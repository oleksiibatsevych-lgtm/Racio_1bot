# -*- coding: utf-8 -*-
"""
Strategy Lab — walk-forward перевірка ВСІХ активних стратегій єдиного реєстру STRATEGIES_META (57+100=157)
з контролем множинного тестування (Benjamini-Hochberg FDR + Bonferroni) та авто-відсівом слабких.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

import reversion_engine as rev
from market_data import download_dukascopy_chunked
from strategy_engine import (
    STRATEGIES_META,
    compute_base_indicators,
    evaluate_strategy_series,
    htf_alignment,
)

PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "CAD/JPY", "CHF/JPY", "EUR/GBP", "EUR/AUD", "EUR/CAD", "EUR/CHF", "GBP/AUD",
    "GBP/CAD", "GBP/CHF", "AUD/CAD", "AUD/CHF", "CAD/CHF",
]
TIMEFRAMES = [1, 3, 5, 15]
EXPIRATIONS = [1, 3, 5, 10, 15, 30]
SPLIT_FRACTION = 0.60
TOP_K = 25
MIN_DISC_TRADES = 120
MIN_CONF_TRADES = 150
CLUSTER_MINUTES = 30


def yahoo_symbol(pair: str) -> str:
    return pair.replace("/", "") + "=X"


def download_and_merge(pairs, data_dir, source: str = "dukascopy", days: int = 90):
    import yfinance as yf
    os.makedirs(data_dir, exist_ok=True)
    for p in pairs:
        path = os.path.join(data_dir, p.replace("/", "_") + ".csv")
        try:
            if source == "dukascopy":
                df = download_dukascopy_chunked(p, days=days)
            else:
                raw = yf.Ticker(yahoo_symbol(p)).history(period="7d", interval="1m", auto_adjust=True)
                if raw is None or raw.empty:
                    df = pd.DataFrame()
                else:
                    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
                    df = raw[cols].rename(columns=str.lower)
                    if "volume" not in df.columns:
                        df["volume"] = 0.0
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert("UTC").tz_localize(None)
            if df is None or df.empty:
                print(f"  {p}: порожньо")
                continue
            if os.path.exists(path):
                old = pd.read_csv(path, index_col=0, parse_dates=True)
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(path)
            print(f"  {p}: {len(df)} хв ({df.index[0]} → {df.index[-1]})")
        except Exception as e:
            print(f"  {p}: помилка {e}")


def load_pair(pair, data_dir):
    path = os.path.join(data_dir, pair.replace("/", "_") + ".csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index().rename(columns=str.lower)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"])
    return df if len(df) > 500 else None


def minutes_of(index) -> np.ndarray:
    return index.values.astype("datetime64[m]").astype(np.int64)


def extract_trades(sig, end_min, m1_min, m1_open, exp, delay=0):
    idx = np.flatnonzero(sig)
    if idx.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int8)
    em = end_min[idx] + delay
    ex = exp[idx] if isinstance(exp, np.ndarray) else np.full(len(idx), int(exp), dtype=np.int64)
    keep, i, n = [], 0, len(em)
    while i < n:
        keep.append(i)
        i = int(np.searchsorted(em, em[i] + ex[i], side="left"))
    keep = np.array(keep)
    em, dirs, ex = em[keep], sig[idx][keep], ex[keep]

    ie, ix = np.searchsorted(m1_min, em), np.searchsorted(m1_min, em + ex)
    n1 = len(m1_min)
    ok = (ie < n1) & (ix < n1)
    ie_c, ix_c = np.minimum(ie, n1 - 1), np.minimum(ix, n1 - 1)
    ok &= (m1_min[ie_c] == em) & (m1_min[ix_c] == em + ex)
    em, dirs, ie_c, ix_c = em[ok], dirs[ok], ie_c[ok], ix_c[ok]
    delta = m1_open[ix_c] - m1_open[ie_c]
    nz = delta != 0
    em, dirs, delta = em[nz], dirs[nz], delta[nz]
    win = ((delta > 0) & (dirs == 1)) | ((delta < 0) & (dirs == -1))
    return em, win.astype(np.int8)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def cluster_stats(win, em, breakeven):
    n = len(win)
    if n == 0:
        return dict(n=0, wr=float("nan"), se=float("nan"), z=float("nan"), p=1.0, lo=float("nan"))
    w = win.sum()
    p = w / n
    _, inv = np.unique(em // CLUSTER_MINUTES, return_inverse=True)
    nb = np.bincount(inv).astype(float)
    wb = np.bincount(inv, weights=win.astype(float))
    resid = wb - p * nb
    se = max(math.sqrt(float((resid ** 2).sum()) / (n ** 2)), math.sqrt(p * (1 - p) / n) * 0.5, 1e-9)
    z = (p - breakeven) / se
    return dict(n=n, wr=p, se=se, z=z, p=norm_sf(z), lo=p - 1.645 * se)


def control_signal(n_bars, seed, density=0.04):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n_bars) < density, rng.choice(np.array([-1, 1], dtype=np.int8), size=n_bars), 0).astype(np.int8)


def collect(data, delay=0, verbose=True):
    store = {}
    pair_stats = []
    t0 = time.time()
    for pi, (pair, df1) in enumerate(data.items()):
        m1_min, m1_open = minutes_of(df1.index), df1["open"].values.astype(float)
        htf_frames = {}
        for h in (15, 60):
            dh = df1.resample(f"{h}min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            if len(dh) >= 60:
                htf_frames[h] = dh

        tf_cache = {}
        for tf in TIMEFRAMES:
            dft = df1 if tf == 1 else df1.resample(f"{tf}min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open", "high", "low", "close"])
            if len(dft) < 120:
                continue
            tf_cache[tf] = (dft, minutes_of(dft.index) + tf, compute_base_indicators(dft))

        htf_ctx_cache = {}
        for s_key, meta in list(STRATEGIES_META.items()):
            tf = meta["tf_min"]
            if tf not in tf_cache:
                continue
            dft, end_min, base_ind = tf_cache[tf]
            if s_key in rev.PRESETS:
                sg = evaluate_strategy_series(dft, s_key)
                label_variants = [(f"BOT_{s_key}", sg, int(meta["exp"]))]
            else:
                htf_m = meta.get("htf")
                htf_ctx = None
                if htf_m:
                    ck = (tf, htf_m)
                    if ck not in htf_ctx_cache:
                        htf_ctx_cache[ck] = htf_alignment(dft, htf_frames.get(htf_m), tf, htf_m)
                    htf_ctx = htf_ctx_cache[ck]
                sg_reg = evaluate_strategy_series(dft, s_key, ind=base_ind, htf_ctx=htf_ctx, regime_filter=True)
                label_variants = [(f"BOT_{s_key}", sg_reg, int(meta["exp"]))]

            for lbl, sg, exp in label_variants:
                em, win = extract_trades(sg, end_min, m1_min, m1_open, exp, delay)
                if len(em):
                    store.setdefault((lbl, tf, str(exp)), []).append((em, win))
                    w_sum, t_sum = int(win.sum()), int(len(win))
                    pair_stats.append({"pair": pair, "strategy": s_key, "wins": w_sum, "losses": t_sum - w_sum, "total": t_sum, "win_rate": w_sum / t_sum})

        for tf, (dft, end_min, _) in tf_cache.items():
            for k in range(5):
                sg_c = control_signal(len(dft), seed=1000 * pi + 10 * tf + k)
                em, win = extract_trades(sg_c, end_min, m1_min, m1_open, tf * 3, delay)
                if len(em):
                    store.setdefault((f"CTRL_random_{k + 1}", tf, str(tf * 3)), []).append((em, win))

        if verbose:
            print(f"  [{pi + 1}/{len(data)}] {pair} готово ({time.time() - t0:.0f}с)", flush=True)

    merged = {}
    for k, parts in store.items():
        em = np.concatenate([p[0] for p in parts])
        win = np.concatenate([p[1] for p in parts])
        order = np.argsort(em, kind="stable")
        merged[k] = (em[order], win[order])
    return merged, pair_stats


def _benjamini_hochberg(pvals: np.ndarray, q: float = 0.10) -> np.ndarray:
    m = len(pvals)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, m + 1) / m) * q
    passed_idx = np.where(ranked <= thresh)[0]
    mask = np.zeros(m, dtype=bool)
    if len(passed_idx):
        max_k = int(passed_idx[-1])
        mask[order[: max_k + 1]] = True
    return mask


def analyze(merged, payout):
    breakeven = 1.0 / (1.0 + payout)
    all_em = np.concatenate([v[0] for v in merged.values()])
    split = np.quantile(all_em, SPLIT_FRACTION)
    rows = []
    for (name, tf, exp), (em, win) in merged.items():
        d, c = em < split, em >= split
        sd = cluster_stats(win[d], em[d], breakeven)
        sc = cluster_stats(win[c], em[c], breakeven)
        tot_n = sd["n"] + sc["n"]
        tot_w = (0 if np.isnan(sd["wr"]) else sd["wr"] * sd["n"]) + (0 if np.isnan(sc["wr"]) else sc["wr"] * sc["n"])
        rows.append(dict(
            strategy=name, tf=tf, exp=exp, control=name.startswith("CTRL"),
            d_n=sd["n"], d_wr=sd["wr"], d_z=sd["z"],
            c_n=sc["n"], c_wr=sc["wr"], c_z=sc["z"], c_p=sc["p"], c_lo=sc["lo"],
            tot_n=tot_n, tot_wr=(tot_w / tot_n) if tot_n else float("nan"),
        ))
    rep = pd.DataFrame(rows)
    return rep, split, breakeven


def prune_config_from_report(rep: pd.DataFrame, config_path: str, threshold: float = 0.45, min_trades: int = 20) -> list[str]:
    if not os.path.exists(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    bot_rows = rep[rep["strategy"].str.startswith("BOT_")].copy()
    if bot_rows.empty:
        return []
    bot_rows["base_key"] = bot_rows["strategy"].str.replace(r"^BOT_", "", regex=True)
    disabled = []
    for _, row in bot_rows.iterrows():
        key, n, wr = row["base_key"], float(row["tot_n"]), float(row["tot_wr"])
        if n >= min_trades and key in cfg and wr < threshold and cfg[key]:
            cfg[key] = False
            disabled.append(f"{key} ({wr * 100:.1f}%, n={int(n)})")
    if disabled:
        tmp = config_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, config_path)
    return disabled


def build_summary(rep, split, breakeven, payout, data_info):
    real = rep[~rep.control & (rep.d_n >= MIN_DISC_TRADES)].copy()
    ctrl = rep[rep.control & (rep.d_n >= MIN_DISC_TRADES)].copy()
    real["BH_PASS"] = False
    cand_mask = (real.c_n >= MIN_CONF_TRADES) & (real.c_wr > breakeven)
    if cand_mask.any():
        real.loc[cand_mask, "BH_PASS"] = _benjamini_hochberg(real.loc[cand_mask, "c_p"].to_numpy(float), q=0.10)
    top = real.sort_values("d_z", ascending=False).head(TOP_K).copy()
    alpha = 0.05 / max(1, len(real))
    top["PASS"] = (top.c_n >= MIN_CONF_TRADES) & (top.c_wr > breakeven) & ((top.c_p < alpha) | top.BH_PASS)

    lines = [
        f"🧪 ЛАБОРАТОРІЯ СТРАТЕГІЙ (УСЬОГО В РЕЄСТРІ: {len(STRATEGIES_META)})",
        f"Дані: {data_info}",
        f"Беззбиток при виплаті {payout * 100:.0f}%: {breakeven * 100:.2f}% | Поправка Бонферроні alpha={alpha:.5f}",
        f"Протестовано конфігурацій: {len(real)} | Поділ 60/40",
        "",
    ]
    if len(ctrl):
        lines.append(f"Контроль (випадкові входи): max пошук {ctrl.d_wr.max() * 100:.1f}% → підтвердження {ctrl.loc[ctrl.d_wr.idxmax(), 'c_wr'] * 100:.1f}%")
    if len(real):
        lines.append(f"Найкращий WR у пошуку: {real.d_wr.max() * 100:.1f}% | Середній OOS WR: {real.c_wr.mean() * 100:.2f}%")
    lines.append(f"\nТОП-{TOP_K} за пошуком → OOS підтвердження:")
    lines.append("стратегія | ТФ | exp | пошук n/WR | OOS n/WR | p | вердикт")
    for _, r in top.iterrows():
        lines.append(f"{r.strategy} | {int(r.tf)}m | {r.exp}хв | {int(r.d_n)}/{r.d_wr * 100:.1f}% | {int(r.c_n)}/{r.c_wr * 100:.1f}% | {r.c_p:.4f} | {'✅ PASS' if r.PASS else '❌'}")
    lines.append(f"\nПІДСУМОК: пройшли жорсткий OOS-відбір {int(top.PASS.sum())} з {len(top)} (BH-FDR усього: {int(real['BH_PASS'].sum())}).")
    return "\n".join(lines), top[top.PASS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="lab_data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--source", choices=("yahoo", "dukascopy"), default="dukascopy")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--delay", type=int, default=0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--prune-config", default="")
    ap.add_argument("--prune-threshold", type=float, default=0.45)
    ap.add_argument("--min-trades", type=int, default=20)
    args = ap.parse_args()

    if args.download:
        print(f"Завантаження 1m-історії ({args.source}, {args.days}д)…")
        download_and_merge(PAIRS, args.data_dir, source=args.source, days=args.days)

    data = {p: d for p in PAIRS if (d := load_pair(p, args.data_dir)) is not None}
    if not data:
        print("Немає даних. Запустіть: python3 lab.py --download --source dukascopy --days 90")
        sys.exit(1)
    first, last = min(d.index[0] for d in data.values()), max(d.index[-1] for d in data.values())
    info = f"{len(data)} пар, {first:%Y-%m-%d} → {last:%Y-%m-%d} ({(last - first).total_seconds() / 86400:.1f} дн.)"
    merged, _ = collect(data, delay=args.delay)
    rep, split, breakeven = analyze(merged, args.payout)
    summary, passed = build_summary(rep, split, breakeven, args.payout, info)

    os.makedirs(args.out, exist_ok=True)
    rep.sort_values("d_z", ascending=False).to_csv(os.path.join(args.out, "lab_report.csv"), index=False)
    with open(os.path.join(args.out, "lab_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.out, "lab_validated.json"), "w", encoding="utf-8") as f:
        json.dump([dict(strategy=r.strategy, tf=int(r.tf), exp=str(r.exp), conf_wr=float(r.c_wr), conf_n=int(r.c_n)) for _, r in passed.iterrows()], f, ensure_ascii=False, indent=2)
    if args.prune_config:
        pruned = prune_config_from_report(rep, args.prune_config, threshold=args.prune_threshold, min_trades=args.min_trades)
        print(f"Відсіяно нижче {args.prune_threshold * 100:.0f}%: {len(pruned)} стратегій")
    print("\n" + summary)


if __name__ == "__main__":
    main()
