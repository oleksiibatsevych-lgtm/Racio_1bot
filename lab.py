# -*- coding: utf-8 -*-
"""
Strategy Lab — чесна перевірка: чи існує конфігурація (стратегія × таймфрейм × експірація),
яка справді перемагає беззбитковість бінарних опціонів.

Принципи (щоб не обманювати себе):
  1. Сигнал на ЗАКРИТІЙ свічці, вхід — по open наступної 1-хв свічки, вихід — по open через `exp` хв.
  2. Угоди не перекриваються (на парі не відкривається нова, поки не завершилась попередня).
  3. Історія ділиться ЗА ЧАСОМ: 60% — «пошук» (discovery), 40% — «підтвердження» (confirmation).
     Конфігурації обираються ТІЛЬКИ за першою частиною; друга частина їх не бачила.
  4. Статистична значущість рахується з кластерною похибкою (угоди в один і той самий час
     на різних парах корельовані) і з поправкою на кількість обраних кандидатів (Bonferroni).
  5. Контрольні «стратегії» (випадковий напрямок) показують, наскільки високо шум може підняти вінрейт.
  6. PASS лише якщо: вінрейт у підтвердженні > беззбитку, n ≥ 200, p < 0.05/K.
     Беззбиток = 1/(1+payout), для 85% це 54.05%. Це поріг прибутковості, а не «65%».

Запуск:
  python3 lab.py --download        # скачати/дописати 1m-історію з Yahoo (накопичується у lab_data/)
  python3 lab.py                   # аналіз вже накопиченого
Результати: lab_summary.txt, lab_report.csv, lab_validated.json
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "EUR/JPY", "GBP/JPY",
         "AUD/JPY", "CAD/JPY", "CHF/JPY", "EUR/GBP", "EUR/AUD", "EUR/CAD", "EUR/CHF", "GBP/AUD",
         "GBP/CAD", "GBP/CHF", "AUD/CAD", "AUD/CHF", "CAD/CHF"]
TIMEFRAMES = [1, 3, 5, 15]
EXPIRATIONS = [1, 2, 3, 5, 10, 15, 30]
SPLIT_FRACTION = 0.60
TOP_K = 15
MIN_DISC_TRADES = 150
MIN_CONF_TRADES = 200
CLUSTER_MINUTES = 30


# ───────────────────────────── дані ─────────────────────────────
def yahoo_symbol(pair: str) -> str:
    return pair.replace("/", "") + "=X"


def download_and_merge(pairs, data_dir):
    import yfinance as yf
    os.makedirs(data_dir, exist_ok=True)
    for p in pairs:
        path = os.path.join(data_dir, p.replace("/", "_") + ".csv")
        try:
            df = yf.Ticker(yahoo_symbol(p)).history(period="7d", interval="1m", auto_adjust=True)
            if df is None or df.empty:
                print(f"  {p}: порожньо")
                continue
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"})[["open", "high", "low", "close"]]
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)
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
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df[~df.index.duplicated(keep="last")].dropna()
    return df if len(df) > 500 else None


def minutes_of(index) -> np.ndarray:
    return index.values.astype("datetime64[m]").astype(np.int64)


# ───────────────────────────── індикатори ─────────────────────────────
class Feat:
    """Кеш індикаторів для одного (пара, ТФ). Усі значення на барі i використовують дані ≤ i."""
    def __init__(self, df):
        self.o, self.h, self.l, self.c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
        self.s = pd.Series(self.c)
        self._cache = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def rsi(self, n):
        def f():
            d = self.s.diff()
            up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
            dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
            return (100 - 100 / (1 + up / (dn + 1e-12))).values
        return self._get(("rsi", n), f)

    def ema(self, n):
        return self._get(("ema", n), lambda: self.s.ewm(span=n, adjust=False).mean().values)

    def sma(self, n):
        return self._get(("sma", n), lambda: self.s.rolling(n).mean().values)

    def std(self, n):
        return self._get(("std", n), lambda: self.s.rolling(n).std().values)


def _prev(a, k=1):
    out = np.empty_like(a)
    out[:k] = np.nan
    out[k:] = a[:-k]
    return out


def _sig(call, put):
    out = np.zeros(len(call), dtype=np.int8)
    out[call] = 1
    out[put & ~call] = -1
    return out


# ───────────────────────────── стратегії ─────────────────────────────
def s_rsi_rev(F, n, lo, hi):
    r = F.rsi(n)
    return _sig(r < lo, r > hi)


def s_bb(F, k, mode):
    ma, sd = F.sma(20), F.std(20)
    below, above = F.c < ma - k * sd, F.c > ma + k * sd
    return _sig(below, above) if mode == "fade" else _sig(above, below)


def s_ema_cross(F, a, b):
    diff = F.ema(a) - F.ema(b)
    pd_ = _prev(diff)
    return _sig((diff > 0) & (pd_ <= 0), (diff < 0) & (pd_ >= 0))


def s_pullback(F):
    e21, e50, r = F.ema(21), F.ema(50), F.rsi(7)
    up, dn = (F.c > e50) & (e21 > e50), (F.c < e50) & (e21 < e50)
    return _sig(up & (r < 35), dn & (r > 65))


def s_macd(F):
    macd = F.ema(12) - F.ema(26)
    hist = macd - pd.Series(macd).ewm(span=9, adjust=False).mean().values
    ph = _prev(hist)
    return _sig((hist > 0) & (ph <= 0), (hist < 0) & (ph >= 0))


def s_donchian(F, n):
    hi = pd.Series(F.h).rolling(n).max().shift(1).values
    lo = pd.Series(F.l).rolling(n).min().shift(1).values
    return _sig(F.c > hi, F.c < lo)


def s_stoch(F):
    lo = pd.Series(F.l).rolling(14).min().values
    hi = pd.Series(F.h).rolling(14).max().values
    k = pd.Series((F.c - lo) / (hi - lo + 1e-12) * 100).rolling(3).mean().values
    d = pd.Series(k).rolling(3).mean().values
    pk, pdd = _prev(k), _prev(d)
    return _sig((k > d) & (pk <= pdd) & (k < 30), (k < d) & (pk >= pdd) & (k > 70))


def s_engulf(F):
    po, pc = _prev(F.o), _prev(F.c)
    bull = (pc < po) & (F.c > F.o) & (F.c >= po) & (F.o <= pc)
    bear = (pc > po) & (F.c < F.o) & (F.c <= po) & (F.o >= pc)
    return _sig(bull, bear)


def s_pin(F):
    rng = F.h - F.l + 1e-12
    body = np.abs(F.c - F.o)
    lw = np.minimum(F.o, F.c) - F.l
    uw = F.h - np.maximum(F.o, F.c)
    return _sig((lw > 2 * body) & (lw > 0.6 * rng), (uw > 2 * body) & (uw > 0.6 * rng))


def s_streak(F, n, mode):
    g = (F.c > F.o).astype(int)
    r = (F.c < F.o).astype(int)
    gs = pd.Series(g).rolling(n).sum().values == n
    rs = pd.Series(r).rolling(n).sum().values == n
    return _sig(rs, gs) if mode == "fade" else _sig(gs, rs)


def s_zscore(F, th):
    z = (F.c - F.sma(30)) / (F.std(30) + 1e-12)
    return _sig(z < -th, z > th)


STRATEGIES = []
for n_, lo_, hi_ in ((7, 30, 70), (7, 20, 80), (14, 30, 70), (14, 25, 75)):
    STRATEGIES.append((f"rsi_rev(n={n_},{lo_}/{hi_})", s_rsi_rev, dict(n=n_, lo=lo_, hi=hi_)))
for k_ in (2.0, 2.5):
    STRATEGIES.append((f"bb_fade(k={k_})", s_bb, dict(k=k_, mode="fade")))
    STRATEGIES.append((f"bb_break(k={k_})", s_bb, dict(k=k_, mode="cont")))
for a_, b_ in ((9, 21), (5, 13)):
    STRATEGIES.append((f"ema_cross({a_},{b_})", s_ema_cross, dict(a=a_, b=b_)))
STRATEGIES += [("trend_pullback", s_pullback, {}), ("macd_flip", s_macd, {}),
               ("stoch_rev", s_stoch, {}), ("engulfing_rev", s_engulf, {}), ("pin_bar_rev", s_pin, {})]
for n_ in (20, 10):
    STRATEGIES.append((f"donchian_break({n_})", s_donchian, dict(n=n_)))
for n_ in (3, 4):
    STRATEGIES.append((f"streak_fade({n_})", s_streak, dict(n=n_, mode="fade")))
    STRATEGIES.append((f"streak_cont({n_})", s_streak, dict(n=n_, mode="cont")))
for th_ in (2.0, 2.5):
    STRATEGIES.append((f"zscore_fade({th_})", s_zscore, dict(th=th_)))

N_CONTROLS = 5


def control_signal(n_bars, seed, density=0.04):
    rng = np.random.default_rng(seed)
    mask = rng.random(n_bars) < density
    dirs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_bars)
    return np.where(mask, dirs, 0).astype(np.int8)


# ───────────────────────────── угоди ─────────────────────────────
def extract_trades(sig, end_min, m1_min, m1_open, exp, delay=0):
    """Повертає (entry_minute[], win[]) для неперекривних угод."""
    idx = np.flatnonzero(sig)
    if idx.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int8)
    em = end_min[idx] + delay
    keep, i, n = [], 0, len(em)
    while i < n:
        keep.append(i)
        i = int(np.searchsorted(em, em[i] + exp, side="left"))
    keep = np.array(keep)
    em, dirs = em[keep], sig[idx][keep]

    ie = np.searchsorted(m1_min, em)
    ix = np.searchsorted(m1_min, em + exp)
    n1 = len(m1_min)
    ok = (ie < n1) & (ix < n1)
    ie_c, ix_c = np.minimum(ie, n1 - 1), np.minimum(ix, n1 - 1)
    ok &= (m1_min[ie_c] == em) & (m1_min[ix_c] == em + exp)
    em, dirs, ie_c, ix_c = em[ok], dirs[ok], ie_c[ok], ix_c[ok]
    delta = m1_open[ix_c] - m1_open[ie_c]
    nz = delta != 0
    em, dirs, delta = em[nz], dirs[nz], delta[nz]
    win = ((delta > 0) & (dirs == 1)) | ((delta < 0) & (dirs == -1))
    return em, win.astype(np.int8)


# ───────────────────────────── статистика ─────────────────────────────
def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def cluster_stats(win, em, breakeven):
    """Вінрейт, кластерна SE (кластер = 30 хв, усі пари разом), z і p (односторонній) проти беззбитку."""
    n = len(win)
    if n == 0:
        return dict(n=0, wr=float("nan"), se=float("nan"), z=float("nan"), p=1.0, lo=float("nan"))
    w = win.sum()
    p = w / n
    _, inv = np.unique(em // CLUSTER_MINUTES, return_inverse=True)
    nb = np.bincount(inv).astype(float)
    wb = np.bincount(inv, weights=win.astype(float))
    resid = wb - p * nb
    var = float((resid ** 2).sum()) / (n ** 2)
    se = max(math.sqrt(var), math.sqrt(p * (1 - p) / n) * 0.5, 1e-9)
    z = (p - breakeven) / se
    return dict(n=n, wr=p, se=se, z=z, p=norm_sf(z), lo=p - 1.645 * se)


# ───────────────────────────── основний цикл ─────────────────────────────
def collect(data, delay=0, verbose=True):
    """Повертає dict[config] -> (em[], win[]) по всіх парах."""
    store = {}
    t0 = time.time()
    for pi, (pair, df1) in enumerate(data.items()):
        m1_min, m1_open = minutes_of(df1.index), df1["open"].values.astype(float)
        for tf in TIMEFRAMES:
            if tf == 1:
                dft = df1
            else:
                dft = df1.resample(f"{tf}min", label="left", closed="left").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            if len(dft) < 120:
                continue
            end_min = minutes_of(dft.index) + tf
            F = Feat(dft)
            sigs = [(name, fn(F, **kw)) for name, fn, kw in STRATEGIES]
            sigs += [(f"CTRL_random_{k + 1}", control_signal(len(dft), seed=1000 * pi + 10 * tf + k)) for k in range(N_CONTROLS)]
            for name, sg in sigs:
                for exp in EXPIRATIONS:
                    em, win = extract_trades(sg, end_min, m1_min, m1_open, exp, delay)
                    if len(em):
                        store.setdefault((name, tf, exp), []).append((em, win))
        if verbose:
            print(f"  [{pi + 1}/{len(data)}] {pair} готово ({time.time() - t0:.0f}с)", flush=True)
    merged = {}
    for k, parts in store.items():
        em = np.concatenate([p[0] for p in parts])
        win = np.concatenate([p[1] for p in parts])
        order = np.argsort(em, kind="stable")
        merged[k] = (em[order], win[order])
    return merged


def analyze(merged, payout):
    breakeven = 1.0 / (1.0 + payout)
    all_em = np.concatenate([v[0] for v in merged.values()])
    split = np.quantile(all_em, SPLIT_FRACTION)
    rows = []
    for (name, tf, exp), (em, win) in merged.items():
        d, c = em < split, em >= split
        sd = cluster_stats(win[d], em[d], breakeven)
        sc = cluster_stats(win[c], em[c], breakeven)
        rows.append(dict(strategy=name, tf=tf, exp=exp, control=name.startswith("CTRL"),
                         d_n=sd["n"], d_wr=sd["wr"], d_z=sd["z"],
                         c_n=sc["n"], c_wr=sc["wr"], c_z=sc["z"], c_p=sc["p"], c_lo=sc["lo"]))
    rep = pd.DataFrame(rows)
    return rep, split, breakeven


def build_summary(rep, split, breakeven, payout, data_info):
    real = rep[~rep.control & (rep.d_n >= MIN_DISC_TRADES)].copy()
    ctrl = rep[rep.control & (rep.d_n >= MIN_DISC_TRADES)].copy()
    n_configs = len(real)
    top = real.sort_values("d_z", ascending=False).head(TOP_K).copy()
    alpha = 0.05 / TOP_K
    top["PASS"] = (top.c_n >= MIN_CONF_TRADES) & (top.c_wr > breakeven) & (top.c_p < alpha)
    lines = []
    A = lines.append
    A("🧪 ЛАБОРАТОРІЯ СТРАТЕГІЙ")
    A(f"Дані: {data_info}")
    A(f"Беззбиток при виплаті {payout * 100:.0f}%: {breakeven * 100:.2f}%  (це поріг прибутку, не 65%)")
    A(f"Перевірено конфігурацій: {n_configs} (стратегія × ТФ × експірація)")
    A(f"Поділ за часом: перші {int(SPLIT_FRACTION * 100)}% — пошук, решта — підтвердження (їх вибір не бачив)")
    A("")
    if len(ctrl):
        A("Контроль (випадковий напрямок):")
        A(f"  найкращий у пошуку: {ctrl.d_wr.max() * 100:.1f}%  → той самий у підтвердженні: "
          f"{ctrl.loc[ctrl.d_wr.idxmax(), 'c_wr'] * 100:.1f}%")
        A(f"  середній вінрейт контролю: пошук {ctrl.d_wr.mean() * 100:.1f}%, підтвердження {ctrl.c_wr.mean() * 100:.1f}%")
    A(f"Найкращий вінрейт серед ВСІХ реальних конфігурацій у пошуку (наївний вибір): {real.d_wr.max() * 100:.1f}%")
    A(f"Середній вінрейт усіх реальних конфігурацій у підтвердженні: {real.c_wr.mean() * 100:.2f}%")
    A("")
    A(f"ТОП-{TOP_K} за пошуком → що з ними на нових даних:")
    A("стратегія | ТФ | exp | пошук n/WR | підтвердж. n/WR | p | вердикт")
    for _, r in top.iterrows():
        A(f"{r.strategy} | {int(r.tf)}m | {int(r.exp)}хв | {int(r.d_n)}/{r.d_wr * 100:.1f}% | "
          f"{int(r.c_n)}/{r.c_wr * 100:.1f}% | {r.c_p:.3f} | {'✅ PASS' if r.PASS else '❌'}")
    n_pass = int(top.PASS.sum())
    A("")
    A(f"ПІДСУМОК: пройшли перевірку {n_pass} з {len(top)}.")
    if n_pass == 0:
        A("Жодна конфігурація не показала стійкої переваги над беззбитком на нових даних.")
    else:
        A("Це кандидати, а не гарантія: потрібне ще підтвердження на живих угодах (демо).")
    return "\n".join(lines), top[top.PASS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="lab_data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--delay", type=int, default=0, help="затримка входу в хвилинах після закриття свічки")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    if args.download:
        print("Завантаження 1m-історії (Yahoo)…")
        download_and_merge(PAIRS, args.data_dir)

    data = {p: d for p in PAIRS if (d := load_pair(p, args.data_dir)) is not None}
    if not data:
        print("Немає даних. Запустіть: python3 lab.py --download")
        sys.exit(1)
    first = min(d.index[0] for d in data.values())
    last = max(d.index[-1] for d in data.values())
    days = (last - first).total_seconds() / 86400
    info = f"{len(data)} пар, {first:%Y-%m-%d} → {last:%Y-%m-%d} ({days:.1f} дн.)"
    print(info)

    merged = collect(data, delay=args.delay)
    rep, split, breakeven = analyze(merged, args.payout)
    summary, passed = build_summary(rep, split, breakeven, args.payout, info)

    os.makedirs(args.out, exist_ok=True)
    rep.sort_values("d_z", ascending=False).to_csv(os.path.join(args.out, "lab_report.csv"), index=False)
    with open(os.path.join(args.out, "lab_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.out, "lab_validated.json"), "w", encoding="utf-8") as f:
        json.dump([dict(strategy=r.strategy, tf=int(r.tf), exp=int(r.exp), conf_wr=float(r.c_wr), conf_n=int(r.c_n))
                   for _, r in passed.iterrows()], f, ensure_ascii=False, indent=2)
    print("\n" + summary)


if __name__ == "__main__":
    main()
