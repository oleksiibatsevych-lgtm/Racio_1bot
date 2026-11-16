# -*- coding: utf-8 -*-
"""Exact-open backtest for legacy REV presets and full multi-strategy suite with pair x strategy matrix."""
import csv
import html
import io
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

import database
from config import AUTO_DISABLE_WR_THRESHOLD, PAIRS_MAP, PAYOUT, YAHOO_PAIRS_MAP
from quality_gate import SignalGate, wilson_bounds
from reversion_engine import (
    PRESETS,
    calculate_indicators as reversion_indicators,
    candidates_from_indicators,
    load_strategy_config,
)
from strategy_engine import (
    STRATEGIES_META,
    calculate_dynamic_expiration,
    compute_base_indicators,
    evaluate_strategy_series,
    htf_alignment,
    load_all_strategy_config,
)
from trading_sessions import session_allowed

logger = logging.getLogger(__name__)
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)


def load_strategy_settings():
    return load_strategy_config(Path(__file__).with_name("strategies_config.json"))


def load_full_strategy_settings():
    return load_all_strategy_config(Path(__file__).with_name("strategies_config.json"))


def _record_trade(pair_name, cand, now, prices, trades, gate, open_reversions):
    strategy = cand.get("strategy", "?")
    if open_reversions.get(strategy, now) > now:
        return
    exp = cand["suggested_exp"]
    exit_time = now + timedelta(minutes=exp)
    if now not in prices.index or exit_time not in prices.index:
        return
    if not gate.check(cand, now_ts=now.timestamp())[0]:
        return
    gate.register(cand, exp, now_ts=now.timestamp())
    open_reversions[strategy] = exit_time
    entry = float(prices.loc[now, "open"])
    exit_p = float(prices.loc[exit_time, "open"])
    if exit_p == entry:
        return
    win = (exit_p > entry) if cand["signal"] == "CALL" else (exit_p < entry)
    trades.append({
        "pair": pair_name, "tf": cand["tf"], "strategy": strategy,
        "signal": cand["signal"], "exp": exp, "win": bool(win),
        "confirmations": cand["confirmations"], "regime": cand.get("regime", ""),
        "entry_time": now, "exit_time": exit_time,
    })


def simulate_reversion_pair(pair_name, df_5m, strategies_status=None, start=None, end=None, pair_enabled=None):
    settings = load_strategy_settings() if strategies_status is None else strategies_status
    indicators = reversion_indicators(df_5m)
    trades, open_reversions = [], {}
    gate = SignalGate()
    for i in range(200, len(indicators)):
        now = indicators.index[i] + timedelta(minutes=5)
        if start is not None and now < start:
            continue
        if end is not None and now >= end:
            break
        candidates = candidates_from_indicators(pair_name, indicators.iloc[:i + 1], now, settings, pair_enabled)
        for cand in candidates:
            if end is not None and now + timedelta(minutes=cand["suggested_exp"]) >= end:
                continue
            _record_trade(pair_name, cand, now, df_5m, trades, gate, open_reversions)
    return trades


def summarize(trades, title, payout: float = PAYOUT, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD):
    if not trades:
        return f"{title}\nСигналів не знайдено."
    be = 1.0 / (1.0 + float(payout))
    df = pd.DataFrame(trades)
    n, w = len(df), int(df["win"].sum())
    lo, hi = wilson_bounds(w, n)
    ev = (w / n) * float(payout) - (1.0 - w / n)
    lines = [
        title,
        f"🎯 Угод: <b>{n}</b> | ✅ {w} | 🏆 Вінрейт: <b>{w / n * 100:.1f}%</b>",
        f"📐 95% інтервал: {lo * 100:.1f}–{hi * 100:.1f}% | Беззбиток (виплата {int(float(payout) * 100)}%): {be * 100:.1f}%",
        f"💰 Очікування на угоду: <b>{ev * 100:+.1f}%</b> від ставки\n",
        "📋 <b>По таймфреймах:</b>",
    ]
    for tf, g in df.groupby("tf"):
        gw, gn = int(g["win"].sum()), len(g)
        ico = "🟢" if gw / gn >= be else ("🟡" if gw / gn >= float(wr_threshold) else "🔴")
        lines.append(f"{ico} {tf}: {gw}/{gn} ({gw / gn * 100:.1f}%)")
    lines.append("\n📋 <b>По стратегіях:</b>")
    stats = []
    for st, g in df.groupby("strategy"):
        gw, gn = int(g["win"].sum()), len(g)
        stats.append((gw / gn, gn, gw, st))
    stats.sort(reverse=True)
    for wr, gn, gw, st in stats:
        ico = "🟢" if wr >= be else ("🟡" if wr >= float(wr_threshold) else "🔴")
        lines.append(f"{ico} {html.escape(st)}: {gw}/{gn} ({wr * 100:.1f}%)")
    return "\n".join(lines)


def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns=str.lower)
    req = [c for c in ("open", "high", "low", "close") if c in out.columns]
    if len(req) < 4:
        return pd.DataFrame()
    if "volume" not in out.columns:
        out["volume"] = 0.0
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    out = out.sort_index().dropna(subset=["open", "high", "low", "close"])
    return out.iloc[:-1] if len(out) > 1 else pd.DataFrame()


def _download_yf(pair_name, interval, period):
    if pair_name not in PAIRS_MAP:
        raise ValueError("Невідома пара")
    df = yf.download(YAHOO_PAIRS_MAP[pair_name], interval=interval, period=period, progress=False, auto_adjust=True)
    return _clean_ohlcv(df)


def _download_yf_batch(interval, period) -> dict[str, pd.DataFrame]:
    tickers = list(YAHOO_PAIRS_MAP.values())
    raw = yf.download(" ".join(tickers), interval=interval, period=period, group_by="ticker", threads=True, progress=False, auto_adjust=True)
    result = {}
    if raw is None or raw.empty:
        return result
    if isinstance(raw.columns, pd.MultiIndex):
        top = set(raw.columns.get_level_values(0))
        for pair, y_ticker in YAHOO_PAIRS_MAP.items():
            if y_ticker in top:
                cleaned = _clean_ohlcv(raw[y_ticker])
                if not cleaned.empty:
                    result[pair] = cleaned
    return result


def load_5m(pair_name):
    return _download_yf(pair_name, "5m", "60d")


def _build_frames_dict(df_1m, df_5m, df_15m, df_60m):
    df_3m = pd.DataFrame()
    if df_1m is not None and not df_1m.empty:
        df_3m = df_1m.resample("3min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open", "high", "low", "close"])
    return {
        "1m": df_1m if df_1m is not None else pd.DataFrame(),
        "3m": df_3m,
        "5m": df_5m if df_5m is not None else pd.DataFrame(),
        "15m": df_15m if df_15m is not None else pd.DataFrame(),
        "60m": df_60m if df_60m is not None else pd.DataFrame(),
    }


def load_frames_for_backtest(pair_name):
    return _build_frames_dict(
        _download_yf(pair_name, "1m", "7d"),
        _download_yf(pair_name, "5m", "60d"),
        _download_yf(pair_name, "15m", "60d"),
        _download_yf(pair_name, "60m", "180d"),
    )


def simulate_strategy_suite(pair_name, frames, strategies_status=None, pair_enabled=None, include_all_strategies=False, expiry_mode="FIX"):
    settings = dict(load_full_strategy_settings() if strategies_status is None else strategies_status)
    use_session = bool(settings.get("session_filter", False))
    use_regime = bool(settings.get("regime_filter", True))

    trades = []
    df_5m = frames.get("5m")
    if df_5m is not None and len(df_5m) >= 201:
        rev_settings = dict(settings)
        if include_all_strategies:
            for k in PRESETS:
                rev_settings[k] = True
            trades.extend(simulate_reversion_pair(pair_name, df_5m, rev_settings, pair_enabled=None))
        elif any(settings.get(k, True) for k in PRESETS):
            trades.extend(simulate_reversion_pair(pair_name, df_5m, rev_settings, pair_enabled=pair_enabled))

    ind_cache = {}
    htf_cache = {}
    df_1m = frames.get("1m")

    for strategy_key, meta in STRATEGIES_META.items():
        if strategy_key in PRESETS:
            continue
        if not include_all_strategies:
            if not settings.get(strategy_key, False):
                continue
            if pair_enabled is not None and not pair_enabled(pair_name, strategy_key):
                continue

        tf_name = meta["tf"]
        tf_min = meta["tf_min"]
        df_tf = frames.get(tf_name)
        min_bars = 201 if (tf_min == 5 or strategy_key.startswith("PSAR_EMA")) else 120
        if df_tf is None or len(df_tf) < min_bars:
            continue

        if tf_name not in ind_cache:
            ind_cache[tf_name] = compute_base_indicators(df_tf)
        ind = ind_cache[tf_name]

        htf_min = meta.get("htf")
        htf_ctx = None
        if htf_min:
            htf_key = (tf_name, htf_min)
            if htf_key not in htf_cache:
                htf_cache[htf_key] = htf_alignment(df_tf, frames.get(f"{htf_min}m"), tf_min, htf_min)
            htf_ctx = htf_cache[htf_key]

        sig_series = evaluate_strategy_series(df_tf, strategy_key, ind=ind, htf_ctx=htf_ctx, regime_filter=use_regime)
        base_exp = int(meta["exp"])
        next_allowed = df_tf.index[0]

        for i in range(min_bars - 1, len(df_tf)):
            sv = int(sig_series[i])
            if sv == 0:
                continue
            entry_time = df_tf.index[i] + timedelta(minutes=tf_min)
            if entry_time < next_allowed:
                continue
            if use_session and not session_allowed(strategy_key, entry_time):
                continue

            if str(expiry_mode).upper() == "DYN":
                vr = float(ind["vol_ratio"][i]) if not pd.isna(ind["vol_ratio"][i]) else 1.0
                exp = calculate_dynamic_expiration([{
                    "strategy": strategy_key,
                    "suggested_exp": base_exp,
                    "tf_minutes": tf_min,
                    "vol_ratio": vr,
                }], "DYN")
            else:
                exp = base_exp

            price_ref = df_1m if (df_1m is not None and not df_1m.empty and exp % tf_min != 0) else df_tf
            exit_time = entry_time + timedelta(minutes=exp)
            if entry_time not in price_ref.index or exit_time not in price_ref.index:
                continue
            entry_p = float(price_ref.loc[entry_time, "open"])
            exit_p = float(price_ref.loc[exit_time, "open"])
            if exit_p == entry_p:
                continue
            win = (exit_p > entry_p) if sv > 0 else (exit_p < entry_p)
            trades.append({
                "pair": pair_name, "tf": tf_name, "strategy": strategy_key,
                "signal": "CALL" if sv > 0 else "PUT", "exp": exp, "win": bool(win),
                "confirmations": 1 if htf_min else 0, "regime": meta["regime"],
                "entry_time": entry_time, "exit_time": exit_time,
            })
            next_allowed = exit_time

    return trades


def compute_pair_strategy_stats(trades: list[dict]) -> list[dict]:
    if not trades:
        return []
    df = pd.DataFrame(trades)
    rows = []
    for (pair, st), grp in df.groupby(["pair", "strategy"]):
        total = int(len(grp))
        wins = int(grp["win"].sum())
        rows.append({
            "pair": pair,
            "strategy": st,
            "wins": wins,
            "losses": total - wins,
            "total": total,
            "win_rate": wins / total if total else 0.0,
        })
    rows.sort(key=lambda r: (r["pair"], -r["win_rate"], -r["total"]))
    return rows


def format_pair_strategy_report(trades: list[dict], title: str, payout: float = PAYOUT, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD) -> str:
    base = summarize(trades, title, payout=payout, wr_threshold=wr_threshold)
    if not trades:
        return base
    be = 1.0 / (1.0 + float(payout))
    df = pd.DataFrame(trades)
    lines = [base, "\n🔎 <b>Детально: Пара × Стратегія</b>"]
    for pair in sorted(df["pair"].unique()):
        pdf = df[df["pair"] == pair]
        pw, pn = int(pdf["win"].sum()), len(pdf)
        p_ico = "🟢" if pw / pn >= be else ("🟡" if pw / pn >= float(wr_threshold) else "🔴")
        lines.append(f"\n{p_ico} <b>{html.escape(pair)}</b> ({pw / pn * 100:.1f}% · {pw}/{pn})")
        st_rows = []
        for st, grp in pdf.groupby("strategy"):
            sw, sn = int(grp["win"].sum()), len(grp)
            st_rows.append((sw / sn, sn, sw, st))
        st_rows.sort(reverse=True)
        for wr, sn, sw, st in st_rows:
            ico = "🟢" if wr >= be else ("🟡" if wr >= float(wr_threshold) else "🔴")
            lines.append(f"  {ico} <code>{html.escape(st)}</code>: {wr * 100:.1f}% ({sw}/{sn})")
    return "\n".join(lines)


def build_backtest_csv(stats_rows: list[dict], wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Pair", "Strategy", "Category", "TF", "ExpiryMin", "Wins", "Losses", "Total", "WinRatePct", "BelowThreshold"])
    for r in stats_rows:
        meta = STRATEGIES_META.get(r["strategy"], {})
        wr_pct = round(r["win_rate"] * 100.0, 2)
        writer.writerow([
            r["pair"],
            r["strategy"],
            meta.get("category", ""),
            meta.get("tf", ""),
            meta.get("exp", ""),
            r["wins"],
            r["losses"],
            r["total"],
            wr_pct,
            "YES" if r["win_rate"] < float(wr_threshold) else "NO",
        ])
    return buf.getvalue().encode("utf-8-sig")


def comparison(pair_name, df, strategies_status=None):
    settings = dict(load_strategy_settings() if strategies_status is None else strategies_status)
    results = {}
    for enabled in (False, True):
        settings["session_filter"] = enabled
        results[enabled] = simulate_reversion_pair(pair_name, df, settings, pair_enabled=database.get_pair_strategy_status)
    return results


def run_backtest(pair_name, period="60d", strategies_status=None):
    df = load_5m(pair_name)
    results = comparison(pair_name, df, strategies_status)
    return "\n\n".join(
        summarize(trades, f"📊 <b>{pair_name}: REV 5m · сесії {'ON' if enabled else 'OFF'}</b>")
        for enabled, trades in results.items()
    ) + "\nResearch only: без новин, SegmentFilter та затримки виконання."


def run_backtest_all_detailed(period="60d", strategies_status=None):
    combined = {False: [], True: []}
    batch_5m = _download_yf_batch("5m", "60d")
    for pair in PAIRS_MAP:
        df_5m = batch_5m.get(pair)
        if df_5m is None or df_5m.empty:
            df_5m = load_5m(pair)
        if df_5m.empty:
            continue
        for enabled, trades in comparison(pair, df_5m, strategies_status).items():
            combined[enabled].extend(trades)
    return [
        summarize(trades, f"🌐 <b>21 пара · REV 5m · сесії {'ON' if enabled else 'OFF'}</b>")
        for enabled, trades in combined.items()
    ]


def run_full_backtest_with_stats(pair_name: str, strategies_status=None, include_all_strategies: bool = True, payout: float = PAYOUT, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD, expiry_mode: str = "FIX") -> tuple[str, list[dict]]:
    frames = load_frames_for_backtest(pair_name)
    trades = simulate_strategy_suite(
        pair_name,
        frames,
        strategies_status,
        pair_enabled=database.get_pair_strategy_status,
        include_all_strategies=include_all_strategies,
        expiry_mode=expiry_mode,
    )
    stats_rows = compute_pair_strategy_stats(trades)
    mode_lbl = "усі 57 стратегій" if include_all_strategies else "активні стратегії"
    report = format_pair_strategy_report(
        trades,
        f"📊 <b>{html.escape(pair_name)} · Бектест ({mode_lbl} · {expiry_mode} · 1m/3m: 7д, 5m/15m: 60д)</b>",
        payout=payout,
        wr_threshold=wr_threshold,
    ) + "\n<i>Research only: без новин і прослизання.</i>"
    return report, stats_rows


def run_full_backtest_all_with_stats(strategies_status=None, include_all_strategies: bool = True, payout: float = PAYOUT, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD, expiry_mode: str = "FIX") -> tuple[str, list[dict]]:
    b1 = _download_yf_batch("1m", "7d")
    b5 = _download_yf_batch("5m", "60d")
    b15 = _download_yf_batch("15m", "60d")
    b60 = _download_yf_batch("60m", "180d")
    all_trades = []
    for pair in PAIRS_MAP:
        frames = _build_frames_dict(b1.get(pair), b5.get(pair), b15.get(pair), b60.get(pair))
        all_trades.extend(
            simulate_strategy_suite(
                pair,
                frames,
                strategies_status,
                pair_enabled=database.get_pair_strategy_status,
                include_all_strategies=include_all_strategies,
                expiry_mode=expiry_mode,
            )
        )
    stats_rows = compute_pair_strategy_stats(all_trades)
    mode_lbl = "усі 57 стратегій" if include_all_strategies else "активні стратегії"
    report = format_pair_strategy_report(
        all_trades,
        f"🌐 <b>21 пара · Бектест ({mode_lbl} · {expiry_mode} · 1m/3m: 7д, 5m/15m: 60д)</b>",
        payout=payout,
        wr_threshold=wr_threshold,
    ) + "\n<i>Research only: без новин і прослизання.</i>"
    return report, stats_rows


def run_full_backtest(pair_name, strategies_status=None, include_all_strategies=True):
    return run_full_backtest_with_stats(pair_name, strategies_status, include_all_strategies)[0]


def run_full_backtest_all(strategies_status=None, include_all_strategies=True):
    return run_full_backtest_all_with_stats(strategies_status, include_all_strategies)[0]
