# -*- coding: utf-8 -*-
"""Exact-open backtest for legacy REV presets and full multi-strategy suite. Research only."""
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

import database
from config import PAIRS_MAP, PAYOUT, YAHOO_PAIRS_MAP
from quality_gate import SignalGate, wilson_bounds
from reversion_engine import (
    PRESETS,
    calculate_indicators as reversion_indicators,
    candidates_from_indicators,
    load_strategy_config,
)
from strategy_engine import (
    STRATEGIES_META,
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
    """Повна 5m-історія для прогріву; точні open входу/експірації, без перекриття пресету."""
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


def summarize(trades, title):
    if not trades:
        return f"{title}\nСигналів не знайдено."
    df = pd.DataFrame(trades)
    n, w = len(df), int(df["win"].sum())
    lo, hi = wilson_bounds(w, n)
    ev = (w / n) * PAYOUT - (1.0 - w / n)
    lines = [
        title,
        f"🎯 Угод: <b>{n}</b> | ✅ {w} | 🏆 Вінрейт: <b>{w / n * 100:.1f}%</b>",
        f"📐 95% інтервал: {lo * 100:.1f}–{hi * 100:.1f}% | Беззбиток (виплата {int(PAYOUT * 100)}%): {BREAK_EVEN * 100:.1f}%",
        f"💰 Очікування на угоду: <b>{ev * 100:+.1f}%</b> від ставки\n",
        "📋 <b>По таймфреймах:</b>",
    ]
    for tf, g in df.groupby("tf"):
        gw, gn = int(g["win"].sum()), len(g)
        ico = "🟢" if gw / gn >= BREAK_EVEN else "🔴"
        lines.append(f"{ico} {tf}: {gw}/{gn} ({gw / gn * 100:.1f}%)")
    lines.append("\n📋 <b>По стратегіях:</b>")
    stats = []
    for st, g in df.groupby("strategy"):
        gw, gn = int(g["win"].sum()), len(g)
        stats.append((gw / gn, gn, gw, st))
    stats.sort(reverse=True)
    for wr, gn, gw, st in stats:
        ico = "🟢" if wr >= BREAK_EVEN else "🔴"
        lines.append(f"{ico} {st}: {gw}/{gn} ({wr * 100:.1f}%)")
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


def simulate_strategy_suite(pair_name, frames, strategies_status=None, pair_enabled=None):
    settings = load_full_strategy_settings() if strategies_status is None else strategies_status
    use_session = bool(settings.get("session_filter", False))
    use_regime = bool(settings.get("regime_filter", True))

    trades = []
    df_5m = frames.get("5m")
    if df_5m is not None and len(df_5m) >= 201 and any(settings.get(k, True) for k in PRESETS):
        trades.extend(simulate_reversion_pair(pair_name, df_5m, settings, pair_enabled=pair_enabled))

    ind_cache = {}
    htf_cache = {}
    df_1m = frames.get("1m")

    for strategy_key, meta in STRATEGIES_META.items():
        if strategy_key in PRESETS or not settings.get(strategy_key, False):
            continue
        if pair_enabled is not None and not pair_enabled(pair_name, strategy_key):
            continue

        tf_name = meta["tf"]
        tf_min = meta["tf_min"]
        df_tf = frames.get(tf_name)
        min_bars = 201 if tf_min == 5 else 120
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
        exp = int(meta["exp"])
        price_ref = df_1m if (df_1m is not None and not df_1m.empty and exp % tf_min != 0) else df_tf
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


def run_full_backtest(pair_name, strategies_status=None):
    frames = load_frames_for_backtest(pair_name)
    trades = simulate_strategy_suite(pair_name, frames, strategies_status, pair_enabled=database.get_pair_strategy_status)
    return summarize(trades, f"📊 <b>{pair_name} · Усі активні стратегії (1m/3m: 7д · 5m/15m: 60д)</b>") + \
        "\nResearch only: без новин, SegmentFilter та прослизання."


def run_full_backtest_all(strategies_status=None):
    b1 = _download_yf_batch("1m", "7d")
    b5 = _download_yf_batch("5m", "60d")
    b15 = _download_yf_batch("15m", "60d")
    b60 = _download_yf_batch("60m", "180d")
    all_trades = []
    for pair in PAIRS_MAP:
        frames = _build_frames_dict(b1.get(pair), b5.get(pair), b15.get(pair), b60.get(pair))
        all_trades.extend(simulate_strategy_suite(pair, frames, strategies_status, pair_enabled=database.get_pair_strategy_status))
    return summarize(all_trades, "🌐 <b>21 пара · Усі активні стратегії (1m/3m: 7д · 5m/15m: 60д)</b>") + \
        "\nResearch only: без новин, SegmentFilter та прослизання."
