"""REV-only exact-open backtest. No news/segment/latency simulation."""
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
import database

from config import YAHOO_PAIRS_MAP, PAIRS_MAP
from quality_gate import SignalGate, wilson_bounds
from reversion_engine import (
    load_strategy_config, candidates_from_indicators,
    calculate_indicators as reversion_indicators,
)

logger = logging.getLogger(__name__)
PAYOUT = 0.85
BREAK_EVEN = 1 / (1 + PAYOUT)


def load_strategy_settings():
    return load_strategy_config(Path(__file__).with_name("strategies_config.json"))


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


def simulate_reversion_pair(pair_name, df_5m, strategies_status=None, start=None, end=None,
                            pair_enabled=None):
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
        candidates = candidates_from_indicators(pair_name, indicators.iloc[:i + 1], now,
                                                 settings, pair_enabled)
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
    ev = (w / n) * PAYOUT - (1 - w / n)
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
    for st, g in df.groupby("strategy"):
        gw, gn = int(g["win"].sum()), len(g)
        ico = "🟢" if gw / gn >= BREAK_EVEN else "🔴"
        lines.append(f"{ico} {st}: {gw}/{gn} ({gw / gn * 100:.1f}%)")
    return "\n".join(lines)


def load_5m(pair_name):
    if pair_name not in PAIRS_MAP:
        raise ValueError("Невідома пара")
    df = yf.download(YAHOO_PAIRS_MAP[pair_name], interval="5m", period="60d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df.sort_index().dropna(subset=["open", "high", "low", "close"]).iloc[:-1]


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
    return "\n\n".join(summarize(trades, f"📊 <b>{pair_name}: REV 5m · сесії {'ON' if enabled else 'OFF'}</b>")
                        for enabled, trades in results.items()) + "\nResearch only: без новин, SegmentFilter та затримки виконання."


def run_backtest_all_detailed(period="60d", strategies_status=None):
    combined = {False: [], True: []}
    for pair in PAIRS_MAP:
        for enabled, trades in comparison(pair, load_5m(pair), strategies_status).items():
            combined[enabled].extend(trades)
    return [summarize(trades, f"🌐 <b>21 пара · REV 5m · сесії {'ON' if enabled else 'OFF'}</b>")
            for enabled, trades in combined.items()]
