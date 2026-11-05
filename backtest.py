# -*- coding: utf-8 -*-
"""
Чесний бектест мульти-ТФ рушія.

Відмінності від старого бектесту:
  * реальні 1m-дані (Yahoo, ~7 днів) ресемплюються у 3m/5m/15m/1h так само, як у бою;
  * сигнал береться лише по ЗАКРИТІЙ свічці, вхід — по close цієї свічки;
  * результат — ціна 1m-свічки через `exp` хвилин (а не «через 3 години»);
  * ті самі фільтри: підтвердження, pivot-фільтр, cooldown/дедуплікація через SignalGate;
  * звіт показує вінрейт ПО ТФ і ПО СТРАТЕГІЯХ, довірчий інтервал Wilson та
    беззбитковий рівень при виплаті 85%.
"""
import os
import logging
from datetime import timedelta

import pandas as pd

from config import YAHOO_PAIRS_MAP, PAIRS_MAP
from signal_pipeline import generate_candidates
from filters import check_pivot_level_proximity, validate_signal_conditions
from quality_gate import SignalGate, wilson_bounds
from timeframe_profiles import PROFILES, ACTIVE_PROFILES

logger = logging.getLogger(__name__)

PERFORMANCE_CACHE_FILE = "pair_strategy_performance.json"
PAYOUT = 0.85
BREAK_EVEN = 1.0 / (1.0 + PAYOUT)  # ≈ 54.1%


def clean_strategy_name(title: str) -> str:
    for ico in ("📊 ", "📉 ", "⚡ ", "📈 ", "📐 ", "🕯️ ", "💥 ", "🚀 "):
        title = title.replace(ico, "")
    return title.strip()


def clear_backtest_cache():
    try:
        if os.path.exists(PERFORMANCE_CACHE_FILE):
            os.remove(PERFORMANCE_CACHE_FILE)
            return True, "Кеш бектесту та результатів стратегій успішно видалено."
        return True, "Кеш файл був відсутній, але стан чистий."
    except Exception as e:
        return False, f"Помилка очищення кешу: {e}"


def _resample(df_1m, rule):
    return df_1m.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()


def load_1m(pair_name: str):
    import yfinance as yf
    ticker = YAHOO_PAIRS_MAP.get(pair_name, f"{pair_name.replace('/', '')}=X")
    df = yf.download(tickers=ticker, period="7d", interval="1m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def simulate_pair(pair_name: str, df_1m: pd.DataFrame, profiles=None, step_cap: int = 300,
                  strategies_status=None, window_1m: int = 1500):
    """
    Крок за кроком «програє» історію: на кожному закритті свічки робочого ТФ викликає
    той самий конвеєр, що й бот. Повертає список угод.
    """
    profiles = profiles or ACTIVE_PROFILES
    trades = []
    gate = SignalGate()
    full_1h = _resample(df_1m, "1h")

    for pkey in profiles:
        prof = PROFILES[pkey]
        tfm = prof["minutes"]
        prim_full = df_1m if tfm == 1 else _resample(df_1m, f"{tfm}min")
        closes = prim_full.index[40:]  # потрібна історія для індикаторів
        if len(closes) > step_cap:
            # рівномірна вибірка по всій історії (а не лише останні N свічок)
            stride = max(1, len(closes) // step_cap)
            closes = closes[::stride]

        for bar_start in closes:
            now = bar_start + timedelta(minutes=tfm) + timedelta(seconds=5)  # одразу після закриття
            hist_1m = df_1m[df_1m.index < now].tail(window_1m)
            if len(hist_1m) < 120:
                continue
            frames = {
                "1m": hist_1m,
                "3m": _resample(hist_1m, "3min"),
                "5m": _resample(hist_1m, "5min"),
                "15m": _resample(hist_1m, "15min"),
                "1h": full_1h[full_1h.index < now].tail(200),
                "1d": pd.DataFrame(),
            }
            cands, _rej = generate_candidates(pair_name, frames, now, strategies_status or {}, profiles=[pkey],
                                              max_age_sec=120)
            for cand in cands:
                ok, _ = validate_signal_conditions(cand)
                if not ok:
                    continue
                ok, _ = check_pivot_level_proximity(cand)
                if not ok:
                    continue
                ok, _ = gate.check(cand, now_ts=now.timestamp())
                if not ok:
                    continue
                exp = cand["suggested_exp"]
                gate.register(cand, exp, now_ts=now.timestamp())

                entry = float(cand["current_price"])
                exit_time = now + timedelta(minutes=exp)
                fut = df_1m[(df_1m.index >= exit_time - timedelta(minutes=1)) & (df_1m.index < exit_time + timedelta(minutes=2))]
                if fut.empty:
                    continue
                exit_p = float(fut["close"].iloc[0])
                if exit_p == entry:
                    continue
                win = (exit_p > entry) if cand["signal"] == "CALL" else (exit_p < entry)
                trades.append({
                    "pair": pair_name, "tf": pkey, "strategy": cand.get("strategy", "?"),
                    "signal": cand["signal"], "exp": exp, "win": bool(win),
                    "confirmations": cand["confirmations"], "regime": cand.get("regime", ""),
                })
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


def run_backtest(pair_name: str, period: str = "7d") -> str:
    try:
        df = load_1m(pair_name)
        if df is None or len(df) < 300:
            return f"❌ Не вдалося завантажити 1m-дані для {pair_name}"
        trades = simulate_pair(pair_name, df)
        return summarize(trades, f"📊 <b>Бектест <code>{pair_name}</code> (останні ~7 днів, 1m→усі ТФ):</b>")
    except Exception as e:
        return f"❌ Помилка виконання бектесту: {e}"


def run_backtest_all_detailed(period: str = "7d") -> list:
    all_trades, per_pair = [], {}
    for p_name in PAIRS_MAP.keys():
        try:
            df = load_1m(p_name)
            if df is None or len(df) < 300:
                continue
            tr = simulate_pair(p_name, df, step_cap=120)
            all_trades.extend(tr)
            if tr:
                per_pair[p_name] = tr
        except Exception as e:
            logger.error(f"Помилка бектесту для {p_name}: {e}")

    messages = [summarize(all_trades, "🌐 <b>ГЛОБАЛЬНИЙ БЕКТЕСТ (останні ~7 днів)</b>\n──────────────────────────────────────")]
    rows = []
    for p, tr in per_pair.items():
        w = sum(1 for t in tr if t["win"])
        rows.append((p, w, len(tr)))
    rows.sort(key=lambda r: r[1] / r[2], reverse=True)
    block = "🔀 <b>ПО ПАРАХ:</b>\n"
    for p, w, n in rows:
        ico = "🟢" if w / n >= BREAK_EVEN else "🔴"
        line = f"{ico} {p}: {w}/{n} ({w / n * 100:.1f}%)\n"
        if len(block + line) > 3500:
            messages.append(block)
            block = ""
        block += line
    if block:
        messages.append(block)
    return messages
