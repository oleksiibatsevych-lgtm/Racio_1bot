# -*- coding: utf-8 -*-
"""
Мульти-таймфреймовий конвеєр кандидатів у сигнали.

Для однієї пари повертає список кандидатів (по одному на кожен ТФ-профіль, де є сигнал).
Гарантії:
  * використовуються ТІЛЬКИ закриті свічки (немає перемальовування);
  * кожен кандидат має bar_id — унікальний ідентифікатор закритої свічки (дедуплікація);
  * кожен кандидат має кількість незалежних підтверджень;
  * кожен кандидат має свій діапазон експірації, прив'язаний до ТФ.
"""
import logging
from datetime import datetime, timedelta

import pandas as pd

from indicators import AdaptiveTechnicalAnalysis
from adaptive_optimizer import calculate_smart_expiration
from timeframe_profiles import PROFILES, ACTIVE_PROFILES, REVERSAL_STRATEGIES, TF_MINUTES

logger = logging.getLogger(__name__)
_analyzer = AdaptiveTechnicalAnalysis()


# ───────────────────────── закриті свічки ─────────────────────────
def closed_only(df: pd.DataFrame, tf_key: str, now: datetime) -> pd.DataFrame:
    """Залишає лише свічки, які вже повністю закрилися на момент now (UTC, naive)."""
    if df is None or df.empty:
        return pd.DataFrame()
    delta = timedelta(minutes=TF_MINUTES.get(tf_key, 1))
    try:
        mask = (df.index + delta) <= now
        return df[mask]
    except Exception:
        # індекс без часу — відкидаємо лише останній рядок
        return df.iloc[:-1]


def build_closed_frames(frames: dict, now: datetime) -> dict:
    out = {}
    for key, df in frames.items():
        if key == "1d":
            out[key] = df if df is not None else pd.DataFrame()
        else:
            out[key] = closed_only(df, key, now)
    return out


# ───────────────────────── підтвердження ─────────────────────────
def _slope_ok(df: pd.DataFrame, direction: str, lookback: int = 3) -> bool:
    if df is None or df.empty or "ema_20" not in df.columns or len(df) <= lookback:
        return False
    now_v, prev_v = float(df["ema_20"].iloc[-1]), float(df["ema_20"].iloc[-1 - lookback])
    return now_v > prev_v if direction == "CALL" else now_v < prev_v


def count_confirmations(direction: str, strategy: str, global_trend: str,
                        ind_confirm: pd.DataFrame, ind_mid: pd.DataFrame) -> tuple:
    """
    Повертає (кількість_підтверджень, список_назв). Кожне підтвердження — окремий,
    слабко корельований з основним сигналом чинник.
    """
    hits = []
    want = "BULLISH" if direction == "CALL" else "BEARISH"

    if strategy in REVERSAL_STRATEGIES:
        # відбій: потрібні ознаки того, що ціна на краю діапазону на старшому ТФ
        if ind_confirm is not None and not ind_confirm.empty and "rsi" in ind_confirm.columns:
            rsi_c = float(ind_confirm["rsi"].iloc[-1])
            if (direction == "CALL" and rsi_c < 45) or (direction == "PUT" and rsi_c > 55):
                hits.append("RSI підтверджуючого ТФ на боці відбою")
            if "bb_upper" in ind_confirm.columns:
                last = ind_confirm.iloc[-1]
                width = float(last["bb_upper"] - last["bb_lower"])
                if width > 0:
                    pct_b = float((last["close"] - last["bb_lower"]) / width)
                    if (direction == "CALL" and pct_b < 0.30) or (direction == "PUT" and pct_b > 0.70):
                        hits.append("Ціна на краю BB підтверджуючого ТФ")
        if ind_mid is not None and not ind_mid.empty and "ema_20" in ind_mid.columns and len(ind_mid) > 3:
            # старший ТФ не має сильно йти проти відбою
            ema_now, ema_prev = float(ind_mid["ema_20"].iloc[-1]), float(ind_mid["ema_20"].iloc[-4])
            price = float(ind_mid["close"].iloc[-1])
            drift = (ema_now - ema_prev) / price if price else 0.0
            against = (direction == "CALL" and drift < -0.0004) or (direction == "PUT" and drift > 0.0004)
            if not against:
                hits.append("Старший ТФ не проти відбою")
    else:
        # за трендом / пробій
        if global_trend == want:
            hits.append("Глобальний тренд (EMA200 1H) за сигналом")
        if _slope_ok(ind_confirm, direction):
            hits.append("EMA20 підтверджуючого ТФ за сигналом")
        if ind_mid is not None and not ind_mid.empty and "macd_hist" in ind_mid.columns:
            mh = float(ind_mid["macd_hist"].iloc[-1])
            if (direction == "CALL" and mh > 0) or (direction == "PUT" and mh < 0):
                hits.append("MACD старшого ТФ за сигналом")

    return len(hits), hits


# ───────────────────────── генерація кандидатів ─────────────────────────
def generate_candidates(pair_name: str, frames: dict, now: datetime,
                        strategies_status: dict = None, profiles=None,
                        max_age_sec: float = None, require_fresh: bool = True) -> tuple:
    """
    frames: {"1d","1h","15m","5m","3m","1m"} (будь-які ТФ, що відсутні, пропускаються).
    Повертає (candidates, rejections), де rejections — {profile: причина} для логів.
    """
    profiles = profiles or ACTIVE_PROFILES
    closed = build_closed_frames(frames, now)
    candidates, rejections = [], {}

    for pkey in profiles:
        prof = PROFILES[pkey]
        p_df = closed.get(prof["primary"])
        c_df = closed.get(prof["confirm"])
        m_df = closed.get(prof["mid"])
        h_df = closed.get("1h")

        if p_df is None or p_df.empty or len(p_df) < 30:
            rejections[pkey] = "мало закритих свічок робочого ТФ"
            continue
        if c_df is None or c_df.empty or len(c_df) < 20:
            rejections[pkey] = "мало даних підтверджуючого ТФ"
            continue

        # свіжість останньої закритої свічки
        bar_end = p_df.index[-1] + timedelta(minutes=prof["minutes"])
        age = (now - bar_end).total_seconds()
        limit = max_age_sec if max_age_sec is not None else min(prof["minutes"] * 60, 90)
        if require_fresh and age > limit:
            rejections[pkey] = f"свічка застаріла ({int(age)}с > {int(limit)}с)"
            continue

        try:
            analysis = _analyzer.analyze_all_timeframes(
                closed.get("1d"), h_df, m_df, pd.DataFrame(), c_df, p_df,
                pair_name=pair_name, strategies_status=strategies_status,
            )
        except Exception as e:
            rejections[pkey] = f"помилка аналізу: {e}"
            continue

        direction = analysis.get("signal", "NONE")
        if direction not in ("CALL", "PUT"):
            rejections[pkey] = analysis.get("reason", "немає сигналу")
            continue

        strategy = analysis.get("strategy", "QUANTUM_MATRIX")
        ind_c = _analyzer.calculate_indicators(c_df)
        ind_m = _analyzer.calculate_indicators(m_df) if m_df is not None and not m_df.empty else pd.DataFrame()
        n_conf, conf_names = count_confirmations(
            direction, strategy, analysis.get("global_trend", "NEUTRAL"), ind_c, ind_m
        )
        if n_conf < prof["min_confirmations"]:
            rejections[pkey] = f"підтверджень {n_conf}/{prof['min_confirmations']}"
            continue

        lo, hi = prof["exp_range"]
        smart = calculate_smart_expiration(
            prof["base_exp"], analysis.get("regime", "RANGE_BOUND"),
            analysis.get("vol_ratio_rel", 1.0), tf=pkey,
        )
        exp = int(max(lo, min(hi, smart)))

        analysis = dict(analysis)
        analysis.update({
            "pair": pair_name,
            "tf": pkey,
            "tf_minutes": prof["minutes"],
            "bar_id": f"{pair_name}|{pkey}|{p_df.index[-1]}",
            "bar_end": bar_end,
            "bar_age_sec": age,
            "confirmations": n_conf,
            "confirmation_names": conf_names,
            "exp_range": prof["exp_range"],
            "suggested_exp": exp,
            "ml_min_auto": prof["ml_min_auto"],
            "ml_min_manual": prof["ml_min_manual"],
        })
        candidates.append(analysis)

    # найкращі спочатку: більше підтверджень, потім більше балів
    candidates.sort(key=lambda c: (c["confirmations"], c.get("confluence_count", 0)), reverse=True)
    return candidates, rejections
