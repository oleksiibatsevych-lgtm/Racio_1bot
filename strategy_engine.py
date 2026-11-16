# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import reversion_engine as rev
from trading_sessions import LONDON, NEW_YORK, is_killzone, session_allowed, session_name

logger = logging.getLogger(__name__)

STRATEGY_CATEGORIES = {
    "REV": "↩️ Базові REV (5m)",
    "MEAN": "🧲 Mean Reversion і Канали",
    "TREND": "📈 Trend & MTF Pullback",
    "VOL": "💥 Волатильність і Сесії",
    "DIV": "📉 Дивергенції та Осцилятори",
    "SMC": "🏦 ICT / Smart Money",
    "PA": "🕯 Price Action (Свічки)",
}

GLOBAL_FILTERS = {
    "session_filter": "🌐 Фільтр торгових сесій",
    "regime_filter": "🧭 Фільтр режиму (CHOP/ADX/ATR)",
    "news_filter": "📰 Карантин макроновин (FF)",
    "auto_disable_49": "🛡 Авто-вимкнення за WR%",
}

CONFLUENCE_MODES = {
    "CONC": "🎯 Єдиний консенсус (без конфліктів)",
    "MAJ": "⚖️ Мажоритарний фільтр",
    "IND": "🔀 Незалежний (кожна окремо)",
}

EXPIRY_MODES = {
    "DYN": "⏱ Динамічна (Золота середина)",
    "FIX": "📌 Фіксована (по стратегії)",
}

STRATEGIES_META = {
    # --- REV (3) ---
    "REV_RSI14": {"title": "REV RSI14", "category": "REV", "tf": "5m", "tf_min": 5, "exp": 60, "htf": None, "regime": "ANY", "priority": 100, "weight": 1.40},
    "REV_BB3": {"title": "REV BB3", "category": "REV", "tf": "5m", "tf_min": 5, "exp": 60, "htf": None, "regime": "ANY", "priority": 99, "weight": 1.35},
    "REV_RSI7": {"title": "REV RSI7", "category": "REV", "tf": "5m", "tf_min": 5, "exp": 45, "htf": None, "regime": "ANY", "priority": 98, "weight": 1.25},

    # --- MEAN (15) ---
    "CRSI_5M": {"title": "Connors RSI 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.10},
    "CRSI_1M": {"title": "Connors RSI 1m", "category": "MEAN", "tf": "1m", "tf_min": 1, "exp": 3, "htf": None, "regime": "RANGE", "priority": 72, "weight": 0.85},
    "VALUE_CH_5M": {"title": "Value Chart 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 85, "weight": 1.10},
    "VALUE_CH_1M": {"title": "Value Chart 1m", "category": "MEAN", "tf": "1m", "tf_min": 1, "exp": 3, "htf": None, "regime": "RANGE", "priority": 71, "weight": 0.85},
    "OSC_CONS_5M": {"title": "RSI+BB+Stoch 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 87, "weight": 1.15},
    "OSC_CONS_3M": {"title": "RSI+BB+Stoch 3m", "category": "MEAN", "tf": "3m", "tf_min": 3, "exp": 6, "htf": None, "regime": "RANGE", "priority": 78, "weight": 0.95},
    "BB_KC_REV_5M": {"title": "BB+KC Envelope 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": 60, "regime": "RANGE", "priority": 84, "weight": 1.15},
    "BB_KC_REV_15M": {"title": "BB+KC Envelope 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": 60, "regime": "RANGE", "priority": 89, "weight": 1.25},
    "PIVOT_REV_5M": {"title": "Daily Pivot S/R 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 83, "weight": 1.05},
    "EXH_STRK_1M": {"title": "Streak Exhaustion 1m", "category": "MEAN", "tf": "1m", "tf_min": 1, "exp": 3, "htf": None, "regime": "RANGE", "priority": 70, "weight": 0.85},
    "EXH_STRK_3M": {"title": "Streak Exhaustion 3m", "category": "MEAN", "tf": "3m", "tf_min": 3, "exp": 6, "htf": None, "regime": "RANGE", "priority": 76, "weight": 0.95},
    "STARC_5M": {"title": "STARC Bands 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.15},
    "STARC_15M": {"title": "STARC Bands 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "RANGE", "priority": 90, "weight": 1.25},
    "LINREG_5M": {"title": "LinReg Channel 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 85, "weight": 1.10},
    "HUBBA_5M": {"title": "Hubba Reversal 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 84, "weight": 1.10},

    # --- TREND (14) ---
    "MTF_PB_1M_15": {"title": "MTF Pullback 1m/15m", "category": "TREND", "tf": "1m", "tf_min": 1, "exp": 5, "htf": 15, "regime": "TREND", "priority": 75, "weight": 0.90},
    "MTF_PB_3M_15": {"title": "MTF Pullback 3m/15m", "category": "TREND", "tf": "3m", "tf_min": 3, "exp": 10, "htf": 15, "regime": "TREND", "priority": 81, "weight": 1.05},
    "MTF_PB_5M_60": {"title": "MTF Pullback 5m/60m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": 60, "regime": "TREND", "priority": 90, "weight": 1.20},
    "EMA_RIB_3M": {"title": "EMA Ribbon 3m", "category": "TREND", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "TREND", "priority": 77, "weight": 0.95},
    "EMA_RIB_5M": {"title": "EMA Ribbon 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 84, "weight": 1.10},
    "MACD_ADX_5M": {"title": "MACD+ADX Trend 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 85, "weight": 1.15},
    "MACD_ADX_15M": {"title": "MACD+ADX Trend 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 91, "weight": 1.25},
    "ST_MOM_3M": {"title": "Dual Supertrend 3m", "category": "TREND", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "TREND", "priority": 79, "weight": 1.00},
    "ST_MOM_5M": {"title": "Dual Supertrend 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 86, "weight": 1.15},
    "PSAR_EMA_5M": {"title": "PSAR+EMA200 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 86, "weight": 1.15},
    "PSAR_EMA_15M": {"title": "PSAR+EMA200 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 90, "weight": 1.25},
    "ICHI_TK_5M": {"title": "Ichimoku Kumo+TK 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 88, "weight": 1.15},
    "ICHI_TK_15M": {"title": "Ichimoku Kumo+TK 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 92, "weight": 1.25},
    "HA_EMA_5M": {"title": "Heikin Ashi+EMA50 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 83, "weight": 1.05},

    # --- VOL (6) ---
    "TTM_SQZ_5M": {"title": "TTM Squeeze 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 92, "weight": 1.20},
    "TTM_SQZ_15M": {"title": "TTM Squeeze 15m", "category": "VOL", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "BREAKOUT", "priority": 94, "weight": 1.30},
    "ORB_LON_5M": {"title": "ORB London 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 93, "weight": 1.20},
    "ORB_NY_5M": {"title": "ORB New York 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 93, "weight": 1.20},
    "DONCH_BRK_5M": {"title": "Donchian Breakout 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 82, "weight": 1.05},
    "JUDAS_LON_5M": {"title": "London Judas Swing 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 91, "weight": 1.20},

    # --- DIV (10) ---
    "DIV_RSI_5M": {"title": "RSI Divergence 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 91, "weight": 1.20},
    "DIV_RSI_15M": {"title": "RSI Divergence 15m", "category": "DIV", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "PA", "priority": 94, "weight": 1.30},
    "DIV_MACD_5M": {"title": "MACD Divergence 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 90, "weight": 1.15},
    "DIV_MACD_15M": {"title": "MACD Divergence 15m", "category": "DIV", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "PA", "priority": 93, "weight": 1.25},
    "AO_TWIN_5M": {"title": "AO Twin Peaks 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 87, "weight": 1.10},
    "AO_ZERO_3M": {"title": "AO Zero Cross 3m", "category": "DIV", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "TREND", "priority": 80, "weight": 0.95},
    "CCI_WPR_5M": {"title": "CCI + Williams %R 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 87, "weight": 1.10},
    "CCI_WPR_3M": {"title": "CCI + Williams %R 3m", "category": "DIV", "tf": "3m", "tf_min": 3, "exp": 6, "htf": None, "regime": "RANGE", "priority": 79, "weight": 0.95},
    "DEM_BB_5M": {"title": "DeMarker + BB 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.10},
    "WT_CROSS_5M": {"title": "WaveTrend Cross 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 88, "weight": 1.15},

    # --- SMC (4) ---
    "LIQ_SWP_3M": {"title": "ICT Liquidity Sweep 3m", "category": "SMC", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "SMC", "priority": 80, "weight": 1.00},
    "LIQ_SWP_5M": {"title": "ICT Liquidity Sweep 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "SMC", "priority": 88, "weight": 1.15},
    "FVG_IMP_3M": {"title": "ICT Fair Value Gap 3m", "category": "SMC", "tf": "3m", "tf_min": 3, "exp": 9, "htf": 15, "regime": "SMC", "priority": 80, "weight": 1.00},
    "FVG_IMP_5M": {"title": "ICT Fair Value Gap 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": 60, "regime": "SMC", "priority": 89, "weight": 1.20},

    # --- PA (5) ---
    "PA_PIN_BB_5M": {"title": "PA Pin Bar + BB 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 84, "weight": 1.10},
    "PA_ENG_SR_5M": {"title": "PA Engulfing S/R 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 85, "weight": 1.10},
    "PA_FAKEY_5M": {"title": "PA Fakey Trap 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 88, "weight": 1.15},
    "PA_MHI_1M": {"title": "MHI 1m Minority", "category": "PA", "tf": "1m", "tf_min": 1, "exp": 1, "htf": None, "regime": "RANGE", "priority": 68, "weight": 0.80},
    "PA_3LS_5M": {"title": "Three Line Strike 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 85, "weight": 1.10},
}

ALL_STRATEGY_KEYS = list(STRATEGIES_META.keys())


def is_known_strategy(cand: dict) -> bool:
    return cand.get("strategy") in STRATEGIES_META


def get_strategy_expiration(cand: dict) -> int:
    if cand.get("suggested_exp"):
        return int(cand["suggested_exp"])
    meta = STRATEGIES_META.get(cand.get("strategy", ""))
    return int(meta["exp"]) if meta else 15


def get_candidate_weight(cand: dict, strategy_weights: dict | None = None) -> float:
    strat = cand.get("strategy", "")
    if strategy_weights and strat in strategy_weights:
        base_w = float(strategy_weights[strat])
    elif "weight" in cand and cand["weight"] is not None:
        base_w = float(cand["weight"])
    else:
        base_w = float(STRATEGIES_META.get(strat, {}).get("weight", 1.0))
    if cand.get("confirmations", 0) > 0:
        base_w += 0.20
    return base_w


def calculate_dynamic_expiration(candidates: list[dict], expiry_mode: str = "DYN", strategy_weights: dict | None = None) -> int:
    """
    Розраховує 'Золоту середину' часу експірації як зважене середнє між
    власною експірацією кожної стратегії та 3*TF, скориговане на волатильність (vol_ratio).
    """
    if not candidates:
        return 15
    if str(expiry_mode).upper() == "FIX":
        return get_strategy_expiration(candidates[0])

    if len(candidates) == 1 and int(candidates[0].get("suggested_exp", 15)) == 1 and int(candidates[0].get("tf_minutes", 5)) == 1:
        return 1

    vals = []
    weights = []
    for c in candidates:
        base_exp = int(c.get("suggested_exp") or STRATEGIES_META.get(c.get("strategy", ""), {}).get("exp", 15))
        tf_min = int(c.get("tf_minutes") or STRATEGIES_META.get(c.get("strategy", ""), {}).get("tf_min", 5))
        golden = (base_exp + tf_min * 3.0) / 2.0
        vr = float(c.get("vol_ratio") or 1.0)
        if vr > 1.3:
            golden *= 0.8
        elif vr < 0.7:
            golden *= 1.2
        w = get_candidate_weight(c, strategy_weights)
        vals.append(golden * w)
        weights.append(w)

    res = int(round(sum(vals) / max(sum(weights), 1e-9)))
    return max(3, min(res, 60))


def resolve_confluence(
    candidates: list[dict],
    conflict_mode: str = "CONC",
    threshold: float = 1.0,
    expiry_mode: str = "DYN",
    strategy_weights: dict | None = None,
) -> tuple[list[dict], str]:
    """
    Усуває суперечливі CALL/PUT сигнали по одній парі та оцінює зважений консенсус.
    Повертає (відібрані кандидати, текстовий статус).
    """
    if not candidates:
        return [], "Немає вхідних умов від стратегій"

    mode = str(conflict_mode or "CONC").upper()
    exp_mode = str(expiry_mode or "DYN").upper()
    th = float(threshold or 1.0)

    if mode == "IND":
        out = []
        for c in candidates:
            c2 = dict(c)
            if exp_mode == "DYN":
                c2["suggested_exp"] = calculate_dynamic_expiration([c2], "DYN", strategy_weights)
                c2["exp_mode"] = "DYN"
            else:
                c2["exp_mode"] = "FIX"
            out.append(c2)
        return out, "OK"

    calls = [c for c in candidates if c.get("signal") == "CALL"]
    puts = [c for c in candidates if c.get("signal") == "PUT"]

    call_score = sum(get_candidate_weight(c, strategy_weights) for c in calls)
    put_score = sum(get_candidate_weight(c, strategy_weights) for c in puts)
    net = call_score - put_score

    if calls and puts and (abs(net) < 1e-9 or (len(calls) == len(puts) and abs(net) < 0.35)):
        return [], f"Конфлікт/нічия: {len(calls)} CALL ({call_score:.1f}) vs {len(puts)} PUT ({put_score:.1f})"

    if abs(net) < th:
        return [], f"Слабка конфлюентність: |{net:+.2f}| < порогу {th:.2f} ({len(calls)} CALL / {len(puts)} PUT)"

    win_dir = "CALL" if net > 0 else "PUT"
    winners = calls if net > 0 else puts
    losers = puts if net > 0 else calls

    winners = sorted(
        winners,
        key=lambda c: (
            get_candidate_weight(c, strategy_weights),
            c.get("priority", STRATEGIES_META.get(c.get("strategy", ""), {}).get("priority", 50)),
        ),
        reverse=True,
    )

    if mode == "MAJ":
        out = []
        for c in winners:
            c2 = dict(c)
            if exp_mode == "DYN":
                c2["suggested_exp"] = calculate_dynamic_expiration([c2], "DYN", strategy_weights)
                c2["exp_mode"] = "DYN"
            else:
                c2["exp_mode"] = "FIX"
            c2["call_votes"] = len(calls)
            c2["put_votes"] = len(puts)
            c2["confluence_score"] = round(abs(net), 2)
            out.append(c2)
        return out, "OK"

    # CONC: об'єднуємо переможців у єдину авторитетну картку
    primary = dict(winners[0])
    primary["signal"] = win_dir
    primary["confluence_score"] = round(abs(net), 2)
    primary["net_score"] = round(net, 2)
    primary["call_votes"] = len(calls)
    primary["put_votes"] = len(puts)
    primary["call_score"] = round(call_score, 2)
    primary["put_score"] = round(put_score, 2)
    primary["agreeing_strategies"] = [c["strategy"] for c in winners]
    primary["opposing_strategies"] = [c["strategy"] for c in losers]
    primary["confirmations"] = len(winners)
    if exp_mode == "DYN":
        primary["suggested_exp"] = calculate_dynamic_expiration(winners, "DYN", strategy_weights)
        primary["exp_mode"] = "DYN"
    else:
        primary["suggested_exp"] = get_strategy_expiration(primary)
        primary["exp_mode"] = "FIX"
    return [primary], "OK"


def load_all_strategy_config(path) -> dict:
    defaults = {k: True for k in ALL_STRATEGY_KEYS}
    defaults["session_filter"] = True
    defaults["regime_filter"] = True
    defaults["news_filter"] = True
    defaults["auto_disable_49"] = True
    p = Path(path)
    if not p.exists():
        return defaults
    try:
        saved = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(saved, dict):
            return defaults
        return {
            key: saved.get(key, val) if isinstance(saved.get(key, val), bool) else val
            for key, val in defaults.items()
        }
    except (OSError, ValueError, TypeError, AttributeError):
        logger.exception("Помилка читання конфігурації стратегій")
        return defaults


def wilder_rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0).ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    down = (-delta).clip(lower=0.0).ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    denom = up + down
    rsi = pd.Series(np.where(denom > 1e-12, 100.0 * up / denom, 50.0), index=series.index)
    rsi.iloc[:length] = np.nan
    return rsi


def compute_true_range(df: pd.DataFrame) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    if len(tr):
        tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def compute_choppiness(df: pd.DataFrame, tr: pd.Series, length: int = 14) -> pd.Series:
    sum_tr = tr.rolling(length, min_periods=length).sum()
    hh = df["high"].rolling(length, min_periods=length).max()
    ll = df["low"].rolling(length, min_periods=length).min()
    span = (hh - ll).replace(0.0, np.nan)
    ratio = (sum_tr / span).clip(lower=1e-9)
    chop = 100.0 * np.log10(ratio) / np.log10(float(length))
    return chop.fillna(50.0).clip(0.0, 100.0)


def compute_adx_di(df: pd.DataFrame, tr: pd.Series, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    dn_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)

    atr = tr.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean() / (atr + 1e-12)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx = dx.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    return adx, plus_di, minus_di


def compute_connors_rsi(close: pd.Series, rsi_len: int = 3, streak_len: int = 2, rank_len: int = 100) -> pd.Series:
    rsi_price = wilder_rsi(close, rsi_len)
    diff = close.diff().to_numpy()
    streak = np.zeros(len(close), dtype=float)
    for i in range(1, len(close)):
        if diff[i] > 0:
            streak[i] = streak[i - 1] + 1.0 if streak[i - 1] > 0 else 1.0
        elif diff[i] < 0:
            streak[i] = streak[i - 1] - 1.0 if streak[i - 1] < 0 else -1.0
        else:
            streak[i] = 0.0
    rsi_streak = wilder_rsi(pd.Series(streak, index=close.index), streak_len)

    roc = close.pct_change(1).to_numpy() * 100.0
    pct_rank = np.full(len(close), np.nan, dtype=float)
    if len(roc) > rank_len:
        win = sliding_window_view(roc, rank_len + 1)
        prev = win[:, :-1]
        curr = win[:, -1:]
        cnt_valid = np.sum(~np.isnan(prev), axis=1)
        cnt_less = np.nansum(prev < curr, axis=1)
        valid_mask = (~np.isnan(curr[:, 0])) & (cnt_valid >= rank_len // 2)
        ranks = np.where(cnt_valid > 0, (cnt_less / np.maximum(cnt_valid, 1)) * 100.0, np.nan)
        pct_rank[rank_len:] = np.where(valid_mask, ranks, np.nan)
    return (rsi_price + rsi_streak + pd.Series(pct_rank, index=close.index)) / 3.0


def compute_value_chart(df: pd.DataFrame, length: int = 5) -> tuple[pd.Series, pd.Series, pd.Series]:
    hl2 = (df["high"] + df["low"]) / 2.0
    var_p = hl2.rolling(length, min_periods=length).mean()
    l_range = ((df["high"] - df["low"]).rolling(length, min_periods=length).mean() * 0.2).replace(0.0, np.nan)
    return (
        (df["high"] - var_p) / l_range,
        (df["low"] - var_p) / l_range,
        (df["close"] - var_p) / l_range,
    )


def compute_supertrend(df: pd.DataFrame, tr: pd.Series, atr_len: int = 7, mult: float = 2.0) -> np.ndarray:
    n = len(df)
    if n < atr_len + 2:
        return np.zeros(n, dtype=np.int8)
    atr = tr.ewm(alpha=1.0 / atr_len, min_periods=atr_len, adjust=False).mean().to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy()
    close = df["close"].to_numpy()

    basic_ub = hl2 + mult * atr
    basic_lb = hl2 - mult * atr
    final_ub = np.copy(basic_ub)
    final_lb = np.copy(basic_lb)
    direction = np.zeros(n, dtype=np.int8)

    start = atr_len
    direction[start] = 1
    for i in range(start + 1, n):
        if np.isnan(atr[i]):
            continue
        if basic_ub[i] < final_ub[i - 1] or close[i - 1] > final_ub[i - 1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i - 1]

        if basic_lb[i] > final_lb[i - 1] or close[i - 1] < final_lb[i - 1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i - 1]

        prev_d = direction[i - 1]
        if prev_d == 1:
            direction[i] = -1 if close[i] < final_lb[i] else 1
        elif prev_d == -1:
            direction[i] = 1 if close[i] > final_ub[i] else -1
        else:
            direction[i] = 1 if close[i] >= final_lb[i] else -1
    return direction


def compute_ttm_squeeze(df: pd.DataFrame, tr: pd.Series, length: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5) -> tuple[pd.Series, pd.Series]:
    sma_c = df["close"].rolling(length, min_periods=length).mean()
    std_c = df["close"].rolling(length, min_periods=length).std(ddof=0)
    bb_up = sma_c + bb_mult * std_c
    bb_lo = sma_c - bb_mult * std_c

    atr_kc = tr.rolling(length, min_periods=length).mean()
    kc_up = sma_c + kc_mult * atr_kc
    kc_lo = sma_c - kc_mult * atr_kc
    sqz_on = (bb_lo > kc_lo) & (bb_up < kc_up)

    donch_mid = (df["high"].rolling(length, min_periods=length).max() + df["low"].rolling(length, min_periods=length).min()) / 2.0
    delta_src = (df["close"] - (donch_mid + sma_c) / 2.0).to_numpy()

    mom = np.full(len(df), np.nan, dtype=float)
    if len(delta_src) >= length:
        k = np.arange(length, dtype=float)
        weights = (1.0 / length) + (6.0 * (k - (length - 1.0) / 2.0)) / (length * (length + 1.0))
        win = sliding_window_view(delta_src, length)
        mom[length - 1:] = np.einsum("ij,j->i", win, weights)
    return sqz_on, pd.Series(mom, index=df.index)


def compute_psar_dir(high: np.ndarray, low: np.ndarray, af_step: float = 0.02, af_max: float = 0.20) -> np.ndarray:
    n = len(high)
    dirs = np.zeros(n, dtype=np.int8)
    if n < 3:
        return dirs
    bull = True
    sar = low[0]
    ep = high[0]
    af = af_step
    dirs[0] = 1
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            if i >= 2:
                sar = min(sar, low[i - 1], low[i - 2])
            else:
                sar = min(sar, low[i - 1])
            if low[i] < sar:
                bull = False
                sar = ep
                ep = low[i]
                af = af_step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            if i >= 2:
                sar = max(sar, high[i - 1], high[i - 2])
            else:
                sar = max(sar, high[i - 1])
            if high[i] > sar:
                bull = True
                sar = ep
                ep = high[i]
                af = af_step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        dirs[i] = 1 if bull else -1
    return dirs


def compute_linreg_channel(close: np.ndarray, atr: np.ndarray, length: int = 50) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    z = np.full(n, np.nan, dtype=float)
    slope_norm = np.full(n, np.nan, dtype=float)
    if n < length:
        return z, slope_norm
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    dx = x - x_mean
    dx_ss = float(np.sum(dx ** 2))
    win = sliding_window_view(close, length)
    y_mean = win.mean(axis=1, keepdims=True)
    slope = np.sum((win - y_mean) * dx, axis=1) / dx_ss
    intercept = y_mean[:, 0] - slope * x_mean
    fitted = slope[:, None] * x[None, :] + intercept[:, None]
    std_res = np.std(win - fitted, axis=1, ddof=0)
    z[length - 1:] = (win[:, -1] - fitted[:, -1]) / (std_res + 1e-12)
    slope_norm[length - 1:] = (slope * length) / (atr[length - 1:] + 1e-12)
    return z, slope_norm


def compute_cci(df: pd.DataFrame, length: int = 20) -> np.ndarray:
    tp = ((df["high"] + df["low"] + df["close"]) / 3.0).to_numpy(dtype=float)
    n = len(tp)
    cci = np.full(n, np.nan, dtype=float)
    if n < length:
        return cci
    win = sliding_window_view(tp, length)
    sma = win.mean(axis=1)
    mad = np.mean(np.abs(win - sma[:, None]), axis=1)
    cci[length - 1:] = (tp[length - 1:] - sma) / (0.015 * mad + 1e-12)
    return cci


def compute_demarker(df: pd.DataFrame, length: int = 14) -> np.ndarray:
    dh = df["high"].diff().clip(lower=0.0)
    dl = (-df["low"].diff()).clip(lower=0.0)
    num = dh.rolling(length, min_periods=length).mean()
    den = num + dl.rolling(length, min_periods=length).mean()
    return (num / (den + 1e-12)).to_numpy(dtype=float)


def compute_wavetrend(df: pd.DataFrame, n1: int = 10, n2: int = 21) -> tuple[np.ndarray, np.ndarray]:
    ap = (df["high"] + df["low"] + df["close"]) / 3.0
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d + 1e-12)
    wt1 = ci.ewm(span=n2, adjust=False).mean()
    wt2 = wt1.rolling(4, min_periods=4).mean()
    return wt1.to_numpy(dtype=float), wt2.to_numpy(dtype=float)


def compute_heikin_ashi(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return ha_o, ha_c


def compute_daily_pivots(df: pd.DataFrame) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    dates = idx.date
    daily = df.groupby(dates).agg({"high": "max", "low": "min", "close": "last"})
    prev = daily.shift(1)
    p = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    r1 = 2.0 * p - prev["low"]
    s1 = 2.0 * p - prev["high"]
    r2 = p + (prev["high"] - prev["low"])
    s2 = p - (prev["high"] - prev["low"])
    mapper = pd.DataFrame({"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2})
    mapped = mapper.reindex(dates)
    return {col: mapped[col].to_numpy() for col in ("P", "R1", "S1", "R2", "S2")}


def compute_base_indicators(df: pd.DataFrame) -> dict:
    tr = compute_true_range(df)
    atr14 = tr.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    atr20 = tr.ewm(alpha=1.0 / 20.0, min_periods=20, adjust=False).mean()
    atr_med = atr14.rolling(200, min_periods=40).median()
    vol_ratio = atr14 / (atr_med + 1e-12)

    close = df["close"]
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    sma6 = close.rolling(6, min_periods=6).mean()
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    sma24 = close.rolling(24, min_periods=24).mean()
    std24 = close.rolling(24, min_periods=24).std(ddof=0)
    sma30 = close.rolling(30, min_periods=30).mean()
    std30 = close.rolling(30, min_periods=30).std(ddof=0)

    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig

    rsi7 = wilder_rsi(close, 7)
    rsi14 = wilder_rsi(close, 14)
    crsi = compute_connors_rsi(close)
    vc_h, vc_l, vc_c = compute_value_chart(df, 5)
    chop = compute_choppiness(df, tr, 14)
    adx, plus_di, minus_di = compute_adx_di(df, tr, 14)

    lo14 = df["low"].rolling(14, min_periods=14).min()
    hi14 = df["high"].rolling(14, min_periods=14).max()
    stoch_k = (100.0 * (close - lo14) / ((hi14 - lo14) + 1e-12)).rolling(3).mean()
    stoch_d = stoch_k.rolling(3).mean()
    wpr14 = -100.0 * (hi14 - close) / ((hi14 - lo14) + 1e-12)

    lo3 = df["low"].rolling(3, min_periods=3).min()
    hi3 = df["high"].rolling(3, min_periods=3).max()
    stoch3_k = (100.0 * (close - lo3) / ((hi3 - lo3) + 1e-12)).rolling(3).mean()
    stoch3_d = stoch3_k.rolling(3).mean()

    hl2 = (df["high"] + df["low"]) / 2.0
    ao = hl2.rolling(5, min_periods=5).mean() - hl2.rolling(34, min_periods=34).mean()

    tenkan = (df["high"].rolling(9, min_periods=9).max() + df["low"].rolling(9, min_periods=9).min()) / 2.0
    kijun = (df["high"].rolling(26, min_periods=26).max() + df["low"].rolling(26, min_periods=26).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = ((df["high"].rolling(52, min_periods=52).max() + df["low"].rolling(52, min_periods=52).min()) / 2.0).shift(26)

    o_np = df["open"].to_numpy(dtype=float)
    h_np = df["high"].to_numpy(dtype=float)
    l_np = df["low"].to_numpy(dtype=float)
    c_np = close.to_numpy(dtype=float)
    atr14_np = atr14.to_numpy(dtype=float)

    st_fast = compute_supertrend(df, tr, 7, 2.0)
    st_slow = compute_supertrend(df, tr, 10, 3.0)
    sqz_on, sqz_mom = compute_ttm_squeeze(df, tr, 20, 2.0, 1.5)
    psar_dir = compute_psar_dir(h_np, l_np)
    lr_z, lr_slope = compute_linreg_channel(c_np, atr14_np, 50)
    cci20 = compute_cci(df, 20)
    dem14 = compute_demarker(df, 14)
    wt1, wt2 = compute_wavetrend(df, 10, 21)
    ha_o, ha_c = compute_heikin_ashi(o_np, h_np, l_np, c_np)
    pivots = compute_daily_pivots(df)

    return {
        "o": o_np, "h": h_np, "l": l_np, "c": c_np,
        "atr14": atr14_np,
        "atr20": atr20.to_numpy(dtype=float),
        "vol_ratio": vol_ratio.to_numpy(dtype=float),
        "ema8": ema8.to_numpy(dtype=float),
        "ema21": ema21.to_numpy(dtype=float),
        "ema50": ema50.to_numpy(dtype=float),
        "ema200": ema200.to_numpy(dtype=float),
        "sma6": sma6.to_numpy(dtype=float),
        "sma20": sma20.to_numpy(dtype=float),
        "std20": std20.to_numpy(dtype=float),
        "sma24": sma24.to_numpy(dtype=float),
        "std24": std24.to_numpy(dtype=float),
        "z30": ((close - sma30) / (std30 + 1e-12)).to_numpy(dtype=float),
        "kc_up_20": (ema21 + 2.0 * atr20).to_numpy(dtype=float),
        "kc_lo_20": (ema21 - 2.0 * atr20).to_numpy(dtype=float),
        "macd_hist": macd_hist.to_numpy(dtype=float),
        "rsi7": rsi7.to_numpy(dtype=float),
        "rsi14": rsi14.to_numpy(dtype=float),
        "crsi": crsi.to_numpy(dtype=float),
        "vc_h": vc_h.to_numpy(dtype=float),
        "vc_l": vc_l.to_numpy(dtype=float),
        "vc_c": vc_c.to_numpy(dtype=float),
        "chop": chop.to_numpy(dtype=float),
        "adx": adx.to_numpy(dtype=float),
        "plus_di": plus_di.to_numpy(dtype=float),
        "minus_di": minus_di.to_numpy(dtype=float),
        "stoch_k": stoch_k.to_numpy(dtype=float),
        "stoch_d": stoch_d.to_numpy(dtype=float),
        "stoch3_k": stoch3_k.to_numpy(dtype=float),
        "stoch3_d": stoch3_d.to_numpy(dtype=float),
        "wpr14": wpr14.to_numpy(dtype=float),
        "ao": ao.to_numpy(dtype=float),
        "tenkan": tenkan.to_numpy(dtype=float),
        "kijun": kijun.to_numpy(dtype=float),
        "cloud_top": np.maximum(senkou_a.to_numpy(dtype=float), senkou_b.to_numpy(dtype=float)),
        "cloud_bot": np.minimum(senkou_a.to_numpy(dtype=float), senkou_b.to_numpy(dtype=float)),
        "st_fast": st_fast,
        "st_slow": st_slow,
        "psar_dir": psar_dir,
        "lr_z": lr_z,
        "lr_slope": lr_slope,
        "cci20": cci20,
        "dem14": dem14,
        "wt1": wt1,
        "wt2": wt2,
        "ha_o": ha_o,
        "ha_c": ha_c,
        "sqz_on": sqz_on.fillna(False).to_numpy(dtype=bool),
        "sqz_mom": sqz_mom.to_numpy(dtype=float),
        "pivots": pivots,
        "donch_hi20": df["high"].rolling(20, min_periods=20).max().shift(1).to_numpy(dtype=float),
        "donch_lo20": df["low"].rolling(20, min_periods=20).min().shift(1).to_numpy(dtype=float),
        "swing_hi20": df["high"].rolling(20, min_periods=20).max().shift(2).to_numpy(dtype=float),
        "swing_lo20": df["low"].rolling(20, min_periods=20).min().shift(2).to_numpy(dtype=float),
        "index": df.index,
    }


def _minutes_of(index: pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.to_numpy().astype("datetime64[m]").astype(np.int64)


def htf_alignment(df_target: pd.DataFrame, df_htf: pd.DataFrame | None, tf_min: int, htf_min: int) -> dict[str, np.ndarray]:
    n = len(df_target)
    zero_trend = np.zeros(n, dtype=np.int8)
    mid_rsi = np.full(n, 50.0, dtype=float)
    if df_htf is None or len(df_htf) < 60:
        return {"trend": zero_trend, "rsi": mid_rsi}

    close = df_htf["close"].astype(float)
    e21 = close.ewm(span=21, adjust=False).mean().to_numpy()
    e50 = close.ewm(span=50, adjust=False).mean().to_numpy()
    c = close.to_numpy()
    trend = np.where((e21 > e50) & (c > e50), 1, np.where((e21 < e50) & (c < e50), -1, 0)).astype(np.int8)
    rsi = wilder_rsi(close, 14).fillna(50.0).to_numpy()

    target_end = _minutes_of(df_target.index) + tf_min
    htf_end = _minutes_of(df_htf.index) + htf_min
    j = np.searchsorted(htf_end, target_end, side="right") - 1
    ok = j >= 0
    jc = np.maximum(j, 0)
    return {
        "trend": np.where(ok, trend[jc], 0).astype(np.int8),
        "rsi": np.where(ok, rsi[jc], 50.0),
    }


def _prev(arr: np.ndarray, k: int = 1) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) > k:
        out[k:] = arr[:-k]
    return out


def _sig_array(call_mask: np.ndarray, put_mask: np.ndarray) -> np.ndarray:
    c = np.asarray(call_mask, dtype=bool)
    p = np.asarray(put_mask, dtype=bool)
    out = np.zeros(len(c), dtype=np.int8)
    out[c] = 1
    out[p & ~c] = -1
    return out


def _non_repainting_divergence(
    ind: dict,
    osc: np.ndarray,
    bull_max_osc: float | None = None,
    bear_min_osc: float | None = None,
    zero_split: bool = False,
    left: int = 5,
    right: int = 2,
) -> np.ndarray:
    h, l, o, c = ind["h"], ind["l"], ind["o"], ind["c"]
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    low_pivots = []
    high_pivots = []

    for i in range(left + right, n):
        p = i - right
        if np.isnan(osc[p]) or np.isnan(osc[i]) or np.isnan(osc[i - 1]):
            continue

        wl = l[p - left : p + right + 1]
        if l[p] == np.min(wl) and np.sum(wl == l[p]) == 1:
            low_pivots.append((p, l[p], osc[p]))
            if len(low_pivots) >= 2:
                pp, pl, po = low_pivots[-2]
                if 5 <= (p - pp) <= 45 and l[p] < pl and osc[p] > po:
                    ok_zone = True
                    if bull_max_osc is not None and (osc[p] > bull_max_osc or po > bull_max_osc):
                        ok_zone = False
                    if zero_split and (osc[p] >= 0.0 or po >= 0.0):
                        ok_zone = False
                    if ok_zone and osc[i] > osc[i - 1] and c[i] > o[i]:
                        out[i] = 1

        wh = h[p - left : p + right + 1]
        if h[p] == np.max(wh) and np.sum(wh == h[p]) == 1:
            high_pivots.append((p, h[p], osc[p]))
            if len(high_pivots) >= 2:
                pp, ph, po = high_pivots[-2]
                if 5 <= (p - pp) <= 45 and h[p] > ph and osc[p] < po:
                    ok_zone = True
                    if bear_min_osc is not None and (osc[p] < bear_min_osc or po < bear_min_osc):
                        ok_zone = False
                    if zero_split and (osc[p] <= 0.0 or po <= 0.0):
                        ok_zone = False
                    if ok_zone and osc[i] < osc[i - 1] and c[i] < o[i]:
                        if out[i] == 0:
                            out[i] = -1
    return out


def _ao_twin_peaks(ind: dict) -> np.ndarray:
    ao, o, c = ind["ao"], ind["o"], ind["c"]
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    troughs = []
    peaks = []
    for i in range(40, n):
        p = i - 1
        if np.isnan(ao[i]) or np.isnan(ao[p]) or np.isnan(ao[i - 2]):
            continue
        if ao[p] < ao[i - 2] and ao[p] < ao[i] and ao[p] < 0.0:
            troughs.append((p, ao[p]))
            if len(troughs) >= 2:
                pp, pv = troughs[-2]
                if 5 <= (p - pp) <= 35 and ao[p] > pv and np.all(ao[pp : i + 1] < 0.0) and c[i] > o[i]:
                    out[i] = 1
        if ao[p] > ao[i - 2] and ao[p] > ao[i] and ao[p] > 0.0:
            peaks.append((p, ao[p]))
            if len(peaks) >= 2:
                pp, pv = peaks[-2]
                if 5 <= (p - pp) <= 35 and ao[p] < pv and np.all(ao[pp : i + 1] > 0.0) and c[i] < o[i]:
                    if out[i] == 0:
                        out[i] = -1
    return out


def _judas_london_series(ind: dict) -> np.ndarray:
    idx = pd.DatetimeIndex(ind["index"])
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    dates = idx_utc.date
    hours = idx_utc.hour
    o, h, l, c, atr = ind["o"], ind["h"], ind["l"], ind["c"], ind["atr14"]
    n = len(c)
    out = np.zeros(n, dtype=np.int8)

    asia = {}
    for i in range(n):
        if 0 <= hours[i] < 7:
            d = dates[i]
            cur = asia.get(d)
            if cur is None:
                asia[d] = [h[i], l[i], 1]
            else:
                cur[0] = max(cur[0], h[i])
                cur[1] = min(cur[1], l[i])
                cur[2] += 1

    triggered = set()
    for i in range(1, n):
        if not (7 <= hours[i] < 10):
            continue
        d = dates[i]
        rng = asia.get(d)
        if not rng or rng[2] < 12:
            continue
        ah, al, _ = rng
        width = ah - al
        if np.isnan(atr[i]) or atr[i] <= 0 or not (0.8 * atr[i] <= width <= 8.0 * atr[i]):
            continue
        if (d, 1) not in triggered and l[i] < al and c[i] > al and c[i] > o[i]:
            out[i] = 1
            triggered.add((d, 1))
        elif (d, -1) not in triggered and h[i] > ah and c[i] < ah and c[i] < o[i]:
            out[i] = -1
            triggered.add((d, -1))
    return out


def _orb_series(ind: dict, zone, start_h: int, start_m: int, end_h: int) -> np.ndarray:
    idx = pd.DatetimeIndex(ind["index"])
    idx_utc = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    local = idx_utc.tz_convert(zone)
    dates, hours, mins = local.date, local.hour, local.minute

    c, o, h, l, atr = ind["c"], ind["o"], ind["h"], ind["l"], ind["atr14"]
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    or_end_m = start_m + 15
    ranges = {}
    for i in range(n):
        if hours[i] == start_h and start_m <= mins[i] < or_end_m:
            d = dates[i]
            cur = ranges.get(d)
            if cur is None:
                ranges[d] = [h[i], l[i], 1]
            else:
                cur[0] = max(cur[0], h[i])
                cur[1] = min(cur[1], l[i])
                cur[2] += 1

    triggered = set()
    for i in range(1, n):
        d = dates[i]
        rng = ranges.get(d)
        if rng is None or rng[2] < 2:
            continue
        in_window = (hours[i] > start_h or (hours[i] == start_h and mins[i] >= or_end_m)) and (hours[i] < end_h)
        if not in_window:
            continue
        orh, orl, _ = rng
        width = orh - orl
        if np.isnan(atr[i]) or atr[i] <= 0 or not (0.8 * atr[i] <= width <= 6.0 * atr[i]):
            continue
        body = abs(c[i] - o[i])
        bar_rng = max(h[i] - l[i], 1e-12)
        if body < 0.40 * bar_rng:
            continue
        if (d, 1) not in triggered and c[i] > orh and c[i - 1] <= orh:
            out[i] = 1
            triggered.add((d, 1))
        elif (d, -1) not in triggered and c[i] < orl and c[i - 1] >= orl:
            out[i] = -1
            triggered.add((d, -1))
    return out


def _fvg_series(ind: dict, htf_trend: np.ndarray) -> np.ndarray:
    o, h, l, c, atr = ind["o"], ind["h"], ind["l"], ind["c"], ind["atr14"]
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    for i in range(25, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        for k in range(2, 11):
            m = i - k
            if m < 2 or np.isnan(atr[m - 1]):
                continue
            gap_bull = l[m] - h[m - 2]
            if gap_bull >= 0.25 * atr[m - 1] and htf_trend[i] >= 0:
                if np.all(c[m + 1 : i] >= h[m - 2]) and l[i] <= l[m] and c[i] > h[m - 2] and c[i] > o[i]:
                    out[i] = 1
                    break
            gap_bear = l[m - 2] - h[m]
            if gap_bear >= 0.25 * atr[m - 1] and htf_trend[i] <= 0:
                if np.all(c[m + 1 : i] <= l[m - 2]) and h[i] >= h[m] and c[i] < l[m - 2] and c[i] < o[i]:
                    out[i] = -1
                    break
    return out


def _mhi_1m_series(ind: dict) -> np.ndarray:
    idx = pd.DatetimeIndex(ind["index"])
    end_mins = (idx + pd.Timedelta(minutes=1)).minute
    o, c, atr = ind["o"], ind["c"], ind["atr14"]
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    for i in range(10, n):
        if end_mins[i] % 5 != 0 or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if (idx[i] - idx[i - 2]) != pd.Timedelta(minutes=2):
            continue
        greens = 0
        reds = 0
        for j in (i - 2, i - 1, i):
            diff = c[j] - o[j]
            if abs(diff) < 0.05 * atr[i]:
                continue
            if diff > 0:
                greens += 1
            else:
                reds += 1
        if greens + reds < 2 or greens == reds:
            continue
        out[i] = 1 if reds > greens else -1
    return out


def evaluate_strategy_series(
    df: pd.DataFrame,
    strategy_key: str,
    ind: dict | None = None,
    htf_ctx: dict | None = None,
    regime_filter: bool = True,
) -> np.ndarray:
    if strategy_key in rev.PRESETS:
        if len(df) < 201:
            return np.zeros(len(df), dtype=np.int8)
        rev_df = rev.calculate_indicators(df)
        return rev_df[strategy_key].to_numpy(dtype=np.int8)

    n = len(df)
    if n < 60:
        return np.zeros(n, dtype=np.int8)
    if ind is None:
        ind = compute_base_indicators(df)

    o, h, l, c = ind["o"], ind["h"], ind["l"], ind["c"]
    atr14 = ind["atr14"]
    vol_ratio = ind["vol_ratio"]
    chop = ind["chop"]
    adx = ind["adx"]
    rsi7, rsi14 = ind["rsi7"], ind["rsi14"]
    sma20, std20 = ind["sma20"], ind["std20"]
    bb_up20 = sma20 + 2.0 * std20
    bb_lo20 = sma20 - 2.0 * std20

    htf_trend = htf_ctx["trend"] if htf_ctx is not None else np.zeros(n, dtype=np.int8)
    htf_rsi = htf_ctx["rsi"] if htf_ctx is not None else np.full(n, 50.0)
    raw = np.zeros(n, dtype=np.int8)

    # --- MEAN & CHANNELS ---
    if strategy_key in ("CRSI_5M", "CRSI_1M"):
        crsi = ind["crsi"]
        raw = _sig_array((crsi < 10.0) & (c > o), (crsi > 90.0) & (c < o))

    elif strategy_key in ("VALUE_CH_5M", "VALUE_CH_1M"):
        raw = _sig_array((ind["vc_l"] <= -8.0) & (ind["vc_c"] <= -6.0), (ind["vc_h"] >= 8.0) & (ind["vc_c"] >= 6.0))

    elif strategy_key in ("OSC_CONS_5M", "OSC_CONS_3M"):
        k, d = ind["stoch_k"], ind["stoch_d"]
        pk, pd_ = _prev(k), _prev(d)
        not_sqz = ~ind["sqz_on"]
        raw = _sig_array(
            not_sqz & (c < bb_lo20) & (rsi7 < 25.0) & (k < 25.0) & (k > d) & (pk <= pd_),
            not_sqz & (c > bb_up20) & (rsi7 > 75.0) & (k > 75.0) & (k < d) & (pk >= pd_),
        )

    elif strategy_key in ("BB_KC_REV_5M", "BB_KC_REV_15M"):
        bb_up25 = sma20 + 2.5 * std20
        bb_lo25 = sma20 - 2.5 * std20
        calm_htf = (htf_rsi > 38.0) & (htf_rsi < 62.0)
        raw = _sig_array(
            calm_htf & (c < bb_lo25) & (c < ind["kc_lo_20"]) & (ind["z30"] < -2.3),
            calm_htf & (c > bb_up25) & (c > ind["kc_up_20"]) & (ind["z30"] > 2.3),
        )

    elif strategy_key == "PIVOT_REV_5M":
        piv = ind["pivots"]
        s1, s2, r1, r2 = piv["S1"], piv["S2"], piv["R1"], piv["R2"]
        tol = 0.15 * atr14
        bull = ((~np.isnan(s1)) & (l <= s1 + tol) & (c > s1) & (c > o) & (rsi14 < 38.0)) | \
               ((~np.isnan(s2)) & (l <= s2 + tol) & (c > s2) & (c > o) & (rsi14 < 38.0))
        bear = ((~np.isnan(r1)) & (h >= r1 - tol) & (c < r1) & (c < o) & (rsi14 > 62.0)) | \
               ((~np.isnan(r2)) & (h >= r2 - tol) & (c < r2) & (c < o) & (rsi14 > 62.0))
        raw = _sig_array(bull, bear)

    elif strategy_key in ("EXH_STRK_1M", "EXH_STRK_3M"):
        g = pd.Series((c > o).astype(int)).rolling(4).sum().to_numpy() == 4
        r = pd.Series((c < o).astype(int)).rolling(4).sum().to_numpy() == 4
        raw = _sig_array(r & (c < bb_lo20) & (rsi7 < 20.0), g & (c > bb_up20) & (rsi7 > 80.0))

    elif strategy_key in ("STARC_5M", "STARC_15M"):
        starc_up = ind["sma6"] + 2.0 * atr14
        starc_lo = ind["sma6"] - 2.0 * atr14
        raw = _sig_array(
            (l < starc_lo) & (c > starc_lo) & (c > o) & (rsi14 < 38.0),
            (h > starc_up) & (c < starc_up) & (c < o) & (rsi14 > 62.0),
        )

    elif strategy_key == "LINREG_5M":
        flat_ch = np.abs(ind["lr_slope"]) < 2.5
        raw = _sig_array(
            flat_ch & (ind["lr_z"] < -2.0) & (rsi14 < 38.0) & (c > o),
            flat_ch & (ind["lr_z"] > 2.0) & (rsi14 > 62.0) & (c < o),
        )

    elif strategy_key == "HUBBA_5M":
        bb_up24 = ind["sma24"] + 1.5 * ind["std24"]
        bb_lo24 = ind["sma24"] - 1.5 * ind["std24"]
        sk, sd = ind["stoch3_k"], ind["stoch3_d"]
        psk, psd = _prev(sk), _prev(sd)
        raw = _sig_array(
            (c < bb_lo24) & (ind["vc_l"] <= -7.5) & (sk > sd) & (psk <= psd) & (sk < 35.0),
            (c > bb_up24) & (ind["vc_h"] >= 7.5) & (sk < sd) & (psk >= psd) & (sk > 65.0),
        )

    # --- TREND & MTF ---
    elif strategy_key in ("MTF_PB_1M_15", "MTF_PB_3M_15", "MTF_PB_5M_60"):
        e21, e50 = ind["ema21"], ind["ema50"]
        raw = _sig_array(
            (htf_trend == 1) & (c > e50) & (l <= e21 + 0.25 * atr14) & (rsi7 < 35.0) & (c > o),
            (htf_trend == -1) & (c < e50) & (h >= e21 - 0.25 * atr14) & (rsi7 > 65.0) & (c < o),
        )

    elif strategy_key in ("EMA_RIB_3M", "EMA_RIB_5M"):
        e8, e21, e50 = ind["ema8"], ind["ema21"], ind["ema50"]
        pc, pe8 = _prev(c), _prev(e8)
        raw = _sig_array(
            (e8 > e21) & (e21 > e50) & (l <= e8) & (c > e50) & (c > e8) & (pc <= pe8),
            (e8 < e21) & (e21 < e50) & (h >= e8) & (c < e50) & (c < e8) & (pc >= pe8),
        )

    elif strategy_key in ("MACD_ADX_5M", "MACD_ADX_15M"):
        hist, ph, e50 = ind["macd_hist"], _prev(ind["macd_hist"]), ind["ema50"]
        raw = _sig_array(
            (c > e50) & (adx > 22.0) & (ind["plus_di"] > ind["minus_di"]) & (hist > 0.0) & (ph <= 0.0),
            (c < e50) & (adx > 22.0) & (ind["minus_di"] > ind["plus_di"]) & (hist < 0.0) & (ph >= 0.0),
        )

    elif strategy_key in ("ST_MOM_3M", "ST_MOM_5M"):
        stf, sts = ind["st_fast"], ind["st_slow"]
        pstf = _prev(stf.astype(float))
        raw = _sig_array((sts == 1) & (stf == 1) & (pstf == -1), (sts == -1) & (stf == -1) & (pstf == 1))

    elif strategy_key in ("PSAR_EMA_5M", "PSAR_EMA_15M"):
        psar, ppsar = ind["psar_dir"], _prev(ind["psar_dir"].astype(float))
        hist, ph = ind["macd_hist"], _prev(ind["macd_hist"])
        e200 = ind["ema200"]
        psar_flip_bull = (psar == 1) & ((ppsar == -1) | ((hist > 0.0) & (ph <= 0.0)))
        psar_flip_bear = (psar == -1) & ((ppsar == 1) | ((hist < 0.0) & (ph >= 0.0)))
        raw = _sig_array((c > e200) & psar_flip_bull, (c < e200) & psar_flip_bear)

    elif strategy_key in ("ICHI_TK_5M", "ICHI_TK_15M"):
        tk, kj = ind["tenkan"], ind["kijun"]
        ptk, pkj = _prev(tk), _prev(kj)
        c26 = _prev(c, 26)
        raw = _sig_array(
            (c > ind["cloud_top"]) & (tk > kj) & (ptk <= pkj) & (c > c26),
            (c < ind["cloud_bot"]) & (tk < kj) & (ptk >= pkj) & (c < c26),
        )

    elif strategy_key == "HA_EMA_5M":
        hac, hao = ind["ha_c"], ind["ha_o"]
        phac, phao = _prev(hac), _prev(hao)
        sk, sd = ind["stoch_k"], ind["stoch_d"]
        raw = _sig_array(
            (hac > hao) & (phac <= phao) & (c > ind["ema50"]) & (sk > sd) & (sk < 75.0) & (c > o),
            (hac < hao) & (phac >= phao) & (c < ind["ema50"]) & (sk < sd) & (sk > 25.0) & (c < o),
        )

    # --- VOL & SESSIONS ---
    elif strategy_key in ("TTM_SQZ_5M", "TTM_SQZ_15M"):
        sqz, mom, pmom = ind["sqz_on"], ind["sqz_mom"], _prev(ind["sqz_mom"])
        was_sqz = pd.Series(sqz.astype(int)).shift(1).rolling(3).sum().to_numpy() == 3
        released = was_sqz & (~sqz)
        raw = _sig_array(released & (mom > 0.0) & (mom > pmom), released & (mom < 0.0) & (mom < pmom))

    elif strategy_key == "ORB_LON_5M":
        raw = _orb_series(ind, LONDON, 8, 0, 12)

    elif strategy_key == "ORB_NY_5M":
        raw = _orb_series(ind, NEW_YORK, 9, 30, 13)

    elif strategy_key == "DONCH_BRK_5M":
        rng = np.maximum(h - l, 1e-12)
        body = np.abs(c - o)
        pc = _prev(c)
        raw = _sig_array(
            (c > ind["donch_hi20"]) & (pc <= ind["donch_hi20"]) & (body >= 0.55 * rng),
            (c < ind["donch_lo20"]) & (pc >= ind["donch_lo20"]) & (body >= 0.55 * rng),
        )

    elif strategy_key == "JUDAS_LON_5M":
        raw = _judas_london_series(ind)

    # --- DIV & OSCILLATORS ---
    elif strategy_key in ("DIV_RSI_5M", "DIV_RSI_15M"):
        raw = _non_repainting_divergence(ind, rsi14, bull_max_osc=40.0, bear_min_osc=60.0)

    elif strategy_key in ("DIV_MACD_5M", "DIV_MACD_15M"):
        raw = _non_repainting_divergence(ind, ind["macd_hist"], zero_split=True)

    elif strategy_key == "AO_TWIN_5M":
        raw = _ao_twin_peaks(ind)

    elif strategy_key == "AO_ZERO_3M":
        ao, pao = ind["ao"], _prev(ind["ao"])
        raw = _sig_array(
            (ao > 0.0) & (pao <= 0.0) & (c > ind["ema50"]),
            (ao < 0.0) & (pao >= 0.0) & (c < ind["ema50"]),
        )

    elif strategy_key in ("CCI_WPR_5M", "CCI_WPR_3M"):
        cci, pcci = ind["cci20"], _prev(ind["cci20"])
        wpr, pwpr = ind["wpr14"], _prev(ind["wpr14"])
        raw = _sig_array(
            (pcci < -100.0) & (cci >= -100.0) & (pwpr < -80.0) & (wpr > pwpr) & (c > o),
            (pcci > 100.0) & (cci <= 100.0) & (pwpr > -20.0) & (wpr < pwpr) & (c < o),
        )

    elif strategy_key == "DEM_BB_5M":
        dem = ind["dem14"]
        pl, ph_ = _prev(l), _prev(h)
        pbb_l, pbb_u = _prev(bb_lo20), _prev(bb_up20)
        raw = _sig_array(
            (dem < 0.30) & ((l <= bb_lo20) | (pl <= pbb_l)) & (c > bb_lo20) & (c > o),
            (dem > 0.70) & ((h >= bb_up20) | (ph_ >= pbb_u)) & (c < bb_up20) & (c < o),
        )

    elif strategy_key == "WT_CROSS_5M":
        wt1, wt2 = ind["wt1"], ind["wt2"]
        pwt1, pwt2 = _prev(wt1), _prev(wt2)
        raw = _sig_array(
            (pwt1 <= pwt2) & (wt1 > wt2) & (wt1 < -53.0) & (c > o),
            (pwt1 >= pwt2) & (wt1 < wt2) & (wt1 > 53.0) & (c < o),
        )

    # --- SMC ---
    elif strategy_key in ("LIQ_SWP_3M", "LIQ_SWP_5M"):
        rng = np.maximum(h - l, 1e-12)
        lw = np.minimum(o, c) - l
        uw = h - np.maximum(o, c)
        sh, sl = ind["swing_hi20"], ind["swing_lo20"]
        raw = _sig_array(
            (l < sl) & (c > sl) & ((sl - l) >= 0.1 * atr14) & ((sl - l) <= 2.0 * atr14) & ((c > o) | (lw >= 0.45 * rng)),
            (h > sh) & (c < sh) & ((h - sh) >= 0.1 * atr14) & ((h - sh) <= 2.0 * atr14) & ((c < o) | (uw >= 0.45 * rng)),
        )

    elif strategy_key in ("FVG_IMP_3M", "FVG_IMP_5M"):
        raw = _fvg_series(ind, htf_trend)

    # --- PA ---
    elif strategy_key == "PA_PIN_BB_5M":
        rng = np.maximum(h - l, 1e-12)
        body = np.abs(c - o)
        lw = np.minimum(o, c) - l
        uw = h - np.maximum(o, c)
        raw = _sig_array(
            (lw > 2.0 * body) & (lw > 0.60 * rng) & (l < bb_lo20) & (c > bb_lo20) & (rsi14 < 35.0),
            (uw > 2.0 * body) & (uw > 0.60 * rng) & (h > bb_up20) & (c < bb_up20) & (rsi14 > 65.0),
        )

    elif strategy_key == "PA_ENG_SR_5M":
        po, pc = _prev(o), _prev(c)
        body = np.abs(c - o)
        raw = _sig_array(
            (pc < po) & (c > o) & (c >= po) & (o <= pc) & (body >= 0.5 * atr14) & ((l <= bb_lo20) | (l <= ind["swing_lo20"] + 0.15 * atr14)),
            (pc > po) & (c < o) & (c <= po) & (o >= pc) & (body >= 0.5 * atr14) & ((h >= bb_up20) | (h >= ind["swing_hi20"] - 0.15 * atr14)),
        )

    elif strategy_key == "PA_FAKEY_5M":
        mh, ml = _prev(h, 2), _prev(l, 2)
        ih, il = _prev(h, 1), _prev(l, 1)
        inside = (ih <= mh) & (il >= ml)
        raw = _sig_array(inside & (l < ml) & (c > il) & (c > o), inside & (h > mh) & (c < ih) & (c < o))

    elif strategy_key == "PA_MHI_1M":
        raw = _mhi_1m_series(ind)

    elif strategy_key == "PA_3LS_5M":
        o1, c1 = _prev(o, 1), _prev(c, 1)
        o2, c2 = _prev(o, 2), _prev(c, 2)
        o3, c3 = _prev(o, 3), _prev(c, 3)
        three_red = (c3 < o3) & (c2 < o2) & (c1 < o1) & (c2 < c3) & (c1 < c2)
        three_grn = (c3 > o3) & (c2 > o2) & (c1 > o1) & (c2 > c3) & (c1 > c2)
        raw = _sig_array(
            three_red & (c > o) & (c > o3) & (rsi14 < 55.0),
            three_grn & (c < o) & (c < o3) & (rsi14 > 45.0),
        )

    if len(raw):
        warmup = min(200 if strategy_key.startswith("PSAR_EMA") else 120, n - 1)
        raw[:warmup] = 0

    if not regime_filter:
        return raw

    meta = STRATEGIES_META.get(strategy_key, {})
    reg_type = meta.get("regime", "ANY")

    liq_ok = (~np.isnan(vol_ratio)) & (vol_ratio >= 0.40) & (vol_ratio <= 3.20)
    if reg_type == "RANGE":
        reg_ok = liq_ok & (chop >= 45.0) & (adx <= 28.0)
    elif reg_type == "TREND":
        reg_ok = liq_ok & (chop <= 50.0) & (adx >= 18.0)
    elif reg_type == "BREAKOUT":
        reg_ok = liq_ok if strategy_key.startswith("TTM_SQZ") else (liq_ok & (chop <= 55.0))
    elif reg_type == "SMC":
        if strategy_key.startswith("LIQ_SWP"):
            kz = np.array([is_killzone(ts) for ts in ind["index"]], dtype=bool)
            reg_ok = liq_ok & (vol_ratio >= 0.55) & kz
        else:
            reg_ok = liq_ok & (adx >= 16.0)
    elif reg_type == "PA":
        reg_ok = liq_ok & (vol_ratio >= 0.50)
    else:
        reg_ok = liq_ok

    return np.where(reg_ok, raw, 0).astype(np.int8)


def _normalize_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out.sort_index()


def generate_all_candidates(
    pair_name: str,
    frames: dict[str, pd.DataFrame],
    now,
    strategies_status: dict | None = None,
    pair_enabled=None,
) -> list[dict]:
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is not None:
        now_ts = now_ts.tz_convert("UTC").tz_localize(None)

    status = strategies_status if strategies_status is not None else {}
    use_session = bool(status.get("session_filter", False))
    use_regime = bool(status.get("regime_filter", True))

    candidates = []

    if any(status.get(k, True) for k in rev.PRESETS):
        rev_cands = rev.generate_candidates(
            pair_name,
            {"5m": frames.get("5m")},
            now_ts,
            status,
            pair_enabled=pair_enabled,
        )
        for rc in rev_cands:
            meta = STRATEGIES_META.get(rc["strategy"], {})
            rc.setdefault("weight", meta.get("weight", 1.3))
            rc.setdefault("priority", meta.get("priority", 98))
            rc.setdefault("category", "REV")
            rc.setdefault("vol_ratio", 1.0)
            candidates.append(rc)

    closed_by_tf = {}
    ind_by_tf = {}
    for tf_name, tf_min in (("1m", 1), ("3m", 3), ("5m", 5), ("15m", 15), ("60m", 60)):
        raw_df = _normalize_frame(frames.get(tf_name))
        if raw_df.empty:
            continue
        closed = raw_df.loc[raw_df.index + pd.Timedelta(minutes=tf_min) <= now_ts]
        if not closed.empty:
            closed_by_tf[tf_name] = closed

    htf_cache = {}
    for strategy_key, meta in STRATEGIES_META.items():
        if strategy_key in rev.PRESETS:
            continue
        if not status.get(strategy_key, False):
            continue

        tf_name = meta["tf"]
        tf_min = meta["tf_min"]
        closed = closed_by_tf.get(tf_name)
        min_bars = 201 if (tf_min == 5 or strategy_key.startswith("PSAR_EMA")) else 120
        if closed is None or len(closed) < min_bars:
            continue

        bar_end = closed.index[-1] + pd.Timedelta(minutes=tf_min)
        age = (now_ts - bar_end).total_seconds()
        max_age = 45 if tf_min <= 1 else (90 if tf_min <= 3 else 180)
        if not (0.0 <= age <= max_age):
            continue

        if use_session and not session_allowed(strategy_key, bar_end):
            continue
        if pair_enabled is not None and not pair_enabled(pair_name, strategy_key):
            continue

        if tf_name not in ind_by_tf:
            ind_by_tf[tf_name] = compute_base_indicators(closed)
        ind = ind_by_tf[tf_name]

        htf_min = meta.get("htf")
        htf_ctx = None
        if htf_min:
            htf_key = (tf_name, htf_min)
            if htf_key not in htf_cache:
                htf_cache[htf_key] = htf_alignment(closed, closed_by_tf.get(f"{htf_min}m"), tf_min, htf_min)
            htf_ctx = htf_cache[htf_key]

        sig_series = evaluate_strategy_series(
            closed, strategy_key, ind=ind, htf_ctx=htf_ctx, regime_filter=use_regime
        )
        sig_val = int(sig_series[-1]) if len(sig_series) else 0
        if sig_val == 0:
            continue

        atr_val = float(ind["atr14"][-1]) if not np.isnan(ind["atr14"][-1]) else 0.0
        if atr_val <= 0:
            continue
        rsi_val = float(ind["rsi14"][-1]) if not np.isnan(ind["rsi14"][-1]) else 50.0
        chop_val = float(ind["chop"][-1]) if not np.isnan(ind["chop"][-1]) else 50.0
        adx_val = float(ind["adx"][-1]) if not np.isnan(ind["adx"][-1]) else 20.0
        vr_val = float(ind["vol_ratio"][-1]) if not np.isnan(ind["vol_ratio"][-1]) else 1.0
        exp = int(meta["exp"])

        candidates.append({
            "pair": pair_name,
            "tf": tf_name,
            "tf_minutes": tf_min,
            "strategy": strategy_key,
            "strategy_title": meta["title"],
            "category": meta["category"],
            "signal": "CALL" if sig_val > 0 else "PUT",
            "current_price": float(closed["close"].iloc[-1]),
            "atr": atr_val,
            "rsi": rsi_val,
            "adx": adx_val,
            "chop": chop_val,
            "vol_ratio": vr_val,
            "weight": meta.get("weight", 1.0),
            "bar_id": f"{pair_name}|{tf_name}|{closed.index[-1]}",
            "bar_end": bar_end,
            "bar_age_sec": age,
            "exp_range": (exp, exp),
            "suggested_exp": exp,
            "session": session_name(bar_end),
            "confirmations": 1 if htf_ctx is not None else 0,
            "confirmation_names": [f"HTF_{htf_min}m"] if htf_min else [],
            "regime": meta["regime"],
            "priority": meta["priority"],
            "reason": f"{meta['title']} ({tf_name}), CHOP={chop_val:.1f}, ADX={adx_val:.1f}, exp={exp} хв",
        })

    candidates.sort(
        key=lambda c: (
            c.get("priority", STRATEGIES_META.get(c["strategy"], {}).get("priority", 50)),
            c.get("confirmations", 0),
        ),
        reverse=True,
    )
    return candidates


def diagnose_pair_market(
    pair_name: str,
    frames: dict[str, pd.DataFrame],
    now,
    strategies_status: dict,
    pair_enabled=None,
    mute_check=None,
) -> dict:
    now_ts = pd.Timestamp(now)
    if now_ts.tzinfo is not None:
        now_ts = now_ts.tz_convert("UTC").tz_localize(None)

    df_5m = _normalize_frame(frames.get("5m"))
    closed_5m = df_5m.loc[df_5m.index + pd.Timedelta(minutes=5) <= now_ts] if not df_5m.empty else pd.DataFrame()
    chop = adx = rsi14 = price = vol_ratio = None
    sess = session_name(now_ts)
    if len(closed_5m) >= 60:
        ind = compute_base_indicators(closed_5m)
        chop = float(ind["chop"][-1])
        adx = float(ind["adx"][-1])
        rsi14 = float(ind["rsi14"][-1])
        vol_ratio = float(ind["vol_ratio"][-1])
        price = float(closed_5m["close"].iloc[-1])
        sess = session_name(closed_5m.index[-1] + pd.Timedelta(minutes=5))

    def _trend_label(df_htf: pd.DataFrame | None) -> str:
        df_h = _normalize_frame(df_htf)
        if df_h.empty or len(df_h) < 55:
            return "—"
        cl = df_h["close"].astype(float)
        e21 = float(cl.ewm(span=21, adjust=False).mean().iloc[-1])
        e50 = float(cl.ewm(span=50, adjust=False).mean().iloc[-1])
        last_c = float(cl.iloc[-1])
        if e21 > e50 and last_c > e50:
            return "🟢 Висхідний"
        if e21 < e50 and last_c < e50:
            return "🔴 Низхідний"
        return "🟡 Боковий"

    checked = off_global = off_pair = off_session = muted = 0
    for key in STRATEGIES_META:
        if not strategies_status.get(key, False):
            off_global += 1
        elif pair_enabled is not None and not pair_enabled(pair_name, key):
            off_pair += 1
        elif mute_check is not None and mute_check(key)[0]:
            muted += 1
        elif strategies_status.get("session_filter", False) and not session_allowed(key, now_ts):
            off_session += 1
        else:
            checked += 1

    return {
        "pair": pair_name,
        "session": sess,
        "price": price,
        "chop": chop,
        "adx": adx,
        "rsi14": rsi14,
        "vol_ratio": vol_ratio,
        "trend_15m": _trend_label(frames.get("15m")),
        "trend_60m": _trend_label(frames.get("60m")),
        "checked": checked,
        "off_global": off_global,
        "off_pair": off_pair,
        "off_session": off_session,
        "muted": muted,
    }
