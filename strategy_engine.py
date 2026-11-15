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
    "MEAN": "🧲 Mean Reversion (Відбій)",
    "TREND": "📈 Trend & MTF Pullback",
    "VOL": "💥 Volatility & ORB (Пробої)",
    "SMC": "🏦 ICT / Smart Money",
    "PA": "🕯 Price Action (Свічки)",
}

GLOBAL_FILTERS = {
    "session_filter": "🌐 Фільтр торгових сесій",
    "regime_filter": "🧭 Фільтр режиму (CHOP/ADX/ATR)",
    "news_filter": "📰 Карантин макроновин (FF)",
}

STRATEGIES_META = {
    "REV_RSI14": {"title": "REV RSI14", "category": "REV", "tf": "5m", "tf_min": 5, "exp": 60, "htf": None, "regime": "ANY", "priority": 100},
    "REV_BB3": {"title": "REV BB3", "category": "REV", "tf": "5m", "tf_min": 5, "exp": 60, "htf": None, "regime": "ANY", "priority": 99},
    "REV_RSI7": {"title": "REV RSI7", "category": "REV", "tf": "5m", "tf_min": 5, "exp": 45, "htf": None, "regime": "ANY", "priority": 98},

    "CRSI_5M": {"title": "Connors RSI 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86},
    "CRSI_1M": {"title": "Connors RSI 1m", "category": "MEAN", "tf": "1m", "tf_min": 1, "exp": 3, "htf": None, "regime": "RANGE", "priority": 72},
    "VALUE_CH_5M": {"title": "Value Chart 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 85},
    "VALUE_CH_1M": {"title": "Value Chart 1m", "category": "MEAN", "tf": "1m", "tf_min": 1, "exp": 3, "htf": None, "regime": "RANGE", "priority": 71},
    "OSC_CONS_5M": {"title": "RSI+BB+Stoch 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 87},
    "OSC_CONS_3M": {"title": "RSI+BB+Stoch 3m", "category": "MEAN", "tf": "3m", "tf_min": 3, "exp": 6, "htf": None, "regime": "RANGE", "priority": 78},
    "BB_KC_REV_5M": {"title": "BB+KC Envelope 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": 60, "regime": "RANGE", "priority": 84},
    "BB_KC_REV_15M": {"title": "BB+KC Envelope 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": 60, "regime": "RANGE", "priority": 89},
    "PIVOT_REV_5M": {"title": "Daily Pivot S/R 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 83},
    "EXH_STRK_1M": {"title": "Streak Exhaustion 1m", "category": "MEAN", "tf": "1m", "tf_min": 1, "exp": 3, "htf": None, "regime": "RANGE", "priority": 70},
    "EXH_STRK_3M": {"title": "Streak Exhaustion 3m", "category": "MEAN", "tf": "3m", "tf_min": 3, "exp": 6, "htf": None, "regime": "RANGE", "priority": 76},

    "MTF_PB_1M_15": {"title": "MTF Pullback 1m/15m", "category": "TREND", "tf": "1m", "tf_min": 1, "exp": 5, "htf": 15, "regime": "TREND", "priority": 75},
    "MTF_PB_3M_15": {"title": "MTF Pullback 3m/15m", "category": "TREND", "tf": "3m", "tf_min": 3, "exp": 10, "htf": 15, "regime": "TREND", "priority": 81},
    "MTF_PB_5M_60": {"title": "MTF Pullback 5m/60m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": 60, "regime": "TREND", "priority": 90},
    "EMA_RIB_3M": {"title": "EMA Ribbon 3m", "category": "TREND", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "TREND", "priority": 77},
    "EMA_RIB_5M": {"title": "EMA Ribbon 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 84},
    "MACD_ADX_5M": {"title": "MACD+ADX Trend 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 85},
    "MACD_ADX_15M": {"title": "MACD+ADX Trend 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 91},
    "ST_MOM_3M": {"title": "Dual Supertrend 3m", "category": "TREND", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "TREND", "priority": 79},
    "ST_MOM_5M": {"title": "Dual Supertrend 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 86},

    "TTM_SQZ_5M": {"title": "TTM Squeeze 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 92},
    "TTM_SQZ_15M": {"title": "TTM Squeeze 15m", "category": "VOL", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "BREAKOUT", "priority": 94},
    "ORB_LON_5M": {"title": "ORB London 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 93},
    "ORB_NY_5M": {"title": "ORB New York 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 93},
    "DONCH_BRK_5M": {"title": "Donchian Breakout 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 82},

    "LIQ_SWP_3M": {"title": "ICT Liquidity Sweep 3m", "category": "SMC", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "SMC", "priority": 80},
    "LIQ_SWP_5M": {"title": "ICT Liquidity Sweep 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "SMC", "priority": 88},
    "FVG_IMP_3M": {"title": "ICT Fair Value Gap 3m", "category": "SMC", "tf": "3m", "tf_min": 3, "exp": 9, "htf": 15, "regime": "SMC", "priority": 80},
    "FVG_IMP_5M": {"title": "ICT Fair Value Gap 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": 60, "regime": "SMC", "priority": 89},

    "PA_PIN_BB_5M": {"title": "PA Pin Bar + BB 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 84},
    "PA_ENG_SR_5M": {"title": "PA Engulfing S/R 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 85},
    "PA_FAKEY_5M": {"title": "PA Fakey Trap 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 88},
}

ALL_STRATEGY_KEYS = list(STRATEGIES_META.keys())


def is_known_strategy(cand: dict) -> bool:
    return cand.get("strategy") in STRATEGIES_META


def get_strategy_expiration(cand: dict) -> int:
    if cand.get("suggested_exp"):
        return int(cand["suggested_exp"])
    meta = STRATEGIES_META.get(cand.get("strategy", ""))
    return int(meta["exp"]) if meta else 15


def load_all_strategy_config(path) -> dict:
    defaults = {k: True for k in ALL_STRATEGY_KEYS}
    defaults["session_filter"] = True
    defaults["regime_filter"] = True
    defaults["news_filter"] = True
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
    return chop.clip(0.0, 100.0)


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
    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
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

    st_fast = compute_supertrend(df, tr, 7, 2.0)
    st_slow = compute_supertrend(df, tr, 10, 3.0)
    sqz_on, sqz_mom = compute_ttm_squeeze(df, tr, 20, 2.0, 1.5)
    pivots = compute_daily_pivots(df)

    return {
        "o": df["open"].to_numpy(dtype=float),
        "h": df["high"].to_numpy(dtype=float),
        "l": df["low"].to_numpy(dtype=float),
        "c": close.to_numpy(dtype=float),
        "atr14": atr14.to_numpy(dtype=float),
        "atr20": atr20.to_numpy(dtype=float),
        "vol_ratio": vol_ratio.to_numpy(dtype=float),
        "ema8": ema8.to_numpy(dtype=float),
        "ema21": ema21.to_numpy(dtype=float),
        "ema50": ema50.to_numpy(dtype=float),
        "sma20": sma20.to_numpy(dtype=float),
        "std20": std20.to_numpy(dtype=float),
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
        "st_fast": st_fast,
        "st_slow": st_slow,
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


def _orb_series(ind: dict, zone, start_h: int, start_m: int, end_h: int) -> np.ndarray:
    idx = pd.DatetimeIndex(ind["index"])
    if idx.tz is None:
        idx_utc = idx.tz_localize("UTC")
    else:
        idx_utc = idx.tz_convert("UTC")
    local = idx_utc.tz_convert(zone)
    dates = local.date
    hours = local.hour
    mins = local.minute

    c = ind["c"]
    o = ind["o"]
    h = ind["h"]
    l = ind["l"]
    atr = ind["atr14"]
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
                if np.all(c[m + 1:i] >= h[m - 2]):
                    if l[i] <= l[m] and c[i] > h[m - 2] and c[i] > o[i]:
                        out[i] = 1
                        break
            gap_bear = l[m - 2] - h[m]
            if gap_bear >= 0.25 * atr[m - 1] and htf_trend[i] <= 0:
                if np.all(c[m + 1:i] <= l[m - 2]):
                    if h[i] >= h[m] and c[i] < l[m - 2] and c[i] < o[i]:
                        out[i] = -1
                        break
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

    if strategy_key in ("CRSI_5M", "CRSI_1M"):
        crsi = ind["crsi"]
        raw = _sig_array((crsi < 10.0) & (c > o), (crsi > 90.0) & (c < o))

    elif strategy_key in ("VALUE_CH_5M", "VALUE_CH_1M"):
        raw = _sig_array(
            (ind["vc_l"] <= -8.0) & (ind["vc_c"] <= -6.0),
            (ind["vc_h"] >= 8.0) & (ind["vc_c"] >= 6.0),
        )

    elif strategy_key in ("OSC_CONS_5M", "OSC_CONS_3M"):
        k, d = ind["stoch_k"], ind["stoch_d"]
        pk, pd_ = _prev(k), _prev(d)
        not_squeezed = ~ind["sqz_on"]
        raw = _sig_array(
            not_squeezed & (c < bb_lo20) & (rsi7 < 25.0) & (k < 25.0) & (k > d) & (pk <= pd_),
            not_squeezed & (c > bb_up20) & (rsi7 > 75.0) & (k > 75.0) & (k < d) & (pk >= pd_),
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
        bull_s1 = (~np.isnan(s1)) & (l <= s1 + tol) & (c > s1) & (c > o) & (rsi14 < 38.0)
        bull_s2 = (~np.isnan(s2)) & (l <= s2 + tol) & (c > s2) & (c > o) & (rsi14 < 38.0)
        bear_r1 = (~np.isnan(r1)) & (h >= r1 - tol) & (c < r1) & (c < o) & (rsi14 > 62.0)
        bear_r2 = (~np.isnan(r2)) & (h >= r2 - tol) & (c < r2) & (c < o) & (rsi14 > 62.0)
        raw = _sig_array(bull_s1 | bull_s2, bear_r1 | bear_r2)

    elif strategy_key in ("EXH_STRK_1M", "EXH_STRK_3M"):
        g = pd.Series((c > o).astype(int)).rolling(4).sum().to_numpy() == 4
        r = pd.Series((c < o).astype(int)).rolling(4).sum().to_numpy() == 4
        raw = _sig_array(r & (c < bb_lo20) & (rsi7 < 20.0), g & (c > bb_up20) & (rsi7 > 80.0))

    elif strategy_key in ("MTF_PB_1M_15", "MTF_PB_3M_15", "MTF_PB_5M_60"):
        e21, e50 = ind["ema21"], ind["ema50"]
        raw = _sig_array(
            (htf_trend == 1) & (c > e50) & (l <= e21 + 0.25 * atr14) & (rsi7 < 35.0) & (c > o),
            (htf_trend == -1) & (c < e50) & (h >= e21 - 0.25 * atr14) & (rsi7 > 65.0) & (c < o),
        )

    elif strategy_key in ("EMA_RIB_3M", "EMA_RIB_5M"):
        e8, e21, e50 = ind["ema8"], ind["ema21"], ind["ema50"]
        pc = _prev(c)
        pe8 = _prev(e8)
        raw = _sig_array(
            (e8 > e21) & (e21 > e50) & (l <= e8) & (c > e50) & (c > e8) & (pc <= pe8),
            (e8 < e21) & (e21 < e50) & (h >= e8) & (c < e50) & (c < e8) & (pc >= pe8),
        )

    elif strategy_key in ("MACD_ADX_5M", "MACD_ADX_15M"):
        hist = ind["macd_hist"]
        ph = _prev(hist)
        e50 = ind["ema50"]
        raw = _sig_array(
            (c > e50) & (adx > 22.0) & (ind["plus_di"] > ind["minus_di"]) & (hist > 0.0) & (ph <= 0.0),
            (c < e50) & (adx > 22.0) & (ind["minus_di"] > ind["plus_di"]) & (hist < 0.0) & (ph >= 0.0),
        )

    elif strategy_key in ("ST_MOM_3M", "ST_MOM_5M"):
        stf, sts = ind["st_fast"], ind["st_slow"]
        pstf = _prev(stf.astype(float))
        raw = _sig_array(
            (sts == 1) & (stf == 1) & (pstf == -1),
            (sts == -1) & (stf == -1) & (pstf == 1),
        )

    elif strategy_key in ("TTM_SQZ_5M", "TTM_SQZ_15M"):
        sqz = ind["sqz_on"]
        mom = ind["sqz_mom"]
        pmom = _prev(mom)
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

    elif strategy_key in ("LIQ_SWP_3M", "LIQ_SWP_5M"):
        rng = np.maximum(h - l, 1e-12)
        lw = np.minimum(o, c) - l
        uw = h - np.maximum(o, c)
        sh, sl = ind["swing_hi20"], ind["swing_lo20"]
        bull_swp = (l < sl) & (c > sl) & ((sl - l) >= 0.1 * atr14) & ((sl - l) <= 2.0 * atr14) & ((c > o) | (lw >= 0.45 * rng))
        bear_swp = (h > sh) & (c < sh) & ((h - sh) >= 0.1 * atr14) & ((h - sh) <= 2.0 * atr14) & ((c < o) | (uw >= 0.45 * rng))
        raw = _sig_array(bull_swp, bear_swp)

    elif strategy_key in ("FVG_IMP_3M", "FVG_IMP_5M"):
        raw = _fvg_series(ind, htf_trend)

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
        bull_eng = (pc < po) & (c > o) & (c >= po) & (o <= pc) & (body >= 0.5 * atr14) & ((l <= bb_lo20) | (l <= ind["swing_lo20"] + 0.15 * atr14))
        bear_eng = (pc > po) & (c < o) & (c <= po) & (o >= pc) & (body >= 0.5 * atr14) & ((h >= bb_up20) | (h >= ind["swing_hi20"] - 0.15 * atr14))
        raw = _sig_array(bull_eng, bear_eng)

    elif strategy_key == "PA_FAKEY_5M":
        mh, ml = _prev(h, 2), _prev(l, 2)
        ih, il = _prev(h, 1), _prev(l, 1)
        inside = (ih <= mh) & (il >= ml)
        raw = _sig_array(
            inside & (l < ml) & (c > il) & (c > o),
            inside & (h > mh) & (c < ih) & (c < o),
        )

    if len(raw):
        warmup = min(120, n - 1)
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
        if strategy_key.startswith("TTM_SQZ"):
            reg_ok = liq_ok
        else:
            reg_ok = liq_ok & (chop <= 55.0)
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
        candidates.extend(rev_cands)

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
        min_bars = 201 if tf_min == 5 else 120
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
                htf_df = closed_by_tf.get(f"{htf_min}m")
                htf_cache[htf_key] = htf_alignment(closed, htf_df, tf_min, htf_min)
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
        sess = session_name(bar_end)

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
            "bar_id": f"{pair_name}|{tf_name}|{closed.index[-1]}",
            "bar_end": bar_end,
            "bar_age_sec": age,
            "exp_range": (exp, exp),
            "suggested_exp": exp,
            "session": sess,
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
