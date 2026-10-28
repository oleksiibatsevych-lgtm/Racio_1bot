# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def detect_market_regime(df_1m: pd.DataFrame, df_3m: pd.DataFrame = None, df_15m: pd.DataFrame = None) -> str:
    """
    Визначає поточний мікро- та макро-режим ринку для адаптації стратегій.
    Режими: STRONG_TREND, TREND_EXHAUSTION, RANGE_BOUND, SQUEEZE, HIGH_VOLATILITY, CHAOS
    """
    target_df = df_3m if (df_3m is not None and not df_3m.empty and len(df_3m) >= 20) else df_1m

    if target_df is None or target_df.empty or len(target_df) < 15:
        return "RANGE_BOUND"

    try:
        current_close = float(target_df['close'].iloc[-1]) if 'close' in target_df.columns else 1.0

        if 'atr' in target_df.columns:
            atr_val = float(target_df['atr'].iloc[-1])
        else:
            high_low = target_df['high'] - target_df['low']
            high_close = (target_df['high'] - target_df['close'].shift()).abs()
            low_close = (target_df['low'] - target_df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr_val = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else 0.001

        volatility_ratio = (atr_val / current_close) if current_close > 0 else 0.0

        if volatility_ratio > 0.035:
            return "HIGH_VOLATILITY"

        adx = float(target_df['adx'].iloc[-1]) if 'adx' in target_df.columns else 20.0
        bb_width = float(target_df['bb_width'].iloc[-1]) if 'bb_width' in target_df.columns else 0.002
        rsi = float(target_df['rsi'].iloc[-1]) if 'rsi' in target_df.columns else 50.0

        bb_widths_tail = target_df['bb_width'].tail(25) if 'bb_width' in target_df.columns else pd.Series([0.002])
        is_squeeze = bb_width <= bb_widths_tail.quantile(0.18)

        if is_squeeze:
            return "SQUEEZE"
        elif adx > 28:
            return "STRONG_TREND"
        elif adx < 20 and (rsi < 32 or rsi > 68):
            return "TREND_EXHAUSTION"
        elif adx < 23:
            return "RANGE_BOUND"
        else:
            return "CHAOS"
    except Exception:
        return "RANGE_BOUND"
