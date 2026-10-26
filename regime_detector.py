# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def detect_market_regime(df_1m: pd.DataFrame, df_3m: pd.DataFrame, df_15m: pd.DataFrame) -> str:
    """
    Визначає поточний мікро-режим ринку для адаптації стратегій.
    Режими: STRONG_TREND, TREND_EXHAUSTION, RANGE_BOUND, SQUEEZE, CHAOS
    """
    if df_3m is None or df_3m.empty or len(df_3m) < 20:
        return "RANGE_BOUND"
    
    try:
        adx = float(df_3m['adx'].iloc[-1]) if 'adx' in df_3m.columns else 20.0
        bb_width = float(df_3m['bb_width'].iloc[-1]) if 'bb_width' in df_3m.columns else 0.002
        rsi = float(df_3m['rsi'].iloc[-1]) if 'rsi' in df_3m.columns else 50.0
        
        # Перевірка на стиснення (Squeeze) смуг Боллінджера
        bb_widths_tail = df_3m['bb_width'].tail(25) if 'bb_width' in df_3m.columns else pd.Series([0.002])
        is_squeeze = bb_width <= bb_widths_tail.quantile(0.18)
        
        if is_squeeze:
            return "SQUEEZE"
        elif adx > 30:
            return "STRONG_TREND"
        elif adx < 20 and (rsi < 32 or rsi > 68):
            return "TREND_EXHAUSTION"
        elif adx < 23:
            return "RANGE_BOUND"
        else:
            return "CHAOS"
    except Exception:
        return "RANGE_BOUND"
