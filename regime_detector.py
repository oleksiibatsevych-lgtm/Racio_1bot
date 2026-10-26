import numpy as np
import pandas as pd

def detect_market_regime(df: pd.DataFrame) -> str:
    """
    Визначає поточний стан ринку:
    - TRENDING: Сильний спрямований тренд (ADX > 25)
    - RANGING: Флет / Боковий рух ціни (ADX <= 25)
    - HIGH_VOLATILITY: Аномально висока волатильність (ATR > 3.5% від ціни)
    """
    if df is None or len(df) < 20:
        return "RANGING"

    # Розрахунок True Range (TR) та ATR (14)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    current_close = df['close'].iloc[-1]
    current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0

    # Оцінка відносної волатильності
    volatility_ratio = (current_atr / current_close) if current_close > 0 else 0.0

    # Перевірка наявності колонки ADX або її розрахунок через відхилення EMA
    if 'adx' in df.columns and not pd.isna(df['adx'].iloc[-1]):
        adx_val = df['adx'].iloc[-1]
    else:
        ema_fast = df['close'].ewm(span=9, adjust=False).mean()
        ema_slow = df['close'].ewm(span=21, adjust=False).mean()
        trend_strength = ((ema_fast - ema_slow).abs() / ema_slow) * 100
        adx_val = trend_strength.iloc[-1] * 10.0

    if volatility_ratio > 0.035:
        return "HIGH_VOLATILITY"
    elif adx_val > 25:
        return "TRENDING"
    else:
        return "RANGING"
