# -*- coding: utf-8 -*-
import pandas as pd
from regime_detector import detect_market_regime
from adaptive_optimizer import get_adaptive_rsi_limits
from ml_model import evaluate_ml_signal

def check_htf_alignment(signal_type: str, htf_df: pd.DataFrame) -> bool:
    """
    Перевіряє відповідність напрямку сигналу 200-періодній EMA на старшому таймфреймі (1H).
    """
    if htf_df is None or htf_df.empty or len(htf_df) < 50:
        return False

    try:
        ema_span = min(200, len(htf_df))
        ema_series = htf_df['close'].ewm(span=ema_span, adjust=False).mean()
        last_close = float(htf_df['close'].iloc[-1])
        last_ema = float(ema_series.iloc[-1])

        if signal_type == "CALL" and last_close > last_ema:
            return True
        elif signal_type == "PUT" and last_close < last_ema:
            return True
    except Exception:
        pass

    return False

def process_signal(df_1m: pd.DataFrame, df_1h: pd.DataFrame = None, ml_filter=None) -> dict:
    """
    Комплексна обробка сигналів: режим ринку, адаптивний RSI,
    мульти-таймфреймова перевірка та оцінка ML.
    """
    if df_1m is None or df_1m.empty or len(df_1m) < 15:
        return {"signal": "NONE", "reason": "Недостатньо даних"}

    regime = detect_market_regime(df_1m)

    high_low = df_1m['high'] - df_1m['low']
    high_close = (df_1m['high'] - df_1m['close'].shift()).abs()
    low_close = (df_1m['low'] - df_1m['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_series = tr.rolling(14).mean()

    current_close = float(df_1m['close'].iloc[-1])
    current_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.001
    base_atr = float(atr_series.mean()) if not pd.isna(atr_series.mean()) else 0.001

    lower_rsi, upper_rsi = get_adaptive_rsi_limits(regime, current_atr, base_atr)

    current_rsi = float(df_1m['rsi'].iloc[-1]) if 'rsi' in df_1m.columns else 50.0

    raw_signal = "NONE"
    if current_rsi <= lower_rsi:
        raw_signal = "CALL"
    elif current_rsi >= upper_rsi:
        raw_signal = "PUT"

    if raw_signal == "NONE":
        return {"signal": "NONE", "regime": regime, "reason": "Значення RSI в нормі"}

    htf_aligned = check_htf_alignment(raw_signal, df_1h) if df_1h is not None else False

    if ml_filter is not None:
        features = {"rsi": current_rsi, "adx": 20, "bb_width": 0.002}
        ml_res = evaluate_ml_signal(ml_filter, features, htf_trend_aligned=htf_aligned)
        if not ml_res["should_trade"]:
            return {
                "signal": "NONE",
                "regime": regime,
                "reason": f"ML відхилив вхід (ймовірність: {ml_res['probability']}%, поріг: {ml_res['threshold']}%)"
            }

    return {
        "signal": raw_signal,
        "regime": regime,
        "htf_aligned": htf_aligned,
        "rsi_limits": (lower_rsi, upper_rsi),
        "reason": "Сигнал успішно перевірено"
    }
