# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def get_adaptive_rsi_limits(regime: str, current_atr: float, base_atr: float) -> tuple:
    """
    Динамічно розраховує межі перекупленості/перепроданості RSI 
    залежно від режиму ринку та рівня волатильності.
    """
    vol_factor = (current_atr / base_atr) if (base_atr and base_atr > 0) else 1.0

    if regime == "STRONG_TREND":
        lower_bound = max(38.0 - (vol_factor * 2.5), 25.0)
        upper_bound = min(62.0 + (vol_factor * 2.5), 75.0)
    elif regime == "RANGE_BOUND":
        lower_bound = 30.0
        upper_bound = 70.0
    elif regime == "SQUEEZE":
        lower_bound = 35.0
        upper_bound = 65.0
    else:
        lower_bound = 32.0
        upper_bound = 68.0

    return round(lower_bound, 2), round(upper_bound, 2)


def calculate_smart_expiration(base_exp: int, regime: str, volatility_ratio: float, tf: str = "1m") -> int:
    """
    Оптимізація часу експірації залежно від волатильності та режиму ринку.
    Підтримує налаштування мікро-експірації (1-3 хвилини).
    """
    vol = max(0.2, float(volatility_ratio))
    
    # Корегування бази за волатильністю (при високій волатильності час скорочується)
    scaled_exp = float(base_exp) * ((1.0 / vol) ** 0.75)
    
    # Модифікація залежно від ринкового режиму
    if regime == "SQUEEZE":
        scaled_exp = min(scaled_exp, 3.0)  # Мікро-експірація на пробої стиснення
    elif regime == "STRONG_TREND":
        scaled_exp = max(scaled_exp, 5.0)  # Даємо тренду розгорнутися
    elif regime == "TREND_EXHAUSTION":
        scaled_exp = min(scaled_exp, 3.0)  # Швидкий відбій на виснаженні
    elif regime == "RANGE_BOUND":
        scaled_exp = np.clip(scaled_exp, 2.0, 7.0)

    final_exp = int(round(scaled_exp))
    return int(max(1, min(60, final_exp)))


def calculate_dynamic_threshold(regime: str, volatility_ratio: float, base_threshold: float = 2.3) -> float:
    """
    Розраховує динамічний поріг балів (score threshold) для фільтрації сигналів.
    """
    vol_adj = 0.25 * (volatility_ratio - 1.0)
    regime_weights = {
        "STRONG_TREND": 1.1,
        "RANGE_BOUND": 1.0,
        "SQUEEZE": 0.85,
        "TREND_EXHAUSTION": 0.95,
        "CHAOS": 1.25
    }
    w = regime_weights.get(regime, 1.0)
    dyn_thresh = (base_threshold * w) + vol_adj
    return float(round(max(1.5, min(4.5, dyn_thresh)), 2))


def get_strategy_adaptive_weight(pair_name: str, strategy_name: str) -> float:
    """
    Динамічна вага стратегії для конкретної валютної пари.
    """
    jpy_pairs = ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY", "CHF/JPY"]
    if pair_name in jpy_pairs and "Breakout" in strategy_name:
        return 1.2
    if "EUR/USD" in pair_name and "Mean Reversion" in strategy_name:
        return 1.15
    return 1.0
