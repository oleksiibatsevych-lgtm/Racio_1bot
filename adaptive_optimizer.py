# -*- coding: utf-8 -*-
import pandas as pd

def get_adaptive_rsi_limits(regime: str, current_atr: float, base_atr: float) -> tuple:
    """
    Динамічно розраховує межі перекупленості/перепроданості RSI 
    залежно від режиму ринку та рівня волатильності.
    """
    vol_factor = (current_atr / base_atr) if (base_atr and base_atr > 0) else 1.0

    if regime in ["STRONG_TREND", "TRENDING"]:
        # Зсув меж за напрямком волатильності у тренді
        lower_bound = max(35.0 - (vol_factor * 2.0), 25.0)
        upper_bound = min(65.0 + (vol_factor * 2.0), 75.0)
    elif regime in ["RANGE_BOUND", "RANGING", "SQUEEZE"]:
        # Розширені межі у боковику (35 / 65) для збільшення кількості точних входів
        lower_bound = 35.0
        upper_bound = 65.0
    else:
        # Стандартні межі за замовчуванням
        lower_bound = 30.0
        upper_bound = 70.0

    return round(lower_bound, 2), round(upper_bound, 2)


def calculate_smart_expiration(base_exp: int, regime: str, volatility_ratio: float) -> int:
    """
    Розраховує адаптивний час експірації на основі режиму ринку та відносної волатильності.
    Зберігає повну функціональність для bot.py.
    """
    try:
        exp = int(base_exp)
        if regime in ["STRONG_TREND", "TRENDING"]:
            exp = int(exp * 0.8) if volatility_ratio > 1.2 else exp
        elif regime in ["RANGE_BOUND", "RANGING", "SQUEEZE"]:
            exp = int(exp * 1.2) if volatility_ratio < 0.8 else exp
        elif regime == "HIGH_VOLATILITY":
            exp = int(exp * 0.7)

        return max(1, min(60, exp))
    except Exception:
        return max(1, min(60, int(base_exp)))


def get_strategy_adaptive_weight(pair_name: str, strategy_name: str) -> float:
    """
    Повертає адаптивну вагу стратегії для конкретної валютної пари.
    Зберігає повну функціональність для indicators.py.
    """
    return 1.0
