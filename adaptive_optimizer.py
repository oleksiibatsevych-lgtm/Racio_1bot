import pandas as pd

def get_adaptive_rsi_limits(regime: str, current_atr: float, base_atr: float) -> tuple:
    """
    Динамічно розраховує межі перекупленості/перепроданості RSI 
    залежно від режиму ринку та рівня волатильності.
    """
    vol_factor = (current_atr / base_atr) if (base_atr and base_atr > 0) else 1.0

    if regime == "TRENDING":
        # Зсув меж за напрямком волатильності у тренді
        lower_bound = max(35.0 - (vol_factor * 2.0), 25.0)
        upper_bound = min(65.0 + (vol_factor * 2.0), 75.0)
    elif regime == "RANGING":
        # Розширені межі у боковику (35 / 65) для збільшення кількості входів
        lower_bound = 35.0
        upper_bound = 65.0
    else:
        # Стандартні межі за замовчуванням
        lower_bound = 30.0
        upper_bound = 70.0

    return round(lower_bound, 2), round(upper_bound, 2)
