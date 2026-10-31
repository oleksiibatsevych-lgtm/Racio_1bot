# -*- coding: utf-8 -*-
import logging
from adaptive_optimizer import calculate_smart_expiration

logger = logging.getLogger(__name__)

def calculate_dynamic_expiration(analysis_dict: dict) -> int:
    """
    Динамічний розрахунок часу експірації з урахуванням режимів та волатильності.
    """
    if not isinstance(analysis_dict, dict):
        return 5

    base_exp = int(analysis_dict.get("suggested_exp", 5))
    regime = str(analysis_dict.get("regime", "RANGE_BOUND"))
    volatility_ratio = float(analysis_dict.get("volatility_ratio", 1.0))

    return calculate_smart_expiration(base_exp, regime, volatility_ratio)


def check_pivot_level_proximity(analysis_dict: dict, threshold_pct: float = 0.0004) -> tuple[bool, str]:
    """
    Перевірка відстані ціни до ключових Pivot або круглих рівнів.
    """
    if not isinstance(analysis_dict, dict):
        return True, "OK"
    
    dist_pivot = float(analysis_dict.get("dist_pivot", 0.01))
    if dist_pivot < threshold_pct:
        return True, f"Ціна у безпосередній близькості до рівня ({dist_pivot*100:.3f}%)"
    return True, "OK"


def validate_signal_conditions(analysis_dict: dict) -> tuple[bool, str]:
    """
    Первинна фільтрація валідності технічних показників сигналу.
    """
    if isinstance(analysis_dict, dict):
        sig_val = str(analysis_dict.get("signal", "NONE"))
        if sig_val in ["NONE", "HOLD"]:
            return False, analysis_dict.get("reason", "Сигнал відсутній")
        rsi = float(analysis_dict.get("rsi", 50))
        if rsi <= 0 or rsi >= 100:
            return False, "Некоректні значення RSI"
        return True, "OK"
    return False, "Некоректні дані аналізу"
