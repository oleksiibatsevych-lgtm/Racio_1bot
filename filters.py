# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)

def calculate_dynamic_expiration(analysis_dict) -> int:
    """
    Динамічний розрахунок експірації з урахуванням типів стратегій та волатильності.
    """
    if not isinstance(analysis_dict, dict):
        return 15

    base_exp = int(analysis_dict.get("suggested_exp", 15))
    adx = float(analysis_dict.get("adx", 20.0))
    volatility_ratio = float(analysis_dict.get("volatility_ratio", 1.0))
    confluence = int(analysis_dict.get("confluence_count", 1))

    # Захист мінімального часу для середньострокових сигналів
    calculated_exp = max(5, base_exp)
    
    if volatility_ratio > 1.4:
        calculated_exp = int(calculated_exp * 0.85)
    elif volatility_ratio < 0.7:
        calculated_exp = int(calculated_exp * 1.15)

    if adx > 35:
        calculated_exp = int(calculated_exp * 0.9)

    return max(1, min(60, calculated_exp))

def check_pivot_level_proximity(analysis_dict, threshold_pct=0.0004) -> tuple[bool, str]:
    if not isinstance(analysis_dict, dict):
        return True, "OK"
    
    dist_pivot = float(analysis_dict.get("dist_pivot", 0.01))
    if dist_pivot < threshold_pct:
        return True, f"Ціна близько до ключового рівня ({dist_pivot*100:.3f}%)"
    return True, "OK"

def validate_signal_conditions(analysis_dict) -> tuple[bool, str]:
    if isinstance(analysis_dict, dict):
        sig_val = str(analysis_dict.get("signal", "NONE"))
        if sig_val in ["NONE", "HOLD"]:
            return False, analysis_dict.get("reason", "Сигнал відсутній")
        rsi = float(analysis_dict.get("rsi", 50))
        if rsi <= 0 or rsi >= 100:
            return False, "Некоректні значення RSI"
        return True, "OK"
    return False, "Некоректні дані"
