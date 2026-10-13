# -*- coding: utf-8 -*-
import logging

logger = logging.getLogger(__name__)

def calculate_dynamic_expiration(analysis_dict) -> int:
    """
    Покращений динамічний розрахунок часу експірації (від 1 до 60 хвилин).
    Захищає середньострокові стратегії (M3, M5, Pivot, Cross, Divergence) від зайвого стискання.
    """
    if not isinstance(analysis_dict, dict):
        return 15

    base_exp = int(analysis_dict.get("suggested_exp", 15))
    adx = float(analysis_dict.get("adx", 20.0))
    volatility_ratio = float(analysis_dict.get("volatility_ratio", 1.0))
    confluence = int(analysis_dict.get("confluence_count", 1))
    strategy_name = str(analysis_dict.get("strategy", ""))
    strategy_title = str(analysis_dict.get("strategy_title", ""))

    # Якщо це середньострокова стратегія (M3, M5, Pivot, Cross, Divergence тощо), захищаємо час
    is_medium_term = any(x in strategy_name.upper() or x in strategy_title.upper() for x in ["M3", "M5", "PIVOT", "CROSS", "DIVERGENCE", "ICHIMOKU", "FIB", "DONCHIAN"])

    if is_medium_term:
        # Не дозволяємо фільтрам знижувати базову експірацію нижче 10 хвилин для середньострокових
        calculated_exp = max(10, base_exp)
        if adx > 45:
            calculated_exp = int(calculated_exp * 0.9)
        return max(1, min(60, calculated_exp))

    multiplier = 1.0

    if volatility_ratio > 1.5:
        multiplier *= 0.8
    elif volatility_ratio < 0.7:
        multiplier *= 1.25

    if adx > 30 and "TREND" in strategy_name.upper():
        multiplier *= 1.2
    elif adx < 20:
        multiplier *= 0.85

    if confluence >= 4:
        multiplier *= 0.9

    calculated_exp = int(round(base_exp * multiplier))
    return max(1, min(60, calculated_exp))

def check_pivot_level_proximity(analysis_dict, threshold_pct=0.0005) -> tuple[bool, str]:
    """
    Перевіряє, чи не знаходиться ціна занадто близько до критичного рівня.
    """
    if not isinstance(analysis_dict, dict):
        return True, "OK"
    
    dist_pivot = float(analysis_dict.get("dist_pivot", 0.01))
    if dist_pivot < threshold_pct:
        return True, f"Ціна надто близько до ключового рівня ({dist_pivot*100:.3f}%), підвищений ризик шуму"
    return True, "OK"

def validate_signal_conditions(analysis_dict) -> tuple[bool, str]:
    """
    Перевіряє коректність сигналу перед подальшою обробкою.
    """
    if isinstance(analysis_dict, dict):
        sig_val = str(analysis_dict.get("signal", "NONE"))
        if sig_val in ["NONE", "HOLD"]:
            return False, analysis_dict.get("reason", "Сигнал відсутній")
        rsi = float(analysis_dict.get("rsi", 50))
        if rsi <= 0 or rsi >= 100:
            return False, "Некоректні значення RSI"
        return True, "OK"
    return False, "Некоректні дані"
