import logging

logger = logging.getLogger(__name__)

def calculate_dynamic_expiration(analysis_dict) -> int:
    """
    Покращений динамічний розрахунок часу експірації (від 1 до 60 хвилин)
    на основі базової стратегії, сили тренду (ADX), волатильності (ATR) та конфлюенції.
    """
    if not isinstance(analysis_dict, dict):
        return 5

    # 1. Отримуємо базову експірацію від стратегії
    base_exp = int(analysis_dict.get("suggested_exp", 15))
    
    # 2. Отримуємо параметри ринку з аналізу
    adx = float(analysis_dict.get("adx", 20.0))
    volatility_ratio = float(analysis_dict.get("volatility_ratio", 1.0))
    confluence = int(analysis_dict.get("confluence_count", 1))
    strategy_name = str(analysis_dict.get("strategy", ""))

    multiplier = 1.0

    # 3. Корекція за волатильністю (ATR)
    # Якщо волатильність занадто висока (> 1.5), ціна рухається швидко -> скорочуємо час
    # Якщо волатильність низька (< 0.7), ціна повільна -> збільшуємо час
    if volatility_ratio > 1.5:
        multiplier *= 0.8
    elif volatility_ratio < 0.7:
        multiplier *= 1.25

    # 4. Корекція за силою тренду (ADX)
    # Сильний тренд (ADX > 30) тримається довше для трендових стратегій
    if adx > 30 and "TREND" in strategy_name.upper():
        multiplier *= 1.2
    # Слабому тренду/флету потрібен швидкий час виконання
    elif adx < 20:
        multiplier *= 0.85

    # 5. Корекція за конфлюенцією (кількістю підтверджених стратегій)
    # Чим більше стратегій зливаються в один сигнал (наприклад, > 5), тим потужніший імпульс
    if confluence >= 6:
        multiplier *= 0.9  # Потужний імпульс відпрацьовує швидше

    # Застосовуємо множник до базової експірації
    calculated_exp = int(round(base_exp * multiplier))

ро 
    # Гарантуємо, що час залишається в межах розумного діапазону бінарних опціонів/скальпінгу (1..60 хв)
    return max(1, min(60, calculated_exp))

def check_pivot_level_proximity(analysis_dict, threshold_pct=0.0003) -> tuple[bool, str]:
    """Півот-рівні використовуються як орієнтир, без блокування."""
    return True, "OK"

def validate_signal_conditions(analysis_dict) -> tuple[bool, str]:
    """Пропускає згенеровані сигнали до ML-моделі без непотрібного блокування."""
    if isinstance(analysis_dict, dict):
        sig_val = str(analysis_dict.get("signal", "NONE"))
        if sig_val in ["NONE", "HOLD"]:
            return False, analysis_dict.get("reason", "Сигнал відсутній")
        return True, "OK"
    return False, "Некоректні дані"
