import logging

logger = logging.getLogger(__name__)

def calculate_dynamic_expiration(analysis_dict) -> int:
    """Отримує та валідує розраховану експірацію у межах 1..60 хвилин."""
    if isinstance(analysis_dict, dict) and "suggested_exp" in analysis_dict:
        try:
            exp_val = int(analysis_dict["suggested_exp"])
            return max(1, min(60, exp_val))
        except (ValueError, TypeError):
            return 5
    return 5

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
