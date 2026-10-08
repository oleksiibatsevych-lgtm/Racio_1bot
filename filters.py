import logging

logger = logging.getLogger(__name__)

def calculate_dynamic_expiration(analysis_dict) -> int:
    """ Використовує розраховану експірацію з indicators.py """
    if isinstance(analysis_dict, dict) and "suggested_exp" in analysis_dict:
        return analysis_dict["suggested_exp"]
    return 5

def check_pivot_level_proximity(analysis_dict, threshold_pct=0.0003) -> tuple[bool, str]:
    """ 
    Успішна логіка (бот №2) використовує півот-рівні для входу (відскоку).
    Тому жорстке блокування сигналів поблизу рівнів відключено.
    """
    return True, "OK"

def validate_signal_conditions(analysis_dict) -> tuple[bool, str]:
    """
    Уся логіка (RSI, ADX, Боллінджер) ВЖЕ перевірена в indicators.py.
    Тут ми просто пропускаємо згенеровані сигнали до ML-моделі без блокування.
    """
    if isinstance(analysis_dict, dict):
        sig_val = str(analysis_dict.get("signal", "NONE"))
        if sig_val in ["NONE", "HOLD"]:
            return False, analysis_dict.get("reason", "Сигнал відсутній")
        return True, "OK"
    return False, "Некоректні дані"
