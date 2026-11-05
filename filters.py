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


def check_pivot_level_proximity(analysis_dict: dict, threshold_pct: float = 0.00025) -> tuple[bool, str]:
    """
    Блокує вхід, якщо ціна впритул до рівня, що стоїть НА ШЛЯХУ угоди:
    CALL — рівень (R1/R2/P) трохи вище ціни; PUT — рівень (S1/S2/P) трохи нижче.
    Сигнали ВІД рівня (відбій) не блокуються.
    Стратегії відбою (PRICE_LEVELS, RANGE_CORRIDOR) та CHAOS-sweep пропускаються.
    """
    if not isinstance(analysis_dict, dict):
        return True, "OK"

    if str(analysis_dict.get("strategy", "")) in ("PRICE_LEVELS", "RANGE_CORRIDOR"):
        return True, "OK"

    price = float(analysis_dict.get("current_price", 0) or 0)
    pivots = analysis_dict.get("pivots") or {}
    direction = str(analysis_dict.get("signal", "NONE"))
    if price <= 0 or direction not in ("CALL", "PUT") or not pivots:
        return True, "OK"

    levels = [float(v) for v in pivots.values() if v and float(v) > 0]
    if direction == "CALL":
        ahead = [lv for lv in levels if lv > price and (lv - price) / price < threshold_pct]
    else:
        ahead = [lv for lv in levels if lv < price and (price - lv) / price < threshold_pct]

    if ahead:
        nearest = min(ahead, key=lambda lv: abs(lv - price))
        return False, f"Рівень {nearest:.5f} впритул на шляху ціни ({direction})"
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
