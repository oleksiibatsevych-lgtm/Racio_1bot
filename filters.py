import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def calculate_dynamic_expiration(
    timeframe: str, adx: float, atr_ratio: float
) -> int:
    """Розраховує тривалість угоди у хвилинах.

    Базовий час прив'язаний до таймфрейму (1-2 свічки). При слабкому ADX (< 25)
    час збільшується на 50% для виходу з імпульсного шуму.
    """
    tf_base_map = {"1m": 3, "5m": 5, "15m": 15, "1h": 30}

    base_exp = tf_base_map.get(str(timeframe).lower(), 5)

    if adx < 25.0:
        base_exp = int(base_exp * 1.5)

    return max(3, min(base_exp, 30))


def validate_signal_conditions(
    adx: float, atr_ratio: float, requested_exp: int
) -> Tuple[bool, str]:
    """Фільтрує сигнали на наявність ринкового шуму та флету."""
    # 1. Заборона угод <= 3 хв при слабкому імпульсі (причина мінусів у 1 пункт)
    if requested_exp <= 3 and adx < 28.0:
        msg = f"Низький ADX ({adx:.1f} < 28) для угоди на {requested_exp} хв (ризик мікрошуму)"
        logger.info(f"🚫 Сигнал відхилено: {msg}")
        return False, msg

    # 2. Фільтр боковика
    if adx < 20.0:
        msg = f"Глибокий флет (ADX {adx:.1f} < 20.0)"
        logger.info(f"🚫 Сигнал відхилено: {msg}")
        return False, msg

    # 3. Фільтр низької волатильності
    if atr_ratio < 0.85:
        msg = f"Низька волатильність (ATR Ratio {atr_ratio:.2f} < 0.85)"
        logger.info(f"🚫 Сигнал відхилено: {msg}")
        return False, msg

    return True, "OK"
