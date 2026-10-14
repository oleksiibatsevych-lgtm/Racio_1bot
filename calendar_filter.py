# -*- coding: utf-8 -*-
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def is_high_impact_news_time() -> tuple[bool, str]:
    """
    Перевіряє наявність ризикових періодів виходу макроекономічних новин.
    """
    try:
        now = datetime.now(timezone.utc)
        # Блокування у п'ятницю в період можливої волатильності (Nonfarm Payrolls тощо)
        if now.weekday() == 4 and 12 <= now.hour <= 15:
            return True, "Вихід важливих новин у п'ятницю (Nonfarm Payrolls)"
        return False, "OK"
    except Exception as e:
        logger.error(f"Помилка перевірки календаря новин: {e}")
        return False, "OK"
