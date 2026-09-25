import logging

logger = logging.getLogger(__name__)

def get_session_adx_threshold(session_info: str) -> float:
    sess = str(session_info).lower()
    if any(s in sess for s in ["азія", "asia", "тихий", "pacific"]):
        return 16.0  
    elif any(s in sess for s in ["лондон", "london", "європа", "europe", "нью-йорк", "ny"]):
        return 22.0  
    return 20.0

def calculate_dynamic_expiration(primary_tf: str, adx: float, volatility_ratio: float, strategy_type: str = "TREND", rsi: float = 50.0, session_info: str = "Азія") -> int:
    """Гібридний розрахунок часу експірації, взятий з логіки Bot 2"""
    # Трендові ривки вимагають більше часу для відпрацювання
    if strategy_type == "TREND":
        if adx > 32.0 or volatility_ratio > 1.5:
            return 15  # Сильний імпульс — даємо ціні 3 свічки по 5хв
        elif adx >= 25.0:
            return 10  # Помірний тренд — 2 свічки
        else:
            return 5
            
    # Відскоки від рівнів зазвичай швидкі
    elif strategy_type == "BOUNCE":
        if rsi >= 70.0 or rsi <= 30.0 or volatility_ratio > 1.2:
            return 3  # Швидкий імпульсний відскок
        return 5      # Стандартний відскок

    return 5

def check_pivot_level_proximity(current_price: float, signal_type: str, pivots: dict, threshold_pct: float = 0.0015) -> tuple[bool, str]:
    """Залишається без змін (дуже хороший захист з Bot 1)"""
    if not pivots or current_price <= 0:
        return True, "OK"
    resistances = [pivots.get("R1"), pivots.get("R2"), pivots.get("R3")]
    supports = [pivots.get("S1"), pivots.get("S2"), pivots.get("S3")]

    if signal_type == "CALL":
        for r in resistances:
            if r and r > 0:
                if 0 <= (r - current_price) / current_price <= threshold_pct:
                    return False, f"Близько до опору R ({r:.5f})"
    elif signal_type == "PUT":
        for s in supports:
            if s and s > 0:
                if 0 <= (current_price - s) / current_price <= threshold_pct:
                    return False, f"Близько до підтримки S ({s:.5f})"
    return True, "OK"

def validate_signal_conditions(adx, volatility_ratio, requested_exp, rsi=50.0, signal_type="CALL", strategy_type="TREND", session_info="Азія"):
    # Цей жорсткий фільтр можна зберегти, оскільки ML та AI тепер роблять основну роботу
    return True, "OK"
