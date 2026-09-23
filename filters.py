import logging

logger = logging.getLogger(__name__)


def get_session_adx_threshold(session_info: str) -> float:
    """Повертає адаптивний поріг ADX залежно від поточної сесії."""
    sess = str(session_info).lower()
    if any(s in sess for s in ["азія", "asia", "тихий", "pacific"]):
        return 16.0  # Тихоокеанська / Азійська сесія (флет, тренди виявляємо раніше)
    elif any(s in sess for s in ["лондон", "london", "європа", "europe", "нью-йорк", "ny"]):
        return 22.0  # Висока волатильність Європи та Америки вимагає сильнішого тренду
    return 20.0


def calculate_dynamic_expiration(
    primary_tf: str,
    adx: float,
    volatility_ratio: float,
    strategy_type: str = "TREND",
    rsi: float = 50.0,
    session_info: str = "Азія",
) -> int:
    """Динамічний розрахунок часу експірації у хвилинах."""
    # 1. СТРАТЕГІЯ "ПО ТРЕНДУ"
    if strategy_type == "TREND":
        if adx >= 30.0 or volatility_ratio > 1.25:
            return 15  # Потужний імпульс — даємо ціні 3 свічки (15 хв)
        return 10      # Стандартний трендовий вхід — 2 свічки (10 хв)

    # 2. СТРАТЕГІЯ "ВІДСКОК ВІД РІВНЯ"
    elif strategy_type == "BOUNCE":
        if rsi >= 70.0 or rsi <= 30.0 or volatility_ratio > 1.2:
            return 3  # Швидкий імпульсний відскок (3 хв)
        return 5      # Стандартний відскок (5 хв)

    # Дефолт за таймфреймом
    return 5


def check_pivot_level_proximity(
    current_price: float,
    signal_type: str,
    pivots: dict,
    threshold_pct: float = 0.0015,
) -> tuple[bool, str]:
    """Захист: блокує купівлю CALL прямо під опором та PUT прямо над підтримкою."""
    if not pivots or current_price <= 0:
        return True, "OK"

    resistances = [pivots.get("R1"), pivots.get("R2"), pivots.get("R3")]
    supports = [pivots.get("S1"), pivots.get("S2"), pivots.get("S3")]

    if signal_type == "CALL":
        for r_level in resistances:
            if r_level and r_level > 0:
                dist = (r_level - current_price) / current_price
                if 0 <= dist <= threshold_pct:
                    return (
                        False,
                        f"Ціна занадто близько до опору R ({r_level:.5f}) — ризик відкату вниз",
                    )
                elif -0.0005 <= dist < 0:
                    return (
                        False,
                        f"Локальний ложний пробій опору ({r_level:.5f})",
                    )

    elif signal_type == "PUT":
        for s_level in supports:
            if s_level and s_level > 0:
                dist = (current_price - s_level) / current_price
                if 0 <= dist <= threshold_pct:
                    return (
                        False,
                        f"Ціна занадто близько до підтримки S ({s_level:.5f}) — ризик відскоку вгору",
                    )
                elif -0.0005 <= dist < 0:
                    return (
                        False,
                        f"Локальний ложний пробій підтримки ({s_level:.5f})",
                    )

    return True, "OK"


def validate_signal_conditions(
    adx: float,
    volatility_ratio: float,
    requested_exp: int,
    rsi: float = 50.0,
    signal_type: str = "CALL",
    strategy_type: str = "TREND",
    session_info: str = "Азія",
) -> tuple[bool, str]:
    """Перевіряє відповідність ринку з адаптивним ADX під сесію."""
    
    min_adx_required = get_session_adx_threshold(session_info)

    # 1. Правила для відскоку від рівнів
    if strategy_type == "BOUNCE":
        if adx < 10.0:
            return False, f"Мертвий ринок (ADX {adx:.1f} < 10.0)"
        if adx > 38.0:
            return False, f"Занадто сильний тренд для відскоку (ADX {adx:.1f} > 38.0) — ризик пробою"
        if volatility_ratio < 0.70:
            return False, f"Занадто низька волатильність ({volatility_ratio:.2f} < 0.70)"

    # 2. Правила для трендової стратегії (з адаптивним ADX)
    else:
        if adx < min_adx_required:
            return (
                False,
                f"Слабкий тренд для торгівлі за імпульсом (ADX {adx:.1f} < {min_adx_required:.1f} для сесії: {session_info})",
            )
        if volatility_ratio < 0.80:
            return False, f"Низька волатильність для тренду ({volatility_ratio:.2f} < 0.80)"

    # 3. Напрямкові фільтри RSI
    if signal_type == "CALL" and rsi > 67.0:
        return False, f"Перекупленість (RSI {rsi:.1f} > 67) — ризик купувати CALL на піку"

    if signal_type == "PUT" and rsi < 33.0:
        return False, f"Перепроданість (RSI {rsi:.1f} < 33) — ризик продавати PUT на дні"

    return True, "OK"
