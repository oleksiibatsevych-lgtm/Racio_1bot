import logging

logger = logging.getLogger(__name__)


def get_session_adx_threshold(session_info: str) -> float:
    """Адаптивний поріг ADX залежно від поточної сесії."""
    sess = str(session_info).lower()
    if any(s in sess for s in ["азія", "asia", "тихий", "pacific"]):
        return 18.0  # Суворіший поріг для Азії (було 16.0)
    elif any(
        s in sess
        for s in ["лондон", "london", "європа", "europe", "нью-йорк", "ny"]
    ):
        return 25.0  # Суворіший поріг для Америки/Європи (було 22.0)
    return 22.0


def calculate_dynamic_expiration(
    primary_tf: str,
    adx: float,
    volatility_ratio: float,
    strategy_type: str = "TREND",
    rsi: float = 50.0,
    session_info: str = "Азія",
) -> int:
    """Гібридний розрахунок часу експірації."""
    if strategy_type == "TREND":
        if adx > 32.0 or volatility_ratio > 1.5:
            return 15  # Сильний імпульс — 3 свічки (15 хв)
        elif adx >= 25.0:
            return 10  # Помірний тренд — 2 свічки (10 хв)
        else:
            return 5

    elif strategy_type == "BOUNCE":
        if rsi >= 70.0 or rsi <= 30.0 or volatility_ratio > 1.2:
            return 3  # Швидкий імпульсний відскок (3 хв)
        return 5  # Стандартний відскок (5 хв)

    return 5


def check_pivot_level_proximity(
    current_price: float,
    signal_type: str,
    pivots: dict,
    threshold_pct: float = 0.0015,
) -> tuple[bool, str]:
    """Блокує купівлю CALL прямо під опором та PUT прямо над підтримкою."""
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
                        f"Занадто близько до опору R ({r_level:.5f}) — ризик відкату",
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
                        f"Занадто близько до підтримки S ({s_level:.5f}) — ризик відскоку",
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
    """Сувора перевірка параметрів волатильності та RSI."""
    min_adx_required = get_session_adx_threshold(session_info)

    if strategy_type == "BOUNCE":
        if adx < 12.0:
            return False, f"Мертвий ринок (ADX {adx:.1f} < 12.0)"
        if adx > 35.0:
            return (
                False,
                f"Занадто сильний тренд для відскоку (ADX {adx:.1f} > 35.0)",
            )
        if volatility_ratio < 0.75:
            return (
                False,
                f"Низька волатильність для відскоку ({volatility_ratio:.2f} < 0.75)",
            )
    else:  # TREND
        if adx < min_adx_required:
            return (
                False,
                f"Слабкий тренд (ADX {adx:.1f} < {min_adx_required:.1f} для сесії {session_info})",
            )
        if volatility_ratio < 0.95:
            return (
                False,
                f"Низька волатильність для тренду ({volatility_ratio:.2f} < 0.95)",
            )

    # Звужені зони безпечного входу по RSI
    if signal_type == "CALL" and rsi > 64.0:
        return (
            False,
            f"Перекупленість (RSI {rsi:.1f} > 64) — ризик купувати на піку",
        )

    if signal_type == "PUT" and rsi < 36.0:
        return (
            False,
            f"Перепроданість (RSI {rsi:.1f} < 36) — ризик продавати на дні",
        )

    return True, "OK"
