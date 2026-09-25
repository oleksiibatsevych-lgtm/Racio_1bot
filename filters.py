import logging

logger = logging.getLogger(__name__)


def calculate_dynamic_expiration(
    primary_tf: str,
    adx: float,
    volatility_ratio: float,
    strategy_type: str = "TREND",
    rsi: float = 50.0,
    session_info: str = "Азія",
) -> int:
    """Розрахунок тривалості експірації."""
    if primary_tf == "1m":
        return 3

    if strategy_type == "TREND":
        if adx >= 30.0 or volatility_ratio > 1.3:
            return 15  # 3 свічки 5m
        return 10      # 2 свічки 5m

    elif strategy_type == "BOUNCE":
        if rsi >= 68.0 or rsi <= 32.0:
            return 3  # Швидкий скальпінг
        return 5      # 1 свічка 5m

    return 5


def check_pivot_level_proximity(
    current_price: float,
    signal_type: str,
    pivots: dict,
    threshold_pct: float = 0.0015,
) -> tuple[bool, str]:
    """Захист від торгівлі в стіну рівнів."""
    if not pivots or current_price <= 0:
        return True, "OK"

    resistances = [pivots.get("R1"), pivots.get("R2"), pivots.get("R3")]
    supports = [pivots.get("S1"), pivots.get("S2"), pivots.get("S3")]

    if signal_type == "CALL":
        for r_level in resistances:
            if r_level and r_level > 0:
                dist = (r_level - current_price) / current_price
                if 0 <= dist <= threshold_pct:
                    return False, f"Ціна занадто близько до опору R ({r_level:.5f})"

    elif signal_type == "PUT":
        for s_level in supports:
            if s_level and s_level > 0:
                dist = (current_price - s_level) / current_price
                if 0 <= dist <= threshold_pct:
                    return False, f"Ціна занадто близько до підтримки S ({s_level:.5f})"

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
    """Розділені вимоги для ФЛЕТУ та ТРЕНДУ."""
    if strategy_type == "BOUNCE":
        if adx > 32.0:
            return False, f"Сильний тренд для відскоку (ADX {adx:.1f} > 32.0)"
        if volatility_ratio < 0.65:
            return False, f"Занадто низька волатильність ({volatility_ratio:.2f} < 0.65)"
    else:
        if adx < 20.0:
            return False, f"Слабкий тренд (ADX {adx:.1f} < 20.0)"
        if volatility_ratio < 0.85:
            return False, f"Слабкий імпульс ({volatility_ratio:.2f} < 0.85)"

    if signal_type == "CALL" and rsi > 66.0:
        return False, f"Перекупленість (RSI {rsi:.1f} > 66)"

    if signal_type == "PUT" and rsi < 34.0:
        return False, f"Перепроданість (RSI {rsi:.1f} < 34)"

    return True, "OK"
