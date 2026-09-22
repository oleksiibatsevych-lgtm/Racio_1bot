import logging

logger = logging.getLogger(__name__)


def calculate_dynamic_expiration(
    primary_tf: str, adx: float, volatility_ratio: float
) -> int:
    """Обчислює рекомендований час експірації залежно від ТФ та стану ринку."""
    if primary_tf == "1h":
        return 30 if adx < 25 else 15
    elif primary_tf == "15m":
        return 15 if adx < 25 else 10
    else:  # 5m або 1m
        if volatility_ratio > 1.3:
            return 3
        elif volatility_ratio < 0.8:
            return 5
        return 5


def check_pivot_level_proximity(
    current_price: float,
    signal_type: str,
    pivots: dict,
    threshold_pct: float = 0.0015,
) -> tuple[bool, str]:
    """Перевіряє, чи не здійснюється купівля в опір або продажу в підтримку (поріг ~0.15%)."""
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
                        f"Ціна занадто близько до рівню опору R ({r_level:.5f}) — високий ризик відкату вниз",
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
                        f"Ціна занадто близько до рівню підтримки S ({s_level:.5f}) — високий ризик відскоку вгору",
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
) -> tuple[bool, str]:
    """Основні жорсткі фільтри стану ринку."""
    if requested_exp <= 3 and adx < 24.0:
        return (
            False,
            f"Низький ADX ({adx:.1f} < 24) для угоди на {requested_exp} хв",
        )

    if adx < 18.0:
        return False, f"Глибокий флет (ADX {adx:.1f} < 18.0)"

    if volatility_ratio < 0.80:
        return (
            False,
            f"Низька волатильність (ATR Ratio {volatility_ratio:.2f} < 0.80)",
        )

    if rsi > 67.0:
        return (
            False,
            f"Перекупленість (RSI {rsi:.1f} > 67) — високий ризик відкату вниз",
        )
    if rsi < 33.0:
        return (
            False,
            f"Перепроданість (RSI {rsi:.1f} < 33) — високий ризик відкату вгору",
        )

    return True, "OK"
