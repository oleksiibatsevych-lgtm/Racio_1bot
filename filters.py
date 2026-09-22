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
) -> tuple[bool, str]:
    """Жорсткі фільтри стану ринку з урахуванням напрямку угоди."""
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

    # Блокуємо КУПІВЛЮ (CALL) на піку перекупленості, але ДОЗВОЛЯЄМО ПРОДАЖ (PUT)
    if signal_type == "CALL" and rsi > 67.0:
        return (
            False,
            f"Перекупленість (RSI {rsi:.1f} > 67) — високий ризик купувати CALL на піку",
        )

    # Блокуємо ПРОДАЖ (PUT) на дні перепроданості, але ДОЗВОЛЯЄМО КУПІВЛЮ (CALL)
    if signal_type == "PUT" and rsi < 33.0:
        return (
            False,
            f"Перепроданість (RSI {rsi:.1f} < 33) — високий ризик продавати PUT на дні",
        )

    return True, "OK"
