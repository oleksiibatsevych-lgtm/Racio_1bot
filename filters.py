import logging

logger = logging.getLogger(__name__)


def calculate_dynamic_expiration(
    primary_tf="5m",
    adx: float = 20.0,
    volatility_ratio: float = 1.0,
    strategy_type: str = "TREND",
    rsi: float = 50.0,
    session_info: str = "Азія",
) -> int:
    """Підтримує як окремі аргументи, так і передачу dict з результатом аналізу."""
    if isinstance(primary_tf, dict):
        d = primary_tf
        tf_val = str(d.get("primary_tf", "5m"))
        adx_val = float(d.get("adx", 20.0))
        vol_val = float(d.get("volatility_ratio", d.get("atr_ratio", 1.0)))
        strat_val = str(d.get("strategy_type", "TREND"))
        rsi_val = float(d.get("rsi", 50.0))
    else:
        tf_val = str(primary_tf)
        adx_val = float(adx)
        vol_val = float(volatility_ratio)
        strat_val = str(strategy_type)
        rsi_val = float(rsi)

    if tf_val == "1m":
        return 3

    if strat_val == "TREND":
        if adx_val >= 30.0 or vol_val > 1.3:
            return 15
        return 10

    elif strat_val == "BOUNCE":
        if rsi_val >= 68.0 or rsi_val <= 32.0:
            return 3
        return 5

    return 5


def check_pivot_level_proximity(
    current_price=0.0,
    signal_type: str = "CALL",
    pivots: dict = None,
    threshold_pct: float = 0.0015,
) -> tuple[bool, str]:
    """Перевірка наближеності до Pivot рівнів."""
    if isinstance(current_price, dict):
        d = current_price
        price_val = float(d.get("current_price", 0.0))
        sig_val = str(d.get("signal", "CALL"))
        pivots_val = d.get("pivots", {})
    else:
        price_val = float(current_price)
        sig_val = str(signal_type)
        pivots_val = pivots or {}

    if not pivots_val or price_val <= 0:
        return True, "OK"

    resistances = [pivots_val.get("R1"), pivots_val.get("R2"), pivots_val.get("R3")]
    supports = [pivots_val.get("S1"), pivots_val.get("S2"), pivots_val.get("S3")]

    if sig_val == "CALL":
        for r_level in resistances:
            if r_level and r_level > 0:
                dist = (r_level - price_val) / price_val
                if 0 <= dist <= threshold_pct:
                    return False, f"Ціна занадто близько до опору R ({r_level:.5f})"

    elif sig_val == "PUT":
        for s_level in supports:
            if s_level and s_level > 0:
                dist = (price_val - s_level) / price_val
                if 0 <= dist <= threshold_pct:
                    return False, f"Ціна занадто близько до підтримки S ({s_level:.5f})"

    return True, "OK"


def validate_signal_conditions(
    adx=20.0,
    volatility_ratio: float = 1.0,
    requested_exp: int = 5,
    rsi: float = 50.0,
    signal_type: str = "CALL",
    strategy_type: str = "TREND",
    session_info: str = "Азія",
    atr_ratio: float = None,
) -> tuple[bool, str]:
    """Фільтрація умов входу."""
    if isinstance(adx, dict):
        d = adx
        if d.get("signal") == "NONE":
            return False, d.get("reason", "Сигнал відсутній")
        adx_val = float(d.get("adx", 20.0))
        vol_val = float(d.get("volatility_ratio", d.get("atr_ratio", 1.0)))
        rsi_val = float(d.get("rsi", 50.0))
        sig_val = str(d.get("signal", "CALL"))
        strat_val = str(d.get("strategy_type", "TREND"))
    else:
        adx_val = float(adx)
        vol_val = float(atr_ratio if atr_ratio is not None else volatility_ratio)
        rsi_val = float(rsi)
        sig_val = str(signal_type)
        strat_val = str(strategy_type)

    if strat_val == "BOUNCE":
        if adx_val > 32.0:
            return False, f"Сильний тренд для відскоку (ADX {adx_val:.1f} > 32.0)"
        if vol_val < 0.65:
            return False, f"Занадто низька волатильність ({vol_val:.2f} < 0.65)"
    else:
        if adx_val < 20.0:
            return False, f"Слабкий тренд (ADX {adx_val:.1f} < 20.0)"
        if vol_val < 0.85:
            return False, f"Слабкий імпульс ({vol_val:.2f} < 0.85)"

    if sig_val == "CALL" and rsi_val > 66.0:
        return False, f"Перекупленість (RSI {rsi_val:.1f} > 66)"

    if sig_val == "PUT" and rsi_val < 34.0:
        return False, f"Перепроданість (RSI {rsi_val:.1f} < 34)"

    return True, "OK"
