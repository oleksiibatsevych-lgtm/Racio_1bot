import logging
import html
from ai_advisor import analyze_signal_with_gemini
from filters import calculate_dynamic_expiration, validate_signal_conditions

logger = logging.getLogger(__name__)


def process_and_filter_signal(
    pair_name: str, market_data: dict, chart_images: list = None
) -> dict | None:
    """Головна функція обробки сигналу перед відправкою користувачу."""
    adx = float(market_data.get("adx", 0.0))
    atr_ratio = float(market_data.get("atr_ratio", 1.0))
    timeframe = str(market_data.get("primary_tf", "5m"))

    # 1. Розраховуємо динамічний час експірації
    calculated_exp = calculate_dynamic_expiration(timeframe, adx, atr_ratio)
    market_data["suggested_exp"] = calculated_exp

    # 2. Перевіряємо жорсткі фільтри ринку (ADX / ATR)
    is_valid, filter_reason = validate_signal_conditions(
        adx=adx, atr_ratio=atr_ratio, requested_exp=calculated_exp
    )

    if not is_valid:
        logger.info(
            f"⏭️ Сигнал для {pair_name} пропущено фільтром: {filter_reason}"
        )
        return None

    # 3. Викликаємо аналіз Gemini 2.5 Flash
    ai_result = analyze_signal_with_gemini(pair_name, market_data, chart_images)

    # 4. Записуємо отримані параметри
    market_data["ai_confidence"] = ai_result["confidence"]
    market_data["ai_reason"] = html.escape(str(ai_result["reason"]))
    market_data["suggested_exp"] = ai_result.get(
        "suggested_expiration", calculated_exp
    )

    # 5. Відкидаємо сигнали з оцінкою нижче 6 або при помилці (0/10)
    if ai_result["confidence"] < 6:
        logger.info(
            f"⏭️ Сигнал для {pair_name} відхилено ШІ. Оцінка: {ai_result['confidence']}/10"
        )
        return None

    logger.info(
        f"🎯 Сигнал {pair_name} готовий: ШІ {ai_result['confidence']}/10, експірація {market_data['suggested_exp']} хв"
    )
    return market_data
