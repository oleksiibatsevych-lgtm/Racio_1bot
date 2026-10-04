import logging
import html
from ai_advisor import analyze_signal_with_gemini
from filters import calculate_dynamic_expiration, validate_signal_conditions

logger = logging.getLogger(__name__)


def process_and_filter_signal(
    pair_name: str, market_data: dict, chart_images: list = None
) -> dict | None:
    """Головна функція обробки сигналу перед відправкою користувачу."""
    adx = float(market_data.get("adx", 20.0))
    volatility_ratio = float(market_data.get("volatility_ratio", market_data.get("atr_ratio", 1.0)))

    # 1. Розраховуємо динамічний час експірації
    calculated_exp = calculate_dynamic_expiration(market_data)
    market_data["suggested_exp"] = calculated_exp

    # 2. Перевіряємо жорсткі фільтри ринку
    is_valid, filter_reason = validate_signal_conditions(market_data)

    if not is_valid:
        logger.info(
            f"⏭️ Сигнал для {pair_name} пропущено фільтром: {filter_reason}"
        )
        return None

    # 3. Викликаємо аналіз Gemini
    ai_result = analyze_signal_with_gemini(pair_name, market_data, chart_images or [])

    # 4. Записуємо отримані параметри
    market_data["ai_confidence"] = ai_result.get("confidence", 0)
    market_data["ai_reason"] = html.escape(str(ai_result.get("reason", "")))
    market_data["suggested_exp"] = ai_result.get(
        "suggested_expiration", calculated_exp
    )

    # 5. Відкидаємо сигнали з оцінкою нижче 6
    if ai_result.get("confidence", 0) < 6:
        logger.info(
            f"⏭️ Сигнал для {pair_name} відхилено ШІ. Оцінка: {ai_result.get('confidence', 0)}/10"
        )
        return None

    logger.info(
        f"🎯 Сигнал {pair_name} готовий: ШІ {ai_result.get('confidence', 0)}/10, експірація {market_data['suggested_exp']} хв"
    )
    return market_data
