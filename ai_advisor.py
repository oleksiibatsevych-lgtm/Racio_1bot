import json
import logging
import os
import re
import google.generativeai as genai

# Налаштування логування
logger = logging.getLogger(__name__)

# Ініціалізація Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("⚠️ GEMINI_API_KEY не знайдено в змінних оточення!")


def analyze_signal_with_gemini(
    pair_name: str, payload: dict, chart_images: list = None
) -> dict:
    """Аналізує ринковий сигнал за допомогою Gemini 2.5 Flash.

    Повертає словник із полями:
    - confidence (int): оцінка 1-10
    - reason (str): короткий висновок
    - optimal_tf (str): рекомендований ТФ
    - suggested_expiration (int): рекомендована експірація у хвилинах
    """
    if not GEMINI_API_KEY:
        logger.error("❌ Gemini API Key відсутній. Перехід на fallback.")
        return _get_fallback_response(
            payload, "Помилка: GEMINI_API_KEY не налаштовано"
        )

    prompt = f"""
Ти є професійним Forex та Binary Options аналітиком. Оціни поточний торговий сигнал.

📊 ПАРАМЕТРИ СИГНАЛУ:
- Валютна пара: {pair_name}
- Напрямок: {payload.get('signal', 'UNKNOWN')}
- Основний ТФ: {payload.get('primary_tf', '5m')}
- Поточний RSI: {payload.get('rsi', 'N/A')}
- Поточний ADX: {payload.get('adx', 'N/A')}
- Наявність дивергенції: {payload.get('divergence', 'NONE')}
- Співвідношення ATR: {payload.get('atr_ratio', 'N/A')}
- Початкове обґрунтування: {payload.get('reason', '')}
- ML-ймовірність: {payload.get('ml_prob', 0)}%

💡 ЗАВДАННЯ:
1. Проаналізуй узгодженість індикаторів та графіків.
2. Дай оцінку впевненості у сигналі від 1 до 10 (де 1-4 — слабкий/небезпечний, 5-7 — середній, 8-10 — сильний).
3. Надай КОРОТКЕ (до 20 слів) конкретне обґрунтування українською мовою (вкажи головний плюс або головний ризик).
4. Визнач оптимальний час експірації у хвилинах.

ВІДПОВІДЬ НАДАЙ СТРОГО У ФОРМАТІ JSON БЕЗ БУДЬ-ЯКОГО ДОДАТКОВОГО ТЕКСТУ ТА МАРКДАУН-ТЕГІВ:
{{
    "confidence": 8,
    "reason": "Сильний імпульс за трендом, але RSI наближається до зони перекупленості",
    "optimal_tf": "{payload.get('primary_tf', '5m')}",
    "suggested_expiration": {payload.get('suggested_exp', 5)}
}}
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.2,
        }

        inputs = [prompt]
        if chart_images and isinstance(chart_images, list):
            inputs.extend(chart_images)

        logger.info(f"🧠 Відправка запиту до Gemini API для {pair_name}...")
        response = model.generate_content(
            inputs, generation_config=generation_config
        )

        raw_text = response.text.strip() if response.text else ""

        # Витягуємо чистий JSON за допомогою регулярних виразів
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
        else:
            clean_json = re.sub(r"```json\s*|\s*```", "", raw_text).strip()

        data = json.loads(clean_json)

        confidence = int(data.get("confidence", 5))
        reason = str(data.get("reason", "Аналіз виконано успішно")).strip()
        optimal_tf = str(
            data.get("optimal_tf", payload.get("primary_tf", "5m"))
        )
        suggested_exp = int(
            data.get("suggested_expiration", payload.get("suggested_exp", 5))
        )

        logger.info(
            f"✅ Gemini успішно опрацював {pair_name}: Оцінка {confidence}/10 | {reason}"
        )

        return {
            "confidence": confidence,
            "reason": reason,
            "optimal_tf": optimal_tf,
            "suggested_expiration": suggested_exp,
        }

    except json.JSONDecodeError as e:
        logger.error(
            f"❌ Помилка парсингу JSON від Gemini для {pair_name}: {e}. Сирий текст: {raw_text}"
        )
        return _get_fallback_response(
            payload, f"Помилка формату JSON від ШІ"
        )

    except Exception as e:
        logger.error(
            f"❌ Критична помилка Gemini API для {pair_name}: {type(e).__name__} - {e}"
        )
        return _get_fallback_response(payload, f"Збій API ШІ: {str(e)[:30]}")


def _get_fallback_response(payload: dict, error_msg: str) -> dict:
    """Маркерна резервна відповідь при помилці API."""
    return {
        "confidence": 0,  # 0 чітко показує у чаті/логах, що стався збій
        "reason": error_msg,
        "optimal_tf": payload.get("primary_tf", "5m"),
        "suggested_expiration": payload.get("suggested_exp", 5),
    }
