import io
import json
import logging
import os
import re
from PIL import Image
import google.generativeai as genai

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("⚠️ GEMINI_API_KEY не знайдено в змінних оточення!")


def get_available_gemini_models() -> list[str]:
    """Динамічно отримує список усіх доступних моделей для генерації контенту за вашим API-ключем."""
    try:
        available_models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                # Отримуємо назву моделі (наприклад, 'models/gemini-1.5-flash' або 'models/gemini-2.0-flash')
                model_name = m.name
                available_models.append(model_name)
        
        logger.info(f"📋 Доступні Gemini моделі для ключа: {available_models}")
        return available_models
    except Exception as e:
        logger.error(f"❌ Помилка отримання списку моделей від Gemini: {e}")
        # Резервний список на випадок помилки запиту списку
        return ["models/gemini-1.5-flash", "models/gemini-1.5-pro", "gemini-1.5-flash"]


def analyze_signal_with_gemini(
    pair_name: str, payload: dict, chart_images: list = None
) -> dict:
    """Аналізує ринковий сигнал за допомогою Gemini API."""
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
1. Проаналізуй узгодженість індикаторів та наданих графіків.
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

    inputs = [prompt]

    # 🛠 Перетворення зображень у формат PIL.Image
    if chart_images and isinstance(chart_images, list):
        for img in chart_images:
            if isinstance(img, io.BytesIO):
                try:
                    img.seek(0)
                    pil_img = Image.open(img)
                    inputs.append(pil_img)
                except Exception as img_err:
                    logger.warning(
                        f"⚠️ Не вдалося відкрити зображення через PIL: {img_err}"
                    )
            elif isinstance(img, Image.Image):
                inputs.append(img)

    # 🔹 Отримуємо СПРАВЖНІЙ динамічний список моделей під ваш ключ
    models_to_try = get_available_gemini_models()
    response = None
    last_error = None

    for model_name in models_to_try:
        try:
            logger.info(
                f"🧠 Спроба аналізу через {model_name} для {pair_name}..."
            )
            model = genai.GenerativeModel(model_name)
            generation_config = {
                "response_mime_type": "application/json",
                "temperature": 0.2,
            }
            response = model.generate_content(
                inputs, generation_config=generation_config
            )
            if response and response.text:
                logger.info(f"✅ Успішно використано модель: {model_name}")
                break
        except Exception as e:
            last_error = e
            logger.warning(f"⚠️ Модель {model_name} видала помилку: {e}")

    if not response or not response.text:
        logger.error(
            f"❌ Усі доступні моделі Gemini виявилися недоступними для {pair_name}: {last_error}"
        )
        return _get_fallback_response(
            payload, f"Збій API ШІ: {str(last_error)[:30]}"
        )

    try:
        raw_text = response.text.strip()
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        clean_json = (
            json_match.group(0)
            if json_match
            else re.sub(r"```json\s*|\s*```", "", raw_text).strip()
        )

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
            f"❌ Помилка парсингу JSON від Gemini для {pair_name}: {e}. Текст: {raw_text}"
        )
        return _get_fallback_response(
            payload, "Помилка формату JSON від ШІ"
        )


def _get_fallback_response(payload: dict, error_msg: str) -> dict:
    return {
        "confidence": 0,
        "reason": error_msg,
        "optimal_tf": payload.get("primary_tf", "5m"),
        "suggested_expiration": payload.get("suggested_exp", 5),
    }
