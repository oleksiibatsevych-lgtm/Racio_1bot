import json
import logging
import re
import google.generativeai as genai


def analyze_signal_with_gemini(
    pair_name: str, payload: dict, chart_images: list = None
) -> dict:
    """Аналіз сигналу за допомогою Gemini 2.5 Flash із примусовим JSON-форматуванням."""
    prompt = f"""
Ти є професійним Forex та Binary Options трейдером. Проаналізуй торгову ситуацію та надай свій вердикт.

📊 ВХІДНІ ДАНІ СИГНАЛУ:
- Пара: {pair_name}
- Напрямок: {payload.get('signal')}
- Основний ТФ: {payload.get('primary_tf', '5m')}
- RSI: {payload.get('rsi')} | ADX: {payload.get('adx')}
- Дивергенція: {payload.get('divergence', 'NONE')}
- Обґрунтування тех. аналізу: {payload.get('reason')}
- ML-ймовірність: {payload.get('ml_prob', 0)}%

💡 ЗАВДАННЯ:
1. Оціни впевненість від 1 до 10 (де 1 - високий ризик/шум, 10 - ідеальний сетап).
2. Вкажи КОНКРЕТНУ причину чи застереження (до 20 слів українською).
3. Визнач оптимальну експірацію у хвилинах.

ВІДПОВІДЬ СТРОГО У ФОРМАТІ JSON (без сміття та обгорток):
{{
    "confidence": <число 1-10>,
    "reason": "<короткий конкретний висновок>",
    "optimal_tf": "{payload.get('primary_tf', '5m')}",
    "suggested_expiration": <число хвилин>
}}
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Примусово вимагаємо чистий JSON від API
        generation_config = {
            "response_mime_type": "application/json",
            "temperature": 0.2,
        }

        inputs = [prompt]
        if chart_images:
            inputs.extend(chart_images)

        response = model.generate_content(
            inputs, generation_config=generation_config
        )

        raw_text = response.text.strip()
        # Додаткова страховка очищення від ```json
        clean_text = re.sub(r"```json\s*|\s*```", "", raw_text).strip()
        data = json.loads(clean_text)

        return {
            "confidence": int(data.get("confidence", 5)),
            "reason": str(data.get("reason", "Аналіз виконано")),
            "optimal_tf": str(data.get("optimal_tf", "5m")),
            "suggested_expiration": int(
                data.get("suggested_expiration", 5)
            ),
        }

    except Exception as e:
        # Логуємо точну помилку в консоль Render
        logging.error(f"❌ Помилка Gemini API у ai_advisor: {e}")

        # Маркерний fallback: якщо 0/10 — одразу видно, що стався збій API або ключа
        return {
            "confidence": 0,
            "reason": f"Збій ШІ: {str(e)[:30]}",
            "optimal_tf": payload.get("primary_tf", "5m"),
            "suggested_expiration": payload.get("suggested_exp", 5),
        }
