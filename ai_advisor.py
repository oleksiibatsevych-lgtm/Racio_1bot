import json
import logging
import os
import google.api_core.exceptions
import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get(
    "GEMINI_KEY"
)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY.strip())


def analyze_signal_with_gemini(symbol, payload, chart_images=[]):
    """ШІ-аналіз торгового сигналу з обробкою таймаутів та обмежень квоти."""
    suggested_exp = payload.get("suggested_exp", 5)

    if not GEMINI_API_KEY:
        logger.warning("⚠️ GEMINI_API_KEY відсутній. ШІ-аналіз пропущено.")
        return {
            "confidence": 0,
            "reason": "GEMINI_API_KEY не налаштовано",
            "suggested_expiration": suggested_exp,
        }

    prompt = f"""
    Проаналізуй торговий сигнал для валютної пари {symbol}:
    - Напрямок: {payload.get('signal')}
    - Оцінка алгоритму: {payload.get('score')}
    - Основний таймфрейм: {payload.get('primary_tf')}
    - Стратегія: {payload.get('strategy_type')}
    - Поточна ціна: {payload.get('current_price')}
    - Pivots: P={payload.get('pivot_p')}, R1={payload.get('pivot_r1')}, S1={payload.get('pivot_s1')}
    - ADX: {payload.get('adx')}, RSI: {payload.get('rsi')}
    - Дивергенція: {payload.get('divergence')}
    - ATR Ratio: {payload.get('atr_ratio')}
    - Причина формування: {payload.get('reason')}
    - ML Ймовірність: {payload.get('ml_prob')}%
    - Рекомендована експірація: {suggested_exp} хв.

    Оціни впевненість від 1 до 10 та надай коротке пояснення (до 2 речень).
    Відповідь надай СУВОРО у форматі JSON без форматування markdown:
    {{
        "confidence": <число від 1 до 10>,
        "reason": "<коротке пояснення>",
        "suggested_expiration": <число в хвилинах>
    }}
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")

        contents = [prompt]
        for img_buf in chart_images:
            if img_buf:
                try:
                    img_buf.seek(0)
                    image = Image.open(img_buf)
                    contents.append(image)
                except Exception as img_err:
                    logger.warning(
                        f"⚠️ Не вдалося прочитати графік для Gemini: {img_err}"
                    )

        # 🟢 Таймаут 5 секунд — бот не зависає в очікуванні відповіді
        response = model.generate_content(
            contents=contents, request_options={"timeout": 5}
        )

        txt = response.text.strip()
        if txt.startswith("```json"):
            txt = txt.replace("```json", "").replace("```", "").strip()
        elif txt.startswith("```"):
            txt = txt.replace("```", "").strip()

        data = json.loads(txt)
        return {
            "confidence": int(data.get("confidence", 0)),
            "reason": str(data.get("reason", "Оцінка ШІ")),
            "suggested_expiration": int(
                data.get("suggested_expiration", suggested_exp)
            ),
        }

    except google.api_core.exceptions.ResourceExhausted:
        # 🟢 Миттєве повернення при перевищенні квоти без 20-секундної паузи
        logger.warning(
            f"⚠️ Перевищено ліміт квоти Gemini (ResourceExhausted) для {symbol}. Миттєвий фолбек!"
        )
        return {
            "confidence": 0,
            "reason": "Перевищено ліміт запитів Gemini (Instant Fallback)",
            "suggested_expiration": suggested_exp,
        }

    except Exception as e:
        logger.warning(f"⚠️ Помилка Gemini API для {symbol}: {e}")
        return {
            "confidence": 0,
            "reason": f"ШІ недоступний ({e})",
            "suggested_expiration": suggested_exp,
        }
