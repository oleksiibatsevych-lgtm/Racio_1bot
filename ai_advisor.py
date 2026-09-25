import json
import logging
import os
import google.api_core.exceptions
import google.generativeai as genai
from PIL import Image

logger = logging.getLogger(__name__)


class AITradingAdvisor:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GEMINI_KEY"
        )
        if self.api_key:
            genai.configure(api_key=self.api_key.strip())

        # Список моделей у порядку пріоритету
        self.models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]

    def evaluate_signal(
        self,
        name: str,
        payload: dict,
        macro_chart=None,
        mid_chart=None,
        micro_chart=None,
    ) -> dict:
        """Здійснює аналіз сигналу за допомогою Gemini з підтримкою мульти-модельності."""
        suggested_exp = payload.get("suggested_exp", 5)

        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY відсутній. ШІ-аналіз пропущено.")
            return {
                "decision": "NO",
                "confidence": 0,
                "suggested_expiration": suggested_exp,
                "reason": "GEMINI_API_KEY не налаштовано",
            }

        prompt = f"""
        Ти професійний трейдер та ризик-менеджер. Проаналізуй ринкові дані та графіки для активу {name}.
        Параметри сигналу:
        - Напрямок: {payload.get('signal')}
        - Поточна ціна: {payload.get('current_price')}
        - RSI: {payload.get('rsi')}
        - ADX: {payload.get('adx')}
        - Глобальний тренд (1h): {payload.get('global_trend', 'Невідомо')}
        - Середній тренд (15m): {payload.get('mid_trend', 'Невідомо')}
        - Дивергенція: {payload.get('divergence', 'NONE')}
        - Технічна причина: {payload.get('reason')}
        - ATR Ratio: {payload.get('atr_ratio')}
        - Рекомендована експірація: {suggested_exp} хв

        Оціни доцільність входу в угоду та підтвердь або скоригуй час експірації.
        Відповідь надай ВИКЛЮЧНО у форматі JSON без жодних додаткових символів чи обгорток markdown:
        {{
            "decision": "YES" або "NO",
            "confidence": <число від 1 до 10>,
            "suggested_expiration": <число в хвилинах>,
            "reason": "<коротке обґрунтування українською мовою>"
        }}
        """

        content_parts = [prompt]
        for chart in [macro_chart, mid_chart, micro_chart]:
            if chart:
                try:
                    chart.seek(0)
                    content_parts.append(Image.open(chart))
                except Exception as e:
                    logger.warning(f"⚠️ Помилка відкриття графіка для ШІ: {e}")

        response_text = None
        for model_name in self.models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    content_parts, request_options={"timeout": 10}
                )
                if response and response.text:
                    response_text = response.text
                    break
            except google.api_core.exceptions.ResourceExhausted:
                logger.warning(f"⚠️ Квота вичерпана для моделі {model_name}.")
                continue
            except Exception as e:
                logger.warning(f"⚠️ Модель {model_name} недоступна: {e}")
                continue

        if not response_text:
            return {
                "decision": "NO",
                "confidence": 1,
                "suggested_expiration": suggested_exp,
                "reason": "Усі моделі Gemini наразі недоступні або перевищено квоту.",
            }

        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json", "", 1)
            if clean_text.startswith("```"):
                clean_text = clean_text.replace("```", "", 1)
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            result = json.loads(clean_text)
            return {
                "decision": str(result.get("decision", "NO")).upper(),
                "confidence": int(result.get("confidence", 0)),
                "suggested_expiration": int(
                    result.get("suggested_expiration", suggested_exp)
                ),
                "reason": str(
                    result.get("reason", "ШІ не надав детального пояснення")
                ),
            }
        except Exception as e:
            logger.error(
                f"⚠️ Помилка парсингу відповіді ШІ: {e}. Текст: {response_text}"
            )
            return {
                "decision": "NO",
                "confidence": 1,
                "suggested_expiration": suggested_exp,
                "reason": "Помилка обробки відповіді ШІ (некоректний JSON)",
            }


ai_advisor_instance = AITradingAdvisor()


def analyze_signal_with_gemini(symbol, payload, chart_images=[]):
    """Обгортка для сумісності з функціональним викликом."""
    c_macro = chart_images[0] if len(chart_images) > 0 else None
    c_mid = chart_images[1] if len(chart_images) > 1 else None
    c_micro = chart_images[2] if len(chart_images) > 2 else None
    return ai_advisor_instance.evaluate_signal(
        symbol, payload, c_macro, c_mid, c_micro
    )
