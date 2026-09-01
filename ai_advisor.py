import os
import json
import google.generativeai as genai
from PIL import Image

class AITradingAdvisor:
    def __init__(self):
        self.models_to_try = [
            "gemini-3.6-flash"
        ]

    def evaluate_signal(self, name, payload, combined_chart):
        """
        Проводить візуальний та технічний аудит сигналу за допомогою Gemini Vision, використовуючи 
        зведене зображення графіків та лаконічний системний промпт[span_3](start_span)[span_3](end_span).
        """
        prompt = f"""
        Ти професійний трейдер. Оціни торговий сигнал для {name} за даними та зведеним графіком (1h/15m/5m):
        - Сигнал: {payload.get('signal')} | RSI: {payload.get('rsi')} | ADX: {payload.get('adx')}
        - Тренд (гл/сер/лок): {payload.get('global_trend')} / {payload.get('mid_trend')} / {payload.get('local_trend')}
        - Причина: {payload.get('reason')} | ATR: {payload.get('atr')}

        Визнач доцільність входу та оптимальний час експірації (хв).
        Відповідь надай ВИКЛЮЧНО у форматі чинного JSON без markdown-обгородок:
        {{
            "decision": "YES" або "NO",
            "confidence": число від 1 до 10,
            "suggested_expiration": число хвилин,
            "reason": "Коротке обґрунтування українською"
        }}
        """

        content_parts = [prompt]
        if combined_chart:
            try:
                combined_chart.seek(0)
                content_parts.append(Image.open(combined_chart))
            except Exception as e:
                print(f"⚠️ Помилка відкриття зведеного графіка для ШІ: {e}")

        response_text = None
        for model_name in self.models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(content_parts)
                if response and response.text:
                    response_text = response.text
                    break
            except Exception as e:
                print(f"⚠️ Модель {model_name} недоступна: {e}")
                continue

        if not response_text:
            return {
                "decision": "NO",
                "confidence": 1,
                "suggested_expiration": 10,
                "reason": "Усі моделі Gemini наразі недоступні."
            }

        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.startswith("```"):
                clean_text = clean_text[3:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            result = json.loads(clean_text)
            return {
                "decision": result.get("decision", "NO"),
                "confidence": int(result.get("confidence", 5)),
                "suggested_expiration": int(result.get("suggested_expiration", 10)),
                "reason": result.get("reason", "ШІ не надав пояснення")
            }
        except Exception as e:
            return {
                "decision": "NO",
                "confidence": 1,
                "suggested_expiration": 10,
                "reason": "Помилка обробки відповіді ШІ"
            }
