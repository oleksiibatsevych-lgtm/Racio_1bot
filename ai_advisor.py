import os
import json
import google.generativeai as genai
from PIL import Image

class AITradingAdvisor:
    def __init__(self):
        self.models_to_try = [
            "gemini-3.6-flash",
            "gemini-2.0-flash"
        ]

    def evaluate_signal(self, name, payload, macro_chart, mid_chart, micro_chart):
        prompt = f"""
        Ти професійний трейдер та ризик-менеджер. Проаналізуй ринкові дані та графіки (1h, 15m, 5m) для активу {name}.
        Параметри сигналу:
        - Сигнал: {payload.get('signal')}
        - RSI: {payload.get('rsi')}
        - ADX: {payload.get('adx')}
        - Глобальний тренд (1h): {payload.get('global_trend')}
        - Середній тренд (15m): {payload.get('mid_trend')}
        - Локальний тренд (5m): {payload.get('local_trend')}
        - Технічна причина: {payload.get('reason')}
        - ATR: {payload.get('atr')}

        Твоє завдання — оцінити доцільність входу в угоду за цим сигналом, а також визначити оптимальний час експірації у хвилинах.
        
        Відповідь надай ВИКЛЮЧНО у форматі JSON без жодних додаткових символів чи markdown-обгородок (чистий JSON):
        {{
            "decision": "YES" або "NO",
            "confidence": число від 1 до 10,
            "suggested_expiration": число хвилин (наприклад, 5, 10, 15),
            "reason": "Коротке та чітке обґрунтування українською мовою"
        }}
        """

        content_parts = [prompt]
        for chart in [macro_chart, mid_chart, micro_chart]:
            if chart:
                try:
                    chart.seek(0)
                    content_parts.append(Image.open(chart))
                except Exception as e:
                    print(f"⚠️ Помилка відкриття графіка для ШІ: {e}")

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
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": 10,
                "reason": "Схвалено за замовчуванням (аудит пропущено)"
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
                "decision": result.get("decision", "YES"),
                "confidence": int(result.get("confidence", 7)),
                "suggested_expiration": int(result.get("suggested_expiration", 10)),
                "reason": result.get("reason", "Схвалено ШІ")
            }
        except Exception as e:
            return {
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": 10,
                "reason": "Схвалено за замовчуванням"
            }
