import os
import json
import google.generativeai as genai
from PIL import Image

class AITradingAdvisor:
    def __init__(self):
        self.models_to_try = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash"
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
        
        Відповідь надай ВИКЛЮЧНО у форматі JSON з такими полями:
        - "decision": "YES" або "NO"
        - "confidence": ціле число від 1 до 10
        - "suggested_expiration": ціле число хвилин (наприклад, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
        - "reason": "Коротке та чітке обґрунтування українською мовою"
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
        generation_config = {"response_mime_type": "application/json"}

        for model_name in self.models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(content_parts, generation_config=generation_config)
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
            result = json.loads(response_text.strip())
            return {
                "decision": result.get("decision", "YES"),
                "confidence": int(result.get("confidence", 7)),
                "suggested_expiration": int(result.get("suggested_expiration", 10)),
                "reason": result.get("reason", "Схвалено ШІ")
            }
        except Exception as e:
            print(f"⚠️ Помилка парсингу JSON від ШІ: {e}")
            return {
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": 10,
                "reason": "Схвалено за замовчуванням"
            }
