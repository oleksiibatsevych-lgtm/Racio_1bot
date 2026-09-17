import os
import json
import requests
import google.generativeai as genai
from PIL import Image

class AITradingAdvisor:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            
        self.gemini_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp"
        ]

    def _evaluate_with_openrouter(self, prompt):
        """Резервний аналіз через OpenRouter (DeepSeek / Llama)"""
        if not self.openrouter_key:
            return None
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }

        models_to_try = [
            "deepseek/deepseek-r1:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free"
        ]

        for model_slug in models_to_try:
            payload = {
                "model": model_slug,
                "messages": [{"role": "user", "content": prompt}]
            }
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=12)
                if response.status_code == 200:
                    data = response.json()
                    res_text = data["choices"][0]["message"]["content"]
                    if res_text:
                        return res_text
            except Exception as e:
                print(f"⚠️ Помилка OpenRouter ({model_slug}): {e}")

        return None

    def evaluate_signal(self, name, payload, macro_chart, mid_chart, micro_chart):
        prompt = f"""
        Ти професійний трейдер та ризик-менеджер. Проаналізуй ринкові дані для активу {name}.
        Параметри сигналу:
        - Сигнал: {payload.get('signal')}
        - RSI: {payload.get('rsi')}
        - ADX: {payload.get('adx')}
        - Глобальний тренд (1h): {payload.get('global_trend')}
        - Середній тренд (15m): {payload.get('mid_trend')}
        - Технічна причина: {payload.get('reason')}
        - ATR: {payload.get('atr')}
        - Розрахована динамічна експірація: {payload.get('suggested_exp')} хв
        - Відстань до EMA: {payload.get('ema_dist', 0)}%
        - Wick Ratio (відношення тіні): {payload.get('wick_ratio', 0)}
        - Волатильність: {payload.get('volatility_ratio', 1)}

        Оціни доцільність входу в угоду та підтвердь або скоригуй час експірації залежно від динаміки ринку.
        Відповідь надай ВИКЛЮЧНО у форматі JSON без жодних додаткових символів чи Markdown обгородок:
        {{
            "decision": "YES" або "NO",
            "confidence": число від 1 до 10,
            "suggested_expiration": число хвилин,
            "reason": "Коротке обґрунтування українською мовою"
        }}
        """

        response_text = None

        # 1. Основна спроба: Gemini
        if self.gemini_key:
            content_parts = [prompt]
            for chart in [macro_chart, mid_chart, micro_chart]:
                if chart:
                    try:
                        chart.seek(0)
                        content_parts.append(Image.open(chart))
                    except Exception as e:
                        print(f"⚠️ Помилка відкриття зображення для Gemini: {e}")

            for model_name in self.gemini_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content(content_parts)
                    if resp and resp.text:
                        response_text = resp.text
                        break
                except Exception as e:
                    print(f"⚠️ Збій Gemini ({model_name}): {e}")

        # 2. Резервна спроба: OpenRouter
        if not response_text and self.openrouter_key:
            print("🔄 Gemini недоступна. Перемикаємося на резервний OpenRouter...")
            response_text = self._evaluate_with_openrouter(prompt)

        # 3. Якщо жоден сервіс не відповів
        if not response_text:
            return {
                "decision": "NO",
                "confidence": 1,
                "suggested_expiration": payload.get('suggested_exp', 5),
                "reason": "Усі AI-моделі недоступні."
            }

        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"): clean_text = clean_text[7:]
            if clean_text.startswith("```"): clean_text = clean_text[3:]
            if clean_text.endswith("```"): clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            result = json.loads(clean_text)
            return {
                "decision": result.get("decision", "NO"),
                "confidence": int(result.get("confidence", 5)),
                "suggested_expiration": int(result.get("suggested_expiration", payload.get('suggested_exp', 5))),
                "reason": result.get("reason", "ШІ не надав детального пояснення")
            }
        except Exception as e:
            print(f"⚠️ Помилка парсингу JSON від ШІ: {e}")
            return {
                "decision": "NO",
                "confidence": 1,
                "suggested_expiration": payload.get('suggested_exp', 5),
                "reason": "Помилка обробки відповіді ШІ"
            }
