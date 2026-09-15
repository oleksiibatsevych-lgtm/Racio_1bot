import os
import json
import base64
import requests
import google.generativeai as genai
from PIL import Image

class AITradingAdvisor:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            
        # Робочі та стабільні назви моделей Gemini
        self.gemini_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]

    def _evaluate_with_openrouter(self, prompt, chart_bytes_list):
        """Резервний виклик через OpenRouter (DeepSeek / Llama)"""
        if not self.openrouter_key:
            return None
            
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json"
        }

        content = [{"type": "text", "text": prompt}]
        
        # Додавання графіків у базі base64, якщо вони присутні
        for buf in chart_bytes_list:
            if buf:
                try:
                    buf.seek(0)
                    b64 = base64.b64encode(buf.read()).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
                except Exception:
                    pass

        payload = {
            "model": "deepseek/deepseek-r1:free",
            "messages": [{"role": "user", "content": content}]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"⚠️ Помилка OpenRouter: {e}")
            
        return None

    def evaluate_signal(self, name, payload, macro_chart, mid_chart, micro_chart):
        prompt = f"""
        Ти професійний трейдер та ризик-менеджер. Проаналізуй ринкові дані та графіки для активу {name}.
        Параметри сигналу:
        - Сигнал: {payload.get('signal')}
        - RSI: {payload.get('rsi')}
        - ADX: {payload.get('adx')}
        - Глобальний тренд (1h): {payload.get('global_trend')}
        - Середній тренд (15m): {payload.get('mid_trend')}
        - Технічна причина: {payload.get('reason')}
        - ATR: {payload.get('atr')}
        - Розрахована динамічна експірація: {payload.get('suggested_exp')} хв

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

        # 1. Перша спроба: Gemini (офіційні моделі 1.5)
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

        # 2. Резервна спроба: OpenRouter (DeepSeek / Llama), якщо Gemini видала помилку або ліміти
        if not response_text and self.openrouter_key:
            print("🔄 Gemini недоступна. Перемикаємося на резервний OpenRouter...")
            charts = [macro_chart, mid_chart, micro_chart]
            response_text = self._evaluate_with_openrouter(prompt, charts)

        # 3. Якщо жоден сервіс не відповів
        if not response_text:
            return {
                "decision": "NO",
                "confidence": 1,
                "suggested_expiration": payload.get('suggested_exp', 5),
                "reason": "Усі AI-моделі (Gemini та OpenRouter) недоступні або вичерпано ліміти."
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
        except Exception:
            return {
                "decision": "NO",
                "confidence": 1,
                "suggested_expiration": payload.get('suggested_exp', 5),
                "reason": "Помилка обробки відповіді ШІ"
            }
