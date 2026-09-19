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
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception as e:
                print(f"⚠️ Помилка ініціалізації Gemini API: {e}")
            
        self.gemini_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp"
        ]

    def _evaluate_with_openrouter(self, prompt):
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
        calculated_signal = payload.get('signal', 'CALL')
        calculated_exp = payload.get('suggested_exp', 5)
        primary_tf = payload.get('primary_tf', '5m')
        score = payload.get('score', 60)

        prompt = f"""
        Ти старший аналітик та ризик-менеджер систем алгоритмічного трейдингу.
        Надано розрахований математичний сигнал для активу {name}:
        
        - Напрямок угоди: {calculated_signal}
        - Сила сигналу (Signal Score): {score}/100
        - Базовий таймфрейм: {primary_tf}
        - Розрахована початкова експірація: {calculated_exp} хв
        - RSI: {payload.get('rsi')}
        - ADX: {payload.get('adx')}
        - Дивергенція: {payload.get('divergence', 'NONE')}
        - Волатильність (ATR Ratio): {payload.get('volatility_ratio', 1.0)}
        - Wick Ratio (відношення тіні): {payload.get('wick_ratio', 0)}
        - Відстань до EMA: {payload.get('ema_dist', 0)}%
        - Математичне обґрунтування: {payload.get('reason')}

        ТВОЄ ЗАВДАННЯ:
        1. Проаналізуй завантажені мульти-таймфреймові графіки (макро, середній, мікро).
        2. Оціни актуальні свічкові паттерни, локальні рівні підтримки/опору та поточний імпульс.
        3. Підтвердь або скоригуй рекомендувану експірацію (у хвилинах) для ідеальної точки входу.
        4. Визнач рівень впевненості (від 1 до 10).
        5. Надай короткий аналітичний коментар українською мовою.

        Відповідь надай ВИКЛЮЧНО у форматі JSON без жодних додаткових символів чи Markdown обгородок:
        {{
            "confidence": число від 1 до 10,
            "suggested_expiration": число хвилин,
            "optimal_tf": "{primary_tf}",
            "reason": "Короткий аналітичний висновок та порада щодо входу"
        }}
        """

        response_text = None

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

        if not response_text and self.openrouter_key:
            print("🔄 Gemini недоступна. Перемикаємося на резервний OpenRouter...")
            response_text = self._evaluate_with_openrouter(prompt)

        if not response_text:
            return {
                "decision": "YES",
                "confidence": 6,
                "suggested_expiration": calculated_exp,
                "optimal_tf": primary_tf,
                "reason": "Аналіз проведено на основі математичної моделі (ШІ-сервіси офлайн)."
            }

        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"): clean_text = clean_text[7:]
            if clean_text.startswith("```"): clean_text = clean_text[3:]
            if clean_text.endswith("```"): clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            result = json.loads(clean_text)
            
            return {
                "decision": "YES",
                "confidence": int(result.get("confidence", 6)),
                "suggested_expiration": int(result.get("suggested_expiration", calculated_exp)),
                "optimal_tf": str(result.get("optimal_tf", primary_tf)),
                "reason": str(result.get("reason", "ШІ підтвердив параметри входу."))
            }
        except Exception as e:
            print(f"⚠️ Помилка парсингу JSON від ШІ: {e}")
            return {
                "decision": "YES",
                "confidence": 5,
                "suggested_expiration": calculated_exp,
                "optimal_tf": primary_tf,
                "reason": "Аналіз сформовано за математичними індикаторами."
            }
