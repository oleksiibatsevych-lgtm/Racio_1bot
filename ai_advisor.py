import os
import json
import requests
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

class AITradingAdvisor:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        
        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception as e:
                logger.error(f"⚠️ Помилка конфігурації Gemini API: {e}")

        # Актуальні робочі моделі Gemini
        self.gemini_models = [
            "gemini-2.0-flash",
            "gemini-1.5-flash"
        ]

    def _evaluate_with_openrouter(self, prompt):
        """Резервний аналіз через OpenRouter"""
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
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    res_text = data["choices"][0]["message"]["content"]
                    if res_text:
                        return res_text
            except Exception as e:
                logger.warning(f"⚠️ Помилка OpenRouter ({model_slug}): {e}")

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

        # 1. Спроба: Gemini
        if self.gemini_key:
            images_parts = []
            for chart in [macro_chart, mid_chart, micro_chart]:
                if chart:
                    try:
                        chart.seek(0)
                        img_bytes = chart.getvalue()
                        if img_bytes:
                            images_parts.append({
                                "mime_type": "image/png",
                                "data": img_bytes
                            })
                    except Exception as e:
                        logger.warning(f"⚠️ Помилка зчитування байтів графіка: {e}")

            content_parts = [prompt] + images_parts

            for model_name in self.gemini_models:
                try:
                    model = genai.GenerativeModel(model_name)
                    resp = model.generate_content(
                        content_parts,
                        safety_settings=[
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        ]
                    )
                    
                    if resp and resp.candidates and resp.candidates[0].content.parts:
                        response_text = resp.text
                        logger.info(f"✅ Успішний аналіз через Gemini ({model_name})")
                        break
                except Exception as e:
                    logger.warning(f"⚠️ Збій Gemini ({model_name}): {e}")

        # 2. Спроба: OpenRouter
        if not response_text and self.openrouter_key:
            logger.info("🔄 Gemini недоступна. Перемикання на OpenRouter...")
            response_text = self._evaluate_with_openrouter(prompt)

        # 3. ФОЛБЕК: При недоступності ШІ пропускаємо за індикаторами (YES)
        if not response_text:
            logger.info(f"ℹ️ ШІ недоступний для {name}. Сигнал пропущено за індикаторами.")
            return {
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": payload.get('suggested_exp', 5),
                "reason": "ШІ недоступний (пройдено за індикаторами)"
            }

        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"): clean_text = clean_text[7:]
            if clean_text.startswith("```"): clean_text = clean_text[3:]
            if clean_text.endswith("```"): clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            result = json.loads(clean_text)
            return {
                "decision": result.get("decision", "YES"),
                "confidence": int(result.get("confidence", 7)),
                "suggested_expiration": int(result.get("suggested_expiration", payload.get('suggested_exp', 5))),
                "reason": result.get("reason", "Схвалено ШІ")
            }
        except Exception as e:
            logger.error(f"⚠️ Помилка парсингу JSON від ШІ: {e}")
            return {
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": payload.get('suggested_exp', 5),
                "reason": "ШІ недоступний (помилка JSON, пройдено за індикаторами)"
            }
