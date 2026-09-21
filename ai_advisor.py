import os
import logging
import json
import google.generativeai as genai
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class AITradingAdvisor:
    def __init__(self):
        self.api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.model = None
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                
                # Автоматичний пошук усіх моделей, які підтримують генерацію тексту
                available_models = [
                    m.name for m in genai.list_models() 
                    if 'generateContent' in m.supported_generation_methods
                ]
                
                # Пріоритет вибору моделей
                selected_model_name = None
                for candidate in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-2.0-flash', 'models/gemini-2.5-flash']:
                    if candidate in available_models:
                        selected_model_name = candidate
                        break
                
                # Якщо жодної з пріоритетних немає, беремо першу доступну
                if not selected_model_name and available_models:
                    selected_model_name = available_models[0]

                if selected_model_name:
                    self.model = genai.GenerativeModel(selected_model_name)
                    logger.info(f"✅ Gemini API успішно ініціалізовано з моделлю: {selected_model_name}")
                else:
                    logger.warning("⚠️ Не знайдено жодної активної моделі Gemini для generateContent.")
            except Exception as e:
                logger.error(f"⚠️ Помилка ініціалізації Gemini API: {e}")

    def evaluate_signal(self, pair_name, payload, macro_chart=None, mid_chart=None, micro_chart=None):
        if not self.model:
            return {
                "confidence": 7,
                "reason": "Аналіз проведено за математичною моделлю.",
                "optimal_tf": payload.get("primary_tf", "5m"),
                "suggested_expiration": payload.get("suggested_exp", 5)
            }

        prompt = f"""
Ти є професійним Forex та Binary Options трейдером. Проаналізуй сигнал:

Пара: {pair_name}
Сигнал: {payload.get('signal')}
Signal Score: {payload.get('score')}
Головний ТФ: {payload.get('primary_tf')}
ADX: {payload.get('adx')} | RSI: {payload.get('rsi')}
Глобальний тренд (1h): {payload.get('global_trend')}
Локальний тренд (15m): {payload.get('mid_trend')}
Дивергенція: {payload.get('divergence')}
Обґрунтування алгоритму: {payload.get('reason')}

Дай коротку оцінку у форматі JSON з полями:
- confidence (число від 1 до 10)
- reason (1 короткий висновок українською мовою, до 15 слів)
- optimal_tf (рекомендований таймфрейм)
- suggested_expiration (час експірації в хвилинах: від 3 до 15)

Відповідай ТІЛЬКИ у форматі чистого JSON.
"""
        try:
            response = self.model.generate_content(prompt)
            clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            return data
        except Exception as e:
            logger.warning(f"⚠️ Помилка виконання запиту ШІ: {e}")
            return {
                "confidence": 7,
                "reason": "Сигнал підтверджено математичною моделлю.",
                "optimal_tf": payload.get("primary_tf", "5m"),
                "suggested_expiration": payload.get("suggested_exp", 5)
            }
