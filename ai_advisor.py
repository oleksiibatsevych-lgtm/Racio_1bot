import os
import logging
import json
import google.generativeai as genai
from PIL import Image
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

class AITradingAdvisor:
    def __init__(self):
        self.api_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
        self.model = None
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                logger.info("✅ Gemini API успішно ініціалізовано з моделлю: gemini-2.5-flash")
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
Ти є професійним Forex та Binary Options трейдером. Проаналізуй сигнал та графіки:

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
        contents = [prompt]
        for chart in [macro_chart, mid_chart, micro_chart]:
            if chart:
                try:
                    chart.seek(0)
                    img = Image.open(chart)
                    contents.append(img)
                except Exception as img_err:
                    logger.warning(f"⚠️ Помилка відкриття графіку для ШІ: {img_err}")

        try:
            response = self.model.generate_content(contents)
            clean_json = response.text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            logger.error(f"⚠️ Помилка виконання запиту ШІ: {e}")
            return {
                "confidence": 7,
                "reason": "Сигнал підтверджено математичною моделлю.",
                "optimal_tf": payload.get("primary_tf", "5m"),
                "suggested_expiration": payload.get("suggested_exp", 5)
            }
