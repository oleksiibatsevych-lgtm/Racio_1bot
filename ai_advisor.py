import io
import json
from google import genai
from google.genai import types

client = genai.Client()

def safe_generate_content(contents, config=None):
    models_to_try = ["gemini-3.6-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(model=model_name, contents=contents, config=config)
            return response
        except Exception as e:
            print(f"⚠️ Модель {model_name} недоступна: {e}")
    raise Exception("Усі моделі Gemini наразі недоступні.")

class AITradingAdvisor:
    def evaluate_signal(self, pair_name: str, signal_data: dict, macro_chart: io.BytesIO, mid_chart: io.BytesIO, micro_chart: io.BytesIO) -> dict:
        prompt = (
            f"Ти професійний квантовий трейдер. Зроби глибокий мультитаймфреймовий аналіз сигналу по парі {pair_name}:\n"
            f"- Запропонований сигнал: {signal_data['signal']}\n"
            f"- Режим ринку та ADX: {signal_data.get('adx')}\n"
            f"- Глобальний тренд (EMA 200 1h): {signal_data['global_trend']}\n"
            f"- Середній тренд (EMA 50 15m): {signal_data['mid_trend']}\n"
            f"- Локальний тренд (EMA 10 5m): {signal_data['local_trend']}\n"
            f"- Причина: {signal_data['reason']}\n"
            f"- RSI: {signal_data.get('rsi')}, ATR: {signal_data.get('atr')}\n\n"
            "Відповідай СУВОРО у форматі JSON без жодних додаткових символів чи markdown-тегів:\n"
            "{\n"
            '  "trend_alignment_analysis": "аналіз умов",\n'
            '  "level_and_bb_analysis": "аналіз рівнів",\n'
            '  "decision": "YES" або "NO",\n'
            '  "confidence": ціле число від 1 до 10,\n'
            '  "expiration": ціле число хвилин,\n'
            '  "reason": "коротке резюме українською"\n'
            "}"
        )
        try:
            macro_chart.seek(0)
            mid_chart.seek(0)
            micro_chart.seek(0)
            img1 = types.Part.from_bytes(data=macro_chart.getvalue(), mime_type="image/png")
            img2 = types.Part.from_bytes(data=mid_chart.getvalue(), mime_type="image/png")
            img3 = types.Part.from_bytes(data=micro_chart.getvalue(), mime_type="image/png")

            response = safe_generate_content(contents=[prompt, img1, img2, img3], config=types.GenerateContentConfig(temperature=0.1))
            raw_text = response.text or "{}"
            bt = chr(96) * 3
            raw_text = raw_text.replace(f"{bt}json", "").replace(bt, "").strip()
            return json.loads(raw_text)
        except Exception as e:
            print(f"❌ Помилка AI Audit: {e}")
            return {
                "decision": "NO",
                "confidence": 0,
                "expiration": 5,
                "reason": "ШІ тимчасово перевантажений, захисне блокування"
            }
