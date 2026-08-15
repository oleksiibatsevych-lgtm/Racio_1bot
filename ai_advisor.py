import json
from google import genai
from google.genai import types
import io

client = genai.Client()


def safe_generate_content(contents, config=None):
  models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash"]
  for model_name in models_to_try:
    try:
      response = client.models.generate_content(
          model=model_name, contents=contents, config=config
      )
      return response
    except Exception as e:
      print(f"⚠️ Модель {model_name} недоступна: {e}")
  raise Exception("Усі моделі Gemini наразі недоступні.")


class AITradingAdvisor:

  def evaluate_signal(
      self,
      pair_name: str,
      signal_data: dict,
      macro_chart: io.BytesIO,
      mid_chart: io.BytesIO,
      micro_chart: io.BytesIO,
  ) -> dict:
    prompt = (
        f"Ти професійний квантовий трейдер. Зроби глибокий мультитаймфреймовий аналіз сигналу по парі {pair_name}:\n"
        f"- Запропонований сигнал: {signal_data['signal']}\n"
        f"- Режим ринку та ADX: {signal_data.get('adx')}\n"
        f"- Глобальний тренд (EMA 200 1h): {signal_data['global_trend']}\n"
        f"- Середній тренд (EMA 20 15m): {signal_data['mid_trend']}\n"
        f"- Локальний тренд (EMA 20 5m): {signal_data['local_trend']}\n"
        f"- Причина: {signal_data['reason']}\n"
        f"- RSI: {signal_data.get('rsi')}, ATR (волатильність): {signal_data.get('atr')}\n\n"
        "Зображення: 1) Макротренд (1h), 2) Середній таймфрейм (15m), 3) Мікроструктура (5m зі рівнями Swing та Смугами Боллінджера).\n"
        "ПРОАНАЛІЗУЙ КРОК ЗА КРОКОМ (Chain of Thought):\n"
        "1. Чи збігаються ринкові умови (флєт/тренд за ADX) та сигнали?\n"
        "2. Наскільки якісний відскок від рівня або межі Боллінджера?\n"
        "3. Чи виправдана поточна волатильність ATR та ADX?\n"
        "Визнач оптимальний час експірації у хвилинах (від 3 до 25 хв) та справедливу оцінку впевненості (confidence) від 1 до 10.\n"
        "Відповідай СУВОРО у форматі JSON без жодних додаткових символів чи markdown-тегів:\n"
        "{\n"
        '  "trend_alignment_analysis": "короткий аналіз умов ринку та таймфреймів",\n'
        '  "level_and_bb_analysis": "аналіз рівнів та Боллінджера",\n'
        '  "decision": "YES" або "NO",\n'
        '  "confidence": ціле число від 1 до 10,\n'
        '  "expiration": ціле число хвилин (наприклад, 5, 8, 12, 15 тощо),\n'
        '  "reason": "коротке резюме українською"\n'
        "}"
    )
    try:
      macro_chart.seek(0)
      mid_chart.seek(0)
      micro_chart.seek(0)
      img1 = types.Part.from_bytes(
          data=macro_chart.getvalue(), mime_type="image/png"
      )
      img2 = types.Part.from_bytes(
          data=mid_chart.getvalue(), mime_type="image/png"
      )
      img3 = types.Part.from_bytes(
          data=micro_chart.getvalue(), mime_type="image/png"
      )

      response = safe_generate_content(
          contents=[prompt, img1, img2, img3],
          config=types.GenerateContentConfig(temperature=0.1),
      )

      raw_text = response.text
      if not raw_text:
        raw_text = "{}"

      bt = chr(96) * 3
      raw_text = (
          raw_text.replace(f"{bt}json", "").replace(bt, "").strip()
      )

      result = json.loads(raw_text)
      return result
    except Exception as e:
      print(f"❌ Помилка AI Audit: {e}")
      return {
          "decision": "NO",
          "confidence": 0,
          "expiration": 5,
          "reason": "ШІ тимчасово перевантажений, захисне блокування",
      }
