import json
import os
import google.generativeai as genai


class AITradingAdvisor:

  def __init__(self):
    # Ініціалізація Gemini API (ключ автоматично зчитується з환경них змінних Render)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
      genai.configure(api_key=api_key)

    # Використовуємо стабільну і перевірену модель gemini-1.5-flash
    self.model = genai.GenerativeModel("gemini-1.5-flash")

  def evaluate_signal(
      self,
      technical_data: dict,
      image_1h,
      image_15m,
      image_5m,
      asset_name: str,
  ) -> dict:
    # Базова відповідь за замовчуванням у разі перевантаження чи помилки API
    default_response = {
        "decision": "NO",
        "confidence": 0,
        "reason": "ШІ тимчасово перевантажений, захисне блокування",
        "expiry": 5,
    }

    prompt = f"""
        Ти професійний фінансовий трейдер та алгоритмічний аналітик. 
        Проаналізуй технічні дані та графіки для активу {asset_name} на трьох таймфреймах (1h, 15m, 5m).
        Технічні дані індикаторів: {json.dumps(technical_data, ensure_ascii=False)}
        
        Дай відповідь суворо у форматі JSON без зайвих символів з полями:
        - "decision": "YES" або "NO" (чи варто відкривати угоду)
        - "confidence": ціле число від 0 до 10 (рівень впевненості)
        - "reason": коротке обґрунтування українською мовою
        - "expiry": рекомендований час експірації в хвилинах (наприклад, 5, 10, 15 тощо)
        """

    try:
      # Надсилаємо промпт разом із трьома скріншотами графіків
      response = self.model.generate_content(
          [prompt, image_1h, image_15m, image_5m]
      )

      text_response = response.text.strip()

      # Очищуємо відповідь від markdown-обгорток типу ```json ... ```
      if text_response.startswith("```json"):
        text_response = (
            text_response.removeprefix("```json").removesuffix("```").strip()
        )
      elif text_response.startswith("```"):
        text_response = (
            text_response.removeprefix("```").removesuffix("```").strip()
        )

      result = json.loads(text_response)

      return {
          "decision": result.get("decision", "NO"),
          "confidence": int(result.get("confidence", 0)),
          "reason": result.get(
              "reason", "Не вдалося отримати детального обґрунтування від ШІ"
          ),
          "expiry": int(result.get("expiry", 5)),
      }

    except Exception as e:
      print(f"Помилка виклику Gemini API або парсингу відповіді ШІ: {e}")
      return default_response
