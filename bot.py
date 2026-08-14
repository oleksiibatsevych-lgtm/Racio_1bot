import os
import requests
from flask import Flask, request
from google import genai

app = Flask(__name__)

# Отримуємо токен Telegram-бота з налаштувань середовища Render
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# Ініціалізація нового клієнта Gemini (автоматично використовує GEMINI_API_KEY з Render)
client = genai.Client()


def ask_ai_trading_advisor(prompt_text):
  """Функція для звернення до Gemini через новий SDK google-genai"""
  try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Ти експерт із технічного аналізу та алгоритмічного трейдингу. "
            f"Дай чіткий і професійний аналіз за цим запитом: {prompt_text}"
        ),
    )
    return response.text
  except Exception as e:
    return f"Помилка звернення до AI-радника: {e}"


@app.route("/", methods=["GET"])
def index():
  return "Racio Bot is active and running!", 200


@app.route(f"/{TELEGRAM_BOT_TOKEN}", methods=["POST"])
def telegram_webhook():
  """Головний обробник вхідних повідомлень від Telegram через вебхук"""
  data = request.get_json()

  if data and "message" in data:
    chat_id = data["message"]["chat"]["id"]
    user_text = data["message"].get("text", "")

    if user_text:
      # Отримуємо аналіз або відповідь від Gemini
      ai_reply = ask_ai_trading_advisor(user_text)

      # Відправляємо результат назад користувачу в Telegram
      send_telegram_message(chat_id, ai_reply)

  return {"status": "ok"}, 200


def send_telegram_message(chat_id, text):
  """Допоміжна функція для відправки повідомлень у Telegram"""
  url = f"{TELEGRAM_API_URL}/sendMessage"
  payload = {"chat_id": chat_id, "text": text}
  try:
    requests.post(url, json=payload, timeout=10)
  except Exception as e:
    print(f"Помилка надсилання повідомлення в Telegram: {e}")


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  app.run(host="0.0.0.0", port=port)
