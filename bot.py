from datetime import datetime, timedelta
import io
import os
import threading
import time
from ai_advisor import AITradingAdvisor
from database import (
    get_active_signals,
    get_consecutive_losses,
    get_statistics,
    init_db,
    save_signal,
    update_signal_result,
)
from flask import Flask, request
from indicators import AdaptiveTechnicalAnalysis
import pandas as pd
import requests
import yfinance as yf

app = Flask(__name__)

TELEGRAM_TOKEN = "8921212255:AAE_Ypn6wCLUxVMjcrrd8TgPncuLTYQRnSg"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def setup_webhook():
  webhook_url = "https://racio-1bot.onrender.com/webhook"
  url = f"{TELEGRAM_API_URL}/setWebhook?url={webhook_url}"
  try:
    response = requests.get(url, timeout=10)
    print(f"Webhook setup response: {response.json()}")
  except Exception as e:
    print(f"❌ Webhook setup error: {e}")


setup_webhook()

PAIRS = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "AUDUSD=X",
    "USDCAD=X",
    "USDCHF=X",
    "NZDUSD=X",
    "EURGBP=X",
    "EURJPY=X",
    "GBPJPY=X",
    "AUDJPY=X",
    "CADJPY=X",
    "CHFJPY=X",
    "EURAUD=X",
    "EURNZD=X",
    "EURCAD=X",
    "GBPAUD=X",
    "GBPNZD=X",
    "GBPCAD=X",
    "AUDCAD=X",
    "AUDNZD=X",
]

tech_analysis = AdaptiveTechnicalAnalysis()
ai_advisor = AITradingAdvisor()

init_db()


def send_telegram_message(chat_id, text, reply_markup=None):
  url = f"{TELEGRAM_API_URL}/sendMessage"
  payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
  if reply_markup:
    payload["reply_markup"] = reply_markup
  try:
    requests.post(url, json=payload, timeout=10)
  except Exception as e:
    print(f"❌ Telegram send error: {e}")


def check_trading_time():
  now_utc = datetime.utcnow()
  return 7 <= now_utc.hour < 19


def background_expired_checker():
  while True:
    try:
      active_signals = get_active_signals()
      now = datetime.utcnow()
      for sig_id, pair, signal_type, entry_price, exp_str in active_signals:
        exp_time = datetime.strptime(exp_str, "%Y-%m-%d %H:%M:%S")
        if now >= exp_time:
          df = yf.download(pair, period="1d", interval="1m", progress=False)
          if df is not None and not df.empty:
            current_price = float(df["Close"].iloc[-1])
            if isinstance(current_price, pd.Series):
              current_price = float(current_price.item())

            result = "LOSS"
            if signal_type == "CALL" and current_price > entry_price:
              result = "WIN"
            elif signal_type == "PUT" and current_price < entry_price:
              result = "WIN"

            update_signal_result(sig_id, result)
            print(
                f"✅ Угоду по {pair} закрито: {result} (Вхід: {entry_price},"
                f" Вихід: {current_price})"
            )
    except Exception as e:
      print(f"⚠️ Помилка у фоновій перевірці: {e}")
    time.sleep(60)


threading.Thread(target=background_expired_checker, daemon=True).start()


@app.route("/")
def index():
  return "Racio_1bot is running!", 200


@app.route("/webhook", methods=["POST"])
def webhook():
  data = request.get_json()
  if not data:
    return "ok", 200

  if "message" in data:
    msg = data["message"]
    chat_id = msg.get("chat", {}).get("id")
    text = msg.get("text", "")

    if text == "/start":
      markup = {
          "inline_keyboard": [
              [{"text": "📊 Масовий аналіз", "callback_data": "run_scan"}],
              [{"text": "📈 Статистика", "callback_data": "show_stats"}],
          ]
      }
      send_telegram_message(
          chat_id,
          "Вітаю! Я торговий бот Racio_1. Оберіть дію:",
          reply_markup=markup,
      )

  elif "callback_query" in data:
    cq = data["callback_query"]
    chat_id = cq["message"]["chat"]["id"]
    data_val = cq["data"]

    requests.post(
        f"{TELEGRAM_API_URL}/answerCallbackQuery",
        json={"callback_query_id": cq["id"]},
    )

    if data_val == "show_stats":
      stats = get_statistics()
      text = (
          f"📊 *Статистика торгівлі:*\n\n"
          f"• Всього угод: {stats['total']}\n"
          f"• Перемог (WIN): {stats['wins']}\n"
          f"• Поразок (LOSS): {stats['losses']}\n"
          f"• WinRate: {stats['win_rate']}%"
      )
      send_telegram_message(chat_id, text)

    elif data_val == "run_scan":
      if not check_trading_time():
        send_telegram_message(
            chat_id,
            "⏳ Позаторговий час (робота з 07:00 до 19:00 UTC). Сканування"
            " призупинено.",
        )
        return "ok", 200

      send_telegram_message(
          chat_id, "🔄 Запуск масового сканування 21 валютної пари..."
      )

      signals_found = 0
      for pair in PAIRS:
        try:
          losses = get_consecutive_losses(pair)
          if losses >= 3:
            continue

          df_1h = yf.download(pair, period="5d", interval="1h", progress=False)
          df_15m = yf.download(pair, period="5d", interval="15m", progress=False)
          df_5m = yf.download(pair, period="2d", interval="5m", progress=False)

          if (
              df_1h is None
              or df_15m is None
              or df_5m is None
              or df_5m.empty
              or len(df_5m) < 30
          ):
            continue

          g_trend = tech_analysis.get_trend(df_1h, span_val=200)
          df_15m_ind = tech_analysis.calculate_indicators(df_15m)
          m_trend = (
              "UP"
              if df_15m_ind["close"].iloc[-1]
              > df_15m_ind["close"].ewm(span=20).mean().iloc[-1]
              else "DOWN"
          )

          df_5m_ind = tech_analysis.calculate_indicators(df_5m)
          signal_data = tech_analysis.generate_signal(
              df_5m_ind, g_trend, m_trend, pair
          )

          if signal_data["signal"] != "HOLD":
            buf_1h = io.BytesIO()
            buf_15m = io.BytesIO()
            buf_5m = io.BytesIO()
            df_1h["Close"].plot().get_figure().savefig(buf_1h, format="png")
            df_15m["Close"].plot().get_figure().savefig(buf_15m, format="png")
            df_5m["Close"].plot().get_figure().savefig(buf_5m, format="png")

            ai_res = ai_advisor.evaluate_signal(
                pair, signal_data, buf_1h, buf_15m, buf_5m
            )

            if ai_res.get("decision") == "YES" and ai_res.get("confidence", 0) >= 7:
              entry_price = float(df_5m["Close"].iloc[-1])
              exp_mins = int(ai_res.get("expiration", 5))
              exp_time = datetime.utcnow() + timedelta(minutes=exp_mins)

              save_signal(
                  pair,
                  signal_data["signal"],
                  entry_price,
                  exp_time.strftime("%Y-%m-%d %H:%M:%S"),
              )

              msg = (
                  f"🚨 *ТОРГОВИЙ СИГНАЛ* 🚨\n\n"
                  f"• Пара: `{pair}`\n"
                  f"• Сигнал: *{signal_data['signal']}*\n"
                  f"• Ціна входу: `{entry_price}`\n"
                  f"• Експертиза ШІ: {ai_res.get('reason')}\n"
                  f"• Впевненість: `{ai_res.get('confidence')}/10`\n"
                  f"• Експірація: `{exp_mins} хв`"
              )
              send_telegram_message(chat_id, msg)
              signals_found += 1
        except Exception as e:
          print(f"Помилка при скануванні {pair}: {e}")

      send_telegram_message(
          chat_id,
          f"✅ Масове сканування завершено! Знайдено сигналів: {signals_found}",
      )

  return "ok", 200


if __name__ == "__main__":
  app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
