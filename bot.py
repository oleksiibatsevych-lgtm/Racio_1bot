from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
from flask import Flask, request
import pandas as pd
import requests
import yfinance as yf

from ai_advisor import AITradingAdvisor, safe_generate_content
from charts import create_chart_image
from config import PAIRS_MAP, RENDER_URL, TELEGRAM_TOKEN
from database import (
    get_all_stats_from_db,
    get_consecutive_losses,
    init_db,
    save_stat_to_db,
)
from indicators import AdaptiveTechnicalAnalysis
import threading
import time

app = Flask(__name__)

init_db()

active_signals_lock = threading.Lock()
active_signals = {}
signal_counter = 0

analyzer = AdaptiveTechnicalAnalysis()
ai_advisor = AITradingAdvisor()


def setup_webhook():
  if TELEGRAM_TOKEN:
    webhook_url = f"{RENDER_URL}/webhook"
    set_url = (
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={webhook_url}"
    )
    try:
      resp = requests.get(set_url, timeout=10)
      print("Webhook setup response:", resp.text)
    except Exception as e:
      print(f"Помилка встановлення вебхука: {e}")


setup_webhook()


def self_ping():
  while True:
    try:
      time.sleep(600)
      requests.get(RENDER_URL, timeout=10)
    except Exception:
      pass


threading.Thread(target=self_ping, daemon=True).start()


def check_expired_signals():
  while True:
    time.sleep(15)
    now = datetime.now(timezone.utc)
    expired_ids = []

    with active_signals_lock:
      current_signals = list(active_signals.items())

    for sig_id, sig in current_signals:
      if now >= sig["expiry_time"]:
        expired_ids.append(sig_id)
        try:
          df_current = yf.download(
              sig["pair_symbol"], period="1d", interval="1m", progress=False
          )
          if df_current.empty:
            continue
          if isinstance(df_current.columns, pd.MultiIndex):
            df_current.columns = df_current.columns.get_level_values(0)
          df_current.columns = [str(c).lower() for c in df_current.columns]

          current_price = float(df_current["close"].iloc[-1])
          entry_price = sig["entry_price"]
          signal_type = sig["signal"]

          if signal_type == "CALL":
            result = "WIN" if current_price > entry_price else "LOSS"
          else:
            result = "WIN" if current_price < entry_price else "LOSS"

          save_stat_to_db(sig["pair_name"], signal_type, result)

          icon = "✅ WIN" if result == "WIN" else "❌ LOSS"
          if sig.get("chat_id") and sig.get("message_id"):
            old_caption = sig.get("caption", "")
            new_caption = (
                f"{old_caption}\n\n*Авто-результат:* Закрито системою —"
                f" **{icon}**"
            )
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageCaption",
                json={
                    "chat_id": sig["chat_id"],
                    "message_id": sig["message_id"],
                    "caption": new_caption,
                    "parse_mode": "Markdown",
                },
            )
        except Exception as e:
          print(f"Помилка автоперевірки сигналу {sig_id}: {e}")

    with active_signals_lock:
      for sig_id in expired_ids:
        if sig_id in active_signals:
          del active_signals[sig_id]


threading.Thread(target=check_expired_signals, daemon=True).start()


def is_trading_time() -> bool:
  current_hour = datetime.now(timezone.utc).hour
  return 7 <= current_hour < 19


def get_atr_dynamic_expiration(df: pd.DataFrame, base_minutes: int = 10) -> int:
  try:
    if "atr" not in df.columns or len(df) < 20:
      return base_minutes
    current_atr = float(df["atr"].iloc[-1])
    mean_atr = float(df["atr"].rolling(20).mean().iloc[-1])
    if np.isnan(mean_atr) or mean_atr == 0:
      return base_minutes
    ratio = current_atr / mean_atr
    exp = int(base_minutes / ratio) if ratio > 0 else base_minutes
    return max(3, min(exp, 30))
  except Exception:
    return base_minutes


def is_news_time_with_ai_sentiment(pair_name: str) -> bool:
  try:
    clean_name = pair_name.replace("=X", "").replace("/", "")
    currencies = (
        [clean_name[:3], clean_name[3:6]]
        if len(clean_name) >= 6
        else [clean_name[:3]]
    )

    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=5)
    if response.status_code != 200:
      return False

    events = response.json()
    now_utc = datetime.now(timezone.utc)

    relevant_events = []
    for event in events:
      if event.get("impact") != "High":
        continue
      if event.get("country") in currencies:
        date_str = event.get("date")
        if date_str:
          event_time = datetime.fromisoformat(date_str).astimezone(timezone.utc)
          diff_minutes = (now_utc - event_time).total_seconds() / 60.0
          if -45 <= diff_minutes <= 45:
            relevant_events.append(
                f"- {event.get('title')} ({event.get('country')}) о"
                f" {event_time.strftime('%H:%M')} UTC"
            )

    if not relevant_events:
      return False

    prompt = (
        f"Оціни ризики для торгівлі парою {pair_name} на основі цих"
        " високоважливих новин:\n"
        + "\n".join(relevant_events)
        + "\nЧи несуть ці події критичну непередбачувану волатильність?"
        " Відповідай ТІЛЬКИ одне слово: 'DANGER' або 'SAFE'."
    )
    res = safe_generate_content(
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.1),
    )
    decision = res.text.strip().upper()
    return "DANGER" in decision
  except Exception:
    return False


def get_statistics():
  history = get_all_stats_from_db()
  now = datetime.now()
  day_ago = now - timedelta(days=1)
  week_ago = now - timedelta(days=7)

  def process_items(items):
    pair_data = {}
    total_wins = 0
    total_valid = 0

    for item in items:
      p = item["pair"]
      res = item["result"]
      if p not in pair_data:
        pair_data[p] = {"requests": 0, "wins": 0, "losses": 0}
      pair_data[p]["requests"] += 1
      if res == "WIN":
        pair_data[p]["wins"] += 1
        total_wins += 1
        total_valid += 1
      elif res == "LOSS":
        pair_data[p]["losses"] += 1
        total_valid += 1

    result = {}
    for p, d in pair_data.items():
      valid = d["wins"] + d["losses"]
      wr = (d["wins"] / valid * 100) if valid > 0 else 0.0
      result[p] = {
          "requests": d["requests"],
          "wins": d["wins"],
          "losses": d["losses"],
          "winrate": round(wr, 1),
      }

    overall_wr = (total_wins / total_valid * 100) if total_valid > 0 else 0.0
    return result, round(overall_wr, 1)

  day_items = [i for i in history if i["timestamp"] >= day_ago]
  week_items = [i for i in history if i["timestamp"] >= week_ago]
  return (
      process_items(day_items),
      process_items(week_items),
      process_items(history),
  )


def format_stats_text(title, data_tuple):
  data, overall_wr = data_tuple
  text = f"📊 *{title}*\n"
  if not data:
    text += "\nЩе немає оцінених угод за цей період."
    return text
  text += f"🏆 **Загальний вінрейт:** `{overall_wr}%`\n\n"
  text += "📋 *По парах*:\n"
  for pair, counts in data.items():
    text += f"🌟 *{pair}* Всього: `{counts['requests']}` | ✅ `{counts['wins']}` ❌ `{counts['losses']}` | Вінрейт: `{counts['winrate']}%`\n"
  return text


def scan_pair(pair_symbol, asset_name, chat_id=None):
  global signal_counter

  with active_signals_lock:
    for s in active_signals.values():
      if s["pair_name"] == asset_name:
        return

  if get_consecutive_losses(asset_name) >= 3:
    print(
        f"⚠️ Пара {asset_name} пропущена через серію збитків (3+ LOSS поспіль)."
    )
    return

  if is_news_time_with_ai_sentiment(asset_name):
    return

  try:
    df_global_raw = yf.download(
        pair_symbol, period="3mo", interval="1h", progress=False
    )
    df_global = (
        analyzer.calculate_indicators(df_global_raw)
        if not df_global_raw.empty
        else None
    )

    df_mid_raw = yf.download(
        pair_symbol, period="14d", interval="15m", progress=False
    )
    df_mid = (
        analyzer.calculate_indicators(df_mid_raw)
        if not df_mid_raw.empty
        else None
    )

    df_local_raw = yf.download(
        pair_symbol, period="5d", interval="5m", progress=False
    )
    if df_local_raw.empty or len(df_local_raw) < 30:
      return
    df_5m = analyzer.calculate_indicators(df_local_raw)

    global_trend = analyzer.get_trend(df_global, 200)
    mid_trend = analyzer.get_trend(df_mid, 50)
    signal_res = analyzer.generate_signal(
        df_5m, global_trend, mid_trend, asset_name
    )

    if signal_res["signal"] != "HOLD" and chat_id:
      macro_buf = (
          create_chart_image(df_global, asset_name, "1h (EMA 200)")
          if df_global is not None
          else create_chart_image(df_5m, asset_name, "1h")
      )
      mid_buf = (
          create_chart_image(df_mid, asset_name, "15m (EMA 50)")
          if df_mid is not None
          else create_chart_image(df_5m, asset_name, "15m")
      )
      micro_buf = create_chart_image(
          df_5m, asset_name, "5m (Swing + Bollinger)"
      )

      atr_exp = get_atr_dynamic_expiration(df_5m, base_minutes=10)
      ai_eval = ai_advisor.evaluate_signal(
          asset_name, signal_res, macro_buf, mid_buf, micro_buf
      )

      if (
          ai_eval.get("decision") == "YES"
          and ai_eval.get("confidence", 0) >= 7
      ):
        micro_buf.seek(0)
        with active_signals_lock:
          signal_counter += 1
          sig_id = signal_counter

        ai_exp = int(ai_eval.get("expiration", 10))
        dynamic_expiration = int((atr_exp + ai_exp) / 2)
        dynamic_expiration = max(3, min(dynamic_expiration, 30))

        adx_val = signal_res.get("adx", 25)
        market_regime = "Флет 📊" if adx_val < 22 else "Тренд 🚀"

        sig_text = (
            "🟢 ВВЕРХ (CALL)"
            if signal_res["signal"] == "CALL"
            else "🔴 ВНИЗ (PUT)"
        )
        caption = (
            f"🎯 **Сигнал #{sig_id} [{market_regime}]**\n"
            f"📊 Пара: `{asset_name}`\n"
            f"📈 Напрямок: {sig_text}\n"
            f"📉 Сила ринку (ADX): `{adx_val}`\n"
            f"🌐 1h: `{signal_res['global_trend']}` | 15m:"
            f" `{signal_res['mid_trend']}` | 5m: `{signal_res['local_trend']}`\n"
            f"⏳ Динамічна експірація (ATR+ШІ): `{dynamic_expiration} хв`\n"
            f"💡 Причина: _{signal_res['reason']}_\n"
            f"🤖 ШІ Конфіденційність: `{ai_eval.get('confidence')}/10`"
        )

        micro_buf.seek(0)
        files = {"photo": (f"{asset_name}.png", micro_buf, "image/png")}
        data = {
            "chat_id": chat_id,
            "caption": caption,
            "parse_mode": "Markdown",
        }
        resp = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
            data=data,
            files=files,
        )

        if resp.status_code == 200:
          resp_json = resp.json()
          sent_message_id = resp_json["result"]["message_id"]

          with active_signals_lock:
            active_signals[sig_id] = {
                "pair_symbol": pair_symbol,
                "pair_name": asset_name,
                "signal": signal_res["signal"],
                "entry_price": float(df_5m["close"].iloc[-1]),
                "expiry_time": datetime.now(timezone.utc)
                + timedelta(minutes=dynamic_expiration),
                "chat_id": chat_id,
                "message_id": sent_message_id,
                "caption": caption,
            }
  except Exception as e:
    print(f"Помилка сканування {pair_symbol}: {e}")


def get_bottom_menu():
  return {
      "keyboard": [
          [{"text": "💱 Пари"}],
          [{"text": "📊 Аналіз усіх пар"}, {"text": "📈 Статистика"}],
      ],
      "resize_keyboard": True,
  }


def get_pairs_inline_keyboard():
  keys = list(PAIRS_MAP.keys())
  inline = [
      [
          {"text": keys[i], "callback_data": f"pair|{keys[i]}"},
          {"text": keys[i + 1], "callback_data": f"pair|{keys[i + 1]}"},
      ]
      if i + 1 < len(keys)
      else [{"text": keys[i], "callback_data": f"pair|{keys[i]}"}]
      for i in range(0, len(keys), 2)
  ]
  return {"inline_keyboard": inline}


@app.route("/webhook", methods=["POST"])
def telegram_webhook():
  update = request.get_json()
  if not update:
    return "OK", 200

  if "message" in update:
    chat_id = update["message"]["chat"]["id"]
    text = update["message"].get("text", "")

    if text in ["/start", "💱 Пари"]:
      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
          json={
              "chat_id": chat_id,
              "text": "Оберіть пару:",
              "reply_markup": get_pairs_inline_keyboard(),
          },
      )
      if text == "/start":
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "Меню:",
                "reply_markup": get_bottom_menu(),
            },
        )

    elif text == "📊 Аналіз усіх пар":
      if not is_trading_time():
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "🌙 Поза межами торгового часу (07:00 - 19:00 UTC).",
            },
        )
        return "OK", 200

      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
          json={
              "chat_id": chat_id,
              "text": (
                  "⏳ Паралельне сканування ринку (ThreadPoolExecutor) з ШІ..."
              ),
          },
      )

      def run_mass():
        with ThreadPoolExecutor(max_workers=5) as executor:
          futures = [
              executor.submit(scan_pair, ticker, name, chat_id)
              for name, ticker in PAIRS_MAP.items()
          ]
          for f in futures:
            try:
              f.result()
            except Exception as e:
              print(f"Помилка в потоці сканування: {e}")

        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": "✅ Масове сканування завершено!"},
        )

      threading.Thread(target=run_mass).start()

    elif text == "📈 Статистика":
      keyboard = {
          "inline_keyboard": [
              [
                  {"text": "📅 За добу", "callback_data": "stats|day"},
                  {"text": "📆 За тиждень", "callback_data": "stats|week"},
              ],
              [{"text": "📈 За весь час", "callback_data": "stats|all"}],
          ]
      }
      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
          json={
              "chat_id": chat_id,
              "text": "📊 **Виберіть період:**",
              "reply_markup": keyboard,
              "parse_mode": "Markdown",
          },
      )

  elif "callback_query" in update:
    query = update["callback_query"]
    chat_id = query["message"]["chat"]["id"]
    message_id = query["message"]["message_id"]
    data = query["data"]

    if data.startswith("pair|"):
      _, pair_name = data.split("|")
      if not is_trading_time():
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "🌙 Поза межами торгового часу (07:00 - 19:00 UTC).",
            },
        )
        return "OK", 200

      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
          json={
              "chat_id": chat_id,
              "text": (
                  f"⏳ Аналізую {pair_name} по трьох таймфреймах (1h, 15m, 5m)..."
              ),
          },
      )
      threading.Thread(
          target=lambda: scan_pair(PAIRS_MAP[pair_name], pair_name, chat_id)
      ).start()

    elif data.startswith("stats|"):
      _, period = data.split("|")
      (stats_day, wr_day), (stats_week, wr_week), (stats_all, wr_all) = (
          get_statistics()
      )

      if period == "day":
        sel_stats, sel_wr = stats_day, wr_day
      elif period == "week":
        sel_stats, sel_wr = stats_week, wr_week
      else:
        sel_stats, sel_wr = stats_all, wr_all

      text_res = format_stats_text(
          f"Статистика ({period})", (sel_stats, sel_wr)
      )
      keyboard = {
          "inline_keyboard": [[{
              "text": "🔄 Оновити",
              "callback_data": f"stats|{period}",
          }]]
      }
      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
          json={
              "chat_id": chat_id,
              "message_id": message_id,
              "text": text_res,
              "reply_markup": keyboard,
              "parse_mode": "Markdown",
          },
      )

  return "OK", 200


@app.route("/")
def home():
  return (
      "Modular Racio_1 Bot (bot.py) with SQLite, ThreadPool & Risk Management"
      " is running!"
  )


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
