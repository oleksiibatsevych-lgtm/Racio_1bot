from datetime import datetime, timedelta, timezone
import io
import json
import os
import threading
import time
from flask import Flask, request
from google import genai
from google.genai import types
import matplotlib
import numpy as np
import pandas as pd
import requests
import yfinance as yf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
RENDER_URL = os.environ.get(
    "RENDER_EXTERNAL_URL", "https://racio-1bot.onrender.com"
)

client = genai.Client()

PAIRS_MAP = {
    "CHF/JPY": "CHFJPY=X",
    "AUD/CAD": "AUDCAD=X",
    "GBP/AUD": "GBPAUD=X",
    "EUR/USD": "EURUSD=X",
    "EUR/CAD": "EURCAD=X",
    "AUD/USD": "AUDUSD=X",
    "AUD/CHF": "AUDCHF=X",
    "CAD/CHF": "CADCHF=X",
    "EUR/CHF": "EURCHF=X",
    "GBP/CHF": "GBPCHF=X",
    "USD/CAD": "USDCAD=X",
    "GBP/USD": "GBPUSD=X",
    "GBP/JPY": "GBPJPY=X",
    "EUR/AUD": "EURAUD=X",
    "CAD/JPY": "CADJPY=X",
    "USD/CHF": "USDCHF=X",
    "EUR/GBP": "EURGBP=X",
    "USD/JPY": "USDJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/CAD": "GBPCAD=X",
}

stats_history = []
active_signals = {}
signal_counter = 0


def safe_generate_content(contents, config=None):
  models_to_try = ["gemini-2.5-flash"]
  for model_name in models_to_try:
    try:
      response = client.models.generate_content(
          model=model_name, contents=contents, config=config
      )
      return response
    except Exception as e:
      print(f"⚠️ Модель {model_name} недоступна: {e}")
  raise Exception("Усі моделі Gemini наразі недоступні.")


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
      requests.get(RENDER_URL)
    except Exception:
      pass


threading.Thread(target=self_ping, daemon=True).start()


def check_expired_signals():
  while True:
    time.sleep(15)
    now = datetime.now(timezone.utc)
    expired_ids = []

    for sig_id, sig in list(active_signals.items()):
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

          stats_history.append({
              "timestamp": datetime.now(),
              "pair": sig["pair_name"],
              "signal": signal_type,
              "result": result,
          })

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

    for sig_id in expired_ids:
      if sig_id in active_signals:
        del active_signals[sig_id]


threading.Thread(target=check_expired_signals, daemon=True).start()


def is_trading_time() -> bool:
  current_hour = datetime.now(timezone.utc).hour
  return 7 <= current_hour < 19


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


def create_chart_image(
    df: pd.DataFrame, asset_name: str, tf_label="5m"
) -> io.BytesIO:
  plot_df = df.tail(60).copy().reset_index(drop=True)
  fig, ax = plt.subplots(figsize=(10, 5))

  if "local_support" in df.columns and "local_resistance" in df.columns:
    last_sup = df["local_support"].iloc[-1]
    last_res = df["local_resistance"].iloc[-1]
  else:
    last_sup = df["low"].rolling(window=15).min().iloc[-1]
    last_res = df["high"].rolling(window=15).max().iloc[-1]

  for i in range(len(plot_df)):
    op = plot_df["open"].iloc[i]
    hi = plot_df["high"].iloc[i]
    lo = plot_df["low"].iloc[i]
    cl = plot_df["close"].iloc[i]

    color = "#26a69a" if cl >= op else "#ef5350"
    ax.vlines(i, lo, hi, color=color, linewidth=1, alpha=0.9)
    ax.bar(
        i,
        abs(cl - op),
        bottom=min(op, cl),
        color=color,
        width=0.6,
        alpha=0.9,
    )

  if "ema_20" in plot_df.columns:
    ax.plot(
        plot_df.index,
        plot_df["ema_20"],
        color="#2962FF",
        linestyle="-",
        linewidth=1.5,
        alpha=0.7,
        label="EMA 20",
    )

  if "bb_upper" in plot_df.columns and "bb_lower" in plot_df.columns:
    ax.plot(
        plot_df.index,
        plot_df["bb_upper"],
        color="#ab47bc",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="BB Upper",
    )
    ax.plot(
        plot_df.index,
        plot_df["bb_lower"],
        color="#ab47bc",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="BB Lower",
    )

  ax.axhline(y=last_sup, color="#00897b", linestyle="--", alpha=0.8, linewidth=1)
  ax.axhline(y=last_res, color="#c62828", linestyle="--", alpha=0.8, linewidth=1)

  ax.set_title(
      f"Asset: {asset_name} [{tf_label}]",
      fontsize=10,
      color="white",
      weight="bold",
  )
  ax.grid(True, color="#2a2e39", alpha=0.5)
  ax.set_facecolor("#131722")
  ax.tick_params(colors="white")
  for spine in ax.spines.values():
    spine.set_edgecolor("#2a2e39")

  fig.patch.set_facecolor("#131722")
  plt.tight_layout()

  buf = io.BytesIO()
  plt.savefig(
      buf,
      format="png",
      dpi=150,
      facecolor=fig.get_facecolor(),
      edgecolor="none",
  )
  buf.seek(0)
  plt.close(fig)
  return buf


def get_statistics():
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

  day_items = [i for i in stats_history if i["timestamp"] >= day_ago]
  week_items = [i for i in stats_history if i["timestamp"] >= week_ago]
  return (
      process_items(day_items),
      process_items(week_items),
      process_items(stats_history),
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
        f"- Глобальний тренд (EMA 200 1h): {signal_data['global_trend']}\n"
        f"- Середній тренд (EMA 20 15m): {signal_data['mid_trend']}\n"
        f"- Локальний тренд (EMA 20 5m): {signal_data['local_trend']}\n"
        f"- Причина: {signal_data['reason']}\n"
        f"- RSI: {signal_data.get('rsi')}, ATR (волатильність): {signal_data.get('atr')}\n\n"
        "Зображення: 1) Макротренд (1h), 2) Середній таймфрейм (15m), 3) Мікроструктура (5m зі рівнями Swing та Смугами Боллінджера).\n"
        "ПРОАНАЛІЗУЙ КРОК ЗА КРОКОМ (Chain of Thought):\n"
        "1. Чи збігаються тренди на 1h, 15m та 5m?\n"
        "2. Наскільки якісний відскок від рівня та чи допомагають Смуги Боллінджера?\n"
        "3. Чи виправдана поточна волатильність ATR?\n"
        "Визнач оптимальний час експірації у хвилинах (від 3 до 25 хв) та справедливу оцінку впевненості (confidence) від 1 до 10.\n"
        "Відповідай СУВОРО у форматі JSON без жодних додаткових символів чи markdown-тегів:\n"
        "{\n"
        '  "trend_alignment_analysis": "короткий аналіз збігу таймфреймів",\n'
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


class AdaptiveTechnicalAnalysis:

  def __init__(self):
    self.rsi_window = 14

  def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    res_df = df.copy()
    if isinstance(res_df.columns, pd.MultiIndex):
      res_df.columns = res_df.columns.get_level_values(0)
    res_df.columns = [str(c).lower() for c in res_df.columns]

    delta = res_df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0.0)
    avg_gain = gain.ewm(com=self.rsi_window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=self.rsi_window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    res_df["rsi"] = 100 - (100 / (1 + rs))

    res_df["ema_20"] = res_df["close"].ewm(span=20, adjust=False).mean()

    bb_window = 20
    res_df["bb_middle"] = res_df["close"].rolling(window=bb_window).mean()
    bb_std = res_df["close"].rolling(window=bb_window).std()
    res_df["bb_upper"] = res_df["bb_middle"] + (bb_std * 2)
    res_df["bb_lower"] = res_df["bb_middle"] - (bb_std * 2)

    res_df["local_support"] = res_df["low"].rolling(window=15).min()
    res_df["local_resistance"] = res_df["high"].rolling(window=15).max()

    tr1 = res_df["high"] - res_df["low"]
    tr2 = (res_df["high"] - res_df["close"].shift(1)).abs()
    tr3 = (res_df["low"] - res_df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res_df["atr"] = tr.ewm(span=14, adjust=False).mean()

    return res_df

  def get_trend(self, df: pd.DataFrame, span_val=200) -> str:
    if df is None or df.empty:
      return "NEUTRAL"
    g_df = df.copy()
    g_df["ema"] = g_df["close"].ewm(span=span_val, adjust=False).mean()
    last_close = g_df["close"].iloc[-1]
    last_ema = g_df["ema"].iloc[-1]
    return "UP" if last_close > last_ema else "DOWN"

  def generate_signal(
      self,
      df_5m: pd.DataFrame,
      global_trend: str,
      mid_trend: str,
      asset_name: str,
  ) -> dict:
    default = {
        "signal": "HOLD",
        "rsi": None,
        "atr": None,
        "reason": "No setup",
        "global_trend": global_trend,
        "mid_trend": mid_trend,
        "local_trend": "NEUTRAL",
    }
    if len(df_5m) < 25:
      return default

    last = df_5m.iloc[-1]
    c = last["close"]
    l_sup = last["local_support"]
    l_res = last["local_resistance"]
    rsi = last["rsi"]
    atr = last["atr"]
    ema20 = last["ema_20"]

    local_trend = "UP" if c > ema20 else "DOWN"
    dist_sup = abs(c - l_sup) / c
    dist_res = abs(c - l_res) / c

    if global_trend == "UP" and mid_trend == "UP" and local_trend == "UP":
      if dist_sup < 0.008:
        return {
            "signal": "CALL",
            "rsi": round(float(rsi), 2),
            "atr": round(float(atr), 5),
            "reason": (
                "Потрійний тренд ВГОРУ (1h+15m+5m) + відскок від підтримки Swing"
            ),
            "global_trend": global_trend,
            "mid_trend": mid_trend,
            "local_trend": local_trend,
        }

    if global_trend == "DOWN" and mid_trend == "DOWN" and local_trend == "DOWN":
      if dist_res < 0.008:
        return {
            "signal": "PUT",
            "rsi": round(float(rsi), 2),
            "atr": round(float(atr), 5),
            "reason": (
                "Потрійний тренд ВНИЗ (1h+15m+5m) + відскок від опору Swing"
            ),
            "global_trend": global_trend,
            "mid_trend": mid_trend,
            "local_trend": local_trend,
        }

    return default


analyzer = AdaptiveTechnicalAnalysis()
ai_advisor = AITradingAdvisor()


def scan_pair(pair_symbol, asset_name, chat_id=None):
  global signal_counter
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

      ai_eval = ai_advisor.evaluate_signal(
          asset_name, signal_res, macro_buf, mid_buf, micro_buf
      )

      if (
          ai_eval.get("decision") == "YES"
          and ai_eval.get("confidence", 0) >= 7
      ):
        micro_buf.seek(0)
        signal_counter += 1
        sig_id = signal_counter

        dynamic_expiration = int(ai_eval.get("expiration", 10))
        dynamic_expiration = max(3, min(dynamic_expiration, 30))

        sig_text = (
            "🟢 ВВЕРХ (CALL)"
            if signal_res["signal"] == "CALL"
            else "🔴 ВНИЗ (PUT)"
        )
        caption = (
            f"🎯 **Трендовий Сигнал #{sig_id}**\n"
            f"📊 Пара: `{asset_name}`\n"
            f"📈 Напрямок: {sig_text}\n"
            f"🌐 1h: `{signal_res['global_trend']}` | 15m: `{signal_res['mid_trend']}` | 5m: `{signal_res['local_trend']}`\n"
            f"⏳ Динамічна експірація: `{dynamic_expiration} хв`\n"
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
                  "⏳ Сканую ринок (1h + 15m + 5m) з ШІ та Боллінджером..."
              ),
          },
      )

      def run_mass():
        for name, ticker in PAIRS_MAP.items():
          scan_pair(ticker, name, chat_id)
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": "✅ Сканування завершено!"},
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
  return "Multi-TF Trading Bot with Gemini 2.5 Flash is running!"


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
