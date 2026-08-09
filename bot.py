from datetime import datetime, timedelta, timezone
import io
import os
import threading
import time
from flask import Flask, request
import matplotlib
import numpy as np
import pandas as pd
import requests
import yfinance as yf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# --- КОНФІГУРАЦІЯ ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
RENDER_URL = os.environ.get(
    "RENDER_EXTERNAL_URL", "https://racio-1bot.onrender.com"
)

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


# --- ПЕРЕВІРКА ВЕБХУКА ---
def setup_webhook():
  if TELEGRAM_TOKEN:
    webhook_url = f"{RENDER_URL}/webhook"
    set_url = (
        f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={webhook_url}"
    )
    try:
      resp = requests.get(set_url, timeout=10)
      print("Webhook setup response:", resp.text)
      info_url = (
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getWebhookInfo"
      )
      info_resp = requests.get(info_url, timeout=10)
      print("Webhook info status:", info_resp.text)
    except Exception as e:
      print(f"Помилка встановлення вебхука: {e}")


setup_webhook()


# --- ЗАХОДИ ПРОТИ ЗАСИНАННЯ НА RENDER ---
def self_ping():
  while True:
    try:
      time.sleep(600)
      requests.get(RENDER_URL)
    except Exception:
      pass


threading.Thread(target=self_ping, daemon=True).start()


# --- ТОРГОВИЙ ЧАС (10:00 - 22:00 за Києвом / 07:00 - 19:00 UTC) ---
def is_trading_time() -> bool:
  current_hour = datetime.now(timezone.utc).hour
  return 7 <= current_hour < 19


def log_stat(pair, signal_type):
  global stats_history
  stats_history.append({
      "timestamp": datetime.now(),
      "pair": pair.replace("=X", ""),
      "signal": signal_type,
  })


def get_statistics():
  now = datetime.now()
  day_ago = now - timedelta(days=1)
  week_ago = now - timedelta(days=7)

  stats_day = {}
  stats_week = {}
  stats_all = {}

  for item in stats_history:
    pair = item["pair"]
    t = item["timestamp"]
    sig = item["signal"]

    for s_dict in [stats_day, stats_week, stats_all]:
      if pair not in s_dict:
        s_dict[pair] = {"requests": 0, "call": 0, "put": 0, "hold": 0}

    stats_all[pair]["requests"] += 1
    if sig == "CALL":
      stats_all[pair]["call"] += 1
    elif sig == "PUT":
      stats_all[pair]["put"] += 1
    else:
      stats_all[pair]["hold"] += 1

    if t >= week_ago:
      stats_week[pair]["requests"] += 1
      if sig == "CALL":
        stats_week[pair]["call"] += 1
      elif sig == "PUT":
        stats_week[pair]["put"] += 1
      else:
        stats_week[pair]["hold"] += 1

    if t >= day_ago:
      stats_day[pair]["requests"] += 1
      if sig == "CALL":
        stats_day[pair]["call"] += 1
      elif sig == "PUT":
        stats_day[pair]["put"] += 1
      else:
        stats_day[pair]["hold"] += 1

  return stats_day, stats_week, stats_all


def format_stats_text(title, data):
  if not data:
    return f"📊 *{title}*:\n\nЗа цей період ще немає збережених даних."
  text = f"📊 *{title} (по парах)*:\n\n"
  for pair, counts in data.items():
    text += f"🌟 *{pair}*:\n"
    text += (
        f"  • Перевірок: `{counts['requests']}` | CALL: `{counts['call']}` | PUT:"
        f" `{counts['put']}` | HOLD: `{counts['hold']}`\n\n"
    )
  return text


# --- КЛАС ТЕХНІЧНОГО АНАЛІЗУ ---
class AdvancedTechnicalAnalysis:

  def __init__(
      self,
      rsi_window: int = 14,
      atr_window: int = 14,
      stoch_window: int = 14,
  ):
    self.rsi_window = rsi_window
    self.atr_window = atr_window
    self.stoch_window = stoch_window

  def calculate_indicators(
      self, df_local: pd.DataFrame, df_global: pd.DataFrame = None
  ) -> pd.DataFrame:
    required_columns = ["open", "high", "low", "close", "volume"]
    if not all(col in df_local.columns for col in required_columns):
      raise ValueError("DataFrame містить не всі необхідні колонки")
    res_df = df_local.copy()

    res_df["EMA_trend"] = res_df["close"].ewm(span=50, adjust=False).mean()

    # RSI
    delta = res_df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0.0)
    avg_gain = gain.ewm(com=self.rsi_window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=self.rsi_window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    res_df["RSI"] = 100 - (100 / (1 + rs))

    # Stochastic Oscillator
    low_min = res_df["low"].rolling(window=self.stoch_window).min()
    high_max = res_df["high"].rolling(window=self.stoch_window).max()
    res_df["Stoch_K"] = (
        (res_df["close"] - low_min) / (high_max - low_min + 1e-9)
    ) * 100
    res_df["Stoch_D"] = res_df["Stoch_K"].rolling(window=3).mean()

    # Локальні рівні (на 5m)
    res_df["Local_Support"] = res_df["low"].rolling(window=20).min()
    res_df["Local_Resistance"] = res_df["high"].rolling(window=20).max()

    # Глобальні рівні (з 1h таймфрейму)
    if df_global is not None and not df_global.empty:
      g_sup = df_global["low"].rolling(window=24).min()
      g_res = df_global["high"].rolling(window=24).max()
      res_df["Global_Support"] = (
          g_sup.reindex(res_df.index, method="ffill")
          .bfill()
          .fillna(res_df["low"].min())
      )
      res_df["Global_Resistance"] = (
          g_res.reindex(res_df.index, method="ffill")
          .bfill()
          .fillna(res_df["high"].max())
      )
    else:
      res_df["Global_Support"] = res_df["Local_Support"]
      res_df["Global_Resistance"] = res_df["Local_Resistance"]

    # ATR
    high_low = res_df["high"].values - res_df["low"].values
    high_close = np.abs(
        res_df["high"].values - res_df["close"].shift(1).values
    )
    low_close = np.abs(res_df["low"].values - res_df["close"].shift(1).values)
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    res_df["ATR"] = (
        pd.Series(true_range, index=res_df.index)
        .rolling(window=self.atr_window)
        .mean()
    )

    return res_df

  def calculate_dynamic_expiration(self, df: pd.DataFrame) -> int:
    if len(df) < 20 or "ATR" not in df.columns:
      return 7
    last_row = df.iloc[-1]
    if pd.isna(last_row["ATR"]) or pd.isna(last_row["close"]):
      return 7

    volatility_pct = (last_row["ATR"] / last_row["close"]) * 100
    recent_vol = ((df["ATR"] / df["close"]) * 100).tail(200)
    if recent_vol.empty or recent_vol.max() == recent_vol.min():
      return 7

    min_v = recent_vol.min()
    max_v = recent_vol.max()
    norm = ((volatility_pct - min_v) / (max_v - min_v + 1e-9)).clip(0, 1)
    # Точний час від 3 до 15 хвилин залежно від волатильності
    exp = round(15 - norm * 12)
    return int(max(3, min(15, exp)))

  def check_pre_alert(self, df: pd.DataFrame) -> dict:
    if len(df) < 25:
      return {"status": False}
    last = df.iloc[-1]
    close = last["close"]
    g_support = last["Global_Support"]
    g_resistance = last["Global_Resistance"]
    rsi = last["RSI"]
    stoch_k = last["Stoch_K"]

    dist_to_sup = abs(close - g_support) / close
    dist_to_res = abs(close - g_resistance) / close

    if dist_to_sup < 0.004 and (30 <= rsi <= 36 or stoch_k <= 25):
      return {
          "status": True,
          "type": "CALL_PREPARE",
          "reason": (
              f"Ціна підходить до глобальної підтримки ({g_support:.5f}), RSI:"
              f" {rsi:.1f}"
          ),
      }

    if dist_to_res < 0.004 and (64 <= rsi <= 70 or stoch_k >= 75):
      return {
          "status": True,
          "type": "PUT_PREPARE",
          "reason": (
              f"Ціна підходить до глобального опору ({g_resistance:.5f}), RSI:"
              f" {rsi:.1f}"
          ),
      }

    return {"status": False}

  def generate_signal(self, df: pd.DataFrame) -> dict:
    default_response = {
        "signal": "HOLD",
        "expiration": 7,
        "rsi": None,
        "stoch": None,
        "reason": "No setup",
    }
    if len(df) < 25:
      return default_response

    last = df.iloc[-1]
    close = last["close"]
    g_support = last["Global_Support"]
    g_resistance = last["Global_Resistance"]
    l_support = last["Local_Support"]
    l_resistance = last["Local_Resistance"]
    rsi = last["RSI"]
    stoch_k = last["Stoch_K"]
    stoch_d = last["Stoch_D"]

    expiration_time = self.calculate_dynamic_expiration(df)

    near_g_support = abs(close - g_support) / close < 0.0025
    near_l_support = abs(close - l_support) / close < 0.0015

    if (near_g_support or near_l_support) and (
        rsi < 32 or (stoch_k < 20 and stoch_k > stoch_d)
    ):
      lvl_type = "глобального" if near_g_support else "локального"
      return {
          "signal": "CALL",
          "expiration": expiration_time,
          "rsi": round(float(rsi), 2),
          "stoch": round(float(stoch_k), 2),
          "reason": (
              f"Відскок від {lvl_type} рівня підтримки з підтвердженням"
              " осцилятора"
          ),
      }

    near_g_resistance = abs(close - g_resistance) / close < 0.0025
    near_l_resistance = abs(close - l_resistance) / close < 0.0015

    if (near_g_resistance or near_l_resistance) and (
        rsi > 68 or (stoch_k > 80 and stoch_k < stoch_d)
    ):
      lvl_type = "глобального" if near_g_resistance else "локального"
      return {
          "signal": "PUT",
          "expiration": expiration_time,
          "rsi": round(float(rsi), 2),
          "stoch": round(float(stoch_k), 2),
          "reason": (
              f"Відскік від {lvl_type} рівня опору з підтвердженням осцилятора"
          ),
      }

    return default_response


# --- ВІДПРАВКА ПОВІДОМЛЕНЬ У TELEGRAM ---
class TelegramSignalSender:

  def __init__(self, token: str, chat_id: str):
    self.token = token
    self.chat_id = chat_id
    self.api_url = f"https://api.telegram.org/bot{self.token}"

  def _create_chart(self, df: pd.DataFrame, asset_name: str) -> io.BytesIO:
    plot_df = df.tail(60)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot(
        plot_df.index,
        plot_df["close"],
        label="Close",
        color="#d1d4dc",
        alpha=0.6,
        linewidth=1,
    )
    ax1.plot(
        plot_df.index,
        plot_df["Global_Support"],
        label="Global Support",
        color="#00897b",
        linestyle="-.",
        linewidth=1.2,
    )
    ax1.plot(
        plot_df.index,
        plot_df["Global_Resistance"],
        label="Global Resistance",
        color="#c62828",
        linestyle="-.",
        linewidth=1.2,
    )
    ax1.plot(
        plot_df.index,
        plot_df["Local_Support"],
        label="Local Support",
        color="#26a69a",
        linestyle="--",
        linewidth=1,
    )
    ax1.plot(
        plot_df.index,
        plot_df["Local_Resistance"],
        label="Local Resistance",
        color="#ef5350",
        linestyle="--",
        linewidth=1,
    )
    ax1.set_title(
        f"Signal: {asset_name}", fontsize=14, color="white", weight="bold"
    )
    ax1.legend(loc="upper left", facecolor="#1e1e1e", labelcolor="white")
    ax1.grid(True, color="#2a2e39", alpha=0.5)

    ax2.plot(
        plot_df.index,
        plot_df["RSI"],
        label="RSI",
        color="#e91e63",
        linewidth=1.2,
    )
    ax2.axhline(70, color="red", linestyle=":", alpha=0.7)
    ax2.axhline(30, color="green", linestyle=":", alpha=0.7)
    ax2.set_ylabel("RSI", color="white")
    ax2.legend(loc="upper left", facecolor="#1e1e1e", labelcolor="white")
    ax2.grid(True, color="#2a2e39", alpha=0.5)

    for ax in [ax1, ax2]:
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

  def send_pre_alert(self, pre_data: dict, asset: str):
    emoji = "⚠️"
    caption = (
        f"{emoji} **ПОПЕРЕДЖЕННЯ ПРО СЕТАП (PRE-ALERT)**\n\n"
        f"📊 **Актив:** `{asset}`\n"
        f"🎯 **Очікування:** `{pre_data['type']}`\n"
        f"💡 **Причина:** _{pre_data['reason']}_\n"
        f"⏳ _Готуйтеся до можливої угоди!_"
    )
    url = f"{self.api_url}/sendMessage"
    payload = {
        "chat_id": self.chat_id,
        "text": caption,
        "parse_mode": "Markdown",
    }
    requests.post(url, json=payload)

  def send_signal(self, df: pd.DataFrame, signal_data: dict, asset: str):
    if signal_data["signal"] == "HOLD":
      return False
    chart_buffer = self._create_chart(df, asset_name=asset)
    emoji = "🟢" if signal_data["signal"] == "CALL" else "🔴"
    caption = (
        f"{emoji} **СІГНАЛ: {signal_data['signal']}**\n\n"
        f"📊 **Актив:** `{asset}`\n"
        f"⏳ **Експірація:** `{signal_data['expiration']} хв`\n"
        f"📈 **RSI:** `{signal_data['rsi']}` | 📉 **Stoch:** `{signal_data['stoch']}`\n"
        f"💡 **Причина:** _{signal_data['reason']}_\n"
    )
    url = f"{self.api_url}/sendPhoto"
    files = {"photo": (f"{asset}_signal.png", chart_buffer, "image/png")}
    data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "Markdown"}
    response = requests.post(url, data=data, files=files)
    return response.json()


analyzer = AdvancedTechnicalAnalysis()


def scan_pair(pair_symbol, asset_name, chat_id=None):
  notifier = (
      TelegramSignalSender(token=TELEGRAM_TOKEN, chat_id=str(chat_id))
      if chat_id
      else None
  )

  try:
    df_global = yf.download(
        pair_symbol, period="1mo", interval="1h", progress=False
    )
    if not df_global.empty:
      if isinstance(df_global.columns, pd.MultiIndex):
        df_global.columns = df_global.columns.get_level_values(0)
      df_global.columns = [c.lower() for c in df_global.columns]
    else:
      df_global = None

    df_local = yf.download(
        pair_symbol, period="5d", interval="5m", progress=False
    )
    if df_local.empty or len(df_local) < 30:
      return pair_symbol, None, None
    if isinstance(df_local.columns, pd.MultiIndex):
      df_local.columns = df_local.columns.get_level_values(0)
    df_local.columns = [c.lower() for c in df_local.columns]

    df_ind = analyzer.calculate_indicators(df_local, df_global)

    pre_res = analyzer.check_pre_alert(df_ind)
    if pre_res["status"] and notifier:
      notifier.send_pre_alert(pre_res, asset=asset_name)

    signal_res = analyzer.generate_signal(df_ind)
    log_stat(pair_symbol, signal_res["signal"])

    if signal_res["signal"] != "HOLD":
      return pair_symbol, df_ind, signal_res

  except Exception as e:
    print(f"Помилка сканування {pair_symbol}: {e}")

  return pair_symbol, None, None


def get_pairs_inline_keyboard():
  keys = list(PAIRS_MAP.keys())
  inline_keyboard = []
  for i in range(0, len(keys), 2):
    row = [{"text": keys[i], "callback_data": f"pair|{keys[i]}"}]
    if i + 1 < len(keys):
      row.append({"text": keys[i + 1], "callback_data": f"pair|{keys[i+1]}"})
    inline_keyboard.append(row)
  inline_keyboard.append([
      {"text": "📊 Аналіз усіх пар", "callback_data": "action|scan_all"},
      {"text": "📈 Статистика", "callback_data": "action|stats_menu"},
  ])
  return {"inline_keyboard": inline_keyboard}


# --- ОБРОБНИК ВЕБХУКА ---
@app.route("/webhook", methods=["POST"])
def telegram_webhook():
  update = request.get_json()
  if not update:
    return "OK", 200

  if "message" in update:
    chat_id = update["message"]["chat"]["id"]
    text = update["message"].get("text", "")

    if text == "/start":
      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
          json={
              "chat_id": chat_id,
              "text": (
                  "Оберіть пару для аналізу (Глобальні/Локальні рівні + RSI +"
                  " Stochastic):"
              ),
              "reply_markup": get_pairs_inline_keyboard(),
          },
      )

  elif "callback_query" in update:
    query = update["callback_query"]
    chat_id = query["message"]["chat"]["id"]
    message_id = query["message"]["message_id"]
    data = query["data"]

    if data.startswith("pair|"):
      _, pair_name = data.split("|")
      ticker = PAIRS_MAP[pair_name]
      if not is_trading_time():
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": (
                    "🌙 Зараз поза межами торгового часу (робочі години: 10:00 -"
                    " 22:00 за Києвом)."
                ),
            },
        )
        return "OK", 200

      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
          json={"chat_id": chat_id, "text": f"⏳ Аналізую {pair_name}..."},
      )

      def run_single():
        try:
          _, df_data, sig_res = scan_pair(
              ticker, asset_name=pair_name, chat_id=chat_id
          )
          notifier = TelegramSignalSender(
              token=TELEGRAM_TOKEN, chat_id=str(chat_id)
          )
          if sig_res and sig_res["signal"] != "HOLD":
            notifier.send_signal(df_data, sig_res, asset=pair_name)
          else:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": (
                        f"📭 По парі {pair_name} чітких сигналів не знайдено біля"
                        " рівнів."
                    ),
                },
            )
        except Exception as e:
          print(f"Помилка поодинокого сканування: {e}")

      threading.Thread(target=run_single).start()

    elif data == "action|scan_all":
      if not is_trading_time():
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": "🌙 Зараз поза межами торгового часу.",
            },
        )
        return "OK", 200

      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
          json={
              "chat_id": chat_id,
              "text": "⏳ Починаю масове сканування за рівнями...",
          },
      )

      def run_mass():
        try:
          found = 0
          for name, ticker in PAIRS_MAP.items():
            _, df_data, sig_res = scan_pair(
                ticker, asset_name=name, chat_id=chat_id
            )
            notifier = TelegramSignalSender(
                token=TELEGRAM_TOKEN, chat_id=str(chat_id)
            )
            if sig_res and sig_res["signal"] != "HOLD":
              notifier.send_signal(df_data, sig_res, asset=name)
              found += 1
          requests.post(
              f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
              json={
                  "chat_id": chat_id,
                  "text": (
                      f"✅ Масове сканування завершено. Знайдено сигналів: {found}"
                  ),
              },
          )
        except Exception as e:
          print(f"Помилка масового сканування: {e}")

      threading.Thread(target=run_mass).start()

    elif data == "action|stats_menu":
      stats_day, stats_week, stats_all = get_statistics()
      keyboard = {
          "inline_keyboard": [
              [
                  {"text": "📅 За добу", "callback_data": "stats|day"},
                  {"text": "📆 За тиждень", "callback_data": "stats|week"},
              ],
              [{"text": "📈 За весь час", "callback_data": "stats|all"}],
              [
                  {
                      "text": "« Назад до вибору пар",
                      "callback_data": "action|pairs_menu",
                  }
              ],
          ]
      }
      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
          json={
              "chat_id": chat_id,
              "message_id": message_id,
              "text": "📊 **Виберіть період для перегляду статистики:**",
              "reply_markup": keyboard,
              "parse_mode": "Markdown",
          },
      )

    elif data == "action|pairs_menu":
      keyboard = get_pairs_inline_keyboard()
      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText",
          json={
              "chat_id": chat_id,
              "message_id": message_id,
              "text": (
                  "Оберіть пару для аналізу (Глобальні/Локальні рівні + RSI +"
                  " Stochastic):"
              ),
              "reply_markup": keyboard,
          },
      )

    elif data.startswith("stats|"):
      _, period = data.split("|")
      stats_day, stats_week, stats_all = get_statistics()
      if period == "day":
        text_res = format_stats_text("Статистика за добу", stats_day)
      elif period == "week":
        text_res = format_stats_text("Статистика за тиждень", stats_week)
      else:
        text_res = format_stats_text("Статистика за весь час", stats_all)

      keyboard = {
          "inline_keyboard": [
              [
                  {"text": "🔄 Оновити", "callback_data": f"stats|{period}"},
                  {
                      "text": "« Назад до меню",
                      "callback_data": "action|stats_menu",
                  },
              ]
          ]
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
  return "Inline Pairs Bot with Levels & Pre-alerts is running!"


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
