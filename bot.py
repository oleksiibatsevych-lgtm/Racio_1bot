from datetime import datetime, timedelta, timezone
import io
import os
import threading
import time
from flask import Flask, request
from google import genai
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

# Ініціалізація нового клієнта Gemini (автоматично підхоплює GEMINI_API_KEY з оточення)
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

# Зберігання даних у пам'яті
stats_history = []  # Перевірені результати сигналів
active_signals = {}  # Активні сигнали для зв'язку з кнопками
signal_counter = 0  # Лічильник для унікальних ID сигналів


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


# --- ТОРГОВИЙ ЧАС ТА НОВИНИ ---
def is_trading_time() -> bool:
  current_hour = datetime.now(timezone.utc).hour
  return 7 <= current_hour < 19


def is_news_time(pair_name: str) -> bool:
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

    for event in events:
      if event.get("impact") != "High":
        continue
      if event.get("country") in currencies:
        date_str = event.get("date")
        if date_str:
          event_time = datetime.fromisoformat(date_str).astimezone(timezone.utc)
          diff_minutes = (now_utc - event_time).total_seconds() / 60.0
          if -30 <= diff_minutes <= 30:
            return True
  except Exception:
    pass
  return False


# --- СТАТИСТИКА РЕАЛЬНИХ РЕЗУЛЬТАТІВ ---
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
  all_items = stats_history

  return (
      process_items(day_items),
      process_items(week_items),
      process_items(all_items),
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
    text += f"🌟 *{pair}*:\n"
    text += (
        f"  • Всього угод: `{counts['requests']}`\n"
        f"  • ✅ WIN: `{counts['wins']}` | ❌ LOSS: `{counts['losses']}`\n"
        f"  • Вінрейт: `{counts['winrate']}%`\n\n"
    )
  return text


# --- ШІ-ВІДПОВІДАЛЬНИЙ ЗА ВАЛІДАЦІЮ СИГНАЛІВ (НОВИЙ SDK) ---
class AITradingAdvisor:

  def __init__(self):
    pass

  def evaluate_signal(
      self, pair_name: str, signal_data: dict, recent_candles_str: str
  ) -> bool:
    """Аналізує сигнал через Gemini перед відправкою користувачу (через новий SDK)"""
    prompt = (
        f"Ти експертний AI-алгоритм для торгівлі бінарними опціонами та Forex. "
        f"Оціни доцільність відкриття угоди для пари {pair_name}:\n"
        f"- Сигнал: {signal_data['signal']}\n"
        f"- Причина/Патерн: {signal_data['reason']}\n"
        f"- Експірація: {signal_data['expiration']} хв\n"
        f"- RSI: {signal_data.get('rsi')}, Stoch: {signal_data.get('stoch')}\n"
        f"- Контекст останніх 5 свічок (Open, High, Low, Close):\n{recent_candles_str}\n\n"
        f"Чи варто відкривати цю угоду? Відповідай ТІЛЬКИ одним словом: 'YES' (якщо ризик виправданий) або 'NO' (якщо ринок шумно-небезпечний)."
    )
    try:
      response = client.models.generate_content(
          model="gemini-2.5-flash",
          contents=prompt,
      )
      decision = response.text.strip().upper()
      print(f"AI Advisor decision for {pair_name}: {decision}")
      return "YES" in decision
    except Exception as e:
      print(f"Помилка ШІ-валідації: {e}")
      return True


# --- КЛАС ТЕХНІЧНОГО ТА ПАТЕРНОВОГО АНАЛІЗУ + BOLLINGER BANDS ---
class AdvancedTechnicalAnalysis:

  def __init__(self):
    self.rsi_window = 14
    self.atr_window = 14
    self.stoch_window = 14
    self.adx_window = 14
    self.bb_window = 20

  def calculate_indicators(
      self, df_local: pd.DataFrame, df_global: pd.DataFrame = None
  ) -> pd.DataFrame:
    res_df = df_local.copy()

    delta = res_df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0.0)
    avg_gain = gain.ewm(com=self.rsi_window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=self.rsi_window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    res_df["RSI"] = 100 - (100 / (1 + rs))

    low_min = res_df["low"].rolling(window=self.stoch_window).min()
    high_max = res_df["high"].rolling(window=self.stoch_window).max()
    res_df["Stoch_K"] = (
        (res_df["close"] - low_min) / (high_max - low_min + 1e-9)
    ) * 100
    res_df["Stoch_D"] = res_df["Stoch_K"].rolling(window=3).mean()

    # Bollinger Bands
    res_df["BB_Middle"] = (
        res_df["close"].rolling(window=self.bb_window).mean()
    )
    bb_std = res_df["close"].rolling(window=self.bb_window).std()
    res_df["BB_Upper"] = res_df["BB_Middle"] + (bb_std * 2)
    res_df["BB_Lower"] = res_df["BB_Middle"] - (bb_std * 2)

    alpha = 1 / self.adx_window
    tr1 = res_df["high"] - res_df["low"]
    tr2 = (res_df["high"] - res_df["close"].shift(1)).abs()
    tr3 = (res_df["low"] - res_df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = res_df["high"] - res_df["high"].shift(1)
    down_move = res_df["low"].shift(1) - res_df["low"]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = (
        pd.Series(plus_dm, index=res_df.index)
        .ewm(alpha=alpha, adjust=False)
        .mean()
    )
    minus_dm_smooth = (
        pd.Series(minus_dm, index=res_df.index)
        .ewm(alpha=alpha, adjust=False)
        .mean()
    )

    plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-9))
    minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    res_df["ADX"] = dx.ewm(alpha=alpha, adjust=False).mean()

    res_df["Local_Support"] = res_df["low"].rolling(window=20).min()
    res_df["Local_Resistance"] = res_df["high"].rolling(window=20).max()

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

    high_low = res_df["high"].values - res_df["low"].values
    high_close = np.abs(res_df["high"].values - res_df["close"].shift(1).values)
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
    last = df.iloc[-1]
    if pd.isna(last["ATR"]) or pd.isna(last["close"]):
      return 7
    volatility_pct = (last["ATR"] / last["close"]) * 100
    recent_vol = ((df["ATR"] / df["close"]) * 100).tail(200)
    if recent_vol.empty or recent_vol.max() == recent_vol.min():
      return 7
    norm = (
        (volatility_pct - recent_vol.min())
        / (recent_vol.max() - recent_vol.min() + 1e-9)
    ).clip(0, 1)
    exp = round(15 - norm * 12)
    return int(max(3, min(15, exp)))

  def check_chart_patterns(self, df: pd.DataFrame) -> dict:
    if len(df) < 50:
      return {"pattern": None, "type": None}

    recent = df.tail(50).copy().reset_index(drop=True)
    recent["is_trough"] = recent["low"] == recent["low"].rolling(
        5, center=True
    ).min()
    recent["is_peak"] = recent["high"] == recent["high"].rolling(
        5, center=True
    ).max()

    troughs = recent[recent["is_trough"]]
    peaks = recent[recent["is_peak"]]
    current_price = recent["close"].iloc[-1]

    if len(troughs) >= 2:
      t1, t2 = troughs.iloc[-2], troughs.iloc[-1]
      if (
          abs(t1["low"] - t2["low"]) / t2["low"] < 0.0015
          and abs(t2.name - t1.name) >= 5
          and current_price > t2["low"]
      ):
        return {
            "pattern": "Double Bottom",
            "type": "CALL",
            "reason": f"Патерн 'Подвійне дно' ({t2['low']:.4f})",
        }

    if len(peaks) >= 2:
      p1, p2 = peaks.iloc[-2], peaks.iloc[-1]
      if (
          abs(p1["high"] - p2["high"]) / p2["high"] < 0.0015
          and abs(p2.name - p1.name) >= 5
          and current_price < p2["high"]
      ):
        return {
            "pattern": "Double Top",
            "type": "PUT",
            "reason": f"Патерн 'Подвійна вершина' ({p2['high']:.4f})",
        }

    return {"pattern": None, "type": None}

  def check_multi_tf_patterns(
      self, df_1m: pd.DataFrame, df_3m: pd.DataFrame, df_5m: pd.DataFrame
  ) -> dict:
    for tf_name, df_tf in [("1m", df_1m), ("3m", df_3m), ("5m", df_5m)]:
      if df_tf is not None and not df_tf.empty:
        res = self.check_chart_patterns(df_tf)
        if res["pattern"] is not None:
          return {
              "pattern": res["pattern"],
              "type": res["type"],
              "reason": f"[{tf_name}] {res['reason']}",
          }
    return {"pattern": None, "type": None}

  def generate_signal(
      self,
      df_5m: pd.DataFrame,
      df_3m: pd.DataFrame = None,
      df_1m: pd.DataFrame = None,
  ) -> dict:
    default = {
        "signal": "HOLD",
        "expiration": 7,
        "rsi": None,
        "stoch": None,
        "reason": "No setup",
    }
    if len(df_5m) < 25:
      return default
    last = df_5m.iloc[-1]
    if not pd.isna(last["ADX"]) and last["ADX"] > 25:
      return default

    exp = self.calculate_dynamic_expiration(df_5m)
    rsi, stoch_k, stoch_d = last["RSI"], last["Stoch_K"], last["Stoch_D"]

    pattern_res = self.check_multi_tf_patterns(df_1m, df_3m, df_5m)
    if pattern_res["pattern"] is not None:
      return {
          "signal": pattern_res["type"],
          "expiration": exp,
          "rsi": round(float(rsi), 2) if not pd.isna(rsi) else 50.0,
          "stoch": round(float(stoch_k), 2) if not pd.isna(stoch_k) else 50.0,
          "reason": pattern_res["reason"],
      }

    c, g_sup, g_res = last["close"], last["Global_Support"], last["Global_Resistance"]

    near_g_sup = abs(c - g_sup) / c < 0.0025 or c <= last["BB_Lower"]
    if near_g_sup and (
        rsi < 32 or (stoch_k < 20 and stoch_k > stoch_d) or c <= last["BB_Lower"]
    ):
      return {
          "signal": "CALL",
          "expiration": exp,
          "rsi": round(float(rsi), 2),
          "stoch": round(float(stoch_k), 2),
          "reason": f"Відскок від рівня / Bollinger Lower (ADX: {last['ADX']:.1f})",
      }

    near_g_res = abs(c - g_res) / c < 0.0025 or c >= last["BB_Upper"]
    if near_g_res and (
        rsi > 68 or (stoch_k > 80 and stoch_k < stoch_d) or c >= last["BB_Upper"]
    ):
      return {
          "signal": "PUT",
          "expiration": exp,
          "rsi": round(float(rsi), 2),
          "stoch": round(float(stoch_k), 2),
          "reason": f"Відскік від рівня / Bollinger Upper (ADX: {last['ADX']:.1f})",
      }

    return default


# --- ВІДПРАВКА ПОВІДОМЛЕНЬ ТА ГРАФІКІВ ---
class TelegramSignalSender:

  def __init__(self, token: str, chat_id: str):
    self.api_url = f"https://api.telegram.org/bot{token}"
    self.chat_id = chat_id

  def _create_chart(self, df: pd.DataFrame, asset_name: str) -> io.BytesIO:
    plot_df = df.tail(60).copy().reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    last_sup = df["Global_Support"].iloc[-1]
    last_res = df["Global_Resistance"].iloc[-1]

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

    # Додавання Смуг Боллінджера та рівнів на графік
    ax.plot(
        plot_df.index,
        plot_df["BB_Upper"],
        color="#2962FF",
        linestyle="--",
        alpha=0.5,
        label="BB Upper",
    )
    ax.plot(
        plot_df.index,
        plot_df["BB_Middle"],
        color="#FF6D00",
        linestyle="-",
        alpha=0.5,
        label="BB Middle",
    )
    ax.plot(
        plot_df.index,
        plot_df["BB_Lower"],
        color="#2962FF",
        linestyle="--",
        alpha=0.5,
        label="BB Lower",
    )

    ax.axhline(
        y=last_sup,
        color="#00897b",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label=f"Support: {last_sup:.4f}",
    )
    ax.axhline(
        y=last_res,
        color="#c62828",
        linestyle="--",
        alpha=0.8,
        linewidth=1.5,
        label=f"Resistance: {last_res:.4f}",
    )

    ax.set_title(
        f"Signal: {asset_name} | Bollinger & S/R",
        fontsize=11,
        color="white",
        weight="bold",
    )
    ax.legend(loc="upper left", facecolor="#1e1e1e", labelcolor="white", fontsize=8)
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

  def send_signal(
      self, df: pd.DataFrame, signal_data: dict, asset: str, sig_id: int
  ):
    chart_buffer = self._create_chart(df, asset)
    sig_text = (
        "🟢 ВВЕРХ (CALL)" if signal_data["signal"] == "CALL" else "🔴 ВНИЗ (PUT)"
    )

    last_row = df.iloc[-1]
    sup_price = last_row["Global_Support"]
    res_price = last_row["Global_Resistance"]

    caption = (
        f"🚨 **СІГНАЛ: {sig_text} (AI Approved)**\n"
        f"📊 `{asset}`\n"
        f"⏳ Експірація: `{signal_data['expiration']} хв`\n"
        f"🟢 Підтримка: `{sup_price:.4f}`\n"
        f"🔴 Опір: `{res_price:.4f}`\n"
        f"📈 RSI: `{signal_data['rsi']}` | 📉 Stoch: `{signal_data['stoch']}`\n"
        f"💡 _{signal_data['reason']}_"
    )

    reply_markup = {
        "inline_keyboard": [[
            {"text": "✅ WIN", "callback_data": f"res|WIN|{sig_id}"},
            {"text": "❌ LOSS", "callback_data": f"res|LOSS|{sig_id}"},
        ]]
    }

    files = {"photo": (f"{asset}.png", chart_buffer, "image/png")}
    data = {
        "chat_id": self.chat_id,
        "caption": caption,
        "parse_mode": "Markdown",
        "reply_markup": str(reply_markup).replace("'", '"'),
    }
    requests.post(f"{self.api_url}/sendPhoto", data=data, files=files)


analyzer = AdvancedTechnicalAnalysis()
ai_advisor = AITradingAdvisor()


def scan_pair(pair_symbol, asset_name, chat_id=None):
  global signal_counter
  if is_news_time(asset_name):
    return

  notifier = (
      TelegramSignalSender(TELEGRAM_TOKEN, str(chat_id)) if chat_id else None
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
      return
    if isinstance(df_local.columns, pd.MultiIndex):
      df_local.columns = df_local.columns.get_level_values(0)
    df_local.columns = [c.lower() for c in df_local.columns]

    df_5m = analyzer.calculate_indicators(df_local, df_global)

    df_1m = yf.download(pair_symbol, period="2d", interval="1m", progress=False)
    df_3m = None
    if not df_1m.empty:
      if isinstance(df_1m.columns, pd.MultiIndex):
        df_1m.columns = df_1m.columns.get_level_values(0)
      df_1m.columns = [c.lower() for c in df_1m.columns]
      if len(df_1m) >= 50:
        df_3m = (
            df_1m.resample("3min")
            .agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            })
            .dropna()
        )

    # Викликаємо генерацію сигналу напряму без пре-алертів
    signal_res = analyzer.generate_signal(df_5m, df_3m, df_1m)

    if signal_res["signal"] != "HOLD" and notifier:
      recent_candles = (
          df_5m.tail(5)[["open", "high", "low", "close"]].to_string()
      )
      is_approved = ai_advisor.evaluate_signal(
          asset_name, signal_res, recent_candles
      )

      if is_approved:
        signal_counter += 1
        sig_id = signal_counter

        active_signals[sig_id] = {
            "pair": asset_name,
            "signal": signal_res["signal"],
        }
        notifier.send_signal(df_5m, signal_res, asset_name, sig_id)
      else:
        print(
            f"🤖 ШІ-фільтр відхилив сигнал по {asset_name} через високий ризик."
        )

  except Exception as e:
    print(f"Помилка сканування {pair_symbol}: {e}")


# --- МЕНЮ ТА WEBHOOK ---
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
                "text": "🌙 Зараз поза межами торгового часу (07:00 - 19:00 UTC).",
            },
        )
        return "OK", 200
      requests.post(
          f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
          json={
              "chat_id": chat_id,
              "text": "⏳ Починаю масове сканування з ШІ-фільтрацією...",
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
              "text": f"⏳ Аналізую {pair_name} з ШІ та Боллінджером...",
          },
      )
      threading.Thread(
          target=lambda: scan_pair(PAIRS_MAP[pair_name], pair_name, chat_id)
      ).start()

    elif data.startswith("res|"):
      _, result, sig_id_str = data.split("|")
      sig_id = int(sig_id_str)

      if sig_id in active_signals:
        sig_info = active_signals.pop(sig_id)
        stats_history.append({
            "timestamp": datetime.now(),
            "pair": sig_info["pair"],
            "signal": sig_info["signal"],
            "result": result,
        })

        old_caption = query["message"].get("caption", "")
        icon = "✅ WIN" if result == "WIN" else "❌ LOSS"
        new_caption = (
            f"{old_caption}\n\n*Статус угоди:* Карточка закрита — **{icon}**"
        )

        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageCaption",
            json={
                "chat_id": chat_id,
                "message_id": message_id,
                "caption": new_caption,
                "parse_mode": "Markdown",
            },
        )
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            json={
                "callback_query_id": query["id"],
                "text": f"Зараховано: {icon}!",
            },
        )
      else:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            json={
                "callback_query_id": query["id"],
                "text": "Цей сигнал вже оброблено або застарів.",
            },
        )

    elif data.startswith("stats|"):
      _, period = data.split("|")
      (stats_day, wr_day), (stats_week, wr_week), (stats_all, wr_all) = (
          get_statistics()
      )

      if period == "day":
        text_res = format_stats_text(
            "Статистика за добу", (stats_day, wr_day)
        )
      elif period == "week":
        text_res = format_stats_text(
            "Статистика за тиждень", (stats_week, wr_week)
        )
      else:
        text_res = format_stats_text(
            "Статистика за весь час", (stats_all, wr_all)
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
  return "Bot with Bollinger Bands and New Gemini SDK is running!"


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)
