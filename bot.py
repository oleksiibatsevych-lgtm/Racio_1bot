import io
import os
import time
import threading
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta
from flask import Flask, request

app = Flask(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://racio-1bot.onrender.com")

PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", 
    "USDCHF=X", "EURGBP=X", "EURJPY=X", "EURCHF=X", "GBPJPY=X", 
    "AUDJPY=X", "AUDCAD=X", "EURAUD=X", "CADJPY=X", "GBPCHF=X"
]

SCAN_TIMEFRAMES = {
    "5m": "5d",
    "15m": "1mo",
    "1h": "3mo"
}

stats_history = []

# --- Автоматичне встановлення вебхука ---
def setup_webhook():
    if TELEGRAM_TOKEN:
        webhook_url = f"{RENDER_URL}/webhook"
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={webhook_url}"
        try:
            resp = requests.get(url, timeout=10)
            print("Webhook setup response:", resp.json())
        except Exception as e:
            print(f"Помилка встановлення вебхука: {e}")

# Викликаємо одразу при завантаженні модуля на сервері
setup_webhook()

# --- Функція для запобігання засинанню на Render ---
def self_ping():
    while True:
        try:
            time.sleep(600)  # кожні 10 хвилин
            requests.get(RENDER_URL)
        except Exception:
            pass

threading.Thread(target=self_ping, daemon=True).start()


# --- Часовий фільтр (активні торгові сесії) ---
def is_trading_time() -> bool:
    # Дозволяємо сканування з 08:00 до 20:00 UTC (10:00 - 22:00 за Києвом)
    current_hour = datetime.utcnow().hour
    return 8 <= current_hour < 20


def log_stat(pair, signal_type):
    global stats_history
    stats_history.append({
        "timestamp": datetime.now(),
        "pair": pair.replace("=X", ""),
        "signal": signal_type
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
        if sig == "CALL": stats_all[pair]["call"] += 1
        elif sig == "PUT": stats_all[pair]["put"] += 1
        else: stats_all[pair]["hold"] += 1
        
        if t >= week_ago:
            stats_week[pair]["requests"] += 1
            if sig == "CALL": stats_week[pair]["call"] += 1
            elif sig == "PUT": stats_week[pair]["put"] += 1
            else: stats_week[pair]["hold"] += 1
            
        if t >= day_ago:
            stats_day[pair]["requests"] += 1
            if sig == "CALL": stats_day[pair]["call"] += 1
            elif sig == "PUT": stats_day[pair]["put"] += 1
            else: stats_day[pair]["hold"] += 1
            
    return stats_day, stats_week, stats_all

def format_stats_text(title, data):
    if not data:
        return f"📊 *{title}*:\n\nЗа цей період ще немає збережених даних по запитах."
    
    text = f"📊 *{title} (по парах)*:\n\n"
    for pair, counts in data.items():
        text += f"🌟 *{pair}*:\n"
        text += f"  • Перевірок: `{counts['requests']}` | CALL: `{counts['call']}` | PUT: `{counts['put']}` | HOLD: `{counts['hold']}`\n\n"
    return text


# --- Клас технічного аналізу (з доданим ADX) ---
class AdvancedTechnicalAnalysis:
    def __init__(
        self,
        fast_ema: int = 9,
        slow_ema: int = 21,
        trend_ema: int = 50,
        atr_window: int = 14,
        rsi_window: int = 14,
        volume_window: int = 20,
        adx_window: int = 14,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.trend_ema = trend_ema
        self.atr_window = atr_window
        self.rsi_window = rsi_window
        self.volume_window = volume_window
        self.adx_window = adx_window

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        required_columns = ["open", "high", "low", "close", "volume"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame обов'язково повинен містити колонки: {required_columns}")

        res_df = df.copy()

        # EMA
        res_df["EMA_fast"] = res_df["close"].ewm(span=self.fast_ema, adjust=False).mean()
        res_df["EMA_slow"] = res_df["close"].ewm(span=self.slow_ema, adjust=False).mean()
        res_df["EMA_trend"] = res_df["close"].ewm(span=self.trend_ema, adjust=False).mean()

        # RSI
        delta = res_df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0.0)
        com_val = self.rsi_window - 1
        avg_gain = gain.ewm(com=com_val, adjust=False).mean()
        avg_loss = loss.ewm(com=com_val, adjust=False).mean()
        rs = avg_gain / avg_loss
        res_df["RSI"] = 100 - (100 / (1 + rs))

        # ATR & True Range
        high_low = res_df["high"].values - res_df["low"].values
        high_close = np.abs(res_df["high"].values - res_df["close"].shift(1).values)
        low_close = np.abs(res_df["low"].values - res_df["close"].shift(1).values)
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        res_df["ATR"] = pd.Series(true_range, index=res_df.index).rolling(window=self.atr_window).mean()

        # Volume MA
        res_df["Volume_MA"] = res_df["volume"].rolling(window=self.volume_window).mean()
        res_df["Volume_Confirm"] = res_df["volume"] > res_df["Volume_MA"]

        # Wick Ratio
        body_size = np.abs(res_df["close"] - res_df["open"])
        total_candle_size = res_df["high"] - res_df["low"]
        total_candle_size = np.where(total_candle_size == 0, 1e-9, total_candle_size)
        res_df["Wick_Ratio"] = (total_candle_size - body_size) / total_candle_size

        # ADX (Average Directional Index) для фільтрації флету
        up_move = res_df["high"].diff()
        down_move = res_df["low"].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr_series = pd.Series(true_range, index=res_df.index)
        tr14 = tr_series.ewm(alpha=1/self.adx_window, adjust=False).mean()
        
        plus_di = 100 * (pd.Series(plus_dm, index=res_df.index).ewm(alpha=1/self.adx_window, adjust=False).mean() / tr14)
        minus_di = 100 * (pd.Series(minus_dm, index=res_df.index).ewm(alpha=1/self.adx_window, adjust=False).mean() / tr14)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        res_df["ADX"] = dx.ewm(alpha=1/self.adx_window, adjust=False).mean()

        return res_df

    def calculate_dynamic_expiration(self, df: pd.DataFrame, window: int = 200) -> int:
        min_exp = 3
        max_exp = 30
        default_exp = 15

        if len(df) < 20 or "ATR" not in df.columns:
            return default_exp

        last_row = df.iloc[-1]
        if pd.isna(last_row["ATR"]) or pd.isna(last_row["close"]):
            return default_exp

        volatility_pct = (last_row["ATR"] / last_row["close"]) * 100
        recent_vol = ((df["ATR"] / df["close"]) * 100).tail(window)
        high_vol_threshold = recent_vol.quantile(0.75)
        low_vol_threshold = recent_vol.quantile(0.25)

        if volatility_pct > high_vol_threshold:
            recommended_time = 3
        elif volatility_pct < low_vol_threshold:
            recommended_time = 30
        else:
            recommended_time = 15

        return max(min_exp, min(recommended_time, max_exp))

    def generate_signal(self, df: pd.DataFrame) -> dict:
        default_response = {
            "signal": "HOLD",
            "expiration": 15,
            "rsi": None,
            "atr": None,
            "adx": None,
            "reason": "Insufficient data or unconfirmed filters"
        }

        if len(df) < self.trend_ema:
            return default_response

        last = df.iloc[-1]
        prev = df.iloc[-2]

        critical_cols = ["EMA_fast", "EMA_slow", "EMA_trend", "RSI", "ATR", "Volume_Confirm", "Wick_Ratio", "ADX"]
        if any(pd.isna(last[col]) for col in critical_cols):
            return default_response

        # 1. Фільтр тіней (шум)
        if last["Wick_Ratio"] > 0.50:
            default_response["reason"] = "High wick ratio (market noise)"
            return default_response

        # 2. Фільтр сили тренду ADX (якщо ADX < 22 — це флет, сигнали не даємо)
        if last["ADX"] < 22:
            default_response["reason"] = f"Low trend strength (ADX: {round(float(last['ADX']), 1)} < 22)"
            return default_response

        expiration_time = self.calculate_dynamic_expiration(df)

        bullish_cross = (prev["EMA_fast"] <= prev["EMA_slow"]) and (last["EMA_fast"] > last["EMA_slow"])
        bearish_cross = (prev["EMA_fast"] >= prev["EMA_slow"]) and (last["EMA_fast"] < last["EMA_slow"])

        uptrend = last["close"] > last["EMA_trend"]
        downtrend = last["close"] < last["EMA_trend"]
        volume_ok = last["Volume_Confirm"]

        signal = "HOLD"
        reason = "No cross or filters mismatch"

        if bullish_cross and uptrend and volume_ok and (40 < last["RSI"] < 70):
            signal = "CALL"
            reason = "Bullish cross, uptrend, volume & ADX confirmed"
        elif bearish_cross and downtrend and volume_ok and (30 < last["RSI"] < 60):
            signal = "PUT"
            reason = "Bearish cross, downtrend, volume & ADX confirmed"

        return {
            "signal": signal,
            "expiration": expiration_time,
            "rsi": round(float(last["RSI"]), 2),
            "atr": round(float(last["ATR"]), 5),
            "adx": round(float(last["ADX"]), 1),
            "reason": reason
        }


class TelegramSignalSender:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{self.token}"

    def _create_chart(self, df: pd.DataFrame, asset_name: str) -> io.BytesIO:
        plot_df = df.tail(60)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]}
        )

        ax1.plot(plot_df.index, plot_df["close"], label="Close", color="#d1d4dc", alpha=0.6, linewidth=1)
        ax1.plot(plot_df.index, plot_df["EMA_fast"], label="EMA Fast", color="#2962ff", linewidth=1.2)
        ax1.plot(plot_df.index, plot_df["EMA_slow"], label="EMA Slow", color="#ff6d00", linewidth=1.2)
        ax1.plot(plot_df.index, plot_df["EMA_trend"], label="EMA Trend", color="#ab47bc", linewidth=1.5, linestyle="--")

        ax1.set_title(f"Signal: {asset_name}", fontsize=14, color="white", weight="bold")
        ax1.legend(loc="upper left", facecolor="#1e1e1e", labelcolor="white")
        ax1.grid(True, color="#2a2e39", alpha=0.5)

        ax2.plot(plot_df.index, plot_df["RSI"], label="RSI", color="#e91e63", linewidth=1.2)
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
        plt.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        buf.seek(0)
        plt.close(fig)

        return buf

    def send_signal(self, df: pd.DataFrame, signal_data: dict, asset: str):
        if signal_data["signal"] == "HOLD":
            return False

        chart_buffer = self._create_chart(df, asset_name=asset)

        emoji = "🟢" if signal_data["signal"] == "CALL" else "🔴"
        caption = (
            f"{emoji} **СИГНАЛ: {signal_data['signal']}**\n\n"
            f"📊 **Актив:** `{asset}`\n"
            f"⏳ **Експірація:** `{signal_data['expiration']} хв`\n"
            f"📈 **RSI:** `{signal_data['rsi']}` | 📉 **ADX:** `{signal_data['adx']}`\n"
            f"💡 **Причина:** _{signal_data['reason']}_\n"
        )

        url = f"{self.api_url}/sendPhoto"
        files = {"photo": (f"{asset}_signal.png", chart_buffer, "image/png")}
        data = {"chat_id": self.chat_id, "caption": caption, "parse_mode": "Markdown"}

        response = requests.post(url, data=data, files=files)
        return response.json()


analyzer = AdvancedTechnicalAnalysis()

def scan_pair(pair_symbol):
    clean_name = pair_symbol.replace("=X", "")
    for tf, period in SCAN_TIMEFRAMES.items():
        try:
            df = yf.download(pair_symbol, period=period, interval=tf, progress=False)
            if df.empty or len(df) < 50:
                continue
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df.columns = [c.lower() for c in df.columns]

            df_ind = analyzer.calculate_indicators(df)
            signal_res = analyzer.generate_signal(df_ind)
            
            log_stat(pair_symbol, signal_res["signal"])

            if signal_res["signal"] != "HOLD":
                return pair_symbol, df_ind, signal_res
        except Exception as e:
            print(f"Помилка по парі {clean_name} на ТФ {tf}: {e}")
            continue
    
    log_stat(pair_symbol, "HOLD")
    return pair_symbol, None, None


@app.route("/webhook", methods=["POST"])
def telegram_webhook():
    update = request.get_json()
    if not update:
        return "OK", 200
        
    print("Вхідний запит від Telegram:", update)

    main_menu_keyboard = {
        "keyboard": [
            [{"text": "📊 Аналізувати пару"}, {"text": "📈 Статистика"}]
        ],
        "resize_keyboard": True
    }

    if "message" in update:
        chat_id = update["message"]["chat"]["id"]
        text = update["message"].get("text", "")

        if text == "/start":
            reply = (
                "👋 Вітаю! Бот технічного аналізу активний.\n\n"
                "Використовуйте кнопки нижче для масового сканування ринку:"
            )
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={
                "chat_id": chat_id, "text": reply, "reply_markup": main_menu_keyboard, "parse_mode": "Markdown"
            })

        elif text in ["/signal", "📊 Аналізувати пару"]:
            # Перевірка на часовий фільтр сесій
            if not is_trading_time():
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={
                    "chat_id": chat_id, "text": "🌙 Зараз поза межами активних торгових сесій (нічний час / низька ліквідність). Сканування тимчасово обмежено для захисту від хибних сигналів."
                })
                return "OK", 200

            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={
                "chat_id": chat_id, "text": "⏳ Починаю масове сканування всіх валютних пар..."
            })
            
            def run_scan():
                try:
                    notifier = TelegramSignalSender(token=TELEGRAM_TOKEN, chat_id=str(chat_id))
                    signals_found = 0

                    for pair in PAIRS:
                        pair_sym, df_data, sig_res = scan_pair(pair)
                        if sig_res and sig_res["signal"] != "HOLD":
                            notifier.send_signal(df_data, sig_res, asset=pair.replace("=X", ""))
                            signals_found += 1

                    if signals_found == 0:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={
                            "chat_id": chat_id, "text": "📭 За результатами сканування активних трендів не знайдено (ринок у флеті або фільтри відхилили сигнали)."
                        })
                    else:
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={
                            "chat_id": chat_id, "text": f"✅ Сканування завершено. Знайдено та відправлено якісних сигналів: {signals_found}"
                        })
                except Exception as e:
                    print(f"Помилка у фоновому скануванні: {e}")

            threading.Thread(target=run_scan).start()

        elif text in ["/stats", "📈 Статистика"]:
            stats_day, stats_week, stats_all = get_statistics()
            keyboard = {
                "inline_keyboard": [
                    [{"text": "📅 За добу", "callback_data": "stats|day"},
                     {"text": "📆 За тиждень", "callback_data": "stats|week"}],
                    [{"text": "📈 За весь час", "callback_data": "stats|all"}]
                ]
            }
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={
                "chat_id": chat_id, "text": "📊 **Виберіть період для перегляду статистики:**", "reply_markup": keyboard, "parse_mode": "Markdown"
            })

    elif "callback_query" in update:
        query = update["callback_query"]
        chat_id = query["message"]["chat"]["id"]
        message_id = query["message"]["message_id"]
        data = query["data"]

        if data.startswith("stats|"):
            _, period = data.split("|")
            stats_day, stats_week, stats_all = get_statistics()
            if period == "day":
                text = format_stats_text("Статистика за добу", stats_day)
            elif period == "week":
                text = format_stats_text("Статистика за тиждень", stats_week)
            else:
                text = format_stats_text("Статистика за весь час", stats_all)
                
            keyboard = {
                "inline_keyboard": [
                    [{"text": "🔄 Оновити", "callback_data": f"stats|{period}"}],
                    [{"text": "« Назад до вибору періоду", "callback_data": "stats|menu"}]
                ]
            }
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText", json={
                "chat_id": chat_id, "message_id": message_id, "text": text, "reply_markup": keyboard, "parse_mode": "Markdown"
            })

        elif data == "stats|menu":
            keyboard = {
                "inline_keyboard": [
                    [{"text": "📅 За добу", "callback_data": "stats|day"},
                     {"text": "📆 За тиждень", "callback_data": "stats|week"}],
                    [{"text": "📈 За весь час", "callback_data": "stats|all"}]
                ]
            }
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/editMessageText", json={
                "chat_id": chat_id, "message_id": message_id, "text": "📊 **Виберіть період для перегляду статистики:**", "reply_markup": keyboard, "parse_mode": "Markdown"
            })

    return "OK", 200

@app.route("/")
def home():
    return "Advanced TA Bot is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
