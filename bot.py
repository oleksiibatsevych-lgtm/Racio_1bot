import html
import io
import logging
import os
import threading
import time
from datetime import datetime, timedelta
import matplotlib
import pandas as pd
import requests
import yfinance as yf

matplotlib.use("Agg")
from flask import Flask, request
from telegram import (Bot, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup, Update)
from telegram.ext import (CallbackQueryHandler, CommandHandler, Dispatcher, Filters, MessageHandler)

from config import FINNHUB_API_KEY, FINNHUB_TOKEN, PAIRS_MAP, YAHOO_PAIRS_MAP
from charts import create_chart_image
import database
from finnhub_ws import get_live_price, start_finnhub_ws
from indicators import AdaptiveTechnicalAnalysis
from ml_model import TradingMLFilter
from ai_advisor import ai_advisor_instance  # Використовуємо новий клас
from filters import calculate_dynamic_expiration, check_pivot_level_proximity

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
app = Flask(__name__)
bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")
if RENDER_URL:
    try:
        bot.set_webhook(url=f"{RENDER_URL}/webhook", allowed_updates=["message", "callback_query"])
    except Exception as e:
        logger.error(f"Помилка Webhook: {e}")

analyzer = AdaptiveTechnicalAnalysis()
ml_filter = TradingMLFilter()
last_sent_signals = {}

database.init_db()
start_finnhub_ws()

# Функції завантаження свічок (fetch_finnhub_candles, fetch_yahoo_fallback, fetch_all_timeframes)
# залишаються такими ж, як у Bot 1, оскільки вони працюють ідеально для Finnhub + Yahoo.

def fetch_finnhub_candles(symbol, resolution="1", count_candles=500):
    token = FINNHUB_API_KEY or FINNHUB_TOKEN
    if not token: return pd.DataFrame()
    end_time = int(time.time())
    start_time = end_time - (count_candles * (60 if resolution == "1" else 3600 if resolution == "60" else 86400))
    url = "https://finnhub.io/api/v1/forex/candle"
    try:
        resp = requests.get(url, params={"symbol": symbol, "resolution": resolution, "from": start_time, "to": end_time, "token": token}, timeout=6)
        if resp.status_code == 200 and resp.json().get("s") == "ok":
            data = resp.json()
            df = pd.DataFrame({"open": data["o"], "high": data["h"], "low": data["l"], "close": data["c"], "volume": data["v"]}, index=pd.to_datetime(data["t"], unit="s"))
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)
            return df
    except Exception: pass
    return pd.DataFrame()

def fetch_yahoo_fallback(ticker, interval="1m", range_period="5d"):
    try:
        tk = yf.Ticker(ticker)
        df_yf = tk.history(period=range_period, interval=interval, auto_adjust=True)
        if not df_yf.empty:
            df_yf.columns = df_yf.columns.str.lower()
            if df_yf.index.tz is not None: df_yf.index = df_yf.index.tz_localize(None)
            return df_yf
    except Exception: pass
    return pd.DataFrame()

def fetch_all_timeframes(ticker_finnhub, ticker_yahoo=""):
    df_1m = fetch_finnhub_candles(ticker_finnhub, "1", 600)
    if df_1m.empty and ticker_yahoo: df_1m = fetch_yahoo_fallback(ticker_yahoo, "1m", "5d")
    if df_1m.empty: return None, None, None, None, None
    df_5m = df_1m.resample("5min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    df_15m = df_1m.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    df_1h = fetch_finnhub_candles(ticker_finnhub, "60", 200)
    if df_1h.empty and ticker_yahoo: df_1h = fetch_yahoo_fallback(ticker_yahoo, "1h", "30d")
    df_daily = fetch_finnhub_candles(ticker_finnhub, "D", 30)
    live_p = get_live_price(ticker_finnhub)
    if live_p and not df_1m.empty:
        df_1m.iloc[-1, df_1m.columns.get_loc("close")] = live_p
        if not df_5m.empty: df_5m.iloc[-1, df_5m.columns.get_loc("close")] = live_p
    return df_daily, df_1h, df_15m, df_5m, df_1m

def get_current_session_info():
    hour = datetime.utcnow().hour
    sessions, code = [], 1
    if 0 <= hour < 8: sessions.append("Азія"); code = 0
    if 7 <= hour < 16: sessions.append("Лондон"); code = 1
    if 13 <= hour < 21: sessions.append("Нью-Йорк"); code = 2
    if 13 <= hour < 16: sessions.append("🔥 Перетин Лондон/Нью-Йорк"); code = 3
    return ", ".join(sessions) if sessions else "Тихоокеанська", code, hour

def process_single_pair(chat_id, name, ticker, ignore_cooldown=False, is_background=False):
    try:
        session_str, session_code, hour = get_current_session_info()
        
        if not ignore_cooldown and ticker in last_sent_signals and (time.time() - last_sent_signals[ticker]) < 180:
            if not is_background: bot.send_message(chat_id=chat_id, text=f"⏳ Пара {name} на кулдауні.")
            return

        ticker_yahoo = YAHOO_PAIRS_MAP.get(name, "")
        df_daily, df_macro, df_mid, df_fast, df_micro = fetch_all_timeframes(ticker, ticker_yahoo)

        if df_macro is None or df_fast is None or df_macro.empty or df_fast.empty:
            if not is_background: bot.send_message(chat_id=chat_id, text=f"⚠️ Немає даних {name}.")
            return

        df_macro = analyzer.calculate_indicators(df_macro)
        df_mid = analyzer.calculate_indicators(df_mid)
        df_fast = analyzer.calculate_indicators(df_fast)
        
        global_trend = analyzer.get_trend(df_macro, 200)
        mid_trend = analyzer.get_trend(df_mid, 50)
        pivots = analyzer.calculate_pivots(df_daily if (df_daily is not None and not df_daily.empty) else df_macro)

        sig_data = analyzer.generate_signal(global_trend=global_trend, mid_trend=mid_trend, df_daily=df_daily, tf_dict={"1m": df_micro, "5m": df_fast, "15m": df_mid, "1h": df_macro})
        signal_type = sig_data.get("signal", "NONE")
        
        if signal_type == "NONE":
             return

        rsi, adx = float(sig_data.get("rsi", 50)), float(sig_data.get("adx", 20))
        bb_width = float(df_fast["bb_width"].iloc[-2]) if "bb_width" in df_fast.columns else 0.001
        current_price = float(df_fast["close"].iloc[-1])
        volatility_ratio = float(sig_data.get("volatility_ratio", 1.0))
        divergence_str = str(sig_data.get("divergence", "NONE"))

        # Перевірка на близькість до рівнів (Захист)
        is_level_ok, level_reason = check_pivot_level_proximity(current_price, signal_type, pivots, 0.0015)
        if not is_level_ok:
            log_msg = f"⏭ {name} відхилено: {level_reason}"
            database.save_filtered_log(chat_id, log_msg)
            if not is_background: bot.send_message(chat_id=chat_id, text=log_msg)
            return

        dist_pivot = (current_price - pivots["P"]) / pivots["P"] if pivots["P"] > 0 else 0.0

        # ЖОРСТКИЙ ML ФІЛЬТР (як у Bot 2)
        raw_prob = ml_filter.predict_signal_probability(rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot, volatility_ratio, 0.0, 0.0)
        win_probability = raw_prob * 100.0 if raw_prob <= 1.0 else raw_prob

        # Поріг піднято до 65% для максимального вінрейту
        if win_probability < 65.0:
            log_msg = f"❌ {name}: ML відхилив (Ймовірність {win_probability:.1f}%)"
            database.save_filtered_log(chat_id, log_msg)
            if not is_background: bot.send_message(chat_id=chat_id, text=f"{log_msg}\nСигнал відсіяно фільтром ML.")
            return

        calculated_expiration = calculate_dynamic_expiration(sig_data.get("primary_tf", "5m"), adx, volatility_ratio, sig_data.get("strategy_type", "TREND"), rsi, session_str)

        macro_chart = create_chart_image(df_macro, name, tf_label="1h")
        mid_chart = create_chart_image(df_mid, name, tf_label="15m")
        micro_chart = create_chart_image(df_fast, name, tf_label="5m")

        ai_payload = {
            "signal": signal_type,
            "current_price": current_price,
            "adx": adx, "rsi": rsi,
            "global_trend": global_trend, "mid_trend": mid_trend,
            "divergence": divergence_str,
            "atr_ratio": volatility_ratio,
            "reason": sig_data.get("reason"),
            "suggested_exp": calculated_expiration,
        }

        # ЖОРСТКИЙ AI ФІЛЬТР (YES/NO)
        ai_audit = ai_advisor_instance.evaluate_signal(name, ai_payload, macro_chart, mid_chart, micro_chart)
        ai_decision = ai_audit.get("decision", "NO")
        ai_confidence = int(ai_audit.get("confidence", 0))
        ai_reason = str(ai_audit.get("reason", ""))
        final_expiration = int(ai_audit.get("suggested_expiration", calculated_expiration))

        is_ai_busy = any(kw in ai_reason.lower() for kw in ["недоступні", "зайняті", "quota", "exhausted", "limit"])

        if is_ai_busy:
            # Якщо ШІ лежить, але ML > 75%, можна ризикнути пропустити. Інакше - блок.
            if win_probability >= 75.0:
                ai_confidence = 7
                ai_reason = f"Авто-схвалення (Надзвичайно високий ML: {win_probability:.1f}%), ШІ зайнятий."
            else:
                log_msg = f"🤖 {name}: ШІ недоступний, ML недостатній для сліпого входу ({win_probability:.1f}%)."
                database.save_filtered_log(chat_id, log_msg)
                return
        elif ai_decision != "YES" or ai_confidence < 7:
            log_msg = f"🤖 {name}: ШІ ВІДХИЛИВ — {ai_reason} (Впевненість: {ai_confidence}/10)"
            database.save_filtered_log(chat_id, log_msg)
            if not is_background: bot.send_message(chat_id=chat_id, text=f"{log_msg}\nСигнал скасовано ШІ.")
            return

        last_sent_signals[ticker] = time.time()
        icon = "🟢 КУПІВЛЯ (CALL)" if signal_type == "CALL" else "🔴 ПРОДАЖ (PUT)"

        msg_text = (
            f"⚡ <b>СИГНАЛ: {name}</b> | {icon}\n"
            f"🎯 <b>Ціна входу:</b> <code>{current_price:.5f}</code>\n"
            f"⏱ <b>Експірація:</b> {final_expiration} хв.\n"
            f"----------------------------------\n"
            f"📈 <b>Тренди (1h / 15m):</b> {global_trend} / {mid_trend}\n"
            f"📊 <b>RSI:</b> {rsi:.1f} | <b>ADX:</b> {adx:.1f} | <b>ATR:</b> {volatility_ratio:.2f}\n"
            f"🌐 <b>Сесія:</b> {session_str}\n"
            f"----------------------------------\n"
            f"🤖 <b>Аналітика:</b>\n"
            f"• ML-ймовірність: <b>{win_probability:.1f}%</b>\n"
            f"• ШІ-впевненість: <b>{ai_confidence}/10</b>\n"
            f"💡 <b>Обґрунтування:</b> <i>{html.escape(ai_reason)}</i>"
        )

        sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="HTML")

        timestamp_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        sig_id = database.save_signal(
            ticker=ticker, signal_type=signal_type, entry_price=current_price, expiration_mins=final_expiration,
            chat_id=chat_id, message_id=sent_msg.message_id, timestamp_str=timestamp_now, message_text=msg_text,
            rsi=rsi, adx=adx, bb_width=bb_width, session_code=session_code, hour=hour, divergence=divergence_str, dist_pivot=dist_pivot
        )

        # Таймер експірації імпортується або використовується з Bot 1
        # schedule_signal_timer(sig_id, timestamp_now, final_expiration) 
        # (Залишаємо логіку таймера з Bot 1, вона стандартна і працює)

    except Exception as e:
        logger.exception(f"Помилка при обробці пари {name}: {e}")

def analyze_all_pairs_async(chat_id):
    logger.info("Фоновий аналіз...")
    for name, ticker in PAIRS_MAP.items():
        try:
            process_single_pair(chat_id, name, ticker, is_background=True)
            time.sleep(3)
        except Exception: pass
    bot.send_message(chat_id=chat_id, text="✅ <b>Аналіз завершено.</b> Відхилені сигнали можна подивитись в логах.", parse_mode="HTML")

# --- Далі йдуть стандартні хендлери Flask та Telegram (з Bot 1) ---
@app.route("/")
def index(): return "Racio_1bot is active"

@app.route("/webhook", methods=["POST"])
def webhook():
    dispatcher.process_update(Update.de_json(request.get_json(force=True), bot))
    return "ok", 200

def handle_message(update, context):
    text = update.message.text
    chat_id = update.effective_chat.id
    if text == "📊 Аналіз усіх пар":
        update.message.reply_text("🔎 <b>Розпочинаю фоновий аналіз...</b>", parse_mode="HTML")
        threading.Thread(target=analyze_all_pairs_async, args=(chat_id,), daemon=True).start()
    elif text == "📋 Логи фільтру":
        logs = database.get_system_logs(chat_id)
        if not logs:
            update.message.reply_text("📭 Логи порожні або застарілі.")
        else:
            msg = "📋 <b>Останні відхилені сигнали:</b>\n\n" + "\n".join(logs)
            if len(msg) > 4000: msg = msg[:4000] + "..."
            update.message.reply_text(msg, parse_mode="HTML")
    # Додайте інші кнопки зі свого коду (Статистика, Пари)

dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
# dispatcher.add_handler(CommandHandler("start", start)) 

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
