import os
import io
import time
import threading
import sqlite3
import requests
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CallbackQueryHandler, CommandHandler, MessageHandler, Filters

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None

from config import TELEGRAM_TOKEN, PAIRS_MAP, FINNHUB_TOKEN, FINNHUB_API_KEY
from indicators import AdaptiveTechnicalAnalysis
import database
from ml_model import TradingMLFilter
from ai_advisor import AITradingAdvisor
from charts import create_chart_image
from finnhub_ws import start_finnhub_ws, get_live_price

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

analyzer = AdaptiveTechnicalAnalysis()
ml_filter = TradingMLFilter()
ai_advisor = AITradingAdvisor()

last_sent_signals = {}

database.init_db()

def init_logs_db():
    try:
        conn = sqlite3.connect("filtered_logs.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS filtered_logs (
                chat_id INTEGER,
                log_text TEXT,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception(f"Помилка ініціалізації БД логів: {e}")

init_logs_db()

def clear_filtered_logs(chat_id):
    try:
        conn = sqlite3.connect("filtered_logs.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM filtered_logs WHERE chat_id = ?", (chat_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception(f"Помилка очищення логів: {e}")

def save_filtered_log(chat_id, log_text):
    try:
        conn = sqlite3.connect("filtered_logs.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO filtered_logs (chat_id, log_text, timestamp) VALUES (?, ?, ?)", 
                       (chat_id, log_text, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception(f"Помилка збереження логу: {e}")

def get_filtered_logs(chat_id):
    try:
        conn = sqlite3.connect("filtered_logs.db", check_same_thread=False)
        cursor = conn.cursor()
        cutoff = time.time() - 3600
        cursor.execute("SELECT log_text FROM filtered_logs WHERE chat_id = ? AND timestamp > ? ORDER BY timestamp DESC", (chat_id, cutoff))
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        logger.exception(f"Помилка читання логів: {e}")
        return []

# --- FINNHUB REST API & RESAMPLING ---

def fetch_finnhub_candles(symbol, resolution="1", count_candles=500):
    """Отримання свічок безпосередньо з Finnhub REST API."""
    token = FINNHUB_API_KEY or FINNHUB_TOKEN
    if not token:
        return pd.DataFrame()

    end_time = int(time.time())
    # Розрахунок часового вікна
    if resolution == "1":
        start_time = end_time - (count_candles * 60)
    elif resolution == "60":
        start_time = end_time - (count_candles * 3600)
    elif resolution == "D":
        start_time = end_time - (count_candles * 86400)
    else:
        start_time = end_time - (count_candles * 300)

    url = "https://finnhub.io/api/v1/forex/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": start_time,
        "to": end_time,
        "token": token
    }

    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("s") == "ok":
                df = pd.DataFrame({
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                    "volume": data["v"]
                }, index=pd.to_datetime(data["t"], unit="s"))
                df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                return df
    except Exception as e:
        logger.warning(f"Finnhub REST API error for {symbol}: {e}")

    return pd.DataFrame()

def fetch_yahoo_fallback(ticker, interval="1m", range_period="7d"):
    """Фолбек на yfinance, якщо Finnhub недоступний."""
    try:
        df_yf = yf.download(tickers=ticker, period=range_period, interval=interval, progress=False, auto_adjust=True)
        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            df_yf = df_yf.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df_yf = df_yf[["open", "high", "low", "close", "volume"]].copy()
            df_yf.dropna(subset=["open", "high", "low", "close"], inplace=True)
            if df_yf.index.tz is not None:
                df_yf.index = df_yf.index.tz_localize(None)
            return df_yf
    except Exception as e:
        logger.warning(f"Yahoo fallback failed for {ticker}: {e}")
    return pd.DataFrame()

def resample_candles(df_1m, rule):
    """Ресемплінг 1m свічок у 5m, 15m без додаткових мережевих запитів."""
    if df_1m.empty:
        return pd.DataFrame()
    resampled = df_1m.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    return resampled

def fetch_all_timeframes(ticker_finnhub, ticker_yahoo=""):
    """
    Завантажує 1m, 1h та D свічки з Finnhub і генерує 5m/15m через Resampling.
    Заощаджує лиміти запитів API.
    """
    # 1. Завантажуємо 1m свічки (достатньо 600 свічок для 1m, 5m, 15m)
    df_1m = fetch_finnhub_candles(ticker_finnhub, resolution="1", count_candles=600)
    
    # Фолбек на Yahoo, якщо Finnhub порожній
    if df_1m.empty and ticker_yahoo:
        df_1m = fetch_yahoo_fallback(ticker_yahoo, interval="1m", range_period="5d")

    if df_1m.empty:
        return None, None, None, None, None

    # Ресемплінг 1m -> 5m та 15m
    df_5m = resample_candles(df_1m, "5min")
    df_15m = resample_candles(df_1m, "15min")

    # 2. Старші таймфрейми (1h та Daily)
    df_1h = fetch_finnhub_candles(ticker_finnhub, resolution="60", count_candles=200)
    if df_1h.empty and ticker_yahoo:
        df_1h = fetch_yahoo_fallback(ticker_yahoo, interval="1h", range_period="30d")

    df_daily = fetch_finnhub_candles(ticker_finnhub, resolution="D", count_candles=30)
    if df_daily.empty and ticker_yahoo:
        df_daily = fetch_yahoo_fallback(ticker_yahoo, interval="1d", range_period="60d")

    # Накладаємо останні живі дані з WebSocket на свічку
    live_p = get_live_price(ticker_finnhub)
    if live_p and not df_1m.empty:
        df_1m.iloc[-1, df_1m.columns.get_loc('close')] = live_p
        if not df_5m.empty:
            df_5m.iloc[-1, df_5m.columns.get_loc('close')] = live_p
        if not df_15m.empty:
            df_15m.iloc[-1, df_15m.columns.get_loc('close')] = live_p

    return df_daily, df_1h, df_15m, df_5m, df_1m

def get_current_session_info():
    now_utc = datetime.utcnow()
    hour = now_utc.hour
    sessions = []
    session_code = 1
    if 0 <= hour < 8:
        sessions.append("Азія")
        session_code = 0
    if 7 <= hour < 16:
        sessions.append("Лондон")
        session_code = 1
    if 13 <= hour < 21:
        sessions.append("Нью-Йорк")
        session_code = 2
    if 13 <= hour < 16:
        sessions.append("🔥 Перетин Лондон/Нью-Йорк")
        session_code = 3
    
    session_str = ", ".join(sessions) if sessions else "Тихоокеанська сесія"
    return session_str, session_code, hour

def process_signal_expiration(sig_id):
    """Обробка підсумку угоди за живими даними з Finnhub WebSocket."""
    try:
        if hasattr(database, "get_signal_by_id") and callable(getattr(database, "get_signal_by_id", None)):
            sig_data = database.get_signal_by_id(sig_id)
            if sig_data:
                ticker = sig_data['ticker']
                entry_price = float(sig_data['entry_price'])
                signal_type = sig_data['signal_type']

                exit_price = get_live_price(ticker)
                if not exit_price:
                    df_check = fetch_finnhub_candles(ticker, resolution="1", count_candles=5)
                    if not df_check.empty:
                        exit_price = float(df_check['close'].iloc[-1])

                if exit_price:
                    multiplier = 1000 if "JPY" in ticker else 100000
                    pips = (exit_price - entry_price) * multiplier if signal_type == "CALL" else (entry_price - exit_price) * multiplier

                    if signal_type == "CALL":
                        result = "WIN" if exit_price > entry_price else ("LOSS" if exit_price < entry_price else "NEUTRAL")
                    else:
                        result = "WIN" if exit_price < entry_price else ("LOSS" if exit_price > entry_price else "NEUTRAL")

                    if hasattr(database, "update_signal_result"):
                        database.update_signal_result(sig_id, result, exit_price, pips)

                    res_icon = "✅ WIN" if result == "WIN" else ("❌ LOSS" if result == "LOSS" else "➖ NEUTRAL")
                    pips_str = f"+{pips:.1f}" if pips > 0 else f"{pips:.1f}"

                    report_str = (
                        f"\n----------------------------------\n"
                        f"🏁 **Результат:** {res_icon} (`{pips_str}` п.)\n"
                        f"📍 Вхід: `{entry_price:.5f}` ➔ Вихід: `{exit_price:.5f}`"
                    )

                    orig_txt = sig_data.get("message_text", "")
                    if orig_txt and "🏁 Результат" not in orig_txt:
                        bot.edit_message_text(
                            chat_id=sig_data["chat_id"], 
                            message_id=sig_data["message_id"], 
                            text=f"{orig_txt}\n{report_str}",
                            parse_mode="Markdown"
                        )
                    return

        # Фолбек перевірка
        res_data = database.evaluate_single_signal(sig_id, fetch_yahoo_data_func=None)
        if res_data and res_data.get("chat_id") and res_data.get("message_id"):
            pips_val = float(res_data.get('pips', 0))
            pips_str = f"+{pips_val:.1f}" if pips_val > 0 else f"{pips_val:.1f}"
            res_result = res_data.get('result', 'NEUTRAL')
            
            res_icon = f"🏁 Результат: WIN ✅ ({pips_str} п.)" if res_result == 'WIN' else (
                f"🏁 Результат: LOSS ❌ ({pips_str} п.)" if res_result == 'LOSS' else f"🏁 Результат: NEUTRAL ➖ ({pips_str} п.)"
            )

            orig_txt = res_data.get("message_text", "")
            if orig_txt and "🏁 Результат" not in orig_txt:
                bot.edit_message_text(
                    chat_id=res_data["chat_id"], 
                    message_id=res_data["message_id"], 
                    text=f"{orig_txt}\n{res_icon}",
                    parse_mode="Markdown"
                )
    except Exception as e:
        logger.exception(f"Помилка таймера експірації {sig_id}: {e}")

def schedule_signal_timer(sig_id, timestamp_val, expiration_mins):
    try:
        if isinstance(timestamp_val, datetime):
            signal_time = timestamp_val
        else:
            signal_time = datetime.strptime(str(timestamp_val), "%Y-%m-%d %H:%M:%S")
            
        expiry_time = signal_time + timedelta(minutes=expiration_mins)
        delay = max((expiry_time - datetime.utcnow()).total_seconds(), 1)
        timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
        timer.daemon = True
        timer.start()
    except Exception as e:
        logger.exception(f"Помилка планування таймера {sig_id}: {e}")

def restore_pending_timers():
    pending = database.get_pending_signals()
    for i, row in enumerate(pending):
        sig_id, _, _, _, expiration_mins, timestamp_str, _, _, _ = row
        try:
            if isinstance(timestamp_str, datetime):
                signal_time = timestamp_str
            else:
                signal_time = datetime.strptime(str(timestamp_str), "%Y-%m-%d %H:%M:%S")
                
            expiry_time = signal_time + timedelta(minutes=expiration_mins)
            delay = max((expiry_time - datetime.utcnow()).total_seconds(), 2 + (i * 2))
            timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
            timer.daemon = True
            timer.start()
        except Exception as e:
            logger.warning(f"⚠️ Помилка відновлення таймера {sig_id}: {e}")
    logger.info(f"⏳ Відновлено активних таймерів: {len(pending)}")

restore_pending_timers()

@app.route("/")
def index():
    return "Racio_1bot is running with Finnhub REST & WebSocket!"

@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok", 200

def start(update, context):
    user = update.effective_user
    database.register_user(user.id, user.username)
    
    keyboard = [
        [KeyboardButton("📊 Аналіз усіх пар"), KeyboardButton("💵 Пари")],
        [KeyboardButton("📈 Статистика"), KeyboardButton("📋 Логи фільтру")]
    ]
    update.message.reply_text("Бот Racio_1 готовий до роботи! 🚀 Оберіть дію в меню:", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

def show_pairs_menu(chat_id):
    buttons = []
    row = []
    for name in PAIRS_MAP.keys():
        row.append(InlineKeyboardButton(name, callback_data=f"pair_{name}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    
    reply_markup = InlineKeyboardMarkup(buttons)
    bot.send_message(chat_id=chat_id, text="Оберіть валютну пару для миттєвого мульти-ТФ аналізу:", reply_markup=reply_markup)

def process_single_pair(chat_id, name, ticker):
    try:
        session_str, session_code, hour = get_current_session_info()
        current_time = time.time()
        
        if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 180:
            bot.send_message(chat_id=chat_id, text=f"⏳ Пара {name} на кулдауні (зачекайте 3 хвилини).")
            return

        # 1. Завантаження даних із підтримкою Finnhub та Resampling
        df_daily, df_macro, df_mid, df_fast, df_micro = fetch_all_timeframes(ticker)
        
        if df_macro is None or df_macro.empty or df_fast is None or df_fast.empty:
            bot.send_message(chat_id=chat_id, text=f"⚠️ Не вдалося завантажити котирування для {name}")
            return

        ws_price = get_live_price(ticker)

        # 2. Обчислення індикаторів
        if df_daily is not None and not df_daily.empty:
            df_daily = analyzer.calculate_indicators(df_daily)
        df_macro = analyzer.calculate_indicators(df_macro)
        df_mid = analyzer.calculate_indicators(df_mid)
        df_fast = analyzer.calculate_indicators(df_fast)
        df_micro = analyzer.calculate_indicators(df_micro)

        global_trend = analyzer.get_trend(df_macro, span_val=200)
        mid_trend = analyzer.get_trend(df_mid, span_val=50)
        pivots = analyzer.calculate_pivots(df_daily if (df_daily is not None and not df_daily.empty) else df_macro)
        
        tf_dict = {'1m': df_micro, '5m': df_fast, '15m': df_mid, '1h': df_macro}
        sig_data = analyzer.generate_signal(
            global_trend=global_trend,
            mid_trend=mid_trend,
            df_daily=df_daily,
            tf_dict=tf_dict
        )
        
        signal_type = sig_data.get('signal', 'CALL')
        signal_score = sig_data.get('score', 60)
        primary_tf = sig_data.get('primary_tf', '5m')
        
        rsi = sig_data.get('rsi', 50)
        adx = sig_data.get('adx', 20)
        bb_width = float(df_fast['bb_width'].iloc[-2]) if 'bb_width' in df_fast.columns and len(df_fast) >= 2 else 0.001
        
        pct_b_val = float(sig_data.get('pct_b', 0.5)) if 'pct_b' in sig_data else 0.5

        divergence_str = str(sig_data.get('divergence', 'NONE'))
        volatility_ratio = sig_data.get('volatility_ratio', 1.0)
        wick_ratio = sig_data.get('wick_ratio', 0.0)
        ema_dist = sig_data.get('ema_dist', 0.0)
        
        strategy_name = sig_data.get('strategy', 'Мульти-ТФ Сигнал')
        strategy_priority = sig_data.get('priority', 2)
        calculated_expiration = sig_data.get('expiration_minutes', 5)
        
        current_price = ws_price if ws_price else (float(df_fast['close'].iloc[-1]) if not df_fast.empty else 0.0)
        dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
        
        win_probability = ml_filter.predict_signal_probability(
            rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot,
            volatility_ratio, wick_ratio, ema_dist
        )
        
        ai_confidence = 7
        ai_reason = "Консультація за індикаторною моделлю"
        optimal_tf_final = primary_tf

        try:
            macro_chart = create_chart_image(df_macro, name, tf_label="1h")
            mid_chart = create_chart_image(df_mid, name, tf_label="15m")
            micro_chart = create_chart_image(df_fast, name, tf_label="5m")

            ai_payload = {
                'signal': signal_type, 'score': signal_score, 'primary_tf': primary_tf,
                'adx': adx, 'global_trend': global_trend, 'mid_trend': mid_trend,
                'reason': sig_data.get('reason'), 'rsi': rsi, 'atr': sig_data.get('atr'),
                'suggested_exp': calculated_expiration, 'strategy': strategy_name,
                'priority': strategy_priority, 'wick_ratio': wick_ratio,
                'ema_dist': ema_dist, 'volatility_ratio': volatility_ratio,
                'divergence': divergence_str
            }

            ai_audit = ai_advisor.evaluate_signal(name, ai_payload, macro_chart, mid_chart, micro_chart) or {}
            ai_confidence = int(ai_audit.get("confidence", 7))
            ai_reason = str(ai_audit.get("reason", "Підтверджено ШІ-консультантом"))
            optimal_tf_final = str(ai_audit.get("optimal_tf", primary_tf))
            
            if ai_audit.get("suggested_expiration"):
                calculated_expiration = int(ai_audit.get("suggested_expiration"))
        except Exception as e:
            logger.warning(f"⚠️ ШІ-сервіси тимчасово офлайн для {name}: {e}")

        last_sent_signals[ticker] = time.time()  
        expiration = calculated_expiration
        
        icon = "🟢" if signal_type == "CALL" else "🔴"
        action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"
        
        msg_text = (
            f"📊 **{name} ({ticker})**\n"
            f"{icon} **{action_text}** | ⏱ Експірація: **{expiration} хв**\n"
            f"⚡ **Signal Score:** `{signal_score}/100` | 📐 ТФ: `{optimal_tf_final}`\n"
            f"🎯 Стратегія: `{strategy_name}`\n"
            f"----------------------------------\n"
            f"📈 **Параметри ринку:**\n"
            f"• Ціна входу: `{current_price:.5f}` {'⚡ (Realtime)' if ws_price else ''}\n"
            f"• Тренди (1h / 15m): `{global_trend} / {mid_trend}`\n"
            f"• RSI: `{rsi}` | ADX: `{adx}` | %B: `{pct_b_val:.2f}`\n"
            f"• Дивергенція: `{divergence_str}`\n"
            f"• ATR Ratio: `{volatility_ratio}` | Відхилення EMA: `{ema_dist}%`\n"
            f"🌐 Сесія: `{session_str}`\n"
            f"----------------------------------\n"
            f"🤖 **Аналітика моделей:**\n"
            f"• ML-ймовірність: `{round(win_probability * 100, 1)}%`\n"
            f"• ШІ-впевненість: `{ai_confidence}/10`\n"
            f"💡 **Обґрунтування:** _{str(sig_data.get('reason'))}_\n"
            f"🛡 **Висновок ШІ:** _{ai_reason}_"
        )
        
        sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="Markdown")
        
        timestamp_dt = datetime.utcnow()
        timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        # 3. Збереження в БД та запуск таймера
        sig_id = database.save_signal(
            chat_id=chat_id,
            message_id=sent_msg.message_id,
            ticker=ticker,
            signal_type=signal_type,
            entry_price=current_price,
            expiration_mins=expiration,
            timestamp_str=timestamp_str,
            message_text=msg_text
        )

        if sig_id:
            schedule_signal_timer(sig_id, timestamp_dt, expiration)

    except Exception as e:
        logger.exception(f"Помилка обробки сигналу для {name}: {e}")
        bot.send_message(chat_id=chat_id, text=f"❌ Сталася помилка під час аналізу {name}.")
