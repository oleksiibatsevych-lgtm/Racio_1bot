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

from config import TELEGRAM_TOKEN, PAIRS_MAP
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

def fetch_yahoo_data(ticker, interval="1m", range_period="7d"):
    """Завантаження котирувань через yfinance з обходом обмежень."""
    try:
        df_yf = yf.download(
            tickers=ticker,
            period=range_period,
            interval=interval,
            progress=False,
            auto_adjust=True
        )
        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            
            df_yf = df_yf.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            df_yf = df_yf[["open", "high", "low", "close", "volume"]].copy()
            df_yf.dropna(subset=["open", "high", "low", "close"], inplace=True)
            if not df_yf.empty:
                if df_yf.index.tz is not None:
                    df_yf.index = df_yf.index.tz_localize(None)
                return df_yf
    except Exception as e:
        logger.warning(f"yf.download failed for {ticker}: {e}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
        }
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": interval, "range": range_period}
        
        if curl_requests:
            response = curl_requests.get(url, headers=headers, params=params, timeout=7, impersonate="chrome120")
        else:
            response = requests.get(url, headers=headers, params=params, timeout=7)

        if response.status_code == 200:
            data = response.json()
            result = data.get("chart", {}).get("result")
            if result:
                res = result[0]
                timestamps = res.get("timestamp", [])
                quotes = res.get("indicators", {}).get("quote", [{}])[0]
                if timestamps and quotes and quotes.get("close"):
                    df = pd.DataFrame({
                        "open": quotes.get("open", []),
                        "high": quotes.get("high", []),
                        "low": quotes.get("low", []),
                        "close": quotes.get("close", []),
                        "volume": quotes.get("volume", [0] * len(timestamps))
                    }, index=pd.to_datetime(timestamps, unit="s"))
                    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                    df["volume"] = df["volume"].fillna(0)
                    if not df.empty:
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)
                        return df
    except Exception as e:
        logger.warning(f"Direct Yahoo query failed for {ticker}: {e}")

    return pd.DataFrame()

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
    try:
        res_data = database.evaluate_single_signal(sig_id, fetch_yahoo_data_func=fetch_yahoo_data)
        if res_data and res_data.get("chat_id") and res_data.get("message_id"):
            pips_val = int(round(float(res_data['pips'])))
            pips_str = f"+{pips_val}" if pips_val > 0 else str(pips_val)
            
            res_result = res_data['result']
            if res_result == 'WIN':
                res_icon = f"🏁 Результат: WIN ✅ ({pips_str} п.)"
            elif res_result == 'NEUTRAL':
                res_icon = f"🏁 Результат: NEUTRAL ➖ ({pips_str} п.)"
            else:
                res_icon = f"🏁 Результат: LOSS ❌ ({pips_str} п.)"

            orig_txt = res_data["message_text"]
            if orig_txt and "🏁 Результат" not in orig_txt:
                bot.edit_message_text(chat_id=res_data["chat_id"], message_id=res_data["message_id"], text=f"{orig_txt}\n{res_icon}")
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
    return "Racio_1bot is running with Finnhub WebSocket!"

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
    # Розподіл 21 пари по 3 кнопки у рядку для компактного вигляду
    for name in PAIRS_MAP.keys():
        row.append(InlineKeyboardButton(name, callback_data=f"pair_{name}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    
    reply_markup = InlineKeyboardMarkup(buttons)
    bot.send_message(chat_id=chat_id, text="Оберіть валютну пару для миттєвого мульти-ТФ аналізу:", reply_markup=reply_markup)

def train_ml_command(update, context):
    _, msg = ml_filter.train_model()
    update.message.reply_text(msg)

def process_single_pair(chat_id, name, ticker):
    try:
        session_str, session_code, hour = get_current_session_info()
        current_time = time.time()
        
        if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 180:
            bot.send_message(chat_id=chat_id, text=f"⏳ Пара {name} на кулдауні (зачекайте 3 хвилини).")
            return

        df_daily = fetch_yahoo_data(ticker, interval="1d", range_period="30d")
        df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
        df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
        df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
        df_micro = fetch_yahoo_data(ticker, interval="1m", range_period="7d")
        
        if df_macro.empty or df_mid.empty or df_fast.empty or df_micro.empty:
            bot.send_message(chat_id=chat_id, text=f"⚠️ Не вдалося завантажити котирування для {name}")
            return

        ws_price = get_live_price(ticker)
        if ws_price:
            df_micro.iloc[-1, df_micro.columns.get_loc('close')] = ws_price
            df_fast.iloc[-1, df_fast.columns.get_loc('close')] = ws_price

        if not df_daily.empty:
            df_daily = analyzer.calculate_indicators(df_daily)
        df_macro = analyzer.calculate_indicators(df_macro)
        df_mid = analyzer.calculate_indicators(df_mid)
        df_fast = analyzer.calculate_indicators(df_fast)
        df_micro = analyzer.calculate_indicators(df_micro)

        global_trend = analyzer.get_trend(df_macro, span_val=200)
        mid_trend = analyzer.get_trend(df_mid, span_val=50)
        pivots = analyzer.calculate_pivots(df_daily if not df_daily.empty else df_macro)
        
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
        
        current_price = ws_price if ws_price else (float(df_fast['close'].iloc[-2]) if len(df_fast) >= 2 else float(df_fast['close'].iloc[-1]))
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
        
        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        sig_id = database.save_signal(
            ticker, signal_type, current_price, expiration, chat_id, sent_msg.message_id,
            ai_decision="YES", ai_confidence=ai_confidence, ai_reason=ai_reason,
            rsi=rsi, adx=adx, bb_width=bb_width,
            session_code=session_code, hour=hour, divergence=divergence_str,
            dist_pivot=dist_pivot, message_text=msg_text,
            volatility_ratio=volatility_ratio, wick_ratio=wick_ratio, ema_dist=ema_dist
        )
        if sig_id:
            schedule_signal_timer(sig_id, timestamp_str, expiration)
    except Exception as e:
        logger.exception(f"Помилка обробки пари {ticker}: {e}")
        bot.send_message(chat_id=chat_id, text=f"❌ Сталася помилка при аналізі {ticker}.")

def run_full_scan_background(chat_id):
    clear_filtered_logs(chat_id)
    bot.send_message(chat_id=chat_id, text="🔍 Розпочато повний аналіз усіх 21 валютних пар...")
    
    def worker():
        for name, ticker in PAIRS_MAP.items():
            process_single_pair(chat_id, name, ticker)
            time.sleep(2)
        bot.send_message(chat_id=chat_id, text="✅ Повний сканер завершив перевірку всіх пар!")

    threading.Thread(target=worker, daemon=True).start()

def show_stats(chat_id):
    try:
        stats_text = database.get_stats_summary()
        bot.send_message(chat_id=chat_id, text=f"📈 **Статистика сигналів:**\n\n{stats_text}", parse_mode="Markdown")
    except Exception as e:
        logger.exception(f"Помилка отримання статистики: {e}")
        bot.send_message(chat_id=chat_id, text="⚠️ Помилка зчитування статистики з бази даних.")

def show_logs(chat_id):
    logs = get_filtered_logs(chat_id)
    if not logs:
        bot.send_message(chat_id=chat_id, text="📋 Логи порожні або застаріли.")
    else:
        text = "📋 **Останні записи логів:**\n\n" + "\n".join(logs[:15])
        bot.send_message(chat_id=chat_id, text=text)

def handle_message(update, context):
    text = update.message.text
    chat_id = update.effective_chat.id
    user = update.effective_user
    database.register_user(chat_id, user.username if user else None)

    if text == "📊 Аналіз усіх пар":
        run_full_scan_background(chat_id)
    elif text == "💵 Пари":
        show_pairs_menu(chat_id)
    elif text == "📈 Статистика":
        show_stats(chat_id)
    elif text == "📋 Логи фільтру":
        show_logs(chat_id)

def button_handler(update, context):
    query = update.callback_query
    query.answer()
    data = query.data

    if data.startswith("pair_"):
        pair_name = data.replace("pair_", "")
        ticker = PAIRS_MAP.get(pair_name)
        if ticker:
            bot.send_message(chat_id=query.message.chat_id, text=f"⏳ Запущено аналіз для **{pair_name}**...", parse_mode="Markdown")
            threading.Thread(target=process_single_pair, args=(query.message.chat_id, pair_name, ticker), daemon=True).start()

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("train_ml", train_ml_command))
dispatcher.add_handler(CallbackQueryHandler(button_handler))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

if __name__ == "__main__":
    start_finnhub_ws()
    
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Запуск сервера на порту {port}")
    app.run(host="0.0.0.0", port=port)
