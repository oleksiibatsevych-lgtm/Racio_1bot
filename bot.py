import os
import io
import time
import threading
import sqlite3
import requests
import json
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CallbackQueryHandler, CommandHandler, MessageHandler, Filters

from config import TELEGRAM_TOKEN, PAIRS_MAP
from indicators import AdaptiveTechnicalAnalysis
import database
from ml_model import TradingMLFilter
from ai_advisor import AITradingAdvisor
from charts import create_chart_image

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
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    })
    
    try:
        session.get("https://finance.yahoo.com", timeout=5)
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": interval, "range": range_period, "includeAdjustedClose": "true"}
        
        response = session.get(url, params=params, timeout=5)
        if response.status_code == 200:
            try:
                data = response.json()
            except (ValueError, json.JSONDecodeError):
                logger.warning(f"Yahoo API returned non-JSON response for {ticker} (possible rate limit or block).")
                data = {}

            result = data.get("chart", {}).get("result")
            if result:
                res = result[0]
                timestamps = res.get("timestamp", [])
                quotes = res.get("indicators", {}).get("quote", [{}])[0]
                if timestamps and quotes:
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
        logger.warning(f"Yahoo API query failed for {ticker}: {e}")

    try:
        yf.pdr_override()
        df_yf = yf.download(ticker, period=range_period, interval=interval, progress=False, session=session)
        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            
            df_yf = df_yf.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            df_yf = df_yf[["open", "high", "low", "close", "volume"]].copy()
            df_yf.dropna(subset=["open", "high", "low", "close"], inplace=True)
            df_yf["volume"] = df_yf["volume"].fillna(0)
            if df_yf.index.tz is not None:
                df_yf.index = df_yf.index.tz_localize(None)
            return df_yf
    except Exception as e:
        logger.warning(f"yfinance fallback failed for {ticker}: {e}")

    return pd.DataFrame()

def get_weekly_extremum_distance(df_1h, current_price):
    if df_1h.empty:
        return 0.0
    weekly_high = df_1h['high'].max()
    weekly_low = df_1h['low'].min()
    weekly_range = weekly_high - weekly_low
    if weekly_range == 0:
        return 0.0
    dist_to_high = (weekly_high - current_price) / weekly_range
    dist_to_low = (current_price - weekly_low) / weekly_range
    return min(dist_to_high, dist_to_low)

def calculate_balanced_expiration(sig_data, df_indicators_5m, global_trend):
    try:
        adx = float(sig_data.get('adx', 20))
        atr = float(sig_data.get('atr', 0))
        if atr == 0 and 'atr' in df_indicators_5m.columns:
            atr = float(df_indicators_5m['atr'].iloc[-1])
            
        close = float(df_indicators_5m['close'].iloc[-1])
        volatility_ratio = (atr / close) * 1000 if close > 0 else 1.0

        is_trend = adx >= 22 and global_trend != "NEUTRAL"

        if is_trend:
            if adx > 32 or volatility_ratio < 1.5:
                return 15
            elif adx >= 25:
                return 10
            else:
                return 3
        else:
            if volatility_ratio < 1.2:
                return 10
            else:
                return 5
    except Exception as e:
        logger.warning(f"Помилка розрахунку експірації: {e}")
    
    return 5

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
        res_data = database.evaluate_single_signal(sig_id, fetch_yahoo_data)
        if res_data and res_data.get("chat_id") and res_data.get("message_id"):
            pips_val = res_data['pips']
            pips_str = f"+{pips_val}" if pips_val > 0 else str(pips_val)
            
            res_result = res_data['result']
            if res_result == 'WIN':
                res_icon = f"🏁 Результат: WIN ✅ ({pips_str} п.)"
            elif res_result == 'NEUTRAL':
                res_icon = f"🏁 Результат: NEUTRAL ➖ ({pips_str} п.)"
            else:
                res_icon = f"🏁 Результат: LOSS ❌ ({pips_str} п.)"

            orig_txt = res_data["message_text"]
            if "🏁 Результат" not in orig_txt:
                bot.edit_message_text(chat_id=res_data["chat_id"], message_id=res_data["message_id"], text=f"{orig_txt}\n{res_icon}")
    except Exception as e:
        logger.exception(f"Помилка таймера експірації {sig_id}: {e}")

def schedule_signal_timer(sig_id, timestamp_str, expiration_mins):
    try:
        signal_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
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
            expiry_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S") + timedelta(minutes=expiration_mins)
            delay = max((expiry_time - datetime.utcnow()).total_seconds(), 2 + (i * 2))
            timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
            timer.daemon = True
            timer.start()
        except:
            pass
    logger.info(f"⏳ Відновлено активних таймерів: {len(pending)}")

restore_pending_timers()

@app.route("/")
def index():
    return "Racio_1bot is running with enhanced ML and dynamic expiration!"

@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok", 200

def start(update, context):
    keyboard = [
        [KeyboardButton("📊 Аналіз усіх пар"), KeyboardButton("💵 Пари")],
        [KeyboardButton("📈 Статистика")]
    ]
    update.message.reply_text("Бот Racio_1 готовий до роботи (З покращеним вінрейтом та прозорими факторами)! 🚀", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

def train_ml_command(update, context):
    _, msg = ml_filter.train_model()
    update.message.reply_text(msg)

def pairs_command(update, context):
    keyboard = []
    row = []
    for name in PAIRS_MAP.keys():
        row.append(InlineKeyboardButton(name, callback_data=f"pair_{name}"))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    update.message.reply_text("Виберіть валютну пару для аналізу:", reply_markup=InlineKeyboardMarkup(keyboard))

def stats_command(update, context):
    stats_text = database.get_global_statistics()
    update.message.reply_text(stats_text)

def handle_text_message(update, context):
    text = update.message.text
    chat_id = update.message.chat_id
    if text == "📊 Аналіз усіх пар":
        update.message.reply_text("⏳ Розпочинаю масове сканування всіх пар у фоновому режимі...")
        threading.Thread(target=run_full_scan_background, args=(chat_id,)).start()
    elif text == "💵 Пари":
        pairs_command(update, context)
    elif text == "📈 Статистика":
        stats_command(update, context)

def handle_callback_query(update, context):
    query = update.callback_query
    query.answer()
    data = query.data
    chat_id = query.message.chat_id
    
    if data.startswith("pair_"):
        name = data.replace("pair_", "")
        if name in PAIRS_MAP:
            ticker = PAIRS_MAP[name]
            query.edit_message_text(text=f"⏳ Аналізую пару {name} ({ticker})... Зачекайте.")
            threading.Thread(target=process_single_pair, args=(chat_id, name, ticker)).start()

def process_single_pair(chat_id, name, ticker):
    try:
        session_str, session_code, hour = get_current_session_info()
        current_time = time.time()
        
        if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 300:
            bot.send_message(chat_id=chat_id, text=f"⏳ Пара {name} на кулдауні. Зачекайте трохи.")
            return

        df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="7d")
        df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
        df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
        df_micro = fetch_yahoo_data(ticker, interval="1m", range_period="7d")
        
        if df_macro.empty or df_mid.empty or df_fast.empty or df_micro.empty:
            bot.send_message(chat_id=chat_id, text=f"⚠️ Не вдалося завантажити дані для {name}")
            return

        global_trend = analyzer.get_trend(df_macro, span_val=200)
        mid_trend = analyzer.get_trend(df_mid, span_val=50)
        pivots = analyzer.calculate_pivots(df_macro)
        
        df_indicators_5m = analyzer.calculate_indicators(df_fast)
        df_indicators_1m = analyzer.calculate_indicators(df_micro)
        
        sig_data = analyzer.generate_signal(df_indicators_1m, df_indicators_5m, global_trend, mid_trend, df_macro)
        
        signal_type = sig_data.get('signal')
        if signal_type not in ['CALL', 'PUT']:
            bot.send_message(chat_id=chat_id, text=f"ℹ️ {name}: Поточний сигнал HOLD (немає чіткої точки входу).")
            return
        
        rsi = sig_data.get('rsi', 50)
        adx = sig_data.get('adx', 20)
        bb_width = float(df_indicators_5m['bb_width'].iloc[-1]) if 'bb_width' in df_indicators_5m.columns else 0.001
        divergence_str = str(sig_data.get('divergence', 'NONE'))
        volume_surge = float(df_indicators_5m['volume_surge'].iloc[-1]) if 'volume_surge' in df_indicators_5m.columns else 1.0
        tf_consistency = analyzer.get_timeframe_consistency(df_macro, df_mid, df_fast)
        
        current_price = float(df_indicators_5m['close'].iloc[-1])
        dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
        dist_weekly_ext = get_weekly_extremum_distance(df_macro, current_price)
        
        win_probability = ml_filter.predict_signal_probability(
            rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot, dist_weekly_ext, volume_surge, tf_consistency
        )
        
        dynamic_threshold = 0.54
        if "JPY" in ticker:
            dynamic_threshold = 0.52
        
        _, _, pair_session_wr = database.get_pair_session_winrate(ticker, session_code)
        session_factor_text = "Нейтрально"
        if pair_session_wr is not None:
            if pair_session_wr < 0.45:
                win_probability *= 0.80
                session_factor_text = f"Історичний WR сесії низький ({round(pair_session_wr*100, 1)}%): коефіцієнт -20%"
            elif pair_session_wr > 0.65:
                win_probability = min(1.0, win_probability * 1.15)
                session_factor_text = f"Історичний WR сесії високий ({round(pair_session_wr*100, 1)}%): коефіцієнт +15%"

        if win_probability < dynamic_threshold:
            log_msg = f"❌ {name}: ML відхилив (Ймовірність {round(win_probability * 100, 1)}% < поріг {round(dynamic_threshold * 100, 1)}%)"
            save_filtered_log(chat_id, log_msg)
            bot.send_message(chat_id=chat_id, text=f"{log_msg}\nСигнал відсіяно фільтром.")
            return
        
        ai_confidence = 5
        ai_reason = "ШІ недоступний"
        
        try:
            macro_chart = create_chart_image(df_macro, name, tf_label="1h")
            mid_chart = create_chart_image(df_mid, name, tf_label="15m")
            micro_chart = create_chart_image(df_indicators_5m, name, tf_label="5m")

            ai_payload = {
                'signal': signal_type,
                'adx': adx,
                'global_trend': global_trend,
                'mid_trend': mid_trend,
                'reason': sig_data.get('reason'),
                'rsi': rsi,
                'atr': sig_data.get('atr'),
                'suggested_exp': calculate_balanced_expiration(sig_data, df_indicators_5m, global_trend)
            }

            ai_audit = ai_advisor.evaluate_signal(name, ai_payload, macro_chart, mid_chart, micro_chart)
            decision = ai_audit.get("decision", "NO")
            rejection_reason = str(ai_audit.get("reason", ""))

            if decision == "FALLBACK" or any(kw in rejection_reason.lower() for kw in ["quota", "429", "limit", "busy", "unavailable", "недоступні"]):
                ai_reason = "[⚠️ ШІ-аудит пропущено через ліміти/квоту]"
                ai_confidence = 5
            else:
                ai_confidence = int(ai_audit.get("confidence", 5))
                ai_reason = rejection_reason if rejection_reason else "Схвалено ШІ"
                if decision != "YES" or ai_confidence < 6:
                    log_msg = f"🤖 {name}: ШІ відхилив — {rejection_reason} (Впевненість: {ai_confidence}/10)"
                    save_filtered_log(chat_id, log_msg)
                    bot.send_message(chat_id=chat_id, text=f"{log_msg}\nСигнал відхилено ШІ-радником.")
                    return
        except Exception as e:
            logger.warning(f"⚠️ Збій ШІ для {name}: {e}")
            ai_reason = "[⚠️ ШІ-аудит пропущено через збій зв'язку]"
            ai_confidence = 5

        last_sent_signals[ticker] = time.time()  
        
        ai_sug_exp = ai_audit.get("suggested_expiration") if 'ai_audit' in locals() and ai_audit.get("decision") == "YES" else None
        expiration = int(ai_sug_exp) if ai_sug_exp in [3, 5, 10, 15] else calculate_balanced_expiration(sig_data, df_indicators_5m, global_trend)
        
        icon = "🟢" if signal_type == "CALL" else "🔴"
        action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"
        
        msg_text = (
            f"📊 {name} ({ticker})\n"
            f"{icon} {action_text} | ⏱ {expiration} хв (ATR-оптимізовано)\n"
            f"🎯 Ціна входу: {current_price:.5f}\n"
            f"📈 Тренд (гл/сер): {global_trend} / {mid_trend} | Узгодженість: {int(tf_consistency*100)}%\n"
            f"📉 RSI: {rsi} | ADX: {adx} | Об'ємний імпульс: {volume_surge:.2f}x\n"
            f"🌐 Сесія: {session_str} | Вплив сесії: {session_factor_text}\n"
            f"🧠 ШІ-успіх (ML): {round(win_probability * 100, 1)}% | ШІ-впевненість: {ai_confidence}/10\n"
            f"💡 Технічна причина: {str(sig_data.get('reason'))}\n"
            f"🤖 Вердикт ШІ: {ai_reason}"
        )
        sent_msg = bot.send_message(chat_id=chat_id, text=msg_text)
        
        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        sig_id = database.save_signal(
            ticker, signal_type, current_price, expiration, chat_id, sent_msg.message_id,
            rsi=rsi, adx=adx, bb_width=bb_width,
            session_code=session_code, hour=hour, divergence=divergence_str,
            dist_pivot=dist_pivot, dist_weekly_ext=dist_weekly_ext, 
            volume_surge=volume_surge, tf_consistency=tf_consistency, message_text=msg_text
        )
        schedule_signal_timer(sig_id, timestamp_str, expiration)
    except Exception as e:
        logger.exception(f"Помилка обробки пари {ticker}: {e}")
        bot.send_message(chat_id=chat_id, text=f"❌ Сталася помилка при аналізі {ticker}.")

def run_full_scan_background(chat_id):
    clear_filtered_logs(chat_id)
    try:
        session_str, session_code, hour = get_current_session_info()
        sent_signals_count = 0
        filtered_count = 0
        current_time = time.time()
        
        for name, ticker in PAIRS_MAP.items():
            try:
                if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 300:
                    continue

                df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="7d")
                df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
                df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
                df_micro = fetch_yahoo_data(ticker, interval="1m", range_period="7d")
                
                if df_macro.empty or df_mid.empty or df_fast.empty or df_micro.empty:
                    continue

                global_trend = analyzer.get_trend(df_macro, span_val=200)
                mid_trend = analyzer.get_trend(df_mid, span_val=50)
                pivots = analyzer.calculate_pivots(df_macro)
                
                df_indicators_5m = analyzer.calculate_indicators(df_fast)
                df_indicators_1m = analyzer.calculate_indicators(df_micro)
                
                sig_data = analyzer.generate_signal(df_indicators_1m, df_indicators_5m, global_trend, mid_trend, df_macro)
                
                signal_type = sig_data.get('signal')
                if signal_type not in ['CALL', 'PUT']:
                    continue
                
                rsi = sig_data.get('rsi', 50)
                adx = sig_data.get('adx', 20)
                bb_width = float(df_indicators_5m['bb_width'].iloc[-1]) if 'bb_width' in df_indicators_5m.columns else 0.001
                divergence_str = str(sig_data.get('divergence', 'NONE'))
                volume_surge = float(df_indicators_5m['volume_surge'].iloc[-1]) if 'volume_surge' in df_indicators_5m.columns else 1.0
                tf_consistency = analyzer.get_timeframe_consistency(df_macro, df_mid, df_fast)
                
                current_price = float(df_indicators_5m['close'].iloc[-1])
                dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
                dist_weekly_ext = get_weekly_extremum_distance(df_macro, current_price)
                
                win_probability = ml_filter.predict_signal_probability(
                    rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot, dist_weekly_ext, volume_surge, tf_consistency
                )
                
                dynamic_threshold = 0.52 if "JPY" in ticker else 0.54
                
                _, _, pair_session_wr = database.get_pair_session_winrate(ticker, session_code)
                if pair_session_wr is not None:
                    if pair_session_wr < 0.45:
                        win_probability *= 0.80
                    elif pair_session_wr > 0.65:
                        win_probability = min(1.0, win_probability * 1.15)

                if win_probability < dynamic_threshold:
                    filtered_count += 1
                    continue
                
                ai_confidence = 5
                ai_reason = "ШІ недоступний"
                
                try:
                    macro_chart = create_chart_image(df_macro, name, tf_label="1h")
                    mid_chart = create_chart_image(df_mid, name, tf_label="15m")
                    micro_chart = create_chart_image(df_indicators_5m, name, tf_label="5m")

                    ai_payload = {
                        'signal': signal_type,
                        'adx': adx,
                        'global_trend': global_trend,
                        'mid_trend': mid_trend,
                        'reason': sig_data.get('reason'),
                        'rsi': rsi,
                        'atr': sig_data.get('atr'),
                        'suggested_exp': calculate_balanced_expiration(sig_data, df_indicators_5m, global_trend)
                    }

                    ai_audit = ai_advisor.evaluate_signal(name, ai_payload, macro_chart, mid_chart, micro_chart)
                    decision = ai_audit.get("decision", "NO")
                    rejection_reason = str(ai_audit.get("reason", ""))

                    if decision == "FALLBACK" or any(kw in rejection_reason.lower() for kw in ["quota", "429", "limit", "busy", "unavailable"]):
                        ai_confidence = 5
                    else:
                        ai_confidence = int(ai_audit.get("confidence", 5))
                        ai_reason = rejection_reason if rejection_reason else "Схвалено ШІ"
                        if decision != "YES" or ai_confidence < 6:
                            filtered_count += 1
                            continue
                except Exception as e:
                    logger.warning(f"⚠️ Збій ШІ в скані для {name}: {e}")

                last_sent_signals[ticker] = time.time()
                ai_sug_exp = ai_audit.get("suggested_expiration") if 'ai_audit' in locals() and ai_audit.get("decision") == "YES" else None
                expiration = int(ai_sug_exp) if ai_sug_exp in [3, 5, 10, 15] else calculate_balanced_expiration(sig_data, df_indicators_5m, global_trend)
                
                icon = "🟢" if signal_type == "CALL" else "🔴"
                action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"
                
                msg_text = (
                    f"📊 {name} ({ticker})\n"
                    f"{icon} {action_text} | ⏱ {expiration} хв (ATR-оптимізовано)\n"
                    f"🎯 Ціна входу: {current_price:.5f}\n"
                    f"📈 Тренд: {global_trend} / {mid_trend} | Узгодженість: {int(tf_consistency*100)}%\n"
                    f"📉 RSI: {rsi} | ADX: {adx} | Об'єм: {volume_surge:.2f}x\n"
                    f"🌐 Сесія: {session_str}\n"
                    f"🧠 ML: {round(win_probability * 100, 1)}% | ШІ: {ai_confidence}/10\n"
                    f"💡 Причина: {str(sig_data.get('reason'))}\n"
                    f"🤖 Вердикт ШІ: {ai_reason}"
                )
                sent_msg = bot.send_message(chat_id=chat_id, text=msg_text)
                sent_signals_count += 1
                
                timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                sig_id = database.save_signal(
                    ticker, signal_type, current_price, expiration, chat_id, sent_msg.message_id,
                    rsi=rsi, adx=adx, bb_width=bb_width, session_code=session_code, hour=hour,
                    divergence=divergence_str, dist_pivot=dist_pivot, dist_weekly_ext=dist_weekly_ext,
                    volume_surge=volume_surge, tf_consistency=tf_consistency, message_text=msg_text
                )
                schedule_signal_timer(sig_id, timestamp_str, expiration)
            except Exception as pair_err:
                logger.exception(f"Помилка пари {ticker} у фоновому скані: {pair_err}")
        
        bot.send_message(chat_id=chat_id, text=f"🏁 Сканування завершено!\n📤 Надіслано сигналів: {sent_signals_count}\n🛡 Відсіяно фільтрами: {filtered_count}")
    except Exception as e:
        logger.exception(f"Помилка фонового сканування: {e}")
        bot.send_message(chat_id=chat_id, text="❌ Сталася помилка під час масового сканування.")

# Реєстрація обробників (Handlers)
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("train_ml", train_ml_command))
dispatcher.add_handler(CommandHandler("pairs", pairs_command))
dispatcher.add_handler(CommandHandler("stats", stats_command))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_message))
dispatcher.add_handler(CallbackQueryHandler(handle_callback_query))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
