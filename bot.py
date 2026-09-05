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
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": interval, "range": range_period, "includeAdjustedClose": "true"}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        response = requests.get(url, headers=headers, params=params, timeout=(3, 5))
        if response.status_code == 200:
            data = response.json()
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
                        return df
    except Exception as e:
        logger.warning(f"Yahoo API query failed for {ticker}: {e}")

    try:
        yf_interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m"}
        mapped_interval = yf_interval_map.get(interval, "1m")
        
        df_yf = yf.download(ticker, period=range_period, interval=mapped_interval, progress=False)
        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            
            df_yf = df_yf.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            df_yf = df_yf[["open", "high", "low", "close", "volume"]].copy()
            df_yf.dropna(subset=["open", "high", "low", "close"], inplace=True)
            df_yf["volume"] = df_yf["volume"].fillna(0)
            return df_yf
    except Exception as e:
        logger.warning(f"yfinance fallback failed for {ticker}: {e}")

    return pd.DataFrame()

def calculate_smart_expiration(sig_data, df_indicators_5m, global_trend):
    """Адаптивний вибір часу експірації (3, 5, 10, 15 хв) для Флету та Тренду"""
    try:
        adx = float(sig_data.get('adx', 20))
        atr = float(sig_data.get('atr', 0))
        if atr == 0 and 'atr' in df_indicators_5m.columns:
            atr = float(df_indicators_5m['atr'].iloc[-1])
            
        close = float(df_indicators_5m['close'].iloc[-1])
        volatility_ratio = (atr / close) * 1000 if close > 0 else 1.0

        # Режим ФЛЕТУ (низький ADX або відсутність глобального тренду)
        if adx < 22 or global_trend == "NEUTRAL":
            if volatility_ratio < 1.0:
                return 15  # Повільний широкий флет
            elif volatility_ratio < 2.5:
                return 10  # Стандартний флет (від рівнів до рівнів)
            else:
                return 5   # Швидкі коливання у вузькому флеті

        # Режим ТРЕНДУ (високий ADX та чіткий глобальний тренд)
        elif adx >= 22:
            if adx > 35:
                return 15  # Потужний довгий тренд
            elif adx >= 28:
                return 10  # Впевнений середній тренд
            elif volatility_ratio > 3.5:
                return 3   # Імпульсний пробій / швидкий відкат
            else:
                return 5   # Звичайний трендовий рух

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

def is_news_blackout_window():
    now_utc = datetime.utcnow()
    if now_utc.minute < 10 and now_utc.hour in [12, 13, 14, 15, 18]:
        return True
    return False

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
    return "Racio_1bot is running!"

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
    update.message.reply_text("Бот Racio_1 готовий до роботи (Мультитаймфрейм + Флет/Тренд експірація)! 🚀", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

def train_ml_command(update, context):
    _, msg = ml_filter.train_model()
    update.message.reply_text(msg)

def run_full_scan_background(chat_id):
    clear_filtered_logs(chat_id)
    try:
        session_str, session_code, hour = get_current_session_info()
        sent_signals_count = 0
        filtered_count = 0
        current_time = time.time()
        
        logger.info(f"Початок фонового сканування для chat_id={chat_id}. Всього пар: {len(PAIRS_MAP)}")

        for name, ticker in PAIRS_MAP.items():
            try:
                logger.info(f"Перевірка пари: {name} ({ticker})")
                if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 300:
                    logger.info(f"Пара {ticker} пропущена через кулдаун")
                    continue

                df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
                df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
                df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
                df_micro = fetch_yahoo_data(ticker, interval="1m", range_period="7d")
                
                if df_macro.empty or df_mid.empty or df_fast.empty or df_micro.empty:
                    logger.warning(f"Не вдалося завантажити всі ТФ для {ticker}")
                    continue

                global_trend = analyzer.get_trend(df_macro, span_val=200)
                mid_trend = analyzer.get_trend(df_mid, span_val=50)
                pivots = analyzer.calculate_pivots(df_macro)
                
                df_indicators_5m = analyzer.calculate_indicators(df_fast)
                df_indicators_1m = analyzer.calculate_indicators(df_micro)
                
                sig_data = analyzer.generate_signal(df_indicators_1m, df_indicators_5m, global_trend, mid_trend)
                
                signal_type = sig_data.get('signal')
                if signal_type not in ['CALL', 'PUT']:
                    logger.info(f"Пара {name}: сигнал HOLD")
                    continue
                
                rsi = sig_data.get('rsi', 50)
                adx = sig_data.get('adx', 20)
                bb_width = float(df_indicators_5m['bb_width'].iloc[-1]) if 'bb_width' in df_indicators_5m.columns else 0.001
                divergence_str = str(sig_data.get('divergence', 'NONE'))
                
                current_price = float(df_indicators_5m['close'].iloc[-1])
                dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
                
                win_probability = ml_filter.predict_signal_probability(
                    rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot
                )
                if win_probability < 0.54:
                    filtered_count += 1
                    log_msg = f"❌ {name}: ML відхилив (Ймовірність {round(win_probability * 100, 1)}%)"
                    logger.info(log_msg)
                    save_filtered_log(chat_id, log_msg)
                    continue
                
                ai_confidence = 7
                ai_reason = "ШІ зайнятий / пройдено за індикаторами"
                ai_audit_failed = False
                
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
                        'suggested_exp': calculate_smart_expiration(sig_data, df_indicators_5m, global_trend)
                    }

                    ai_audit = ai_advisor.evaluate_signal(name, ai_payload, macro_chart, mid_chart, micro_chart)
                    ai_confidence = int(ai_audit.get("confidence", 5))
                    rejection_reason = str(ai_audit.get("reason", ""))
                    decision = ai_audit.get("decision", "NO")

                    busy_keywords = ["недоступні", "зайняті", "quota", "429", "resource", "exhausted", "limit", "busy", "unavailable"]
                    is_ai_busy = any(kw in rejection_reason.lower() for kw in busy_keywords)

                    if is_ai_busy:
                        ai_reason = "ШІ зайнятий (пройдено за індикаторами)"
                        ai_confidence = 7
                    elif decision != "YES" or ai_confidence < 7:
                        filtered_count += 1
                        log_msg = f"🤖 {name}: ШІ відхилив — {rejection_reason} (Впевненість: {ai_confidence}/10)"
                        logger.info(log_msg)
                        save_filtered_log(chat_id, log_msg)
                        ai_audit_failed = True
                    else:
                        ai_reason = rejection_reason if rejection_reason else "Схвалено ШІ"
                except Exception as e:
                    logger.warning(f"⚠️ Ліміт або недоступність ШІ для {name}: {e}")
                    ai_reason = "ШІ недоступний (пройдено за індикаторами)"
                    ai_confidence = 7

                if ai_audit_failed:
                    continue

                sent_signals_count += 1
                last_sent_signals[ticker] = time.time()  
                
                ai_sug_exp = ai_audit.get("suggested_expiration") if 'ai_audit' in locals() and not ai_audit_failed else None
                expiration = int(ai_sug_exp) if ai_sug_exp in [3, 5, 10, 15] else calculate_smart_expiration(sig_data, df_indicators_5m, global_trend)
                
                icon = "🟢" if signal_type == "CALL" else "🔴"
                action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"
                
                msg_text = (
                    f"📊 {name} ({ticker})\n"
                    f"{icon} {action_text} | ⏱ {expiration} хв\n"
                    f"🎯 Ціна входу: {current_price:.5f}\n"
                    f"📈 Тренд (гл/сер): {global_trend} / {mid_trend}\n"
                    f"📉 RSI: {rsi} | ADX: {adx} | Дивергенція: {divergence_str}\n"
                    f"🌐 Сесія: {session_str}\n"
                    f"🧠 ШІ-успіх (ML): {round(win_probability * 100, 1)}% | ШІ-впевненість: {ai_confidence}/10\n"
                    f"💡 Технічна причина: {str(sig_data.get('reason'))}\n"
                    f"🤖 Візуальний вердикт ШІ: {ai_reason}"
                )
                sent_msg = bot.send_message(chat_id=chat_id, text=msg_text)
                
                timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                sig_id = database.save_signal(
                    ticker, signal_type, current_price, expiration, chat_id, sent_msg.message_id,
                    rsi=rsi, adx=adx, bb_width=bb_width,
                    session_code=session_code, hour=hour, divergence=divergence_str,
                    dist_pivot=dist_pivot, message_text=msg_text
                )
                schedule_signal_timer(sig_id, timestamp_str, expiration)
                
                time.sleep(5)
            except Exception as e:
                logger.exception(f"Помилка обробки пари {ticker}: {e}")
                
        finish_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Переглянути чому відсіяно", callback_data="show_filtered_log")]])
        bot.send_message(
            chat_id=chat_id, 
            text=f"✅ Сканування завершено!\n📤 Надіслано сигналів: {sent_signals_count}\n🛡 Відсіяно фільтрами (ML + ШІ): {filtered_count}",
            reply_markup=finish_keyboard
        )
    except Exception as e:
        logger.exception(f"Помилка у фоновому скануванні: {e}")

def handle_text_menu(update, context):
    chat_id = update.message.chat_id
    text = update.message.text
    
    if text == "💵 Пари":
        pairs = list(PAIRS_MAP.items())
        keyboard = []
        for i in range(0, len(pairs), 2):
            row = [InlineKeyboardButton(pairs[i][0], callback_data=f"scan_{pairs[i][1]}")]
            if i + 1 < len(pairs): 
                row.append(InlineKeyboardButton(pairs[i+1][0], callback_data=f"scan_{pairs[i+1][1]}"))
            keyboard.append(row)
        update.message.reply_text("📌 Оберіть пару для аналізу:", reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif text == "📊 Аналіз усіх пар":
        if is_news_blackout_window():
            update.message.reply_text("⚠️ Увага: Зараз період підвищеної новинної волатильності. Сканування тимчасово призупинено.")
            return

        update.message.reply_text("🔄 Глибоке сканування запущено у фоновому режимі...")
        threading.Thread(target=run_full_scan_background, args=(chat_id,)).start()
        
    elif text == "📈 Статистика":
        update.message.reply_text("🔄 Розрахунок правдивої статистики...")
        try:
            stats = database.get_overall_stats()
            ml_filter.train_model()
            stats_text = (
                f"📈 Правдива статистика трейдингу:\n"
                f"• Успішних угод (WIN): {stats.get('wins', 0)}\n"
                f"• Нейтральних угод (BE): {stats.get('neutral', 0)}\n"
                f"• Збиткових угод (LOSS): {stats.get('losses', 0)}\n"
                f"• Усього перевірених угод: {stats.get('total', 0)}\n"
                f"• Реальний вінрейт: {stats.get('winrate', 0)}%\n\n"
                f"🤖 Модель успішно перенавчена."
            )
            update.message.reply_text(stats_text, reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 Звіт ШІ", callback_data="get_ai_report")]]))
        except Exception as e:
            logger.exception(f"Помилка статистики: {e}")
            update.message.reply_text(f"Помилка статистики: {e}")

def button_callback(update, context):
    query = update.callback_query
    chat_id = query.message.chat_id
    try: query.answer()
    except: pass
    
    if query.data == "show_filtered_log":
        logs = get_filtered_logs(chat_id)
        if not logs:
            query.answer("Немає записів про відсіяні пари у цьому сеансі.", show_alert=True)
            return
        
        log_text = "🛡 Причини відхилення сигналів:\n\n" + "\n".join(logs[:15])
        if len(log_text) > 4000:
            log_text = log_text[:4000] + "...\n(спискок скорочено)"
            
        query.edit_message_text(text=log_text)
        return

    if query.data == "get_ai_report":
        query.edit_message_text(text=ml_filter.generate_strategy_report())
        return

    if query.data.startswith("scan_"):
        if is_news_blackout_window():
            query.edit_message_text(text="⚠️ Новинне вікно високого ризику. Сканування заблоковано.")
            return

        ticker = query.data.replace("scan_", "")
        name = next((k for k, v in PAIRS_MAP.items() if v == ticker), ticker)
        session_str, session_code, hour = get_current_session_info()
        
        query.edit_message_text(text=f"🔄 Глибокий аналіз {name} (ML + ШІ-аудит)...")
        try:
            df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
            df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
            df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
            df_micro = fetch_yahoo_data(ticker, interval="1m", range_period="7d")
            
            if df_macro.empty or df_mid.empty or df_fast.empty or df_micro.empty:
                query.edit_message_text(text=f"❌ Помилка завантаження даних для {ticker}")
                return

            global_trend = analyzer.get_trend(df_macro, span_val=200)
            mid_trend = analyzer.get_trend(df_mid, span_val=50)
            pivots = analyzer.calculate_pivots(df_macro)
            
            df_indicators_5m = analyzer.calculate_indicators(df_fast)
            df_indicators_1m = analyzer.calculate_indicators(df_micro)
            
            sig_data = analyzer.generate_signal(df_indicators_1m, df_indicators_5m, global_trend, mid_trend)
            
            signal_type = sig_data.get('signal')
            if signal_type not in ['CALL', 'PUT']:
                query.edit_message_text(text=f"ℹ️ По {name} наразі сигнал HOLD (умови не сформовані).")
                return

            rsi = sig_data.get('rsi', 50)
            adx = sig_data.get('adx', 20)
            bb_width = float(df_indicators_5m['bb_width'].iloc[-1]) if 'bb_width' in df_indicators_5m.columns else 0.001
            divergence_str = str(sig_data.get('divergence', 'NONE'))
            
            current_price = float(df_indicators_5m['close'].iloc[-1])
            dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
            
            win_probability = ml_filter.predict_signal_probability(
                rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot
            )
            if win_probability < 0.54:
                query.edit_message_text(text=f"🛡 ШІ (ML) відхилив сигнал по {name} (Ймовірність нижче порогової).")
                return

            ai_confidence = 7
            ai_reason = "ШІ зайнятий / пройдено за індикаторами"
            
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
                    'suggested_exp': calculate_smart_expiration(sig_data, df_indicators_5m, global_trend)
                }

                ai_audit = ai_advisor.evaluate_signal(name, ai_payload, macro_chart, mid_chart, micro_chart)
                ai_confidence = int(ai_audit.get("confidence", 5))
                rejection_reason = str(ai_audit.get("reason", ""))
                decision = ai_audit.get("decision", "NO")

                busy_keywords = ["недоступні", "зайняті", "quota", "429", "resource", "exhausted", "limit", "busy", "unavailable"]
                is_ai_busy = any(kw in rejection_reason.lower() for kw in busy_keywords)

                if is_ai_busy:
                    ai_reason = "ШІ зайнятий (пройдено за індикаторами)"
                    ai_confidence = 7
                elif decision != "YES" or ai_confidence < 7:
                    query.edit_message_text(text=f"🛡 Візуальний ШІ-аудит відхилив сигнал по {name} (впевненість {ai_confidence}/10):\n{rejection_reason}")
                    return
                else:
                    ai_reason = rejection_reason if rejection_reason else "Схвалено ШІ"
            except Exception as e:
                logger.warning(f"⚠️ Ліміт або недоступність ШІ для {name}: {e}")
                ai_reason = "ШІ недоступний (пройдено за індикаторами)"
                ai_confidence = 7

            ai_sug_exp = ai_audit.get("suggested_expiration") if 'ai_audit' in locals() else None
            expiration = int(ai_sug_exp) if ai_sug_exp in [3, 5, 10, 15] else calculate_smart_expiration(sig_data, df_indicators_5m, global_trend)

            icon = "🟢" if signal_type == "CALL" else "🔴"
            action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"

            text = (
                f"📊 {name} ({ticker})\n"
                f"{icon} {action_text} | ⏱ {expiration} хв\n"
                f"🎯 Ціна входу: {current_price:.5f}\n"
                f"📈 Тренд (гл/сер): {global_trend} / {mid_trend}\n"
                f"📉 RSI: {rsi} | ADX: {adx} | Дивергенція: {divergence_str}\n"
                f"🌐 Сесія: {session_str}\n"
                f"🧠 ШІ-успіх (ML): {round(win_probability * 100, 1)}% | ШІ-впевненість: {ai_confidence}/10\n"
                f"💡 Технічна причина: {str(sig_data.get('reason'))}\n"
                f"🤖 Візуальний вердикт ШІ: {ai_reason}"
            )
            query.edit_message_text(text=text)
        except Exception as e:
            logger.exception(f"Помилка аналізу пари {ticker}: {e}")
            query.edit_message_text(text=f"❌ Помилка аналізу пари: {e}")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("train", train_ml_command))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_menu))
dispatcher.add_handler(CallbackQueryHandler(button_callback))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
