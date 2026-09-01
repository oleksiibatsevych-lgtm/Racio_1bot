import os
import io
import time
import threading
import sqlite3
import requests
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
from charts import create_combined_charts_image

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
        print(f"Помилка ініціалізації БД логів: {e}")

init_logs_db()

def clear_filtered_logs(chat_id):
    try:
        conn = sqlite3.connect("filtered_logs.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM filtered_logs WHERE chat_id = ?", (chat_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Помилка очищення логів: {e}")

def save_filtered_log(chat_id, log_text):
    try:
        conn = sqlite3.connect("filtered_logs.db", check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO filtered_logs (chat_id, log_text, timestamp) VALUES (?, ?, ?)", 
                       (chat_id, log_text, time.time()))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Помилка збереження логу: {e}")

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
        print(f"Помилка читання логів: {e}")
        return []

def fetch_yahoo_data(ticker, interval="15m", range_period="10d"):
    try:
        yf_interval_map = {"5m": "5m", "15m": "15m", "1h": "60m"}
        mapped_interval = yf_interval_map.get(interval, "15m")
        
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        df_yf = yf.download(ticker, period=range_period, interval=mapped_interval, progress=False, session=session)
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
        print(f"❌ Помилка завантаження yfinance для {ticker}: {e}")

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

def is_news_blackout_window():
    now_utc = datetime.utcnow()
    if now_utc.minute < 10 and now_utc.hour in [12, 13, 14, 15, 18]:
        return True
    return False

def calculate_dynamic_expiration(df_fast, atr, adx=20):
    try:
        if atr is None or pd.isna(atr) or atr == 0:
            return 10
        price = float(df_fast['close'].iloc[-1])
        atr_pct = (atr / price) * 100
        if adx > 30: 
            exp = 5
        elif adx > 22: 
            exp = 10
        elif adx < 16: 
            exp = 20
        else: 
            exp = 15 if atr_pct < 0.05 else (10 if atr_pct < 0.12 else 5)
        return max(5, min(20, exp))
    except:
        return 10

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
        print(f"Помилка таймера експірації {sig_id}: {e}")

def schedule_signal_timer(sig_id, timestamp_str, expiration_mins):
    try:
        signal_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        expiry_time = signal_time + timedelta(minutes=expiration_mins)
        delay = max((expiry_time - datetime.utcnow()).total_seconds(), 1)
        timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
        timer.daemon = True
        timer.start()
    except Exception as e:
        print(f"Помилка планування таймера {sig_id}: {e}")

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
    print(f"⏳ Відновлено активних таймерів: {len(pending)}")

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
    update.message.reply_text("Бот Racio_1 готовий до роботи (RSI + BB + Дивергенції + ШІ-аудит)! 🚀", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

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
        
        for name, ticker in PAIRS_MAP.items():
            try:
                if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 300:
                    time.sleep(4)
                    continue

                df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
                df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
                df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
                
                if df_macro.empty or df_mid.empty or df_fast.empty:
                    time.sleep(4)
                    continue

                global_trend = analyzer.get_trend(df_macro, span_val=200)
                mid_trend = analyzer.get_trend(df_mid, span_val=50)
                pivots = analyzer.calculate_pivots(df_macro)
                
                df_indicators = analyzer.calculate_indicators(df_fast)
                sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)
                
                signal_type = sig_data.get('signal')
                if signal_type not in ['CALL', 'PUT']:
                    time.sleep(4)
                    continue
                
                rsi = sig_data.get('rsi', 50)
                adx = sig_data.get('adx', 20)
                bb_width = float(df_indicators['bb_width'].iloc[-1]) if 'bb_width' in df_indicators.columns else 0.001
                divergence_str = str(sig_data.get('divergence', 'NONE'))
                
                current_price = float(df_fast['close'].iloc[-1])
                dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
                
                win_probability = ml_filter.predict_signal_probability(
                    rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot
                )
                if win_probability < 0.54:
                    filtered_count += 1
                    save_filtered_log(chat_id, f"❌ {name}: ML відхилив (Ймовірність {round(win_probability * 100, 1)}% < 54%)")
                    time.sleep(4)
                    continue
                
                ai_confidence = 5
                ai_reason = "Аудит пропущено"
                ai_audit_failed = False
                try:
                    combined_chart = create_combined_charts_image(df_macro, df_mid, df_fast, name)

                    ai_payload = {
                        'signal': signal_type,
                        'adx': adx,
                        'global_trend': global_trend,
                        'mid_trend': mid_trend,
                        'local_trend': analyzer.get_trend(df_fast, span_val=10),
                        'reason': sig_data.get('reason'),
                        'rsi': rsi,
                        'atr': sig_data.get('atr')
                    }

                    ai_audit = ai_advisor.evaluate_signal(name, ai_payload, combined_chart)
                    if ai_audit.get("decision") != "YES":
                        filtered_count += 1
                        rejection_reason = ai_audit.get("reason", "ШІ не схвалив")
                        save_filtered_log(chat_id, f"🤖 {name}: ШІ відхилив — _{rejection_reason}_")
                        ai_audit_failed = True
                    else:
                        ai_confidence = ai_audit.get("confidence", 7)
                        ai_reason = ai_audit.get("reason", "Схвалено ШІ")
                except Exception as e:
                    print(f"⚠️ Помилка ШІ-аудиту для {name}: {e}")

                if ai_audit_failed:
                    time.sleep(4)
                    continue

                sent_signals_count += 1
                last_sent_signals[ticker] = time.time()  
                
                atr = sig_data.get('atr')
                expiration = calculate_dynamic_expiration(df_fast, atr, adx=adx)
                
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
                    f"💡 Технічна причина: {sig_data.get('reason')}\n"
                    f"🤖 Візуальний вердикт ШІ: {ai_reason}"
                )
                sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="Markdown")
                
                timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                sig_id = database.save_signal(
                    ticker, signal_type, current_price, expiration, chat_id, sent_msg.message_id,
                    rsi=rsi, adx=adx, bb_width=bb_width,
                    session_code=session_code, hour=hour, divergence=divergence_str,
                    dist_pivot=dist_pivot, message_text=msg_text
                )
                schedule_signal_timer(sig_id, timestamp_str, expiration)
                
                time.sleep(4)
            except Exception as e:
                print(f"Помилка {ticker}: {e}")
                time.sleep(4)
                
        finish_keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Переглянути чому відсіяно", callback_data="show_filtered_log")]])
        bot.send_message(
            chat_id=chat_id, 
            text=f"✅ Сканування завершено!\n📤 Надіслано сигналів: {sent_signals_count}\n🛡 Відсіяно фільтрами (ML + ШІ): {filtered_count}",
            reply_markup=finish_keyboard
        )
    except Exception as e:
        print(f"Помилка у фоновому скануванні: {e}")

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
                f"📈 **Правдива статистика трейдингу:**\n"
                f"• Успішних угод (WIN): {stats.get('wins', 0)}\n"
                f"• Нейтральних угод (BE): {stats.get('neutral', 0)}\n"
                f"• Збиткових угод (LOSS): {stats.get('losses', 0)}\n"
                f"• Усього перевірених угод: {stats.get('total', 0)}\n"
                f"• Реальний вінрейт: **{stats.get('winrate', 0)}%**\n\n"
                f"🤖 *Модель успішно перенавчена.*"
            )
            update.message.reply_text(stats_text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 Звіт ШІ", callback_data="get_ai_report")]]))
        except Exception as e:
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
        
        log_text = "🛡 **Причини відхилення сигналів:**\n\n" + "\n".join(logs[:15])
        if len(log_text) > 4000:
            log_text = log_text[:4000] + "...\n(список скорочено)"
            
        query.edit_message_text(text=log_text, parse_mode="Markdown")
        return

    if query.data == "get_ai_report":
        query.edit_message_text(text=ml_filter.generate_strategy_report(), parse_mode="Markdown")
        return

    if query.data.startswith("scan_"):
        if is_news_blackout_window():
            query.edit_message_text(text="⚠️ Новинне вікно високого ризику. Сканування заблоковано.")
            return

        ticker = query.data.replace("scan_", "")
        name = next((k for k, v in PAIRS_MAP.items() if v == ticker), ticker)
        session_str, session_code, hour = get_current_session_info()
        
        query.edit_message_text(text=f"🔄 Глибокий аналіз **{name}** (ML + ШІ-аудит)...", parse_mode="Markdown")
        try:
            df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
            df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
            df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
            
            if df_macro.empty or df_mid.empty or df_fast.empty:
                query.edit_message_text(text=f"❌ Помилка завантаження даних для {ticker}")
                return

            global_trend = analyzer.get_trend(df_macro, span_val=200)
            mid_trend = analyzer.get_trend(df_mid, span_val=50)
            pivots = analyzer.calculate_pivots(df_macro)
            
            df_indicators = analyzer.calculate_indicators(df_fast)
            sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)
            
            signal_type = sig_data.get('signal')
            if signal_type not in ['CALL', 'PUT']:
                query.edit_message_text(text=f"ℹ️ По **{name}** наразі сигнал **HOLD** (умови не сформовані).", parse_mode="Markdown")
                return

            rsi = sig_data.get('rsi', 50)
            adx = sig_data.get('adx', 20)
            bb_width = float(df_indicators['bb_width'].iloc[-1]) if 'bb_width' in df_indicators.columns else 0.001
            divergence_str = str(sig_data.get('divergence', 'NONE'))
            
            current_price = float(df_fast['close'].iloc[-1])
            dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
            
            win_probability = ml_filter.predict_signal_probability(
                rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot
            )
            if win_probability < 0.54:
                query.edit_message_text(text=f"🛡 ШІ (ML) відхилив сигнал по **{name}** (Ймовірність: {round(win_probability * 100, 1)}% нижче порогової).", parse_mode="Markdown")
                return

            ai_confidence = 5
            ai_reason = "Аудит пропущено"
            try:
                combined_chart = create_combined_charts_image(df_macro, df_mid, df_fast, name)

                ai_payload = {
                    'signal': signal_type,
                    'adx': adx,
                    'global_trend': global_trend,
                    'mid_trend': mid_trend,
                    'local_trend': analyzer.get_trend(df_fast, span_val=10),
                    'reason': sig_data.get('reason'),
                    'rsi': rsi,
                    'atr': sig_data.get('atr')
                }

                ai_audit = ai_advisor.evaluate_signal(name, ai_payload, combined_chart)
                if ai_audit.get("decision") != "YES":
                    query.edit_message_text(text=f"🛡 Візуальний ШІ-аудит відхилив сигнал по **{name}**:\n_{ai_audit.get('reason')}_", parse_mode="Markdown")
                    return
                    
                ai_confidence = ai_audit.get("confidence", 7)
                ai_reason = ai_audit.get("reason", "Схвалено ШІ")
            except Exception as e:
                print(f"⚠️ Помилка ШІ-аудиту для {name}: {e}")

            atr = sig_data.get('atr')
            expiration = calculate_dynamic_expiration(df_fast, atr, adx=adx)

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
                f"💡 Технічна причина: {sig_data.get('reason')}\n"
                f"🤖 Візуальний вердикт ШІ: {ai_reason}"
            )
            query.edit_message_text(text=text, parse_mode="Markdown")
        except Exception as e:
            query.edit_message_text(text=f"❌ Помилка аналізу пари: {e}")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("train", train_ml_command))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_menu))
dispatcher.add_handler(CallbackQueryHandler(button_callback))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
