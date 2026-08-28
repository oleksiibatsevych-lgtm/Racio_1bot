import os
import io
import time
import threading
import requests
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CallbackQueryHandler, CommandHandler, MessageHandler, Filters

from config import TELEGRAM_TOKEN, PAIRS_MAP
from indicators import AdaptiveTechnicalAnalysis
import database
from ml_model import TradingMLFilter
from from ai_advisor import AITradingAdvisor
from charts import create_chart_image

app = Flask(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

analyzer = AdaptiveTechnicalAnalysis()
ml_filter = TradingMLFilter()
ai_advisor = AITradingAdvisor()

last_sent_signals = {}

def fetch_yahoo_data(ticker, interval="3m", range_period="5d"):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {"interval": interval, "range": range_period, "includeAdjustedClose": "true"}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, headers=headers, params=params, timeout=(3, 5))
        if response.status_code != 200:
            return pd.DataFrame()
            
        data = response.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return pd.DataFrame()
            
        res = result[0]
        timestamps = res.get("timestamp", [])
        quotes = res.get("indicators", {}).get("quote", [{}])[0]
        
        if not timestamps or not quotes:
            return pd.DataFrame()
            
        df = pd.DataFrame({
            "open": quotes.get("open", []),
            "high": quotes.get("high", []),
            "low": quotes.get("low", []),
            "close": quotes.get("close", []),
            "volume": quotes.get("volume", [0] * len(timestamps))
        }, index=pd.to_datetime(timestamps, unit="s"))
        
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df["volume"] = df["volume"].fillna(0)
        return df
    except Exception as e:
        print(f"Помилка завантаження {ticker}: {e}")
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
            return 15
        price = float(df_fast['close'].iloc[-1])
        atr_pct = (atr / price) * 100
        # Розширений час експірації для згладжування початкових коливань та відкатів
        if adx > 30: return 10
        elif adx > 22: return 15
        elif adx < 16: return 25
        else: return 20 if atr_pct < 0.05 else (15 if atr_pct < 0.12 else 10)
    except:
        return 15

def process_signal_expiration(sig_id):
    try:
        res_data = database.evaluate_single_signal(sig_id, fetch_yahoo_data)
        if res_data and res_data.get("chat_id") and res_data.get("message_id"):
            pips_val = res_data['pips']
            pips_str = f"+{pips_val}" if pips_val > 0 else str(pips_val)
            
            res_icon = f"🏁 Результат: WIN ✅ ({pips_str} п.)" if res_data['result'] == 'WIN' else f"🏁 Результат: LOSS ❌ ({pips_str} п.)"
            orig_txt = res_data["message_text"]
            if "🏁 Результат" not in orig_txt:
                bot.edit_message_text(chat_id=res_data["chat_id"], message_id=res_data["message_id"], text=f"{orig_txt}\n{res_icon}")
    except Exception as e:
        print(f"Помилка таймера експірації {sig_id}: {e}")

def schedule_signal_timer(sig_id, timestamp_str, expiration_mins):
    try:
        signal_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        expiry_time = signal_time + timedelta(minutes=expiration_mins)
        delay = max((expiry_time - datetime.now()).total_seconds(), 1)
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
            delay = max((expiry_time - datetime.now()).total_seconds(), 2 + (i * 2))
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
    update.message.reply_text("Бот Racio_1 готовий до роботи (3m таймфрейм + Z-Score + ШІ-аудит)! 🚀", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

def train_ml_command(update, context):
    _, msg = ml_filter.train_model()
    update.message.reply_text(msg)

def handle_text_menu(update, context):
    chat_id = update.message.chat_id
    text = update.message.text
    
    if text == "💵 Пари":
        pairs = list(PAIRS_MAP.items())
        keyboard = []
        for i in range(0, len(pairs), 2):
            row = [InlineKeyboardButton(pairs[i][0], callback_data=f"scan_{pairs[i][1]}")]
            if i + 1 < len(pairs): row.append(InlineKeyboardButton(pairs[i+1][0], callback_data=f"scan_{pairs[i+1][1]}"))
            keyboard.append(row)
        update.message.reply_text("📌 Оберіть пару для аналізу:", reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif text == "📊 Аналіз усіх пар":
        if is_news_blackout_window():
            update.message.reply_text("⚠️ Увага: Зараз період підвищеної новинної волатильності. Сканування призупинено.")
            return

        update.message.reply_text("🔄 Сканування на 3m таймфреймі з перевіркою Z-Score та мультимодальним ШІ-аудитом...")
        session_str, session_code, hour = get_current_session_info()
        
        sent_signals_count = 0
        filtered_count = 0
        current_time = time.time()
        
        for name, ticker in PAIRS_MAP.items():
            try:
                if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 300:
                    continue

                df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
                df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
                df_fast = fetch_yahoo_data(ticker, interval="3m", range_period="3d")
                
                if df_macro.empty or df_mid.empty or df_fast.empty: continue

                global_trend = analyzer.get_trend(df_macro, span_val=200)
                mid_trend = analyzer.get_trend(df_mid, span_val=50)
                pivots = analyzer.calculate_pivots(df_macro)
                
                df_indicators = analyzer.calculate_indicators(df_fast)
                sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)
                
                signal_type = sig_data.get('signal')
                if signal_type not in ['CALL', 'PUT']: continue
                
                rsi = sig_data.get('rsi', 50)
                adx = sig_data.get('adx', 20)
                z_score = sig_data.get('z_score', 0.0)
                bb_width = float(df_indicators['bb_width'].iloc[-1]) if 'bb_width' in df_indicators.columns else 0.001
                div_code = sig_data.get('divergence', 'NONE')
                
                current_price = float(df_fast['close'].iloc[-1])
                dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
                
                win_probability = ml_filter.predict_signal_probability(rsi, adx, bb_width, z_score, session_code, hour, div_code, dist_pivot)
                if win_probability < 0.54:
                    filtered_count += 1
                    continue
                
                # Мультимодальний ШІ-аудит за графіками
                macro_chart = create_chart_image(df_macro, name, "1h")
                mid_chart = create_chart_image(df_mid, name, "15m")
                micro_chart = create_chart_image(df_fast, name, "3m")
                
                ai_audit = ai_advisor.evaluate_signal(name, sig_data, macro_chart, mid_chart, micro_chart)
                if ai_audit.get("decision") == "NO":
                    filtered_count += 1
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
                    f"📉 RSI: {rsi} | Z-Score: {z_score} | ADX: {adx}\n"
                    f"🌐 Сесія: {session_str}\n"
                    f"🧠 ШІ-успіх: {round(win_probability * 100, 1)}% (Аудит: {ai_audit.get('confidence')}/10)\n"
                    f"💡 Причина: {sig_data.get('reason')}"
                )
                
                sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="Markdown")
                
                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sig_id = database.save_signal(
                    ticker=ticker, signal=signal_type, entry_price=current_price, expiration_mins=expiration,
                    chat_id=chat_id, message_id=sent_msg.message_id, rsi=rsi, adx=adx, bb_width=bb_width,
                    z_score=z_score, session_code=session_code, hour=hour, divergence=div_code,
                    dist_pivot=dist_pivot, message_text=msg_text
                )
                schedule_signal_timer(sig_id, timestamp_str, expiration)
                time.sleep(0.5)
            except Exception as e:
                print(f"Помилка {ticker}: {e}")
                
        bot.send_message(chat_id=chat_id, text=f"✅ Сканування завершено!\n📤 Надіслано сигналів: {sent_signals_count}\n🛡 Відсіяно фільтрами: {filtered_count}")
        
    elif text == "📈 Статистика":
        update.message.reply_text("🔄 Розрахунок статистики та перенавчання моделі...")
        try:
            stats = database.get_overall_stats()
            ml_filter.train_model()
            stats_text = (
                f"📈 **Правдива статистика трейдингу:**\n"
                f"• Успішних угод (WIN): {stats.get('wins', 0)}\n"
                f"• Усього перевірених угод: {stats.get('total', 0)}\n"
                f"• Реальний вінрейт: **{stats.get('winrate', 0)}%**\n\n"
                f"🤖 *Модель успішно адаптована за реальними даними.*"
            )
            update.message.reply_text(stats_text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 Звіт ШІ", callback_data="get_ai_report")]]))
        except Exception as e:
            update.message.reply_text(f"Помилка статистики: {e}")

def button_callback(update, context):
    query = update.callback_query
    try: query.answer()
    except: pass
    
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
        
        query.edit_message_text(text=f"🔄 Глибокий аналіз **{name}** (3m + Z-Score + AI)...", parse_mode="Markdown")
        try:
            df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
            df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
            df_fast = fetch_yahoo_data(ticker, interval="3m", range_period="3d")
            
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
                query.edit_message_text(text=f"ℹ️ По **{name}** наразі сигнал **HOLD** (умови тренду або Z-Score відкату не сформовані).", parse_mode="Markdown")
                return

            rsi = sig_data.get('rsi', 50)
            adx = sig_data.get('adx', 20)
            z_score = sig_data.get('z_score', 0.0)
            bb_width = float(df_indicators['bb_width'].iloc[-1]) if 'bb_width' in df_indicators.columns else 0.001
            div_code = sig_data.get('divergence', 'NONE')
            
            current_price = float(df_fast['close'].iloc[-1])
            dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
            
            win_probability = ml_filter.predict_signal_probability(rsi, adx, bb_width, z_score, session_code, hour, div_code, dist_pivot)
            if win_probability < 0.54:
                query.edit_message_text(text=f"🛡 ШІ відхилив сигнал по **{name}** (Ймовірність: {round(win_probability * 100, 1)}%).", parse_mode="Markdown")
                return

            macro_chart = create_chart_image(df_macro, name, "1h")
            mid_chart = create_chart_image(df_mid, name, "15m")
            micro_chart = create_chart_image(df_fast, name, "3m")
            ai_audit = ai_advisor.evaluate_signal(name, sig_data, macro_chart, mid_chart, micro_chart)
            
            if ai_audit.get("decision") == "NO":
                query.edit_message_text(text=f"🛡 ШІ-аудит графіків відхилив угоду по **{name}**: {ai_audit.get('reason')}", parse_mode="Markdown")
                return

            atr = sig_data.get('atr')
            expiration = calculate_dynamic_expiration(df_fast, atr, adx=adx)
            icon = "🟢" if signal_type == "CALL" else "🔴"
            action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"

            text = (
                f"📊 {name} ({ticker})\n"
                f"{icon} {action_text} | ⏱ {expiration} хв\n"
                f"🎯 Ціна входу: {current_price:.5f}\n"
                f"📈 Тренд (гл/сер): {global_trend} / {mid_trend}\n"
                f"📉 RSI: {rsi} | Z-Score: {z_score} | ADX: {adx}\n"
                f"🌐 Сесія: {session_str}\n"
                f"🧠 ШІ-успіх: {round(win_probability * 100, 1)}% (Аудит: {ai_audit.get('confidence')}/10)\n"
                f"💡 Причина: {sig_data.get('reason')}"
            )
            query.edit_message_text(text=text, parse_mode="Markdown")
            
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sig_id = database.save_signal(
                ticker=ticker, signal=signal_type, entry_price=current_price, expiration_mins=expiration,
                chat_id=query.message.chat_id, message_id=query.message.message_id, rsi=rsi, adx=adx,
                bb_width=bb_width, z_score=z_score, session_code=session_code, hour=hour,
                divergence=div_code, dist_pivot=dist_pivot, message_text=text
            )
            schedule_signal_timer(sig_id, timestamp_str, expiration)
        except Exception as e:
            query.edit_message_text(text=f"❌ Помилка обробки: {str(e)}")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("train_ml", train_ml_command))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_menu))
dispatcher.add_handler(CallbackQueryHandler(button_callback))
