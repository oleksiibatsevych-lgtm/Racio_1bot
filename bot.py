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

from config import TELEGRAM_TOKEN, PAIRS_MAP[span_8](start_span)[span_8](end_span)
from indicators import AdaptiveTechnicalAnalysis[span_9](start_span)[span_9](end_span)
import database[span_10](start_span)[span_10](end_span)
from ml_model import TradingMLFilter[span_11](start_span)[span_11](end_span)
from ai_advisor import AITradingAdvisor[span_12](start_span)[span_12](end_span)
from charts import create_chart_image[span_13](start_span)[span_13](end_span)

app = Flask(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

analyzer = AdaptiveTechnicalAnalysis()[span_14](start_span)[span_14](end_span)
ml_filter = TradingMLFilter()[span_15](start_span)[span_15](end_span)
advisor = AITradingAdvisor()[span_16](start_span)[span_16](end_span)

last_sent_signals = {}

def fetch_yahoo_data(ticker, interval="15m", range_period="10d"):
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
            return 10
        price = float(df_fast['close'].iloc[-1])
        atr_pct = (atr / price) * 100
        if adx > 30: return 5
        elif adx > 22: return 10
        elif adx < 16: return 20
        else: return 15 if atr_pct < 0.05 else (10 if atr_pct < 0.12 else 5)
    except:
        return 10

def process_signal_expiration(sig_id):
    try:
        res_data = database.evaluate_single_signal(sig_id, fetch_yahoo_data)[span_17](start_span)[span_17](end_span)
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
        delay = max((expiry_time - datetime.utcnow()).total_seconds(), 1)
        timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
        timer.daemon = True
        timer.start()
    except Exception as e:
        print(f"Помилка планування таймера {sig_id}: {e}")

def restore_pending_timers():
    pending = database.get_pending_signals()[span_18](start_span)[span_18](end_span)
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
    update.message.reply_text("Бот Racio_1 готовий до роботи (RSI + BB + Дивергенції + ШІ-аудит з графіками)! 🚀", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

def train_ml_command(update, context):
    _, msg = ml_filter.train_model()[span_19](start_span)[span_19](end_span)
    update.message.reply_text(msg)

def handle_text_menu(update, context):
    chat_id = update.message.chat_id
    text = update.message.text
    
    if text == "💵 Пари":
        pairs = list(PAIRS_MAP.items())[span_20](start_span)[span_20](end_span)
        keyboard = []
        for i in range(0, len(pairs), 2):
            row = [InlineKeyboardButton(pairs[i][0], callback_data=f"scan_{pairs[i][1]}")]
            if i + 1 < len(pairs): row.append(InlineKeyboardButton(pairs[i+1][0], callback_data=f"scan_{pairs[i+1][1]}"))
            keyboard.append(row)
        update.message.reply_text("📌 Оберіть пару для аналізу:", reply_markup=InlineKeyboardMarkup(keyboard))
        
    elif text == "📊 Аналіз усіх пар":
        if is_news_blackout_window():
            update.message.reply_text("⚠️ Увага: Зараз період підвищеної новинної волатильності. Сканування тимчасово призупинено.")
            return

        update.message.reply_text("🔄 Глибоке сканування та ШІ-аудит за оновленою стратегією...")
        session_str, session_code, hour = get_current_session_info()
        
        sent_signals_count = 0
        filtered_count = 0
        current_time = time.time()
        
        for name, ticker in PAIRS_MAP.items():[span_21](start_span)[span_21](end_span)
            try:
                if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 300:
                    continue

                df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
                df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
                df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
                
                if df_macro.empty or df_mid.empty or df_fast.empty: continue

                global_trend = analyzer.get_trend(df_macro, span_val=200)[span_22](start_span)[span_22](end_span)
                mid_trend = analyzer.get_trend(df_mid, span_val=50)[span_23](start_span)[span_23](end_span)
                local_trend = analyzer.get_trend(df_fast, span_val=10)[span_24](start_span)[span_24](end_span)
                pivots = analyzer.calculate_pivots(df_macro)[span_25](start_span)[span_25](end_span)
                
                df_indicators = analyzer.calculate_indicators(df_fast)[span_26](start_span)[span_26](end_span)
                sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)[span_27](start_span)[span_27](end_span)
                
                signal_type = sig_data.get('signal')
                if signal_type not in ['CALL', 'PUT']: continue
                
                rsi = sig_data.get('rsi', 50)
                adx = sig_data.get('adx', 20)
                bb_width = float(df_indicators['bb_width'].iloc[-1]) if 'bb_width' in df_indicators.columns else 0.001
                divergence_str = str(sig_data.get('divergence', 'NONE'))
                
                current_price = float(df_fast['close'].iloc[-1])
                dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
                
                win_probability = ml_filter.predict_signal_probability([span_28](start_span)[span_28](end_span)
                    rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot
                )
                if win_probability < 0.54:
                    filtered_count += 1
                    continue
                
                # Мультитаймфреймовий ШІ-аудит з графіками[span_29](start_span)[span_29](end_span)[span_30](start_span)[span_30](end_span)
                sig_data['global_trend'] = global_trend
                sig_data['mid_trend'] = mid_trend
                sig_data['local_trend'] = local_trend
                
                macro_chart = create_chart_image(df_macro, name, tf_label="1h")[span_31](start_span)[span_31](end_span)
                mid_chart = create_chart_image(df_mid, name, tf_label="15m")[span_32](start_span)[span_32](end_span)
                micro_chart = create_chart_image(df_fast, name, tf_label="5m")[span_33](start_span)[span_33](end_span)
                
                ai_verdict = advisor.evaluate_signal(name, sig_data, macro_chart, mid_chart, micro_chart)[span_34](start_span)[span_34](end_span)
                if ai_verdict.get("decision") != "YES":
                    filtered_count += 1
                    continue

                sent_signals_count += 1
                last_sent_signals[ticker] = time.time()  
                
                atr = sig_data.get('atr')
                expiration = int(ai_verdict.get("expiration", calculate_dynamic_expiration(df_fast, atr, adx=adx)))
                
                icon = "🟢" if signal_type == "CALL" else "🔴"
                action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"
                
                msg_text = (
                    f"📊 {name} ({ticker})\n"
                    f"{icon} {action_text} | ⏱ {expiration} хв\n"
                    f"🎯 Ціна входу: {current_price:.5f}\n"
                    f"📈 Тренд (гл/сер/лок): {global_trend} / {mid_trend} / {local_trend}\n"
                    f"📉 RSI: {rsi} | ADX: {adx} | Дивергенція: {divergence_str}\n"
                    f"🌐 Сесія: {session_str}\n"
                    f"🧠 ШІ-впевненість: {ai_verdict.get('confidence', 7)}/10\n"
                    f"💡 Вердикт ШІ: {ai_verdict.get('reason', sig_data.get('reason'))}"
                )
                
                # Надсилаємо графік 5m разом із текстом сигналу[span_35](start_span)[span_35](end_span)
                micro_chart.seek(0)
                sent_msg = bot.send_photo(chat_id=chat_id, photo=micro_chart, caption=msg_text, parse_mode="Markdown")
                
                timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                sig_id = database.save_signal([span_36](start_span)[span_36](end_span)
                    ticker, signal_type, current_price, expiration, chat_id, sent_msg.message_id,
                    rsi=rsi, adx=adx, bb_width=bb_width,
                    session_code=session_code, hour=hour, divergence=divergence_str,
                    dist_pivot=dist_pivot, message_text=msg_text
                )
                schedule_signal_timer(sig_id, timestamp_str, expiration)
                time.sleep(1)
            except Exception as e:
                print(f"Помилка {ticker}: {e}")
                
        bot.send_message(chat_id=chat_id, text=f"✅ Сканування завершено!\n📤 Надіслано сигналів: {sent_signals_count}\n🛡 Відсіяно фільтрами/ШІ: {filtered_count}")
        
    elif text == "📈 Статистика":
        update.message.reply_text("🔄 Розрахунок правдивої статистики...")
        try:
            stats = database.get_overall_stats()[span_37](start_span)[span_37](end_span)
            ml_filter.train_model()[span_38](start_span)[span_38](end_span)
            stats_text = (
                f"📈 **Правдива статистика трейдингу:**\n"
                f"• Успішних угод (WIN): {stats.get('wins', 0)}\n"
                f"• Усього перевірених угод: {stats.get('total', 0)}\n"
                f"• Реальний вінрейт: **{stats.get('winrate', 0)}%**\n\n"
                f"🤖 *Модель успішно перенавчена.*"
            )
            update.message.reply_text(stats_text, parse_mode="Markdown", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("📊 Звіт ШІ", callback_data="get_ai_report")]]))
        except Exception as e:
            update.message.reply_text(f"Помилка статистики: {e}")

def button_callback(update, context):
    query = update.callback_query
    try: query.answer()
    except: pass
    
    if query.data == "get_ai_report":
        query.edit_message_text(text=ml_filter.generate_strategy_report(), parse_mode="Markdown")[span_39](start_span)[span_39](end_span)
        return

    if query.data.startswith("scan_"):
        if is_news_blackout_window():
            query.edit_message_text(text="⚠️ Новинне вікно високого ризику. Сканування заблоковано.")
            return

        ticker = query.data.replace("scan_", "")
        name = next((k for k, v in PAIRS_MAP.items() if v == ticker), ticker)[span_40](start_span)[span_40](end_span)
        session_str, session_code, hour = get_current_session_info()
        
        query.edit_message_text(text=f"🔄 Мультитаймфреймовий аналіз та ШІ-аудит **{name}**...", parse_mode="Markdown")
        try:
            df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
            df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
            df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
            
            if df_macro.empty or df_mid.empty or df_fast.empty:
                query.edit_message_text(text=f"❌ Помилка завантаження даних для {ticker}")
                return

            global_trend = analyzer.get_trend(df_macro, span_val=200)[span_41](start_span)[span_41](end_span)
            mid_trend = analyzer.get_trend(df_mid, span_val=50)[span_42](start_span)[span_42](end_span)
            local_trend = analyzer.get_trend(df_fast, span_val=10)[span_43](start_span)[span_43](end_span)
            pivots = analyzer.calculate_pivots(df_macro)[span_44](start_span)[span_44](end_span)
            
            df_indicators = analyzer.calculate_indicators(df_fast)[span_45](start_span)[span_45](end_span)
            sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)[span_46](start_span)[span_46](end_span)
            
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
            
            win_probability = ml_filter.predict_signal_probability([span_47](start_span)[span_47](end_span)
                rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot
            )
            if win_probability < 0.54:
                query.edit_message_text(text=f"🛡 ШІ відхилив сигнал по **{name}** (Ймовірність: {round(win_probability * 100, 1)}% є нижчою за поріг безпеки).", parse_mode="Markdown")
                return

            sig_data['global_trend'] = global_trend
            sig_data['mid_trend'] = mid_trend
            sig_data['local_trend'] = local_trend
            
            macro_chart = create_chart_image(df_macro, name, tf_label="1h")[span_48](start_span)[span_48](end_span)
            mid_chart = create_chart_image(df_mid, name, tf_label="15m")[span_49](start_span)[span_49](end_span)
            micro_chart = create_chart_image(df_fast, name, tf_label="5m")[span_50](start_span)[span_50](end_span)
            
            ai_verdict = advisor.evaluate_signal(name, sig_data, macro_chart, mid_chart, micro_chart)[span_51](start_span)[span_51](end_span)
            if ai_verdict.get("decision") != "YES":
                query.edit_message_text(text=f"🛡 Gemini AI відхилив торгову ідею по **{name}**:\n_{ai_verdict.get('reason', 'Низька якість сетапу')}_", parse_mode="Markdown")
                return

            atr = sig_data.get('atr')
            expiration = int(ai_verdict.get("expiration", calculate_dynamic_expiration(df_fast, atr, adx=adx)))
            icon = "🟢" if signal_type == "CALL" else "🔴"
            action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"

            text = (
                f"📊 {name} ({ticker})\n"
                f"{icon} {action_text} | ⏱ {expiration} хв\n"
                f"🎯 Ціна входу: {current_price:.5f}\n"
                f"📈 Тренд (гл/сер/лок): {global_trend} / {mid_trend} / {local_trend}\n"
                f"📉 RSI: {rsi} | ADX: {adx} | Дивергенція: {divergence_str}\n"
                f"🌐 Сесія: {session_str}\n"
                f"🧠 ШІ-впевненість: {ai_verdict.get('confidence', 7)}/10\n"
                f"💡 Вердикт ШІ: {ai_verdict.get('reason', sig_data.get('reason'))}"
            )
            
            micro_chart.seek(0)
            bot.send_photo(chat_id=query.message.chat_id, photo=micro_chart, caption=text, parse_mode="Markdown")
            try:
                bot.delete_message(chat_id=query.message.chat_id, message_id=query.message.message_id)
            except:
                pass
            
            timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            sig_id = database.save_signal([span_52](start_span)[span_52](end_span)
                ticker, signal_type, current_price, expiration, query.message.chat_id, query.message.message_id,
                rsi=rsi, adx=adx, bb_width=bb_width,
                session_code=session_code, hour=hour, divergence=divergence_str,
                dist_pivot=dist_pivot, message_text=text
            )
            schedule_signal_timer(sig_id, timestamp_str, expiration)
        except Exception as e:
            query.edit_message_text(text=f"❌ Помилка обробки: {str(e)}")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("train_ml", train_ml_command))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_menu))
dispatcher.add_handler(CallbackQueryHandler(button_callback))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
