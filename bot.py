import os
import io
import time
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

app = Flask(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

analyzer = AdaptiveTechnicalAnalysis()
ml_filter = TradingMLFilter()

last_sent_signals = {}

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9"
})

def fetch_yahoo_data(ticker, interval="15m", range_period="10d"):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "interval": interval,
            "range": range_period,
            "includeAdjustedClose": "true"
        }
        
        response = session.get(url, params=params, timeout=(3, 5))
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
            
        opens = quotes.get("open", [])
        highs = quotes.get("high", [])
        lows = quotes.get("low", [])
        closes = quotes.get("close", [])
        volumes = quotes.get("volume", [])
        
        vol_data = volumes if volumes else [0] * len(timestamps)
        
        df = pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vol_data
        }, index=pd.to_datetime(timestamps, unit="s"))
        
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        df["volume"] = df["volume"].fillna(0)
        
        return df
    except Exception as e:
        print(f"Помилка завантаження {ticker}: {e}")
        return pd.DataFrame()

def calculate_dynamic_expiration(df_fast, atr, adx=20):
    """
    Покращений підбір експірації: враховує і волатильність (ATR%), і силу тренду (ADX).
    - Високий ADX (>30) = швидкий спрямований рух -> менший час (5 хв)
    - Низький ADX (<18) = флет/коливання -> більший час (15-20 хв) для відпрацювання
    """
    try:
        if atr is None or pd.isna(atr) or atr == 0:
            return 10
        price = float(df_fast['close'].iloc[-1])
        atr_pct = (atr / price) * 100
        
        if adx > 30:
            return 5   # Сильний імпульсний тренд
        elif adx > 22:
            return 10  # Помірний тренд
        elif adx < 16:
            return 20  # Широкий флет, потрібен більший запас часу
        else:
            if atr_pct < 0.05:
                return 15
            elif atr_pct < 0.12:
                return 10
            else:
                return 5
    except:
        return 10

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
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    update.message.reply_text(
        "Бот Racio_1 успішно запущений! 🚀 Оберіть потрібну опцію в меню нижче:",
        reply_markup=reply_markup
    )

def train_ml_command(update, context):
    success, msg = ml_filter.train_model()
    update.message.reply_text(msg)

def ai_report_command(update, context):
    report = ml_filter.generate_strategy_report()
    update.message.reply_text(report, parse_mode="Markdown")

def handle_text_menu(update, context):
    chat_id = update.message.chat_id
    
    # 🔄 Автоперевірка завершених угод при натисканні будь-якої кнопки меню
    try:
        stats = database.evaluate_and_get_stats(fetch_yahoo_data)
        for sig in stats.get("updated_signals", []):
            if sig.get("chat_id") and sig.get("message_id") and sig.get("message_text"):
                pips_val = sig['pips']
                pips_str = f"+{pips_val}" if pips_val > 0 else str(pips_val)
                
                if sig['result'] == 'WIN':
                    res_icon = f"🏁 Результат: WIN ✅ ({pips_str} п.)"
                else:
                    res_icon = f"🏁 Результат: LOSS ❌ ({pips_str} п.)"
                
                orig_txt = sig["message_text"]
                if "🏁 Результат" not in orig_txt:
                    new_text = f"{orig_txt}\n{res_icon}"
                    bot.edit_message_text(
                        chat_id=sig["chat_id"],
                        message_id=sig["message_id"],
                        text=new_text
                    )
    except Exception as e:
        print(f"Помилка автоперевірки угод: {e}")

    text = update.message.text
    
    if text == "💵 Пари":
        pairs = list(PAIRS_MAP.items())
        keyboard = []
        for i in range(0, len(pairs), 2):
            row = [InlineKeyboardButton(pairs[i][0], callback_data=f"scan_{pairs[i][1]}")]
            if i + 1 < len(pairs):
                row.append(InlineKeyboardButton(pairs[i+1][0], callback_data=f"scan_{pairs[i+1][1]}"))
            keyboard.append(row)
            
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text("📌 Оберіть пару для технічного аналізу:", reply_markup=reply_markup)
        
    elif text == "📊 Аналіз усіх пар":
        update.message.reply_text("🔄 Сканування та ШІ-фільтрація всіх пар (1h + 15m + 5m)...")
        
        sent_signals_count = 0
        filtered_count = 0
        current_time = time.time()
        
        for name, ticker in PAIRS_MAP.items():
            try:
                if ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 300:
                    continue

                df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
                df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
                df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
                
                if df_macro.empty or df_mid.empty or df_fast.empty:
                    continue

                global_trend = analyzer.get_trend(df_macro, span_val=200)
                mid_trend = analyzer.get_trend(df_mid, span_val=50)
                
                df_indicators = analyzer.calculate_indicators(df_fast)
                sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)
                
                signal_type = sig_data.get('signal')
                if signal_type not in ['CALL', 'PUT']:
                    continue
                
                rsi = sig_data.get('rsi', 50)
                adx = sig_data.get('adx', 20)
                bb_width = float(df_indicators['bb_width'].iloc[-1]) if 'bb_width' in df_indicators.columns else 0.001
                
                win_probability = ml_filter.predict_signal_probability(rsi, adx, bb_width)
                if win_probability < 0.52:
                    filtered_count += 1
                    continue
                
                sent_signals_count += 1
                last_sent_signals[ticker] = time.time()  
                
                atr = sig_data.get('atr')
                expiration = calculate_dynamic_expiration(df_fast, atr, adx=adx)
                current_price = float(df_fast['close'].iloc[-1])
                
                icon = "🟢" if signal_type == "CALL" else "🔴"
                action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"
                
                msg_text = (
                    f"📊 {name} ({ticker})\n"
                    f"{icon} {action_text} | ⏱ {expiration} хв\n"
                    f"🎯 Ціна входу: {current_price:.5f}\n"
                    f"📈 Тренд (гл/сер): {global_trend} / {mid_trend}\n"
                    f"📉 RSI: {rsi} | ADX (5m): {adx}\n"
                    f"🧠 ШІ-успіх: {round(win_probability * 100, 1)}%\n"
                    f"💡 Причина: {sig_data.get('reason')}"
                )
                sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="Markdown")
                
                database.save_signal(
                    ticker, signal_type, current_price, expiration, chat_id, sent_msg.message_id,
                    rsi=rsi, adx=adx, bb_width=bb_width, message_text=msg_text
                )
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Помилка сканування {ticker}: {e}")
                
        bot.send_message(
            chat_id=chat_id, 
            text=f"✅ Сканування завершено!\n📤 Надіслано сильних сигналів: {sent_signals_count}\n🛡 Відсіяно ШІ як ризикові: {filtered_count}"
        )
        
    elif text == "📈 Статистика":
        update.message.reply_text("🔄 Оновлення статистики та перевірка завершених угод...")
        try:
            stats = database.evaluate_and_get_stats(fetch_yahoo_data)
            ml_filter.train_model()
            
            stats_text = (
                f"📈 **Правдива статистика трейдингу:**\n"
                f"• Успішних угод (WIN): {stats.get('wins', 0)}\n"
                f"• Усього перевірених угод: {stats.get('total', 0)}\n"
                f"• Реальний вінрейт: **{stats.get('winrate', 0)}%**\n\n"
                f"🤖 *Модель успішно перенавчена на найсвіжіших даних.*"
            )
            
            keyboard = [[InlineKeyboardButton("📊 Отримати звіт ШІ та поради", callback_data="get_ai_report")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            update.message.reply_text(stats_text, parse_mode="Markdown", reply_markup=reply_markup)
        except Exception as e:
            update.message.reply_text(f"• Помилка розрахунку статистики: {e}")

def button_callback(update, context):
    query = update.callback_query
    try:
        query.answer()
    except Exception as e:
        print(f"Попередження query.answer(): {e}")
        
    data = query.data
    
    if data == "get_ai_report":
        report = ml_filter.generate_strategy_report()
        query.edit_message_text(text=report, parse_mode="Markdown")
        return

    if data.startswith("scan_"):
        ticker = data.replace("scan_", "")
        name = next((k for k, v in PAIRS_MAP.items() if v == ticker), ticker)
        
        query.edit_message_text(
            text=f"🔄 Аналіз для **{name} ({ticker})** (1h + 15m + 5m)...",
            parse_mode="Markdown"
        )
        try:
            df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
            df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
            df_fast = fetch_yahoo_data(ticker, interval="5m", range_period="5d")
            
            if df_macro.empty or df_mid.empty or df_fast.empty:
                query.edit_message_text(text=f"❌ Помилка завантаження даних для {ticker}")
                return

            global_trend = analyzer.get_trend(df_macro, span_val=200)
            mid_trend = analyzer.get_trend(df_mid, span_val=50)
            
            df_indicators = analyzer.calculate_indicators(df_fast)
            sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)
            
            signal_type = sig_data.get('signal')
            if signal_type not in ['CALL', 'PUT']:
                query.edit_message_text(
                    text=f"ℹ️ По 🟢 **{name} ({ticker})** наразі сигнал **HOLD**. Торгові можливості відсутні.",
                    parse_Mode="Markdown"
                )
                return

            rsi = sig_data.get('rsi', 50)
            adx = sig_data.get('adx', 20)
            bb_width = float(df_indicators['bb_width'].iloc[-1]) if 'bb_width' in df_indicators.columns else 0.001
            
            win_probability = ml_filter.predict_signal_probability(rsi, adx, bb_width)
            if win_probability < 0.52:
                query.edit_message_text(
                    text=f"🛡 По **{name}** знайдено сигнал `{signal_type}`, але ШІ-фільтр відхилив його через високий ризик (Ймовірність успіху: `{round(win_probability * 100, 1)}%`).",
                    parse_mode="Markdown"
                )
                return

            atr = sig_data.get('atr')
            expiration = calculate_dynamic_expiration(df_fast, atr, adx=adx)
            current_price = float(df_fast['close'].iloc[-1])
            
            icon = "🟢" if signal_type == "CALL" else "🔴"
            action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"

            text = (
                f"📊 {name} ({ticker})\n"
                f"{icon} {action_text} | ⏱ {expiration} хв\n"
                f"🎯 Ціна входу: {current_price:.5f}\n"
                f"📈 Тренд (гл/сер): {global_trend} / {mid_trend}\n"
                f"📉 RSI: {rsi} | ADX (5m): {adx}\n"
                f"🧠 ШІ-успіх: {round(win_probability * 100, 1)}%\n"
                f"💡 Причина: {sig_data.get('reason')}"
            )
            query.edit_message_text(text=text, parse_mode="Markdown")
            
            database.save_signal(
                ticker, signal_type, current_price, expiration, query.message.chat_id, query.message.message_id,
                rsi=rsi, adx=adx, bb_width=bb_width, message_text=text
            )
            
        except Exception as e:
            query.edit_message_text(text=f"❌ Помилка обробки: {str(e)}")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("train_ml", train_ml_command))
dispatcher.add_handler(CommandHandler("ai_report", ai_report_command))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_menu))
dispatcher.add_handler(CallbackQueryHandler(button_callback))
