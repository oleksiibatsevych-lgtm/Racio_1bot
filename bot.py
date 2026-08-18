import os
import io
import time
import requests
import pandas as pd
from flask import Flask, request
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CallbackQueryHandler, CommandHandler, MessageHandler, Filters

from config import TELEGRAM_TOKEN, PAIRS_MAP
from indicators import AdaptiveTechnicalAnalysis
import database

app = Flask(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

analyzer = AdaptiveTechnicalAnalysis()

# Сесія з повними заголовками реального браузера для обходу захисту Yahoo
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9"
})

def fetch_yahoo_data(ticker, interval="1h", range_period="60d"):
    """Пряме завантаження історичних даних з Yahoo API з назвами колонок у нижньому регістрі"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "interval": interval,
            "range": range_period,
            "includeAdjustedClose": "true"
        }
        
        session.get("https://finance.yahoo.com", timeout=5)
        response = session.get(url, params=params, timeout=10)
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

def calculate_dynamic_expiration(df_mid, atr):
    """Розрахунок динамічної експірації на основі волатильності (ATR) та ціни"""
    try:
        if atr is None or pd.isna(atr) or atr == 0:
            return 5
        price = df_mid['close'].iloc[-1]
        atr_pct = (atr / price) * 100
        # Низька волатильність -> більша експірація, висока -> менша
        if atr_pct < 0.05:
            return 15
        elif atr_pct < 0.15:
            return 10
        else:
            return 5
    except:
        return 5

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

def handle_text_menu(update, context):
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
        chat_id = update.message.chat_id
        update.message.reply_text("🔄 Запущено сканування всіх валютних пар (HOLD сигнали пропускаються)...")
        
        sent_signals_count = 0
        for name, ticker in PAIRS_MAP.items():
            try:
                df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
                df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
                
                if df_macro.empty or df_mid.empty:
                    continue

                global_trend = analyzer.get_trend(df_macro, span_val=200)
                mid_trend = analyzer.get_trend(df_mid, span_val=50)
                df_indicators = analyzer.calculate_indicators(df_mid)
                sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)
                
                signal_type = sig_data.get('signal')
                # Повністю пропускаємо HOLD
                if signal_type not in ['CALL', 'PUT']:
                    continue
                
                sent_signals_count += 1
                atr = sig_data.get('atr')
                expiration = calculate_dynamic_expiration(df_mid, atr)
                current_price = df_mid['close'].iloc[-1]
                
                # Зберігаємо сигнал для статистики
                database.save_signal(ticker, signal_type, current_price, expiration)

                icon = "🟢" if signal_type == "CALL" else "🔴"
                action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"
                
                msg = (
                    f"📊 **Сканування: {name} ({ticker})**\n"
                    f"{icon} Сигнал: **{action_text}**\n"
                    f"⏱ Експірація: **{expiration} хв**\n"
                    f"• Тренд (глоб/сер): {global_trend} / {mid_trend}\n"
                    f"• RSI: {sig_data.get('rsi')} | ADX: {sig_data.get('adx')}\n"
                    f"• Причина: {sig_data.get('reason')}"
                )
                bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Помилка сканування {ticker}: {e}")
                
        bot.send_message(chat_id=chat_id, text=f"✅ Сканування завершено! Знайдено активних сигналів: {sent_signals_count}")
        
    elif text == "📈 Статистика":
        update.message.reply_text("🔄 Оновлення та перевірка правдивої статистики за результатами ринку...")
        try:
            stats = database.evaluate_and_get_stats(fetch_yahoo_data)
            stats_text = (
                f"📈 **Правдива статистика трейдингу:**\n"
                f"• Успішних угод (WIN): {stats.get('wins', 0)}\n"
                f"• Усього перевірених угод: {stats.get('total', 0)}\n"
                f"• Реальний вінрейт: **{stats.get('winrate', 0)}%**"
            )
        except Exception as e:
            stats_text = f"• Помилка розрахунку статистики: {e}"
        
        update.message.reply_text(stats_text, parse_mode="Markdown")

def button_callback(update, context):
    query = update.callback_query
    try:
        query.answer()
    except Exception as e:
        print(f"Попередження query.answer(): {e}")
        
    data = query.data
    
    if data.startswith("scan_"):
        ticker = data.replace("scan_", "")
        query.edit_message_text(
            text=f"🔄 Завантаження даних та розрахунок індикаторів для **{ticker}**...",
            parse_mode="Markdown"
        )
        try:
            df_macro = fetch_yahoo_data(ticker, interval="1h", range_period="60d")
            df_mid = fetch_yahoo_data(ticker, interval="15m", range_period="10d")
            
            if df_macro.empty or df_mid.empty:
                query.edit_message_text(text=f"❌ Помилка: не вдалося завантажити дані для {ticker}")
                return

            global_trend = analyzer.get_trend(df_macro, span_val=200)
            mid_trend = analyzer.get_trend(df_mid, span_val=50)
            df_indicators = analyzer.calculate_indicators(df_mid)
            sig_data = analyzer.generate_signal(df_indicators, global_trend, mid_trend, ticker)
            
            signal_type = sig_data.get('signal')
            if signal_type not in ['CALL', 'PUT']:
                query.edit_message_text(
                    text=f"ℹ️ По 🟢 **{ticker}** наразі сигнал **HOLD** (утримання). Торгові можливості відсутні.",
                    parse_mode="Markdown"
                )
                return

            atr = sig_data.get('atr')
            expiration = calculate_dynamic_expiration(df_mid, atr)
            current_price = df_mid['close'].iloc[-1]
            
            # Зберігаємо для статистики
            database.save_signal(ticker, signal_type, current_price, expiration)

            icon = "🟢" if signal_type == "CALL" else "🔴"
            action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"

            text = (
                f"📈 **Технічний аналіз: {ticker}**\n"
                f"{icon} Сигнал: **{action_text}**\n"
                f"⏱ Динамічна експірація: **{expiration} хв**\n"
                f"• Глобальний тренд: {global_trend}\n"
                f"• Середній тренд: {mid_trend}\n"
                f"• RSI: {sig_data.get('rsi')} | ADX: {sig_data.get('adx')} | ATR: {sig_data.get('atr')}\n"
                f"• Локальний тренд: {sig_data.get('local_trend')}\n"
                f"• Причина: {sig_data.get('reason')}"
            )
            query.edit_message_text(text=text, parse_mode="Markdown")
        except Exception as e:
            query.edit_message_text(text=f"❌ Помилка обробки: {str(e)}")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_menu))
dispatcher.add_handler(CallbackQueryHandler(button_callback))
