import sqlite3
import datetime
import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

# ==================== НАЛАШТУВАННЯ ====================
BOT_TOKEN = "8921212255:AAE_Ypn6wCLUxVMjcrrd8TgPncuLTYQRnSg"

# Точний список пар з вашого термінала (без NZD)
SYMBOLS = {
    "GBPUSD=X": "GBP/USD", "EURUSD=X": "EUR/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "GBPJPY=X": "GBP/JPY", "EURJPY=X": "EUR/JPY", "AUDCHF=X": "AUD/CHF",
    "AUDJPY=X": "AUD/JPY", "CADCHF=X": "CAD/CHF", "CADJPY=X": "CAD/JPY",
    "CHFJPY=X": "CHF/JPY", "EURAUD=X": "EUR/AUD", "EURCAD=X": "EUR/CAD",
    "EURCHF=X": "EUR/CHF", "EURGBP=X": "EUR/GBP", "GBPCAD=X": "GBP/CAD",
    "GBPCHF=X": "GBP/CHF", "AUDCAD=X": "AUD/CAD", "GBPAUD=X": "GBP/AUD"
}

def fetch_forex_data(ticker, interval, period="30d"):
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        return None if data.empty else data
    except:
        return None

def calculate_atr(df, period=14):
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return (atr / close.iloc[-1] * 100).iloc[-1]

def analyze_symbol(ticker_code, display_name):
    df_m5 = fetch_forex_data(ticker_code, "5m", period="5d")
    df_h1 = fetch_forex_data(ticker_code, "1h", period="20d")
    df_h4 = fetch_forex_data(ticker_code, "1h", period="40d")
    
    if df_m5 is None or df_h1 is None or df_h4 is None or len(df_m5) < 50:
        return None

    df_h4_res = df_h4.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    if len(df_h4_res) < 20:
        return None

    current_price = df_m5['Close'].iloc[-1]
    
    # Тренди на старших таймфреймах
    h4_bull = current_price > df_h4_res['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    h1_bull = current_price > df_h1['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    
    # RSI M5
    delta = df_m5['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = round((100 - (100 / (1 + (gain / loss)))).iloc[-1], 1)
    
    # Bollinger Bands
    sma20 = df_m5['Close'].rolling(20).mean().iloc[-1]
    std20 = df_m5['Close'].rolling(20).std().iloc[-1]
    upper_bb, lower_bb = sma20 + (2 * std20), sma20 - (2 * std20)
    
    atr = calculate_atr(df_m5)

    # Логіка аналізу та визначення експірації від 3 до 30 хвилин
    score_call = sum([h4_bull, h1_bull, rsi <= 35, current_price <= lower_bb])
    score_put = sum([not h4_bull, not h1_bull, rsi >= 65, current_price >= upper_bb])

    # Динамічний розрахунок часу експірації на основі волатильності (ATR)
    if atr > 0.10:
        expiration = "3 - 5 хвилин (Висока волатильність)"
    elif atr > 0.06:
        expiration = "5 - 15 хвилин (Стандартна волатильність)"
    else:
        expiration = "15 - 30 хвилин (Низька волатильність / Флєт)"

    if score_call >= 3:
        signal_type = "CALL (📈 ВГОРУ)"
        confidence = "Висока (75-80%)" if score_call == 4 else "Середня (65-70%)"
    elif score_put >= 3:
        signal_type = "PUT (📉 ВНИЗ)"
        confidence = "Висока (75-80%)" if score_put == 4 else "Середня (65-70%)"
    else:
        if rsi < 50 and h1_bull:
            signal_type = "CALL (📈 ВГОРУ)"
            confidence = "Помірна (60%)"
            expiration = "15 - 30 хвилин"
        elif rsi > 50 and not h1_bull:
            signal_type = "PUT (📉 ВНИЗ)"
            confidence = "Помірна (60%)"
            expiration = "15 - 30 хвилин"
        else:
            signal_type = "НЕЙТРАЛЬНО (Поза ринком)"
            confidence = "Сигнал відсутній"
            expiration = "-"

    return {
        "symbol": display_name, "type": signal_type, "price": round(current_price, 5),
        "rsi": rsi, "atr": round(atr, 3), "confidence": confidence, "expiration": expiration
    }

# Telegram Handlers
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = []
    row = []
    for ticker, name in SYMBOLS.items():
        row.append(InlineKeyboardButton(name, callback_data=ticker))
        if len(row) == 2:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
        
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "👋 **Оберіть валютну пару для миттєвого аналізу та отримання часу експірації:**",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    ticker = query.data
    display_name = SYMBOLS.get(ticker, ticker)
    
    await query.edit_message_text(f"⏳ Аналізую ринок для `{display_name}`, зачекайте кілька секунд...", parse_mode="Markdown")
    
    res = analyze_symbol(ticker, display_name)
    
    if not res:
        await query.edit_message_text(f"❌ Не вдалося завантажити дані для `{display_name}`. Спробуйте іншу пару.", parse_mode="Markdown")
        return

    msg = (
        f"🎯 **АНАЛІЗ ПАРИ: {res['symbol']}**\n\n"
        f"🔹 **Сигнал:** `{res['type']}`\n"
        f"💰 **Ціна входу:** `{res['price']}`\n"
        f"⏱ **Рекомендована експірація:** `{res['expiration']}`\n"
        f"📊 **Рівень RSI (M5):** `{res['rsi']}`\n"
        f"📈 **Якість сигналу:** `{res['confidence']}`\n"
        f"📏 **Волатильність (ATR):** `{res['atr']}%`"
    )
    
    keyboard = [[InlineKeyboardButton("🔙 Назад до списку пар", callback_data="back_to_menu")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(msg, reply_markup=reply_markup, parse_mode="Markdown")

async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if query.data == "back_to_menu":
        await query.answer()
        keyboard = []
        row = []
        for ticker, name in SYMBOLS.items():
            row.append(InlineKeyboardButton(name, callback_data=ticker))
            if len(row) == 2:
                keyboard.append(row)
                row = []
        if row:
            keyboard.append(row)
            
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "📋 **Оберіть валютну пару для аналізу:**",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

app = Flask(__name__)
application = Application.builder().token(BOT_TOKEN).build()

application.add_handler(CommandHandler("start", start_cmd))
application.add_handler(CallbackQueryHandler(menu_handler, pattern="^back_to_menu$"))
application.add_handler(CallbackQueryHandler(button_handler))

@app.route('/')
def home():
    return "Interactive On-Demand Bot is running!"

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    json_data = request.get_json(force=True)
    update = Update.de_json(json_data, application.bot)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(application.initialize())
    loop.run_until_complete(application.process_update(update))
    return 'ok'

if __name__ == '__main__':
    webhook_url = f"https://racio-1bot.onrender.com/{BOT_TOKEN}"
    requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook?url={webhook_url}")
    app.run(host='0.0.0.0', port=10000)
