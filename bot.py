import time
import sqlite3
import datetime
import asyncio
import threading
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request
from telegram import ReplyKeyboardMarkup, KeyboardButton, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = "8921212255:AAE_Ypn6wCLUxVMjcrrd8TgPncuLTYQRnSg"
CHAT_ID = 859749941

SYMBOLS = {
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "NZDUSD=X": "NZD/USD", "EURGBP=X": "EUR/GBP", "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY", "AUDJPY=X": "AUD/JPY", "CADJPY=X": "CAD/JPY",
    "EURAUD=X": "EUR/AUD", "EURCAD=X": "EUR/CAD", "GBPCAD=X": "GBP/CAD",
    "GBPAUD=X": "GBP/AUD", "AUDNZD=X": "AUD/NZD", "AUDCAD=X": "AUD/CAD",
    "NZDJPY=X": "NZD/JPY", "CHFJPY=X": "CHF/JPY", "EURCHF=X": "EUR/CHF"
}

def init_db():
    conn = sqlite3.connect('stats.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            signal_type TEXT,
            entry_price REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_signal(symbol, signal_type, entry_price):
    conn = sqlite3.connect('stats.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO signals (symbol, signal_type, entry_price, status) VALUES (?, ?, ?, ?)',
                   (symbol, signal_type, entry_price, 'PENDING'))
    conn.commit()
    conn.close()

def get_stats(days=None):
    conn = sqlite3.connect('stats.db')
    cursor = conn.cursor()
    if days:
        date_limit = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute("SELECT status FROM signals WHERE timestamp >= ? AND status != 'PENDING'", (date_limit,))
    else:
        cursor.execute("SELECT status FROM signals WHERE status != 'PENDING'")
        
    rows = cursor.fetchall()
    conn.close()
    
    total = len(rows)
    wins = sum(1 for r in rows if r[0] == 'WIN')
    losses = sum(1 for r in rows if r[0] == 'LOSS')
    winrate = (wins / total * 100) if total > 0 else 0.0
    return total, wins, losses, winrate

def is_night_time():
    current_hour = datetime.datetime.now().hour
    return current_hour >= 22 or current_hour < 8

def fetch_forex_data(ticker, interval, period="30d"):
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        return None if data.empty else data
    except:
        return None

def analyze_forex_symbol(ticker_code, display_name):
    df_m5_raw = fetch_forex_data(ticker_code, "5m", period="10d")
    df_m15 = fetch_forex_data(ticker_code, "15m", period="15d")
    df_h1 = fetch_forex_data(ticker_code, "1h", period="30d")
    df_h4_raw = fetch_forex_data(ticker_code, "1h", period="60d")
    
    if df_m5_raw is None or df_m15 is None or df_h1 is None or df_h4_raw is None:
        return None
        
    df_m5 = df_m5_raw.tail(500)

    if len(df_m5) < 50 or len(df_m15) < 50 or len(df_h1) < 50:
        return None

    df_h4 = df_h4_raw.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    if len(df_h4) < 30:
        return None

    current_price = df_m5['Close'].iloc[-1]
    ema50_h4 = df_h4['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    h4_bull = current_price > ema50_h4
    h4_bear = current_price < ema50_h4

    ema50_h1 = df_h1['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    h1_bull = current_price > ema50_h1
    h1_bear = current_price < ema50_h1

    ema50_m15 = df_m15['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    m15_bull = current_price > ema50_m15
    m15_bear = current_price < ema50_m15

    delta = df_m5['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = round((100 - (100 / (1 + rs))).iloc[-1], 1)

    sma20 = df_m5['Close'].rolling(20).mean().iloc[-1]
    std20 = df_m5['Close'].rolling(20).std().iloc[-1]
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)

    vol_ma = df_m5['Volume'].rolling(20).mean().iloc[-1]
    vol_surge = df_m5['Volume'].iloc[-1] >= vol_ma if vol_ma > 0 else True

    min_support = df_m5['Low'].tail(500).min()
    max_resistance = df_m5['High'].tail(500).max()
    near_support = abs(current_price - min_support) / current_price <= 0.003
    near_resistance = abs(current_price - max_resistance) / current_price <= 0.003

    score_call = sum([h4_bull, h1_bull, m15_bull, rsi <= 25 or current_price <= lower_bb, near_support, vol_surge])
    score_put = sum([h4_bear, h1_bear, m15_bear, rsi >= 75 or current_price >= upper_bb, near_resistance, vol_surge])

    max_score = max(score_call, score_put)
    signal_type = "CALL (ВГОРУ)" if score_call >= score_put else "PUT (ВНИЗ)"

    return {
        "symbol": display_name, "type": signal_type,
        "price": round(current_price, 5), "rsi": rsi,
        "score": max_score, "full_signal": max_score >= 5
    }

async def background_scanner(application):
    await asyncio.sleep(15)
    while True:
        try:
            if not is_night_time():
                for ticker, name in SYMBOLS.items():
                    res = analyze_forex_symbol(ticker, name)
                    if res and res["full_signal"]:
                        msg = (
                            f"🚀 **АВТО-СИГНАЛ (H1/H4 Trend): {res['type']}**\n\n"
                            f"🔹 **Пара:** `{res['symbol']}`\n"
                            f"💰 **Ціна входу:** `{res['price']}`\n"
                            f"⏳ **Експірація:** `10–15 хвилин`\n\n"
                            f"🎯 **Мультитаймфрейм підтверджено! ✅**"
                        )
                        await application.bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="Markdown")
                        save_signal(res['symbol'], res['type'], res['price'])
                        await asyncio.sleep(60)
            await asyncio.sleep(300)
        except Exception as e:
            print(f"Помилка сканера: {e}")
            await asyncio.sleep(60)

def main_keyboard():
    keyboard = [
        [KeyboardButton("📅 День"), KeyboardButton("🗓 Тиждень"), KeyboardButton("♾ Увесь час")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Автоматичного бота запущено у хмарі!", reply_markup=main_keyboard())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    if text == "📅 День":
        t, w, l, wr = get_stats(1)
        await update.message.reply_text(f"📅 День: Всього: {t} | Win Rate: {wr:.1f}%")
    elif "🗓 Тиждень" in text:
        t, w, l, wr = get_stats(7)
        await update.message.reply_text(f"🗓 Тиждень: Всього: {t} | Win Rate: {wr:.1f}%")
    elif "♾ Увесь час" in text:
        t, w, l, wr = get_stats(None)
        await update.message.reply_text(f"♾ За весь час: Всього: {t} | Win Rate: {wr:.1f}%")

app = Flask(__name__)
application = Application.builder().token(BOT_TOKEN).build()

application.add_handler(CommandHandler("start", start_cmd))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

@app.route('/')
def home():
    return "Bot is alive and running!"

@app.route(f'/{BOT_TOKEN}', methods=['POST'])
def webhook():
    update = Update.de_json(request.get_json(force=True), application.bot)
    asyncio.run(application.process_update(update))
    return 'ok'

def run_scanner():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(background_scanner(application))

if __name__ == '__main__':
    init_db()
    
    webhook_url = f"https://racio-1bot.onrender.com/{BOT_TOKEN}"
    requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook?url={webhook_url}")
    
    t = threading.Thread(target=run_scanner, daemon=True)
    t.start()
    
    app.run(host='0.0.0.0', port=10000)
