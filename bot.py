import os
import time
import sqlite3
import datetime
import logging
import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from telegram import ReplyKeyboardMarkup, KeyboardButton, Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN", "8921212255:AAE_Ypn6wCLUxVMjcrrd8TgPncuLTYQRnSg")

SYMBOLS = {
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "EURGBP=X": "EUR/GBP", "EURJPY=X": "EUR/JPY", "GBPJPY=X": "GBP/JPY", 
    "CADJPY=X": "CAD/JPY", "EURCAD=X": "EUR/CAD", "GBPCAD=X": "GBP/CAD", 
    "CHFJPY=X": "CHF/JPY", "EURCHF=X": "EUR/CHF"
}

NIGHT_MODE = True
SUBSCRIBED_CHATS = set()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

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

def save_signal_to_db(symbol, signal_type, price):
    conn = sqlite3.connect('stats.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO signals (symbol, signal_type, entry_price, status)
        VALUES (?, ?, ?, 'PENDING')
    ''', (symbol, signal_type, price))
    conn.commit()
    conn.close()

async def check_pending_signals():
    """Фонова перевірка результатів через 15 хвилин для підрахунку статистики (Win/Loss)"""
    while True:
        await asyncio.sleep(60)
        try:
            conn = sqlite3.connect('stats.db')
            cursor = conn.cursor()
            cursor.execute("SELECT id, symbol, signal_type, entry_price FROM signals WHERE status = 'PENDING' AND timestamp <= datetime('now', '-15 minutes')")
            rows = cursor.fetchall()
            
            for row in rows:
                sig_id, symbol, sig_type, entry_price = row
                ticker_code = next((k for k, v in SYMBOLS.items() if v == symbol), None)
                if ticker_code:
                    df = fetch_forex_data(ticker_code, "5m", period="1d")
                    if df is not None and not df.empty:
                        current_price = df['Close'].iloc[-1]
                        if "ВГОРУ" in sig_type:
                            res_status = 'WIN' if current_price > entry_price else 'LOSS'
                        else:
                            res_status = 'WIN' if current_price < entry_price else 'LOSS'
                        
                        cursor.execute("UPDATE signals SET status = ? WHERE id = ?", (res_status, sig_id))
                        conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error checking pending signals: {e}")

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

def fetch_forex_data(ticker, interval, period="5d"):
    try:
        if interval in ["1h", "60m", "1d"]:
            period = "60d"
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        return None if data.empty else data
    except:
        return None

def analyze_forex_symbol(ticker_code, display_name):
    df_m5 = fetch_forex_data(ticker_code, "5m")
    df_m15 = fetch_forex_data(ticker_code, "15m")
    df_h1 = fetch_forex_data(ticker_code, "1h")
    df_h4_raw = fetch_forex_data(ticker_code, "1h", period="60d")
    
    if df_m5 is None or df_m15 is None or df_h1 is None or df_h4_raw is None:
        return None
        
    if len(df_m5) < 50 or len(df_m15) < 50 or len(df_h1) < 50:
        return None

    df_h4 = df_h4_raw.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    if len(df_h4) < 30:
        return None

    current_price = df_m5['Close'].iloc[-1]
    
    # Трендові фільтри по EMA 50
    ema50_h4 = df_h4['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    h4_bull = current_price > ema50_h4
    h4_bear = current_price < ema50_h4

    ema50_h1 = df_h1['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    h1_bull = current_price > ema50_h1
    h1_bear = current_price < ema50_h1

    ema50_m15 = df_m15['Close'].ewm(span=50, adjust=False).mean().iloc[-1]
    m15_bull = current_price > ema50_m15
    m15_bear = current_price < ema50_m15

    # RSI (14)
    delta = df_m5['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = round((100 - (100 / (1 + rs))).iloc[-1], 1)

    # Bollinger Bands
    sma20 = df_m5['Close'].rolling(20).mean().iloc[-1]
    std20 = df_m5['Close'].rolling(20).std().iloc[-1]
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)

    # Об'єми
    vol_ma = df_m5['Volume'].rolling(20).mean().iloc[-1]
    vol_surge = df_m5['Volume'].iloc[-1] >= vol_ma if vol_ma > 0 else True

    # Рівні підтримки/опору за 100 свічок
    min_support = df_m5['Low'].tail(100).min()
    max_resistance = df_m5['High'].tail(100).max()
    near_support = abs(current_price - min_support) / current_price <= 0.003
    near_resistance = abs(current_price - max_resistance) / current_price <= 0.003

    # Підрахунок балів
    score_call = sum([h4_bull, h1_bull, m15_bull, rsi <= 30, current_price <= lower_bb, near_support, vol_surge])
    score_put = sum([h4_bear, h1_bear, m15_bear, rsi >= 70, current_price >= upper_bb, near_resistance, vol_surge])

    max_score = max(score_call, score_put)
    is_call = score_call >= score_put

    if is_call:
        signal_header = "🟢🚀 ТОП-СИГНАЛ (H1/H4 Trend): ВГОРУ"
        signal_text_type = "ВГОРУ"
    else:
        signal_header = "🔴⚓ ТОП-СИГНАЛ (H1/H4 Trend): ВНИЗ"
        signal_text_type = "ВНИЗ"

    # Суворі фільтри якості: сигнал пройде тільки при високій кількості балів та екстремальному RSI
    is_valid_call = (score_call >= 6) and (rsi <= 35)
    is_valid_put = (score_put >= 6) and (rsi >= 65)
    strict_full_signal = is_valid_call if is_call else is_valid_put

    return {
        "symbol": display_name, "header": signal_header, "type_str": signal_text_type,
        "price": round(current_price, 5), "rsi": rsi,
        "score": max_score, "full_signal": strict_full_signal
    }

def main_keyboard():
    night_status = "УВІМК 🟢" if NIGHT_MODE else "ВИМК 🔴"
    keyboard = [
        [KeyboardButton("🔍 СКАНУВАТИ РИНОК ЗАРАЗ")],
        [KeyboardButton("📅 День"), KeyboardButton("🗓 Тиждень"), KeyboardButton("♾ Увесь час")],
        [KeyboardButton(f"🌙 Нічний режим: {night_status}")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    SUBSCRIBED_CHATS.add(chat_id)
    await update.message.reply_text("👋 Бот запущено! Фоновий сканер активний і надсилатиме лише найкращі сигнали.", reply_markup=main_keyboard())

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global NIGHT_MODE
    if not update.message or not update.message.text:
        return
    chat_id = update.effective_chat.id
    SUBSCRIBED_CHATS.add(chat_id)
    text = update.message.text

    if "СКАНУВАТИ" in text:
        await update.message.reply_text(f"🔎 Примусове сканування {len(SYMBOLS)} пар...")
        found = 0
        for ticker, name in SYMBOLS.items():
            res = analyze_forex_symbol(ticker, name)
            if res and res["full_signal"]:
                found += 1
                save_signal_to_db(res['symbol'], res['type_str'], res['price'])
                msg = (
                    f"*{res['header']}*\n\n"
                    f"🔹 Пара: `{res['symbol']}`\n"
                    f"💰 Ціна входу: `{res['price']}`\n"
                    f"⏳ Експірація: 10–15 хвилин\n\n"
                    f"🎯 Мультитаймфрейм підтверджено! ✅"
                )
                await update.message.reply_text(msg, parse_mode="Markdown")
        if found == 0:
            await update.message.reply_text("😴 Зараз ідеальних сигналів за суворими критеріями немає.")
    elif "День" in text:
        t, w, l, wr = get_stats(1)
        await update.message.reply_text(f"📅 День: Всього: {t} (Виграшів: {w}, Програшів: {l}) | Win Rate: {wr:.1f}%")
    elif "Тиждень" in text:
        t, w, l, wr = get_stats(7)
        await update.message.reply_text(f"🗓 Тиждень: Всього: {t} (Виграшів: {w}, Програшів: {l}) | Win Rate: {wr:.1f}%")
    elif "Увесь час" in text:
        t, w, l, wr = get_stats(None)
        await update.message.reply_text(f"♾ За весь час: Всього: {t} (Виграшів: {w}, Програшів: {l}) | Win Rate: {wr:.1f}%")
    elif "Нічний режим" in text:
        NIGHT_MODE = not NIGHT_MODE
        await update.message.reply_text("🌙 Режим змінено!", reply_markup=main_keyboard())
    else:
        await update.message.reply_text("Скористайтеся кнопками нижче 👇", reply_markup=main_keyboard())

async def background_scanner(application):
    await asyncio.sleep(15)
    while True:
        try:
            if SUBSCRIBED_CHATS:
                for ticker, name in SYMBOLS.items():
                    res = analyze_forex_symbol(ticker, name)
                    if res and res["full_signal"]:
                        save_signal_to_db(res['symbol'], res['type_str'], res['price'])
                        msg = (
                            f"*{res['header']}*\n\n"
                            f"🔹 Пара: `{res['symbol']}`\n"
                            f"💰 Ціна входу: `{res['price']}`\n"
                            f"⏳ Експірація: 10–15 хвилин\n\n"
                            f"🎯 Мультитаймфрейм підтверджено! ✅"
                        )
                        for chat_id in SUBSCRIBED_CHATS:
                            try:
                                await application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                            except:
                                pass
                    await asyncio.sleep(15)
        except Exception as e:
            logging.error(f"Background scanner error: {e}")
        
        # Пауза перед наступним колом сканування ринку
        await asyncio.sleep(300)

if __name__ == '__main__':
    init_db()
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    loop = asyncio.get_event_loop()
    loop.create_task(background_scanner(app))
    loop.create_task(check_pending_signals())
    
    app.run_polling()
