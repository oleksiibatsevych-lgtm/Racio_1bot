import sqlite3
import datetime
import asyncio
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

BOT_TOKEN = "8921212255:AAE_Ypn6wCLUxVMjcrrd8TgPncuLTYQRnSg"

SYMBOLS = {
    "GBPUSD=X": "GBP/USD", "EURUSD=X": "EUR/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "GBPJPY=X": "GBP/JPY", "EURJPY=X": "EUR/JPY", "AUDCHF=X": "AUD/CHF",
    "AUDJPY=X": "AUD/JPY", "CADCHF=X": "CAD/CHF", "CADJPY=X": "CAD/JPY",
    "CHFJPY=X": "CHF/JPY", "EURAUD=X": "EUR/AUD", "EURCAD=X": "EUR/CAD",
    "EURCHF=X": "EUR/CHF", "EURGBP=X": "EUR/GBP", "GBPCAD=X": "GBP/CAD",
    "GBPCHF=X": "GBP/CHF", "AUDCAD=X": "AUD/CAD", "GBPAUD=X": "GBP/AUD"
}

# ==================== РОБОТА З БД (SQLITE) ====================
def init_db():
    conn = sqlite3.connect('bot_stats.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            signal_type TEXT,
            price REAL,
            confidence TEXT,
            status TEXT DEFAULT 'PENDING',
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_signal_to_db(user_id, symbol, signal_type, price, confidence):
    try:
        conn = sqlite3.connect('bot_stats.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signals (user_id, symbol, signal_type, price, confidence, status, timestamp)
            VALUES (?, ?, ?, ?, ?, 'PENDING', ?)
        ''', (user_id, symbol, signal_type, price, confidence, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        signal_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return signal_id
    except Exception as e:
        print(f"Помилка збереження в БД: {e}")
        return None

def update_signal_status(signal_id, status):
    try:
        conn = sqlite3.connect('bot_stats.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE signals SET status = ? WHERE id = ?', (status, signal_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Помилка оновлення статусу: {e}")

# ==================== АНАЛІЗ РИНКУ ====================
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

def calculate_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def analyze_symbol(ticker_code, display_name):
    df_m5 = fetch_forex_data(ticker_code, "5m", period="5d")
    df_h1 = fetch_forex_data(ticker_code, "1h", period="30d")
    
    if df_m5 is None or df_h1 is None or len(df_m5) < 50 or len(df_h1) < 50:
        return None

    current_price = df_m5['Close'].iloc[-1]
    
    h1_ema200 = df_h1['Close'].ewm(span=200, adjust=False).mean().iloc[-1]
    global_bullish = current_price > h1_ema200

    delta = df_m5['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = round((100 - (100 / (1 + (gain / loss)))).iloc[-1], 1)
    
    macd_val, signal_val = calculate_macd(df_m5)
    macd_bullish = macd_val > signal_val

    current_vol = df_m5['Volume'].iloc[-1] if 'Volume' in df_m5.columns else 0
    avg_vol = df_m5['Volume'].rolling(20).mean().iloc[-1] if 'Volume' in df_m5.columns else 1
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

    atr = calculate_atr(df_m5)

    if atr > 0.10 and vol_ratio > 1.3:
        expiration = "3 хвилини (Сплеск об'єму + Імпульс)"
    elif atr > 0.07 or vol_ratio > 1.1:
        expiration = "5 хвилин (Активна фаза руху)"
    elif atr > 0.04:
        expiration = "10 - 15 хвилин (Стандартний робочий рух)"
    else:
        expiration = "20 - 30 хвилин (Низький об'єм / Флєт)"

    if global_bullish and macd_bullish and rsi < 60:
        signal_type = "CALL (📈 ВГОРУ)"
        confidence = "Висока (Тренд H1 + MACD + RSI)" if rsi < 45 else "Помірна (Тренд + MACD)"
    elif not global_bullish and not macd_bullish and rsi > 40:
        signal_type = "PUT (📉 ВНИЗ)"
        confidence = "Висока (Тренд H1 + MACD + RSI)" if rsi > 55 else "Помірна (Тренд + MACD)"
    elif rsi <= 30:
        signal_type = "CALL (📈 ВГОРУ)"
        confidence = "Висока (Зона глибокої перепроданості RSI)"
    elif rsi >= 70:
        signal_type = "PUT (📉 ВНИЗ)"
        confidence = "Висока (Зона глибокої перекупленості RSI)"
    else:
        if macd_bullish:
            signal_type = "CALL (📈 ВГОРУ)"
            confidence = "Обережна (За імпульсом MACD)"
        else:
            signal_type = "PUT (📉 ВНИЗ)"
            confidence = "Обережна (За імпульсом MACD)"

    return {
        "symbol": display_name, "type": signal_type, "price": round(current_price, 5),
        "rsi": rsi, "atr": round(atr, 3), "vol_ratio": round(vol_ratio, 2),
        "confidence": confidence, "expiration": expiration
    }

# ==================== TELEGRAM HANDLERS ====================
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
        "👋 **Оберіть пару для аналізу:**\n"
        "📊 Використовуйте /stats для перегляду успішності та статистики по кожній парі.",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    try:
        conn = sqlite3.connect('bot_stats.db')
        cursor = conn.cursor()
        
        # Загальна статистика
        cursor.execute("SELECT COUNT(*) FROM signals WHERE user_id = ? AND status != 'PENDING'", (user_id,))
        total_finished = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signals WHERE user_id = ? AND status = 'WIN'", (user_id,))
        wins = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM signals WHERE user_id = ? AND status = 'LOSS'", (user_id,))
        losses = cursor.fetchone()[0]
        
        total_winrate = round((wins / total_finished * 100), 1) if total_finished > 0 else 0

        # Статистика згрупована по валютних парах
        cursor.execute('''
            SELECT symbol, 
                   COUNT(*) as total, 
                   SUM(CASE WHEN status = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN status = 'LOSS' THEN 1 ELSE 0 END) as losses
            FROM signals 
            WHERE user_id = ? AND status != 'PENDING' 
            GROUP BY symbol
            ORDER BY total DESC
        ''', (user_id,))
        pair_rows = cursor.fetchall()
        conn.close()
        
        msg = (
            f"📊 **ЗАГАЛЬНА СТАТИСТИКА УГОД**\n\n"
            f"🎯 Всього закрито угод: `{total_finished}`\n"
            f"✅ Успішних (Плюс): `{wins}`\n"
            f"❌ Неуспішних (Мінус): `{losses}`\n"
            f"📈 **Загальний Winrate:** `{total_winrate}%`\n\n"
            f"💱 **СТАТИСТИКА ПО ВАЛЮТНИХ ПАРАХ:**\n"
        )
        
        if not pair_rows:
            msg += "_Статистика по парах з'явиться після того, як ви закриєте угоди й позначите їх результати._"
        else:
            for row in pair_rows:
                p_symbol, p_total, p_wins, p_losses = row
                p_winrate = round((p_wins / p_total * 100), 1) if p_total > 0 else 0
                msg += f"🔹 **{p_symbol}**: угод: `{p_total}` | Плюс: `{p_wins}` | Мінус: `{p_losses}` | Winrate: **`{p_winrate}%`**\n"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка завантаження статистики: {e}")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith("win_") or data.startswith("loss_"):
        status = "WIN" if data.startswith("win_") else "LOSS"
        signal_id = data.split("_")[1]
        
        update_signal_status(signal_id, status)
        
        status_text = "🟢 Зараховано як ПЛЮС (Win)" if status == "WIN" else "🔴 Зараховано як МІНУС (Loss)"
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton(f"Статус: {status_text}", callback_data="none")
        ], [
            InlineKeyboardButton("🔙 Назад до списку пар", callback_data="back_to_menu")
        ]]))
        return

    ticker = data
    display_name = SYMBOLS.get(ticker, ticker)
    user_id = query.from_user.id
    
    await query.edit_message_text(f"⏳ Розраховую об'єми та індикатори для `{display_name}`...", parse_mode="Markdown")
    
    res = analyze_symbol(ticker, display_name)
    
    if not res:
        await query.edit_message_text(f"❌ Не вдалося завантажити дані для `{display_name}`.", parse_mode="Markdown")
        return

    signal_id = save_signal_to_db(user_id, res['symbol'], res['type'], res['price'], res['confidence'])

    msg = (
        f"🎯 **АНАЛІЗ ПАРИ: {res['symbol']}**\n\n"
        f"🔹 **Рекомендація:** `{res['type']}`\n"
        f"💰 **Ціна входу:** `{res['price']}`\n"
        f"⏱ **Рекомендована експірація:** `{res['expiration']}`\n"
        f"📊 **Рівень RSI (M5):** `{res['rsi']}`\n"
        f"📈 **Якість сигналу:** `{res['confidence']}`\n"
        f"📦 **Співвідношення об'єму:** `{res['vol_ratio']}x від середнього`\n"
        f"📏 **Волатильність (ATR):** `{res['atr']}%`\n\n"
        f"👇 **Позначте результат угоди після її завершення:**"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("✅ Плюс (Win)", callback_data=f"win_{signal_id}"),
            InlineKeyboardButton("❌ Мінус (Loss)", callback_data=f"loss_{signal_id}")
        ],
        [
            InlineKeyboardButton("🔙 Назад до списку пар", callback_data="back_to_menu")
        ]
    ]
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

# ==================== FLASK & TELEGRAM SETUP ====================
app = Flask(__name__)
application = Application.builder().token(BOT_TOKEN).build()

application.add_handler(CommandHandler("start", start_cmd))
application.add_handler(CommandHandler("stats", stats_cmd))
application.add_handler(CallbackQueryHandler(menu_handler, pattern="^back_to_menu$"))
application.add_handler(CallbackQueryHandler(button_handler))

@app.route('/')
def home():
    return "Bot with Detailed Pair Statistics is running!"

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
    init_db()
    webhook_url = f"https://racio-1bot.onrender.com/{BOT_TOKEN}"
    requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook?url={webhook_url}")
    app.run(host='0.0.0.0', port=10000)
