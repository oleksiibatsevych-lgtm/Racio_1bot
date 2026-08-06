import sqlite3
import datetime
import asyncio
import requests
import io
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters, ContextTypes

BOT_TOKEN = "8921212255:AAE_Ypn6wCLUxVMjcrrd8TgPncuLTYQRnSg"

SYMBOLS = {
    "GBP/USD": "GBPUSD=X", "EUR/USD": "EURUSD=X", "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
    "GBP/JPY": "GBPJPY=X", "EUR/JPY": "EURJPY=X", "AUD/CHF": "AUDCHF=X",
    "AUD/JPY": "AUDJPY=X", "CAD/CHF": "CADCHF=X", "CAD/JPY": "CADJPY=X",
    "CHF/JPY": "CHFJPY=X", "EUR/AUD": "EURAUD=X", "EUR/CAD": "EURCAD=X",
    "EUR/CHF": "EURCHF=X", "EUR/GBP": "EURGBP=X", "GBP/CAD": "GBPCAD=X",
    "GBP/CHF": "GBPCHF=X", "AUD/CAD": "AUDCAD=X", "GBP/AUD": "GBPAUD=X"
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

# ==================== ДОПОМІЖНІ ФУНКЦІЇ ====================
def format_minutes(n):
    if n % 10 == 1 and n % 100 != 11:
        return f"{n} хвилина"
    elif 2 <= n % 10 <= 4 and (n % 100 < 10 or n % 100 >= 20):
        return f"{n} хвилини"
    else:
        return f"{n} хвилин"

def get_main_keyboard():
    return ReplyKeyboardMarkup([
        [KeyboardButton("💱 Обрати валютну пару")],
        [KeyboardButton("📊 Статистика")]
    ], resize_keyboard=True)

def get_pairs_inline_keyboard():
    symbols_list = list(SYMBOLS.keys())
    keyboard = []
    for i in range(0, len(symbols_list), 2):
        row = [InlineKeyboardButton(symbols_list[i], callback_data=f"pair_{symbols_list[i]}")]
        if i + 1 < len(symbols_list):
            row.append(InlineKeyboardButton(symbols_list[i + 1], callback_data=f"pair_{symbols_list[i + 1]}"))
        keyboard.append(row)
    keyboard.append([InlineKeyboardButton("❌ Закрити", callback_data="close_pairs")])
    return InlineKeyboardMarkup(keyboard)

# ==================== АНАЛІЗ РИНКУ ТА ГРАФІК ====================
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

def generate_chart(df, display_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label='Close Price', color='#1f77b4', linewidth=2)
    plt.title(f'Technical Chart: {display_name}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

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

    score = (atr / 0.10) * 0.5 + (min(vol_ratio, 2.0) / 2.0) * 0.5
    score = float(np.clip(score, 0.0, 1.0))
    exact_mins = int(round(17 - score * (17 - 3)))
    exact_mins = max(3, min(17, exact_mins))
    expiration = format_minutes(exact_mins)

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
    await update.message.reply_text(
        "👋 Вітаю! Скористайтеся кнопками внизу для вибору пари або статистики.",
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )

async def stats_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message = update.effective_message
    try:
        conn = sqlite3.connect('bot_stats.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM signals WHERE user_id = ? AND status != 'PENDING'", (user_id,))
        res_tot = cursor.fetchone()
        total_finished = res_tot[0] if res_tot else 0
        
        cursor.execute("SELECT COUNT(*) FROM signals WHERE user_id = ? AND status = 'WIN'", (user_id,))
        res_wins = cursor.fetchone()
        wins = res_wins[0] if res_wins else 0
        
        cursor.execute("SELECT COUNT(*) FROM signals WHERE user_id = ? AND status = 'LOSS'", (user_id,))
        res_loss = cursor.fetchone()
        losses = res_loss[0] if res_loss else 0
        
        total_winrate = round((wins / total_finished * 100), 1) if total_finished > 0 else 0

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
        
        if message:
            await message.reply_text(msg, reply_markup=get_main_keyboard(), parse_mode="Markdown")
    except Exception as e:
        print(f"Stats error: {e}")
        if message:
            await message.reply_text(f"❌ Помилка завантаження статистики: {e}", reply_markup=get_main_keyboard())

async def text_message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    
    if text == "📊 Статистика":
        await stats_handler(update, context)
        return
        
    if text == "💱 Обрати валютну пару":
        await update.message.reply_text(
            "📋 **Оберіть валютну пару:**",
            reply_markup=get_pairs_inline_keyboard(),
            parse_mode="Markdown"
        )
        return

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = update.effective_user.id
    
    if data == "close_pairs":
        await query.message.delete()
        return

    if data.startswith("pair_"):
        display_name = data.split("_", 1)[1]
        ticker_code = SYMBOLS[display_name]
        
        await query.edit_message_text(
            text=f"⏳ Розраховую точний час та об'єми для `{display_name}`...",
            parse_mode="Markdown"
        )
        
        res = analyze_symbol(ticker_code, display_name)
        
        if not res:
            await query.edit_message_text(text=f"❌ Не вдалося завантажити дані для `{display_name}`.")
            return

        signal_id = save_signal_to_db(user_id, res['symbol'], res['type'], res['price'], res['confidence'])

        df_chart = fetch_forex_data(ticker_code, "5m", period="2d")
        photo_buf = generate_chart(df_chart, res['symbol']) if df_chart is not None else None

        msg = (
            f"🎯 **АНАЛІЗ ПАРИ: {res['symbol']}**\n\n"
            f"🔹 **Рекомендація:** `{res['type']}`\n"
            f"💰 **Ціна входу:** `{res['price']}`\n"
            f"⏱ **Точна експірація:** `{res['expiration']}`\n"
            f"📊 **Рівень RSI (M5):** `{res['rsi']}`\n"
            f"📈 **Якість сигналу:** `{res['confidence']}`\n"
            f"📦 **Співвідношення об'єму:** `{res['vol_ratio']}x від середнього`\n"
            f"📏 **Волатильність (ATR):** `{res['atr']}%`\n\n"
            f"👇 **Позначте результат після завершення угоди:**"
        )
        
        keyboard = [
            [
                InlineKeyboardButton("✅ Плюс (Win)", callback_data=f"win_{signal_id}"),
                InlineKeyboardButton("❌ Мінус (Loss)", callback_data=f"loss_{signal_id}")
            ]
        ]
        
        await query.message.delete()
        if photo_buf:
            await context.bot.send_photo(
                chat_id=user_id,
                photo=photo_buf,
                caption=msg,
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await context.bot.send_message(
                chat_id=user_id,
                text=msg,
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        return

    if data.startswith("win_") or data.startswith("loss_"):
        status = "WIN" if data.startswith("win_") else "LOSS"
        signal_id = data.split("_")[1]
        
        update_signal_status(signal_id, status)
        
        status_text = "🟢 Зараховано як ПЛЮС (Win)" if status == "WIN" else "🔴 Зараховано як МІНУС (Loss)"
        await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([[
            InlineKeyboardButton(f"Статус: {status_text}", callback_data="none")
        ]]))

# ==================== FLASK & TELEGRAM SETUP ====================
app = Flask(__name__)
application = Application.builder().token(BOT_TOKEN).build()

application.add_handler(CommandHandler("start", start_cmd))
application.add_handler(CommandHandler("stats", stats_handler))
application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message_handler))
application.add_handler(CallbackQueryHandler(callback_handler))

@app.route('/')
def home():
    return "Bot is running!"

@app.route('/webhook', methods=['POST'])
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
    webhook_url = "https://racio-1bot.onrender.com/webhook"
    requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook?url={webhook_url}")
    app.run(host='0.0.0.0', port=10000)
