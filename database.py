import sqlite3
import time
import pandas as pd
from datetime import datetime, timedelta

DB_NAME = "trading_stats.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            signal TEXT,
            entry_price REAL,
            expiration_mins INTEGER,
            timestamp TEXT,
            status TEXT DEFAULT 'PENDING',
            result TEXT DEFAULT 'UNKNOWN',
            chat_id INTEGER,
            message_id INTEGER,
            pips INTEGER DEFAULT 0,
            rsi REAL,
            adx REAL,
            bb_width REAL,
            message_text TEXT
        )
    ''')
    
    cursor.execute("PRAGMA table_info(signals)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    required_columns = {
        "ticker": "TEXT", "signal": "TEXT", "entry_price": "REAL",
        "expiration_mins": "INTEGER", "timestamp": "TEXT", "status": "TEXT DEFAULT 'PENDING'",
        "result": "TEXT DEFAULT 'UNKNOWN'", "chat_id": "INTEGER", "message_id": "INTEGER",
        "pips": "INTEGER DEFAULT 0", "rsi": "REAL", "adx": "REAL", "bb_width": "REAL", "message_text": "TEXT"
    }
    for col_name, col_type in required_columns.items():
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE signals ADD COLUMN {col_name} {col_type}")
            except:
                pass
    conn.commit()
    conn.close()

def save_signal(ticker, signal, entry_price, expiration_mins, chat_id=None, message_id=None, rsi=0.0, adx=0.0, bb_width=0.0, message_text=""):
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO signals (ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, rsi, adx, bb_width, message_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, rsi, adx, bb_width, message_text))
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return signal_id

def get_pending_signals():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Автоматично закриваємо старі завислі сигнали (старші за 2 години), щоб вони не блокували статистику
    two_hours_ago = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE signals SET status = 'EXPIRED', result = 'EXPIRED' WHERE status = 'PENDING' AND timestamp < ?", (two_hours_ago,))
    conn.commit()
    
    cursor.execute("SELECT id, ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, message_text FROM signals WHERE status = 'PENDING'")
    rows = cursor.fetchall()
    conn.close()
    return rows

def evaluate_single_signal(sig_id, fetch_data_func):
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT ticker, signal, entry_price, chat_id, message_id, message_text, status FROM signals WHERE id = ?", (sig_id,))
    row = cursor.fetchone()
    if not row or row[6] != 'PENDING':
        conn.close()
        return None
        
    ticker, signal, entry_price, chat_id, message_id, message_text, _ = row
    
    df = pd.DataFrame()
    for _ in range(3):
        df = fetch_data_func(ticker, interval="15m", range_period="10d")
        if not df.empty:
            break
        time.sleep(2)
        
    if df.empty:
        conn.close()
        return None
        
    current_price = float(df['close'].iloc[-1])
    diff = current_price - entry_price
    multiplier = 1000 if "JPY" in ticker.upper() else 100000
    pips = int(round(diff * multiplier))
    
    if signal == 'CALL':
        result = 'WIN' if current_price > entry_price else 'LOSS'
    else:
        result = 'WIN' if current_price < entry_price else 'LOSS'
        
    cursor.execute("UPDATE signals SET status = 'COMPLETED', result = ?, pips = ? WHERE id = ?", (result, pips, sig_id))
    conn.commit()
    conn.close()
    
    return {
        "chat_id": chat_id,
        "message_id": message_id,
        "result": result,
        "pips": pips,
        "message_text": message_text
    }

def get_overall_stats():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) FROM signals WHERE status = 'COMPLETED'")
    row = cursor.fetchone()
    conn.close()
    
    total = row[0] if row and row[0] else 0
    wins = row[1] if row and row[1] else 0
    winrate = round((wins / total) * 100, 1) if total > 0 else 0
    return {"total": total, "wins": wins, "winrate": winrate}
