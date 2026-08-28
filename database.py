import sqlite3
import time
import pandas as pd
from datetime import datetime, timedelta

DB_NAME = "trading_stats.db"

def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
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
            session_code INTEGER,
            hour INTEGER,
            divergence TEXT,
            dist_pivot REAL,
            message_text TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_signal(ticker, signal, entry_price, expiration_mins, chat_id=None, message_id=None, 
                rsi=0.0, adx=0.0, bb_width=0.0, session_code=1, hour=12, 
                divergence="NONE", dist_pivot=0.0, message_text=""):
    init_db()
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO signals (ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, 
                             rsi, adx, bb_width, session_code, hour, divergence, dist_pivot, message_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, 
          rsi, adx, bb_width, session_code, hour, divergence, dist_pivot, message_text))
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return signal_id

def get_pending_signals():
    init_db()
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    
    two_hours_ago = (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("UPDATE signals SET status = 'EXPIRED', result = 'EXPIRED' WHERE status = 'PENDING' AND timestamp < ?", (two_hours_ago,))
    conn.commit()
    
    cursor.execute("SELECT id, ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, message_text FROM signals WHERE status = 'PENDING'")
    rows = cursor.fetchall()
    conn.close()
    return rows

def evaluate_single_signal(sig_id, fetch_data_func):
    init_db()
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT ticker, signal, entry_price, chat_id, message_id, message_text, status, timestamp, expiration_mins 
        FROM signals WHERE id = ?
    """, (sig_id,))
    row = cursor.fetchone()
    if not row or row[6] != 'PENDING':
        conn.close()
        return None
        
    ticker, signal, entry_price, chat_id, message_id, message_text, _, timestamp_str, expiration_mins = row
    
    signal_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    expiry_time = signal_time + timedelta(minutes=expiration_mins)
    
    df = pd.DataFrame()
    for _ in range(3):
        df = fetch_data_func(ticker, interval="3m", range_period="2d")
        if not df.empty:
            break
        time.sleep(2)
        
    if df.empty:
        cursor.execute("UPDATE signals SET status = 'ERROR', result = 'ERROR' WHERE id = ?", (sig_id,))
        conn.commit()
        conn.close()
        return None
        
    future_df = df[df.index >= expiry_time]
    if not future_df.empty:
        current_price = float(future_df['close'].iloc[0])
    else:
        current_price = float(df['close'].iloc[-1])
        
    multiplier = 1000 if "JPY" in ticker.upper() else 100000
    
    if signal == 'CALL':
        diff_pips = current_price - entry_price
        result = 'WIN' if current_price > entry_price else 'LOSS'
    else: 
        diff_pips = entry_price - current_price
        result = 'WIN' if current_price < entry_price else 'LOSS'
        
    pips = int(round(diff_pips * multiplier))
    
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
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) FROM signals WHERE status = 'COMPLETED'")
    row = cursor.fetchone()
    conn.close()
    
    total = row[0] if row and row[0] else 0
    wins = row[1] if row and row[1] else 0
    winrate = round((wins / total) * 100, 1) if total > 0 else 0
    return {"total": total, "wins": wins, "winrate": winrate}
