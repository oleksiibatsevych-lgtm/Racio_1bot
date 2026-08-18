import sqlite3
import time
import pandas as pd
from datetime import datetime

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
            result TEXT DEFAULT 'UNKNOWN'
        )
    ''')
    conn.commit()
    conn.close()

def save_signal(ticker, signal, entry_price, expiration_mins):
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO signals (ticker, signal, entry_price, expiration_mins, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (ticker, signal, entry_price, expiration_mins, timestamp))
    signal_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return signal_id

def fetch_with_retry(fetch_data_func, ticker, retries=3, delay=2):
    """Безпечне завантаження з повторними спробами для обходу лімітів Yahoo"""
    for attempt in range(retries):
        df = fetch_data_func(ticker, interval="5m", range_period="1d")
        if not df.empty:
            return df
        time.sleep(delay)
    return pd.DataFrame()

def evaluate_single_signal(signal_id, fetch_data_func):
    """Перевірка результату сигналу з урахуванням повторних спроб завантаження ціни"""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT ticker, signal, entry_price, status FROM signals WHERE id = ?", (signal_id,))
    row = cursor.fetchone()
    if not row or row[3] != 'PENDING':
        conn.close()
        return None, 0
        
    ticker, signal, entry_price, _ = row
    conn.close()
    
    # Використовуємо захищене завантаження з повторами
    df = fetch_with_retry(fetch_data_func, ticker)
    
    if df.empty:
        return None, 0
        
    current_price = df['close'].iloc[-1]
    
    # Розрахунок різниці в пунктах (pips)
    diff = current_price - entry_price
    multiplier = 100 if "JPY" in ticker.upper() else 10000
    pips = round(diff * multiplier, 1)
    
    if signal == 'CALL':
        result = 'WIN' if current_price > entry_price else 'LOSS'
    else:
        result = 'WIN' if current_price < entry_price else 'LOSS'
        
    # Зберігаємо результат у базу
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE signals SET status = 'COMPLETED', result = ? WHERE id = ?", (result, signal_id))
    conn.commit()
    conn.close()
    
    return result, pips

def evaluate_and_get_stats(fetch_data_func):
    """Масова перевірка всіх залишкових PENDING сигналів"""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, ticker, signal, entry_price FROM signals WHERE status = 'PENDING'")
    rows = cursor.fetchall()
    
    for row in rows:
        sig_id, ticker, signal, entry_price = row
        df = fetch_with_retry(fetch_data_func, ticker)
        if not df.empty:
            current_price = df['close'].iloc[-1]
            if signal == 'CALL':
                result = 'WIN' if current_price > entry_price else 'LOSS'
            else:
                result = 'WIN' if current_price < entry_price else 'LOSS'
            cursor.execute("UPDATE signals SET status = 'COMPLETED', result = ? WHERE id = ?", (result, sig_id))
            
    conn.commit()
    
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) FROM signals WHERE status = 'COMPLETED'")
    row = cursor.fetchone()
    conn.close()
    
    total = row[0] if row and row[0] else 0
    wins = row[1] if row and row[1] else 0
    winrate = round((wins / total) * 100, 1) if total > 0 else 0
    
    return {"total": total, "wins": wins, "winrate": winrate}
