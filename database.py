import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta

try:
    import psycopg2
except ImportError:
    psycopg2 = None

logger = logging.getLogger(__name__)
DB_NAME = "trading_bot.db"
DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_connection():
    if DATABASE_URL and psycopg2:
        try:
            url = DATABASE_URL
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql://", 1)
            return psycopg2.connect(url), "pg"
        except Exception as e:
            logger.warning(f"Не вдалося підключитися до PostgreSQL, використовується SQLite: {e}")
    
    conn = sqlite3.connect(DB_NAME)
    return conn, "sqlite"

def init_db():
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        
        if db_type == "pg":
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(20),
                    signal VARCHAR(10),
                    entry_price DOUBLE PRECISION,
                    expiration_mins INT,
                    timestamp VARCHAR(30),
                    chat_id BIGINT,
                    message_id BIGINT,
                    rsi DOUBLE PRECISION,
                    adx DOUBLE PRECISION,
                    bb_width DOUBLE PRECISION,
                    session_code INT,
                    hour INT,
                    divergence VARCHAR(20),
                    dist_pivot DOUBLE PRECISION,
                    message_text TEXT,
                    exit_price DOUBLE PRECISION,
                    result VARCHAR(10),
                    pips DOUBLE PRECISION,
                    volatility_ratio DOUBLE PRECISION DEFAULT 1.0,
                    wick_ratio DOUBLE PRECISION DEFAULT 0.0,
                    ema_dist DOUBLE PRECISION DEFAULT 0.0
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT,
                    signal TEXT,
                    entry_price REAL,
                    expiration_mins INTEGER,
                    timestamp TEXT,
                    chat_id INTEGER,
                    message_id INTEGER,
                    rsi REAL,
                    adx REAL,
                    bb_width REAL,
                    session_code INTEGER,
                    hour INTEGER,
                    divergence TEXT,
                    dist_pivot REAL,
                    message_text TEXT,
                    exit_price REAL,
                    result TEXT,
                    pips REAL,
                    volatility_ratio REAL DEFAULT 1.0,
                    wick_ratio REAL DEFAULT 0.0,
                    ema_dist REAL DEFAULT 0.0
                )
            ''')
        
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.exception(f"Помилка ініціалізації бази даних: {e}")

init_db()

def save_signal(ticker, signal_type, entry_price, expiration_mins, chat_id, message_id,
                rsi=0, adx=0, bb_width=0, session_code=0, hour=0, divergence='NONE',
                dist_pivot=0, message_text='', volatility_ratio=1.0, wick_ratio=0.0, ema_dist=0.0):
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        placeholder = "%s" if db_type == "pg" else "?"
        query = f'''
            INSERT INTO signals (
                ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id,
                rsi, adx, bb_width, session_code, hour, divergence, dist_pivot, message_text,
                volatility_ratio, wick_ratio, ema_dist
            ) VALUES ({','.join([placeholder]*18)})
        '''
        if db_type == "pg":
            query += " RETURNING id"
            cursor.execute(query, (
                ticker, signal_type, entry_price, expiration_mins, timestamp_str, chat_id, message_id,
                rsi, adx, bb_width, session_code, hour, divergence, dist_pivot, message_text,
                volatility_ratio, wick_ratio, ema_dist
            ))
            sig_id = cursor.fetchone()[0]
        else:
            cursor.execute(query, (
                ticker, signal_type, entry_price, expiration_mins, timestamp_str, chat_id, message_id,
                rsi, adx, bb_width, session_code, hour, divergence, dist_pivot, message_text,
                volatility_ratio, wick_ratio, ema_dist
            ))
            sig_id = cursor.lastrowid
            
        conn.commit()
        cursor.close()
        conn.close()
        return sig_id
    except Exception as e:
        logger.exception(f"Помилка збереження сигналу в БД: {e}")
        return None

def get_pending_signals():
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, message_text
            FROM signals 
            WHERE result IS NULL
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.exception(f"Помилка отримання незавершених сигналів: {e}")
        return []

def evaluate_single_signal(sig_id, fetch_yahoo_data_func):
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        placeholder = "%s" if db_type == "pg" else "?"
        
        cursor.execute(f"""
            SELECT ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, message_text
            FROM signals 
            WHERE id = {placeholder}
        """, (sig_id,))
        row = cursor.fetchone()
        
        if not row:
            cursor.close()
            conn.close()
            return None
            
        ticker, signal_type, entry_price, expiration_mins, timestamp_str, chat_id, message_id, message_text = row
        
        entry_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        expiry_time = entry_time + timedelta(minutes=expiration_mins)
        
        df = fetch_yahoo_data_func(ticker, interval="1m", range_period="2d")
        
        if df.empty:
            cursor.close()
            conn.close()
            return None

        df_filtered = df[(df.index >= expiry_time - timedelta(seconds=30)) & 
                         (df.index <= expiry_time + timedelta(minutes=2))]
        
        if not df_filtered.empty:
            exit_price = float(df_filtered['close'].iloc[0])
        else:
            exit_price = float(df['close'].iloc[-1])

        pip_size = 0.01 if "JPY" in ticker else 0.0001
        price_diff = (exit_price - entry_price) if signal_type == "CALL" else (entry_price - exit_price)
        pips = round(price_diff / pip_size, 1)

        result = "WIN" if pips > 0.1 else ("LOSS" if pips < -0.1 else "NEUTRAL")

        cursor.execute(f"""
            UPDATE signals 
            SET exit_price = {placeholder}, result = {placeholder}, pips = {placeholder}
            WHERE id = {placeholder}
        """, (exit_price, result, pips, sig_id))
        
        conn.commit()
        cursor.close()
        conn.close()

        return {
            "chat_id": chat_id,
            "message_id": message_id,
            "message_text": message_text,
            "result": result,
            "pips": pips,
            "exit_price": exit_price
        }
    except Exception as e:
        logger.exception(f"Помилка під час оцінки сигналу {sig_id}: {e}")
        return None

def get_overall_stats():
    try:
        conn, db_type = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT result FROM signals WHERE result IS NOT NULL")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        wins = sum(1 for r in rows if r[0] == 'WIN')
        losses = sum(1 for r in rows if r[0] == 'LOSS')
        neutral = sum(1 for r in rows if r[0] == 'NEUTRAL')
        total = len(rows)
        winrate = round((wins / (wins + losses)) * 100, 1) if (wins + losses) > 0 else 0.0
        
        return {
            "wins": wins,
            "losses": losses,
            "neutral": neutral,
            "total": total,
            "winrate": winrate
        }
    except Exception as e:
        logger.exception(f"Помилка розрахунку статистики: {e}")
        return {"wins": 0, "losses": 0, "neutral": 0, "total": 0, "winrate": 0.0}
