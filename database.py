import sqlite3
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
DB_NAME = "trading_bot.db"

def init_db():
    """Створення таблиці сигналів при старті."""
    try:
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
                pips REAL
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception(f"Помилка ініціалізації бази даних: {e}")

# Автоматична ініціалізація БД при імпорті модуля
init_db()

def save_signal(ticker, signal_type, entry_price, expiration_mins, chat_id, message_id,
                rsi=0, adx=0, bb_width=0, session_code=0, hour=0, divergence='NONE',
                dist_pivot=0, message_text=''):
    """Збереження нового сигналу в БД."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        timestamp_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO signals (
                ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id,
                rsi, adx, bb_width, session_code, hour, divergence, dist_pivot, message_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker, signal_type, entry_price, expiration_mins, timestamp_str, chat_id, message_id,
            rsi, adx, bb_width, session_code, hour, divergence, dist_pivot, message_text
        ))
        sig_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return sig_id
    except Exception as e:
        logger.exception(f"Помилка збереження сигналу в БД: {e}")
        return None

def get_pending_signals():
    """Отримання всіх сигналів, які ще не завершилися (результат IS NULL)."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, message_text
            FROM signals 
            WHERE result IS NULL
        """)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        logger.exception(f"Помилка отримання незавершених сигналів: {e}")
        return []

def evaluate_single_signal(sig_id, fetch_yahoo_data_func):
    """
    Точна перевірка результату угоди за часом експірації та типом сигналу (CALL/PUT).
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ticker, signal, entry_price, expiration_mins, timestamp, chat_id, message_id, message_text
            FROM signals 
            WHERE id = ?
        """, (sig_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
            
        ticker, signal_type, entry_price, expiration_mins, timestamp_str, chat_id, message_id, message_text = row
        
        # 1. Визначаємо точний час закінчення експірації (UTC)
        entry_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        expiry_time = entry_time + timedelta(minutes=expiration_mins)
        
        # 2. Отримуємо 1-хвилинні дані з Yahoo Finance
        df = fetch_yahoo_data_func(ticker, interval="1m", range_period="2d")
        
        if df.empty:
            conn.close()
            return None

        # 3. Знаходимо свічку, найближчу до моменту закінчення експірації
        df_filtered = df[(df.index >= expiry_time - timedelta(seconds=30)) & 
                         (df.index <= expiry_time + timedelta(minutes=2))]
        
        if not df_filtered.empty:
            exit_price = float(df_filtered['close'].iloc[0])
        else:
            exit_price = float(df['close'].iloc[-1])

        # 4. Визначення розміру пункту (Pip size: 0.01 для пар з JPY, 0.0001 для інших)
        pip_size = 0.01 if "JPY" in ticker else 0.0001

        # 5. Розрахунок різниці ціни з урахуванням напрямку (CALL vs PUT)
        if signal_type == "CALL":
            price_diff = exit_price - entry_price
        else:  # PUT
            price_diff = entry_price - exit_price

        pips = round(price_diff / pip_size, 1)

        # 6. Визначення фінального статусу угоди
        if pips > 0.1:
            result = "WIN"
        elif pips < -0.1:
            result = "LOSS"
        else:
            result = "NEUTRAL"

        # 7. Запис результату в базу даних
        cursor.execute("""
            UPDATE signals 
            SET exit_price = ?, result = ?, pips = ?
            WHERE id = ?
        """, (exit_price, result, pips, sig_id))
        
        conn.commit()
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
    """Розрахунок загальної статистики результатів."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT result FROM signals WHERE result IS NOT NULL")
        rows = cursor.fetchall()
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
