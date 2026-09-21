import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL не вказано у змінних оточення!")
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def init_db():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGINT PRIMARY KEY,
                username VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT,
                message_id BIGINT,
                ticker VARCHAR(50),
                signal_type VARCHAR(10),
                entry_price NUMERIC,
                primary_tf VARCHAR(10),
                score INT,
                expiration_mins INT,
                timestamp_str VARCHAR(50),
                status VARCHAR(20) DEFAULT 'PENDING',
                result VARCHAR(10),
                exit_price NUMERIC,
                pips INT,
                message_text TEXT,
                rsi NUMERIC,
                adx NUMERIC,
                bb_width NUMERIC,
                session_code INT,
                hour INT,
                divergence VARCHAR(50),
                dist_pivot NUMERIC,
                volatility_ratio NUMERIC,
                wick_ratio NUMERIC,
                ema_dist NUMERIC,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Автоматична міграція колонок для існуючих баз даних
        columns_to_check = [
            ("timestamp_str", "VARCHAR(50)"),
            ("message_text", "TEXT"),
            ("rsi", "NUMERIC"),
            ("adx", "NUMERIC"),
            ("bb_width", "NUMERIC"),
            ("session_code", "INT"),
            ("hour", "INT"),
            ("divergence", "VARCHAR(50)"),
            ("dist_pivot", "NUMERIC"),
            ("volatility_ratio", "NUMERIC"),
            ("wick_ratio", "NUMERIC"),
            ("ema_dist", "NUMERIC")
        ]
        
        for col_name, col_type in columns_to_check:
            cursor.execute(f"""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='signals' AND column_name='{col_name}'
                    ) THEN
                        ALTER TABLE signals ADD COLUMN {col_name} {col_type};
                    END IF;
                END $$;
            """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Базу даних ініціалізовано: структуру таблиць та міграції перевірено.")
    except Exception as e:
        logger.error(f"⚠️ Помилка ініціалізації бази даних: {e}")

def register_user(user_id, username=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (user_id, username)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE 
            SET username = EXCLUDED.username;
        """, (user_id, username))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка реєстрації користувача {user_id}: {e}")

def save_signal(chat_id=None, message_id=None, ticker=None, signal_type=None, 
                entry_price=None, primary_tf=None, score=None, expiration_mins=None, 
                timestamp_str=None, message_text=None, rsi=None, adx=None, bb_width=None,
                session_code=None, hour=None, divergence=None, dist_pivot=None,
                volatility_ratio=None, wick_ratio=None, ema_dist=None, **kwargs):
    ticker = ticker or kwargs.get('pair')
    signal_type = signal_type or kwargs.get('signal')
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signals (
                chat_id, message_id, ticker, signal_type, entry_price, primary_tf, 
                score, expiration_mins, timestamp_str, message_text, rsi, adx, bb_width,
                session_code, hour, divergence, dist_pivot, volatility_ratio, wick_ratio, ema_dist, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING')
            RETURNING id;
        """, (
            chat_id, message_id, ticker, signal_type, entry_price, primary_tf, 
            score, expiration_mins, timestamp_str, message_text, rsi, adx, bb_width,
            session_code, hour, divergence, dist_pivot, volatility_ratio, wick_ratio, ema_dist
        ))
        signal_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"✅ Успішно збережено сигнал #{signal_id} для {ticker}")
        return signal_id
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження сигналу в БД: {e}")
        return None

def get_signal_by_id(signal_id):
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM signals WHERE id = %s;", (signal_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        return row
    except Exception as e:
        logger.error(f"⚠️ Помилка отримання сигналу #{signal_id}: {e}")
        return None

def get_pending_signals():
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM signals WHERE status = 'PENDING' ORDER BY created_at ASC;")
        signals = cursor.fetchall()
        cursor.close()
        conn.close()
        return signals
    except Exception as e:
        logger.error(f"⚠️ Помилка отримання PENDING сигналів: {e}")
        return []

def update_signal_result(signal_id, result, exit_price, pips=0, status='CLOSED'):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE signals 
            SET status = %s, result = %s, exit_price = %s, pips = %s
            WHERE id = %s;
        """, (status, result, exit_price, pips, signal_id))
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"✅ Сигнал #{signal_id} закрито: {result} (Ціна виходу: {exit_price})")
    except Exception as e:
        logger.error(f"⚠️ Помилка оновлення сигналу #{signal_id}: {e}")

def get_filtered_logs(chat_id):
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT id, ticker, signal_type, score, result, pips, timestamp_str 
            FROM signals 
            WHERE chat_id = %s 
            ORDER BY created_at DESC LIMIT 15;
        """, (chat_id,))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        logs = []
        for r in rows:
            res = r['result'] or 'PENDING'
            pips_str = f"({r['pips']} p.)" if r['pips'] is not None else ""
            logs.append(f"#{r['id']} | {r['ticker']} | {r['signal_type']} | Score: {r['score']} | Status: {res} {pips_str}")
        return logs
    except Exception as e:
        logger.error(f"⚠️ Помилка отримання логів для {chat_id}: {e}")
        return []

def get_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE result = 'WIN') as wins,
                COUNT(*) FILTER (WHERE result = 'LOSS') as losses,
                COUNT(*) FILTER (WHERE status = 'PENDING') as pending
            FROM signals;
        """)
        stats = cursor.fetchone()
        cursor.close()
        conn.close()
        return stats
    except Exception as e:
        logger.error(f"⚠️ Помилка отримання статистики: {e}")
        return {"total": 0, "wins": 0, "losses": 0, "pending": 0}

def get_stats_summary():
    stats = get_stats()
    total = stats.get('total', 0) or 0
    wins = stats.get('wins', 0) or 0
    losses = stats.get('losses', 0) or 0
    pending = stats.get('pending', 0) or 0
    
    closed = wins + losses
    winrate = round((wins / closed * 100), 1) if closed > 0 else 0.0
    
    return (
        f"🎯 Всього згенеровано: <b>{total}</b>\n"
        f"✅ Успішних (WIN): <b>{wins}</b>\n"
        f"❌ Невдалих (LOSS): <b>{losses}</b>\n"
        f"⏳ В очікуванні: <b>{pending}</b>\n\n"
        f"📈 <b>Winrate:</b> <b>{winrate}%</b>"
    )

def clear_signals_safely():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE signals RESTART IDENTITY;")
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Таблицю signals безпечно очищено.")
    except Exception as e:
        logger.error(f"⚠️ Помилка очищення таблиці signals: {e}")
