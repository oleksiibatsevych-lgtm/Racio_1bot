import os
import logging
import time
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")


def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL не вказано у змінних оточення!")
    raw_url = DATABASE_URL
    if raw_url.startswith("postgres://"):
        raw_url = raw_url.replace("postgres://", "postgresql://", 1)
    raw_url = raw_url.replace("&channel_binding=require", "").replace(
        "?channel_binding=require", ""
    )
    return psycopg2.connect(raw_url, sslmode="require")


def init_db():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGINT PRIMARY KEY,
                username VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        )

        cursor.execute(
            """
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
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS filtered_logs (
                id SERIAL PRIMARY KEY,
                chat_id BIGINT,
                log_text TEXT,
                timestamp REAL
            );
        """
        )

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Базу даних PostgreSQL успішно ініціалізовано.")
    except Exception as e:
        logger.error(f"⚠️ Помилка ініціалізації БД: {e}")


def register_user(user_id, username=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (user_id, username)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE 
            SET username = EXCLUDED.username;
        """,
            (user_id, username),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка реєстрації користувача {user_id}: {e}")


def save_signal(
    chat_id=None,
    message_id=None,
    ticker=None,
    signal_type=None,
    entry_price=None,
    primary_tf=None,
    score=None,
    expiration_mins=None,
    timestamp_str=None,
    message_text=None,
    rsi=None,
    adx=None,
    bb_width=None,
    session_code=None,
    hour=None,
    divergence=None,
    dist_pivot=None,
    volatility_ratio=None,
    wick_ratio=None,
    ema_dist=None,
    **kwargs,
):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO signals (
                chat_id, message_id, ticker, signal_type, entry_price, primary_tf, 
                score, expiration_mins, timestamp_str, message_text, rsi, adx, bb_width,
                session_code, hour, divergence, dist_pivot, volatility_ratio, wick_ratio, ema_dist, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING')
            RETURNING id;
        """,
            (
                chat_id,
                message_id,
                ticker,
                signal_type,
                entry_price,
                primary_tf,
                score,
                expiration_mins,
                timestamp_str,
                message_text,
                rsi,
                adx,
                bb_width,
                session_code,
                hour,
                divergence,
                dist_pivot,
                volatility_ratio,
                wick_ratio,
                ema_dist,
            ),
        )
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
        cursor.execute(
            "SELECT * FROM signals WHERE status = 'PENDING' ORDER BY created_at ASC;"
        )
        signals = cursor.fetchall()
        cursor.close()
        conn.close()
        return signals
    except Exception as e:
        logger.error(f"⚠️ Помилка отримання PENDING сигналів: {e}")
        return []


def update_signal_result(
    signal_id, result, exit_price, pips=0, status="CLOSED"
):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE signals 
            SET status = %s, result = %s, exit_price = %s, pips = %s
            WHERE id = %s;
        """,
            (status, result, exit_price, pips, signal_id),
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(
            f"✅ Сигнал #{signal_id} закрито: {result} (Ціна виходу: {exit_price})"
        )
    except Exception as e:
        logger.error(f"⚠️ Помилка оновлення сигналу #{signal_id}: {e}")


def get_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE result = 'WIN') as wins,
                COUNT(*) FILTER (WHERE result = 'LOSS') as losses,
                COUNT(*) FILTER (WHERE status = 'PENDING') as pending
            FROM signals;
        """
        )
        stats = cursor.fetchone()
        cursor.close()
        conn.close()
        return stats
    except Exception as e:
        logger.error(f"⚠️ Помилка отримання статистики: {e}")
        return {"total": 0, "wins": 0, "losses": 0, "pending": 0}


def save_filtered_log(chat_id, log_text):
    """Зберігає лог відхиленого сигналу у PostgreSQL."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO filtered_logs (chat_id, log_text, timestamp) VALUES (%s, %s, %s)",
            (chat_id, log_text, time.time()),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження логу відхилення: {e}")


def get_system_logs(chat_id, minutes=120):
    """Отримує останні відхилені signals за останні N хвилин."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cutoff = time.time() - (minutes * 60)
        cursor.execute(
            "SELECT log_text FROM filtered_logs WHERE chat_id = %s AND timestamp > %s ORDER BY timestamp DESC LIMIT 20",
            (chat_id, cutoff),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        logger.error(f"⚠️ Помилка читання логів: {e}")
        return []


def clear_filtered_logs(chat_id):
    """Очищає застарілі логи відхилень для чату."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM filtered_logs WHERE chat_id = %s", (chat_id,)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка очищення логів: {e}")
