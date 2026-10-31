# -*- coding: utf-8 -*-
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
    raw_url = raw_url.replace("&channel_binding=require", "").replace("?channel_binding=require", "")
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
                strategy VARCHAR(50),
                ai_review TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS strategy VARCHAR(50);")
        cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS ai_review техно;")

        # Таблиця для персоналізації стратегій по конкретних парах (ВКЛ/ВИКЛ)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS pair_strategy_settings (
                pair_name VARCHAR(50),
                strategy_key VARCHAR(50),
                is_active BOOLEAN DEFAULT TRUE,
                PRIMARY KEY (pair_name, strategy_key)
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

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_prompts (
                prompt_key VARCHAR(100) PRIMARY KEY,
                prompt_text TEXT
            );
            """
        )

        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Базу даних PostgreSQL успішно ініціалізовано з урахуванням матриці персоналізації та Shadow Mode.")
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
        logger.error(f"⚠️ Помилка реєстрації користувача: {e}")


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
    strategy="QUANTUM_MATRIX",
    status="PENDING",
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
                session_code, hour, divergence, dist_pivot, volatility_ratio, wick_ratio, ema_dist, strategy, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                chat_id, message_id, ticker, signal_type, entry_price, primary_tf,
                score, expiration_mins, timestamp_str, message_text, rsi, adx, bb_width,
                session_code, hour, divergence, dist_pivot, volatility_ratio, wick_ratio, ema_dist, strategy, status
            ),
        )
        signal_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        return signal_id
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження сигналу: {e}")
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
        logger.error(f"⚠️ Помилка PENDING сигналів: {e}")
        return []


def update_signal_result(signal_id, result, exit_price, pips=0, status="CLOSED"):
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
    except Exception as e:
        logger.error(f"⚠️ Помилка оновлення результату: {e}")


def save_signal_ai_review(signal_id, review_text):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE signals SET ai_review = %s WHERE id = %s;", (review_text, signal_id))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження AI-огляду: {e}")


def get_recent_failed_reviews_summary(ticker: str, limit=3) -> str:
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT ai_review, signal_type, strategy FROM signals 
            WHERE ticker = %s AND result = 'LOSS' AND ai_review IS NOT NULL 
            ORDER BY id DESC LIMIT %s;
            """,
            (ticker, limit)
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if not rows:
            return "Немає попередніх помилок для аналізу."
        lines = []
        for r in rows:
            lines.append(f"- Стратегія [{r['strategy']} ({r['signal_type']})]: {r['ai_review']}")
        return "\n".join(lines)
    except Exception:
        return "Помилка завантаження історії помилок."


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
            FROM signals WHERE status != 'SHADOW';
            """
        )
        stats = cursor.fetchone()
        cursor.close()
        conn.close()
        return stats
    except Exception:
        return {"total": 0, "wins": 0, "losses": 0, "pending": 0}


def get_detailed_pair_strategy_stats(ticker_filter=None, strategy_filter=None):
    """
    Повертає деталізовану статистику зрізу: Валютна пара ✖ Стратегія (включно з Shadow Mode).
    """
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        query = """
            SELECT 
                ticker,
                COALESCE(strategy, 'QUANTUM_MATRIX') as strategy,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE result = 'WIN') as wins,
                COUNT(*) FILTER (WHERE result = 'LOSS') as losses
            FROM signals 
            WHERE result IN ('WIN', 'LOSS')
        """
        params = []
        if ticker_filter:
            query += " AND ticker = %s"
            params.append(ticker_filter)
        if strategy_filter:
            query += " AND strategy = %s"
            params.append(strategy_filter)
            
        query += " GROUP BY ticker, strategy ORDER BY ticker ASC, wins DESC;"
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"⚠️ Помилка вибірки деталізованої статистики: {e}")
        return []


def get_pair_strategy_status(pair_name: str, strategy_key: str) -> bool:
    """
    Перевіряє, чи активна конкретна стратегія для конкретної пари. За замовчуванням True.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT is_active FROM pair_strategy_settings WHERE pair_name = %s AND strategy_key = %s;",
            (pair_name, strategy_key)
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if row is not None:
            return bool(row[0])
        return True
    except Exception:
        return True


def set_pair_strategy_status(pair_name: str, strategy_key: str, is_active: bool):
    """
    Зберігає стан стратегії для конкретної пари.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO pair_strategy_settings (pair_name, strategy_key, is_active)
            VALUES (%s, %s, %s)
            ON CONFLICT (pair_name, strategy_key) DO UPDATE 
            SET is_active = EXCLUDED.is_active;
            """,
            (pair_name, strategy_key, is_active)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження налаштування пари та стратегії: {e}")


def get_strategy_time_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        cursor.execute("""
            SELECT 
                COALESCE(strategy, 'MATRIX') as strategy,
                EXTRACT(HOUR FROM created_at) as trade_hour,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE result = 'WIN') as wins,
                COUNT(*) FILTER (WHERE result = 'LOSS') as losses
            FROM signals 
            WHERE result IN ('WIN', 'LOSS')
            GROUP BY strategy, trade_hour
            ORDER BY strategy, trade_hour;
        """)
        hour_rows = cursor.fetchall()

        cursor.execute("""
            SELECT 
                COALESCE(strategy, 'MATRIX') as strategy,
                EXTRACT(DOW FROM created_at) as trade_dow,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE result = 'WIN') as wins,
                COUNT(*) FILTER (WHERE result = 'LOSS') as losses
            FROM signals 
            WHERE result IN ('WIN', 'LOSS')
            GROUP BY strategy, trade_dow
            ORDER BY strategy, trade_dow;
        """)
        dow_rows = cursor.fetchall()
        
        cursor.close()
        conn.close()
        return {"hours": hour_rows, "dow": dow_rows}
    except Exception as e:
        logger.error(f"⚠️ Помилка вибірки статистики за часом: {e}")
        return {"hours": [], "dow": []}


def save_filtered_log(chat_id, log_text):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO filtered_logs (chat_id, log_text, timestamp) VALUES (%s, %s, %s)", (chat_id, log_text, time.time()))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception:
        pass


def get_system_logs(chat_id, minutes=120):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cutoff = time.time() - (minutes * 60)
        cursor.execute("SELECT log_text FROM filtered_logs WHERE chat_id = %s AND timestamp > %s ORDER BY timestamp DESC LIMIT 20", (chat_id, cutoff))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in rows]
    except Exception:
        return []


def clear_filtered_logs(chat_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM filtered_logs WHERE chat_id = %s", (chat_id,))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception:
        pass


def clear_all_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE signals, filtered_logs, pair_strategy_settings RESTART IDENTITY;")
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception:
        return False
