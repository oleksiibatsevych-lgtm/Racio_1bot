import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

# Отримання URL бази даних зі змінних оточення (Render)
DATABASE_URL = os.environ.get("DATABASE_URL", "")

def get_connection():
    """Створення та повернення підключення до PostgreSQL."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL не вказано у змінних оточення!")
    return psycopg2.connect(DATABASE_URL, sslmode='require')

def init_db():
    """Створення та перевірка наявності всіх необхідних таблиць."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Створення таблиці користувачів
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGINT PRIMARY KEY,
                username VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Створення таблиці сигналів
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
                status VARCHAR(20) DEFAULT 'PENDING',
                result VARCHAR(10),
                exit_price NUMERIC,
                pips INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Базу даних ініціалізовано: таблиці 'users' та 'signals' перевірено/створено.")
    except Exception as e:
        logger.error(f"⚠️ Помилка ініціалізації бази даних: {e}")

def register_user(user_id, username=None):
    """Реєстрація або оновлення даних користувача в БД."""
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

def save_signal(chat_id, message_id, ticker, signal_type, entry_price, primary_tf, score, expiration_mins):
    """Збереження нового сигналу у статус PENDING."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO signals (chat_id, message_id, ticker, signal_type, entry_price, primary_tf, score, expiration_mins, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'PENDING')
            RETURNING id;
        """, (chat_id, message_id, ticker, signal_type, entry_price, primary_tf, score, expiration_mins))
        signal_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"✅ Успішно збережено сигнал #{signal_id} для {ticker}")
        return signal_id
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження сигналу в БД: {e}")
        return None

def get_pending_signals():
    """Отримання всіх незакритих сигналів для перевірки результату."""
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT * FROM signals WHERE status = 'PENDING' ORDER BY created_at ASC;
        """)
        signals = cursor.fetchall()
        cursor.close()
        conn.close()
        return signals
    except Exception as e:
        logger.error(f"⚠️ Помилка отримання PENDING сигналів: {e}")
        return []

def update_signal_result(signal_id, status, result, exit_price, pips=0):
    """Оновлення результату угоди (WIN/LOSS) після експірації."""
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

def clear_signals_safely():
    """Безпечне очищення таблиці сигналів (зберігає структуру таблиці)."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE signals RESTART IDENTITY;")
        conn.commit()
        cursor.close()
        conn.close()
        logger.info("✅ Таблицю signals безпечно очищено (TRUNCATE).")
    except Exception as e:
        logger.error(f"⚠️ Помилка очищення таблиці signals: {e}")

def get_stats():
    """Отримання загальної статистики сигналів бота."""
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
