import psycopg2
from config import DATABASE_URL

def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL не вказано в змінних оточення.")
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Створення та оновлення таблиць при запуску бота"""
    query_users = """
    CREATE TABLE IF NOT EXISTS users (
        user_id BIGINT PRIMARY KEY,
        username TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    query_signals = """
    CREATE TABLE IF NOT EXISTS signals (
        id SERIAL PRIMARY KEY,
        pair TEXT NOT NULL,
        signal_type TEXT NOT NULL,
        entry_price NUMERIC,
        expiration INT,
        ai_decision TEXT,
        ai_confidence INT,
        ai_reason TEXT,
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    query_add_status_col = """
    ALTER TABLE signals ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'PENDING';
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_users)
                cur.execute(query_signals)
                cur.execute(query_add_status_col)
            conn.commit()
        print("✅ База даних PostgreSQL успішно ініціалізована.")
    except Exception as e:
        print(f"⚠️ Помилка ініціалізації бази даних: {e}")

def register_user(user_id, username=None):
    """Збереження або оновлення користувача"""
    query = """
    INSERT INTO users (user_id, username) 
    VALUES (%s, %s) 
    ON CONFLICT (user_id) DO UPDATE SET username = EXCLUDED.username;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (user_id, username))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Помилка реєстрації користувача {user_id}: {e}")

def get_all_users():
    """Отримання списку всіх користувачів"""
    query = "SELECT user_id FROM users;"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                return [row[0] for row in rows]
    except Exception as e:
        print(f"⚠️ Помилка отримання користувачів: {e}")
        return []

def save_signal(pair, signal_type, entry_price, expiration, ai_decision, ai_confidence, ai_reason, *args, **kwargs):
    """Збереження нового сигналу з підтримкою додаткових аргументів"""
    query = """
    INSERT INTO signals (pair, signal_type, entry_price, expiration, ai_decision, ai_confidence, ai_reason, status)
    VALUES (%s, %s, %s, %s, %s, %s, %s, 'PENDING') RETURNING id;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (pair, signal_type, entry_price, expiration, ai_decision, ai_confidence, ai_reason))
                signal_id = cur.fetchone()[0]
            conn.commit()
            return signal_id
    except Exception as e:
        print(f"⚠️ Помилка збереження сигналу для {pair}: {e}")
        return None

def get_pending_signals():
    """Отримання незавершених сигналів"""
    query = "SELECT id, pair, signal_type, entry_price, expiration, created_at FROM signals WHERE status = 'PENDING';"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
    except Exception as e:
        print(f"⚠️ Помилка отримання pending сигналів: {e}")
        return []

def update_signal_status(signal_id, status):
    """Оновлення статусу сигналу (WIN / LOSS / EXPIRED)"""
    query = "UPDATE signals SET status = %s WHERE id = %s;"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (status, signal_id))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Помилка оновлення статусу сигналу {signal_id}: {e}")
