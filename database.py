import psycopg2
from config import DATABASE_URL

def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL не вказано в змінних оточення.")
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Створення необхідних таблиць при запуску бота"""
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
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_users)
                cur.execute(query_signals)
            conn.commit()
        print("✅ База даних PostgreSQL (Neon) успішно ініціалізована.")
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

def save_signal(pair, signal_type, entry_price, expiration, ai_decision, ai_confidence, ai_reason):
    """Збереження торгового сигналу"""
    query = """
    INSERT INTO signals (pair, signal_type, entry_price, expiration, ai_decision, ai_confidence, ai_reason)
    VALUES (%s, %s, %s, %s, %s, %s, %s);
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (pair, signal_type, entry_price, expiration, ai_decision, ai_confidence, ai_reason))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Помилка збереження сигналу для {pair}: {e}")
