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
        pair TEXT,
        signal_type TEXT,
        entry_price NUMERIC,
        expiration INT,
        ai_decision TEXT,
        ai_confidence INT,
        ai_reason TEXT,
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    columns_to_add = [
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pair TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS signal_type TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS entry_price NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS expiration INT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ai_decision TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ai_confidence INT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ai_reason TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'PENDING';"
    ]
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_users)
                cur.execute(query_signals)
                for col_query in columns_to_add:
                    cur.execute(col_query)
            conn.commit()
        print("✅ База даних PostgreSQL успішно ініціалізована та оновлена.")
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

def save_signal(*args, **kwargs):
    """Універсальне збереження сигналу в БД"""
    pair = kwargs.get('pair') or (args[0] if len(args) > 0 else "UNKNOWN")
    signal_type = kwargs.get('signal_type') or (args[1] if len(args) > 1 else "HOLD")
    entry_price = kwargs.get('entry_price') or (args[2] if len(args) > 2 else 0.0)
    expiration = kwargs.get('expiration') or (args[3] if len(args) > 3 else 5)
    ai_decision = kwargs.get('ai_decision') or (args[4] if len(args) > 4 else "NO")
    ai_confidence = kwargs.get('ai_confidence') or (args[5] if len(args) > 5 else 0)
    ai_reason = kwargs.get('ai_reason') or (args[6] if len(args) > 6 else "")

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

def get_overall_stats():
    """Розрахунок загальної статистики угод з бази даних"""
    query = """
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN status = 'WIN' THEN 1 END) as wins,
        COUNT(CASE WHEN status = 'LOSS' THEN 1 END) as losses,
        COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending
    FROM signals;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                row = cur.fetchone()
                if row:
                    total, wins, losses, pending = row[0], row[1], row[2], row[3]
                    completed = wins + losses
                    winrate = round((wins / completed * 100), 1) if completed > 0 else 0.0
                    return {
                        'total': total,
                        'wins': wins,
                        'losses': losses,
                        'pending': pending,
                        'completed': completed,
                        'winrate': winrate,
                        'win_rate': winrate
                    }
    except Exception as e:
        print(f"⚠️ Помилка отримання загальної статистики: {e}")
        
    return {
        'total': 0,
        'wins': 0,
        'losses': 0,
        'pending': 0,
        'completed': 0,
        'winrate': 0.0,
        'win_rate': 0.0
    }
