import time
import psycopg2
from config import DATABASE_URL

def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL не вказано в змінних оточення.")
    return psycopg2.connect(DATABASE_URL)

def init_db():
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
        exit_price NUMERIC,
        expiration INT,
        ai_decision TEXT,
        ai_confidence INT,
        ai_reason TEXT,
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    query_logs = """
    CREATE TABLE IF NOT EXISTS filtered_logs (
        id SERIAL PRIMARY KEY,
        chat_id BIGINT,
        log_text TEXT,
        timestamp DOUBLE PRECISION
    );
    """
    
    columns_to_add = [
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pair TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS signal_type TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS entry_price NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS exit_price NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS expiration INT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ai_decision TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ai_confidence INT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ai_reason TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'PENDING';",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS chat_id BIGINT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS message_id BIGINT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS rsi NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS adx NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS bb_width NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS session_code INT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS hour INT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS divergence TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS dist_pivot NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS volatility_ratio NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS wick_ratio NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS ema_dist NUMERIC;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS message_text TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS result TEXT;",
        "ALTER TABLE signals ADD COLUMN IF NOT EXISTS pips NUMERIC;"
    ]
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query_users)
                cur.execute(query_signals)
                cur.execute(query_logs)
                for col_query in columns_to_add:
                    cur.execute(col_query)
            conn.commit()
        print("✅ База даних PostgreSQL успішно ініціалізована та оновлена.")
    except Exception as e:
        print(f"⚠️ Помилка ініціалізації бази даних: {e}")

def register_user(user_id, username=None):
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
    pair = kwargs.get('ticker') or kwargs.get('pair') or (args[0] if len(args) > 0 else "UNKNOWN")
    signal_type = kwargs.get('signal_type') or (args[1] if len(args) > 1 else "HOLD")
    entry_price = kwargs.get('entry_price') or (args[2] if len(args) > 2 else 0.0)
    expiration = kwargs.get('expiration_mins') or kwargs.get('expiration') or (args[3] if len(args) > 3 else 5)
    chat_id = kwargs.get('chat_id') or (args[4] if len(args) > 4 else None)
    message_id = kwargs.get('message_id') or (args[5] if len(args) > 5 else None)

    ai_decision = kwargs.get('ai_decision', 'YES')
    ai_confidence = kwargs.get('ai_confidence', 7)
    ai_reason = kwargs.get('ai_reason', '')
    rsi = kwargs.get('rsi')
    adx = kwargs.get('adx')
    bb_width = kwargs.get('bb_width')
    session_code = kwargs.get('session_code')
    hour = kwargs.get('hour')
    divergence = kwargs.get('divergence')
    dist_pivot = kwargs.get('dist_pivot')
    message_text = kwargs.get('message_text')
    volatility_ratio = kwargs.get('volatility_ratio')
    wick_ratio = kwargs.get('wick_ratio')
    ema_dist = kwargs.get('ema_dist')

    query = """
    INSERT INTO signals (
        pair, signal_type, entry_price, expiration, chat_id, message_id,
        ai_decision, ai_confidence, ai_reason, status,
        rsi, adx, bb_width, session_code, hour, divergence, dist_pivot,
        message_text, volatility_ratio, wick_ratio, ema_dist
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'PENDING', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    pair, signal_type, entry_price, expiration, chat_id, message_id,
                    ai_decision, ai_confidence, ai_reason,
                    rsi, adx, bb_width, session_code, hour, divergence, dist_pivot,
                    message_text, volatility_ratio, wick_ratio, ema_dist
                ))
                signal_id = cur.fetchone()[0]
            conn.commit()
            return signal_id
    except Exception as e:
        print(f"⚠️ Помилка збереження сигналу для {pair}: {e}")
        return None

def get_signal_by_id(sig_id):
    query = """
    SELECT id, pair, signal_type, entry_price, expiration, chat_id, message_id, message_text, status, result 
    FROM signals WHERE id = %s;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (sig_id,))
                row = cur.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'ticker': row[1],
                        'pair': row[1],
                        'signal_type': row[2],
                        'entry_price': row[3],
                        'expiration': row[4],
                        'chat_id': row[5],
                        'message_id': row[6],
                        'message_text': row[7],
                        'status': row[8],
                        'result': row[9]
                    }
    except Exception as e:
        print(f"⚠️ Помилка отримання сигналу {sig_id}: {e}")
    return None

def update_signal_result(sig_id, result, exit_price=None, pips=None):
    query = """
    UPDATE signals 
    SET result = %s, status = %s, exit_price = %s, pips = %s 
    WHERE id = %s;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (result, result, exit_price, pips, sig_id))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Помилка оновлення результату сигналу {sig_id}: {e}")

def get_pending_signals():
    query = """
    SELECT id, pair, signal_type, entry_price, expiration, 
           TO_CHAR(created_at, 'YYYY-MM-DD HH24:MI:SS') as created_at_str, 
           chat_id, message_id, message_text 
    FROM signals WHERE status = 'PENDING';
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
    except Exception as e:
        print(f"⚠️ Помилка отримання pending сигналів: {e}")
        return []

def evaluate_single_signal(sig_id, fetch_yahoo_data_func=None):
    query = "SELECT id, pair, signal_type, entry_price, expiration, chat_id, message_id, message_text FROM signals WHERE id = %s;"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (sig_id,))
                row = cur.fetchone()
                if not row:
                    return None
                
                sig_id, pair, signal_type, entry_price, expiration, chat_id, message_id, message_text = row
                entry_price = float(entry_price) if entry_price is not None else 0.0
                
                if fetch_yahoo_data_func:
                    df = fetch_yahoo_data_func(pair, interval="1m", range_period="1d")
                    if df is not None and not df.empty and 'close' in df.columns:
                        exit_price = float(df['close'].iloc[-1])
                    else:
                        exit_price = entry_price
                else:
                    exit_price = entry_price

                pips_multiplier = 1000.0 if "JPY" in str(pair) else 100000.0
                raw_diff = exit_price - entry_price

                if signal_type == "CALL":
                    pips = round(raw_diff * pips_multiplier, 1)
                    if exit_price > entry_price:
                        res = "WIN"
                    elif exit_price < entry_price:
                        res = "LOSS"
                    else:
                        res = "NEUTRAL"
                elif signal_type in ["PUT", "PRODAZH"]:
                    pips = round(-raw_diff * pips_multiplier, 1)
                    if exit_price < entry_price:
                        res = "WIN"
                    elif exit_price > entry_price:
                        res = "LOSS"
                    else:
                        res = "NEUTRAL"
                else:
                    pips = 0.0
                    res = "NEUTRAL"

                update_query = "UPDATE signals SET result = %s, status = %s, exit_price = %s, pips = %s WHERE id = %s;"
                cur.execute(update_query, (res, res, exit_price, pips, sig_id))
            conn.commit()

        return {
            "chat_id": chat_id,
            "message_id": message_id,
            "pips": pips,
            "result": res,
            "message_text": message_text
        }
    except Exception as e:
        print(f"⚠️ Помилка оцінки сигналу {sig_id}: {e}")
        return None

def update_signal_status(signal_id, status):
    query = "UPDATE signals SET status = %s WHERE id = %s;"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (status, signal_id))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Помилка оновлення статусу сигналу {signal_id}: {e}")

def get_overall_stats():
    query = """
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN result = 'WIN' OR status = 'WIN' THEN 1 END) as wins,
        COUNT(CASE WHEN result = 'LOSS' OR status = 'LOSS' THEN 1 END) as losses,
        COUNT(CASE WHEN result = 'NEUTRAL' OR status = 'NEUTRAL' THEN 1 END) as neutral,
        COUNT(CASE WHEN status = 'PENDING' THEN 1 END) as pending
    FROM signals;
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                row = cur.fetchone()
                if row:
                    total, wins, losses, neutral, pending = row[0], row[1], row[2], row[3], row[4]
                    completed = wins + losses
                    winrate = round((wins / completed * 100), 1) if completed > 0 else 0.0
                    return {
                        'total': total,
                        'wins': wins,
                        'losses': losses,
                        'neutral': neutral,
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
        'neutral': 0,
        'pending': 0,
        'completed': 0,
        'winrate': 0.0,
        'win_rate': 0.0
    }

def get_stats_summary():
    stats = get_overall_stats()
    return (
        f"📊 **Всього сигналів:** `{stats['total']}`\n"
        f"✅ **Успішних (WIN):** `{stats['wins']}`\n"
        f"❌ **Неуспішних (LOSS):** `{stats['losses']}`\n"
        f"➖ **Нейтральних:** `{stats['neutral']}`\n"
        f"⏳ **В очікуванні:** `{stats['pending']}`\n"
        f"🎯 **Вінрейт:** `{stats['winrate']}%`"
    )

def clear_filtered_logs(chat_id):
    query = "DELETE FROM filtered_logs WHERE chat_id = %s;"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (chat_id,))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Помилка очищення логів: {e}")

def save_filtered_log(chat_id, log_text):
    query = "INSERT INTO filtered_logs (chat_id, log_text, timestamp) VALUES (%s, %s, %s);"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (chat_id, log_text, time.time()))
            conn.commit()
    except Exception as e:
        print(f"⚠️ Помилка збереження логу: {e}")

def get_filtered_logs(chat_id):
    cutoff = time.time() - 3600
    query = "SELECT log_text FROM filtered_logs WHERE chat_id = %s AND timestamp > %s ORDER BY timestamp DESC;"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (chat_id, cutoff))
                rows = cur.fetchall()
                return [row[0] for row in rows]
    except Exception as e:
        print(f"⚠️ Помилка читання логів: {e}")
        return []
