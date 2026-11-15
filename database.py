# -*- coding: utf-8 -*-
import logging
import os
import threading
import time
import psycopg2
from psycopg2.extras import RealDictCursor

from config import (
    AUTO_DISABLE_MIN_BT_GLOBAL,
    AUTO_DISABLE_MIN_BT_PAIR,
    AUTO_DISABLE_MIN_LIVE_GLOBAL,
    AUTO_DISABLE_MIN_LIVE_PAIR,
    AUTO_DISABLE_WR_THRESHOLD,
    normalize_pair_name,
)

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

_STATUS_LOCK = threading.Lock()
_STATUS_CACHE = {}
_BULK_CACHE_TS = 0.0
_STATUS_TTL_SEC = 60


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
                auto_scan BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS auto_scan BOOLEAN DEFAULT FALSE;")

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
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS strategy VARCHAR(50);")
        cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS ai_review TEXT;")
        cursor.execute("ALTER TABLE signals ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP;")

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
            CREATE TABLE IF NOT EXISTS strategy_auto_status (
                scope_type VARCHAR(10) NOT NULL,
                pair_name VARCHAR(50) NOT NULL,
                strategy_key VARCHAR(50) NOT NULL,
                is_auto_disabled BOOLEAN DEFAULT FALSE,
                source VARCHAR(20),
                wins INT DEFAULT 0,
                total INT DEFAULT 0,
                win_rate NUMERIC DEFAULT 0,
                reason TEXT,
                reset_after_signal_id BIGINT DEFAULT 0,
                manual_override BOOLEAN DEFAULT FALSE,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (scope_type, pair_name, strategy_key)
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS backtest_results (
                pair_name VARCHAR(50) NOT NULL,
                strategy_key VARCHAR(50) NOT NULL,
                wins INT NOT NULL DEFAULT 0,
                total INT NOT NULL DEFAULT 0,
                win_rate NUMERIC NOT NULL DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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

        cursor.execute("UPDATE signals SET status = 'PENDING' WHERE status = 'PROCESSING';")

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
            SET username = COALESCE(EXCLUDED.username, users.username);
            """,
            (user_id, username),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка реєстрації користувача: {e}")


def set_user_autoscan(user_id, enabled: bool):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (user_id, auto_scan)
            VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE
            SET auto_scan = EXCLUDED.auto_scan;
            """,
            (user_id, bool(enabled)),
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження стану автоскану: {e}")


def get_autoscan_chats() -> set[int]:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM users WHERE auto_scan = TRUE;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return {int(r[0]) for r in rows}
    except Exception as e:
        logger.error(f"⚠️ Помилка завантаження чатів автоскану: {e}")
        return set()


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
    expires_at=None,
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
                session_code, hour, divergence, dist_pivot, volatility_ratio, wick_ratio, ema_dist, strategy, status, expires_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """,
            (
                chat_id, message_id, ticker, signal_type, entry_price, primary_tf,
                score, expiration_mins, timestamp_str, message_text, rsi, adx, bb_width,
                session_code, hour, divergence, dist_pivot, volatility_ratio, wick_ratio, ema_dist, strategy, status, expires_at
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


def get_chat_signals(chat_id, mode="stats", since=None, until=None):
    clauses = ["chat_id = %s"]
    parameters = [chat_id]
    if mode == "active":
        clauses.append("status = 'PENDING'")
    elif mode in ("history", "stats"):
        clauses.append("status IN ('CLOSED', 'VOID')")
    else:
        raise ValueError("Unknown signal view")
    if since is not None:
        clauses.append("created_at >= %s")
        parameters.append(since)
    if until is not None:
        clauses.append("created_at < %s")
        parameters.append(until)
    query = "SELECT * FROM signals WHERE " + " AND ".join(clauses) + " ORDER BY created_at DESC, id DESC"
    if mode == "history":
        query = query.replace(" ORDER BY", " AND (status = 'VOID' OR result IN ('WIN', 'LOSS', 'NEUTRAL')) ORDER BY")
        query += " LIMIT 15"
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, parameters)
            return cursor.fetchall()
    finally:
        conn.close()


def has_open_reversion_signal(ticker, strategy):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM signals WHERE ticker = %s AND strategy = %s "
                "AND status IN ('PENDING', 'PROCESSING'));",
                (ticker, strategy),
            )
            return bool(cursor.fetchone()[0])
    finally:
        conn.close()


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
            (ticker, limit),
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        if not rows:
            return "Немає попередніх помилок для аналізу."
        return "\n".join(
            f"- Стратегія [{r['strategy']} ({r['signal_type']})]: {r['ai_review']}" for r in rows
        )
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
            FROM signals WHERE status NOT IN ('SHADOW', 'VOID');
            """
        )
        stats = cursor.fetchone()
        cursor.close()
        conn.close()
        return stats
    except Exception:
        return {"total": 0, "wins": 0, "losses": 0, "pending": 0}


def get_detailed_pair_strategy_stats(ticker_filter=None, strategy_filter=None):
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
            WHERE result IN ('WIN', 'LOSS') AND status != 'VOID'
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


EXPORT_COLUMNS = [
    "id", "timestamp_str", "ticker", "signal_type", "strategy", "primary_tf", "expiration_mins", "score",
    "entry_price", "exit_price", "pips", "result", "rsi", "adx", "bb_width", "session_code", "hour",
    "divergence", "dist_pivot", "volatility_ratio", "wick_ratio", "ema_dist",
]


def export_resolved_signals():
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT " + ", ".join(EXPORT_COLUMNS) +
            " FROM signals WHERE result IN ('WIN', 'LOSS', 'NEUTRAL') ORDER BY id ASC;"
        )
        rows = cursor.fetchall()
        cursor.close()
        return EXPORT_COLUMNS, rows
    finally:
        conn.close()


def get_segment_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT ticker,
                   COALESCE(primary_tf, '5m') AS tf,
                   COALESCE(strategy, 'QUANTUM_MATRIX') AS strategy,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (WHERE result = 'WIN') AS wins
            FROM signals
            WHERE result IN ('WIN', 'LOSS') AND status != 'VOID'
            GROUP BY ticker, COALESCE(primary_tf, '5m'), COALESCE(strategy, 'QUANTUM_MATRIX');
            """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"⚠️ Помилка статистики сегментів: {e}")
        return []


def claim_signal(signal_id) -> bool:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE signals SET status = 'PROCESSING' WHERE id = %s AND status = 'PENDING' RETURNING id;",
            (signal_id,),
        )
        got = cursor.fetchone() is not None
        conn.commit()
        cursor.close()
        conn.close()
        return got
    except Exception as e:
        logger.error(f"⚠️ Помилка claim_signal: {e}")
        return False


def release_signal(signal_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE signals SET status = 'PENDING' WHERE id = %s AND status = 'PROCESSING';", (signal_id,))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка release_signal: {e}")


def _refresh_pair_strategy_cache():
    global _BULK_CACHE_TS
    now = time.time()
    with _STATUS_LOCK:
        if now - _BULK_CACHE_TS < _STATUS_TTL_SEC:
            return
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT pair_name, strategy_key, is_active FROM pair_strategy_settings;")
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            _STATUS_CACHE.clear()
            for pair_name, strategy_key, is_active in rows:
                _STATUS_CACHE[(pair_name, strategy_key)] = bool(is_active)
            _BULK_CACHE_TS = time.time()
        except Exception:
            _BULK_CACHE_TS = time.time()


def get_pair_strategy_status(pair_name: str, strategy_key: str) -> bool:
    _refresh_pair_strategy_cache()
    with _STATUS_LOCK:
        return _STATUS_CACHE.get((pair_name, strategy_key), True)


def set_pair_strategy_status(pair_name: str, strategy_key: str, is_active: bool):
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
            (pair_name, strategy_key, is_active),
        )
        with _STATUS_LOCK:
            _STATUS_CACHE[(pair_name, strategy_key)] = bool(is_active)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження налаштування пари та стратегії: {e}")


def set_pair_bulk_strategy_status(pair_name: str, strategy_keys: list[str], is_active: bool):
    if not strategy_keys:
        return
    try:
        conn = get_connection()
        cursor = conn.cursor()
        for key in strategy_keys:
            cursor.execute(
                """
                INSERT INTO pair_strategy_settings (pair_name, strategy_key, is_active)
                VALUES (%s, %s, %s)
                ON CONFLICT (pair_name, strategy_key) DO UPDATE
                SET is_active = EXCLUDED.is_active;
                """,
                (pair_name, key, bool(is_active)),
            )
            with _STATUS_LOCK:
                _STATUS_CACHE[(pair_name, key)] = bool(is_active)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка групового збереження налаштувань пари: {e}")


def upsert_backtest_results(rows: list[dict]):
    if not rows:
        return
    try:
        conn = get_connection()
        cursor = conn.cursor()
        for r in rows:
            pair = normalize_pair_name(r["pair"])
            strategy = r["strategy"]
            wins = int(r["wins"])
            total = int(r["total"])
            wr = float(wins / total) if total > 0 else 0.0
            cursor.execute(
                """
                INSERT INTO backtest_results (pair_name, strategy_key, wins, total, win_rate, updated_at)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (pair_name, strategy_key) DO UPDATE
                SET wins = EXCLUDED.wins,
                    total = EXCLUDED.total,
                    win_rate = EXCLUDED.win_rate,
                    updated_at = CURRENT_TIMESTAMP;
                """,
                (pair, strategy, wins, total, wr),
            )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка збереження результатів бектесту: {e}")


def get_backtest_results(pair_filter: str | None = None) -> list[dict]:
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        if pair_filter:
            cursor.execute(
                "SELECT * FROM backtest_results WHERE pair_name = %s ORDER BY win_rate DESC, total DESC;",
                (normalize_pair_name(pair_filter),),
            )
        else:
            cursor.execute("SELECT * FROM backtest_results ORDER BY pair_name ASC, win_rate DESC;")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"⚠️ Помилка читання результатів бектесту: {e}")
        return []


def get_auto_disabled_items() -> list[dict]:
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT scope_type, pair_name, strategy_key, is_auto_disabled, source,
                   wins, total, win_rate, reason, reset_after_signal_id, manual_override, updated_at
            FROM strategy_auto_status
            WHERE is_auto_disabled = TRUE
            ORDER BY scope_type ASC, pair_name ASC, win_rate ASC;
            """
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        logger.error(f"⚠️ Помилка читання авто-вимкнених стратегій: {e}")
        return []


def get_auto_disabled_map() -> dict[tuple[str, str, str], dict]:
    rows = get_auto_disabled_items()
    return {(r["scope_type"], r["pair_name"], r["strategy_key"]): r for r in rows}


def _get_auto_status_record(cursor, scope_type: str, pair_name: str, strategy_key: str) -> dict | None:
    cursor.execute(
        """
        SELECT scope_type, pair_name, strategy_key, is_auto_disabled, source,
               wins, total, win_rate, reset_after_signal_id, manual_override
        FROM strategy_auto_status
        WHERE scope_type = %s AND pair_name = %s AND strategy_key = %s;
        """,
        (scope_type, pair_name, strategy_key),
    )
    row = cursor.fetchone()
    if not row:
        return None
    if isinstance(row, dict):
        return row
    cols = (
        "scope_type", "pair_name", "strategy_key", "is_auto_disabled", "source",
        "wins", "total", "win_rate", "reset_after_signal_id", "manual_override",
    )
    return dict(zip(cols, row))


def mark_strategy_manual_enable(scope_type: str, pair_name: str, strategy_key: str):
    norm_pair = "*" if scope_type == "GLOBAL" else normalize_pair_name(pair_name)
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COALESCE(MAX(id), 0) FROM signals;")
        max_row = cursor.fetchone()
        max_sig_id = int(max_row[0]) if max_row and max_row[0] is not None else 0
        cursor.execute(
            """
            INSERT INTO strategy_auto_status (
                scope_type, pair_name, strategy_key, is_auto_disabled, source,
                wins, total, win_rate, reason, reset_after_signal_id, manual_override, updated_at
            )
            VALUES (%s, %s, %s, FALSE, 'MANUAL', 0, 0, 0, 'Ввімкнено вручну', %s, TRUE, CURRENT_TIMESTAMP)
            ON CONFLICT (scope_type, pair_name, strategy_key) DO UPDATE
            SET is_auto_disabled = FALSE,
                manual_override = TRUE,
                reset_after_signal_id = GREATEST(strategy_auto_status.reset_after_signal_id, EXCLUDED.reset_after_signal_id),
                reason = 'Ввімкнено вручну',
                updated_at = CURRENT_TIMESTAMP;
            """,
            (scope_type, norm_pair, strategy_key, max_sig_id),
        )
        if scope_type == "PAIR":
            cursor.execute(
                """
                INSERT INTO pair_strategy_settings (pair_name, strategy_key, is_active)
                VALUES (%s, %s, TRUE)
                ON CONFLICT (pair_name, strategy_key) DO UPDATE
                SET is_active = TRUE;
                """,
                (norm_pair, strategy_key),
            )
            with _STATUS_LOCK:
                _STATUS_CACHE[(norm_pair, strategy_key)] = True
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка ручного ввімкнення стратегії {strategy_key}: {e}")


def evaluate_and_apply_auto_disable(
    source: str = "LIVE",
    pair_filter: str | None = None,
    strategy_filter: str | None = None,
    threshold: float = AUTO_DISABLE_WR_THRESHOLD,
    min_pair: int | None = None,
    min_global: int | None = None,
) -> dict[str, list[dict]]:
    """
    Перевіряє результати (LIVE або BACKTEST) і автоматично вимикає стратегії
    (по парі та/або глобально), якщо вінрейт < threshold при достатній вибірці.
    """
    src = source.upper()
    req_pair = min_pair if min_pair is not None else (
        AUTO_DISABLE_MIN_BT_PAIR if src == "BACKTEST" else AUTO_DISABLE_MIN_LIVE_PAIR
    )
    req_global = min_global if min_global is not None else (
        AUTO_DISABLE_MIN_BT_GLOBAL if src == "BACKTEST" else AUTO_DISABLE_MIN_LIVE_GLOBAL
    )
    events = {"disabled_global": [], "disabled_pair": []}
    norm_filter = normalize_pair_name(pair_filter) if pair_filter else None

    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        if src == "BACKTEST":
            cursor.execute("SELECT pair_name, strategy_key, wins, total FROM backtest_results;")
            bt_rows = cursor.fetchall()
            pair_agg = {}
            global_agg = {}
            for r in bt_rows:
                p = normalize_pair_name(r["pair_name"])
                s = r["strategy_key"]
                w, t = int(r["wins"]), int(r["total"])
                pair_agg[(p, s)] = [w, t]
                g = global_agg.setdefault(s, [0, 0])
                g[0] += w
                g[1] += t

            for (p, s), (w, t) in pair_agg.items():
                if norm_filter and p != norm_filter:
                    continue
                if strategy_filter and s != strategy_filter:
                    continue
                if t < req_pair or (w / t) >= threshold:
                    continue
                rec = _get_auto_status_record(cursor, "PAIR", p, s)
                if rec and (rec.get("manual_override") or rec.get("is_auto_disabled")):
                    continue
                wr = w / t
                reason = f"Бектест {wr * 100:.1f}% ({w}/{t}) < {threshold * 100:.0f}%"
                cursor.execute(
                    """
                    INSERT INTO strategy_auto_status (
                        scope_type, pair_name, strategy_key, is_auto_disabled, source,
                        wins, total, win_rate, reason, manual_override, updated_at
                    )
                    VALUES ('PAIR', %s, %s, TRUE, 'BACKTEST', %s, %s, %s, %s, FALSE, CURRENT_TIMESTAMP)
                    ON CONFLICT (scope_type, pair_name, strategy_key) DO UPDATE
                    SET is_auto_disabled = TRUE, source = 'BACKTEST',
                        wins = EXCLUDED.wins, total = EXCLUDED.total,
                        win_rate = EXCLUDED.win_rate, reason = EXCLUDED.reason,
                        manual_override = FALSE, updated_at = CURRENT_TIMESTAMP;
                    """,
                    (p, s, w, t, wr, reason),
                )
                cursor.execute(
                    """
                    INSERT INTO pair_strategy_settings (pair_name, strategy_key, is_active)
                    VALUES (%s, %s, FALSE)
                    ON CONFLICT (pair_name, strategy_key) DO UPDATE SET is_active = FALSE;
                    """,
                    (p, s),
                )
                with _STATUS_LOCK:
                    _STATUS_CACHE[(p, s)] = False
                events["disabled_pair"].append({"pair": p, "strategy": s, "wins": w, "total": t, "wr": wr, "source": "BACKTEST"})

            if not norm_filter:
                for s, (w, t) in global_agg.items():
                    if strategy_filter and s != strategy_filter:
                        continue
                    if t < req_global or (w / t) >= threshold:
                        continue
                    rec = _get_auto_status_record(cursor, "GLOBAL", "*", s)
                    if rec and (rec.get("manual_override") or rec.get("is_auto_disabled")):
                        continue
                    wr = w / t
                    reason = f"Бектест {wr * 100:.1f}% ({w}/{t}) < {threshold * 100:.0f}%"
                    cursor.execute(
                        """
                        INSERT INTO strategy_auto_status (
                            scope_type, pair_name, strategy_key, is_auto_disabled, source,
                            wins, total, win_rate, reason, manual_override, updated_at
                        )
                        VALUES ('GLOBAL', '*', %s, TRUE, 'BACKTEST', %s, %s, %s, %s, FALSE, CURRENT_TIMESTAMP)
                        ON CONFLICT (scope_type, pair_name, strategy_key) DO UPDATE
                        SET is_auto_disabled = TRUE, source = 'BACKTEST',
                            wins = EXCLUDED.wins, total = EXCLUDED.total,
                            win_rate = EXCLUDED.win_rate, reason = EXCLUDED.reason,
                            manual_override = FALSE, updated_at = CURRENT_TIMESTAMP;
                        """,
                        (s, w, t, wr, reason),
                    )
                    events["disabled_global"].append({"strategy": s, "wins": w, "total": t, "wr": wr, "source": "BACKTEST"})

        else:
            cursor.execute(
                """
                SELECT id, ticker, COALESCE(strategy, 'QUANTUM_MATRIX') AS strategy, result
                FROM signals
                WHERE status = 'CLOSED' AND result IN ('WIN', 'LOSS')
                ORDER BY id ASC;
                """
            )
            sig_rows = cursor.fetchall()

            strategies_seen = {r["strategy"] for r in sig_rows}
            pairs_seen = {(normalize_pair_name(r["ticker"]), r["strategy"]) for r in sig_rows}

            for p, s in pairs_seen:
                if norm_filter and p != norm_filter:
                    continue
                if strategy_filter and s != strategy_filter:
                    continue
                rec = _get_auto_status_record(cursor, "PAIR", p, s)
                if rec and rec.get("is_auto_disabled"):
                    continue
                reset_id = int(rec["reset_after_signal_id"]) if rec and rec.get("reset_after_signal_id") else 0
                sub = [r for r in sig_rows if normalize_pair_name(r["ticker"]) == p and r["strategy"] == s and int(r["id"]) > reset_id]
                t = len(sub)
                if t < req_pair:
                    continue
                w = sum(1 for r in sub if r["result"] == "WIN")
                wr = w / t
                if wr < threshold:
                    reason = f"Живі угоди {wr * 100:.1f}% ({w}/{t}) < {threshold * 100:.0f}%"
                    cursor.execute(
                        """
                        INSERT INTO strategy_auto_status (
                            scope_type, pair_name, strategy_key, is_auto_disabled, source,
                            wins, total, win_rate, reason, reset_after_signal_id, manual_override, updated_at
                        )
                        VALUES ('PAIR', %s, %s, TRUE, 'LIVE', %s, %s, %s, %s, %s, FALSE, CURRENT_TIMESTAMP)
                        ON CONFLICT (scope_type, pair_name, strategy_key) DO UPDATE
                        SET is_auto_disabled = TRUE, source = 'LIVE',
                            wins = EXCLUDED.wins, total = EXCLUDED.total,
                            win_rate = EXCLUDED.win_rate, reason = EXCLUDED.reason,
                            manual_override = FALSE, updated_at = CURRENT_TIMESTAMP;
                        """,
                        (p, s, w, t, wr, reason, reset_id),
                    )
                    cursor.execute(
                        """
                        INSERT INTO pair_strategy_settings (pair_name, strategy_key, is_active)
                        VALUES (%s, %s, FALSE)
                        ON CONFLICT (pair_name, strategy_key) DO UPDATE SET is_active = FALSE;
                        """,
                        (p, s),
                    )
                    with _STATUS_LOCK:
                        _STATUS_CACHE[(p, s)] = False
                    events["disabled_pair"].append({"pair": p, "strategy": s, "wins": w, "total": t, "wr": wr, "source": "LIVE"})

            for s in strategies_seen:
                if strategy_filter and s != strategy_filter:
                    continue
                rec = _get_auto_status_record(cursor, "GLOBAL", "*", s)
                if rec and rec.get("is_auto_disabled"):
                    continue
                reset_id = int(rec["reset_after_signal_id"]) if rec and rec.get("reset_after_signal_id") else 0
                sub = [r for r in sig_rows if r["strategy"] == s and int(r["id"]) > reset_id]
                t = len(sub)
                if t < req_global:
                    continue
                w = sum(1 for r in sub if r["result"] == "WIN")
                wr = w / t
                if wr < threshold:
                    reason = f"Живі угоди {wr * 100:.1f}% ({w}/{t}) < {threshold * 100:.0f}%"
                    cursor.execute(
                        """
                        INSERT INTO strategy_auto_status (
                            scope_type, pair_name, strategy_key, is_auto_disabled, source,
                            wins, total, win_rate, reason, reset_after_signal_id, manual_override, updated_at
                        )
                        VALUES ('GLOBAL', '*', %s, TRUE, 'LIVE', %s, %s, %s, %s, %s, FALSE, CURRENT_TIMESTAMP)
                        ON CONFLICT (scope_type, pair_name, strategy_key) DO UPDATE
                        SET is_auto_disabled = TRUE, source = 'LIVE',
                            wins = EXCLUDED.wins, total = EXCLUDED.total,
                            win_rate = EXCLUDED.win_rate, reason = EXCLUDED.reason,
                            manual_override = FALSE, updated_at = CURRENT_TIMESTAMP;
                        """,
                        (s, w, t, wr, reason, reset_id),
                    )
                    events["disabled_global"].append({"strategy": s, "wins": w, "total": t, "wr": wr, "source": "LIVE"})

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"⚠️ Помилка оцінки авто-вимкнення ({src}): {e}")
    return events


def get_strategy_time_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT
                COALESCE(strategy, 'MATRIX') as strategy,
                EXTRACT(HOUR FROM created_at) as trade_hour,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE result = 'WIN') as wins,
                COUNT(*) FILTER (WHERE result = 'LOSS') as losses
            FROM signals
            WHERE result IN ('WIN', 'LOSS') AND status != 'VOID'
            GROUP BY strategy, trade_hour
            ORDER BY strategy, trade_hour;
            """
        )
        hour_rows = cursor.fetchall()

        cursor.execute(
            """
            SELECT
                COALESCE(strategy, 'MATRIX') as strategy,
                EXTRACT(DOW FROM created_at) as trade_dow,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE result = 'WIN') as wins,
                COUNT(*) FILTER (WHERE result = 'LOSS') as losses
            FROM signals
            WHERE result IN ('WIN', 'LOSS') AND status != 'VOID'
            GROUP BY strategy, trade_dow
            ORDER BY strategy, trade_dow;
            """
        )
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
        cursor.execute("TRUNCATE TABLE signals, filtered_logs, backtest_results, strategy_auto_status RESTART IDENTITY;")
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"⚠️ Помилка очищення статистики: {e}")
        return False
