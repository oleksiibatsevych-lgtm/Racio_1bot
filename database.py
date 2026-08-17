import sqlite3
from datetime import datetime

DB_NAME = "trading_bot.db"


def init_db():
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT,
            signal TEXT,
            entry_price REAL,
            expiration_time TEXT,
            status TEXT DEFAULT 'ACTIVE',
            result TEXT DEFAULT 'PENDING',
            created_at TEXT
        )
    """)
  conn.commit()
  conn.close()


def save_signal(
    pair: str, signal: str, entry_price: float, expiration_time_str: str
):
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute(
      """
        INSERT INTO signals (pair, signal, entry_price, expiration_time, status, result, created_at)
        VALUES (?, ?, ?, ?, 'ACTIVE', 'PENDING', ?)
    """,
      (
          pair,
          signal,
          entry_price,
          expiration_time_str,
          datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
      ),
  )
  conn.commit()
  conn.close()


def get_active_signals():
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute(
      "SELECT id, pair, signal, entry_price, expiration_time FROM signals"
      " WHERE status = 'ACTIVE'"
  )
  rows = cursor.fetchall()
  conn.close()
  return rows


def update_signal_result(signal_id: int, result: str):
  """result може бути 'WIN' або 'LOSS'"""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute(
      """
        UPDATE signals 
        SET status = 'CLOSED', result = ? 
        WHERE id = ?
    """,
      (result, signal_id),
  )
  conn.commit()
  conn.close()


def get_statistics():
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()

  # Загальна кількість закритих сигналів
  cursor.execute(
      "SELECT COUNT(*) FROM signals WHERE result IN ('WIN', 'LOSS')"
  )
  total = cursor.fetchone()[0]

  # Кількість перемог
  cursor.execute("SELECT COUNT(*) FROM signals WHERE result = 'WIN'")
  wins = cursor.fetchone()[0]

  # Кількість поразок
  cursor.execute("SELECT COUNT(*) FROM signals WHERE result = 'LOSS'")
  losses = cursor.fetchone()[0]

  conn.close()

  win_rate = (wins / total * 100) if total > 0 else 0.0
  return {
      "total": total,
      "wins": wins,
      "losses": losses,
      "win_rate": round(win_rate, 2),
  }


def get_consecutive_losses(pair: str) -> int:
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute(
      """
        SELECT result FROM signals 
        WHERE pair = ? AND result IN ('WIN', 'LOSS') 
        ORDER BY id DESC LIMIT 5
    """,
      (pair,),
  )
  rows = cursor.fetchall()
  conn.close()

  consecutive = 0
  for row in rows:
    if row[0] == "LOSS":
      consecutive += 1
    else:
      break
  return consecutive
