from datetime import datetime, timezone
import sqlite3
import threading
from config import DB_FILE

db_lock = threading.Lock()


def init_db():
  with db_lock:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
            CREATE TABLE IF NOT EXISTS stats_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                pair TEXT,
                signal TEXT,
                result TEXT
            )
        """)
    conn.commit()
    conn.close()


def save_stat_to_db(pair, signal, result):
  with db_lock:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO stats_history (timestamp, pair, signal, result) VALUES"
        " (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), pair, signal, result),
    )
    conn.commit()
    conn.close()


def get_all_stats_from_db():
  with db_lock:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, pair, signal, result FROM stats_history"
    )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
      history.append({
          "timestamp": datetime.fromisoformat(row[0]),
          "pair": row[1],
          "signal": row[2],
          "result": row[3],
      })
    return history


def get_consecutive_losses(pair_name: str) -> int:
  history = get_all_stats_from_db()
  pair_items = [i for i in history if i["pair"] == pair_name]
  if not pair_items:
    return 0
  losses = 0
  for item in reversed(pair_items):
    if item["result"] == "LOSS":
      losses += 1
    else:
      break
  return losses
