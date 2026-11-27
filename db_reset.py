# -*- coding: utf-8 -*-
"""Повне безпечне очищення PostgreSQL (схема public), скидання deleted_strategies.json та strategies_config.json до еталона GitHub."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import database
from strategy_engine import FULL_STRATEGIES_META, restore_all_deleted_strategies

logger = logging.getLogger(__name__)

LEGACY_DISABLED = {
    "CRSI_1M", "VALUE_CH_1M", "EXH_STRK_1M", "EXH_STRK_3M",
    "MTF_PB_1M_15", "EMA_RIB_3M", "ST_MOM_3M", "PA_MHI_1M",
}


def build_clean_github_config() -> dict:
    cfg = {k: (k not in LEGACY_DISABLED) for k in FULL_STRATEGIES_META}
    cfg.update({
        "session_filter": True,
        "regime_filter": True,
        "news_filter": True,
        "auto_disable_49": True,
    })
    return cfg


def _safe_ident(name: str) -> str:
    clean = "".join(ch for ch in str(name) if ch.isalnum() or ch == "_")
    return f'"{clean}"'


def reset_database_and_config(config_path: str | Path | None = None, reset_config: bool = True) -> dict:
    cleared_tables = []
    conn = database.get_connection()
    if conn is None:
        raise RuntimeError("Немає підключення до бази даних")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename;")
            rows = cur.fetchall() or []
            tables = [r["tablename"] if isinstance(r, dict) else r[0] for r in rows]
            if tables:
                quoted = ", ".join(_safe_ident(t) for t in tables)
                cur.execute(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE;")
                cleared_tables = tables
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if hasattr(database, "clear_in_memory_caches"):
        database.clear_in_memory_caches()

    config_reset = False
    if reset_config:
        cfg_file = Path(config_path) if config_path else Path(__file__).with_name("strategies_config.json")
        restore_all_deleted_strategies(config_path=cfg_file, deleted_path=cfg_file.with_name("deleted_strategies.json"))
        tmp_file = cfg_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(build_clean_github_config(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_file, cfg_file)
        config_reset = True

    return {"tables": cleared_tables, "config_reset": config_reset}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Повне очищення БД і повернення конфігурації до чистого GitHub.")
    parser.add_argument("--yes", action="store_true", help="Підтвердити повне видалення всіх даних у БД")
    parser.add_argument("--keep-config", action="store_true", help="Не перезаписувати strategies_config.json")
    args = parser.parse_args()
    if not args.yes:
        print("Додайте прапорець --yes для підтвердження очищення БД.")
        sys.exit(1)
    res = reset_database_and_config(reset_config=not args.keep_config)
    print(f"Очищено таблиці: {res['tables']} | Конфіг скинуто: {res['config_reset']}")
