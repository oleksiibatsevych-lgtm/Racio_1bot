# -*- coding: utf-8 -*-
"""Повне безпечне очищення PostgreSQL (схема public) та скидання strategies_config.json до чистого еталона GitHub."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import database

logger = logging.getLogger(__name__)

CLEAN_GITHUB_CONFIG = {
    "REV_RSI14": True,
    "REV_BB3": True,
    "REV_RSI7": True,
    "CRSI_5M": True,
    "CRSI_1M": False,
    "VALUE_CH_5M": True,
    "VALUE_CH_1M": False,
    "OSC_CONS_5M": True,
    "OSC_CONS_3M": True,
    "BB_KC_REV_5M": True,
    "BB_KC_REV_15M": True,
    "PIVOT_REV_5M": True,
    "EXH_STRK_1M": False,
    "EXH_STRK_3M": False,
    "STARC_5M": True,
    "STARC_15M": True,
    "LINREG_5M": True,
    "HUBBA_5M": True,
    "MTF_PB_1M_15": False,
    "MTF_PB_3M_15": True,
    "MTF_PB_5M_60": True,
    "EMA_RIB_3M": False,
    "EMA_RIB_5M": True,
    "MACD_ADX_5M": True,
    "MACD_ADX_15M": True,
    "ST_MOM_3M": False,
    "ST_MOM_5M": True,
    "PSAR_EMA_5M": True,
    "PSAR_EMA_15M": True,
    "ICHI_TK_5M": True,
    "ICHI_TK_15M": True,
    "HA_EMA_5M": True,
    "TTM_SQZ_5M": True,
    "TTM_SQZ_15M": True,
    "ORB_LON_5M": True,
    "ORB_NY_5M": True,
    "DONCH_BRK_5M": True,
    "JUDAS_LON_5M": True,
    "DIV_RSI_5M": True,
    "DIV_RSI_15M": True,
    "DIV_MACD_5M": True,
    "DIV_MACD_15M": True,
    "AO_TWIN_5M": True,
    "AO_ZERO_3M": True,
    "CCI_WPR_5M": True,
    "CCI_WPR_3M": True,
    "DEM_BB_5M": True,
    "WT_CROSS_5M": True,
    "LIQ_SWP_3M": True,
    "LIQ_SWP_5M": True,
    "FVG_IMP_3M": True,
    "FVG_IMP_5M": True,
    "PA_PIN_BB_5M": True,
    "PA_ENG_SR_5M": True,
    "PA_FAKEY_5M": True,
    "PA_MHI_1M": False,
    "PA_3LS_5M": True,
    "session_filter": True,
    "regime_filter": True,
    "news_filter": True,
    "auto_disable_49": True,
}


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
        tmp_file = cfg_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(CLEAN_GITHUB_CONFIG, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
