# -*- coding: utf-8 -*-
import logging
import threading
import time
from datetime import datetime, timezone

import requests

from config import NEWS_BLACKOUT_MINS

logger = logging.getLogger(__name__)

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_TTL_SEC = 3600
FAIL_RETRY_SEC = 300

_LOCK = threading.Lock()
_CACHE = {
    "ts": 0.0,
    "fail_ts": 0.0,
    "events": [],
}

GLOBAL_USD_KEYWORDS = (
    "FOMC", "NON-FARM", "NONFARM", "NFP", "CPI", "CORE CPI",
    "FEDERAL FUNDS", "POWELL", "GDP", "PPI", "RETAIL SALES",
)


def _extract_currencies(pair_name: str | None) -> set[str]:
    if not pair_name:
        return {"USD", "EUR", "GBP", "JPY"}
    clean = str(pair_name).replace("OANDA:", "").replace("IC MARKETS:", "").strip()
    if "/" in clean:
        parts = clean.split("/", 1)
        return {parts[0].strip().upper(), parts[1].strip().upper()}
    if "_" in clean:
        parts = clean.split("_", 1)
        return {parts[0].strip().upper(), parts[1].strip().upper()}
    if len(clean) == 6:
        return {clean[:3].upper(), clean[3:].upper()}
    return {"USD"}


def _parse_events(raw_items: list) -> list[dict]:
    parsed = []
    for item in raw_items or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("impact", "")).strip().lower() != "high":
            continue
        date_str = item.get("date")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
        except Exception:
            continue
        parsed.append({
            "country": str(item.get("country", "")).strip().upper(),
            "title": str(item.get("title", "High Impact News")).strip(),
            "dt": dt,
        })
    return parsed


def _get_events() -> tuple[list[dict], bool]:
    now_ts = time.time()
    with _LOCK:
        if _CACHE["events"] and (now_ts - _CACHE["ts"]) < CACHE_TTL_SEC:
            return _CACHE["events"], True
        if _CACHE["fail_ts"] and (now_ts - _CACHE["fail_ts"]) < FAIL_RETRY_SEC:
            return _CACHE["events"], bool(_CACHE["events"])

    try:
        resp = requests.get(
            CALENDAR_URL,
            timeout=6,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RacioBot/1.0)"},
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                events = _parse_events(data)
                with _LOCK:
                    _CACHE["events"] = events
                    _CACHE["ts"] = time.time()
                    _CACHE["fail_ts"] = 0.0
                return events, True
    except Exception as e:
        logger.warning(f"Не вдалося завантажити календар ForexFactory: {e}")

    with _LOCK:
        _CACHE["fail_ts"] = time.time()
        return _CACHE["events"], bool(_CACHE["events"])


def is_high_impact_news_time(
    pair_name: str | None = None,
    now: datetime | None = None,
    blackout_mins: int | None = None,
) -> tuple[bool, str]:
    """
    Перевіряє, чи перебуває пара у вікні важливих макроекономічних новин (±blackout_mins).
    """
    try:
        now_utc = now or datetime.now(timezone.utc)
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)
        else:
            now_utc = now_utc.astimezone(timezone.utc)

        window_sec = max(5, int(blackout_mins if blackout_mins is not None else NEWS_BLACKOUT_MINS)) * 60
        currencies = _extract_currencies(pair_name)
        events, ok = _get_events()

        if ok:
            for ev in events:
                diff = abs((now_utc - ev["dt"]).total_seconds())
                if diff > window_sec:
                    continue
                country = ev["country"]
                title_up = ev["title"].upper()
                is_global_usd = country == "USD" and any(k in title_up for k in GLOBAL_USD_KEYWORDS)
                if country in currencies or is_global_usd:
                    return True, f"Новина ({country}): {ev['title']} о {ev['dt']:%H:%M} UTC"
            return False, "OK"

        if now_utc.weekday() == 4 and 12 <= now_utc.hour <= 15:
            return True, "Резервний фільтр новин у п'ятницю (Nonfarm Payrolls)"
        return False, "OK"
    except Exception as e:
        logger.error(f"Помилка перевірки календаря новин: {e}")
        return False, "OK"
