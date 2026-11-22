# -*- coding: utf-8 -*-
import logging
import os
import threading
import time
from datetime import datetime, timezone

import requests

from config import BIQUOTE_ENABLED, NEWS_BLACKOUT_MINS

logger = logging.getLogger(__name__)

PRIMARY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
BIQUOTE_CALENDAR_URL = "https://biquote.io/api/calendar/upcoming?importance=high&limit=200"
FALLBACK_CALENDAR_URL = os.environ.get("FALLBACK_CALENDAR_URL", "").strip()
CACHE_TTL_SEC = 3600
FAIL_RETRY_SEC = 300

_LOCK = threading.Lock()
_CACHE = {
    "ts": 0.0,
    "fail_ts": 0.0,
    "events": [],
    "source": "none",
}

GLOBAL_USD_KEYWORDS = (
    "FOMC", "NON-FARM", "NONFARM", "NFP", "CPI", "CORE CPI",
    "FEDERAL FUNDS", "POWELL", "GDP", "PPI", "RETAIL SALES", "PCE",
)

CURRENCY_MAP = {
    "US": "USD", "USA": "USD", "USD": "USD",
    "EU": "EUR", "EUR": "EUR", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR", "EMU": "EUR",
    "UK": "GBP", "GB": "GBP", "GBP": "GBP",
    "JP": "JPY", "JPY": "JPY",
    "AU": "AUD", "AUD": "AUD",
    "CA": "CAD", "CAD": "CAD",
    "CH": "CHF", "CHF": "CHF",
}


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


def _parse_dt(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except Exception:
        return None


def _parse_generic_events(raw_payload) -> list[dict]:
    items = raw_payload if isinstance(raw_payload, list) else (
        raw_payload.get("events") or raw_payload.get("data") or [] if isinstance(raw_payload, dict) else []
    )
    parsed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() == "holiday":
            continue
        imp = str(item.get("impact") or item.get("importance") or "").strip().lower()
        if imp not in ("high", "3", "critical"):
            continue
        dt = _parse_dt(item.get("date") or item.get("time") or item.get("datetime") or item.get("timestamp"))
        if dt is None:
            continue
        time_mode = str(item.get("timeMode") or "exact").strip().lower()
        raw_c = str(item.get("currency") or item.get("country") or item.get("countryCode") or "").strip().upper()
        ccy = CURRENCY_MAP.get(raw_c, raw_c)
        if not ccy:
            continue
        parsed.append({
            "country": ccy,
            "title": str(item.get("title") or item.get("name") or item.get("event") or "High Impact News").strip(),
            "dt": dt,
            "exact_time": time_mode == "exact",
        })
    return parsed


def _get_events() -> tuple[list[dict], bool]:
    now_ts = time.time()
    with _LOCK:
        if _CACHE["events"] and (now_ts - _CACHE["ts"]) < CACHE_TTL_SEC:
            return _CACHE["events"], True
        if _CACHE["fail_ts"] and (now_ts - _CACHE["fail_ts"]) < FAIL_RETRY_SEC:
            return _CACHE["events"], bool(_CACHE["events"])

    headers = {"User-Agent": "Mozilla/5.0 (compatible; RacioBot/2.2)"}
    urls = [("primary", PRIMARY_CALENDAR_URL)]
    if FALLBACK_CALENDAR_URL:
        urls.append(("env_fallback", FALLBACK_CALENDAR_URL))
    if BIQUOTE_ENABLED:
        urls.append(("biquote", BIQUOTE_CALENDAR_URL))

    for src_name, url in urls:
        try:
            resp = requests.get(url, timeout=6, headers=headers)
            if resp.status_code == 200:
                events = _parse_generic_events(resp.json())
                if events or src_name == "biquote":
                    with _LOCK:
                        _CACHE["events"] = events
                        _CACHE["ts"] = time.time()
                        _CACHE["fail_ts"] = 0.0
                        _CACHE["source"] = src_name
                    return events, True
        except Exception as e:
            logger.warning(f"Календар {src_name} недоступний: {e}")

    with _LOCK:
        _CACHE["fail_ts"] = time.time()
        return _CACHE["events"], bool(_CACHE["events"])


def is_high_impact_news_time(
    pair_name: str | None = None,
    now: datetime | None = None,
    blackout_mins: int | None = None,
) -> tuple[bool, str]:
    try:
        now_utc = now or datetime.now(timezone.utc)
        now_utc = now_utc.replace(tzinfo=timezone.utc) if now_utc.tzinfo is None else now_utc.astimezone(timezone.utc)

        window_sec = max(5, int(blackout_mins if blackout_mins is not None else NEWS_BLACKOUT_MINS)) * 60
        currencies = _extract_currencies(pair_name)
        events, ok = _get_events()

        if ok:
            for ev in events:
                if not ev.get("exact_time", True):
                    continue
                diff = abs((now_utc - ev["dt"]).total_seconds())
                if diff > window_sec:
                    continue
                country = ev["country"]
                title_up = ev["title"].upper()
                is_global_usd = country == "USD" and any(k in title_up for k in GLOBAL_USD_KEYWORDS)
                if country in currencies or is_global_usd:
                    return True, f"Новина ({country}): {ev['title']} о {ev['dt']:%H:%M} UTC"
            return False, "OK"

        if now_utc.weekday() == 4 and 1 <= now_utc.day <= 7 and 12 <= now_utc.hour <= 15:
            return True, "Резервний захист від NFP (перша п'ятниця місяця)"
        if now_utc.weekday() == 2 and now_utc.hour in (18, 19):
            return True, "Резервний захист від можливого засідання FOMC"
        return False, "OK"
    except Exception as e:
        logger.error(f"Помилка перевірки календаря новин: {e}")
        return False, "OK"


def get_upcoming_news(pair_name: str | None = None, now: datetime | None = None, hours: int = 12) -> list[dict]:
    try:
        now_utc = now or datetime.now(timezone.utc)
        now_utc = now_utc.replace(tzinfo=timezone.utc) if now_utc.tzinfo is None else now_utc.astimezone(timezone.utc)
        currencies = _extract_currencies(pair_name)
        events, ok = _get_events()
        if not ok:
            return []
        upcoming = []
        for ev in events:
            diff_sec = (ev["dt"] - now_utc).total_seconds()
            if -900 <= diff_sec <= hours * 3600:
                if ev["country"] in currencies or ev["country"] == "USD":
                    upcoming.append(ev)
        upcoming.sort(key=lambda e: e["dt"])
        return upcoming[:5]
    except Exception:
        return []
