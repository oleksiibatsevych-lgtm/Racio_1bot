# -*- coding: utf-8 -*-
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
KYIV = ZoneInfo("Europe/Kyiv")
LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")

ALLOWED_SESSIONS = {
    "REV_RSI14": {"London", "Overlap", "Other"},
    "REV_BB3": {"NY", "Other"},
    "REV_RSI7": {"London", "NY", "Overlap", "Other"},
}

STRATEGY_SESSIONS = {
    "ORB_LON_5M": {"London", "Overlap"},
    "ORB_NY_5M": {"Overlap", "NY"},
    "LIQ_SWP_3M": {"London", "Overlap", "NY"},
    "LIQ_SWP_5M": {"London", "Overlap", "NY"},
    "FVG_IMP_3M": {"London", "Overlap", "NY"},
    "FVG_IMP_5M": {"London", "Overlap", "NY"},
    "TTM_SQZ_5M": {"London", "Overlap", "NY"},
    "TTM_SQZ_15M": {"London", "Overlap", "NY"},
    "DONCH_BRK_5M": {"London", "Overlap", "NY"},
    "MTF_PB_1M_15": {"London", "Overlap", "NY"},
    "MTF_PB_3M_15": {"London", "Overlap", "NY"},
    "MTF_PB_5M_60": {"London", "Overlap", "NY"},
    "EMA_RIB_3M": {"London", "Overlap", "NY"},
    "EMA_RIB_5M": {"London", "Overlap", "NY"},
    "MACD_ADX_5M": {"London", "Overlap", "NY"},
    "MACD_ADX_15M": {"London", "Overlap", "NY"},
    "ST_MOM_3M": {"London", "Overlap", "NY"},
    "ST_MOM_5M": {"London", "Overlap", "NY"},
    "PA_PIN_BB_5M": {"London", "Overlap", "NY", "Other"},
    "PA_ENG_SR_5M": {"London", "Overlap", "NY", "Other"},
    "PA_FAKEY_5M": {"London", "Overlap", "NY"},
    "CRSI_1M": {"London", "Overlap", "NY", "Other"},
    "CRSI_5M": {"London", "Overlap", "NY", "Other"},
    "VALUE_CH_1M": {"London", "Overlap", "NY", "Other"},
    "VALUE_CH_5M": {"London", "Overlap", "NY", "Other"},
    "OSC_CONS_3M": {"London", "Overlap", "NY", "Other"},
    "OSC_CONS_5M": {"London", "Overlap", "NY", "Other"},
    "BB_KC_REV_5M": {"London", "Overlap", "NY", "Other"},
    "BB_KC_REV_15M": {"London", "Overlap", "NY", "Other"},
    "PIVOT_REV_5M": {"London", "Overlap", "NY", "Other"},
    "EXH_STRK_1M": {"London", "Overlap", "NY", "Other"},
    "EXH_STRK_3M": {"London", "Overlap", "NY", "Other"},
}


def utc_datetime(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def london_time(value):
    return utc_datetime(value).astimezone(LONDON)


def ny_time(value):
    return utc_datetime(value).astimezone(NEW_YORK)


def session_name(value):
    value = utc_datetime(value)
    london = 8 <= value.astimezone(LONDON).hour < 16
    ny = 8 <= value.astimezone(NEW_YORK).hour < 16
    if london and ny:
        return "Overlap"
    if london:
        return "London"
    if ny:
        return "NY"
    return "Asia" if value.hour < 7 else "Other"


def is_killzone(value):
    return session_name(value) in {"London", "Overlap", "NY"}


def session_allowed(preset, value):
    if preset in ALLOWED_SESSIONS:
        return session_name(value) in ALLOWED_SESSIONS[preset]
    allowed = STRATEGY_SESSIONS.get(preset, {"London", "Overlap", "NY"})
    return session_name(value) in allowed


def next_daily_report(now):
    local = utc_datetime(now).astimezone(KYIV)
    day = local.date() + timedelta(days=local.time() >= time(21))
    return datetime.combine(day, time(21), KYIV).astimezone(UTC)


def period_start(period, now):
    now = utc_datetime(now)
    if period == "today":
        return datetime.combine(now.astimezone(KYIV).date(), time(), KYIV).astimezone(UTC).replace(tzinfo=None)
    if period in ("7", "30"):
        return (now - timedelta(days=int(period))).replace(tzinfo=None)
    if period == "all":
        return None
    raise ValueError("Unknown statistics period")
