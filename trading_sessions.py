# -*- coding: utf-8 -*-
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
KYIV = ZoneInfo("Europe/Kyiv")
LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")

SUPPORTED_TIMEZONES = {
    "Europe/Kyiv": ("Київ", KYIV),
    "UTC": ("UTC", UTC),
    "Europe/London": ("Лондон", LONDON),
    "America/New_York": ("Нью-Йорк", NEW_YORK),
}

ALLOWED_SESSIONS = {
    "REV_RSI14": {"London", "Overlap", "Other"},
    "REV_BB3": {"NY", "Other"},
    "REV_RSI7": {"London", "NY", "Overlap", "Other"},
}

ACTIVE_FLOW = {"London", "Overlap", "NY"}
ALL_EXCEPT_DEAD = {"London", "Overlap", "NY", "Other"}
ANY_SESSION = {"London", "Overlap", "NY", "Asia", "Other"}

STRATEGY_SESSIONS = {
    "CRSI_5M": ALL_EXCEPT_DEAD,
    "CRSI_1M": ACTIVE_FLOW,
    "VALUE_CH_5M": ALL_EXCEPT_DEAD,
    "VALUE_CH_1M": ACTIVE_FLOW,
    "OSC_CONS_5M": ALL_EXCEPT_DEAD,
    "OSC_CONS_3M": ACTIVE_FLOW,
    "BB_KC_REV_5M": ALL_EXCEPT_DEAD,
    "BB_KC_REV_15M": ALL_EXCEPT_DEAD,
    "PIVOT_REV_5M": ALL_EXCEPT_DEAD,
    "EXH_STRK_1M": ACTIVE_FLOW,
    "EXH_STRK_3M": ALL_EXCEPT_DEAD,
    "STARC_5M": ALL_EXCEPT_DEAD,
    "STARC_15M": ALL_EXCEPT_DEAD,
    "LINREG_5M": ALL_EXCEPT_DEAD,
    "HUBBA_5M": ALL_EXCEPT_DEAD,
    "MTF_PB_1M_15": ACTIVE_FLOW,
    "MTF_PB_3M_15": ACTIVE_FLOW,
    "MTF_PB_5M_60": ACTIVE_FLOW,
    "EMA_RIB_3M": ACTIVE_FLOW,
    "EMA_RIB_5M": ACTIVE_FLOW,
    "MACD_ADX_5M": ACTIVE_FLOW,
    "MACD_ADX_15M": ACTIVE_FLOW,
    "ST_MOM_3M": ACTIVE_FLOW,
    "ST_MOM_5M": ACTIVE_FLOW,
    "PSAR_EMA_5M": ACTIVE_FLOW,
    "PSAR_EMA_15M": ACTIVE_FLOW,
    "ICHI_TK_5M": ACTIVE_FLOW,
    "ICHI_TK_15M": ACTIVE_FLOW,
    "HA_EMA_5M": ACTIVE_FLOW,
    "TTM_SQZ_5M": ACTIVE_FLOW,
    "TTM_SQZ_15M": ACTIVE_FLOW,
    "ORB_LON_5M": {"London", "Overlap"},
    "ORB_NY_5M": {"Overlap", "NY"},
    "DONCH_BRK_5M": ACTIVE_FLOW,
    "JUDAS_LON_5M": {"London", "Other"},
    "DIV_RSI_5M": ALL_EXCEPT_DEAD,
    "DIV_RSI_15M": ALL_EXCEPT_DEAD,
    "DIV_MACD_5M": ALL_EXCEPT_DEAD,
    "DIV_MACD_15M": ALL_EXCEPT_DEAD,
    "AO_TWIN_5M": ALL_EXCEPT_DEAD,
    "AO_ZERO_3M": ACTIVE_FLOW,
    "CCI_WPR_5M": ALL_EXCEPT_DEAD,
    "CCI_WPR_3M": ACTIVE_FLOW,
    "DEM_BB_5M": ALL_EXCEPT_DEAD,
    "WT_CROSS_5M": ALL_EXCEPT_DEAD,
    "LIQ_SWP_3M": ACTIVE_FLOW,
    "LIQ_SWP_5M": ACTIVE_FLOW,
    "FVG_IMP_3M": ACTIVE_FLOW,
    "FVG_IMP_5M": ACTIVE_FLOW,
    "PA_PIN_BB_5M": ALL_EXCEPT_DEAD,
    "PA_ENG_SR_5M": ALL_EXCEPT_DEAD,
    "PA_FAKEY_5M": ACTIVE_FLOW,
    "PA_MHI_1M": ANY_SESSION,
    "PA_3LS_5M": ALL_EXCEPT_DEAD,
}


def utc_datetime(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def get_tz_info(tz_name: str = "Europe/Kyiv"):
    return SUPPORTED_TIMEZONES.get(tz_name, ("Київ", KYIV))


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
    return session_name(value) in ACTIVE_FLOW


def session_allowed(preset, value):
    if preset in ALLOWED_SESSIONS:
        return session_name(value) in ALLOWED_SESSIONS[preset]
    return session_name(value) in STRATEGY_SESSIONS.get(preset, ACTIVE_FLOW)


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
