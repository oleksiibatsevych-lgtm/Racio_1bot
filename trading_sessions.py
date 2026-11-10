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


def utc_datetime(value):
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


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


def session_allowed(preset, value):
    return session_name(value) in ALLOWED_SESSIONS.get(preset, set())


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
