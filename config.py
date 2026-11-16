# -*- coding: utf-8 -*-
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def _sanitize_database_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    url = raw_url.strip()
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    try:
        parts = urlsplit(url)
        query_items = [
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if k.lower() != "channel_binding"
        ]
        host = (parts.hostname or "").lower()
        has_ssl = any(k.lower() == "sslmode" for k, _ in query_items)
        if host and host not in ("localhost", "127.0.0.1", "::1") and not has_ssl:
            query_items.append(("sslmode", "require"))
        clean_query = urlencode(query_items)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, clean_query, parts.fragment))
    except Exception:
        return url


def _parse_int_set(env_value: str) -> set[int]:
    result = set()
    for part in str(env_value or "").split(","):
        p = part.strip()
        if p.lstrip("-").isdigit():
            result.add(int(p))
    return result


TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

FINNHUB_TOKEN = os.environ.get("FINNHUB_TOKEN") or os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_API_KEY = FINNHUB_TOKEN
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")

ADMIN_IDS = _parse_int_set(os.environ.get("ADMIN_IDS", ""))
ALLOWED_CHAT_IDS = _parse_int_set(os.environ.get("ALLOWED_CHAT_IDS", ""))
ALLOW_OPEN_ADMIN = os.environ.get("ALLOW_OPEN_ADMIN", "false").lower() in ("1", "true", "yes")

DATABASE_URL = _sanitize_database_url(os.environ.get("DATABASE_URL", ""))

PAYOUT = float(os.environ.get("PAYOUT", "0.85"))
NEWS_BLACKOUT_MINS = int(os.environ.get("NEWS_BLACKOUT_MINS", "15"))
DISABLE_FINNHUB_CANDLES = os.environ.get("DISABLE_FINNHUB_CANDLES", "false").lower() in ("1", "true", "yes")
FINNHUB_CONNECT_DELAY_SEC = float(os.environ.get("FINNHUB_CONNECT_DELAY_SEC", "4.0"))

PRUNE_WR_THRESHOLD = float(os.environ.get("PRUNE_WR_THRESHOLD", "0.45"))
AUTO_DISABLE_WR_THRESHOLD = float(os.environ.get("AUTO_DISABLE_WR_THRESHOLD", "0.49"))
AUTO_DISABLE_MIN_LIVE_PAIR = int(os.environ.get("AUTO_DISABLE_MIN_LIVE_PAIR", "8"))
AUTO_DISABLE_MIN_LIVE_GLOBAL = int(os.environ.get("AUTO_DISABLE_MIN_LIVE_GLOBAL", "15"))
AUTO_DISABLE_MIN_BT_PAIR = int(os.environ.get("AUTO_DISABLE_MIN_BT_PAIR", "15"))
AUTO_DISABLE_MIN_BT_GLOBAL = int(os.environ.get("AUTO_DISABLE_MIN_BT_GLOBAL", "30"))

CONFLUENCE_MODE = os.environ.get("CONFLUENCE_MODE", "CONC").upper()
CONFLUENCE_THRESHOLD = float(os.environ.get("CONFLUENCE_THRESHOLD", "1.0"))
EXPIRY_MODE = os.environ.get("EXPIRY_MODE", "DYN").upper()
DEFAULT_TIMEZONE = os.environ.get("DEFAULT_TIMEZONE", "Europe/Kyiv")

DEFAULT_DEPOSIT = 0.0
DEFAULT_RISK_PCT = 0.0
MAX_ACTIVE_PER_CURRENCY = int(os.environ.get("MAX_ACTIVE_PER_CURRENCY", "2"))

PAIRS_MAP = {
    "EUR/USD": "OANDA:EUR_USD",
    "GBP/USD": "OANDA:GBP_USD",
    "USD/JPY": "OANDA:USD_JPY",
    "AUD/USD": "OANDA:AUD_USD",
    "USD/CAD": "OANDA:USD_CAD",
    "USD/CHF": "OANDA:USD_CHF",
    "EUR/JPY": "OANDA:EUR_JPY",
    "GBP/JPY": "OANDA:GBP_JPY",
    "AUD/JPY": "OANDA:AUD_JPY",
    "CAD/JPY": "OANDA:CAD_JPY",
    "CHF/JPY": "OANDA:CHF_JPY",
    "EUR/GBP": "OANDA:EUR_GBP",
    "EUR/AUD": "OANDA:EUR_AUD",
    "EUR/CAD": "OANDA:EUR_CAD",
    "EUR/CHF": "OANDA:EUR_CHF",
    "GBP/AUD": "OANDA:GBP_AUD",
    "GBP/CAD": "OANDA:GBP_CAD",
    "GBP/CHF": "OANDA:GBP_CHF",
    "AUD/CAD": "OANDA:AUD_CAD",
    "AUD/CHF": "OANDA:AUD_CHF",
    "CAD/CHF": "OANDA:CAD_CHF",
}

YAHOO_PAIRS_MAP = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "USDCAD=X",
    "USD/CHF": "USDCHF=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "CAD/JPY": "CADJPY=X",
    "CHF/JPY": "CHFJPY=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/AUD": "EURAUD=X",
    "EUR/CAD": "EURCAD=X",
    "EUR/CHF": "EURCHF=X",
    "GBP/AUD": "GBPAUD=X",
    "GBP/CAD": "GBPCAD=X",
    "GBP/CHF": "GBPCHF=X",
    "AUD/CAD": "AUDCAD=X",
    "AUD/CHF": "AUDCHF=X",
    "CAD/CHF": "CADCHF=X",
}

REVERSE_PAIRS_MAP = {ticker: name for name, ticker in PAIRS_MAP.items()}
REVERSE_YAHOO_MAP = {ticker: name for name, ticker in YAHOO_PAIRS_MAP.items()}


def normalize_pair_name(ticker: str | None) -> str:
    if not ticker:
        return ""
    raw = str(ticker).strip()
    if raw in PAIRS_MAP:
        return raw
    if raw in REVERSE_PAIRS_MAP:
        return REVERSE_PAIRS_MAP[raw]
    if raw in REVERSE_YAHOO_MAP:
        return REVERSE_YAHOO_MAP[raw]
    cleaned = raw.replace("OANDA:", "").replace("IC MARKETS:", "").replace("=X", "").strip()
    if "_" in cleaned:
        cleaned = cleaned.replace("_", "/")
    elif "/" not in cleaned and len(cleaned) == 6 and cleaned.isalpha():
        cleaned = f"{cleaned[:3]}/{cleaned[3:]}"
    return cleaned
