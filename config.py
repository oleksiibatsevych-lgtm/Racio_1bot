import os

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Перевіряємо обидві можливі назви ключа Finnhub у змінних оточення Render
FINNHUB_TOKEN = os.environ.get("FINNHUB_TOKEN") or os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_API_KEY = FINNHUB_TOKEN

# Безпечне коригування URL бази даних PostgreSQL
raw_db_url = os.environ.get("DATABASE_URL", "")
if raw_db_url.startswith("postgres://"):
    raw_db_url = raw_db_url.replace("postgres://", "postgresql://", 1)

DATABASE_URL = raw_db_url.replace("&channel_binding=require", "").replace("?channel_binding=require", "")

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
    "CAD/CHF": "OANDA:CAD_CHF"
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
    "CAD/CHF": "CADCHF=X"
}
