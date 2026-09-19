import json
import logging
import threading
import time
import websocket
from config import FINNHUB_TOKEN

logger = logging.getLogger(__name__)

LIVE_PRICES = {}

FINNHUB_SYMBOL_MAP = {
    "EURUSD=X": "OANDA:EUR_USD",
    "GBPUSD=X": "OANDA:GBP_USD",
    "USDJPY=X": "OANDA:USD_JPY",
    "AUDUSD=X": "OANDA:AUD_USD",
    "USDCAD=X": "OANDA:USD_CAD",
    "USDCHF=X": "OANDA:USD_CHF",
    "EURJPY=X": "OANDA:EUR_JPY",
    "GBPJPY=X": "OANDA:GBP_JPY",
    "AUDJPY=X": "OANDA:AUD_JPY",
    "CADJPY=X": "OANDA:CAD_JPY",
    "CHFJPY=X": "OANDA:CHF_JPY",
    "EURGBP=X": "OANDA:EUR_GBP",
    "EURAUD=X": "OANDA:EUR_AUD",
    "EURCAD=X": "OANDA:EUR_CAD",
    "EURCHF=X": "OANDA:EUR_CHF",
    "GBPAUD=X": "OANDA:GBP_AUD",
    "GBPCAD=X": "OANDA:GBP_CAD",
    "GBPCHF=X": "OANDA:GBP_CHF",
    "AUDCAD=X": "OANDA:AUD_CAD",
    "AUDCHF=X": "OANDA:AUD_CHF",
    "CADCHF=X": "OANDA:CAD_CHF"
}

def on_message(ws, message):
    try:
        data = json.loads(message)
        if data.get("type") == "trade":
            for item in data.get("data", []):
                symbol = item.get("s")
                price = item.get("p")
                if symbol and price is not None:
                    fl_price = float(price)
                    LIVE_PRICES[symbol] = fl_price
                    for yf_ticker, fh_symbol in FINNHUB_SYMBOL_MAP.items():
                        if fh_symbol == symbol:
                            LIVE_PRICES[yf_ticker] = fl_price
    except Exception as e:
        logger.error(f"⚠️ Помилка обробки WebSocket Finnhub: {e}")

def on_error(ws, error):
    logger.error(f"❌ Finnhub WebSocket Помилка: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.warning("🔌 З'єднання Finnhub WebSocket закрито. Повторне підключення через 5 секунд...")
    time.sleep(5)
    start_finnhub_ws()

def on_open(ws):
    logger.info("🟢 Finnhub WebSocket підключено успішно!")
    for yf_ticker, fh_symbol in FINNHUB_SYMBOL_MAP.items():
        subscribe_msg = json.dumps({"type": "subscribe", "symbol": fh_symbol})
        ws.send(subscribe_msg)
        logger.info(f"📡 Підписано на символ Finnhub: {fh_symbol}")

_ws_thread_started = False

def start_finnhub_ws():
    global _ws_thread_started
    if _ws_thread_started:
        return
    if not FINNHUB_TOKEN:
        logger.error("❌ FINNHUB_TOKEN відсутній у змінних оточення!")
        return

    ws_url = f"wss://ws.finnhub.io?token={FINNHUB_TOKEN}"
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()
    _ws_thread_started = True

def get_live_price(ticker):
    return LIVE_PRICES.get(ticker)
