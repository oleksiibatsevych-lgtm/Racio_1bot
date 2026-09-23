import json
import logging
import os
import threading
import time
import websocket
from config import FINNHUB_API_KEY, FINNHUB_TOKEN, PAIRS_MAP

logger = logging.getLogger(__name__)

live_prices = {}
ws_started = False
is_rate_limited = False


def get_live_price(symbol: str):
    """Отримання поточної ціни з WebSocket."""
    if not symbol:
        return None
    clean_sym = str(symbol).strip()
    return live_prices.get(clean_sym)


def on_message(ws, message):
    try:
        data = json.loads(message)
        if data.get("type") == "trade":
            for item in data.get("data", []):
                sym = item.get("s")
                price = item.get("p")
                if sym and price:
                    live_prices[sym] = float(price)
    except Exception as e:
        logger.error(f"⚠️ Помилка обробки WS повідомлення: {e}")


def on_error(ws, error):
    global is_rate_limited
    err_str = str(error)
    if "429" in err_str:
        logger.error(
            "❌ Finnhub WS 429: Стара сесія ще активна на серверах Finnhub. Вмикаємо затримку..."
        )
        is_rate_limited = True
    else:
        logger.error(f"❌ Finnhub WebSocket Error: {error}")


def on_close(ws, close_status_code, close_msg):
    logger.warning(
        f"⚠️ З'єднання Finnhub WebSocket закрите (Код: {close_status_code})"
    )


def on_open(ws):
    global is_rate_limited
    is_rate_limited = False
    logger.info("🟢 Finnhub WebSocket успішно підключено!")
    for name, symbol in PAIRS_MAP.items():
        if symbol and not symbol.startswith("IC MARKETS:"):
            sub_msg = json.dumps(
                {"type": "subscribe", "symbol": symbol.strip()}
            )
            ws.send(sub_msg)


def run_websocket_loop():
    global is_rate_limited
    token = (FINNHUB_API_KEY or FINNHUB_TOKEN or "").strip()
    if not token:
        logger.error("❌ Finnhub API Key відсутній!")
        return

    ws_url = f"wss://ws.finnhub.io?token={token}"

    while True:
        try:
            logger.info("🔌 Підключення до Finnhub WebSocket...")
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            logger.error(f"⚠️ Помилка в циклі WS: {e}")

        # Якщо отримали 429, чекаємо 75 сек, щоб старий контейнер/сесія повністю завершилася на боці Finnhub
        if is_rate_limited:
            logger.warning(
                "⏳ Чекаємо 75 секунд для анулювання попередньої сесії на Finnhub..."
            )
            time.sleep(75)
            is_rate_limited = False
        else:
            logger.warning("⏳ Перепідключення WS через 15 секунд...")
            time.sleep(15)


def start_finnhub_ws():
    """Запуск WebSocket у фоновому потоці."""
    global ws_started
    if ws_started:
        return
    ws_started = True

    thread = threading.Thread(target=run_websocket_loop, daemon=True)
    thread.start()
