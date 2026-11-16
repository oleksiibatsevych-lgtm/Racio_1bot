# -*- coding: utf-8 -*-
import json
import logging
import re
import threading
import time
import websocket

from config import FINNHUB_API_KEY, FINNHUB_CONNECT_DELAY_SEC, FINNHUB_TOKEN, PAIRS_MAP
import tick_candles

logger = logging.getLogger(__name__)

live_prices = {}
ws_started = False
is_rate_limited = False
_stop_event = threading.Event()
_ws_lock = threading.Lock()
_active_ws = None
_last_tick_ts = 0.0


def _scrub_token(text: str) -> str:
    return re.sub(r"token=[^&\s'\"]+", "token=***", str(text))


def get_live_price(symbol: str):
    if not symbol:
        return None
    quote = get_live_quote(symbol)
    return quote[0] if quote else None


def get_live_quote(symbol: str):
    return live_prices.get(str(symbol).strip()) if symbol else None


def get_ws_health() -> dict:
    age = (time.time() - _last_tick_ts) if _last_tick_ts else None
    return {
        "started": ws_started,
        "rate_limited": is_rate_limited,
        "symbols_cached": len(live_prices),
        "last_tick_age_sec": round(age, 1) if age is not None else None,
    }


def on_message(ws, message):
    global _last_tick_ts
    try:
        data = json.loads(message)
        if data.get("type") == "trade":
            for item in data.get("data", []):
                sym = item.get("s")
                price = item.get("p")
                if sym and price:
                    ts = float(item.get("t") or time.time() * 1000) / 1000.0
                    live_prices[sym] = (float(price), ts)
                    _last_tick_ts = time.time()
                    tick_candles.add_tick(sym, price, item.get("t"))
    except Exception as e:
        logger.error(f"⚠️ Помилка обробки WS повідомлення: {_scrub_token(e)}")


def on_error(ws, error):
    global is_rate_limited
    err_str = _scrub_token(error)
    if "429" in err_str:
        logger.error("❌ Finnhub WS 429: Стара сесія ще активна на серверах Finnhub. Вмикаємо паузу...")
        is_rate_limited = True
    else:
        logger.error(f"❌ Finnhub WebSocket Error: {err_str}")


def on_close(ws, close_status_code, close_msg):
    logger.warning(f"⚠️ З'єднання Finnhub WebSocket закрите (Код: {close_status_code})")


def on_open(ws):
    global is_rate_limited
    is_rate_limited = False
    logger.info("🟢 Finnhub WebSocket успішно підключено!")
    for _, symbol in PAIRS_MAP.items():
        if symbol and not symbol.startswith("IC MARKETS:"):
            ws.send(json.dumps({"type": "subscribe", "symbol": symbol.strip()}))


def run_websocket_loop():
    global is_rate_limited, _active_ws
    token = (FINNHUB_API_KEY or FINNHUB_TOKEN or "").strip()
    if not token:
        logger.error("❌ Finnhub API Key відсутній!")
        return

    delay = max(0.0, float(FINNHUB_CONNECT_DELAY_SEC))
    if delay > 0:
        logger.info(f"⏳ Очікування {delay:.1f}с перед запуском Finnhub WS...")
        if _stop_event.wait(delay):
            return

    ws_url = f"wss://ws.finnhub.io?token={token}"

    while not _stop_event.is_set():
        try:
            logger.info("🔌 Підключення до Finnhub WebSocket...")
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            with _ws_lock:
                _active_ws = ws
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            logger.error(f"⚠️ Помилка в циклі WS: {_scrub_token(e)}")
        finally:
            with _ws_lock:
                _active_ws = None

        if _stop_event.is_set():
            break

        if is_rate_limited:
            logger.warning("⏳ Чекаємо 75 секунд для анулювання попередньої сесії на Finnhub...")
            _stop_event.wait(75)
            is_rate_limited = False
        else:
            logger.warning("⏳ Перепідключення WS через 15 секунд...")
            _stop_event.wait(15)


def start_finnhub_ws():
    global ws_started
    if ws_started:
        return
    ws_started = True
    _stop_event.clear()
    thread = threading.Thread(target=run_websocket_loop, daemon=True)
    thread.start()


def stop_finnhub_ws():
    _stop_event.set()
    with _ws_lock:
        if _active_ws is not None:
            try:
                _active_ws.close()
            except Exception:
                pass
