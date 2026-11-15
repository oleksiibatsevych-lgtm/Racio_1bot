# -*- coding: utf-8 -*-
import csv
import html
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import requests
import yfinance as yf
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import CallbackQueryHandler, CommandHandler, Filters, MessageHandler, Updater

load_dotenv()
import database
import tick_candles
from ai_advisor import ai_advisor_instance
from backtest import run_backtest, run_backtest_all_detailed, run_full_backtest, run_full_backtest_all
from calendar_filter import is_high_impact_news_time
from config import DISABLE_FINNHUB_CANDLES, FINNHUB_API_KEY, FINNHUB_TOKEN, PAIRS_MAP, YAHOO_PAIRS_MAP
from finnhub_ws import get_live_price, get_live_quote, start_finnhub_ws
from quality_gate import SegmentFilter, SignalGate
from reversion_engine import PRESETS, ReversionGuard, is_reversion
from strategy_engine import (
    ALL_STRATEGY_KEYS,
    GLOBAL_FILTERS,
    STRATEGIES_META,
    STRATEGY_CATEGORIES,
    generate_all_candidates as generate_candidates,
    get_strategy_expiration,
    is_known_strategy,
    load_all_strategy_config,
)
from telegram_views import PERIODS, active_text, history_text, settlement_text, signal_card, stats_text, wr_line
from trading_sessions import next_daily_report, period_start, session_allowed, utc_datetime

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
ADMIN_IDS = {int(v.strip()) for v in os.environ.get("ADMIN_IDS", "").split(",") if v.strip()}
STRATEGIES_CONFIG_FILE = Path(__file__).with_name("strategies_config.json")
strategies_status = load_all_strategy_config(STRATEGIES_CONFIG_FILE)
AUTO_SCAN_CHATS = set()
signal_gate = SignalGate()
reversion_guard = ReversionGuard()
segment_filter = SegmentFilter(database, break_even=0.54, min_trades=40)
bot = None
LAB_PROC = None

_DATA_LOCK = threading.Lock()
_FINNHUB_CANDLE_DISABLED_UNTIL = 0.0
_YAHOO_CACHE = {}        # (yahoo_ticker, interval) -> (ts, df)
_MERGED_1M_CACHE = {}    # finnhub_ticker -> (ts, df_1m)
_YF_TTL_BY_INTERVAL = {"1m": 45, "5m": 120, "15m": 600, "60m": 1800, "1h": 1800}


def require_admin(update):
    if not ADMIN_IDS or update.effective_user.id in ADMIN_IDS:
        return True
    update.effective_message.reply_text("⛔ Ця дія доступна лише адміністратору.")
    return False


def save_strategies_config(cfg):
    STRATEGIES_CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_dt(value):
    if not value:
        return get_utc_now()
    return utc_datetime(pd.Timestamp(value).to_pydatetime()).replace(tzinfo=None)


def normalize_yahoo_ticker(ticker, name=""):
    if name and name in YAHOO_PAIRS_MAP:
        return YAHOO_PAIRS_MAP[name]
    clean = ticker.replace("OANDA:", "").replace("IC MARKETS:", "").replace("_", "").replace("/", "")
    if len(clean) == 6 and not clean.endswith("=X"):
        return f"{clean}=X"
    return ticker


def fetch_finnhub_candles(symbol, resolution="1", count_candles=500):
    global _FINNHUB_CANDLE_DISABLED_UNTIL
    if DISABLE_FINNHUB_CANDLES:
        return pd.DataFrame()
    with _DATA_LOCK:
        if time.time() < _FINNHUB_CANDLE_DISABLED_UNTIL:
            return pd.DataFrame()

    token = FINNHUB_API_KEY or FINNHUB_TOKEN
    if not token:
        return pd.DataFrame()

    end_time = int(time.time())
    if resolution == "1":
        start_time = end_time - (count_candles * 60)
    elif resolution == "60":
        start_time = end_time - (count_candles * 3600)
    elif resolution == "D":
        start_time = end_time - (count_candles * 86400)
    else:
        start_time = end_time - (count_candles * 300)

    url = "https://finnhub.io/api/v1/forex/candle"
    params = {"symbol": str(symbol).strip(), "resolution": str(resolution).strip(), "from": start_time, "to": end_time, "token": str(token).strip()}

    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code in (401, 403):
            with _DATA_LOCK:
                _FINNHUB_CANDLE_DISABLED_UNTIL = time.time() + 21600
            logger.warning("Finnhub /forex/candle повернув %s (обмеження free tier) — переходжу на Yahoo + Tick Candles на 6 год.", resp.status_code)
            return pd.DataFrame()
        if resp.status_code == 200:
            data = resp.json()
            if data.get("s") == "ok":
                df = pd.DataFrame({"open": data["o"], "high": data["h"], "low": data["l"], "close": data["c"], "volume": data["v"]}, index=pd.to_datetime(data["t"], unit="s"))
                df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                return df
    except Exception as e:
        logger.warning(f"Finnhub REST API error for {symbol}: {e}")
    return pd.DataFrame()


def _clean_yf_df(df_yf: pd.DataFrame) -> pd.DataFrame:
    if df_yf is None or df_yf.empty:
        return pd.DataFrame()
    df_out = df_yf.copy()
    if isinstance(df_out.columns, pd.MultiIndex):
        df_out.columns = df_out.columns.get_level_values(0)
    df_out = df_out.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df_out.columns]
    if len(cols) < 4:
        return pd.DataFrame()
    df_out = df_out[cols].copy()
    if "volume" not in df_out.columns:
        df_out["volume"] = 0.0
    df_out.dropna(subset=["open", "high", "low", "close"], inplace=True)
    if df_out.index.tz is not None:
        df_out.index = df_out.index.tz_convert("UTC").tz_localize(None)
    return df_out.sort_index()


def fetch_yahoo_fallback(ticker, interval="1m", range_period="5d"):
    ttl = _YF_TTL_BY_INTERVAL.get(interval, 60)
    cache_key = (ticker, interval)
    with _DATA_LOCK:
        cached = _YAHOO_CACHE.get(cache_key)
        if cached and (time.time() - cached[0]) < ttl:
            return cached[1].copy()

    try:
        tk = yf.Ticker(ticker)
        df_yf = tk.history(period=range_period, interval=interval, auto_adjust=True)
        if df_yf is None or df_yf.empty:
            df_yf = yf.download(tickers=ticker, period=range_period, interval=interval, progress=False, auto_adjust=True)
        cleaned = _clean_yf_df(df_yf)
        if not cleaned.empty:
            with _DATA_LOCK:
                _YAHOO_CACHE[cache_key] = (time.time(), cleaned)
            return cleaned.copy()
    except Exception as e:
        logger.warning(f"Yahoo fallback failed for {ticker} ({interval}): {e}")
    return pd.DataFrame()


def prefetch_yahoo_batch(interval="1m", range_period="5d"):
    ttl = _YF_TTL_BY_INTERVAL.get(interval, 60)
    now_ts = time.time()
    needed = []
    with _DATA_LOCK:
        for y_ticker in YAHOO_PAIRS_MAP.values():
            cached = _YAHOO_CACHE.get((y_ticker, interval))
            if not cached or (now_ts - cached[0]) >= ttl:
                needed.append(y_ticker)
    if not needed:
        return
    try:
        raw = yf.download(
            tickers=" ".join(needed),
            period=range_period,
            interval=interval,
            group_by="ticker",
            threads=True,
            progress=False,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            return
        stamp = time.time()
        with _DATA_LOCK:
            if isinstance(raw.columns, pd.MultiIndex):
                top_levels = set(raw.columns.get_level_values(0))
                for y_ticker in needed:
                    if y_ticker in top_levels:
                        cleaned = _clean_yf_df(raw[y_ticker])
                        if not cleaned.empty:
                            _YAHOO_CACHE[(y_ticker, interval)] = (stamp, cleaned)
            elif len(needed) == 1:
                cleaned = _clean_yf_df(raw)
                if not cleaned.empty:
                    _YAHOO_CACHE[(needed[0], interval)] = (stamp, cleaned)
    except Exception as e:
        logger.warning(f"Yahoo batch prefetch error ({interval}): {e}")


def resample_candles(df_1m, rule):
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    return df_1m.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["open", "high", "low", "close"])


def get_exit_price(ticker, expires_at, name=""):
    quote = get_live_quote(ticker)
    now = get_utc_now()
    if quote:
        price, ts = quote
        quote_time = datetime.fromtimestamp(ts, timezone.utc).replace(tzinfo=None)
        if price > 0 and 0 <= (now - quote_time).total_seconds() <= 20 and abs((quote_time - expires_at).total_seconds()) <= 20:
            return float(price)
    minute = pd.Timestamp(expires_at).floor("min")
    ticks = tick_candles.get_1m(ticker, now_ts=now.replace(tzinfo=timezone.utc).timestamp())
    if ticks is not None and not ticks.empty and minute in ticks.index and ticks.loc[minute, "volume"] > 0 and minute + pd.Timedelta(minutes=1) <= now:
        return float(ticks.loc[minute, "close"])
    yahoo_ticker = normalize_yahoo_ticker(ticker, name)
    if yahoo_ticker:
        df_yf = fetch_yahoo_fallback(yahoo_ticker, interval="1m", range_period="1d")
        if df_yf is not None and not df_yf.empty and minute in df_yf.index and minute + pd.Timedelta(minutes=1) <= now:
            return float(df_yf.loc[minute, "close"])
    return None


def schedule_signal_timer(sig_id, timestamp_val, expiration_mins):
    try:
        signal_time = parse_dt(timestamp_val)
        delay = max(1, (signal_time + timedelta(minutes=expiration_mins) - get_utc_now()).total_seconds())
        timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
        timer.daemon = True
        timer.start()
    except Exception as e:
        logger.exception(f"Помилка планування таймера: {e}")


def start_background_checker():
    def loop():
        while True:
            try:
                for row in database.get_pending_signals():
                    expiry = parse_dt(row["expires_at"]) if row.get("expires_at") else parse_dt(row["timestamp_str"]) + timedelta(minutes=row["expiration_mins"])
                    if get_utc_now() >= expiry:
                        process_signal_expiration(row["id"])
            except Exception as e:
                logger.error(f"Помилка перевірки сигналів: {e}")
            time.sleep(20)
    threading.Thread(target=loop, daemon=True).start()


def _warm_scan_caches():
    prefetch_yahoo_batch(interval="1m", range_period="5d")
    if any(strategies_status.get(k, False) for k in ("BB_KC_REV_15M", "MACD_ADX_15M", "TTM_SQZ_15M", "MTF_PB_1M_15", "MTF_PB_3M_15", "FVG_IMP_3M")):
        prefetch_yahoo_batch(interval="15m", range_period="30d")
    if any(strategies_status.get(k, False) for k in ("MTF_PB_5M_60", "BB_KC_REV_5M", "BB_KC_REV_15M", "FVG_IMP_5M")):
        prefetch_yahoo_batch(interval="60m", range_period="60d")


def start_auto_scanner_loop():
    def loop():
        while True:
            try:
                if AUTO_SCAN_CHATS:
                    _warm_scan_caches()
                    for chat_id in list(AUTO_SCAN_CHATS):
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            futures = [
                                executor.submit(analyze_single_pair, chat_id, pair_name, ticker, auto_mode=True)
                                for pair_name, ticker in PAIRS_MAP.items()
                            ]
                            for f in futures:
                                try:
                                    f.result(timeout=30)
                                except Exception as ex:
                                    logger.error(f"Помилка потоку авто-сканера: {ex}")
            except Exception as e:
                logger.error(f"Помилка авто-сканера: {e}")
            time.sleep(max(1.0, 62.0 - (time.time() % 60)))
    threading.Thread(target=loop, daemon=True).start()


def handle_lab_run_command(update, context):
    """/lab_run — запускає лабораторію у фоні (скачує 1m-історію і перевіряє конфігурації)."""
    global LAB_PROC
    if not require_admin(update):
        return
    if LAB_PROC is not None and LAB_PROC.poll() is None:
        update.message.reply_text("⏳ Лабораторія вже працює. Результат: /lab")
        return
    base = os.path.dirname(os.path.abspath(__file__))
    try:
        log = open(os.path.join(base, "lab.log"), "w")
        LAB_PROC = subprocess.Popen(["nice", "-n", "19", sys.executable, "lab.py", "--download"],
                                    cwd=base, stdout=log, stderr=subprocess.STDOUT)
        update.message.reply_text("🧪 Лабораторію запущено у фоні (зазвичай 2–5 хв). Коли завершиться, надішліть /lab")
    except Exception as e:
        update.message.reply_text(f"❌ Не вдалося запустити лабораторію: {e}")


def handle_lab_command(update, context):
    """/lab — надсилає останній звіт лабораторії."""
    if not require_admin(update):
        return
    base = os.path.dirname(os.path.abspath(__file__))
    if LAB_PROC is not None and LAB_PROC.poll() is None:
        update.message.reply_text("⏳ Лабораторія ще рахує, спробуйте через хвилину.")
        return
    path = os.path.join(base, "lab_summary.txt")
    if not os.path.exists(path):
        update.message.reply_text("📭 Звіту ще немає. Запустіть /lab_run")
        return
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for i in range(0, len(text), 3500):
        update.message.reply_text(text[i:i + 3500])
    csv_path = os.path.join(base, "lab_report.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "rb") as f:
            update.message.reply_document(document=f, filename="lab_report.csv", caption="Повний звіт по всіх конфігураціях")


def fetch_5m(ticker, yahoo):
    minute = fetch_finnhub_candles(ticker, count_candles=1500)
    if minute.empty:
        minute = fetch_yahoo_fallback(yahoo, interval="1m", range_period="5d")
    minute = tick_candles.merge_with_history(ticker, minute)
    if minute is not None and not minute.empty:
        with _DATA_LOCK:
            _MERGED_1M_CACHE[ticker] = (time.time(), minute)
        return resample_candles(minute, "5min")
    return pd.DataFrame()


def build_mtf_frames(ticker, yahoo, df_5m):
    frames = {"5m": df_5m}
    with _DATA_LOCK:
        cached = _MERGED_1M_CACHE.get(ticker)
        df_1m = cached[1] if (cached and (time.time() - cached[0]) < 120) else pd.DataFrame()

    if (df_1m is None or df_1m.empty) and (df_5m is None or df_5m.empty):
        return frames

    if df_1m is not None and not df_1m.empty:
        frames["1m"] = df_1m
        frames["3m"] = resample_candles(df_1m, "3min")
        if df_5m is None or df_5m.empty:
            frames["5m"] = resample_candles(df_1m, "5min")

    need_15m = any(strategies_status.get(k, False) for k in ("BB_KC_REV_15M", "MACD_ADX_15M", "TTM_SQZ_15M", "MTF_PB_1M_15", "MTF_PB_3M_15", "FVG_IMP_3M"))
    if need_15m:
        df_15m = fetch_yahoo_fallback(yahoo, interval="15m", range_period="30d")
        if df_15m.empty and df_1m is not None and not df_1m.empty:
            df_15m = resample_candles(df_1m, "15min")
        frames["15m"] = df_15m

    need_60m = any(strategies_status.get(k, False) for k in ("MTF_PB_5M_60", "BB_KC_REV_5M", "BB_KC_REV_15M", "FVG_IMP_5M"))
    if need_60m:
        df_60m = fetch_yahoo_fallback(yahoo, interval="60m", range_period="60d")
        if not df_60m.empty:
            frames["60m"] = df_60m

    return frames


def notify_settlement(sig, result, exit_price, pips):
    report = settlement_text(result, exit_price, pips)
    markup = None
    if ai_advisor_instance.api_key and result in ("WIN", "LOSS", "NEUTRAL"):
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Розбір закритої угоди", callback_data=f"review|{sig['id']}")]])
    try:
        bot.edit_message_text(chat_id=sig["chat_id"], message_id=sig["message_id"],
                              text=(sig.get("message_text") or "") + "\n\n" + report,
                              parse_mode="HTML", reply_markup=markup)
    except Exception:
        logger.exception("Не вдалося оновити картку #%s", sig["id"])
    try:
        bot.send_message(chat_id=sig["chat_id"], text=f"#{sig['id']} " + report, parse_mode="HTML",
                         reply_to_message_id=sig["message_id"], allow_sending_without_reply=True)
    except Exception:
        logger.exception("Не вдалося надіслати результат #%s", sig["id"])


def process_signal_expiration(sig_id):
    claimed = False
    try:
        sig = database.get_signal_by_id(sig_id)
        if not sig or sig.get("status") != "PENDING":
            return
        expiry = parse_dt(sig["expires_at"]) if sig.get("expires_at") else parse_dt(sig["timestamp_str"]) + timedelta(minutes=sig["expiration_mins"])
        late = (get_utc_now() - expiry).total_seconds()
        if late < 0:
            timer = threading.Timer(-late, process_signal_expiration, args=[sig_id])
            timer.daemon = True
            timer.start()
            return
        if not database.claim_signal(sig_id):
            return
        claimed = True
        pair = next((name for name, ticker in PAIRS_MAP.items() if ticker == sig["ticker"]), "")
        exit_price = get_exit_price(sig["ticker"], expiry, pair)
        if not exit_price:
            if late <= 180:
                database.release_signal(sig_id)
                timer = threading.Timer(30, process_signal_expiration, args=[sig_id])
                timer.daemon = True
                timer.start()
                return
            database.update_signal_result(sig_id, "NEUTRAL", None, 0, status="VOID")
            notify_settlement(sig, "VOID", None, 0)
            return
        entry = float(sig["entry_price"])
        move = exit_price - entry if sig["signal_type"] == "CALL" else entry - exit_price
        result = "NEUTRAL" if exit_price == entry else ("WIN" if move > 0 else "LOSS")
        pips = int(round(move * (1000 if "JPY" in sig["ticker"] else 100000)))
        database.update_signal_result(sig_id, result, exit_price, pips)
        notify_settlement(sig, result, exit_price, pips)
    except Exception:
        logger.exception("Помилка таймера #%s", sig_id)
        if claimed:
            database.release_signal(sig_id)


def process_candidate(chat_id, pair_name, ticker_finnhub, cand, frames=None, auto_mode=False, status_msg_id=None, news_is_bad=False):
    if not is_known_strategy(cand):
        return False, "Невідома стратегія"
    if news_is_bad:
        return False, "Пауза через новини"
    tf_label = cand.get("tf") or "5m"
    tf_minutes = int(cand.get("tf_minutes") or 5)

    with reversion_guard.reserve(pair_name, cand["strategy"], ticker_finnhub, database.has_open_reversion_signal) as (ok, reason):
        if not ok:
            return False, reason
        if strategies_status.get("session_filter", False) and not session_allowed(cand["strategy"], cand["bar_end"]):
            return False, "Сесію відфільтровано"
        ok, reason = segment_filter.check(ticker_finnhub, tf_label, cand["strategy"], get_utc_now().hour)
        if not ok:
            return False, reason
        ok, reason = signal_gate.check(cand)
        if not ok:
            return False, reason
        live = get_live_price(ticker_finnhub)
        if live:
            adverse = cand["current_price"] - live if cand["signal"] == "CALL" else live - cand["current_price"]
            if adverse > 0.5 * cand["atr"]:
                return False, "Ціна пішла проти сигналу більше ніж на 0.5 ATR"
        now = get_utc_now()
        max_age = 45 if tf_minutes <= 1 else (90 if tf_minutes <= 3 else 180)
        if not 0 <= (now - cand["bar_end"]).total_seconds() <= max_age:
            return False, "Сигнал застарів"

        entry = float(live or cand["current_price"])
        expiration = get_strategy_expiration(cand)
        text = signal_card(pair_name, cand, entry, now)
        message = bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
        sig_id = database.save_signal(
            chat_id=chat_id,
            message_id=message.message_id,
            ticker=ticker_finnhub,
            signal_type=cand["signal"],
            entry_price=entry,
            primary_tf=tf_label,
            score=cand.get("confirmations", 0),
            expiration_mins=expiration,
            message_text=text,
            timestamp_str=now.strftime("%Y-%m-%d %H:%M:%S"),
            expires_at=now + timedelta(minutes=expiration),
            rsi=cand.get("rsi"),
            adx=cand.get("adx"),
            volatility_ratio=cand.get("vol_ratio"),
            divergence=cand.get("regime"),
            hour=now.hour,
            strategy=cand["strategy"],
        )
        if not sig_id:
            bot.send_message(chat_id=chat_id, text="⚠️ Сигнал не збережено в БД; автоматичний результат не відстежується.")
            return False, "Помилка збереження сигналу"
        signal_gate.register(cand, expiration)
        reversion_guard.register(pair_name, cand["strategy"], expiration)
        schedule_signal_timer(sig_id, now, expiration)
        return True, "OK"


def analyze_single_pair(chat_id, pair_name, ticker_finnhub, auto_mode=False, status_msg_id=None):
    try:
        if strategies_status.get("news_filter", True):
            bad_news, reason = is_high_impact_news_time(pair_name)
            if bad_news:
                return False
        yahoo = normalize_yahoo_ticker(ticker_finnhub, pair_name)
        df_5m = fetch_5m(ticker_finnhub, yahoo)
        frames = build_mtf_frames(ticker_finnhub, yahoo, df_5m)
        candidates = generate_candidates(
            pair_name,
            frames,
            get_utc_now(),
            dict(strategies_status),
            pair_enabled=database.get_pair_strategy_status,
        )
        for cand in candidates:
            ok, reason = process_candidate(chat_id, pair_name, ticker_finnhub, cand, frames=frames, auto_mode=auto_mode)
            if ok:
                return True
        return False
    except Exception:
        logger.exception("Помилка аналізу %s", pair_name)
        return False


def main_keyboard(chat_id):
    return ReplyKeyboardMarkup([
        [KeyboardButton("📡 Сигнал зараз"), KeyboardButton("🤖 Автоскан ON" if chat_id in AUTO_SCAN_CHATS else "🤖 Автоскан OFF")],
        [KeyboardButton("⏳ Активні угоди"), KeyboardButton("📜 Історія")],
        [KeyboardButton("📊 Статистика"), KeyboardButton("⚙️ Стратегії")],
    ], resize_keyboard=True)


def stats_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton(title, callback_data="stats|" + period)
                                  for period, title in PERIODS.items()]])


def strategy_keyboard(category=None, pair=None):
    if pair:
        buttons = []
        row = []
        for key in ALL_STRATEGY_KEYS:
            mark = "✅ " if database.get_pair_strategy_status(pair, key) else "❌ "
            row.append(InlineKeyboardButton(mark + key, callback_data=f"pair_toggle|{pair}|{key}"))
            if len(row) == 2:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        buttons.append([InlineKeyboardButton("⬅️ Назад до пар", callback_data="pairs")])
        return InlineKeyboardMarkup(buttons)

    if category and category in STRATEGY_CATEGORIES:
        buttons = []
        for key, meta in STRATEGIES_META.items():
            if meta["category"] != category:
                continue
            mark = "✅ " if strategies_status.get(key, False) else "❌ "
            buttons.append([InlineKeyboardButton(f"{mark}{meta['title']} ({meta['tf']}, {meta['exp']}хв)", callback_data="toggle|" + key)])
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="cat|MAIN")])
        return InlineKeyboardMarkup(buttons)

    buttons = []
    for cat_key, cat_title in STRATEGY_CATEGORIES.items():
        enabled_cnt = sum(1 for k, m in STRATEGIES_META.items() if m["category"] == cat_key and strategies_status.get(k, False))
        total_cnt = sum(1 for m in STRATEGIES_META.values() if m["category"] == cat_key)
        buttons.append([InlineKeyboardButton(f"{cat_title} [{enabled_cnt}/{total_cnt}]", callback_data="cat|" + cat_key)])
    for f_key, f_title in GLOBAL_FILTERS.items():
        mark = "✅ " if strategies_status.get(f_key, False) else "❌ "
        buttons.append([InlineKeyboardButton(mark + f_title, callback_data="toggle|" + f_key)])
    buttons.append([InlineKeyboardButton("🎯 Налаштування по парах", callback_data="pairs")])
    return InlineKeyboardMarkup(buttons)


def start(update, context):
    database.register_user(update.effective_chat.id, update.effective_user.username)
    update.effective_message.reply_text(
        "📡 Мульти-стратегічний бот (1m–15m + MTF/ICT/ORB/REV).\n"
        "Перевірте нові стратегії на ваших даних через /lab_run або /backtest перед торгівлею.",
        reply_markup=main_keyboard(update.effective_chat.id),
    )


def show_stats(chat_id, period="all"):
    rows = database.get_chat_signals(chat_id, since=period_start(period, get_utc_now()))
    return stats_text(rows, period)


def handle_message(update, context):
    chat_id = update.effective_chat.id
    message = update.effective_message
    text = message.text
    if text == "📡 Сигнал зараз":
        status = message.reply_text("📡 Оновлюю дані та перевіряю 21 пару по всіх активних стратегіях…")
        def scan():
            _warm_scan_caches()
            count = sum(analyze_single_pair(chat_id, pair, ticker) for pair, ticker in PAIRS_MAP.items())
            bot.edit_message_text(chat_id=chat_id, message_id=status.message_id, text=f"📡 Завершено · сигналів: {count}")
        threading.Thread(target=scan, daemon=True).start()
    elif text in ("🤖 Автоскан ON", "🤖 Автоскан OFF"):
        if chat_id in AUTO_SCAN_CHATS:
            AUTO_SCAN_CHATS.remove(chat_id)
            enabled = False
        else:
            AUTO_SCAN_CHATS.add(chat_id)
            enabled = True
        try:
            database.set_user_autoscan(chat_id, enabled)
        except Exception:
            pass
        message.reply_text("🤖 Автоскан " + ("ON" if enabled else "OFF"), reply_markup=main_keyboard(chat_id))
    elif text == "⏳ Активні угоди":
        reply_report(message, active_text(database.get_chat_signals(chat_id, "active"), get_utc_now()))
    elif text == "📜 Історія":
        reply_report(message, history_text(database.get_chat_signals(chat_id, "history")))
    elif text == "📊 Статистика":
        message.reply_text(show_stats(chat_id), parse_mode="HTML", reply_markup=stats_keyboard())
    elif text == "⚙️ Стратегії":
        if require_admin(update):
            message.reply_text("⚙️ Категорії стратегій та глобальні фільтри", reply_markup=strategy_keyboard())
    else:
        message.reply_text("Оберіть дію з меню.", reply_markup=main_keyboard(chat_id))


def reply_report(message, text):
    block = ""
    for line in text.splitlines():
        if len(block) + len(line) > 3500:
            message.reply_text(block, parse_mode="HTML")
            block = ""
        block += line + "\n"
    if block:
        message.reply_text(block, parse_mode="HTML")


def send_chunked_html(chat_id, text):
    block = ""
    for line in text.splitlines():
        if len(block) + len(line) > 3500:
            bot.send_message(chat_id=chat_id, text=block, parse_mode="HTML")
            block = ""
        block += line + "\n"
    if block:
        bot.send_message(chat_id=chat_id, text=block, parse_mode="HTML")


def handle_export_command(update, context):
    if not require_admin(update):
        return
    columns, rows = database.export_resolved_signals()
    if not rows:
        update.effective_message.reply_text("Завершених угод немає.")
        return
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows(rows)
    update.effective_message.reply_document(document=io.BytesIO(buffer.getvalue().encode("utf-8-sig")), filename="signals_export.csv")


def handle_callback(update, context):
    query = update.callback_query
    query.answer()
    chat_id = update.effective_chat.id
    parts = query.data.split("|")
    action = parts[0]
    if action == "stats" and len(parts) == 2 and parts[1] in PERIODS:
        query.edit_message_text(show_stats(chat_id, parts[1]), parse_mode="HTML", reply_markup=stats_keyboard())
    elif action in ("toggle", "cat", "pairs", "pair", "pair_toggle"):
        if not require_admin(update):
            return
        if action == "cat" and len(parts) == 2:
            cat = parts[1]
            if cat == "MAIN":
                query.edit_message_text("⚙️ Категорії стратегій та глобальні фільтри", reply_markup=strategy_keyboard())
            elif cat in STRATEGY_CATEGORIES:
                query.edit_message_text(STRATEGY_CATEGORIES[cat], reply_markup=strategy_keyboard(category=cat))
        elif action == "toggle" and len(parts) == 2 and (parts[1] in STRATEGIES_META or parts[1] in GLOBAL_FILTERS):
            key = parts[1]
            changed = dict(strategies_status)
            changed[key] = not changed.get(key, False)
            save_strategies_config(changed)
            strategies_status.update(changed)
            if key in STRATEGIES_META:
                cat = STRATEGIES_META[key]["category"]
                query.edit_message_text(STRATEGY_CATEGORIES[cat], reply_markup=strategy_keyboard(category=cat))
            else:
                query.edit_message_text("⚙️ Категорії стратегій та глобальні фільтри", reply_markup=strategy_keyboard())
        elif action == "pairs":
            pair_list = list(PAIRS_MAP.keys())
            rows = []
            row = []
            for pair in pair_list:
                row.append(InlineKeyboardButton(pair, callback_data="pair|" + pair))
                if len(row) == 3:
                    rows.append(row)
                    row = []
            if row:
                rows.append(row)
            rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="cat|MAIN")])
            query.edit_message_text("🎯 Оберіть пару", reply_markup=InlineKeyboardMarkup(rows))
        elif action == "pair" and len(parts) == 2 and parts[1] in PAIRS_MAP:
            query.edit_message_text(f"🎯 {parts[1]} · перемикачі стратегій", reply_markup=strategy_keyboard(pair=parts[1]))
        elif action == "pair_toggle" and len(parts) == 3 and parts[1] in PAIRS_MAP and parts[2] in STRATEGIES_META:
            pair, key = parts[1:]
            database.set_pair_strategy_status(pair, key, not database.get_pair_strategy_status(pair, key))
            query.edit_message_text(f"🎯 {pair} · перемикачі стратегій", reply_markup=strategy_keyboard(pair=pair))
    elif action == "review" and len(parts) == 2 and parts[1].isdigit():
        sig = database.get_signal_by_id(int(parts[1]))
        if not sig or sig["chat_id"] != chat_id or sig["status"] != "CLOSED":
            return
        def review():
            result = ai_advisor_instance.evaluate_closed_trade(sig)
            database.save_signal_ai_review(sig["id"], result)
            bot.send_message(chat_id=chat_id, text="🤖 <b>Розбір (не причинний доказ):</b>\n" + html.escape(result[:3000]), parse_mode="HTML")
        threading.Thread(target=review, daemon=True).start()


def handle_backtest_command(update, context):
    args = [a.upper() for a in (context.args or [])]
    rev_only = "REV" in args
    args = [a for a in args if a != "REV"]
    pair = args[0] if args else "ALL"
    if pair != "ALL" and pair not in PAIRS_MAP:
        update.effective_message.reply_text("Невідома пара. Приклади: /backtest EUR/USD, /backtest ALL, /backtest EUR/USD REV")
        return
    chat_id = update.effective_chat.id
    mode_label = "REV 5m (сесії OFF/ON)" if rev_only else "всі активні стратегії (1m–15m + 60m HTF)"
    update.effective_message.reply_text(f"📊 Запускаю бектест ({pair} · {mode_label})…")
    settings = dict(strategies_status)

    def worker():
        try:
            if rev_only:
                reports = run_backtest_all_detailed(strategies_status=settings) if pair == "ALL" else [run_backtest(pair, strategies_status=settings)]
            else:
                reports = [run_full_backtest_all(strategies_status=settings)] if pair == "ALL" else [run_full_backtest(pair, strategies_status=settings)]
            for report in reports:
                send_chunked_html(chat_id, report)
        except Exception:
            logger.exception("Помилка бектесту")
            bot.send_message(chat_id=chat_id, text="❌ Не вдалося виконати бектест.")
    threading.Thread(target=worker, daemon=True).start()


def send_daily_report(now=None):
    now = now or get_utc_now()
    for chat_id in list(AUTO_SCAN_CHATS):
        try:
            rows = database.get_chat_signals(chat_id, since=period_start("today", now), until=utc_datetime(now).replace(tzinfo=None))
            lines = ["🌙 <b>Денний звіт · сьогодні (Київ)</b>", wr_line("Усього", rows)]
            for preset in PRESETS:
                lines.append(wr_line(preset, [row for row in rows if row.get("strategy") == preset]))
            active_new = sorted({r.get("strategy") for r in rows if r.get("strategy") and r.get("strategy") not in PRESETS})
            for st in active_new[:10]:
                lines.append(wr_line(st, [row for row in rows if row.get("strategy") == st]))
            bot.send_message(chat_id=chat_id, text="\n".join(lines), parse_mode="HTML")
        except Exception:
            logger.exception("Помилка денного звіту для %s", chat_id)


def start_daily_report_loop():
    def loop():
        while True:
            now = utc_datetime(get_utc_now())
            target = next_daily_report(now)
            time.sleep(max(1, (target - now).total_seconds()))
            send_daily_report()
    threading.Thread(target=loop, daemon=True).start()


def main():
    global bot
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN не задано")
    if not ADMIN_IDS:
        logger.warning("ADMIN_IDS не задано: глобальне керування доступне всім користувачам.")
    bot = Bot(token=token)
    database.init_db()
    AUTO_SCAN_CHATS.update(database.get_autoscan_chats())
    start_finnhub_ws()
    start_background_checker()
    start_auto_scanner_loop()
    start_daily_report_loop()
    bot.delete_webhook(drop_pending_updates=True)
    updater = Updater(token=token, use_context=True)
    updater.dispatcher.add_handler(CommandHandler("start", start))
    updater.dispatcher.add_handler(CommandHandler("backtest", handle_backtest_command))
    updater.dispatcher.add_handler(CommandHandler("export", handle_export_command))
    updater.dispatcher.add_handler(CommandHandler("lab_run", handle_lab_run_command))
    updater.dispatcher.add_handler(CommandHandler("lab", handle_lab_command))
    updater.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.dispatcher.add_handler(CallbackQueryHandler(handle_callback))
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
