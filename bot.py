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
from backtest import (
    build_backtest_csv,
    run_backtest,
    run_backtest_all_detailed,
    run_full_backtest_all_with_stats,
    run_full_backtest_with_stats,
)
from calendar_filter import get_upcoming_news, is_high_impact_news_time
from config import (
    AUTO_DISABLE_WR_THRESHOLD,
    CONFLUENCE_MODE,
    CONFLUENCE_THRESHOLD,
    DEFAULT_DEPOSIT,
    DEFAULT_RISK_PCT,
    DEFAULT_TIMEZONE,
    DISABLE_FINNHUB_CANDLES,
    EXPIRY_MODE,
    FINNHUB_API_KEY,
    FINNHUB_TOKEN,
    NEWS_BLACKOUT_MINS,
    PAIRS_MAP,
    PAYOUT,
    YAHOO_PAIRS_MAP,
    normalize_pair_name,
)
from finnhub_ws import get_live_price, get_live_quote, start_finnhub_ws
from quality_gate import SegmentFilter, SignalGate
from reversion_engine import PRESETS, ReversionGuard
from strategy_engine import (
    ALL_STRATEGY_KEYS,
    CONFLUENCE_MODES,
    EXPIRY_MODES,
    GLOBAL_FILTERS,
    STRATEGIES_META,
    STRATEGY_CATEGORIES,
    diagnose_pair_market,
    generate_all_candidates as generate_candidates,
    get_strategy_expiration,
    is_known_strategy,
    load_all_strategy_config,
    resolve_confluence,
)
from telegram_views import (
    PERIODS,
    active_text,
    auto_disabled_text,
    history_text,
    pair_radar_text,
    settlement_text,
    signal_card,
    stats_matrix_text,
    stats_text,
    strategy_stat_text,
    wr_line,
)
from trading_sessions import SUPPORTED_TIMEZONES, next_daily_report, period_start, session_allowed, utc_datetime

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

STRATEGIES_15M = tuple(k for k, m in STRATEGIES_META.items() if m["tf"] == "15m" or m.get("htf") == 15)
STRATEGIES_60M = tuple(k for k, m in STRATEGIES_META.items() if m.get("htf") == 60)

_DATA_LOCK = threading.Lock()
_FINNHUB_CANDLE_DISABLED_UNTIL = 0.0
_YAHOO_CACHE = {}
_MERGED_1M_CACHE = {}
_YF_TTL_BY_INTERVAL = {"1m": 45, "5m": 120, "15m": 600, "60m": 1800, "1h": 1800}


def require_admin(update):
    if not ADMIN_IDS or update.effective_user.id in ADMIN_IDS:
        return True
    update.effective_message.reply_text("⛔ Ця дія доступна лише адміністратору.")
    return False


def save_strategies_config(cfg):
    STRATEGIES_CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_runtime_settings():
    defaults = {
        "wr_threshold": float(AUTO_DISABLE_WR_THRESHOLD),
        "confluence_threshold": float(CONFLUENCE_THRESHOLD),
        "conflict_mode": str(CONFLUENCE_MODE),
        "expiry_mode": str(EXPIRY_MODE),
        "news_blackout_mins": int(NEWS_BLACKOUT_MINS),
        "payout": float(PAYOUT),
        "timezone": str(DEFAULT_TIMEZONE),
        "risk_deposit": float(DEFAULT_DEPOSIT),
        "risk_pct": float(DEFAULT_RISK_PCT),
    }
    try:
        raw = database.get_all_bot_settings()
        if not isinstance(raw, dict):
            return defaults
        if "wr_threshold" in raw:
            defaults["wr_threshold"] = float(raw["wr_threshold"])
        if "confluence_threshold" in raw:
            defaults["confluence_threshold"] = float(raw["confluence_threshold"])
        if "conflict_mode" in raw and raw["conflict_mode"] in CONFLUENCE_MODES:
            defaults["conflict_mode"] = raw["conflict_mode"]
        if "expiry_mode" in raw and raw["expiry_mode"] in EXPIRY_MODES:
            defaults["expiry_mode"] = raw["expiry_mode"]
        if "news_blackout_mins" in raw:
            defaults["news_blackout_mins"] = int(float(raw["news_blackout_mins"]))
        if "payout" in raw:
            defaults["payout"] = float(raw["payout"])
        if "timezone" in raw and raw["timezone"] in SUPPORTED_TIMEZONES:
            defaults["timezone"] = raw["timezone"]
        if "risk_deposit" in raw:
            defaults["risk_deposit"] = float(raw["risk_deposit"])
        if "risk_pct" in raw:
            defaults["risk_pct"] = float(raw["risk_pct"])
    except Exception:
        pass
    return defaults


def _safe_edit(query, text, parse_mode=None, reply_markup=None):
    try:
        query.edit_message_text(text=text, parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as e:
        if "not modified" not in str(e).lower():
            raise


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
            logger.warning("Finnhub /forex/candle повернув %s — переходжу на Yahoo + Tick Candles на 6 год.", resp.status_code)
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


def _clean_yf_df(df_yf):
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
    if any(strategies_status.get(k, False) for k in STRATEGIES_15M):
        prefetch_yahoo_batch(interval="15m", range_period="30d")
    if any(strategies_status.get(k, False) for k in STRATEGIES_60M):
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


def _sync_auto_disable_events(events, notify_chat_id=None, wr_threshold=None):
    if not isinstance(events, dict):
        return ""
    th = float(wr_threshold) if isinstance(wr_threshold, (int, float)) else _get_runtime_settings()["wr_threshold"]
    disabled_global = events.get("disabled_global") or []
    disabled_pair = events.get("disabled_pair") or []
    lines = []

    if disabled_global:
        changed = dict(strategies_status)
        for ev in disabled_global:
            s_key = ev["strategy"]
            if s_key in changed:
                changed[s_key] = False
            lines.append(
                f"⛔ Глобально вимкнено <code>{html.escape(s_key)}</code>: "
                f"{ev['wr'] * 100:.1f}% ({ev['wins']}/{ev['total']}) &lt; {th * 100:.0f}%"
            )
        save_strategies_config(changed)
        strategies_status.update(changed)

    for ev in disabled_pair:
        lines.append(
            f"⛔ По парі <b>{html.escape(ev['pair'])}</b> вимкнено <code>{html.escape(ev['strategy'])}</code>: "
            f"{ev['wr'] * 100:.1f}% ({ev['wins']}/{ev['total']}) &lt; {th * 100:.0f}%"
        )

    summary = "\n".join(lines)
    if summary and notify_chat_id and bot is not None:
        try:
            bot.send_message(
                chat_id=notify_chat_id,
                text=f"🛡 <b>Авто-контроль (&lt;{th * 100:.0f}% WR):</b>\n" + summary + "\n<i>Можна ввімкнути вручну у меню «⚙️ Стратегії».</i>",
                parse_mode="HTML",
            )
        except Exception:
            logger.exception("Не вдалося надіслати сповіщення про авто-вимкнення")
    return summary


def _apply_live_auto_disable(sig):
    try:
        if not isinstance(sig, dict) or not sig.get("strategy"):
            return
        if not strategies_status.get("auto_disable_49", True):
            return
        rt = _get_runtime_settings()
        events = database.evaluate_and_apply_auto_disable(
            source="LIVE",
            pair_filter=sig.get("ticker"),
            strategy_filter=sig.get("strategy"),
            threshold=rt["wr_threshold"],
        )
        _sync_auto_disable_events(events, notify_chat_id=sig.get("chat_id"), wr_threshold=rt["wr_threshold"])
    except Exception:
        logger.exception("Помилка перевірки авто-вимкнення після закриття угоди")


def handle_lab_run_command(update, context):
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


def build_mtf_frames(ticker, yahoo, df_5m, include_all_tfs=False):
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

    if include_all_tfs or any(strategies_status.get(k, False) for k in STRATEGIES_15M):
        df_15m = fetch_yahoo_fallback(yahoo, interval="15m", range_period="30d")
        if df_15m.empty and df_1m is not None and not df_1m.empty:
            df_15m = resample_candles(df_1m, "15min")
        frames["15m"] = df_15m

    if include_all_tfs or any(strategies_status.get(k, False) for k in STRATEGIES_60M):
        df_60m = fetch_yahoo_fallback(yahoo, interval="60m", range_period="60d")
        if not df_60m.empty:
            frames["60m"] = df_60m

    return frames


def signal_action_keyboard(strategy_key):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🔕 Пауза 1г", callback_data=f"mute1h|{strategy_key}"),
            InlineKeyboardButton("📊 Стат. стратегії", callback_data=f"st_stat|{strategy_key}"),
        ],
        [
            InlineKeyboardButton("💰 Розрахунок ставки", callback_data="risk|menu"),
            InlineKeyboardButton("🏠 Меню", callback_data="main_menu"),
        ],
    ])


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
        if result in ("WIN", "LOSS"):
            _apply_live_auto_disable(sig)
    except Exception:
        logger.exception("Помилка таймера #%s", sig_id)
        if claimed:
            database.release_signal(sig_id)


def process_candidate(chat_id, pair_name, ticker_finnhub, cand, frames=None, auto_mode=False, status_msg_id=None, news_is_bad=False):
    if not is_known_strategy(cand):
        return False, "Невідома стратегія"
    if news_is_bad:
        return False, "Пауза через новини"
    try:
        mute_res = database.is_strategy_muted(cand["strategy"])
        if isinstance(mute_res, tuple) and mute_res[0]:
            return False, f"Пауза на {mute_res[1]} хв"
    except Exception:
        pass

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

        rt = _get_runtime_settings()
        entry = float(live or cand["current_price"])
        expiration = get_strategy_expiration(cand)
        risk_amt = rt["risk_deposit"] * (rt["risk_pct"] / 100.0)
        text = signal_card(pair_name, cand, entry, now, tz_name=rt["timezone"], risk_amount=risk_amt)
        message = bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=signal_action_keyboard(cand["strategy"]),
        )
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


def _filter_unmuted(candidates):
    eligible = []
    for c in candidates:
        try:
            m = database.is_strategy_muted(c["strategy"])
            if isinstance(m, tuple) and m[0]:
                continue
        except Exception:
            pass
        eligible.append(c)
    return eligible


def analyze_single_pair(chat_id, pair_name, ticker_finnhub, auto_mode=False, status_msg_id=None):
    try:
        rt = _get_runtime_settings()
        if strategies_status.get("news_filter", True):
            bad_news, reason = is_high_impact_news_time(pair_name, blackout_mins=rt["news_blackout_mins"])
            if bad_news:
                return False
        yahoo = normalize_yahoo_ticker(ticker_finnhub, pair_name)
        df_5m = fetch_5m(ticker_finnhub, yahoo)
        frames = build_mtf_frames(ticker_finnhub, yahoo, df_5m)
        raw_candidates = generate_candidates(
            pair_name,
            frames,
            get_utc_now(),
            dict(strategies_status),
            pair_enabled=database.get_pair_strategy_status,
        )
        eligible = _filter_unmuted(raw_candidates)
        if rt["conflict_mode"] != "IND" or rt["expiry_mode"] != "FIX":
            candidates, _ = resolve_confluence(
                eligible,
                conflict_mode=rt["conflict_mode"],
                threshold=rt["confluence_threshold"],
                expiry_mode=rt["expiry_mode"],
            )
        else:
            candidates = eligible

        for cand in candidates:
            ok, reason = process_candidate(chat_id, pair_name, ticker_finnhub, cand, frames=frames, auto_mode=auto_mode)
            if ok:
                return True
        return False
    except Exception:
        logger.exception("Помилка аналізу %s", pair_name)
        return False


def analyze_single_pair_report(chat_id, pair_name, ticker_finnhub):
    try:
        rt = _get_runtime_settings()
        if strategies_status.get("news_filter", True):
            bad_news, reason = is_high_impact_news_time(pair_name, blackout_mins=rt["news_blackout_mins"])
            if bad_news:
                return f"📰 <b>{html.escape(pair_name)}</b>: сканування призупинено\nПричина: {html.escape(reason)}"

        yahoo = normalize_yahoo_ticker(ticker_finnhub, pair_name)
        df_5m = fetch_5m(ticker_finnhub, yahoo)
        frames = build_mtf_frames(ticker_finnhub, yahoo, df_5m)
        if df_5m is None or df_5m.empty:
            return f"⚠️ <b>{html.escape(pair_name)}</b>: не вдалося отримати котирування."

        now = get_utc_now()
        raw_candidates = generate_candidates(
            pair_name,
            frames,
            now,
            dict(strategies_status),
            pair_enabled=database.get_pair_strategy_status,
        )
        eligible = _filter_unmuted(raw_candidates)
        candidates, conf_reason = resolve_confluence(
            eligible,
            conflict_mode=rt["conflict_mode"],
            threshold=rt["confluence_threshold"],
            expiry_mode=rt["expiry_mode"],
        )

        skipped = []
        for cand in candidates:
            ok, reason = process_candidate(chat_id, pair_name, ticker_finnhub, cand, frames=frames, auto_mode=False)
            if ok:
                return (
                    f"✅ <b>{html.escape(pair_name)}</b>: сигнал сформовано!\n"
                    f"<code>{html.escape(cand['strategy'])}</code> · {cand['signal']} · {get_strategy_expiration(cand)} хв"
                )
            skipped.append(f"{cand['strategy']}: {reason}")

        diag = diagnose_pair_market(
            pair_name,
            frames,
            now,
            dict(strategies_status),
            pair_enabled=database.get_pair_strategy_status,
            mute_check=database.is_strategy_muted,
        )
        chop_s = f"{diag['chop']:.1f}" if diag["chop"] is not None else "—"
        adx_s = f"{diag['adx']:.1f}" if diag["adx"] is not None else "—"
        rsi_s = f"{diag['rsi14']:.1f}" if diag["rsi14"] is not None else "—"
        msg = (
            f"📡 <b>{html.escape(pair_name)}</b>: зараз сигналів немає\n"
            f"📊 5m режим: CHOP <b>{chop_s}</b> · ADX <b>{adx_s}</b> · RSI14 <b>{rsi_s}</b> · Сесія: {diag['session']}\n"
            f"⚙️ Активних стратегій: <b>{diag['checked']}</b> (глоб.вимк: {diag['off_global']}, пара: {diag['off_pair']}, пауза: {diag.get('muted', 0)}, сесія: {diag['off_session']})"
        )
        if not raw_candidates:
            msg += "\n📉 Причина: індикатори ще не сформували вхідний паттерн на закритій свічці."
        elif not candidates:
            msg += f"\n⚖️ Причина: {html.escape(conf_reason)}"
        elif skipped:
            msg += "\n⏭ Відфільтровано: " + html.escape("; ".join(skipped[:3]))
        return msg
    except Exception:
        logger.exception("Помилка діагностики %s", pair_name)
        return f"❌ Помилка під час аналізу {html.escape(pair_name)}."


def main_keyboard(chat_id):
    return ReplyKeyboardMarkup([
        [KeyboardButton("📡 Сигнал зараз"), KeyboardButton("🤖 Автоскан ON" if chat_id in AUTO_SCAN_CHATS else "🤖 Автоскан OFF")],
        [KeyboardButton("⏳ Активні угоди"), KeyboardButton("📜 Історія")],
        [KeyboardButton("📊 Статистика"), KeyboardButton("🧪 Бектест")],
        [KeyboardButton("🎛 Налаштування"), KeyboardButton("⚙️ Стратегії")],
    ], resize_keyboard=True)


def inline_main_keyboard(chat_id):
    auto_lbl = "🤖 Автоскан: ON" if chat_id in AUTO_SCAN_CHATS else "🤖 Автоскан: OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📡 Сигнал зараз (пара / всі)", callback_data="scan|menu")],
        [
            InlineKeyboardButton("🔍 Радар пари", callback_data="check_menu"),
            InlineKeyboardButton("🧪 Бектест", callback_data="bt_menu|all"),
        ],
        [
            InlineKeyboardButton("📊 Статистика", callback_data="stats|all"),
            InlineKeyboardButton("⏳ Активні угоди", callback_data="active_refresh"),
        ],
        [
            InlineKeyboardButton("🎛 Пороги і фільтри", callback_data="settings|main"),
            InlineKeyboardButton("⚙️ Стратегії", callback_data="cat|MAIN"),
        ],
        [
            InlineKeyboardButton("💰 Ризик-калькулятор", callback_data="risk|menu"),
            InlineKeyboardButton(auto_lbl, callback_data="settings|main"),
        ],
    ])


def stats_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(title, callback_data="stats|" + period) for period, title in PERIODS.items()],
        [InlineKeyboardButton("📋 Матриця Пара × Стратегія", callback_data="stats_matrix")],
        [
            InlineKeyboardButton("🧹 Очистити статистику", callback_data="stats_clear_ask"),
            InlineKeyboardButton("🏠 Меню", callback_data="main_menu"),
        ],
    ])


def clear_stats_confirm_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Так, очистити все", callback_data="stats_clear_confirm"),
            InlineKeyboardButton("❌ Скасувати", callback_data="stats|all"),
        ]
    ])


def active_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Оновити ціни (Live)", callback_data="active_refresh")],
        [InlineKeyboardButton("🏠 Головне меню", callback_data="main_menu")],
    ])


def scan_keyboard():
    buttons = [[InlineKeyboardButton("🌐 Сканувати всі 21 пару", callback_data="scan|all")]]
    row = []
    for pair in PAIRS_MAP:
        row.append(InlineKeyboardButton(pair, callback_data=f"scan|pair|{pair}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([
        InlineKeyboardButton("🔍 Радар пари (діагностика)", callback_data="check_menu"),
        InlineKeyboardButton("🏠 Меню", callback_data="main_menu"),
    ])
    return InlineKeyboardMarkup(buttons)


def check_pairs_keyboard():
    buttons = []
    row = []
    for pair in PAIRS_MAP:
        row.append(InlineKeyboardButton(pair, callback_data=f"check_go|{pair}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("🏠 Головне меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


def backtest_keyboard(mode="all"):
    next_mode = "act" if mode == "all" else "all"
    mode_txt = "📊 Режим: УСІ 57 стратегій (змінити)" if mode == "all" else "📊 Режим: Лише активні (змінити)"
    buttons = [
        [InlineKeyboardButton(mode_txt, callback_data="bt_menu|" + next_mode)],
        [InlineKeyboardButton("🌐 Бектест ВСІХ 21 пар", callback_data=f"bt_go|ALL|{mode}")],
    ]
    row = []
    for pair in PAIRS_MAP:
        row.append(InlineKeyboardButton(pair, callback_data=f"bt_go|{pair}|{mode}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("🏠 Головне меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


def settings_menu_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    wr_pct = int(round(rt["wr_threshold"] * 100))
    pay_pct = int(round(rt["payout"] * 100))
    tz_lbl = SUPPORTED_TIMEZONES.get(rt["timezone"], ("Київ",))[0]
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎚 Поріг авто-вимкнення: {wr_pct}%", callback_data="settings|thresholds")],
        [InlineKeyboardButton(f"🧩 Конфлюентність: {rt['conflict_mode']} · ≥ {rt['confluence_threshold']:.1f}", callback_data="settings|confluence")],
        [InlineKeyboardButton(f"⏱ Експірація: {rt['expiry_mode']} (Золота сер. / Фікс)", callback_data="settings|expiry")],
        [InlineKeyboardButton(f"📰 Фільтри / Новини ({rt['news_blackout_mins']}хв) · Випл. {pay_pct}%", callback_data="settings|filters")],
        [
            InlineKeyboardButton(f"🕒 Пояс: {tz_lbl}", callback_data="settings|tz"),
            InlineKeyboardButton(f"💰 Ризик: ${rt['risk_deposit']:.0f} ({rt['risk_pct']:.0f}%)", callback_data="settings|risk"),
        ],
        [
            InlineKeyboardButton("⚙️ Категорії стратегій", callback_data="cat|MAIN"),
            InlineKeyboardButton("🎯 По парах", callback_data="pairs"),
        ],
        [InlineKeyboardButton("🏠 Головне меню", callback_data="main_menu")],
    ])


def thresholds_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    cur = int(round(rt["wr_threshold"] * 100))
    auto_on = "✅ Авто-вимкнення: ON" if strategies_status.get("auto_disable_49", True) else "❌ Авто-вимкнення: OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(auto_on, callback_data="ftog|auto_disable_49")],
        [
            InlineKeyboardButton(("• 45% •" if cur == 45 else "45%"), callback_data="set_wr|45"),
            InlineKeyboardButton(("• 49% •" if cur == 49 else "49%"), callback_data="set_wr|49"),
            InlineKeyboardButton(("• 52% •" if cur == 52 else "52%"), callback_data="set_wr|52"),
        ],
        [
            InlineKeyboardButton(("• 54% •" if cur == 54 else "54%"), callback_data="set_wr|54"),
            InlineKeyboardButton(("• 55% •" if cur == 55 else "55%"), callback_data="set_wr|55"),
        ],
        [InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")],
    ])


def confluence_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    c_mod = rt["conflict_mode"]
    c_th = rt["confluence_threshold"]
    buttons = []
    for k, title in CONFLUENCE_MODES.items():
        mark = "✅ " if c_mod == k else ""
        buttons.append([InlineKeyboardButton(mark + title, callback_data=f"set_cmod|{k}")])
    buttons.append([
        InlineKeyboardButton(("• 1.0 •" if abs(c_th - 1.0) < 0.05 else "1.0"), callback_data="set_conf|1.0"),
        InlineKeyboardButton(("• 1.5 •" if abs(c_th - 1.5) < 0.05 else "1.5"), callback_data="set_conf|1.5"),
        InlineKeyboardButton(("• 2.0 •" if abs(c_th - 2.0) < 0.05 else "2.0"), callback_data="set_conf|2.0"),
    ])
    buttons.append([
        InlineKeyboardButton(("• 2.5 •" if abs(c_th - 2.5) < 0.05 else "2.5"), callback_data="set_conf|2.5"),
        InlineKeyboardButton(("• 3.0 •" if abs(c_th - 3.0) < 0.05 else "3.0"), callback_data="set_conf|3.0"),
    ])
    buttons.append([InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")])
    return InlineKeyboardMarkup(buttons)


def expiry_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    e_mod = rt["expiry_mode"]
    buttons = []
    for k, title in EXPIRY_MODES.items():
        mark = "✅ " if e_mod == k else ""
        buttons.append([InlineKeyboardButton(mark + title, callback_data=f"set_emod|{k}")])
    buttons.append([InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")])
    return InlineKeyboardMarkup(buttons)


def filters_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    nm = rt["news_blackout_mins"]
    pay = int(round(rt["payout"] * 100))
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(("✅ " if strategies_status.get("news_filter", True) else "❌ ") + "Новини", callback_data="ftog|news_filter"),
            InlineKeyboardButton(("✅ " if strategies_status.get("session_filter", True) else "❌ ") + "Сесії", callback_data="ftog|session_filter"),
            InlineKeyboardButton(("✅ " if strategies_status.get("regime_filter", True) else "❌ ") + "Режим", callback_data="ftog|regime_filter"),
        ],
        [
            InlineKeyboardButton(("• ±5хв •" if nm == 5 else "±5хв"), callback_data="set_news|5"),
            InlineKeyboardButton(("• ±15хв •" if nm == 15 else "±15хв"), callback_data="set_news|15"),
            InlineKeyboardButton(("• ±30хв •" if nm == 30 else "±30хв"), callback_data="set_news|30"),
        ],
        [
            InlineKeyboardButton(("• 80% •" if pay == 80 else "80%"), callback_data="set_pay|80"),
            InlineKeyboardButton(("• 85% •" if pay == 85 else "85%"), callback_data="set_pay|85"),
            InlineKeyboardButton(("• 88% •" if pay == 88 else "88%"), callback_data="set_pay|88"),
            InlineKeyboardButton(("• 90% •" if pay == 90 else "90%"), callback_data="set_pay|90"),
        ],
        [InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")],
    ])


def tz_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    cur = rt["timezone"]
    buttons = []
    for k, (label, _) in SUPPORTED_TIMEZONES.items():
        mark = "✅ " if cur == k else ""
        buttons.append([InlineKeyboardButton(f"{mark}{label} ({k})", callback_data=f"set_tz|{k}")])
    buttons.append([InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")])
    return InlineKeyboardMarkup(buttons)


def risk_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    dep = int(round(rt["risk_deposit"]))
    rsk = int(round(rt["risk_pct"]))
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(("• $100 •" if dep == 100 else "$100"), callback_data="set_dep|100"),
            InlineKeyboardButton(("• $500 •" if dep == 500 else "$500"), callback_data="set_dep|500"),
            InlineKeyboardButton(("• $1000 •" if dep == 1000 else "$1000"), callback_data="set_dep|1000"),
            InlineKeyboardButton(("• $5000 •" if dep == 5000 else "$5000"), callback_data="set_dep|5000"),
        ],
        [
            InlineKeyboardButton(("• 1% •" if rsk == 1 else "1%"), callback_data="set_rsk|1"),
            InlineKeyboardButton(("• 2% •" if rsk == 2 else "2%"), callback_data="set_rsk|2"),
            InlineKeyboardButton(("• 3% •" if rsk == 3 else "3%"), callback_data="set_rsk|3"),
            InlineKeyboardButton(("• 5% •" if rsk == 5 else "5%"), callback_data="set_rsk|5"),
        ],
        [InlineKeyboardButton("⬅️ Налаштування", callback_data="settings|main"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")],
    ])


def pairs_keyboard(auto_map=None):
    auto_map = auto_map or {}
    rows = []
    row = []
    for pair in PAIRS_MAP:
        has_auto = any(st == "PAIR" and p == pair for (st, p, _) in auto_map)
        label = ("⚠️ " + pair) if has_auto else pair
        row.append(InlineKeyboardButton(label, callback_data="pair|" + pair))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="cat|MAIN")])
    return InlineKeyboardMarkup(rows)


def auto_disabled_keyboard(items):
    buttons = []
    for it in (items if isinstance(items, (list, tuple)) else [])[:25]:
        key = it["strategy_key"]
        wr = float(it.get("win_rate") or 0.0) * 100.0
        if it["scope_type"] == "GLOBAL":
            label = f"🔓 Глобально: {key} ({wr:.0f}%)"
            cb = f"autodis_en|G|{key}"
        else:
            p = it["pair_name"]
            label = f"🔓 {p}: {key} ({wr:.0f}%)"
            cb = f"autodis_en|P|{p}|{key}"
        buttons.append([InlineKeyboardButton(label, callback_data=cb)])
    buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="cat|MAIN")])
    return InlineKeyboardMarkup(buttons)


def strategy_keyboard(category=None, pair=None, auto_map=None):
    if auto_map is None:
        try:
            raw_map = database.get_auto_disabled_map()
            auto_map = raw_map if isinstance(raw_map, dict) else {}
        except Exception:
            auto_map = {}

    if pair and category and category in STRATEGY_CATEGORIES:
        buttons = []
        for key, meta in STRATEGIES_META.items():
            if meta["category"] != category:
                continue
            enabled = database.get_pair_strategy_status(pair, key)
            auto_rec = auto_map.get(("PAIR", pair, key))
            if auto_rec and not enabled:
                wr = float(auto_rec.get("win_rate") or 0.0) * 100.0
                mark = f"⛔ ({wr:.0f}%) "
            else:
                mark = "✅ " if enabled else "❌ "
            buttons.append([InlineKeyboardButton(f"{mark}{meta['title']} ({meta['tf']}, {meta['exp']}хв)", callback_data=f"pair_toggle|{pair}|{key}")])
        buttons.append([
            InlineKeyboardButton("✅ Увімкнути групу", callback_data=f"pcat_all|{pair}|{category}|1"),
            InlineKeyboardButton("❌ Вимкнути групу", callback_data=f"pcat_all|{pair}|{category}|0"),
        ])
        buttons.append([InlineKeyboardButton("⬅️ До категорій пари", callback_data=f"pair|{pair}")])
        return InlineKeyboardMarkup(buttons)

    if pair:
        buttons = []
        for cat_key, cat_title in STRATEGY_CATEGORIES.items():
            cat_keys = [k for k, m in STRATEGIES_META.items() if m["category"] == cat_key]
            enabled_cnt = sum(1 for k in cat_keys if database.get_pair_strategy_status(pair, k))
            has_auto = any(("PAIR", pair, k) in auto_map for k in cat_keys)
            warn = " ⛔" if has_auto else ""
            buttons.append([InlineKeyboardButton(f"{cat_title} [{enabled_cnt}/{len(cat_keys)}]{warn}", callback_data=f"pcat|{pair}|{cat_key}")])
        buttons.append([
            InlineKeyboardButton("✅ Увімкнути всі для пари", callback_data=f"pair_all|{pair}|1"),
            InlineKeyboardButton("❌ Вимкнути всі", callback_data=f"pair_all|{pair}|0"),
        ])
        buttons.append([InlineKeyboardButton("⬅️ Назад до пар", callback_data="pairs")])
        return InlineKeyboardMarkup(buttons)

    if category and category in STRATEGY_CATEGORIES:
        buttons = []
        for key, meta in STRATEGIES_META.items():
            if meta["category"] != category:
                continue
            enabled = strategies_status.get(key, False)
            auto_rec = auto_map.get(("GLOBAL", "*", key))
            if auto_rec and not enabled:
                wr = float(auto_rec.get("win_rate") or 0.0) * 100.0
                mark = f"⛔ ({wr:.0f}%) "
            else:
                mark = "✅ " if enabled else "❌ "
            buttons.append([InlineKeyboardButton(f"{mark}{meta['title']} ({meta['tf']}, {meta['exp']}хв)", callback_data="toggle|" + key)])
        buttons.append([
            InlineKeyboardButton("✅ Увімкнути групу", callback_data=f"cat_all|{category}|1"),
            InlineKeyboardButton("❌ Вимкнути групу", callback_data=f"cat_all|{category}|0"),
        ])
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="cat|MAIN")])
        return InlineKeyboardMarkup(buttons)

    buttons = []
    for cat_key, cat_title in STRATEGY_CATEGORIES.items():
        cat_keys = [k for k, m in STRATEGIES_META.items() if m["category"] == cat_key]
        enabled_cnt = sum(1 for k in cat_keys if strategies_status.get(k, False))
        has_auto = any(("GLOBAL", "*", k) in auto_map for k in cat_keys)
        warn = " ⛔" if has_auto else ""
        buttons.append([InlineKeyboardButton(f"{cat_title} [{enabled_cnt}/{len(cat_keys)}]{warn}", callback_data="cat|" + cat_key)])
    for f_key, f_title in GLOBAL_FILTERS.items():
        mark = "✅ " if strategies_status.get(f_key, False) else "❌ "
        buttons.append([InlineKeyboardButton(mark + f_title, callback_data="toggle|" + f_key)])
    buttons.append([InlineKeyboardButton("🎯 Налаштування по парах", callback_data="pairs")])
    auto_cnt = len(auto_map)
    buttons.append([
        InlineKeyboardButton(f"🚫 Авто-вимкнені [{auto_cnt}]", callback_data="autodis_list"),
        InlineKeyboardButton("🎛 Пороги", callback_data="settings|main"),
    ])
    return InlineKeyboardMarkup(buttons)


def _settings_summary_text(rt=None):
    rt = rt or _get_runtime_settings()
    be = 100.0 / (1.0 + rt["payout"])
    stake = rt["risk_deposit"] * (rt["risk_pct"] / 100.0)
    return (
        "🎛 <b>Панель керування порогами та фільтрами</b>\n"
        f"• Авто-вимкнення WR: <b>&lt;{rt['wr_threshold'] * 100:.0f}%</b>\n"
        f"• Конфлюентність: <b>{rt['conflict_mode']}</b> (поріг <b>{rt['confluence_threshold']:.1f}</b>)\n"
        f"• Експірація: <b>{rt['expiry_mode']}</b>\n"
        f"• Новини: <b>±{rt['news_blackout_mins']} хв</b> · Виплата: <b>{int(rt['payout'] * 100)}%</b> (беззбиток {be:.1f}%)\n"
        f"• Депозит: <b>${rt['risk_deposit']:.0f}</b> · Ризик: <b>{rt['risk_pct']:.1f}%</b> (ставка <b>${stake:.2f}</b>)"
    )


def _build_pair_radar(pair_name):
    rt = _get_runtime_settings()
    ticker = PAIRS_MAP[pair_name]
    yahoo = normalize_yahoo_ticker(ticker, pair_name)
    df_5m = fetch_5m(ticker, yahoo)
    frames = build_mtf_frames(ticker, yahoo, df_5m, include_all_tfs=True)
    diag = diagnose_pair_market(
        pair_name,
        frames,
        get_utc_now(),
        dict(strategies_status),
        pair_enabled=database.get_pair_strategy_status,
        mute_check=database.is_strategy_muted,
    )
    news_st = is_high_impact_news_time(pair_name, blackout_mins=rt["news_blackout_mins"])
    upcoming = get_upcoming_news(pair_name, hours=12)
    live_rows = database.get_detailed_pair_strategy_stats(ticker_filter=ticker)
    bt_rows = database.get_backtest_results(pair_filter=pair_name)
    lp = get_live_price(ticker)
    return pair_radar_text(
        diag,
        news_st,
        upcoming,
        live_rows,
        bt_rows,
        live_price=lp,
        payout=rt["payout"],
        wr_threshold=rt["wr_threshold"],
    )


def start(update, context):
    database.register_user(update.effective_chat.id, update.effective_user.username)
    update.effective_message.reply_text(
        "📡 <b>Racio_1bot · 57 стратегій (1m–15m) + Конфлюентність</b>\n"
        "Оберіть дію на панелі або відкрийте інлайн-меню (/menu):",
        parse_mode="HTML",
        reply_markup=main_keyboard(update.effective_chat.id),
    )


def handle_menu_command(update, context):
    update.effective_message.reply_text(
        "🏠 <b>Головне інлайн-меню</b>\nОберіть потрібний розділ:",
        parse_mode="HTML",
        reply_markup=inline_main_keyboard(update.effective_chat.id),
    )


def handle_check_command(update, context):
    args = [a.upper() for a in (context.args or [])]
    if not args:
        update.effective_message.reply_text(
            "🔍 <b>Радар пари</b>\nОберіть валютну пару для миттєвої діагностики:",
            parse_mode="HTML",
            reply_markup=check_pairs_keyboard(),
        )
        return
    pair = normalize_pair_name(args[0])
    if pair not in PAIRS_MAP:
        update.effective_message.reply_text("Невідома пара. Приклад: /check EUR/USD")
        return
    msg = update.effective_message.reply_text(f"🔍 Аналізую <b>{html.escape(pair)}</b>…", parse_mode="HTML")
    def worker():
        txt = _build_pair_radar(pair)
        bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=msg.message_id,
            text=txt,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton(f"📡 Шукати сигнал ({pair})", callback_data=f"scan|pair|{pair}")],
                [InlineKeyboardButton("⬅️ Інша пара", callback_data="check_menu")],
            ]),
        )
    threading.Thread(target=worker, daemon=True).start()


def handle_size_command(update, context):
    if not require_admin(update):
        return
    args = context.args or []
    if len(args) >= 2:
        try:
            dep = max(10.0, float(args[0]))
            pct = max(0.5, min(25.0, float(args[1])))
            database.set_bot_setting("risk_deposit", f"{dep:.2f}")
            database.set_bot_setting("risk_pct", f"{pct:.2f}")
        except ValueError:
            update.effective_message.reply_text("Формат: /size 1000 2 (депозит і ризик %)")
            return
    rt = _get_runtime_settings()
    stake = rt["risk_deposit"] * (rt["risk_pct"] / 100.0)
    update.effective_message.reply_text(
        f"💰 <b>Ризик-калькулятор</b>\nДепозит: <b>${rt['risk_deposit']:.2f}</b> · Ризик: <b>{rt['risk_pct']:.1f}%</b>\n"
        f"Рекомендована фіксована ставка: <b>${stake:.2f}</b>",
        parse_mode="HTML",
        reply_markup=risk_keyboard(rt),
    )


def show_stats(chat_id, period="all"):
    rows = database.get_chat_signals(chat_id, since=period_start(period, get_utc_now()))
    rt = _get_runtime_settings()
    try:
        raw_map = database.get_auto_disabled_map()
        auto_global = {k for (scope, _, k) in raw_map} if isinstance(raw_map, dict) else set()
    except Exception:
        auto_global = set()
    return stats_text(rows, period, auto_disabled_global=auto_global, payout=rt["payout"], wr_threshold=rt["wr_threshold"])


def handle_message(update, context):
    chat_id = update.effective_chat.id
    message = update.effective_message
    text = message.text
    if text == "📡 Сигнал зараз":
        message.reply_text(
            "📡 <b>Ручний аналіз сигналів</b>\nОберіть конкретну пару або запустіть сканування всіх 21 пар:",
            parse_mode="HTML",
            reply_markup=scan_keyboard(),
        )
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
        rows = database.get_chat_signals(chat_id, "active")
        message.reply_text(active_text(rows, get_utc_now()), parse_mode="HTML", reply_markup=active_keyboard())
    elif text == "📜 Історія":
        reply_report(message, history_text(database.get_chat_signals(chat_id, "history")))
    elif text == "📊 Статистика":
        message.reply_text(show_stats(chat_id), parse_mode="HTML", reply_markup=stats_keyboard())
    elif text == "🧪 Бектест":
        message.reply_text(
            "🧪 <b>Інтерактивний бектест</b>\nОберіть конкретну пару або запустіть тест по всіх 21 парах:",
            parse_mode="HTML",
            reply_markup=backtest_keyboard("all"),
        )
    elif text == "🎛 Налаштування":
        if require_admin(update):
            rt = _get_runtime_settings()
            message.reply_text(_settings_summary_text(rt), parse_mode="HTML", reply_markup=settings_menu_keyboard(rt))
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
            time.sleep(0.35)
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


def _run_backtest_async(chat_id, pair, include_all):
    settings = dict(strategies_status)
    rt = _get_runtime_settings()
    def worker():
        try:
            if pair == "ALL":
                report, stats_rows = run_full_backtest_all_with_stats(
                    strategies_status=settings,
                    include_all_strategies=include_all,
                    payout=rt["payout"],
                    wr_threshold=rt["wr_threshold"],
                    expiry_mode=rt["expiry_mode"],
                )
            else:
                report, stats_rows = run_full_backtest_with_stats(
                    pair,
                    strategies_status=settings,
                    include_all_strategies=include_all,
                    payout=rt["payout"],
                    wr_threshold=rt["wr_threshold"],
                    expiry_mode=rt["expiry_mode"],
                )

            if stats_rows:
                database.upsert_backtest_results(stats_rows)
                if settings.get("auto_disable_49", True):
                    events = database.evaluate_and_apply_auto_disable(
                        source="BACKTEST",
                        pair_filter=None if pair == "ALL" else pair,
                        threshold=rt["wr_threshold"],
                    )
                    auto_msg = _sync_auto_disable_events(events, wr_threshold=rt["wr_threshold"])
                    if auto_msg:
                        report += f"\n\n🛡 <b>Авто-вимкнення (&lt;{rt['wr_threshold'] * 100:.0f}% WR) за бектестом:</b>\n" + auto_msg + \
                                  "\n<i>Можна ввімкнути вручну у меню «⚙️ Стратегії».</i>"

            send_chunked_html(chat_id, report)
            if pair == "ALL" and stats_rows:
                csv_bytes = build_backtest_csv(stats_rows, wr_threshold=rt["wr_threshold"])
                bot.send_document(
                    chat_id=chat_id,
                    document=io.BytesIO(csv_bytes),
                    filename="backtest_matrix.csv",
                    caption="📊 Повна матриця бектесту: Пара × Стратегія",
                )
        except Exception:
            logger.exception("Помилка бектесту (%s)", pair)
            bot.send_message(chat_id=chat_id, text="❌ Не вдалося виконати бектест.")
    threading.Thread(target=worker, daemon=True).start()


def handle_callback(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    parts = query.data.split("|")
    action = parts[0]

    if action == "st_stat" and len(parts) == 2 and parts[1] in STRATEGIES_META:
        s_key = parts[1]
        rt = _get_runtime_settings()
        live_rows = database.get_detailed_pair_strategy_stats(strategy_filter=s_key)
        bt_rows = database.get_backtest_results(strategy_filter=s_key)
        alert_txt = strategy_stat_text(s_key, live_rows, bt_rows, payout=rt["payout"])
        query.answer(alert_txt[:200], show_alert=True)
        return

    query.answer()

    if action == "main_menu":
        _safe_edit(
            query,
            "🏠 <b>Головне інлайн-меню</b>\nОберіть потрібний розділ:",
            parse_mode="HTML",
            reply_markup=inline_main_keyboard(chat_id),
        )

    elif action == "stats" and len(parts) == 2 and parts[1] in PERIODS:
        _safe_edit(query, show_stats(chat_id, parts[1]), parse_mode="HTML", reply_markup=stats_keyboard())

    elif action == "stats_matrix":
        rt = _get_runtime_settings()
        live_rows = database.get_detailed_pair_strategy_stats()
        bt_rows = database.get_backtest_results()
        text = stats_matrix_text(live_rows, bt_rows, payout=rt["payout"], wr_threshold=rt["wr_threshold"])
        if len(text) <= 3800:
            _safe_edit(query, text, parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ До статистики", callback_data="stats|all")]]))
        else:
            _safe_edit(query, "📋 Повну матрицю надіслано частинами нижче:", reply_markup=stats_keyboard())
            send_chunked_html(chat_id, text)

    elif action == "active_refresh":
        rows = database.get_chat_signals(chat_id, "active")
        live_map = {}
        for r in (rows if isinstance(rows, (list, tuple)) else []):
            tkr = r.get("ticker")
            if tkr and tkr not in live_map:
                live_map[tkr] = get_live_price(tkr)
        _safe_edit(query, active_text(rows, get_utc_now(), live_prices=live_map), parse_mode="HTML", reply_markup=active_keyboard())

    elif action == "check_menu":
        _safe_edit(
            query,
            "🔍 <b>Радар пари</b>\nОберіть валютну пару для перегляду індикаторів, новин і статистики:",
            parse_mode="HTML",
            reply_markup=check_pairs_keyboard(),
        )

    elif action == "check_go" and len(parts) == 2 and parts[1] in PAIRS_MAP:
        pair = parts[1]
        _safe_edit(query, f"🔍 Завантажую дані та індикатори для <b>{html.escape(pair)}</b>…", parse_mode="HTML")
        msg_id = query.message.message_id
        def worker():
            try:
                txt = _build_pair_radar(pair)
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=txt,
                    parse_mode="HTML",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton(f"📡 Шукати сигнал ({pair})", callback_data=f"scan|pair|{pair}")],
                        [InlineKeyboardButton("⬅️ Інша пара", callback_data="check_menu"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")],
                    ]),
                )
            except Exception:
                logger.exception("Помилка радара %s", pair)
                bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text="❌ Помилка завантаження радара.")
        threading.Thread(target=worker, daemon=True).start()

    elif action == "scan":
        sub = parts[1] if len(parts) > 1 else "menu"
        if sub in ("menu", "pairs_menu"):
            _safe_edit(
                query,
                "📡 <b>Ручний аналіз сигналів</b>\nОберіть пару або скануйте всі 21:",
                parse_mode="HTML",
                reply_markup=scan_keyboard(),
            )
        elif sub == "all" or (sub == "pair" and len(parts) == 3 and parts[2] in PAIRS_MAP):
            target = "ALL" if sub == "all" else parts[2]
            _safe_edit(
                query,
                "📡 Оновлюю дані та перевіряю 21 пару…" if target == "ALL" else f"📡 Аналізую <b>{html.escape(target)}</b>…",
                parse_mode="HTML",
            )
            msg_id = query.message.message_id
            def scan():
                try:
                    if target == "ALL":
                        _warm_scan_caches()
                        count = sum(analyze_single_pair(chat_id, p, t) for p, t in PAIRS_MAP.items())
                        bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=msg_id,
                            text=f"📡 Сканування 21 пари завершено · знайдено сигналів: <b>{count}</b>",
                            parse_mode="HTML",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("🔄 Повторити вибір", callback_data="scan|menu")],
                                [InlineKeyboardButton("🏠 Меню", callback_data="main_menu")],
                            ]),
                        )
                    else:
                        report = analyze_single_pair_report(chat_id, target, PAIRS_MAP[target])
                        bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=msg_id,
                            text=report,
                            parse_mode="HTML",
                            reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton(f"🔄 Ще раз {target}", callback_data=f"scan|pair|{target}")],
                                [InlineKeyboardButton("🔍 Радар пари", callback_data=f"check_go|{target}"), InlineKeyboardButton("⬅️ Інша пара", callback_data="scan|menu")],
                            ]),
                        )
                except Exception:
                    logger.exception("Помилка ручного сканування %s", target)
                    bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text="❌ Помилка під час сканування.")
            threading.Thread(target=scan, daemon=True).start()

    elif action == "bt_menu" and len(parts) == 2 and parts[1] in ("all", "act"):
        _safe_edit(
            query,
            "🧪 <b>Інтерактивний бектест</b>\nОберіть конкретну пару або запустіть тест по всіх 21 парах:",
            parse_mode="HTML",
            reply_markup=backtest_keyboard(parts[1]),
        )

    elif action == "bt_go" and len(parts) == 3 and parts[2] in ("all", "act"):
        pair = parts[1]
        if pair != "ALL" and pair not in PAIRS_MAP:
            return
        include_all = parts[2] == "all"
        mode_lbl = "усі 57 стратегій" if include_all else "лише активні"
        _safe_edit(
            query,
            f"⏳ Запускаю бектест: <b>{html.escape(pair)}</b> ({mode_lbl})… Результат з'явиться нижче.",
            parse_mode="HTML",
        )
        _run_backtest_async(chat_id, pair, include_all)

    elif action == "risk":
        rt = _get_runtime_settings()
        stake = rt["risk_deposit"] * (rt["risk_pct"] / 100.0)
        _safe_edit(
            query,
            f"💰 <b>Ризик-калькулятор</b>\nДепозит: <b>${rt['risk_deposit']:.2f}</b> · Ризик: <b>{rt['risk_pct']:.1f}%</b>\n"
            f"Рекомендована фіксована ставка: <b>${stake:.2f}</b>",
            parse_mode="HTML",
            reply_markup=risk_keyboard(rt),
        )

    elif action in (
        "toggle", "ftog", "cat", "cat_all", "pairs", "pair", "pair_all",
        "pcat", "pcat_all", "pair_toggle",
        "stats_clear_ask", "stats_clear_confirm",
        "autodis_list", "autodis_en", "mute1h",
        "settings", "set_wr", "set_conf", "set_cmod", "set_emod", "set_news", "set_pay", "set_tz", "set_dep", "set_rsk",
    ):
        if not require_admin(update):
            return

        if action == "mute1h" and len(parts) == 2 and parts[1] in STRATEGIES_META:
            s_key = parts[1]
            database.mute_strategy_until(s_key, 3600)
            new_markup = InlineKeyboardMarkup([
                [InlineKeyboardButton(f"🔕 {s_key} на паузі (1г)", callback_data=f"st_stat|{s_key}")],
                [InlineKeyboardButton("🏠 Меню", callback_data="main_menu")],
            ])
            try:
                query.edit_message_reply_markup(reply_markup=new_markup)
            except Exception:
                pass

        elif action == "stats_clear_ask":
            _safe_edit(
                query,
                "⚠️ <b>Очистити всю статистику?</b>\n"
                "Буде очищено історію сигналів, збережену матрицю бектесту та список авто-вимкнень.\n"
                "Ручні налаштування стратегій і параметрів залишаться без змін.",
                parse_mode="HTML",
                reply_markup=clear_stats_confirm_keyboard(),
            )

        elif action == "stats_clear_confirm":
            ok = database.clear_all_stats()
            text = "🧹 <b>Статистику повністю очищено.</b>" if ok else "❌ Не вдалося очистити статистику."
            _safe_edit(query, text, parse_mode="HTML", reply_markup=stats_keyboard())

        elif action == "settings":
            sub = parts[1] if len(parts) > 1 else "main"
            rt = _get_runtime_settings()
            if sub == "main":
                _safe_edit(query, _settings_summary_text(rt), parse_mode="HTML", reply_markup=settings_menu_keyboard(rt))
            elif sub == "thresholds":
                _safe_edit(
                    query,
                    f"🎚 <b>Поріг авто-вимкнення слабких стратегій</b>\nПоточний поріг: <b>{rt['wr_threshold'] * 100:.0f}%</b>\nОберіть поріг WR:",
                    parse_mode="HTML",
                    reply_markup=thresholds_keyboard(rt),
                )
            elif sub == "confluence":
                _safe_edit(
                    query,
                    f"🧩 <b>Налаштування конфлюєнтності</b>\nРежим: <b>{rt['conflict_mode']}</b> · Поріг сили: <b>{rt['confluence_threshold']:.1f}</b>\n"
                    "• <b>CONC</b>: усуває конфлікти та об'єднує в одну картку\n"
                    "• <b>MAJ</b>: блокує меншість/нічию без об'єднання\n"
                    "• <b>IND</b>: кожна стратегія працює незалежно",
                    parse_mode="HTML",
                    reply_markup=confluence_keyboard(rt),
                )
            elif sub == "expiry":
                _safe_edit(
                    query,
                    f"⏱ <b>Режим часу експірації</b>\nПоточний режим: <b>{rt['expiry_mode']}</b>\n"
                    "• <b>DYN (Золота середина)</b>: баланс стратегії, 3×ТФ та волатильності\n"
                    "• <b>FIX</b>: фіксований час кожної стратегії",
                    parse_mode="HTML",
                    reply_markup=expiry_keyboard(rt),
                )
            elif sub == "filters":
                be = 100.0 / (1.0 + rt["payout"])
                _safe_edit(
                    query,
                    f"📰 <b>Фільтри, новини та виплата</b>\n"
                    f"Новинний карантин: <b>±{rt['news_blackout_mins']} хв</b> · Виплата: <b>{int(rt['payout'] * 100)}%</b> (беззбиток {be:.1f}%)",
                    parse_mode="HTML",
                    reply_markup=filters_keyboard(rt),
                )
            elif sub == "tz":
                _safe_edit(
                    query,
                    f"🕒 <b>Часовий пояс карток сигналів</b>\nПоточний: <b>{rt['timezone']}</b>",
                    parse_mode="HTML",
                    reply_markup=tz_keyboard(rt),
                )
            elif sub == "risk":
                stake = rt["risk_deposit"] * (rt["risk_pct"] / 100.0)
                _safe_edit(
                    query,
                    f"💰 <b>Ризик-калькулятор</b>\nДепозит: <b>${rt['risk_deposit']:.2f}</b> · Ризик: <b>{rt['risk_pct']:.1f}%</b>\n"
                    f"Рекомендована ставка: <b>${stake:.2f}</b>",
                    parse_mode="HTML",
                    reply_markup=risk_keyboard(rt),
                )

        elif action == "set_wr" and len(parts) == 2:
            val = max(0.30, min(0.70, float(parts[1]) / 100.0))
            database.set_bot_setting("wr_threshold", f"{val:.2f}")
            rt = _get_runtime_settings()
            _safe_edit(query, f"🎚 Поріг авто-вимкнення оновлено: <b>{val * 100:.0f}%</b>", parse_mode="HTML", reply_markup=thresholds_keyboard(rt))

        elif action == "set_cmod" and len(parts) == 2 and parts[1] in CONFLUENCE_MODES:
            database.set_bot_setting("conflict_mode", parts[1])
            rt = _get_runtime_settings()
            _safe_edit(query, f"🧩 Режим конфлюєнтності: <b>{parts[1]}</b> · Поріг: <b>{rt['confluence_threshold']:.1f}</b>", parse_mode="HTML", reply_markup=confluence_keyboard(rt))

        elif action == "set_conf" and len(parts) == 2:
            val = max(0.5, min(5.0, float(parts[1])))
            database.set_bot_setting("confluence_threshold", f"{val:.1f}")
            rt = _get_runtime_settings()
            _safe_edit(query, f"🧩 Поріг конфлюєнтності оновлено: <b>{val:.1f}</b> (режим {rt['conflict_mode']})", parse_mode="HTML", reply_markup=confluence_keyboard(rt))

        elif action == "set_emod" and len(parts) == 2 and parts[1] in EXPIRY_MODES:
            database.set_bot_setting("expiry_mode", parts[1])
            rt = _get_runtime_settings()
            _safe_edit(query, f"⏱ Режим експірації оновлено: <b>{parts[1]}</b>", parse_mode="HTML", reply_markup=expiry_keyboard(rt))

        elif action == "set_news" and len(parts) == 2:
            mins = max(5, min(60, int(parts[1])))
            database.set_bot_setting("news_blackout_mins", str(mins))
            rt = _get_runtime_settings()
            _safe_edit(query, f"📰 Новинний карантин оновлено: <b>±{mins} хв</b>", parse_mode="HTML", reply_markup=filters_keyboard(rt))

        elif action == "set_pay" and len(parts) == 2:
            pay = max(0.50, min(0.98, float(parts[1]) / 100.0))
            database.set_bot_setting("payout", f"{pay:.2f}")
            rt = _get_runtime_settings()
            _safe_edit(query, f"💹 Виплату оновлено: <b>{int(pay * 100)}%</b> (беззбиток {100.0 / (1.0 + pay):.1f}%)", parse_mode="HTML", reply_markup=filters_keyboard(rt))

        elif action == "set_tz" and len(parts) == 2 and parts[1] in SUPPORTED_TIMEZONES:
            database.set_bot_setting("timezone", parts[1])
            rt = _get_runtime_settings()
            _safe_edit(query, f"🕒 Часовий пояс оновлено: <b>{parts[1]}</b>", parse_mode="HTML", reply_markup=tz_keyboard(rt))

        elif action == "set_dep" and len(parts) == 2:
            dep = max(10.0, float(parts[1]))
            database.set_bot_setting("risk_deposit", f"{dep:.2f}")
            rt = _get_runtime_settings()
            stake = rt["risk_deposit"] * (rt["risk_pct"] / 100.0)
            _safe_edit(query, f"💰 Депозит: <b>${rt['risk_deposit']:.2f}</b> · Ризик: <b>{rt['risk_pct']:.1f}%</b> · Ставка: <b>${stake:.2f}</b>", parse_mode="HTML", reply_markup=risk_keyboard(rt))

        elif action == "set_rsk" and len(parts) == 2:
            rsk = max(0.5, min(25.0, float(parts[1])))
            database.set_bot_setting("risk_pct", f"{rsk:.2f}")
            rt = _get_runtime_settings()
            stake = rt["risk_deposit"] * (rt["risk_pct"] / 100.0)
            _safe_edit(query, f"💰 Депозит: <b>${rt['risk_deposit']:.2f}</b> · Ризик: <b>{rt['risk_pct']:.1f}%</b> · Ставка: <b>${stake:.2f}</b>", parse_mode="HTML", reply_markup=risk_keyboard(rt))

        elif action == "ftog" and len(parts) == 2 and parts[1] in GLOBAL_FILTERS:
            key = parts[1]
            changed = dict(strategies_status)
            changed[key] = not changed.get(key, False)
            save_strategies_config(changed)
            strategies_status.update(changed)
            rt = _get_runtime_settings()
            if key == "auto_disable_49":
                _safe_edit(query, f"🎚 Поріг авто-вимкнення: <b>{rt['wr_threshold'] * 100:.0f}%</b>", parse_mode="HTML", reply_markup=thresholds_keyboard(rt))
            else:
                _safe_edit(query, "📰 Фільтри, новини та виплата оновлені:", parse_mode="HTML", reply_markup=filters_keyboard(rt))

        elif action == "cat" and len(parts) == 2:
            cat = parts[1]
            if cat == "MAIN":
                _safe_edit(query, "⚙️ Категорії стратегій та глобальні фільтри", reply_markup=strategy_keyboard())
            elif cat in STRATEGY_CATEGORIES:
                _safe_edit(query, STRATEGY_CATEGORIES[cat], reply_markup=strategy_keyboard(category=cat))

        elif action == "cat_all" and len(parts) == 3 and parts[1] in STRATEGY_CATEGORIES and parts[2] in ("0", "1"):
            cat, val = parts[1], parts[2] == "1"
            changed = dict(strategies_status)
            for k, m in STRATEGIES_META.items():
                if m["category"] == cat:
                    changed[k] = val
                    if val:
                        database.mark_strategy_manual_enable("GLOBAL", "*", k)
            save_strategies_config(changed)
            strategies_status.update(changed)
            _safe_edit(query, STRATEGY_CATEGORIES[cat], reply_markup=strategy_keyboard(category=cat))

        elif action == "toggle" and len(parts) == 2 and (parts[1] in STRATEGIES_META or parts[1] in GLOBAL_FILTERS):
            key = parts[1]
            changed = dict(strategies_status)
            new_val = not changed.get(key, False)
            changed[key] = new_val
            if new_val and key in STRATEGIES_META:
                database.mark_strategy_manual_enable("GLOBAL", "*", key)
            save_strategies_config(changed)
            strategies_status.update(changed)
            if key in STRATEGIES_META:
                cat = STRATEGIES_META[key]["category"]
                _safe_edit(query, STRATEGY_CATEGORIES[cat], reply_markup=strategy_keyboard(category=cat))
            else:
                _safe_edit(query, "⚙️ Категорії стратегій та глобальні фільтри", reply_markup=strategy_keyboard())

        elif action == "pairs":
            try:
                auto_map = database.get_auto_disabled_map()
                if not isinstance(auto_map, dict):
                    auto_map = {}
            except Exception:
                auto_map = {}
            _safe_edit(query, "🎯 Оберіть пару", reply_markup=pairs_keyboard(auto_map))

        elif action == "pair" and len(parts) == 2 and parts[1] in PAIRS_MAP:
            pair = parts[1]
            _safe_edit(query, f"🎯 {pair} · оберіть категорію", reply_markup=strategy_keyboard(pair=pair))

        elif action == "pair_all" and len(parts) == 3 and parts[1] in PAIRS_MAP and parts[2] in ("0", "1"):
            pair, val = parts[1], parts[2] == "1"
            database.set_pair_bulk_strategy_status(pair, ALL_STRATEGY_KEYS, val)
            if val:
                for k in ALL_STRATEGY_KEYS:
                    database.mark_strategy_manual_enable("PAIR", pair, k)
            _safe_edit(query, f"🎯 {pair} · оберіть категорію", reply_markup=strategy_keyboard(pair=pair))

        elif action == "pcat" and len(parts) == 3 and parts[1] in PAIRS_MAP and parts[2] in STRATEGY_CATEGORIES:
            pair, cat = parts[1], parts[2]
            _safe_edit(
                query,
                f"🎯 {pair} · {STRATEGY_CATEGORIES[cat]}",
                reply_markup=strategy_keyboard(category=cat, pair=pair),
            )

        elif action == "pcat_all" and len(parts) == 4 and parts[1] in PAIRS_MAP and parts[2] in STRATEGY_CATEGORIES and parts[3] in ("0", "1"):
            pair, cat, val = parts[1], parts[2], parts[3] == "1"
            cat_keys = [k for k, m in STRATEGIES_META.items() if m["category"] == cat]
            database.set_pair_bulk_strategy_status(pair, cat_keys, val)
            if val:
                for k in cat_keys:
                    database.mark_strategy_manual_enable("PAIR", pair, k)
            _safe_edit(
                query,
                f"🎯 {pair} · {STRATEGY_CATEGORIES[cat]}",
                reply_markup=strategy_keyboard(category=cat, pair=pair),
            )

        elif action == "pair_toggle" and len(parts) == 3 and parts[1] in PAIRS_MAP and parts[2] in STRATEGIES_META:
            pair, key = parts[1], parts[2]
            new_val = not database.get_pair_strategy_status(pair, key)
            database.set_pair_strategy_status(pair, key, new_val)
            if new_val:
                database.mark_strategy_manual_enable("PAIR", pair, key)
            cat = STRATEGIES_META[key]["category"]
            _safe_edit(
                query,
                f"🎯 {pair} · {STRATEGY_CATEGORIES[cat]}",
                reply_markup=strategy_keyboard(category=cat, pair=pair),
            )

        elif action == "autodis_list":
            rt = _get_runtime_settings()
            items = database.get_auto_disabled_items()
            _safe_edit(
                query,
                auto_disabled_text(items, wr_threshold=rt["wr_threshold"]),
                parse_mode="HTML",
                reply_markup=auto_disabled_keyboard(items),
            )

        elif action == "autodis_en":
            if len(parts) == 3 and parts[1] == "G" and parts[2] in STRATEGIES_META:
                key = parts[2]
                changed = dict(strategies_status)
                changed[key] = True
                save_strategies_config(changed)
                strategies_status.update(changed)
                database.mark_strategy_manual_enable("GLOBAL", "*", key)
            elif len(parts) == 4 and parts[1] == "P" and parts[2] in PAIRS_MAP and parts[3] in STRATEGIES_META:
                pair, key = parts[2], parts[3]
                database.set_pair_strategy_status(pair, key, True)
                database.mark_strategy_manual_enable("PAIR", pair, key)
            rt = _get_runtime_settings()
            items = database.get_auto_disabled_items()
            _safe_edit(
                query,
                auto_disabled_text(items, wr_threshold=rt["wr_threshold"]),
                parse_mode="HTML",
                reply_markup=auto_disabled_keyboard(items),
            )

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
    if not args:
        update.effective_message.reply_text(
            "🧪 <b>Інтерактивний бектест</b>\nОберіть конкретну пару або запустіть тест по всіх 21 парах:",
            parse_mode="HTML",
            reply_markup=backtest_keyboard("all"),
        )
        return
    rev_only = "REV" in args
    args = [a for a in args if a != "REV"]
    pair = args[0] if args else "ALL"
    if pair != "ALL" and pair not in PAIRS_MAP:
        update.effective_message.reply_text("Невідома пара. Приклади: /backtest EUR/USD, /backtest ALL, /backtest EUR/USD REV")
        return
    chat_id = update.effective_chat.id
    if rev_only:
        update.effective_message.reply_text(f"📊 REV 5m ({pair}) · порівнюю сесії OFF / ON…")
        settings = dict(strategies_status)
        def worker():
            try:
                reports = run_backtest_all_detailed(strategies_status=settings) if pair == "ALL" else [run_backtest(pair, strategies_status=settings)]
                for report in reports:
                    send_chunked_html(chat_id, report)
            except Exception:
                logger.exception("Помилка REV бектесту")
                bot.send_message(chat_id=chat_id, text="❌ Не вдалося виконати бектест.")
        threading.Thread(target=worker, daemon=True).start()
    else:
        update.effective_message.reply_text(f"🧪 Запускаю повний бектест по всіх 57 стратегіях для <b>{html.escape(pair)}</b>…", parse_mode="HTML")
        _run_backtest_async(chat_id, pair, include_all=True)


def send_daily_report(now=None):
    now = now or get_utc_now()
    rt = _get_runtime_settings()
    for chat_id in list(AUTO_SCAN_CHATS):
        try:
            rows = database.get_chat_signals(chat_id, since=period_start("today", now), until=utc_datetime(now).replace(tzinfo=None))
            lines = ["🌙 <b>Денний звіт · сьогодні (Київ)</b>", wr_line("Усього", rows, payout=rt["payout"], wr_threshold=rt["wr_threshold"])]
            for preset in PRESETS:
                lines.append(wr_line(preset, [row for row in rows if row.get("strategy") == preset], payout=rt["payout"], wr_threshold=rt["wr_threshold"]))
            active_new = sorted({r.get("strategy") for r in rows if r.get("strategy") and r.get("strategy") not in PRESETS})
            for st in active_new[:12]:
                lines.append(wr_line(st, [row for row in rows if row.get("strategy") == st], payout=rt["payout"], wr_threshold=rt["wr_threshold"]))
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
    updater.dispatcher.add_handler(CommandHandler("menu", handle_menu_command))
    updater.dispatcher.add_handler(CommandHandler("check", handle_check_command))
    updater.dispatcher.add_handler(CommandHandler("size", handle_size_command))
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
