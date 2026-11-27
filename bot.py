# -*- coding: utf-8 -*-
import csv
import html
import io
import json
import logging
import os
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
import db_reset
import lab
import sr_levels
import tick_candles
from ai_advisor import ai_advisor_instance
from backtest import (
    build_backtest_csv,
    prune_strategies_below_threshold,
    run_backtest,
    run_backtest_all_detailed,
    run_full_backtest_all_with_stats,
    run_full_backtest_with_stats,
)
from calendar_filter import get_upcoming_news, is_high_impact_news_time
from config import (
    ALLOW_OPEN_ADMIN,
    AUTO_DISABLE_WR_THRESHOLD,
    BIQUOTE_ENABLED,
    CONFLUENCE_MODE,
    CONFLUENCE_THRESHOLD,
    DEFAULT_TIMEZONE,
    DISABLE_FINNHUB_CANDLES,
    EXPIRY_MODE,
    FINNHUB_API_KEY,
    FINNHUB_TOKEN,
    LAB_DAYS,
    LAB_SOURCE,
    MAX_ACTIVE_PER_CURRENCY,
    NEWS_BLACKOUT_MINS,
    PAIRS_MAP,
    PAYOUT,
    PRUNE_MIN_TRADES,
    PRUNE_WR_THRESHOLD,
    YAHOO_PAIRS_MAP,
    normalize_pair_name,
)
from finnhub_ws import get_live_price, get_live_quote, start_finnhub_ws, stop_finnhub_ws
from market_data import fetch_biquote_ohlc, fetch_biquote_quote
from quality_gate import SegmentFilter, SignalGate
from reversion_engine import ReversionGuard
from strategy_engine import (
    ALL_STRATEGY_KEYS,
    CONFLUENCE_MODES,
    EXPIRY_MODES,
    GLOBAL_FILTERS,
    STRATEGIES_META,
    STRATEGY_CATEGORIES,
    delete_strategies_permanently,
    diagnose_pair_market,
    generate_all_candidates as generate_candidates,
    get_strategy_expiration,
    is_known_strategy,
    load_all_strategy_config,
    load_deleted_strategies,
    resolve_confluence,
    restore_all_deleted_strategies,
)
from telegram_views import (
    PERIODS,
    active_text,
    auto_disabled_text,
    format_wr_tag,
    history_text,
    localized_session_name,
    pair_radar_text,
    settlement_text,
    signal_card,
    stats_matrix_text,
    stats_text,
    strategy_ranks_text,
    strategy_stat_text,
    wr_line,
)
from trading_sessions import SUPPORTED_TIMEZONES, next_daily_report, period_start, session_allowed, utc_datetime

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
ADMIN_IDS = {int(v.strip()) for v in os.environ.get("ADMIN_IDS", "").split(",") if v.strip().lstrip("-").isdigit()}
STRATEGIES_CONFIG_FILE = Path(__file__).with_name("strategies_config.json")
_CFG_LOCK = threading.RLock()
strategies_status = load_all_strategy_config(STRATEGIES_CONFIG_FILE)
AUTO_SCAN_CHATS = set()


def _get_runtime_settings():
    defaults = {
        "wr_threshold": float(AUTO_DISABLE_WR_THRESHOLD),
        "prune_threshold": float(PRUNE_WR_THRESHOLD),
        "prune_min_trades": int(PRUNE_MIN_TRADES),
        "bt_source": str(LAB_SOURCE),
        "bt_days": int(LAB_DAYS),
        "confluence_threshold": float(CONFLUENCE_THRESHOLD),
        "conflict_mode": str(CONFLUENCE_MODE),
        "expiry_mode": str(EXPIRY_MODE),
        "news_blackout_mins": int(NEWS_BLACKOUT_MINS),
        "payout": float(PAYOUT),
        "timezone": str(DEFAULT_TIMEZONE),
    }
    try:
        raw = database.get_all_bot_settings()
        if not isinstance(raw, dict):
            return defaults
        if "wr_threshold" in raw:
            defaults["wr_threshold"] = float(raw["wr_threshold"])
        if "prune_threshold" in raw:
            defaults["prune_threshold"] = float(raw["prune_threshold"])
        if "prune_min_trades" in raw:
            defaults["prune_min_trades"] = int(float(raw["prune_min_trades"]))
        if raw.get("bt_source") in ("yahoo", "dukascopy"):
            defaults["bt_source"] = raw["bt_source"]
        if "bt_days" in raw:
            defaults["bt_days"] = int(float(raw["bt_days"]))
        if "confluence_threshold" in raw:
            defaults["confluence_threshold"] = float(raw["confluence_threshold"])
        if raw.get("conflict_mode") in CONFLUENCE_MODES:
            defaults["conflict_mode"] = raw["conflict_mode"]
        if raw.get("expiry_mode") in EXPIRY_MODES:
            defaults["expiry_mode"] = raw["expiry_mode"]
        if "news_blackout_mins" in raw:
            defaults["news_blackout_mins"] = int(float(raw["news_blackout_mins"]))
        if "payout" in raw:
            defaults["payout"] = float(raw["payout"])
        if raw.get("timezone") in SUPPORTED_TIMEZONES:
            defaults["timezone"] = raw["timezone"]
    except Exception:
        pass
    return defaults


signal_gate = SignalGate(max_ccy_exposure=MAX_ACTIVE_PER_CURRENCY)
reversion_guard = ReversionGuard()
segment_filter = SegmentFilter(database, break_even=lambda: 1.0 / (1.0 + _get_runtime_settings()["payout"]), min_trades=40)
bot = None
_LAB_LOCK = threading.Lock()
_LAB_RUNNING = False

_DATA_LOCK = threading.Lock()
_FINNHUB_CANDLE_DISABLED_UNTIL = 0.0
_YAHOO_CACHE = {}
_MERGED_1M_CACHE = {}
_YF_TTL_BY_INTERVAL = {"1m": 45, "5m": 120, "15m": 600, "60m": 1800, "1h": 1800}


def _tfs_15m():
    return tuple(k for k, m in STRATEGIES_META.items() if m["tf"] == "15m" or m.get("htf") == 15)


def _tfs_60m():
    return tuple(k for k, m in STRATEGIES_META.items() if m.get("htf") == 60)


def require_admin(update):
    uid = getattr(getattr(update, "effective_user", None), "id", None)
    if (ADMIN_IDS and uid in ADMIN_IDS) or (not ADMIN_IDS and ALLOW_OPEN_ADMIN):
        return True
    cq = getattr(update, "callback_query", None)
    if cq is not None and getattr(cq, "data", None):
        try:
            cq.answer("⛔ Ця дія доступна лише адміністратору.", show_alert=True)
        except Exception:
            pass
    elif getattr(update, "effective_message", None):
        update.effective_message.reply_text("⛔ Ця дія доступна лише адміністратору.")
    return False


def save_strategies_config(cfg):
    with _CFG_LOCK:
        tmp = STRATEGIES_CONFIG_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, STRATEGIES_CONFIG_FILE)


def _safe_edit(query, text, parse_mode=None, reply_markup=None):
    try:
        query.edit_message_text(text=text, parse_mode=parse_mode, reply_markup=reply_markup)
    except Exception as e:
        if "not modified" not in str(e).lower():
            raise


def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def parse_dt(value):
    return get_utc_now() if not value else utc_datetime(pd.Timestamp(value).to_pydatetime()).replace(tzinfo=None)


def normalize_yahoo_ticker(ticker, name=""):
    if name and name in YAHOO_PAIRS_MAP:
        return YAHOO_PAIRS_MAP[name]
    clean = ticker.replace("OANDA:", "").replace("IC MARKETS:", "").replace("_", "").replace("/", "")
    return f"{clean}=X" if len(clean) == 6 and not clean.endswith("=X") else ticker


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
    mult = 60 if resolution == "1" else (3600 if resolution == "60" else (86400 if resolution == "D" else 300))
    try:
        resp = requests.get("https://finnhub.io/api/v1/forex/candle", params={"symbol": str(symbol).strip(), "resolution": str(resolution).strip(), "from": end_time - count_candles * mult, "to": end_time, "token": str(token).strip()}, timeout=6)
        if resp.status_code in (401, 403):
            with _DATA_LOCK:
                _FINNHUB_CANDLE_DISABLED_UNTIL = time.time() + 21600
            return pd.DataFrame()
        if resp.status_code == 200:
            data = resp.json()
            if data.get("s") == "ok":
                df = pd.DataFrame({"open": data["o"], "high": data["h"], "low": data["l"], "close": data["c"], "volume": data["v"]}, index=pd.to_datetime(data["t"], unit="s"))
                df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                return df
    except Exception:
        pass
    return pd.DataFrame()


def _clean_yf_df(df_yf):
    if df_yf is None or df_yf.empty:
        return pd.DataFrame()
    df_out = df_yf.copy()
    if isinstance(df_out.columns, pd.MultiIndex):
        df_out.columns = df_out.columns.get_level_values(0)
    df_out = df_out.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    if len([c for c in ("open", "high", "low", "close") if c in df_out.columns]) < 4:
        return pd.DataFrame()
    if "volume" not in df_out.columns:
        df_out["volume"] = 0.0
    df_out = df_out[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
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
        df_yf = yf.Ticker(ticker).history(period=range_period, interval=interval, auto_adjust=True)
        if df_yf is None or df_yf.empty:
            df_yf = yf.download(tickers=ticker, period=range_period, interval=interval, progress=False, auto_adjust=True)
        cleaned = _clean_yf_df(df_yf)
        if not cleaned.empty:
            with _DATA_LOCK:
                _YAHOO_CACHE[cache_key] = (time.time(), cleaned)
            return cleaned.copy()
    except Exception:
        pass
    if BIQUOTE_ENABLED:
        pair_clean = normalize_pair_name(ticker)
        if pair_clean:
            bq = fetch_biquote_ohlc(pair_clean, interval=interval, limit=500)
            if not bq.empty:
                with _DATA_LOCK:
                    _YAHOO_CACHE[cache_key] = (time.time(), bq)
                return bq.copy()
    return pd.DataFrame()


def prefetch_yahoo_batch(interval="1m", range_period="5d"):
    ttl = _YF_TTL_BY_INTERVAL.get(interval, 60)
    now_ts = time.time()
    with _DATA_LOCK:
        needed = [y for y in YAHOO_PAIRS_MAP.values() if not _YAHOO_CACHE.get((y, interval)) or (now_ts - _YAHOO_CACHE[(y, interval)][0]) >= ttl]
    if not needed:
        return
    try:
        raw = yf.download(tickers=" ".join(needed), period=range_period, interval=interval, group_by="ticker", threads=True, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return
        stamp = time.time()
        with _DATA_LOCK:
            if isinstance(raw.columns, pd.MultiIndex):
                top = set(raw.columns.get_level_values(0))
                for y in needed:
                    if y in top:
                        cleaned = _clean_yf_df(raw[y])
                        if not cleaned.empty:
                            _YAHOO_CACHE[(y, interval)] = (stamp, cleaned)
            elif len(needed) == 1:
                cleaned = _clean_yf_df(raw)
                if not cleaned.empty:
                    _YAHOO_CACHE[(needed[0], interval)] = (stamp, cleaned)
    except Exception:
        pass


def resample_candles(df_1m, rule):
    if df_1m is None or df_1m.empty:
        return pd.DataFrame()
    return df_1m.resample(rule, label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna(subset=["open", "high", "low", "close"])


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
    if BIQUOTE_ENABLED and abs((now - expires_at).total_seconds()) <= 45:
        bq_p = fetch_biquote_quote(name or ticker, allow_stale=False)
        if bq_p:
            return float(bq_p)
    return None


def schedule_signal_timer(sig_id, timestamp_val, expiration_mins):
    try:
        delay = max(1, (parse_dt(timestamp_val) + timedelta(minutes=expiration_mins) - get_utc_now()).total_seconds())
        t = threading.Timer(delay, process_signal_expiration, args=[sig_id])
        t.daemon = True
        t.start()
    except Exception:
        pass


def _recover_pending_timers():
    try:
        for row in database.get_pending_signals():
            exp_dt = parse_dt(row["expires_at"]) if row.get("expires_at") else parse_dt(row["timestamp_str"]) + timedelta(minutes=row["expiration_mins"])
            t = threading.Timer(max(1, (exp_dt - get_utc_now()).total_seconds()), process_signal_expiration, args=[row["id"]])
            t.daemon = True
            t.start()
    except Exception:
        pass


def start_background_checker():
    def loop():
        while True:
            try:
                for row in database.get_pending_signals():
                    expiry = parse_dt(row["expires_at"]) if row.get("expires_at") else parse_dt(row["timestamp_str"]) + timedelta(minutes=row["expiration_mins"])
                    if get_utc_now() >= expiry:
                        process_signal_expiration(row["id"])
            except Exception:
                pass
            time.sleep(20)
    threading.Thread(target=loop, daemon=True).start()


def _warm_scan_caches():
    prefetch_yahoo_batch("1m", "5d")
    if any(strategies_status.get(k, False) for k in _tfs_15m()):
        prefetch_yahoo_batch("15m", "30d")
    if any(strategies_status.get(k, False) for k in _tfs_60m()):
        prefetch_yahoo_batch("60m", "60d")


def start_auto_scanner_loop():
    def loop():
        while True:
            try:
                if AUTO_SCAN_CHATS:
                    _warm_scan_caches()
                    for chat_id in list(AUTO_SCAN_CHATS):
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            for f in [executor.submit(analyze_single_pair, chat_id, p, t, auto_mode=True) for p, t in PAIRS_MAP.items()]:
                                try:
                                    f.result(timeout=30)
                                except Exception:
                                    pass
            except Exception:
                pass
            time.sleep(max(1.0, 62.0 - (time.time() % 60)))
    threading.Thread(target=loop, daemon=True).start()


def _sync_auto_disable_events(events, notify_chat_id=None, wr_threshold=None):
    if not isinstance(events, dict):
        return ""
    th = float(wr_threshold) if isinstance(wr_threshold, (int, float)) else _get_runtime_settings()["wr_threshold"]
    lines = []
    disabled_global = events.get("disabled_global") or []
    if disabled_global:
        with _CFG_LOCK:
            changed = dict(strategies_status)
            for ev in disabled_global:
                s_key = ev["strategy"]
                if s_key in changed:
                    changed[s_key] = False
                lines.append(f"⛔ Глобально вимкнено <code>{html.escape(s_key)}</code>: {ev['wr'] * 100:.1f}% ({ev['wins']}/{ev['total']}) &lt; {th * 100:.0f}%")
            save_strategies_config(changed)
            strategies_status.update(changed)
    for ev in (events.get("disabled_pair") or []):
        lines.append(f"⛔ По парі <b>{html.escape(ev['pair'])}</b> вимкнено <code>{html.escape(ev['strategy'])}</code>: {ev['wr'] * 100:.1f}% ({ev['wins']}/{ev['total']})")
    summary = "\n".join(lines)
    if summary and notify_chat_id and bot is not None:
        try:
            bot.send_message(chat_id=notify_chat_id, text=f"🛡 <b>Авто-контроль (&lt;{th * 100:.0f}% WR):</b>\n{summary}", parse_mode="HTML")
        except Exception:
            pass
    return summary


def _apply_live_auto_disable(sig):
    try:
        if not isinstance(sig, dict) or not sig.get("strategy") or not strategies_status.get("auto_disable_49", True):
            return
        rt = _get_runtime_settings()
        events = database.evaluate_and_apply_auto_disable(source="LIVE", pair_filter=sig.get("ticker"), strategy_filter=sig.get("strategy"), threshold=rt["wr_threshold"])
        _sync_auto_disable_events(events, notify_chat_id=sig.get("chat_id"), wr_threshold=rt["wr_threshold"])
    except Exception:
        pass


def _run_lab_in_thread(chat_id: int):
    global _LAB_RUNNING
    with _LAB_LOCK:
        if _LAB_RUNNING:
            bot.send_message(chat_id=chat_id, text="⏳ Лабораторія вже виконується у фоні.")
            return
        _LAB_RUNNING = True

    def worker():
        global _LAB_RUNNING
        try:
            rt = _get_runtime_settings()
            src, days, payout = rt["bt_source"], rt["bt_days"], rt["payout"]
            prune_th, min_tr = rt["prune_threshold"], rt["prune_min_trades"]
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_data")
            lab.download_and_merge(lab.PAIRS, data_dir, source=src, days=days)
            data = {p: d for p in lab.PAIRS if (d := lab.load_pair(p, data_dir)) is not None}
            if not data:
                bot.send_message(chat_id=chat_id, text="❌ Не вдалося завантажити дані для Лабораторії.")
                return
            merged, pair_stats = lab.collect(data)
            if pair_stats:
                database.upsert_backtest_results(pair_stats)
            rep, split, breakeven = lab.analyze(merged, payout)
            first, last = min(d.index[0] for d in data.values()), max(d.index[-1] for d in data.values())
            info = f"{len(data)} пар, {first:%Y-%m-%d} → {last:%Y-%m-%d} ({src})"
            summary, _ = lab.build_summary(rep, split, breakeven, payout, info)
            pruned = lab.prune_config_from_report(rep, str(STRATEGIES_CONFIG_FILE), threshold=prune_th, min_trades=min_tr)
            with _CFG_LOCK:
                strategies_status.update(load_all_strategy_config(STRATEGIES_CONFIG_FILE))
            rep_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_report.csv")
            rep.sort_values("d_z", ascending=False).to_csv(rep_path, index=False)

            msg = summary + f"\n\n🧹 <b>Авто-відсів (&lt;{prune_th * 100:.0f}%, ≥{min_tr} угод):</b> вимкнено {len(pruned)} стратегій."
            send_chunked_html(chat_id, msg)
            bot.send_message(
                chat_id=chat_id,
                text="🗑 Можете одразу видалити всі відсіяні стратегії з бота:",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton(f"🗑 Видалити відсіяні (<{prune_th * 100:.0f}%)", callback_data="del_pruned_ask")],
                    [InlineKeyboardButton("⚙️ До стратегій (% WR)", callback_data="cat|MAIN")],
                ]),
            )
            if os.path.exists(rep_path):
                with open(rep_path, "rb") as f:
                    bot.send_document(chat_id=chat_id, document=f, filename="lab_report.csv")
        except Exception as e:
            logger.exception("Помилка Lab")
            bot.send_message(chat_id=chat_id, text=f"❌ Помилка Лабораторії: {html.escape(str(e)[:120])}")
        finally:
            with _LAB_LOCK:
                _LAB_RUNNING = False
    threading.Thread(target=worker, daemon=True).start()


def handle_lab_run_command(update, context):
    if require_admin(update):
        update.message.reply_text("🔬 Запускаю повну Лабораторію по всіх активних стратегіях…")
        _run_lab_in_thread(update.effective_chat.id)


def handle_lab_command(update, context):
    if not require_admin(update):
        return
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_summary.txt")
    if not os.path.exists(path):
        update.message.reply_text("📭 Звіту ще немає. Запустіть через меню «🧪 Бектест» → «🔬 Запустити Lab».")
        return
    with open(path, encoding="utf-8") as f:
        send_chunked_html(update.effective_chat.id, f"<pre>{html.escape(f.read())}</pre>")


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
    if include_all_tfs or any(strategies_status.get(k, False) for k in _tfs_15m()):
        df_15m = fetch_yahoo_fallback(yahoo, interval="15m", range_period="30d")
        if df_15m.empty and df_1m is not None and not df_1m.empty:
            df_15m = resample_candles(df_1m, "15min")
        frames["15m"] = df_15m
    if include_all_tfs or any(strategies_status.get(k, False) for k in _tfs_60m()):
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
        [InlineKeyboardButton("🏠 Меню", callback_data="main_menu")],
    ])


def notify_settlement(sig, result, exit_price, pips):
    report = settlement_text(result, exit_price, pips, pair=sig.get("ticker"), entry_price=sig.get("entry_price"), signal_type=sig.get("signal_type"))
    markup = InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Розбір закритої угоди", callback_data=f"review|{sig['id']}")]]) if (ai_advisor_instance.api_key and result in ("WIN", "LOSS", "NEUTRAL")) else None
    try:
        bot.edit_message_text(chat_id=sig["chat_id"], message_id=sig["message_id"], text=(sig.get("message_text") or "") + "\n\n" + report, parse_mode="HTML", reply_markup=markup)
    except Exception:
        pass
    try:
        bot.send_message(chat_id=sig["chat_id"], text=f"#{sig['id']} " + report, parse_mode="HTML", reply_to_message_id=sig["message_id"], allow_sending_without_reply=True)
    except Exception:
        pass


def process_signal_expiration(sig_id):
    claimed = False
    try:
        sig = database.get_signal_by_id(sig_id)
        if not sig or sig.get("status") != "PENDING":
            return
        expiry = parse_dt(sig["expires_at"]) if sig.get("expires_at") else parse_dt(sig["timestamp_str"]) + timedelta(minutes=sig["expiration_mins"])
        late = (get_utc_now() - expiry).total_seconds()
        if late < 0:
            t = threading.Timer(-late, process_signal_expiration, args=[sig_id])
            t.daemon = True
            t.start()
            return
        if not database.claim_signal(sig_id):
            return
        claimed = True
        pair = next((name for name, ticker in PAIRS_MAP.items() if ticker == sig["ticker"]), "")
        exit_price = get_exit_price(sig["ticker"], expiry, pair)
        if not exit_price:
            if late <= 180:
                database.release_signal(sig_id)
                t = threading.Timer(30, process_signal_expiration, args=[sig_id])
                t.daemon = True
                t.start()
                return
            database.update_signal_result(sig_id, "NEUTRAL", None, 0.0, status="VOID")
            notify_settlement(sig, "VOID", None, 0.0)
            return
        entry = float(sig["entry_price"])
        move = exit_price - entry if sig["signal_type"] == "CALL" else entry - exit_price
        result = "NEUTRAL" if exit_price == entry else ("WIN" if move > 0 else "LOSS")
        pips = round(move * (1000.0 if "JPY" in normalize_pair_name(sig["ticker"]) else 100000.0), 1)
        database.update_signal_result(sig_id, result, exit_price, pips)
        notify_settlement(sig, result, exit_price, pips)
        if result in ("WIN", "LOSS"):
            _apply_live_auto_disable(sig)
    except Exception:
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

    tf_label, tf_minutes = cand.get("tf") or "5m", int(cand.get("tf_minutes") or 5)
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
        live = get_live_price(ticker_finnhub) or (fetch_biquote_quote(pair_name, allow_stale=False) if BIQUOTE_ENABLED else None)
        if live:
            adverse = cand["current_price"] - live if cand["signal"] == "CALL" else live - cand["current_price"]
            if adverse > 0.5 * cand["atr"]:
                return False, "Ціна пішла проти сигналу > 0.5 ATR"
        now = get_utc_now()
        max_age = 45 if tf_minutes <= 1 else (90 if tf_minutes <= 3 else 180)
        if not 0 <= (now - cand["bar_end"]).total_seconds() <= max_age:
            return False, "Сигнал застарів"

        rt = _get_runtime_settings()
        entry = float(live or cand["current_price"])
        expiration = get_strategy_expiration(cand)
        text = signal_card(pair_name, cand, entry, now, tz_name=rt["timezone"])
        message = bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML", reply_markup=signal_action_keyboard(cand["strategy"]))
        sig_id = database.save_signal(
            chat_id=chat_id, message_id=message.message_id, ticker=ticker_finnhub, signal_type=cand["signal"],
            entry_price=entry, primary_tf=tf_label, score=cand.get("confirmations", 0), expiration_mins=expiration,
            message_text=text, timestamp_str=now.strftime("%Y-%m-%d %H:%M:%S"), expires_at=now + timedelta(minutes=expiration),
            rsi=cand.get("rsi"), adx=cand.get("adx"), volatility_ratio=cand.get("vol_ratio"), divergence=cand.get("regime"),
            hour=now.hour, strategy=cand["strategy"],
        )
        if not sig_id:
            bot.send_message(chat_id=chat_id, text="⚠️ Сигнал не збережено в БД.")
            return False, "Помилка збереження сигналу"
        signal_gate.register(cand, expiration)
        reversion_guard.register(pair_name, cand["strategy"], expiration)
        schedule_signal_timer(sig_id, now, expiration)
        return True, "OK"


def _filter_unmuted(candidates):
    out = []
    for c in candidates:
        try:
            m = database.is_strategy_muted(c["strategy"])
            if isinstance(m, tuple) and m[0]:
                continue
        except Exception:
            pass
        out.append(c)
    return out


def analyze_single_pair(chat_id, pair_name, ticker_finnhub, auto_mode=False, status_msg_id=None):
    try:
        rt = _get_runtime_settings()
        if strategies_status.get("news_filter", True) and is_high_impact_news_time(pair_name, blackout_mins=rt["news_blackout_mins"])[0]:
            return False
        yahoo = normalize_yahoo_ticker(ticker_finnhub, pair_name)
        df_5m = fetch_5m(ticker_finnhub, yahoo)
        frames = build_mtf_frames(ticker_finnhub, yahoo, df_5m)
        raw_cands = generate_candidates(pair_name, frames, get_utc_now(), dict(strategies_status), pair_enabled=database.get_pair_strategy_status)
        eligible = _filter_unmuted(raw_cands)
        cands = resolve_confluence(eligible, conflict_mode=rt["conflict_mode"], threshold=rt["confluence_threshold"], expiry_mode=rt["expiry_mode"])[0] if (rt["conflict_mode"] != "IND" or rt["expiry_mode"] != "FIX") else eligible
        for cand in cands:
            if process_candidate(chat_id, pair_name, ticker_finnhub, cand, frames=frames, auto_mode=auto_mode)[0]:
                return True
        return False
    except Exception:
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
        raw_cands = generate_candidates(pair_name, frames, now, dict(strategies_status), pair_enabled=database.get_pair_strategy_status)
        eligible = _filter_unmuted(raw_cands)
        cands, conf_reason = resolve_confluence(eligible, conflict_mode=rt["conflict_mode"], threshold=rt["confluence_threshold"], expiry_mode=rt["expiry_mode"])
        skipped = []
        for cand in cands:
            ok, reason = process_candidate(chat_id, pair_name, ticker_finnhub, cand, frames=frames, auto_mode=False)
            if ok:
                return f"✅ <b>{html.escape(pair_name)}</b>: сигнал сформовано!\nНапрямок: {cand['signal']} · Експірація: <b>{get_strategy_expiration(cand)} ХВ</b>"
            skipped.append(f"{cand['strategy']}: {reason}")
        diag = diagnose_pair_market(pair_name, frames, now, dict(strategies_status), pair_enabled=database.get_pair_strategy_status, mute_check=database.is_strategy_muted)
        chop_s = f"{diag['chop']:.1f}" if diag["chop"] is not None else "—"
        adx_s = f"{diag['adx']:.1f}" if diag["adx"] is not None else "—"
        rsi_s = f"{diag['rsi14']:.1f}" if diag["rsi14"] is not None else "—"
        msg = (
            f"📡 <b>{html.escape(pair_name)}</b>: зараз сигналів немає\n"
            f"📊 5m режим: CHOP <b>{chop_s}</b> · ADX <b>{adx_s}</b> · RSI14 <b>{rsi_s}</b> · Сесія: {localized_session_name(now)}\n"
            f"⚙️ Активних стратегій: <b>{diag['checked']}</b> (глоб.вимк: {diag['off_global']}, пара: {diag['off_pair']}, пауза: {diag.get('muted', 0)}, сесія: {diag['off_session']})"
        )
        if not raw_cands:
            msg += "\n📉 Причина: індикатори ще не сформували вхідний паттерн на закритій свічці."
        elif not cands:
            msg += f"\n⚖️ Причина: {html.escape(conf_reason)}"
        elif skipped:
            msg += "\n⏭ Відфільтровано: " + html.escape("; ".join(skipped[:3]))
        return msg
    except Exception:
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
        [InlineKeyboardButton("🔍 Радар пари (S/R)", callback_data="check_menu"), InlineKeyboardButton("🧪 Бектест і Lab", callback_data="bt_menu|all")],
        [InlineKeyboardButton("📊 Статистика", callback_data="stats|all"), InlineKeyboardButton("⏳ Активні угоди", callback_data="active_refresh")],
        [InlineKeyboardButton("🎛 Усі налаштування", callback_data="settings|main"), InlineKeyboardButton(f"⚙️ Стратегії ({len(STRATEGIES_META)})", callback_data="cat|MAIN")],
        [InlineKeyboardButton(auto_lbl, callback_data="autoscan_tog"), InlineKeyboardButton("🗑 Скинути БД (GitHub)", callback_data="db_reset_ask")],
    ])


def stats_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(title, callback_data="stats|" + period) for period, title in PERIODS.items()],
        [InlineKeyboardButton("📋 Матриця Пара×Страт", callback_data="stats_matrix"), InlineKeyboardButton("📊 Рейтинг (% WR)", callback_data="ranks|0")],
        [InlineKeyboardButton("🧹 Очистити статистику", callback_data="stats_clear_ask"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")],
    ])


def clear_stats_confirm_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("✅ Так, очистити все", callback_data="stats_clear_confirm"), InlineKeyboardButton("❌ Скасувати", callback_data="stats|all")]])


def db_reset_confirm_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔥 ТАК, ОБНУЛИТИ БД + КОНФІГ", callback_data="db_reset_confirm"), InlineKeyboardButton("❌ Скасувати", callback_data="settings|main")]])


def active_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Оновити ціни (Live)", callback_data="active_refresh")], [InlineKeyboardButton("🏠 Головне меню", callback_data="main_menu")]])


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
    buttons.append([InlineKeyboardButton("🔍 Радар пари", callback_data="check_menu"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


def check_pairs_keyboard():
    buttons, row = [], []
    for pair in PAIRS_MAP:
        row.append(InlineKeyboardButton(pair, callback_data=f"check_go|{pair}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton("🏠 Головне меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


def backtest_keyboard(mode="all", rt=None):
    rt = rt or _get_runtime_settings()
    next_mode = "act" if mode == "all" else "all"
    cnt = len(STRATEGIES_META)
    mode_txt = f"📊 Режим: УСІ {cnt} стратегій (змінити)" if mode == "all" else "📊 Режим: Лише активні (змінити)"
    src_lbl = "Dukascopy" if rt["bt_source"] == "dukascopy" else "Yahoo"
    buttons = [
        [InlineKeyboardButton(mode_txt, callback_data="bt_menu|" + next_mode)],
        [InlineKeyboardButton(f"🔬 Запустити Lab ({cnt} · {src_lbl} {rt['bt_days']}д)", callback_data="lab_run_tg")],
        [InlineKeyboardButton("🧹 Відсіяти слабкі зараз", callback_data="prune_now"), InlineKeyboardButton("🗑 Видалити відсіяні", callback_data="del_menu|0")],
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
    buttons.append([InlineKeyboardButton("🎛 Налаштування Lab", callback_data="settings|lab"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


def settings_menu_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    wr_pct, pr_pct, pay_pct = int(round(rt["wr_threshold"] * 100)), int(round(rt["prune_threshold"] * 100)), int(round(rt["payout"] * 100))
    tz_lbl = SUPPORTED_TIMEZONES.get(rt["timezone"], ("Київ",))[0]
    src_lbl = "Dukascopy" if rt["bt_source"] == "dukascopy" else "Yahoo"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"🎚 Авто-вимк: {wr_pct}% · Відсів: {pr_pct}%", callback_data="settings|thresholds")],
        [InlineKeyboardButton(f"🔬 Lab: {src_lbl} {rt['bt_days']}д (мін.{rt['prune_min_trades']} угод)", callback_data="settings|lab")],
        [InlineKeyboardButton(f"🧩 Конфлюентність: {rt['conflict_mode']} · ≥ {rt['confluence_threshold']:.1f}", callback_data="settings|confluence")],
        [InlineKeyboardButton(f"⏱ Експірація: {rt['expiry_mode']} (Золота сер. / Фікс)", callback_data="settings|expiry")],
        [InlineKeyboardButton(f"📰 Фільтри / Новини (±{rt['news_blackout_mins']}хв) · {pay_pct}%", callback_data="settings|filters")],
        [InlineKeyboardButton(f"🕒 Пояс: {tz_lbl}", callback_data="settings|tz"), InlineKeyboardButton("🗑 Видалити слабкі", callback_data="del_menu|0")],
        [InlineKeyboardButton(f"⚙️ Групи стратегій ({len(STRATEGIES_META)})", callback_data="cat|MAIN"), InlineKeyboardButton("🎯 По парах", callback_data="pairs")],
        [InlineKeyboardButton("🗑 Повне скидання БД (GitHub)", callback_data="db_reset_ask")],
        [InlineKeyboardButton("🏠 Головне меню", callback_data="main_menu")],
    ])


def lab_settings_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    src, days, mt = rt["bt_source"], rt["bt_days"], rt["prune_min_trades"]
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(("• Dukascopy •" if src == "dukascopy" else "Dukascopy"), callback_data="set_bt_src|dukascopy"),
            InlineKeyboardButton(("• Yahoo •" if src == "yahoo" else "Yahoo"), callback_data="set_bt_src|yahoo"),
        ],
        [
            InlineKeyboardButton(("• 30д •" if days == 30 else "30д"), callback_data="set_bt_days|30"),
            InlineKeyboardButton(("• 90д •" if days == 90 else "90д"), callback_data="set_bt_days|90"),
            InlineKeyboardButton(("• 180д •" if days == 180 else "180д"), callback_data="set_bt_days|180"),
        ],
        [
            InlineKeyboardButton(("• ≥10 угод •" if mt == 10 else "≥10"), callback_data="set_mintr|10"),
            InlineKeyboardButton(("• ≥15 •" if mt == 15 else "≥15"), callback_data="set_mintr|15"),
            InlineKeyboardButton(("• ≥20 •" if mt == 20 else "≥20"), callback_data="set_mintr|20"),
            InlineKeyboardButton(("• ≥30 •" if mt == 30 else "≥30"), callback_data="set_mintr|30"),
        ],
        [InlineKeyboardButton("🔬 Запустити Lab зараз", callback_data="lab_run_tg"), InlineKeyboardButton("🧹 Відсіяти зараз", callback_data="prune_now")],
        [InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")],
    ])


def thresholds_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    cur, pcur = int(round(rt["wr_threshold"] * 100)), int(round(rt["prune_threshold"] * 100))
    auto_on = "✅ Авто-вимкнення: ON" if strategies_status.get("auto_disable_49", True) else "❌ Авто-вимкнення: OFF"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(auto_on, callback_data="ftog|auto_disable_49")],
        [InlineKeyboardButton((f"• Авто {v}% •" if cur == v else f"Авто {v}%"), callback_data=f"set_wr|{v}") for v in (45, 49, 52)],
        [InlineKeyboardButton((f"• Авто {v}% •" if cur == v else f"Авто {v}%"), callback_data=f"set_wr|{v}") for v in (54, 55)],
        [InlineKeyboardButton((f"• Відсів {v}% •" if pcur == v else f"Відсів {v}%"), callback_data=f"set_prune|{v}") for v in (40, 45, 49, 52)],
        [InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")],
    ])


def confluence_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    c_mod, c_th = rt["conflict_mode"], rt["confluence_threshold"]
    buttons = [[InlineKeyboardButton(("✅ " if c_mod == k else "") + title, callback_data=f"set_cmod|{k}")] for k, title in CONFLUENCE_MODES.items()]
    buttons.append([InlineKeyboardButton((f"• {v} •" if abs(c_th - v) < 0.05 else str(v)), callback_data=f"set_conf|{v}") for v in (1.0, 1.5, 2.0)])
    buttons.append([InlineKeyboardButton((f"• {v} •" if abs(c_th - v) < 0.05 else str(v)), callback_data=f"set_conf|{v}") for v in (2.5, 3.0)])
    buttons.append([InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")])
    return InlineKeyboardMarkup(buttons)


def expiry_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    buttons = [[InlineKeyboardButton(("✅ " if rt["expiry_mode"] == k else "") + title, callback_data=f"set_emod|{k}")] for k, title in EXPIRY_MODES.items()]
    buttons.append([InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")])
    return InlineKeyboardMarkup(buttons)


def filters_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    nm, pay = rt["news_blackout_mins"], int(round(rt["payout"] * 100))
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(("✅ " if strategies_status.get("news_filter", True) else "❌ ") + "Новини", callback_data="ftog|news_filter"),
            InlineKeyboardButton(("✅ " if strategies_status.get("session_filter", True) else "❌ ") + "Сесії", callback_data="ftog|session_filter"),
            InlineKeyboardButton(("✅ " if strategies_status.get("regime_filter", True) else "❌ ") + "Режим", callback_data="ftog|regime_filter"),
        ],
        [InlineKeyboardButton((f"• ±{v}хв •" if nm == v else f"±{v}хв"), callback_data=f"set_news|{v}") for v in (5, 15, 30)],
        [InlineKeyboardButton((f"• {v}% •" if pay == v else f"{v}%"), callback_data=f"set_pay|{v}") for v in (80, 85, 88, 90)],
        [InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")],
    ])


def tz_keyboard(rt=None):
    rt = rt or _get_runtime_settings()
    buttons = [[InlineKeyboardButton(("✅ " if rt["timezone"] == k else "") + f"{label} ({k})", callback_data=f"set_tz|{k}")] for k, (label, _) in SUPPORTED_TIMEZONES.items()]
    buttons.append([InlineKeyboardButton("⬅️ Назад до налаштувань", callback_data="settings|main")])
    return InlineKeyboardMarkup(buttons)


def pairs_keyboard(auto_map=None):
    auto_map = auto_map or {}
    rows, row = [], []
    for pair in PAIRS_MAP:
        has_auto = any(st == "PAIR" and p == pair for (st, p, _) in auto_map)
        row.append(InlineKeyboardButton(("⚠️ " + pair) if has_auto else pair, callback_data="pair|" + pair))
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="cat|MAIN")])
    return InlineKeyboardMarkup(rows)


def auto_disabled_keyboard(items, page: int = 0, per_page: int = 12):
    all_items = list(items) if isinstance(items, (list, tuple)) else []
    total_pages = max(1, (len(all_items) + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))
    buttons = []
    for it in all_items[page * per_page : (page + 1) * per_page]:
        key, wr = it["strategy_key"], float(it.get("win_rate") or 0.0) * 100.0
        if it["scope_type"] == "GLOBAL":
            buttons.append([InlineKeyboardButton(f"🔓 Глоб: {key} ({wr:.0f}%)", callback_data=f"autodis_en|G|{key}")])
        else:
            p = it["pair_name"]
            buttons.append([InlineKeyboardButton(f"🔓 {p}: {key} ({wr:.0f}%)", callback_data=f"autodis_en|P|{p}|{key}")])
    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("⬅️ Попер.", callback_data=f"autodis_list|{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("Наст. ➡️", callback_data=f"autodis_list|{page + 1}"))
        buttons.append(nav)
    buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="cat|MAIN")])
    return InlineKeyboardMarkup(buttons)


def _get_prunable_and_disabled(wr_summary: dict, prune_th: float, min_tr: int) -> list[tuple[str, str]]:
    out = []
    for sk, meta in STRATEGIES_META.items():
        d = wr_summary.get(sk) or {}
        tot, wr = int(d.get("total") or 0), d.get("wr")
        is_weak = (wr is not None and tot >= min_tr and (wr / 100.0) < prune_th)
        if is_weak or not strategies_status.get(sk, True):
            tag = f"{wr:.0f}%·{tot}" if wr is not None else "вимк"
            out.append((sk, f"🗑 {meta['title']} [{tag}]"))
    return out


def delete_strategies_keyboard(wr_summary: dict | None = None, page: int = 0, per_page: int = 10, rt: dict | None = None):
    rt = rt or _get_runtime_settings()
    if wr_summary is None:
        wr_summary = database.get_strategy_winrate_summary()
    pr_th, min_tr = rt["prune_threshold"], rt["prune_min_trades"]
    candidates = _get_prunable_and_disabled(wr_summary, pr_th, min_tr)
    deleted_cnt = len(load_deleted_strategies())

    total_pages = max(1, (len(candidates) + per_page - 1) // per_page)
    page = max(0, min(page, total_pages - 1))
    buttons = [[InlineKeyboardButton(f"🔥 Видалити ВСІ відсіяні (<{pr_th * 100:.0f}%)", callback_data="del_pruned_ask")]]
    for sk, lbl in candidates[page * per_page : (page + 1) * per_page]:
        buttons.append([InlineKeyboardButton(lbl, callback_data=f"del_one|{sk}")])
    if total_pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("⬅️", callback_data=f"del_menu|{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{total_pages}", callback_data="noop"))
        if page < total_pages - 1:
            nav.append(InlineKeyboardButton("➡️", callback_data=f"del_menu|{page + 1}"))
        buttons.append(nav)
    if deleted_cnt:
        buttons.append([InlineKeyboardButton(f"♻️ Відновити видалені ({deleted_cnt})", callback_data="del_restore_ask")])
    buttons.append([InlineKeyboardButton("⬅️ До стратегій", callback_data="cat|MAIN"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


def strategy_keyboard(category=None, pair=None, auto_map=None, page: int = 0, per_page: int = 10, wr_summary: dict | None = None):
    if auto_map is None:
        try:
            raw_map = database.get_auto_disabled_map()
            auto_map = raw_map if isinstance(raw_map, dict) else {}
        except Exception:
            auto_map = {}
    if wr_summary is None:
        try:
            wr_summary = database.get_strategy_winrate_summary(pair_filter=pair)
        except Exception:
            wr_summary = {}

    if pair and category and category in STRATEGY_CATEGORIES:
        cat_items = [(k, m) for k, m in STRATEGIES_META.items() if m["category"] == category]
        total_pages = max(1, (len(cat_items) + per_page - 1) // per_page)
        page = max(0, min(page, total_pages - 1))
        buttons = []
        for key, meta in cat_items[page * per_page : (page + 1) * per_page]:
            enabled = database.get_pair_strategy_status(pair, key)
            auto_rec = auto_map.get(("PAIR", pair, key))
            mark = "⛔ " if (auto_rec and not enabled) else ("✅ " if enabled else "❌ ")
            wr_tag = format_wr_tag(wr_summary.get(key))
            buttons.append([InlineKeyboardButton(f"{mark}{meta['title']} {wr_tag}", callback_data=f"pair_toggle|{pair}|{key}|{page}")])
        if total_pages > 1:
            nav = []
            if page > 0:
                nav.append(InlineKeyboardButton("⬅️", callback_data=f"pcat|{pair}|{category}|{page - 1}"))
            nav.append(InlineKeyboardButton(f"Стор. {page + 1}/{total_pages}", callback_data="noop"))
            if page < total_pages - 1:
                nav.append(InlineKeyboardButton("➡️", callback_data=f"pcat|{pair}|{category}|{page + 1}"))
            buttons.append(nav)
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
            if not cat_keys:
                continue
            enabled_cnt = sum(1 for k in cat_keys if database.get_pair_strategy_status(pair, k))
            has_auto = any(("PAIR", pair, k) in auto_map for k in cat_keys)
            wrs = [wr_summary[k]["wr"] for k in cat_keys if k in wr_summary and wr_summary[k].get("wr") is not None]
            avg_s = f"|{sum(wrs) / len(wrs):.0f}%" if wrs else ""
            warn = " ⛔" if has_auto else ""
            buttons.append([InlineKeyboardButton(f"{cat_title} [{enabled_cnt}/{len(cat_keys)}{avg_s}]{warn}", callback_data=f"pcat|{pair}|{cat_key}|0")])
        buttons.append([
            InlineKeyboardButton("✅ Увімкнути всі для пари", callback_data=f"pair_all|{pair}|1"),
            InlineKeyboardButton("❌ Вимкнути всі", callback_data=f"pair_all|{pair}|0"),
        ])
        buttons.append([InlineKeyboardButton("⬅️ Назад до пар", callback_data="pairs")])
        return InlineKeyboardMarkup(buttons)

    if category and category in STRATEGY_CATEGORIES:
        cat_items = [(k, m) for k, m in STRATEGIES_META.items() if m["category"] == category]
        total_pages = max(1, (len(cat_items) + per_page - 1) // per_page)
        page = max(0, min(page, total_pages - 1))
        buttons = []
        for key, meta in cat_items[page * per_page : (page + 1) * per_page]:
            enabled = strategies_status.get(key, False)
            auto_rec = auto_map.get(("GLOBAL", "*", key))
            mark = "⛔ " if (auto_rec and not enabled) else ("✅ " if enabled else "❌ ")
            wr_tag = format_wr_tag(wr_summary.get(key))
            buttons.append([InlineKeyboardButton(f"{mark}{meta['title']} {wr_tag}", callback_data=f"toggle|{key}|{page}")])
        if total_pages > 1:
            nav = []
            if page > 0:
                nav.append(InlineKeyboardButton("⬅️", callback_data=f"cat|{category}|{page - 1}"))
            nav.append(InlineKeyboardButton(f"Стор. {page + 1}/{total_pages}", callback_data="noop"))
            if page < total_pages - 1:
                nav.append(InlineKeyboardButton("➡️", callback_data=f"cat|{category}|{page + 1}"))
            buttons.append(nav)
        buttons.append([
            InlineKeyboardButton("✅ Увімкнути групу", callback_data=f"cat_all|{category}|1"),
            InlineKeyboardButton("❌ Вимкнути групу", callback_data=f"cat_all|{category}|0"),
        ])
        buttons.append([InlineKeyboardButton("⬅️ Назад", callback_data="cat|MAIN")])
        return InlineKeyboardMarkup(buttons)

    buttons = []
    for cat_key, cat_title in STRATEGY_CATEGORIES.items():
        cat_keys = [k for k, m in STRATEGIES_META.items() if m["category"] == cat_key]
        if not cat_keys:
            continue
        enabled_cnt = sum(1 for k in cat_keys if strategies_status.get(k, False))
        has_auto = any(("GLOBAL", "*", k) in auto_map for k in cat_keys)
        wrs = [wr_summary[k]["wr"] for k in cat_keys if k in wr_summary and wr_summary[k].get("wr") is not None]
        avg_s = f"|{sum(wrs) / len(wrs):.0f}%" if wrs else ""
        warn = " ⛔" if has_auto else ""
        buttons.append([InlineKeyboardButton(f"{cat_title} [{enabled_cnt}/{len(cat_keys)}{avg_s}]{warn}", callback_data=f"cat|{cat_key}|0")])
    for f_key, f_title in GLOBAL_FILTERS.items():
        mark = "✅ " if strategies_status.get(f_key, False) else "❌ "
        buttons.append([InlineKeyboardButton(mark + f_title, callback_data=f"toggle|{f_key}|0")])
    buttons.append([
        InlineKeyboardButton("🎯 По парах", callback_data="pairs"),
        InlineKeyboardButton("📊 Рейтинг % WR", callback_data="ranks|0"),
    ])
    buttons.append([
        InlineKeyboardButton(f"🚫 Авто-вимк [{len(auto_map)}]", callback_data="autodis_list|0"),
        InlineKeyboardButton("🗑 Видалення слабких", callback_data="del_menu|0"),
    ])
    buttons.append([InlineKeyboardButton("🎛 Налаштування", callback_data="settings|main"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(buttons)


def _settings_summary_text(rt=None):
    rt = rt or _get_runtime_settings()
    be = 100.0 / (1.0 + rt["payout"])
    src_lbl = "Dukascopy" if rt["bt_source"] == "dukascopy" else "Yahoo"
    return (
        "🎛 <b>Повна панель керування ботом</b>\n"
        f"• Стратегій у реєстрі: <b>{len(STRATEGIES_META)}</b> (видалено: {len(load_deleted_strategies())})\n"
        f"• Авто-вимкнення: <b>&lt;{rt['wr_threshold'] * 100:.0f}%</b> · Поріг видалення: <b>&lt;{rt['prune_threshold'] * 100:.0f}%</b> (≥{rt['prune_min_trades']} угод)\n"
        f"• Lab: <b>{src_lbl} {rt['bt_days']}д</b> · Конфлюентність: <b>{rt['conflict_mode']}</b> (≥{rt['confluence_threshold']:.1f})\n"
        f"• Експірація: <b>{rt['expiry_mode']}</b> · Новини: <b>±{rt['news_blackout_mins']} хв</b> · Виплата: <b>{int(rt['payout'] * 100)}%</b> (беззбиток {be:.1f}%)"
    )


def _build_pair_radar(pair_name):
    rt = _get_runtime_settings()
    ticker = PAIRS_MAP[pair_name]
    yahoo = normalize_yahoo_ticker(ticker, pair_name)
    df_5m = fetch_5m(ticker, yahoo)
    frames = build_mtf_frames(ticker, yahoo, df_5m, include_all_tfs=True)
    diag = diagnose_pair_market(pair_name, frames, get_utc_now(), dict(strategies_status), pair_enabled=database.get_pair_strategy_status, mute_check=database.is_strategy_muted)
    news_st = is_high_impact_news_time(pair_name, blackout_mins=rt["news_blackout_mins"])
    upcoming = get_upcoming_news(pair_name, hours=12)
    live_rows = database.get_detailed_pair_strategy_stats(ticker_filter=ticker)
    bt_rows = database.get_backtest_results(pair_filter=pair_name)
    lp = get_live_price(ticker) or (fetch_biquote_quote(pair_name, allow_stale=False) if BIQUOTE_ENABLED else None)
    sr_plan = sr_levels.analyze_pair_sr(pair_name, frames.get("5m"), frames.get("15m"), frames.get("60m"), live_price=lp)
    return pair_radar_text(diag, news_st, upcoming, live_rows, bt_rows, live_price=lp, payout=rt["payout"], wr_threshold=rt["wr_threshold"], sr_plan=sr_plan)


def start(update, context):
    database.register_user(update.effective_chat.id, update.effective_user.username)
    update.effective_message.reply_text(
        f"📡 <b>Racio_1bot · {len(STRATEGIES_META)} стратегій (8 груп) + S/R Радар</b>\nОберіть дію на панелі або /menu:",
        parse_mode="HTML", reply_markup=main_keyboard(update.effective_chat.id),
    )


def handle_menu_command(update, context):
    update.effective_message.reply_text("🏠 <b>Головне інлайн-меню</b>\nПовне керування через Telegram:", parse_mode="HTML", reply_markup=inline_main_keyboard(update.effective_chat.id))


def handle_check_command(update, context):
    args = [a.upper() for a in (context.args or [])]
    if not args:
        update.effective_message.reply_text("🔍 <b>Радар пари</b>\nОберіть валютну пару:", parse_mode="HTML", reply_markup=check_pairs_keyboard())
        return
    pair = normalize_pair_name(args[0])
    if pair not in PAIRS_MAP:
        update.effective_message.reply_text("Невідома пара. Приклад: /check EUR/USD")
        return
    msg = update.effective_message.reply_text(f"🔍 Аналізую <b>{html.escape(pair)}</b>…", parse_mode="HTML")
    threading.Thread(target=lambda: bot.edit_message_text(chat_id=update.effective_chat.id, message_id=msg.message_id, text=_build_pair_radar(pair), parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(f"📡 Сигнал ({pair})", callback_data=f"scan|pair|{pair}")], [InlineKeyboardButton("⬅️ Інша пара", callback_data="check_menu")]])), daemon=True).start()


def handle_size_command(update, context):
    update.effective_message.reply_text("ℹ️ Розрахунок рекомендованої ставки вимкнено. Орієнтуйтесь на рівень впевненості сигналу.")


def show_stats(chat_id, period="all"):
    rows = database.get_chat_signals(chat_id, since=period_start(period, get_utc_now()))
    rt = _get_runtime_settings()
    try:
        raw_map = database.get_auto_disabled_map()
        auto_global = {k for (_, _, k) in raw_map} if isinstance(raw_map, dict) else set()
    except Exception:
        auto_global = set()
    return stats_text(rows, period, auto_disabled_global=auto_global, payout=rt["payout"], wr_threshold=rt["wr_threshold"])


def handle_message(update, context):
    chat_id, message = update.effective_chat.id, update.effective_message
    text = message.text
    if text == "📡 Сигнал зараз":
        message.reply_text("📡 <b>Ручний аналіз сигналів</b>", parse_mode="HTML", reply_markup=scan_keyboard())
    elif text in ("🤖 Автоскан ON", "🤖 Автоскан OFF"):
        enabled = chat_id not in AUTO_SCAN_CHATS
        AUTO_SCAN_CHATS.add(chat_id) if enabled else AUTO_SCAN_CHATS.discard(chat_id)
        try:
            database.set_user_autoscan(chat_id, enabled)
        except Exception:
            pass
        message.reply_text("🤖 Автоскан " + ("ON" if enabled else "OFF"), reply_markup=main_keyboard(chat_id))
    elif text == "⏳ Активні угоди":
        message.reply_text(active_text(database.get_chat_signals(chat_id, "active"), get_utc_now()), parse_mode="HTML", reply_markup=active_keyboard())
    elif text == "📜 Історія":
        reply_report(message, history_text(database.get_chat_signals(chat_id, "history")))
    elif text == "📊 Статистика":
        message.reply_text(show_stats(chat_id), parse_mode="HTML", reply_markup=stats_keyboard())
    elif text == "🧪 Бектест":
        message.reply_text("🧪 <b>Інтерактивний бектест і Lab</b>", parse_mode="HTML", reply_markup=backtest_keyboard("all"))
    elif text == "🎛 Налаштування":
        if require_admin(update):
            rt = _get_runtime_settings()
            message.reply_text(_settings_summary_text(rt), parse_mode="HTML", reply_markup=settings_menu_keyboard(rt))
    elif text == "⚙️ Стратегії":
        if require_admin(update):
            message.reply_text(f"⚙️ <b>Групи стратегій ({len(STRATEGIES_META)}) та % вінрейту</b>", parse_mode="HTML", reply_markup=strategy_keyboard())
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
    settings, rt = dict(strategies_status), _get_runtime_settings()
    def worker():
        try:
            if pair == "ALL":
                report, stats_rows = run_full_backtest_all_with_stats(strategies_status=settings, include_all_strategies=include_all, payout=rt["payout"], wr_threshold=rt["wr_threshold"], expiry_mode=rt["expiry_mode"], source=rt["bt_source"], days=rt["bt_days"])
            else:
                report, stats_rows = run_full_backtest_with_stats(pair, strategies_status=settings, include_all_strategies=include_all, payout=rt["payout"], wr_threshold=rt["wr_threshold"], expiry_mode=rt["expiry_mode"], source=rt["bt_source"], days=rt["bt_days"])

            if stats_rows:
                database.upsert_backtest_results(stats_rows)
                if settings.get("auto_disable_49", True):
                    events = database.evaluate_and_apply_auto_disable(source="BACKTEST", pair_filter=None if pair == "ALL" else pair, threshold=rt["wr_threshold"])
                    auto_msg = _sync_auto_disable_events(events, wr_threshold=rt["wr_threshold"])
                    if auto_msg:
                        report += f"\n\n🛡 <b>Авто-вимкнення (&lt;{rt['wr_threshold'] * 100:.0f}% WR):</b>\n" + auto_msg
                if pair == "ALL":
                    pruned = prune_strategies_below_threshold(stats_rows, STRATEGIES_CONFIG_FILE, min_wr=rt["prune_threshold"], min_trades=rt["prune_min_trades"])
                    if pruned["disabled"]:
                        with _CFG_LOCK:
                            strategies_status.update(load_all_strategy_config(STRATEGIES_CONFIG_FILE))
                        report += f"\n\n🧹 <b>Відсіяно нижче {rt['prune_threshold'] * 100:.0f}%:</b> {len(pruned['disabled'])} стратегій (готові до повного видалення в меню «🗑 Видалити відсіяні»)."

            send_chunked_html(chat_id, report)
            if pair == "ALL" and stats_rows:
                bot.send_document(chat_id=chat_id, document=io.BytesIO(build_backtest_csv(stats_rows, wr_threshold=rt["wr_threshold"], prune_th=rt["prune_threshold"])), filename="backtest_matrix.csv", caption="📊 Матриця бектесту: Пара × Стратегія")
        except Exception:
            logger.exception("Помилка бектесту (%s)", pair)
            bot.send_message(chat_id=chat_id, text="❌ Не вдалося виконати бектест.")
    threading.Thread(target=worker, daemon=True).start()


def _build_review_context(sig: dict) -> dict:
    enriched = dict(sig)
    try:
        pair = normalize_pair_name(sig.get("ticker"))
        if sig.get("created_at"):
            enriched["session"] = localized_session_name(sig["created_at"])
        if pair in PAIRS_MAP:
            yahoo = normalize_yahoo_ticker(sig["ticker"], pair)
            df5 = fetch_yahoo_fallback(yahoo, "5m", "5d")
            df60 = fetch_yahoo_fallback(yahoo, "60m", "30d")
            if df5 is not None and not df5.empty:
                sr = sr_levels.analyze_pair_sr(pair, df5, df_60m=df60, live_price=float(sig.get("entry_price") or 0) or None)
                if sr.get("ok"):
                    ls = sr["local_support"]["price"] if sr.get("local_support") else None
                    lr = sr["local_resistance"]["price"] if sr.get("local_resistance") else None
                    gs = sr["global_support"]["price"] if sr.get("global_support") else None
                    gr = sr["global_resistance"]["price"] if sr.get("global_resistance") else None
                    enriched["sr_context"] = f"Loc S={ls} / R={lr}; Glob S={gs} / R={gr}"
                if sig.get("created_at") and sig.get("expires_at") and sig.get("entry_price"):
                    t0, t1 = parse_dt(sig["created_at"]), parse_dt(sig["expires_at"])
                    win_df = df5.loc[(df5.index >= t0 - timedelta(minutes=5)) & (df5.index <= t1 + timedelta(minutes=5))]
                    if not win_df.empty:
                        ep, mult = float(sig["entry_price"]), (1000.0 if "JPY" in pair else 100000.0)
                        hi_p, lo_p = float(win_df["high"].max()), float(win_df["low"].min())
                        mfe = (hi_p - ep) * mult if str(sig.get("signal_type", "")).upper() == "CALL" else (ep - lo_p) * mult
                        mae = (lo_p - ep) * mult if str(sig.get("signal_type", "")).upper() == "CALL" else (ep - hi_p) * mult
                        enriched["price_path_summary"] = f"MFE={mfe:+.1f} пт, MAE={mae:+.1f} пт"
    except Exception:
        pass
    return enriched


def handle_callback(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id
    parts = query.data.split("|")
    action = parts[0]

    if action == "noop":
        query.answer()
        return

    if action == "st_stat" and len(parts) == 2 and parts[1] in STRATEGIES_META:
        s_key = parts[1]
        rt = _get_runtime_settings()
        alert_txt = strategy_stat_text(s_key, database.get_detailed_pair_strategy_stats(strategy_filter=s_key), database.get_backtest_results(strategy_filter=s_key), payout=rt["payout"])
        query.answer(alert_txt[:200], show_alert=True)
        return

    query.answer()

    if action == "main_menu":
        _safe_edit(query, "🏠 <b>Головне інлайн-меню</b>\nПовне керування ботом через Telegram:", parse_mode="HTML", reply_markup=inline_main_keyboard(chat_id))
    elif action == "autoscan_tog":
        enabled = chat_id not in AUTO_SCAN_CHATS
        AUTO_SCAN_CHATS.add(chat_id) if enabled else AUTO_SCAN_CHATS.discard(chat_id)
        try:
            database.set_user_autoscan(chat_id, enabled)
        except Exception:
            pass
        _safe_edit(query, f"🤖 Автоскан тепер <b>{'ON' if enabled else 'OFF'}</b>.", parse_mode="HTML", reply_markup=inline_main_keyboard(chat_id))
    elif action == "stats" and len(parts) == 2 and parts[1] in PERIODS:
        _safe_edit(query, show_stats(chat_id, parts[1]), parse_mode="HTML", reply_markup=stats_keyboard())
    elif action == "ranks":
        page = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        rt = _get_runtime_settings()
        txt, total_p = strategy_ranks_text(database.get_strategy_winrate_summary(), page=page, payout=rt["payout"], wr_threshold=rt["wr_threshold"])
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("⬅️", callback_data=f"ranks|{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{total_p}", callback_data="noop"))
        if page < total_p - 1:
            nav.append(InlineKeyboardButton("➡️", callback_data=f"ranks|{page + 1}"))
        _safe_edit(query, txt, parse_mode="HTML", reply_markup=InlineKeyboardMarkup([nav, [InlineKeyboardButton("⚙️ До стратегій", callback_data="cat|MAIN"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")]]))
    elif action == "stats_matrix":
        rt = _get_runtime_settings()
        text = stats_matrix_text(database.get_detailed_pair_strategy_stats(), database.get_backtest_results(), payout=rt["payout"], wr_threshold=rt["wr_threshold"])
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
                live_map[tkr] = get_live_price(tkr) or (fetch_biquote_quote(tkr, allow_stale=False) if BIQUOTE_ENABLED else None)
        _safe_edit(query, active_text(rows, get_utc_now(), live_prices=live_map), parse_mode="HTML", reply_markup=active_keyboard())
    elif action == "check_menu":
        _safe_edit(query, "🔍 <b>Радар пари (локальні та глобальні S/R)</b>\nОберіть валютну пару:", parse_mode="HTML", reply_markup=check_pairs_keyboard())
    elif action == "check_go" and len(parts) == 2 and parts[1] in PAIRS_MAP:
        pair, msg_id = parts[1], query.message.message_id
        _safe_edit(query, f"🔍 Завантажую рівні S/R для <b>{html.escape(pair)}</b>…", parse_mode="HTML")
        threading.Thread(target=lambda: bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=_build_pair_radar(pair), parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(f"📡 Шукати сигнал ({pair})", callback_data=f"scan|pair|{pair}")], [InlineKeyboardButton("⬅️ Інша пара", callback_data="check_menu"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")]])), daemon=True).start()
    elif action == "scan":
        sub = parts[1] if len(parts) > 1 else "menu"
        if sub in ("menu", "pairs_menu"):
            _safe_edit(query, "📡 <b>Ручний аналіз сигналів</b>", parse_mode="HTML", reply_markup=scan_keyboard())
        elif sub == "all" or (sub == "pair" and len(parts) == 3 and parts[2] in PAIRS_MAP):
            target, msg_id = ("ALL" if sub == "all" else parts[2]), query.message.message_id
            _safe_edit(query, "📡 Перевіряю 21 пару…" if target == "ALL" else f"📡 Аналізую <b>{html.escape(target)}</b>…", parse_mode="HTML")
            def scan():
                try:
                    if target == "ALL":
                        _warm_scan_caches()
                        cnt = sum(analyze_single_pair(chat_id, p, t) for p, t in PAIRS_MAP.items())
                        bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=f"📡 Сканування 21 пари завершено · сигналів: <b>{cnt}</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔄 Повторити", callback_data="scan|menu"), InlineKeyboardButton("🏠 Меню", callback_data="main_menu")]]))
                    else:
                        bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=analyze_single_pair_report(chat_id, target, PAIRS_MAP[target]), parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(f"🔄 Ще раз {target}", callback_data=f"scan|pair|{target}")], [InlineKeyboardButton("🔍 Радар", callback_data=f"check_go|{target}"), InlineKeyboardButton("⬅️ Пари", callback_data="scan|menu")]]))
                except Exception:
                    bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text="❌ Помилка під час сканування.")
            threading.Thread(target=scan, daemon=True).start()
    elif action == "bt_menu" and len(parts) == 2 and parts[1] in ("all", "act"):
        _safe_edit(query, "🧪 <b>Інтерактивний бектест і Lab</b>", parse_mode="HTML", reply_markup=backtest_keyboard(parts[1]))
    elif action == "bt_go" and len(parts) == 3 and parts[2] in ("all", "act"):
        pair = parts[1]
        if pair != "ALL" and pair not in PAIRS_MAP:
            return
        _safe_edit(query, f"⏳ Запускаю бектест: <b>{html.escape(pair)}</b>…", parse_mode="HTML")
        _run_backtest_async(chat_id, pair, parts[2] == "all")
    elif action == "risk":
        _safe_edit(query, "ℹ️ Калькулятор ставки вилучено.", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🏠 Меню", callback_data="main_menu")]]))
    elif action in (
        "toggle", "ftog", "cat", "cat_all", "pairs", "pair", "pair_all", "pcat", "pcat_all", "pair_toggle",
        "stats_clear_ask", "stats_clear_confirm", "db_reset_ask", "db_reset_confirm",
        "autodis_list", "autodis_en", "mute1h", "settings", "set_wr", "set_prune", "set_mintr",
        "set_bt_src", "set_bt_days", "set_conf", "set_cmod", "set_emod", "set_news", "set_pay", "set_tz",
        "lab_run_tg", "prune_now", "del_menu", "del_pruned_ask", "del_pruned_confirm",
        "del_one", "del_one_ok", "del_restore_ask", "del_restore_confirm", "review",
    ):
        if not require_admin(update):
            return
        if action == "mute1h" and len(parts) == 2 and parts[1] in STRATEGIES_META:
            database.mute_strategy_until(parts[1], 3600)
            try:
                query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔕 На паузі (1г)", callback_data=f"st_stat|{parts[1]}")], [InlineKeyboardButton("🏠 Меню", callback_data="main_menu")]]))
            except Exception:
                pass
        elif action == "lab_run_tg":
            _safe_edit(query, "🔬 <b>Запущено повний Lab по всіх активних стратегіях...</b>\nЗвіт і файл з'являться в чаті після завершення.", parse_mode="HTML", reply_markup=backtest_keyboard("all"))
            _run_lab_in_thread(chat_id)
        elif action == "prune_now":
            rt = _get_runtime_settings()
            res = prune_strategies_below_threshold(None, STRATEGIES_CONFIG_FILE, min_wr=rt["prune_threshold"], min_trades=rt["prune_min_trades"])
            with _CFG_LOCK:
                strategies_status.update(load_all_strategy_config(STRATEGIES_CONFIG_FILE))
            cnt = len(res["disabled"])
            txt = f"🧹 <b>Авто-відсів (&lt;{rt['prune_threshold'] * 100:.0f}%, ≥{rt['prune_min_trades']} угод):</b> вимкнено {cnt} стратегій."
            _safe_edit(query, txt, parse_mode="HTML", reply_markup=delete_strategies_keyboard(rt=rt))
        elif action == "del_menu":
            page = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            rt = _get_runtime_settings()
            _safe_edit(query, f"🗑 <b>Повне видалення слабких/вимкнених стратегій</b>\nПоріг відсіву: <b>&lt;{rt['prune_threshold'] * 100:.0f}%</b> (≥{rt['prune_min_trades']} угод).", parse_mode="HTML", reply_markup=delete_strategies_keyboard(page=page, rt=rt))
        elif action == "del_pruned_ask":
            rt = _get_runtime_settings()
            wr_sum = database.get_strategy_winrate_summary()
            cands = _get_prunable_and_disabled(wr_sum, rt["prune_threshold"], rt["prune_min_trades"])
            _safe_edit(query, f"🚨 <b>Видалити {len(cands)} відсіяних/вимкнених стратегій?</b>\nВони будуть повністю видалені з меню, сканера, бектесту й таблиць статистики.", parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(f"🔥 ТАК, ВИДАЛИТИ ({len(cands)})", callback_data="del_pruned_confirm"), InlineKeyboardButton("❌ Скасувати", callback_data="del_menu|0")]]))
        elif action == "del_pruned_confirm":
            rt = _get_runtime_settings()
            wr_sum = database.get_strategy_winrate_summary()
            keys = [k for k, _ in _get_prunable_and_disabled(wr_sum, rt["prune_threshold"], rt["prune_min_trades"])]
            removed = delete_strategies_permanently(keys, STRATEGIES_CONFIG_FILE)
            database.delete_strategies_data(removed)
            with _CFG_LOCK:
                for k in removed:
                    strategies_status.pop(k, None)
            _safe_edit(query, f"✅ <b>Повністю видалено {len(removed)} стратегій.</b>\nАктивних у реєстрі: {len(STRATEGIES_META)}", parse_mode="HTML", reply_markup=delete_strategies_keyboard(rt=rt))
        elif action == "del_one" and len(parts) == 2 and parts[1] in STRATEGIES_META:
            sk = parts[1]
            _safe_edit(query, f"🗑 <b>Видалити стратегію <code>{html.escape(sk)}</code>?</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("🔥 Так, видалити", callback_data=f"del_one_ok|{sk}"), InlineKeyboardButton("❌ Скасувати", callback_data="del_menu|0")]]))
        elif action == "del_one_ok" and len(parts) == 2:
            sk = parts[1]
            removed = delete_strategies_permanently([sk], STRATEGIES_CONFIG_FILE)
            database.delete_strategies_data(removed)
            with _CFG_LOCK:
                strategies_status.pop(sk, None)
            _safe_edit(query, f"✅ Стратегію <code>{html.escape(sk)}</code> повністю видалено.", parse_mode="HTML", reply_markup=delete_strategies_keyboard())
        elif action == "del_restore_ask":
            _safe_edit(query, "♻️ <b>Відновити всі видалені стратегії?</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("✅ Так, відновити", callback_data="del_restore_confirm"), InlineKeyboardButton("❌ Скасувати", callback_data="del_menu|0")]]))
        elif action == "del_restore_confirm":
            cnt = restore_all_deleted_strategies(STRATEGIES_CONFIG_FILE)
            with _CFG_LOCK:
                strategies_status.update(load_all_strategy_config(STRATEGIES_CONFIG_FILE))
            _safe_edit(query, f"♻️ Відновлено <b>{cnt}</b> стратегій. У реєстрі: {len(STRATEGIES_META)}.", parse_mode="HTML", reply_markup=strategy_keyboard())
        elif action == "stats_clear_ask":
            _safe_edit(query, "⚠️ <b>Очистити всю статистику?</b>", parse_mode="HTML", reply_markup=clear_stats_confirm_keyboard())
        elif action == "stats_clear_confirm":
            ok = database.clear_all_stats()
            _safe_edit(query, "🧹 <b>Статистику очищено.</b>" if ok else "❌ Помилка.", parse_mode="HTML", reply_markup=stats_keyboard())
        elif action == "db_reset_ask":
            _safe_edit(query, "🚨 <b>ПОВНЕ ОБНУЛЕННЯ БОТА ДО СТАНУ GITHUB</b>\nОчистить БД, видалені стратегії та конфіг.", parse_mode="HTML", reply_markup=db_reset_confirm_keyboard())
        elif action == "db_reset_confirm":
            try:
                res = db_reset.reset_database_and_config(STRATEGIES_CONFIG_FILE, reset_config=True)
                with _CFG_LOCK:
                    strategies_status.clear()
                    strategies_status.update(load_all_strategy_config(STRATEGIES_CONFIG_FILE))
                AUTO_SCAN_CHATS.clear()
                signal_gate.reset()
                _safe_edit(query, f"✅ <b>БД і конфіг обнулено.</b> Таблиці: <code>{html.escape(', '.join(res.get('tables') or []))}</code>", parse_mode="HTML", reply_markup=inline_main_keyboard(chat_id))
            except Exception as e:
                _safe_edit(query, f"❌ Помилка: {html.escape(str(e)[:120])}", parse_mode="HTML", reply_markup=settings_menu_keyboard())
        elif action == "settings":
            sub, rt = (parts[1] if len(parts) > 1 else "main"), _get_runtime_settings()
            if sub == "main":
                _safe_edit(query, _settings_summary_text(rt), parse_mode="HTML", reply_markup=settings_menu_keyboard(rt))
            elif sub == "thresholds":
                _safe_edit(query, f"🎚 <b>Пороги WR%</b>\nАвто-вимкнення: <b>{rt['wr_threshold'] * 100:.0f}%</b> · Відсів: <b>{rt['prune_threshold'] * 100:.0f}%</b>", parse_mode="HTML", reply_markup=thresholds_keyboard(rt))
            elif sub == "lab":
                _safe_edit(query, f"🔬 <b>Налаштування Lab / Бектесту</b>\nДжерело: <b>{rt['bt_source']}</b> · Глибина: <b>{rt['bt_days']}д</b> · Мін. угод: <b>{rt['prune_min_trades']}</b>", parse_mode="HTML", reply_markup=lab_settings_keyboard(rt))
            elif sub == "confluence":
                _safe_edit(query, f"🧩 <b>Конфлюентність</b> ({rt['conflict_mode']} · ≥ {rt['confluence_threshold']:.1f})", parse_mode="HTML", reply_markup=confluence_keyboard(rt))
            elif sub == "expiry":
                _safe_edit(query, f"⏱ <b>Режим експірації: {rt['expiry_mode']}</b>", parse_mode="HTML", reply_markup=expiry_keyboard(rt))
            elif sub == "filters":
                _safe_edit(query, f"📰 <b>Фільтри, новини (±{rt['news_blackout_mins']}хв) та виплата ({int(rt['payout'] * 100)}%)</b>", parse_mode="HTML", reply_markup=filters_keyboard(rt))
            elif sub == "tz":
                _safe_edit(query, f"🕒 <b>Часовий пояс: {rt['timezone']}</b>", parse_mode="HTML", reply_markup=tz_keyboard(rt))
        elif action == "set_wr" and len(parts) == 2:
            val = max(0.30, min(0.70, float(parts[1]) / 100.0))
            database.set_bot_setting("wr_threshold", f"{val:.2f}")
            _safe_edit(query, f"🎚 Поріг авто-вимкнення: <b>{val * 100:.0f}%</b>", parse_mode="HTML", reply_markup=thresholds_keyboard())
        elif action == "set_prune" and len(parts) == 2:
            val = max(0.30, min(0.70, float(parts[1]) / 100.0))
            database.set_bot_setting("prune_threshold", f"{val:.2f}")
            _safe_edit(query, f"🧹 Поріг відсіву оновлено: <b>{val * 100:.0f}%</b>", parse_mode="HTML", reply_markup=thresholds_keyboard())
        elif action == "set_mintr" and len(parts) == 2:
            val = max(5, min(200, int(parts[1])))
            database.set_bot_setting("prune_min_trades", str(val))
            _safe_edit(query, f"🔬 Мін. угод для відсіву: <b>{val}</b>", parse_mode="HTML", reply_markup=lab_settings_keyboard())
        elif action == "set_bt_src" and len(parts) == 2 and parts[1] in ("yahoo", "dukascopy"):
            database.set_bot_setting("bt_source", parts[1])
            _safe_edit(query, f"🔬 Джерело Lab: <b>{parts[1]}</b>", parse_mode="HTML", reply_markup=lab_settings_keyboard())
        elif action == "set_bt_days" and len(parts) == 2:
            val = max(7, min(730, int(parts[1])))
            database.set_bot_setting("bt_days", str(val))
            _safe_edit(query, f"🔬 Глибина історії: <b>{val} дн.</b>", parse_mode="HTML", reply_markup=lab_settings_keyboard())
        elif action == "set_cmod" and len(parts) == 2 and parts[1] in CONFLUENCE_MODES:
            database.set_bot_setting("conflict_mode", parts[1])
            _safe_edit(query, f"🧩 Режим конфлюєнтності: <b>{parts[1]}</b>", parse_mode="HTML", reply_markup=confluence_keyboard())
        elif action == "set_conf" and len(parts) == 2:
            val = max(0.5, min(5.0, float(parts[1])))
            database.set_bot_setting("confluence_threshold", f"{val:.1f}")
            _safe_edit(query, f"🧩 Поріг конфлюєнтності: <b>{val:.1f}</b>", parse_mode="HTML", reply_markup=confluence_keyboard())
        elif action == "set_emod" and len(parts) == 2 and parts[1] in EXPIRY_MODES:
            database.set_bot_setting("expiry_mode", parts[1])
            _safe_edit(query, f"⏱ Режим експірації: <b>{parts[1]}</b>", parse_mode="HTML", reply_markup=expiry_keyboard())
        elif action == "set_news" and len(parts) == 2:
            mins = max(5, min(60, int(parts[1])))
            database.set_bot_setting("news_blackout_mins", str(mins))
            _safe_edit(query, f"📰 Новинний карантин: <b>±{mins} хв</b>", parse_mode="HTML", reply_markup=filters_keyboard())
        elif action == "set_pay" and len(parts) == 2:
            pay = max(0.50, min(0.98, float(parts[1]) / 100.0))
            database.set_bot_setting("payout", f"{pay:.2f}")
            _safe_edit(query, f"💹 Виплата: <b>{int(pay * 100)}%</b>", parse_mode="HTML", reply_markup=filters_keyboard())
        elif action == "set_tz" and len(parts) == 2 and parts[1] in SUPPORTED_TIMEZONES:
            database.set_bot_setting("timezone", parts[1])
            _safe_edit(query, f"🕒 Часовий пояс: <b>{parts[1]}</b>", parse_mode="HTML", reply_markup=tz_keyboard())
        elif action == "ftog" and len(parts) == 2 and parts[1] in GLOBAL_FILTERS:
            with _CFG_LOCK:
                changed = dict(strategies_status)
                changed[parts[1]] = not changed.get(parts[1], False)
                save_strategies_config(changed)
                strategies_status.update(changed)
            _safe_edit(query, "⚙️ Фільтр оновлено:", reply_markup=thresholds_keyboard() if parts[1] == "auto_disable_49" else filters_keyboard())
        elif action == "cat" and len(parts) >= 2:
            cat = parts[1]
            page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            if cat == "MAIN":
                _safe_edit(query, f"⚙️ <b>Групи стратегій ({len(STRATEGIES_META)}) та % вінрейту</b>", parse_mode="HTML", reply_markup=strategy_keyboard())
            elif cat in STRATEGY_CATEGORIES:
                _safe_edit(query, STRATEGY_CATEGORIES[cat], reply_markup=strategy_keyboard(category=cat, page=page))
        elif action == "cat_all" and len(parts) == 3 and parts[1] in STRATEGY_CATEGORIES:
            cat, on = parts[1], parts[2] == "1"
            with _CFG_LOCK:
                changed = dict(strategies_status)
                for k, m in STRATEGIES_META.items():
                    if m["category"] == cat:
                        changed[k] = on
                save_strategies_config(changed)
                strategies_status.update(changed)
            _safe_edit(query, STRATEGY_CATEGORIES[cat], reply_markup=strategy_keyboard(category=cat))
        elif action == "toggle" and len(parts) >= 2:
            key = parts[1]
            page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            if key in STRATEGIES_META or key in GLOBAL_FILTERS:
                with _CFG_LOCK:
                    changed = dict(strategies_status)
                    changed[key] = not changed.get(key, False)
                    save_strategies_config(changed)
                    strategies_status.update(changed)
                cat = STRATEGIES_META[key]["category"] if key in STRATEGIES_META else "MAIN"
                title = STRATEGY_CATEGORIES.get(cat, "⚙️ Групи стратегій")
                _safe_edit(query, title, reply_markup=strategy_keyboard(category=None if cat == "MAIN" else cat, page=page))
        elif action == "pairs":
            _safe_edit(query, "🎯 Оберіть пару:", reply_markup=pairs_keyboard(database.get_auto_disabled_map()))
        elif action == "pair" and len(parts) == 2 and parts[1] in PAIRS_MAP:
            _safe_edit(query, f"🎯 Стратегії для <b>{html.escape(parts[1])}</b>", parse_mode="HTML", reply_markup=strategy_keyboard(pair=parts[1]))
        elif action == "pair_all" and len(parts) == 3 and parts[1] in PAIRS_MAP:
            database.set_pair_bulk_strategy_status(parts[1], list(ALL_STRATEGY_KEYS), parts[2] == "1")
            _safe_edit(query, f"🎯 Стратегії для <b>{html.escape(parts[1])}</b>", parse_mode="HTML", reply_markup=strategy_keyboard(pair=parts[1]))
        elif action == "pcat" and len(parts) >= 3 and parts[1] in PAIRS_MAP and parts[2] in STRATEGY_CATEGORIES:
            page = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            _safe_edit(query, f"🎯 {html.escape(parts[1])} · {html.escape(STRATEGY_CATEGORIES[parts[2]])}", parse_mode="HTML", reply_markup=strategy_keyboard(category=parts[2], pair=parts[1], page=page))
        elif action == "pcat_all" and len(parts) == 4 and parts[1] in PAIRS_MAP and parts[2] in STRATEGY_CATEGORIES:
            database.set_pair_bulk_strategy_status(parts[1], [k for k, m in STRATEGIES_META.items() if m["category"] == parts[2]], parts[3] == "1")
            _safe_edit(query, f"🎯 {html.escape(parts[1])} · {html.escape(STRATEGY_CATEGORIES[parts[2]])}", parse_mode="HTML", reply_markup=strategy_keyboard(category=parts[2], pair=parts[1]))
        elif action == "pair_toggle" and len(parts) >= 3 and parts[1] in PAIRS_MAP and parts[2] in STRATEGIES_META:
            pair, key = parts[1], parts[2]
            page = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            database.set_pair_strategy_status(pair, key, not database.get_pair_strategy_status(pair, key))
            cat = STRATEGIES_META[key]["category"]
            _safe_edit(query, f"🎯 {html.escape(pair)} · {html.escape(STRATEGY_CATEGORIES[cat])}", parse_mode="HTML", reply_markup=strategy_keyboard(category=cat, pair=pair, page=page))
        elif action == "autodis_list":
            page = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            items, rt = database.get_auto_disabled_items(), _get_runtime_settings()
            _safe_edit(query, auto_disabled_text(items, wr_threshold=rt["wr_threshold"]), parse_mode="HTML", reply_markup=auto_disabled_keyboard(items, page=page))
        elif action == "autodis_en" and len(parts) >= 3:
            if parts[1] == "G":
                database.mark_strategy_manual_enable("GLOBAL", "*", parts[2])
                with _CFG_LOCK:
                    changed = dict(strategies_status)
                    if parts[2] in changed:
                        changed[parts[2]] = True
                        save_strategies_config(changed)
                        strategies_status.update(changed)
            elif parts[1] == "P" and len(parts) == 4:
                database.mark_strategy_manual_enable("PAIR", parts[2], parts[3])
            items, rt = database.get_auto_disabled_items(), _get_runtime_settings()
            _safe_edit(query, auto_disabled_text(items, wr_threshold=rt["wr_threshold"]), parse_mode="HTML", reply_markup=auto_disabled_keyboard(items))
        elif action == "review" and len(parts) == 2 and parts[1].isdigit():
            sig_id = int(parts[1])
            def review():
                try:
                    sig = database.get_signal_by_id(sig_id)
                    if not sig:
                        bot.send_message(chat_id=chat_id, text="⚠️ Угоду не знайдено.")
                        return
                    text = sig.get("ai_review")
                    if not text:
                        text = ai_advisor_instance.evaluate_closed_trade(_build_review_context(sig))
                        if text and not text.startswith("⚠️"):
                            database.save_signal_ai_review(sig_id, text)
                    bot.send_message(chat_id=chat_id, text=f"🔍 <b>AI-розбір угоди #{sig_id}:</b>\n{html.escape(text)}", parse_mode="HTML")
                except Exception:
                    bot.send_message(chat_id=chat_id, text="❌ Помилка під час AI-розбору.")
            threading.Thread(target=review, daemon=True).start()


def handle_backtest_command(update, context):
    args = [a.upper() for a in (context.args or [])]
    if not args:
        update.effective_message.reply_text("🧪 <b>Інтерактивний бектест і Lab</b>", parse_mode="HTML", reply_markup=backtest_keyboard("all"))
        return
    rev_only = "REV" in args
    args = [a for a in args if a != "REV"]
    pair = args[0] if args else "ALL"
    if pair != "ALL" and pair not in PAIRS_MAP:
        update.effective_message.reply_text("Невідома пара. Приклад: /backtest EUR/USD або /backtest ALL")
        return
    chat_id = update.effective_chat.id
    if rev_only:
        update.effective_message.reply_text(f"📊 REV 5m ({pair}) · порівнюю сесії OFF / ON…")
        settings = dict(strategies_status)
        threading.Thread(target=lambda: [send_chunked_html(chat_id, r) for r in (run_backtest_all_detailed(strategies_status=settings) if pair == "ALL" else [run_backtest(pair, strategies_status=settings)])], daemon=True).start()
    else:
        update.effective_message.reply_text(f"🧪 Запускаю бектест по всіх {len(STRATEGIES_META)} стратегіях для <b>{html.escape(pair)}</b>…", parse_mode="HTML")
        _run_backtest_async(chat_id, pair, include_all=True)


def send_daily_report(now=None):
    now, rt = now or get_utc_now(), _get_runtime_settings()
    for chat_id in list(AUTO_SCAN_CHATS):
        try:
            rows = database.get_chat_signals(chat_id, since=period_start("today", now), until=utc_datetime(now).replace(tzinfo=None))
            lines = ["🌙 <b>Денний звіт · сьогодні (Київ)</b>", wr_line("Усього", rows, payout=rt["payout"], wr_threshold=rt["wr_threshold"])]
            for st in sorted({r.get("strategy") for r in rows if r.get("strategy")})[:15]:
                lines.append(wr_line(st, [row for row in rows if row.get("strategy") == st], payout=rt["payout"], wr_threshold=rt["wr_threshold"]))
            bot.send_message(chat_id=chat_id, text="\n".join(lines), parse_mode="HTML")
        except Exception:
            pass


def start_daily_report_loop():
    def loop():
        while True:
            now = utc_datetime(get_utc_now())
            time.sleep(max(1, (next_daily_report(now) - now).total_seconds()))
            send_daily_report()
    threading.Thread(target=loop, daemon=True).start()


def main():
    global bot
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN не задано")
    if not ADMIN_IDS and not ALLOW_OPEN_ADMIN:
        logger.warning("ADMIN_IDS не задано: адмін-команди заблоковано.")
    bot = Bot(token=token)
    database.init_db()
    AUTO_SCAN_CHATS.update(database.get_autoscan_chats())
    _recover_pending_timers()
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
    try:
        updater.start_polling()
        updater.idle()
    finally:
        stop_finnhub_ws()


if __name__ == "__main__":
    main()
