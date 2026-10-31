# -*- coding: utf-8 -*-
import html
import io
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

import matplotlib
import pandas as pd
import requests
import yfinance as yf

matplotlib.use("Agg")
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    Filters,
    MessageHandler,
    Updater,
)

from config import FINNHUB_API_KEY, FINNHUB_TOKEN, PAIRS_MAP, YAHOO_PAIRS_MAP
from charts import create_chart_image
import database
from finnhub_ws import get_live_price, start_finnhub_ws
from indicators import AdaptiveTechnicalAnalysis
from ml_model import TradingMLFilter
from ai_advisor import ai_advisor_instance
from filters import (
    check_pivot_level_proximity,
    validate_signal_conditions,
    calculate_dynamic_expiration,
)
from calendar_filter import is_high_impact_news_time
from backtest import (
    run_backtest,
    run_backtest_all_detailed,
    clean_strategy_name,
    PERFORMANCE_CACHE_FILE,
    clear_backtest_cache,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не знайдено!")
    raise ValueError("❌ TELEGRAM_TOKEN не знайдено у змінних оточення!")

bot = Bot(token=TELEGRAM_TOKEN)

analyzer = AdaptiveTechnicalAnalysis()
ml_filter = TradingMLFilter()

last_sent_signals = {}
MACRO_CACHE = {}
CACHE_TTL = 900
AUTO_SCAN_CHATS = set()
ML_FILTER_ENABLED = True

STRATEGIES_CONFIG_FILE = "strategies_config.json"

GROUPED_STRATEGIES = {
    "trend_multi": "🚀 TREND_IMPULSE (Тренд)",
    "flat_corridor": "📉 RANGE_CORRIDOR (Флет)",
    "price_action": "🕯️ PRICE_LEVELS (Price Action)",
    "oscillators": "⚡ MICRO_BREAKOUT (Пробій)"
}

def load_strategies_config():
    default_cfg = {key: True for key in GROUPED_STRATEGIES.keys()}
    if os.path.exists(STRATEGIES_CONFIG_FILE):
        try:
            with open(STRATEGIES_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k in default_cfg:
                    if k not in saved:
                        saved[k] = True
                return saved
        except Exception:
            pass
    return default_cfg

def save_strategies_config(cfg):
    try:
        with open(STRATEGIES_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Помилка збереження конфігу стратегій: {e}")

strategies_status = load_strategies_config()

database.init_db()
start_finnhub_ws()


def get_utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def make_progress_bar(val_pct):
    score = max(0, min(10, int(round(val_pct / 10))))
    filled = "█" * score
    empty = "░" * (10 - score)
    return f"[{filled}{empty}]"


def clean_ai_html(text_val: str) -> str:
    if not text_val:
        return ""
    txt = html.escape(str(text_val))
    txt = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', txt)
    txt = re.sub(r'\*(.*?)\*', r'<i>\1</i>', txt)
    return txt


def parse_dt(dt_val):
    if isinstance(dt_val, datetime):
        return dt_val.replace(tzinfo=None)
    if not dt_val:
        return get_utc_now()
    dt_str = str(dt_val).split(".")[0].replace("T", " ")
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return get_utc_now()


def normalize_yahoo_ticker(ticker, name=""):
    if name and name in YAHOO_PAIRS_MAP:
        return YAHOO_PAIRS_MAP[name]
    clean = ticker.replace("OANDA:", "").replace("IC MARKETS:", "").replace("_", "").replace("/", "")
    if len(clean) == 6 and not clean.endswith("=X"):
        return f"{clean}=X"
    return ticker


def fetch_finnhub_candles(symbol, resolution="1", count_candles=500):
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
        if resp.status_code == 200:
            data = resp.json()
            if data.get("s") == "ok":
                df = pd.DataFrame({"open": data["o"], "high": data["h"], "low": data["l"], "close": data["c"], "volume": data["v"]}, index=pd.to_datetime(data["t"], unit="s"))
                df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                return df
    except Exception as e:
        logger.warning(f"Finnhub REST API error for {symbol}: {e}")
    return pd.DataFrame()


def fetch_yahoo_fallback(ticker, interval="1m", range_period="5d"):
    try:
        tk = yf.Ticker(ticker)
        df_yf = tk.history(period=range_period, interval=interval, auto_adjust=True)
        if df_yf is None or df_yf.empty:
            df_yf = yf.download(tickers=ticker, period=range_period, interval=interval, progress=False, auto_adjust=True)
        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            df_yf = df_yf.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df_yf = df_yf[["open", "high", "low", "close", "volume"]].copy()
            df_yf.dropna(subset=["open", "high", "low", "close"], inplace=True)
            if df_yf.index.tz is not None:
                df_yf.index = df_yf.index.tz_localize(None)
            return df_yf
    except Exception as e:
        logger.warning(f"Yahoo fallback failed for {ticker}: {e}")
    return pd.DataFrame()


def resample_candles(df_1m, rule):
    if df_1m.empty:
        return pd.DataFrame()
    return df_1m.resample(rule).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def fetch_all_timeframes(ticker_finnhub, ticker_yahoo=""):
    global MACRO_CACHE
    df_1m = fetch_finnhub_candles(ticker_finnhub, resolution="1", count_candles=600)
    if df_1m.empty and ticker_yahoo:
        df_1m = fetch_yahoo_fallback(ticker_yahoo, interval="1m", range_period="5d")
    if df_1m.empty:
        return None, None, None, None, None, None

    df_3m = resample_candles(df_1m, "3min")
    df_5m = resample_candles(df_1m, "5min")
    df_15m = resample_candles(df_1m, "15min")

    current_time = time.time()
    cache_key = ticker_finnhub
    cached_data = MACRO_CACHE.get(cache_key)

    if cached_data and (current_time - cached_data["timestamp"] < CACHE_TTL):
        df_1h, df_daily = cached_data["1h"], cached_data["1d"]
    else:
        df_1h = fetch_finnhub_candles(ticker_finnhub, resolution="60", count_candles=200)
        if df_1h.empty and ticker_yahoo:
            df_1h = fetch_yahoo_fallback(ticker_yahoo, interval="1h", range_period="30d")
        df_daily = fetch_finnhub_candles(ticker_finnhub, resolution="D", count_candles=30)
        if df_daily.empty and ticker_yahoo:
            df_daily = fetch_yahoo_fallback(ticker_yahoo, interval="1d", range_period="60d")
        if not df_1h.empty and not df_daily.empty:
            MACRO_CACHE[cache_key] = {"timestamp": current_time, "1h": df_1h, "1d": df_daily}

    live_p = get_live_price(ticker_finnhub) or (get_live_price(ticker_yahoo) if ticker_yahoo else None)
    if live_p and not df_1m.empty:
        df_1m.iloc[-1, df_1m.columns.get_loc("close")] = live_p
        if not df_3m.empty: df_3m.iloc[-1, df_3m.columns.get_loc("close")] = live_p
        if not df_5m.empty: df_5m.iloc[-1, df_5m.columns.get_loc("close")] = live_p

    return df_daily, df_1h, df_15m, df_5m, df_3m, df_1m


def get_current_session_info():
    now_utc = get_utc_now()
    hour = now_utc.hour
    sessions = []
    session_code = 1
    if 0 <= hour < 8:
        sessions.append("Азія")
        session_code = 0
    if 7 <= hour < 16:
        sessions.append("Лондон")
        session_code = 1
    if 13 <= hour < 21:
        sessions.append("Нью-Йорк")
        session_code = 2
    if 13 <= hour < 16:
        sessions.append("🔥 Перетин Лондон/Нью-Йорк")
        session_code = 3
    return ", ".join(sessions) if sessions else "Тихоокеанська сесія", session_code, hour


def get_exit_price(ticker, name=""):
    price = get_live_price(ticker)
    if price and float(price) > 0:
        return float(price)
    df = fetch_finnhub_candles(ticker, resolution="1", count_candles=5)
    if df is not None and not df.empty:
        return float(df["close"].iloc[-1])
    yahoo_ticker = normalize_yahoo_ticker(ticker, name)
    if yahoo_ticker:
        price_yf = get_live_price(yahoo_ticker)
        if price_yf and float(price_yf) > 0:
            return float(price_yf)
        df_yf = fetch_yahoo_fallback(yahoo_ticker, interval="1m", range_period="1d")
        if df_yf is not None and not df_yf.empty:
            return float(df_yf["close"].iloc[-1])
    return None


def process_signal_expiration(sig_id):
    try:
        sig_data = database.get_signal_by_id(sig_id)
        if not sig_data or sig_data.get("status") != "PENDING":
            return

        ticker = sig_data["ticker"]
        entry_price = float(sig_data["entry_price"])
        signal_type = sig_data["signal_type"]
        expiration_mins = int(sig_data.get("expiration_mins", 5))

        created_at = parse_dt(sig_data.get("timestamp_str"))
        elapsed_seconds = (get_utc_now() - created_at).total_seconds()
        target_seconds = expiration_mins * 60

        if 0 <= elapsed_seconds < (target_seconds - 10):
            timer = threading.Timer(target_seconds - elapsed_seconds, process_signal_expiration, args=[sig_id])
            timer.daemon = True
            timer.start()
            return

        pair_name = next((k for k, v in PAIRS_MAP.items() if v == ticker), "")
        exit_price = get_exit_price(ticker, pair_name)
        if not exit_price:
            timer = threading.Timer(30, process_signal_expiration, args=[sig_id])
            timer.daemon = True
            timer.start()
            return

        if exit_price == entry_price:
            result = "NEUTRAL"
        elif signal_type == "CALL":
            result = "WIN" if exit_price > entry_price else "LOSS"
        else:
            result = "WIN" if exit_price < entry_price else "LOSS"

        multiplier = 1000 if "JPY" in ticker else 100000
        pips = int(round((exit_price - entry_price) * multiplier if signal_type == "CALL" else (entry_price - exit_price) * multiplier))

        database.update_signal_result(sig_id, result, exit_price, pips)
        if result in ["WIN", "LOSS"]:
            stats = database.get_stats()
            finished = stats.get('wins', 0) + stats.get('losses', 0)
            if finished >= 10 and finished % 5 == 0:
                threading.Thread(target=ml_filter.train_model, daemon=True).start()

        res_icon = "✅ WIN" if result == "WIN" else ("❌ LOSS" if result == "LOSS" else "➖ NEUTRAL")
        pips_str = f"+{pips}" if pips > 0 else f"{pips}"
        
        report_str = (
            f"\n──────────────────────────────────────\n"
            f"🏁 <b>РЕЗУЛЬТАТ УГОДИ: {res_icon} (<code>{pips_str}</code> pips)</b>\n"
            f"📍 Вхід: {entry_price:.5f} ➔ 🏁 Закриття: {exit_price:.5f}"
        )
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔍 Аналіз помилки (ШІ)", callback_data=f"review_{sig_id}")],
            [InlineKeyboardButton("📊 Статистика пари", callback_data=f"pair_stats_{pair_name}")]
        ])

        orig_txt = sig_data.get("message_text", "")
        if orig_txt and "РЕЗУЛЬТАТ УГОДИ" not in orig_txt:
            try:
                bot.edit_message_text(chat_id=sig_data["chat_id"], message_id=sig_data["message_id"], text=f"{orig_txt}\n{report_str}", parse_mode="HTML", reply_markup=keyboard)
            except Exception:
                try:
                    bot.send_message(
                        chat_id=sig_data["chat_id"], 
                        text=f"🏁 <b>Результат угоди #{sig_id}:</b>\n{res_icon} (<code>{pips_str}</code> п.)", 
                        parse_mode="HTML", 
                        reply_markup=keyboard
                    )
                except Exception:
                    pass
    except Exception as e:
        logger.exception(f"Помилка таймера {sig_id}: {e}")


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
                    if get_utc_now() >= parse_dt(row["timestamp_str"]) + timedelta(minutes=row["expiration_mins"]):
                        process_signal_expiration(row["id"])
            except Exception as e:
                logger.error(f"Помилка перевірки сигналів: {e}")
            time.sleep(20)
    threading.Thread(target=loop, daemon=True).start()


def start_auto_scanner_loop():
    def loop():
        while True:
            try:
                if AUTO_SCAN_CHATS:
                    for chat_id in list(AUTO_SCAN_CHATS):
                        with ThreadPoolExecutor(max_workers=8) as executor:
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
            time.sleep(90)
    threading.Thread(target=loop, daemon=True).start()


start_background_checker()
start_auto_scanner_loop()


def analyze_single_pair(chat_id, pair_name, ticker_finnhub, auto_mode=False, status_msg_id=None):
    try:
        news_is_bad, news_reason = is_high_impact_news_time()
        if news_is_bad:
            if not auto_mode:
                database.save_filtered_log(chat_id, f"❌ {pair_name}: Пауза через новини ({news_reason})")
            if status_msg_id:
                try:
                    bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text=f"⏸ <b><code>{pair_name}</code>: Пауза через новини</b>\n\n💡 <i>{news_reason}</i>", parse_mode="HTML")
                except Exception:
                    pass
            return False

        yahoo_ticker = normalize_yahoo_ticker(ticker_finnhub, pair_name)
        df_daily, df_1h, df_15m, df_5m, df_3m, df_1m = fetch_all_timeframes(ticker_finnhub, yahoo_ticker)

        def report_no_signal(reason_text):
            if not auto_mode:
                database.save_filtered_log(chat_id, f"❌ {pair_name}: {reason_text}")
            if status_msg_id:
                try:
                    bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text=f"⏸ <b><code>{pair_name}</code>: Сигнал відсутній</b>\n\n💡 <i>Причина: {clean_ai_html(reason_text)}</i>", parse_mode="HTML")
                except Exception:
                    pass

        if df_1m is None or df_1m.empty or len(df_1m) < 10:
            report_no_signal("Недостатньо даних котирувань.")
            return False

        analysis = analyzer.analyze_all_timeframes(df_daily, df_1h, df_15m, df_5m, df_3m, df_1m, pair_name=pair_name, strategies_status=strategies_status)
        
        strat_title = analysis.get("strategy_title", "")
        signal_type_engine = analysis.get("signal_type_engine", "trend_multi")

        is_valid, reason_val = validate_signal_conditions(analysis)
        if not is_valid:
            report_no_signal(reason_val)
            return False

        session_str, s_code, s_hour = get_current_session_info()
        win_probability = ml_filter.predict_proba(analysis)

        if ML_FILTER_ENABLED:
            min_ml = 53.0 if auto_mode else 44.0
            if win_probability < min_ml:
                report_no_signal(f"ML-модель оцінила ймовірність у {win_probability:.1f}%")
                return False

        is_pivot_ok, pivot_reason = check_pivot_level_proximity(analysis)
        if not is_pivot_ok:
            report_no_signal(f"Невдале розташування відносно рівнів: {pivot_reason}")
            return False

        exp_time = calculate_dynamic_expiration(analysis)
        regime = analysis.get("regime", "RANGE_BOUND")
        dyn_thresh = analysis.get("dyn_threshold", 2.3)

        payload = {
            "signal": analysis.get("signal"),
            "strategy": analysis.get("strategy", "QUANTUM_MATRIX"),
            "strategy_title": strat_title,
            "confluence_count": analysis.get("confluence_count", 1),
            "current_price": analysis.get("current_price"),
            "rsi": analysis.get("rsi"),
            "adx": analysis.get("adx"),
            "reason": analysis.get("reason"),
            "suggested_exp": exp_time,
        }

        ai_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
        if ai_key:
            c_macro = create_chart_image(df_1h, f"{pair_name} - 1H", tf_label="1H")
            c_mid = create_chart_image(df_15m, f"{pair_name} - 15M", tf_label="15M")
            c_micro = create_chart_image(df_1m, f"{pair_name} - 1M", tf_label="1M")
            
            recent_mistakes = database.get_recent_failed_reviews_summary(pair_name, limit=3)
            payload["recent_mistakes"] = recent_mistakes

            ai_res = ai_advisor_instance.evaluate_signal(pair_name, payload, c_macro, c_mid, c_micro)
        else:
            ai_res = {"decision": "YES", "confidence": 7, "suggested_expiration": exp_time, "reason": "ШІ вимкнено"}

        if ai_res.get("decision", "YES") != "YES" or ai_res.get("confidence", 6) < (6 if auto_mode else 5):
            report_no_signal(f"ШІ відхилив сигнал: {ai_res.get('reason', '')}")
            return False

        ai_exp = ai_res.get("suggested_expiration", exp_time)
        
        direction_icon = "🟢 CALL (ВХІД ВГОРУ)" if payload["signal"] == "CALL" else "🔴 PUT (ВХІД ВНИЗ)"
        risk_badge = "🔴 Новинний ризик" if news_is_bad else "🟢 Ринок стабільний"
        
        ml_badge = f"{make_progress_bar(win_probability)} {win_probability:.1f}%"
        ai_conf = ai_res.get('confidence', 6)
        ai_badge = f"{make_progress_bar(ai_conf * 10)} {ai_conf}/10"

        msg_text = (
            f"🎯 <b>[ {pair_name} ] ─── {direction_icon}</b>\n"
            f"──────────────────────────────────────\n"
            f"⏳ Експірація:  ⏱ <b>{ai_exp} хв</b> (Адаптивна)\n"
            f"📍 Ціна входу:  <b>{payload['current_price']:.5f}</b>\n"
            f"📊 ML-Прогноз:  <b>{ml_badge}</b>\n"
            f"🤖 Оцінка ШІ:   <b>{ai_badge}</b>\n"
            f"🌐 Режим ринку: ⚡ <b>{regime}</b>\n"
            f"──────────────────────────────────────\n"
            f"📋 Стратегія:   <b>{payload['strategy_title']}</b>\n"
            f"💡 Причина:      <i>{clean_ai_html(ai_res.get('reason', ''))}</i>\n"
            f"🌐 Сесія: {session_str} | Статус: {risk_badge}"
        )

        keyboard_markup = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("⚙️ Змінити актив", callback_data="change_pair"),
                InlineKeyboardButton("📊 Статистика пари", callback_data=f"pair_stats_{pair_name}")
            ]
        ])

        sent_msg_id = status_msg_id
        if status_msg_id:
            try:
                bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text=msg_text, parse_mode="HTML", reply_markup=keyboard_markup)
            except Exception:
                sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="HTML", reply_markup=keyboard_markup)
                sent_msg_id = sent_msg.message_id
        else:
            sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="HTML", reply_markup=keyboard_markup)
            sent_msg_id = sent_msg.message_id

        sig_id = database.save_signal(
            chat_id=chat_id, message_id=sent_msg_id, ticker=ticker_finnhub,
            signal_type=payload["signal"], entry_price=payload["current_price"],
            expiration_mins=ai_exp, message_text=msg_text,
            timestamp_str=get_utc_now().strftime("%Y-%m-%d %H:%M:%S"),
            rsi=payload.get("rsi"), adx=payload.get("adx"), strategy=payload.get("strategy")
        )
        schedule_signal_timer(sig_id, get_utc_now(), ai_exp)
        return True
    except Exception as e:
        logger.exception(f"Помилка аналізу {pair_name}: {e}")
        if status_msg_id:
            try:
                bot.edit_message_text(
                    chat_id=chat_id, 
                    message_id=status_msg_id, 
                    text=f"❌ <b>Помилка аналізу <code>{pair_name}</code></b>\n\n💡 <i>Деталі: {str(e)}</i>", 
                    parse_mode="HTML"
                )
            except Exception:
                pass
        return False


def run_mass_analysis(chat_id):
    bot.send_message(chat_id=chat_id, text="🔍 Запускаю квантовий масовий аналіз усіх пар...")
    database.clear_filtered_logs(chat_id)
    
    found = False
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(analyze_single_pair, chat_id, pair_name, ticker): pair_name
            for pair_name, ticker in PAIRS_MAP.items()
        }
        for f in futures:
            try:
                if f.result(timeout=45):
                    found = True
            except Exception as ex:
                logger.error(f"Помилка в масовому аналізі: {ex}")

    bot.send_message(chat_id=chat_id, text="✅ Масовий аналіз завершено!" if found else "✅ Аналіз завершено. Сигналів не знайдено.")


def handle_backtest_command(update, context):
    chat_id = update.effective_chat.id
    args = context.args  
    
    if args:
        pair = args[0].upper()
        if pair == "ALL":
            run_backtest_all_pairs(chat_id)
            return
        bot.send_message(chat_id=chat_id, text=f"⏳ Запускаю бектест для <code>{pair}</code> (30 днів)...", parse_mode="HTML")
        
        def run_bg():
            try:
                report = run_backtest(pair, "30d")
                bot.send_message(chat_id=chat_id, text=report, parse_mode="HTML")
            except Exception as e:
                bot.send_message(chat_id=chat_id, text=f"❌ Помилка бектесту: {e}")

        threading.Thread(target=run_bg, daemon=True).start()
    else:
        buttons = [InlineKeyboardButton(name, callback_data=f"bt_{name}") for name in PAIRS_MAP.keys()]
        markup_list = [buttons[i:i+3] for i in range(0, len(buttons), 3)]
        markup_list.insert(0, [InlineKeyboardButton("🌐 Бектест для УСІХ пар одразу", callback_data="bt_ALL")])
        markup_list.insert(1, [InlineKeyboardButton("🗑 Скинути / Очистити кеш бектесту", callback_data="bt_clear_cache")])
        markup = InlineKeyboardMarkup(markup_list)
        update.message.reply_text("📊 <b>Оберіть валютну пару для бектесту або керування кешем:</b>", parse_mode="HTML", reply_markup=markup)


def run_backtest_all_pairs(chat_id):
    bot.send_message(chat_id=chat_id, text="🌐 <b>Запускаю паралельний бектест по УСІХ парах та УСІХ стратегіях (30d)...</b>", parse_mode="HTML")
    
    def worker():
        try:
            reports = run_backtest_all_detailed("30d")
            for rep in reports:
                bot.send_message(chat_id=chat_id, text=rep, parse_mode="HTML")
        except Exception as e:
            bot.send_message(chat_id=chat_id, text=f"❌ Помилка бектесту: {e}")

    threading.Thread(target=worker, daemon=True).start()


def send_strategies_menu(update_obj, is_callback=False):
    global strategies_status
    buttons = []
    for key, title in GROUPED_STRATEGIES.items():
        active = strategies_status.get(key, True)
        status_icon = "🟢 [ВКЛ]" if active else "🔴 [ВИКЛ]"
        buttons.append([InlineKeyboardButton(f"{status_icon} {title}", callback_data=f"toggle_group_{key}")])
    
    buttons.append([InlineKeyboardButton("🎯 Персоналізація по парах", callback_data="custom_pairs_menu")])
    buttons.append([InlineKeyboardButton("🔙 Головне меню", callback_data="back_to_main_menu")])
    
    markup = InlineKeyboardMarkup(buttons)
    text_msg = (
        "⚙️ <b>Панель глобального керування стратегіями</b>:\n\n"
        "Натисніть на групу нижче, щоб увімкнути/вимкнути її глобально, або скористайтеся персоналізацією по парах:"
    )
    
    if is_callback:
        try:
            update_obj.edit_message_text(text_msg, parse_mode="HTML", reply_markup=markup)
        except Exception:
            pass
    else:
        update_obj.message.reply_text(text_msg, parse_mode="HTML", reply_markup=markup)


def start(update, context):
    user = update.effective_user
    database.register_user(user.id, user.username)
    ml_txt = "🤖 ML-Модель: ВКЛ" if ML_FILTER_ENABLED else "🤖 ML-Модель: ВИКЛ"
    
    keyboard = [
        [KeyboardButton("⚡ Масовий сканер"), KeyboardButton("🔔 Авто-сканер [ВКЛ]" if update.effective_chat.id in AUTO_SCAN_CHATS else "🔔 Авто-сканер [ВИКЛ]"), KeyboardButton("💵 Вибір пари")],
        [KeyboardButton("📈 Статистика"), KeyboardButton("📊 Бектест (30d)"), KeyboardButton("⚙️ Конфігуратор")],
        [KeyboardButton("📋 Системні логи"), KeyboardButton(ml_txt)]
    ]
    update.message.reply_text(
        "🤖 <b>Квантовий бот Racio_1bot успішно активовано!</b>\n\n"
        "Інтерфейс оптимізовано, 4 комплекси стратегій та динамічні пороги працюють у реальному часі.",
        parse_mode="HTML",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )


def handle_message(update, context):
    text, chat_id = update.message.text, update.effective_chat.id
    if text == "⚡ Масовий сканер" or text == "📊 Аналіз усіх пар":
        threading.Thread(target=run_mass_analysis, args=[chat_id], daemon=True).start()
    elif text == "📊 Бектест (30d)" or text == "📊 Бектест":
        handle_backtest_command(update, context)
    elif text == "⚙️ Конфігуратор" or text == "⚙️ Стратегії":
        send_strategies_menu(update, is_callback=False)
    elif "🔔 Авто-сканер" in text:
        if chat_id in AUTO_SCAN_CHATS:
            AUTO_SCAN_CHATS.remove(chat_id)
            update.message.reply_text("🔴 Авто-сканер ВИМКНЕНО.")
        else:
            AUTO_SCAN_CHATS.add(chat_id)
            update.message.reply_text("🟢 Авто-сканер УВІМКНЕНО (кожні 1.5 хв).")
    elif text in ["🤖 ML-Модель: ВКЛ", "🤖 ML-Модель: ВИКЛ", "🟢 ML-фільтр: ВКЛ", "🔴 ML-фільтр: ВИКЛ"]:
        global ML_FILTER_ENABLED
        ML_FILTER_ENABLED = not ML_FILTER_ENABLED
        ml_txt = "🤖 ML-Модель: ВКЛ" if ML_FILTER_ENABLED else "🤖 ML-Модель: ВИКЛ"
        keyboard = [
            [KeyboardButton("⚡ Масовий сканер"), KeyboardButton("🔔 Авто-сканер"), KeyboardButton("💵 Вибір пари")],
            [KeyboardButton("📈 Статистика"), KeyboardButton("📊 Бектест (30d)"), KeyboardButton("⚙️ Конфігуратор")],
            [KeyboardButton("📋 Системні логи"), KeyboardButton(ml_txt)]
        ]
        update.message.reply_text(f"⚙️ ML-фільтр {'увімкнено' if ML_FILTER_ENABLED else 'вимкнено'}.", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))
    elif text == "💵 Вибір пари" or text == "💵 Пари":
        buttons = [InlineKeyboardButton(name, callback_data=f"analyze_{name}") for name in PAIRS_MAP.keys()]
        update.message.reply_text("💵 <b>Оберіть валютну пару для аналізу:</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup([buttons[i:i+3] for i in range(0, len(buttons), 3)]))
    elif text == "📈 Статистика":
        stats = database.get_stats()
        finished = stats.get('wins', 0) + stats.get('losses', 0)
        winrate = (stats.get('wins', 0) / finished * 100) if finished > 0 else 0.0
        
        msg = (
            "📊 <b>ДАШБОРД ЕФЕКТИВНОСТІ МАТРИЦІ</b>\n"
            "──────────────────────────────────────\n"
            f"🏆 Загальний вінрейт: <b>{winrate:.1f}%</b>  [{stats.get('wins', 0)} WIN / {stats.get('losses', 0)} LOSS]\n"
            f"🎯 Угод у базі даних: <b>{stats.get('total', 0)} угод</b>\n"
            f"⚡ Активна ML-модель: <b>XGBoost v2.4 (Активна)</b>\n"
            "──────────────────────────────────────\n"
            "Оберіть зріз аналітики для деталізації:"
        )
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("💵 По парах", callback_data="dash_pairs"), InlineKeyboardButton("📋 По стратегіях", callback_data="dash_strategies")],
            [InlineKeyboardButton("🔀 Пара ✖ Стратегія", callback_data="dash_matrix"), InlineKeyboardButton("📊 По годинах/днях", callback_data="strategy_time_stats")],
            [InlineKeyboardButton("🤖 Перенавчити ШІ", callback_data="retrain_ml"), InlineKeyboardButton("🗑 Очистити", callback_data="clear_stats_confirm")]
        ])
        update.message.reply_text(msg, parse_mode="HTML", reply_markup=keyboard)
    elif text == "📋 Системні логи" or text == "📋 Логи фільтру":
        logs = database.get_system_logs(chat_id)
        update.message.reply_text("📋 <b>Останні логи системи:</b>\n\n" + "\n".join([clean_ai_html(l) for l in logs]) if logs else "📋 Логи порожні.", parse_mode="HTML")


def handle_callback(update, context):
    query = update.callback_query
    query.answer()
    chat_id, data = query.message.chat_id, query.data
    
    if data.startswith("analyze_"):
        pair = data.replace("analyze_", "")
        if PAIRS_MAP.get(pair):
            msg = query.edit_message_text(f"🔍 Аналізую {pair}...")
            threading.Thread(target=analyze_single_pair, args=[chat_id, pair, PAIRS_MAP[pair], False, msg.message_id], daemon=True).start()
    elif data == "change_pair":
        buttons = [InlineKeyboardButton(name, callback_data=f"analyze_{name}") for name in PAIRS_MAP.keys()]
        query.edit_message_text("💵 <b>Оберіть новий актив для аналізу:</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup([buttons[i:i+3] for i in range(0, len(buttons), 3)]))
    elif data.startswith("pair_stats_"):
        pair = data.replace("pair_stats_", "")
        rows = database.get_detailed_pair_strategy_stats(ticker_filter=next((v for k, v in PAIRS_MAP.items() if k == pair), pair))
        lines = [f"📊 <b>СТАТИСТИКА АКТИВУ [ {pair} ]:</b>\n──────────────────────────────────────"]
        for r in rows:
            tot = r['total']
            wins = r['wins']
            wr = (wins / tot * 100) if tot > 0 else 0
            ico = "🟢" if wr >= 55 else ("🟡" if wr >= 48 else "🔴")
            lines.append(f"{ico} <b>{r['strategy']}</b>: {wins}/{tot} WIN ({wr:.1f}% WR)")
        if len(rows) == 0:
            lines.append("⚠️ Недостатньо закритих угод для цього активу.")
        query.edit_message_text("\n".join(lines), parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад до дашборду", callback_data="back_to_stats")]]))
    elif data == "dash_matrix":
        rows = database.get_detailed_pair_strategy_stats()
        lines = ["🔀 <b>МАТРИЦЯ: ПАРА ✖ СТРАТЕГІЯ</b>\n──────────────────────────────────────"]
        for r in rows[:15]:
            tot = r['total']
            wins = r['wins']
            wr = (wins / tot * 100) if tot > 0 else 0
            p_name = next((k for k, v in PAIRS_MAP.items() if v == r['ticker']), r['ticker'])
            ico = "🟢" if wr >= 55 else ("🟡" if wr >= 48 else "🔴")
            lines.append(f"{ico} <code>{p_name}</code> | {r['strategy']}: <b>{wr:.1f}%</b> ({wins}/{tot})")
        if not rows:
            lines.append("⚠️ База даних угод порожня.")
        query.edit_message_text("\n".join(lines), parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад до дашборду", callback_data="back_to_stats")]]))
    elif data == "custom_pairs_menu":
        buttons = [InlineKeyboardButton(name, callback_data=f"cpair_{name}") for name in PAIRS_MAP.keys()]
        markup = InlineKeyboardMarkup([buttons[i:i+3] for i in range(0, len(buttons), 3)] + [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_strategies")]])
        query.edit_message_text("🎯 <b>Оберіть актив для персоналізації стратегій:</b>", parse_mode="HTML", reply_markup=markup)
    elif data.startswith("cpair_"):
        pair_name = data.replace("cpair_", "")
        buttons = []
        for s_key, s_title in GROUPED_STRATEGIES.items():
            is_act = database.get_pair_strategy_status(pair_name, s_key)
            status_ico = "🟢 [ВКЛ]" if is_act else "🔴 [ВИКЛ]"
            buttons.append([InlineKeyboardButton(f"{status_ico} {s_title}", callback_data=f"toggle_cpair_{pair_name}_{s_key}")])
        buttons.append([InlineKeyboardButton("🔙 До списку пар", callback_data="custom_pairs_menu")])
        query.edit_message_text(f"⚙️ <b>Стратегії для активу [ {pair_name} ]:</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))
    elif data.startswith("toggle_cpair_"):
        parts = data.replace("toggle_cpair_", "").split("_", 1)
        pair_name, s_key = parts[0], parts[1]
        current_st = database.get_pair_strategy_status(pair_name, s_key)
        database.set_pair_strategy_status(pair_name, s_key, not current_st)
        
        buttons = []
        for sk, st in GROUPED_STRATEGIES.items():
            is_act = database.get_pair_strategy_status(pair_name, sk)
            status_ico = "🟢 [ВКЛ]" if is_act else "🔴 [ВИКЛ]"
            buttons.append([InlineKeyboardButton(f"{status_ico} {st}", callback_data=f"toggle_cpair_{pair_name}_{sk}")])
        buttons.append([InlineKeyboardButton("🔙 До списку пар", callback_data="custom_pairs_menu")])
        try:
            query.edit_message_text(f"⚙️ <b>Стратегії для активу [ {pair_name} ]:</b>", parse_mode="HTML", reply_markup=InlineKeyboardMarkup(buttons))
        except Exception:
            pass
    elif data == "back_to_strategies":
        send_strategies_menu(query, is_callback=True)
    elif data == "bt_ALL":
        query.edit_message_text("🌐 Запускаю глобальний бектест для усіх пар...", parse_mode="HTML")
        run_backtest_all_pairs(chat_id)
    elif data == "bt_clear_cache":
        success, msg_info = clear_backtest_cache()
        query.edit_message_text(f"{'✅' if success else '❌'} {msg_info}\n\nТепер бектест працюватиме з новими параметрами стратегій з чистого аркуша.")
    elif data.startswith("bt_"):
        pair = data.replace("bt_", "")
        query.edit_message_text(f"⏳ Запускаю детальний бектест для <code>{pair}</code> (30 днів)...", parse_mode="HTML")
        def run_bt():
            try:
                report = run_backtest(pair, "30d")
                bot.send_message(chat_id=chat_id, text=report, parse_mode="HTML")
            except Exception as e:
                bot.send_message(chat_id=chat_id, text=f"❌ Помилка бектесту: {e}")
        threading.Thread(target=run_bt, daemon=True).start()
    elif data == "strategy_time_stats":
        query.edit_message_text("⏳ Формую аналіз за годинами та днями... ⏳")
        try:
            time_stats = database.get_strategy_time_stats()
            hours_data = time_stats.get("hours", [])
            dow_data = time_stats.get("dow", [])

            if not hours_data and not dow_data:
                query.edit_message_text(
                    "⚠️ Недостатньо закритих угод для часового аналізу.",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_stats")]]))
                return

            dow_names = {0: "Неділя", 1: "Понеділок", 2: "Вівторок", 3: "Середа", 4: "Четвер", 5: "П'ятниця", 6: "Субота"}
            report_lines = ["📈 <b>Аналіз ефективності за часом:</b>\n"]

            if hours_data:
                report_lines.append("🕒 <b>По годинах (UTC):</b>")
                hour_grouped = {}
                for row in hours_data:
                    h = int(row['trade_hour'])
                    if h not in hour_grouped:
                        hour_grouped[h] = {"wins": 0, "total": 0}
                    hour_grouped[h]["wins"] += int(row['wins'])
                    hour_grouped[h]["total"] += int(row['total'])

                sorted_hours = sorted(hour_grouped.items(), key=lambda x: (x[1]["wins"]/x[1]["total"] if x[1]["total"]>0 else 0), reverse=True)
                for h, d in sorted_hours[:10]:
                    tot = d["total"]
                    wins = d["wins"]
                    wr = (wins / tot * 100) if tot > 0 else 0
                    ico = "🟢" if wr >= 53 else ("🟡" if wr >= 48 else "🔴")
                    report_lines.append(f"{ico} <b>{h:02d}:00 - {h+1:02d}:00</b>: {wins}/{tot} ({wr:.1f}% WR)")

            if dow_data:
                report_lines.append("\n📅 <b>По днях тижня:</b>")
                dow_grouped = {}
                for row in dow_data:
                    dw = int(row['trade_dow'])
                    if dw not in dow_grouped:
                        dow_grouped[dw] = {"wins": 0, "total": 0}
                    dow_grouped[dw]["wins"] += int(row['wins'])
                    dow_grouped[dw]["total"] += int(row['total'])

                for dw in range(7):
                    if dw in dow_grouped:
                        d = dow_grouped[dw]
                        tot = d["total"]
                        wins = d["wins"]
                        wr = (wins / tot * 100) if tot > 0 else 0
                        ico = "🟢" if wr >= 53 else ("🟡" if wr >= 48 else "🔴")
                        d_name = dow_names.get(dw, "День")
                        report_lines.append(f"{ico} <b>{d_name}</b>: {wins}/{tot} ({wr:.1f}% WR)")

            full_report = "\n".join(report_lines)
            query.edit_message_text(full_report, parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_stats")]]))
        except Exception as e:
            query.edit_message_text(f"❌ Помилка: {e}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_stats")]]))
    elif data == "back_to_stats":
        stats = database.get_stats()
        finished = stats.get('wins', 0) + stats.get('losses', 0)
        winrate = (stats.get('wins', 0) / finished * 100) if finished > 0 else 0.0
        msg = (
            "📊 <b>ДАШБОРД ЕФЕКТИВНОСТІ МАТРИЦІ</b>\n"
            "──────────────────────────────────────\n"
            f"🏆 Загальний вінрейт: <b>{winrate:.1f}%</b>  [{stats.get('wins', 0)} WIN / {stats.get('losses', 0)} LOSS]\n"
            f"🎯 Угод у базі даних: <b>{stats.get('total', 0)} угод</b>\n"
            f"⚡ Активна ML-модель: <b>XGBoost v2.4 (Активна)</b>\n"
            "──────────────────────────────────────\n"
            "Оберіть зріз аналітики для деталізації:"
        )
        query.edit_message_text(
            msg, 
            parse_mode="HTML", 
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💵 По парах", callback_data="dash_pairs"), InlineKeyboardButton("📋 По стратегіях", callback_data="dash_strategies")],
                [InlineKeyboardButton("🔀 Пара ✖ Стратегія", callback_data="dash_matrix"), InlineKeyboardButton("📊 По годинах/днях", callback_data="strategy_time_stats")],
                [InlineKeyboardButton("🤖 Перенавчити ШІ", callback_data="retrain_ml"), InlineKeyboardButton("🗑 Очистити", callback_data="clear_stats_confirm")]
            ])
        )
    elif data == "back_to_main_menu":
        try:
            query.message.delete()
        except Exception:
            pass
        query.message.reply_text("🏠 Головне меню активне.", reply_markup=query.message.reply_markup)
    elif data.startswith("toggle_group_"):
        group_key = data.replace("toggle_group_", "")
        if group_key in strategies_status:
            strategies_status[group_key] = not strategies_status[group_key]
            save_strategies_config(strategies_status)
        send_strategies_menu(query, is_callback=True)
    elif data.startswith("review_"):
        sig_id = int(data.replace("review_", ""))
        query.edit_message_text("🔍 Формую ретроспективний аналіз... ⏳")
        def review():
            sig = database.get_signal_by_id(sig_id)
            rev = ai_advisor_instance.evaluate_closed_trade(sig) if sig else "Не знайдено"
            database.save_signal_ai_review(sig_id, rev)
            bot.edit_message_text(chat_id=chat_id, message_id=query.message.message_id, text=f"{sig.get('message_text', '')}\n\n🤖 <b>Розбір ШІ:</b>\n{clean_ai_html(rev)}", parse_mode="HTML")
        threading.Thread(target=review, daemon=True).start()
    elif data == "clear_stats_confirm":
        query.edit_message_text("Видалити всю статистику?", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⚠️ ТАК", callback_data="clear_stats_do")], [InlineKeyboardButton("❌ Скасувати", callback_data="cancel_action")]]))
    elif data == "clear_stats_do":
        database.clear_all_stats()
        query.edit_message_text("✅ Статистику очищено!")
    elif data == "cancel_action":
        query.edit_message_text("Дію скасовано.")
    elif data == "retrain_ml":
        query.edit_message_text("🤖 Перенавчання ШІ... ⏳")
        def train():
            success, info = ml_filter.train_model()
            bot.send_message(chat_id=chat_id, text=f"{'✅' if success else '❌'} {info}")
        threading.Thread(target=train, daemon=True).start()


if __name__ == "__main__":
    bot.delete_webhook(drop_pending_updates=True)
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    updater.dispatcher.add_handler(CommandHandler("start", start))
    updater.dispatcher.add_handler(CommandHandler("backtest", handle_backtest_command))
    updater.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    updater.dispatcher.add_handler(CallbackQueryHandler(handle_callback))
    logger.info("🚀 Квантовий бот Racio_1bot із повним оновленням UI та логіки запущено!")
    updater.start_polling()
    updater.idle()
