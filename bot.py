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
    calculate_dynamic_expiration,
    check_pivot_level_proximity,
    validate_signal_conditions,
)
from calendar_filter import is_high_impact_news_time
from backtest import run_backtest, clean_strategy_name, PERFORMANCE_CACHE_FILE

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

# --- 7 Головних логічних груп стратегій для керування через меню ---
GROUPED_STRATEGIES = {
    "trend_multi": "📊 Мульти-Трендові Системи (1H/15M/3M)",
    "flat_corridor": "📉 Флетові Коридори та Відбої (BB)",
    "oscillators": "⚡ Осцилятори Екстремумів (RSI/Stoch)",
    "moving_averages": "📈 Ковзні Середні та Імпульси (EMA/MACD)",
    "market_levels": "📐 Рівні та Геометрія Ринку (Pivot)",
    "price_action": "🕯️ Price Action та Патерни",
    "volume_volatility": "💥 Об'єми та Волатильність (ATR/Vol)"
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

        multiplier = 1000 if "JPY" in ticker else 100000
        raw_pips = (exit_price - entry_price) * multiplier if signal_type == "CALL" else (entry_price - exit_price) * multiplier
        pips = int(round(raw_pips))
        result = "WIN" if pips > 0 else ("LOSS" if pips < 0 else "NEUTRAL")

        database.update_signal_result(sig_id, result, exit_price, pips)
        if result in ["WIN", "LOSS"]:
            stats = database.get_stats()
            finished = stats.get('wins', 0) + stats.get('losses', 0)
            if finished >= 10 and finished % 5 == 0:
                threading.Thread(target=ml_filter.train_model, daemon=True).start()

        res_icon = "✅ WIN" if result == "WIN" else ("❌ LOSS" if result == "LOSS" else "➖ NEUTRAL")
        pips_str = f"+{pips}" if pips > 0 else f"{pips}"
        report_str = f"\n----------------------------------\n🏁 <b>Результат:</b> {res_icon} (<code>{pips_str}</code> п.)\n📍 Вхід: {entry_price:.5f} ➔ 🏁 Закриття: {exit_price:.5f}"
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🔍 Аналіз помилки (ШІ)", callback_data=f"review_{sig_id}")]])

        orig_txt = sig_data.get("message_text", "")
        if orig_txt and "🏁 Результат" not in orig_txt:
            try:
                bot.edit_message_text(chat_id=sig_data["chat_id"], message_id=sig_data["message_id"], text=f"{orig_txt}\n{report_str}", parse_mode="HTML", reply_markup=keyboard)
            except Exception:
                bot.send_message(chat_id=sig_data["chat_id"], text=f"🏁 <b>Результат угоди #{sig_id}:</b>\n{res_icon} (<code>{pips_str}</code> п.)", parse_mode="HTML", reply_to_message_id=sig_data["message_id"], reply_markup=keyboard)
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
                        with ThreadPoolExecutor(max_workers=5) as executor:
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
            time.sleep(180)
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

        analysis = analyzer.analyze_all_timeframes(df_daily, df_1h, df_15m, df_5m, df_3m, df_1m)
        
        strat_title = analysis.get("strategy_title", "")
        clean_name = clean_strategy_name(strat_title)

        signal_type_engine = analysis.get("signal_type_engine", "trend_multi")
        if signal_type_engine in strategies_status and not strategies_status[signal_type_engine]:
            report_no_signal(f"Група стратегій вимкнена в меню.")
            return False

        if os.path.exists(PERFORMANCE_CACHE_FILE):
            try:
                with open(PERFORMANCE_CACHE_FILE, "r", encoding="utf-8") as f:
                    perf_cache = json.load(f)
                    if pair_name in perf_cache and clean_name in perf_cache[pair_name]:
                        strat_perf = perf_cache[pair_name][clean_name]
                        if strat_perf["total"] >= 3 and strat_perf["wr"] < 50.0:
                            report_no_signal(f"Авто-фільтр бектесту: слабкий WR ({strat_perf['wr']}%) на {pair_name}.")
                            return False
            except Exception:
                pass

        is_valid, reason_val = validate_signal_conditions(analysis)
        if not is_valid:
            report_no_signal(reason_val)
            return False

        session_str, s_code, s_hour = get_current_session_info()
        win_probability = ml_filter.predict_proba(analysis)

        if ML_FILTER_ENABLED:
            min_ml = 58.0 if auto_mode else 45.0
            if win_probability < min_ml:
                report_no_signal(f"ML-модель оцінила ймовірність у {win_probability:.1f}%")
                return False

        is_pivot_ok, pivot_reason = check_pivot_level_proximity(analysis)
        if not is_pivot_ok:
            report_no_signal(f"Невдале розташування відносно рівнів: {pivot_reason}")
            return False

        exp_time = calculate_dynamic_expiration(analysis)
        payload = {
            "signal": analysis.get("signal"),
            "strategy": analysis.get("strategy", "MULTI_TF_DUAL_ENGINE"),
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
            ai_res = ai_advisor_instance.evaluate_signal(pair_name, payload, c_macro, c_mid, c_micro)
        else:
            ai_res = {"decision": "YES", "confidence": 7, "suggested_expiration": exp_time, "reason": "ШІ вимкнено"}

        if ai_res.get("decision", "YES") != "YES" or ai_res.get("confidence", 6) < (6 if auto_mode else 5):
            report_no_signal(f"ШІ відхилив сигнал: {ai_res.get('reason', '')}")
            return False

        ai_exp = ai_res.get("suggested_expiration", exp_time)
        direction_icon = "🟢 CALL (ВХІД ВГОРУ)" if payload["signal"] == "CALL" else "🔴 PUT (ВХІД ВНИЗ)"
        
        msg_text = (
            f"🎯 <b>АКТИВ: <code>{pair_name}</code></b>\n\n"
            f"📊 Напрямок: <b>{direction_icon}</b>\n"
            f"⏳ Експірація: <b>{ai_exp} хв</b>\n"
            f"📍 Ціна: {payload['current_price']:.5f}\n\n"
            f"📋 Стратегія: <b>{payload['strategy_title']}</b>\n"
            f"🧠 ML: <b>{make_progress_bar(win_probability)} {win_probability:.1f}%</b>\n"
            f"🤖 ШІ: <b>{make_progress_bar(ai_res.get('confidence', 6)*10)} {ai_res.get('confidence', 6)}/10</b>\n"
            f"🔥 Збігів: <b>{payload['confluence_count']} балів</b>\n\n"
            f"💡 Причина: <i>{clean_ai_html(ai_res.get('reason', ''))}</i>\n"
            f"🌐 Сесія: {session_str}"
        )

        sent_msg_id = status_msg_id
        if status_msg_id:
            try:
                bot.edit_message_text(chat_id=chat_id, message_id=status_msg_id, text=msg_text, parse_mode="HTML")
            except Exception:
                sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="HTML")
                sent_msg_id = sent_msg.message_id
        else:
            sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="HTML")
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
    bot.send_message(chat_id=chat_id, text="🔍 Запускаю паралельний масовий аналіз усіх пар...")
    database.clear_filtered_logs(chat_id)
    
    found = False
    with ThreadPoolExecutor(max_workers=5) as executor:
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
        bot.send_message(chat_id=chat_id, text=f"⏳ Запускаю детальний бектест для <code>{pair}</code> (30 днів)...", parse_mode="HTML")
        
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
        markup = InlineKeyboardMarkup(markup_list)
        update.message.reply_text("📊 <b>Оберіть валютну пару для бектесту:</b>", parse_mode="HTML", reply_markup=markup)


def run_backtest_all_pairs(chat_id):
    bot.send_message(chat_id=chat_id, text="🌐 <b>Запускаю паралельний бектест для УСІХ пар (30 днів)...</b>", parse_mode="HTML")
    
    def worker():
        summary_lines = ["🌐 <b>Глобальний звіт бектесту по всіх парах (30d):</b>\n"]
        pair_results = []
        
        for p_name in PAIRS_MAP.keys():
            try:
                yahoo_ticker = YAHOO_PAIRS_MAP.get(p_name, "EURUSD=X")
                df = yf.download(tickers=yahoo_ticker, period="30d", interval="1h", progress=False, auto_adjust=True)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                    df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                    
                    tot_s, tot_w = 0, 0
                    mult = 1000 if "JPY" in yahoo_ticker else 100000
                    for i in range(30, len(df) - 3):
                        s_df = df.iloc[:i]
                        an = analyzer.analyze_all_timeframes(s_df, s_df, s_df, s_df, s_df, s_df)
                        if an.get("signal") in ["CALL", "PUT"]:
                            ep = float(s_df["close"].iloc[-1])
                            exp_p = float(df["close"].iloc[i + 3])
                            pips = int(round((exp_p - ep) * mult if an.get("signal") == "CALL" else (ep - exp_p) * mult))
                            if pips != 0:
                                tot_s += 1
                                if pips > 0: tot_w += 1
                    wr = (tot_w / tot_s * 100) if tot_s > 0 else 0.0
                    pair_results.append((p_name, wr, tot_s))
            except Exception:
                continue
                
        pair_results.sort(key=lambda x: x[1], reverse=True)
        for p_n, wr_val, sigs in pair_results:
            ico = "🟢" if wr_val >= 53 else ("🟡" if wr_val >= 48 else "🔴")
            summary_lines.append(f"{ico} <b>{p_n}</b>: <b>{wr_val:.1f}% WR</b> (Сигналів: {sigs})")
            
        summary_lines.append("\n💡 <i>Кеш стратегій оновлено! Авто-фільтр активовано.</i>")
        bot.send_message(chat_id=chat_id, text="\n".join(summary_lines), parse_mode="HTML")

    threading.Thread(target=worker, daemon=True).start()


def send_strategies_menu(update_obj, is_callback=False):
    global strategies_status
    buttons = []
    for key, title in GROUPED_STRATEGIES.items():
        active = strategies_status.get(key, True)
        status_icon = "🟢" if active else "🔴"
        buttons.append([InlineKeyboardButton(f"{status_icon} {title}", callback_data=f"toggle_group_{key}")])
    
    markup = InlineKeyboardMarkup(buttons)
    text_msg = "⚙️ <b>Керування групами стратегій</b>:\nНатисніть на групу, щоб увімкнути/вимкнути її."
    
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
    ml_txt = "🟢 ML-фільтр: ВКЛ" if ML_FILTER_ENABLED else "🔴 ML-фільтр: ВИКЛ"
    keyboard = [
        [KeyboardButton("📊 Аналіз усіх пар"), KeyboardButton("📊 Бектест")],
        [KeyboardButton("⚙️ Стратегії"), KeyboardButton("🔔 Авто-сканер")],
        [KeyboardButton("💵 Пари"), KeyboardButton("📈 Статистика")],
        [KeyboardButton("📋 Логи фільтру"), KeyboardButton(ml_txt)]
    ]
    update.message.reply_text("🤖 Бот із Двоядерною матрицею та груповим керуванням готовий!", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))


def handle_message(update, context):
    text, chat_id = update.message.text, update.effective_chat.id
    if text == "📊 Аналіз усіх пар":
        threading.Thread(target=run_mass_analysis, args=[chat_id], daemon=True).start()
    elif text == "📊 Бектест":
        handle_backtest_command(update, context)
    elif text == "⚙️ Стратегії":
        send_strategies_menu(update, is_callback=False)
    elif text == "🔔 Авто-сканер":
        if chat_id in AUTO_SCAN_CHATS:
            AUTO_SCAN_CHATS.remove(chat_id)
            update.message.reply_text("🔴 Авто-сканер ВИМКНЕНО.")
        else:
            AUTO_SCAN_CHATS.add(chat_id)
            update.message.reply_text("🟢 Авто-сканер УВІМКНЕНО (кожні 3 хв).")
    elif text in ["🟢 ML-фільтр: ВКЛ", "🔴 ML-фільтр: ВИКЛ"]:
        global ML_FILTER_ENABLED
        ML_FILTER_ENABLED = not ML_FILTER_ENABLED
        ml_txt = "🟢 ML-фільтр: ВКЛ" if ML_FILTER_ENABLED else "🔴 ML-фільтр: ВИКЛ"
        keyboard = [
            [KeyboardButton("📊 Аналіз усіх пар"), KeyboardButton("📊 Бектест")],
            [KeyboardButton("⚙️ Стратегії"), KeyboardButton("🔔 Авто-сканер")],
            [KeyboardButton("💵 Пари"), KeyboardButton("📈 Статистика")],
            [KeyboardButton("📋 Логи фільтру"), KeyboardButton(ml_txt)]
        ]
        update.message.reply_text(f"⚙️ ML-фільтр {'увімкнено' if ML_FILTER_ENABLED else 'вимкнено'}.", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))
    elif text == "💵 Пари":
        buttons = [InlineKeyboardButton(name, callback_data=f"analyze_{name}") for name in PAIRS_MAP.keys()]
        update.message.reply_text("Оберіть пару:", reply_markup=InlineKeyboardMarkup([buttons[i:i+3] for i in range(0, len(buttons), 3)]))
    elif text == "📈 Статистика":
        stats = database.get_stats()
        finished = stats.get('wins', 0) + stats.get('losses', 0)
        winrate = (stats.get('wins', 0) / finished * 100) if finished > 0 else 0.0
        msg = f"📊 <b>Статистика матриці:</b>\n\nВсього: {stats.get('total', 0)}\n✅ WIN: {stats.get('wins', 0)}\n❌ LOSS: {stats.get('losses', 0)}\n🏆 Вінрейт: {winrate:.1f}%"
        update.message.reply_text(
            msg, 
            parse_mode="HTML", 
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Аналіз по годинах/днях", callback_data="strategy_time_stats")],
                [InlineKeyboardButton("🤖 Перенавчити ШІ", callback_data="retrain_ml")], 
                [InlineKeyboardButton("🗑 Очистити", callback_data="clear_stats_confirm")]
            ])
        )
    elif text == "📋 Логи фільтру":
        logs = database.get_system_logs(chat_id)
        update.message.reply_text("📋 <b>Останні логи:</b>\n\n" + "\n".join([clean_ai_html(l) for l in logs]) if logs else "📋 Логи порожні.", parse_mode="HTML")


def handle_callback(update, context):
    query = update.callback_query
    query.answer()
    chat_id, data = query.message.chat_id, query.data
    
    if data.startswith("analyze_"):
        pair = data.replace("analyze_", "")
        if PAIRS_MAP.get(pair):
            msg = query.edit_message_text(f"🔍 Аналізую {pair}...")
            threading.Thread(target=analyze_single_pair, args=[chat_id, pair, PAIRS_MAP[pair], False, msg.message_id], daemon=True).start()
    elif data == "bt_ALL":
        query.edit_message_text("🌐 Запускаю глобальний бектест для усіх пар...", parse_mode="HTML")
        run_backtest_all_pairs(chat_id)
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
        query.edit_message_text("⏳ Формую детальний аналіз вінрейту за годинами та днями... ⏳")
        try:
            time_stats = database.get_strategy_time_stats()
            hours_data = time_stats.get("hours", [])
            dow_data = time_stats.get("dow", [])

            if not hours_data and not dow_data:
                query.edit_message_text(
                    "⚠️ Недостатньо закритих угод у базі даних для формування часового аналізу.",
                    reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_stats")]]))
                return

            dow_names = {0: "Неділя", 1: "Понеділок", 2: "Вівторок", 3: "Середа", 4: "Четвер", 5: "П'ятниця", 6: "Субота"}
            
            report_lines = ["📈 <b>Аналіз ефективності за часом та днями:</b>\n"]

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
            if len(full_report) > 4000:
                full_report = full_report[:3950] + "\n<i>...звіт скорочено через ліміт довжини.</i>"

            query.edit_message_text(full_report, parse_mode="HTML", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад до статистики", callback_data="back_to_stats")]]))
        except Exception as e:
            query.edit_message_text(f"❌ Помилка генерації аналітики: {e}", reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Назад", callback_data="back_to_stats")]]))
    elif data == "back_to_stats":
        stats = database.get_stats()
        finished = stats.get('wins', 0) + stats.get('losses', 0)
        winrate = (stats.get('wins', 0) / finished * 100) if finished > 0 else 0.0
        msg = f"📊 <b>Статистика матриці:</b>\n\nВсього: {stats.get('total', 0)}\n✅ WIN: {stats.get('wins', 0)}\n❌ LOSS: {stats.get('losses', 0)}\n🏆 Вінрейт: {winrate:.1f}%"
        query.edit_message_text(
            msg, 
            parse_mode="HTML", 
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Аналіз по годинах/днях", callback_data="strategy_time_stats")],
                [InlineKeyboardButton("🤖 Перенавчити ШІ", callback_data="retrain_ml")], 
                [InlineKeyboardButton("🗑 Очистити", callback_data="clear_stats_confirm")]
            ])
        )
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
    logger.info("🚀 Бот запущено з груповим керуванням стратегіями!")
    updater.start_polling()
    updater.idle()
