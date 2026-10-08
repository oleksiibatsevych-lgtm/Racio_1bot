import html
import io
import logging
import os
import threading
import time
from datetime import datetime, timedelta

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

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.error("❌ TELEGRAM_TOKEN не знайдено!")

bot = Bot(token=TELEGRAM_TOKEN)

analyzer = AdaptiveTechnicalAnalysis()
ml_filter = TradingMLFilter()

last_sent_signals = {}
MACRO_CACHE = {}
CACHE_TTL = 900

database.init_db()
start_finnhub_ws()


def parse_dt(dt_val):
    if isinstance(dt_val, datetime):
        return dt_val
    if not dt_val:
        return datetime.utcnow()
    dt_str = str(dt_val).split(".")[0].replace("T", " ")
    try:
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return datetime.utcnow()


def normalize_yahoo_ticker(ticker, name=""):
    if name and name in YAHOO_PAIRS_MAP:
        return YAHOO_PAIRS_MAP[name]
    clean = (
        ticker.replace("OANDA:", "")
        .replace("IC MARKETS:", "")
        .replace("_", "")
        .replace("/", "")
    )
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

    url = "[https://finnhub.io/api/v1/forex/candle](https://finnhub.io/api/v1/forex/candle)"
    params = {
        "symbol": str(symbol).strip(),
        "resolution": str(resolution).strip(),
        "from": start_time,
        "to": end_time,
        "token": str(token).strip(),
    }

    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("s") == "ok":
                df = pd.DataFrame(
                    {
                        "open": data["o"],
                        "high": data["h"],
                        "low": data["l"],
                        "close": data["c"],
                        "volume": data["v"],
                    },
                    index=pd.to_datetime(data["t"], unit="s"),
                )
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
            df_yf = yf.download(
                tickers=ticker,
                period=range_period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

        if not df_yf.empty:
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            df_yf = df_yf.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
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
    return (
        df_1m.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )


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
        df_1h = cached_data["1h"]
        df_daily = cached_data["1d"]
    else:
        df_1h = fetch_finnhub_candles(ticker_finnhub, resolution="60", count_candles=200)
        if df_1h.empty and ticker_yahoo:
            df_1h = fetch_yahoo_fallback(ticker_yahoo, interval="1h", range_period="30d")

        df_daily = fetch_finnhub_candles(ticker_finnhub, resolution="D", count_candles=30)
        if df_daily.empty and ticker_yahoo:
            df_daily = fetch_yahoo_fallback(ticker_yahoo, interval="1d", range_period="60d")

        if not df_1h.empty and not df_daily.empty:
            MACRO_CACHE[cache_key] = {
                "timestamp": current_time,
                "1h": df_1h,
                "1d": df_daily,
            }

    live_p = get_live_price(ticker_finnhub) or (get_live_price(ticker_yahoo) if ticker_yahoo else None)
    if live_p and not df_1m.empty:
        df_1m.iloc[-1, df_1m.columns.get_loc("close")] = live_p
        if not df_3m.empty: df_3m.iloc[-1, df_3m.columns.get_loc("close")] = live_p
        if not df_5m.empty: df_5m.iloc[-1, df_5m.columns.get_loc("close")] = live_p

    return df_daily, df_1h, df_15m, df_5m, df_3m, df_1m


def get_current_session_info():
    now_utc = datetime.utcnow()
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
        if not sig_data or sig_data.get("status") == "CLOSED":
            return

        ticker = sig_data["ticker"]
        entry_price = float(sig_data["entry_price"])
        signal_type = sig_data["signal_type"]
        expiration_mins = int(sig_data.get("expiration_mins", 5))

        created_at = parse_dt(sig_data.get("timestamp_str"))
        elapsed_seconds = (datetime.utcnow() - created_at).total_seconds()
        target_seconds = expiration_mins * 60

        if 0 <= elapsed_seconds < (target_seconds - 10):
            remaining_delay = target_seconds - elapsed_seconds
            timer = threading.Timer(remaining_delay, process_signal_expiration, args=[sig_id])
            timer.daemon = True
            timer.start()
            return

        pair_name = ""
        for k_name, k_ticker in PAIRS_MAP.items():
            if k_ticker == ticker:
                pair_name = k_name
                break

        exit_price = get_exit_price(ticker, pair_name)

        if not exit_price:
            timer = threading.Timer(30, process_signal_expiration, args=[sig_id])
            timer.daemon = True
            timer.start()
            return

        multiplier = 1000 if "JPY" in ticker else 100000
        raw_pips = (
            (exit_price - entry_price) * multiplier
            if signal_type == "CALL"
            else (entry_price - exit_price) * multiplier
        )
        pips = int(round(raw_pips))

        if pips > 0:
            result = "WIN"
        elif pips < 0:
            result = "LOSS"
        else:
            result = "NEUTRAL"

        database.update_signal_result(sig_id, result, exit_price, pips)

        res_icon = "✅ WIN" if result == "WIN" else ("❌ LOSS" if result == "LOSS" else "➖ NEUTRAL")
        pips_str = f"+{pips}" if pips > 0 else f"{pips}"

        report_str = (
            f"\n----------------------------------\n"
            f"🏁 <b>Результат:</b> {res_icon} (<code>{pips_str}</code> п.)\n"
            f"📍 Вхід: <code>{entry_price:.5f}</code> ➔ 🏁 Закриття: <code>{exit_price:.5f}</code>"
        )

        # Додаємо інлайн-кнопку «🔍 Аналіз помилки» для збиткових чи інших закритих угод
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🔍 Аналіз помилки (ШІ)", callback_data=f"review_{sig_id}")]
        ])

        orig_txt = sig_data.get("message_text", "")
        if orig_txt and "🏁 Результат" not in orig_txt:
            try:
                bot.edit_message_text(
                    chat_id=sig_data["chat_id"],
                    message_id=sig_data["message_id"],
                    text=f"{orig_txt}\n{report_str}",
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
            except Exception:
                bot.send_message(
                    chat_id=sig_data["chat_id"],
                    text=f"🏁 <b>Результат угоди #{sig_id} ({ticker}):</b>\n{res_icon} (<code>{pips_str}</code> п.)\n📍 Вхід: <code>{entry_price:.5f}</code> ➔ 🏁 Закриття: <code>{exit_price:.5f}</code>",
                    parse_mode="HTML",
                    reply_to_message_id=sig_data["message_id"],
                    reply_markup=keyboard,
                )

    except Exception as e:
        logger.exception(f"Помилка таймера {sig_id}: {e}")


def schedule_signal_timer(sig_id, timestamp_val, expiration_mins):
    try:
        signal_time = parse_dt(timestamp_val)
        expiry_time = signal_time + timedelta(minutes=expiration_mins)
        delay = (expiry_time - datetime.utcnow()).total_seconds()
        if delay <= 0: delay = 1
        timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
        timer.daemon = True
        timer.start()
    except Exception as e:
        logger.exception(f"Помилка таймера {sig_id}: {e}")


def start_background_checker():
    def loop():
        while True:
            try:
                pending = database.get_pending_signals()
                now = datetime.utcnow()
                for row in pending:
                    sig_id = row["id"]
                    expiration_mins = row["expiration_mins"]
                    created_at = parse_dt(row["timestamp_str"])
                    if now >= created_at + timedelta(minutes=expiration_mins):
                        process_signal_expiration(sig_id)
            except Exception as e:
                logger.error(f"Помилка перевірки сигналів: {e}")
            time.sleep(20)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


start_background_checker()


def analyze_single_pair(chat_id, pair_name, ticker_finnhub):
    try:
        yahoo_ticker = normalize_yahoo_ticker(ticker_finnhub, pair_name)
        df_daily, df_1h, df_15m, df_5m, df_3m, df_1m = fetch_all_timeframes(
            ticker_finnhub, yahoo_ticker
        )

        if df_1m is None or df_1m.empty or len(df_1m) < 10:
            database.save_filtered_log(chat_id, f"⚠️ {pair_name}: Недостатньо даних котирувань.")
            return False

        analysis = analyzer.analyze_all_timeframes(
            df_daily, df_1h, df_15m, df_5m, df_3m, df_1m
        )
        
        is_valid, reason_val = validate_signal_conditions(analysis)

        if not is_valid:
            database.save_filtered_log(chat_id, f"❌ {pair_name}: {reason_val}")
            return False

        session_str, s_code, s_hour = get_current_session_info()
        ml_features = {
            "rsi": analysis.get("rsi", 50),
            "adx": analysis.get("adx", 20),
            "bb_width": analysis.get("bb_width", 0.001),
            "session_code": s_code,
            "hour": s_hour,
            "divergence": analysis.get("divergence", "NONE"),
            "volatility_ratio": analysis.get("volatility_ratio", 1.0)
        }
        win_probability = ml_filter.predict_proba(ml_features)

        # Знижено поріг ML з 56% до 50% для збільшення кількості сигналів
        if win_probability < 50.0:
            log_msg = f"❌ {pair_name}: ML відхилив (Ймовірність {win_probability:.1f}% нижче 50.0%)"
            database.save_filtered_log(chat_id, log_msg)
            return False

        is_pivot_ok, pivot_reason = check_pivot_level_proximity(analysis)
        if not is_pivot_ok:
            log_msg = f"⏭ {pair_name} відхилено (Рівень): {pivot_reason}"
            database.save_filtered_log(chat_id, log_msg)
            return False

        exp_time = calculate_dynamic_expiration(analysis)
        payload = {
            "signal": analysis.get("signal"),
            "current_price": analysis.get("current_price"),
            "rsi": analysis.get("rsi"),
            "adx": analysis.get("adx"),
            "global_trend": analysis.get("global_trend"),
            "mid_trend": analysis.get("mid_trend"),
            "divergence": analysis.get("divergence"),
            "reason": analysis.get("reason"),
            "atr_ratio": analysis.get("atr_ratio"),
            "suggested_exp": exp_time,
        }

        ai_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
        
        if ai_key:
            c_macro = create_chart_image(df_1h, f"{pair_name} - 1H Global", tf_label="1H")
            c_mid = create_chart_image(df_15m, f"{pair_name} - 15M Mid", tf_label="15M")
            c_micro = create_chart_image(df_1m, f"{pair_name} - 1M Entry", tf_label="1M")

            ai_res = ai_advisor_instance.evaluate_signal(
                pair_name, payload, c_macro, c_mid, c_micro
            )
        else:
            ai_res = {
                "decision": "YES", 
                "confidence": 7, 
                "suggested_expiration": exp_time, 
                "reason": "Миттєвий вхід (ШІ вимкнено)"
            }

        is_ai_busy = ai_res.get("confidence", 0) <= 1 or "недоступний" in ai_res.get("reason", "").lower()

        if is_ai_busy:
            if win_probability >= 50.0:
                ai_decision = "YES"
                ai_confidence = 7
                ai_exp = exp_time
                ai_reason = f"Авто-схвалення (ML {win_probability:.1f}%)."
            else:
                log_msg = f"🤖 {pair_name}: ШІ недоступний, ML ({win_probability:.1f}%) недостатній."
                database.save_filtered_log(chat_id, log_msg)
                return False
        else:
            ai_decision = ai_res.get("decision", "NO")
            ai_confidence = ai_res.get("confidence", 0)
            ai_exp = ai_res.get("suggested_expiration", exp_time)
            ai_reason = ai_res.get("reason", "")

            if ai_decision != "YES" or ai_confidence < 5:
                log_msg = f"🤖 {pair_name}: ШІ відхилив (Оцінка: {ai_confidence}/10). Причина: {ai_reason}"
                database.save_filtered_log(chat_id, log_msg)
                return False

        direction_icon = "🟢 CALL (ВХІД ВГОРУ)" if payload["signal"] == "CALL" else "🔴 PUT (ВХІД ВНИЗ)"
        msg_text = (
            f"🎯 <b>СИГНАЛ: {pair_name}</b>\n\n"
            f"📊 Напрямок: <b>{direction_icon}</b>\n"
            f"⏳ Час експірації: <b>{ai_exp} хв</b>\n"
            f"📍 Поточна ціна: <code>{payload['current_price']:.5f}</code>\n\n"
            f"🧠 ML Впевненість: <b>{win_probability:.1f}%</b>\n"
            f"🤖 ШІ Оцінка: <b>{ai_confidence}/10</b>\n"
            f"💡 Причина: <i>{ai_reason}</i>\n\n"
            f"🌐 Сесія: {session_str}"
        )

        sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="HTML")
        sig_id = database.save_signal(
            chat_id=chat_id,
            message_id=sent_msg.message_id,
            ticker=ticker_finnhub,
            signal_type=payload["signal"],
            entry_price=payload["current_price"],
            expiration_mins=ai_exp,
            message_text=msg_text,
            timestamp_str=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        )
        schedule_signal_timer(sig_id, datetime.utcnow(), ai_exp)
        return True

    except Exception as e:
        logger.exception(f"Помилка аналізу {pair_name}: {e}")
        return False


def run_mass_analysis(chat_id):
    bot.send_message(chat_id=chat_id, text="🔍 Запускаю фоновий аналіз...")
    database.clear_filtered_logs(chat_id)

    found_any = False
    for pair_name, ticker in PAIRS_MAP.items():
        if analyze_single_pair(chat_id, pair_name, ticker):
            found_any = True

    reply_text = (
        "✅ Масовий аналіз завершено!"
        if found_any
        else "✅ Масовий аналіз завершено! Відхилені сигнали переглядайте у меню «📋 Логи фільтру»."
    )
    bot.send_message(chat_id=chat_id, text=reply_text)


def start(update, context):
    user = update.effective_user
    database.register_user(user.id, user.username)

    keyboard = [
        [KeyboardButton("📊 Аналіз усіх пар"), KeyboardButton("💵 Пари")],
        [KeyboardButton("📈 Статистика"), KeyboardButton("📋 Логи фільтру")],
    ]
    update.message.reply_text(
        "Бот Racio_1 готовий! 🚀 Оберіть дію:",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True),
    )


def handle_message(update, context):
    text = update.message.text
    chat_id = update.effective_chat.id

    if text == "📊 Аналіз усіх пар":
        threading.Thread(target=run_mass_analysis, args=[chat_id], daemon=True).start()
    elif text == "💵 Пари":
        buttons = []
        row = []
        for pair_name in PAIRS_MAP.keys():
            row.append(InlineKeyboardButton(pair_name, callback_data=f"analyze_{pair_name}"))
            if len(row) == 3:
                buttons.append(row)
                row = []
        if row:
            buttons.append(row)
        update.message.reply_text("Оберіть пару для аналізу:", reply_markup=InlineKeyboardMarkup(buttons))
    elif text == "📈 Статистика":
        stats = database.get_stats()
        total = stats.get('total', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        pending = stats.get('pending', 0)
        
        finished = wins + losses
        winrate = (wins / finished * 100) if finished > 0 else 0.0

        msg = (
            f"📊 <b>Загальна статистика:</b>\n\n"
            f"Всього сигналів: {total}\n"
            f"✅ Перемог (WIN): {wins}\n"
            f"❌ Збитків (LOSS): {losses}\n"
            f"⏳ В очікуванні: {pending}\n\n"
            f"🏆 <b>Поточний вінрейт: {winrate:.1f}%</b>"
        )
        
        keyboard = [
            [InlineKeyboardButton("🤖 Перенавчити ML модель", callback_data="retrain_ml")],
            [InlineKeyboardButton("🗑 Очистити статистику", callback_data="clear_stats_confirm")]
        ]
        
        update.message.reply_text(
            msg, 
            parse_mode="HTML", 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    elif text == "📋 Логи фільтру":
        logs = database.get_system_logs(chat_id)
        if not logs:
            update.message.reply_text("📋 Логи фільтру порожні.")
        else:
            msg = "📋 <b>Останні відхилені сигнали:</b>\n\n" + "\n".join(logs)
            update.message.reply_text(msg, parse_mode="HTML")


def handle_callback(update, context):
    query = update.callback_query
    query.answer()
    chat_id = query.message.chat_id
    data = query.data

    if data.startswith("analyze_"):
        pair_name = data.replace("analyze_", "")
        ticker = PAIRS_MAP.get(pair_name)
        if ticker:
            query.edit_message_text(f"🔍 Аналізую {pair_name}...")
            threading.Thread(
                target=analyze_single_pair,
                args=[chat_id, pair_name, ticker],
                daemon=True,
            ).start()

    elif data.startswith("review_"):
        sig_id = int(data.replace("review_", ""))
        query.edit_message_text(f"🔍 Формую ретроспективний ШІ-аналіз для угоди #{sig_id}... ⏳")
        
        def run_ai_review():
            try:
                sig_data = database.get_signal_by_id(sig_id)
                if not sig_data:
                    bot.send_message(chat_id=chat_id, text="❌ Дані угоди не знайдено.")
                    return
                
                review_text = ai_advisor_instance.evaluate_closed_trade(sig_data)
                database.save_signal_ai_review(sig_id, review_text)
                
                orig_text = sig_data.get("message_text", "")
                result_part = f"\n🏁 <b>Результат:</b> {sig_data.get('result')} ({sig_data.get('pips')} п.)"
                review_formatted = f"\n\n🤖 <b>Ретроспективний розбір ШІ:</b>\n<i>{review_text}</i>"
                
                full_updated_text = orig_text + result_part + review_formatted
                
                bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=query.message.message_id,
                    text=full_updated_text,
                    parse_mode="HTML"
                )
            except Exception as e:
                bot.send_message(chat_id=chat_id, text=f"❌ Помилка формування аналізу: {e}")

        threading.Thread(target=run_ai_review, daemon=True).start()
            
    elif data == "clear_stats_confirm":
        keyboard = [
            [InlineKeyboardButton("⚠️ ТАК, ВИДАЛИТИ", callback_data="clear_stats_do")],
            [InlineKeyboardButton("❌ Скасувати", callback_data="cancel_action")]
        ]
        query.edit_message_text(
            "Ви впевнені, що хочете видалити всю статистику сигналів? Дія незворотня.", 
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
    elif data == "clear_stats_do":
        database.clear_all_stats()
        query.edit_message_text("✅ Статистику успішно очищено!")
        
    elif data == "cancel_action":
        query.edit_message_text("Дію скасовано.")
        
    elif data == "retrain_ml":
        query.edit_message_text(
            "🤖 Запускаю перенавчання ML-моделі... ⏳", 
            parse_mode="HTML"
        )
        
        def run_ml_retrain():
            try:
                success, info_msg = ml_filter.train_model()
                status_icon = "✅" if success else "❌"
                bot.send_message(
                    chat_id=chat_id, 
                    text=f"{status_icon} <b>Результат перенавчання:</b>\n\n📝 <i>{info_msg}</i>", 
                    parse_mode="HTML"
                )
            except Exception as e:
                bot.send_message(chat_id=chat_id, text=f"❌ Помилка перенавчання: {e}")
                
        threading.Thread(target=run_ml_retrain, daemon=True).start()


if __name__ == "__main__":
    bot.delete_webhook(drop_pending_updates=True)
    logger.info("🧹 Старий вебхук очищено.")

    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dispatcher.add_handler(CallbackQueryHandler(handle_callback))

    logger.info("🚀 Бот запущено в режимі Polling!")
    updater.start_polling()
    updater.idle()
