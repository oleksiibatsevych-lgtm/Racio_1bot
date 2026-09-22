import html
import io
import logging
import os
import threading
import time
from datetime import datetime, timedelta
import matplotlib
import pandas as pd
import requests
import yfinance as yf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request
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
    Dispatcher,
    Filters,
    MessageHandler,
)

from config import FINNHUB_API_KEY, FINNHUB_TOKEN, PAIRS_MAP, YAHOO_PAIRS_MAP
from charts import create_chart_image
import database
from finnhub_ws import get_live_price, start_finnhub_ws
from indicators import AdaptiveTechnicalAnalysis
from ml_model import TradingMLFilter

from ai_advisor import analyze_signal_with_gemini
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
    logger.error("❌ TELEGRAM_TOKEN не знайдено в Environment Variables!")

app = Flask(__name__)
bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

RENDER_URL = os.environ.get("RENDER_EXTERNAL_URL", "")
if RENDER_URL:
    webhook_url = f"{RENDER_URL}/webhook"
    try:
        bot.set_webhook(
            url=webhook_url, allowed_updates=["message", "callback_query"]
        )
        logger.info(f"✅ Webhook оновлено: {webhook_url}")
    except Exception as e:
        logger.error(f"⚠️ Помилка встановлення Webhook: {e}")

analyzer = AdaptiveTechnicalAnalysis()
ml_filter = TradingMLFilter()

last_sent_signals = {}

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

    url = "https://finnhub.io/api/v1/forex/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": start_time,
        "to": end_time,
        "token": token,
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
        df_yf = tk.history(
            period=range_period, interval=interval, auto_adjust=True
        )

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
    df_1m = fetch_finnhub_candles(
        ticker_finnhub, resolution="1", count_candles=600
    )
    if df_1m.empty and ticker_yahoo:
        df_1m = fetch_yahoo_fallback(
            ticker_yahoo, interval="1m", range_period="5d"
        )

    if df_1m.empty:
        return None, None, None, None, None

    df_5m = resample_candles(df_1m, "5min")
    df_15m = resample_candles(df_1m, "15min")

    df_1h = fetch_finnhub_candles(
        ticker_finnhub, resolution="60", count_candles=200
    )
    if df_1h.empty and ticker_yahoo:
        df_1h = fetch_yahoo_fallback(
            ticker_yahoo, interval="1h", range_period="30d"
        )

    df_daily = fetch_finnhub_candles(
        ticker_finnhub, resolution="D", count_candles=30
    )
    if df_daily.empty and ticker_yahoo:
        df_daily = fetch_yahoo_fallback(
            ticker_yahoo, interval="1d", range_period="60d"
        )

    live_p = get_live_price(ticker_finnhub) or (
        get_live_price(ticker_yahoo) if ticker_yahoo else None
    )
    if live_p and not df_1m.empty:
        df_1m.iloc[-1, df_1m.columns.get_loc("close")] = live_p
        if not df_5m.empty:
            df_5m.iloc[-1, df_5m.columns.get_loc("close")] = live_p
        if not df_15m.empty:
            df_15m.iloc[-1, df_15m.columns.get_loc("close")] = live_p

    return df_daily, df_1h, df_15m, df_5m, df_1m


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

    return (
        ", ".join(sessions) if sessions else "Тихоокеанська сесія",
        session_code,
        hour,
    )


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
        df_yf = fetch_yahoo_fallback(
            yahoo_ticker, interval="1m", range_period="1d"
        )
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
            timer = threading.Timer(
                remaining_delay, process_signal_expiration, args=[sig_id]
            )
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
            logger.warning(
                f"⚠️ Повторне отримання ціни для {ticker} (ID: {sig_id}) через 30 сек..."
            )
            timer = threading.Timer(
                30, process_signal_expiration, args=[sig_id]
            )
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

        res_icon = (
            "✅ WIN"
            if result == "WIN"
            else ("❌ LOSS" if result == "LOSS" else "➖ NEUTRAL")
        )
        pips_str = f"+{pips}" if pips > 0 else f"{pips}"

        report_str = (
            f"\n----------------------------------\n"
            f"🏁 <b>Результат:</b> {res_icon} (<code>{pips_str}</code> п.)\n"
            f"📍 Вхід: <code>{entry_price:.5f}</code> ➔ 🏁 Закриття: <code>{exit_price:.5f}</code>"
        )

        orig_txt = sig_data.get("message_text", "")
        if orig_txt and "🏁 Результат" not in orig_txt:
            try:
                bot.edit_message_text(
                    chat_id=sig_data["chat_id"],
                    message_id=sig_data["message_id"],
                    text=f"{orig_txt}\n{report_str}",
                    parse_mode="HTML",
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Редагування не вдалося для {sig_id}: {e}. Надсилаємо окремо..."
                )
                bot.send_message(
                    chat_id=sig_data["chat_id"],
                    text=f"🏁 <b>Результат угоди #{sig_id} ({ticker}):</b>\n{res_icon} (<code>{pips_str}</code> п.)\n📍 Вхід: <code>{entry_price:.5f}</code> ➔ 🏁 Закриття: <code>{exit_price:.5f}</code>",
                    parse_mode="HTML",
                    reply_to_message_id=sig_data["message_id"],
                )

    except Exception as e:
        logger.exception(f"Помилка таймера експірації {sig_id}: {e}")


def schedule_signal_timer(sig_id, timestamp_val, expiration_mins):
    try:
        signal_time = parse_dt(timestamp_val)
        expiry_time = signal_time + timedelta(minutes=expiration_mins)
        delay = (expiry_time - datetime.utcnow()).total_seconds()

        if delay <= 0:
            delay = 1

        timer = threading.Timer(
            delay, process_signal_expiration, args=[sig_id]
        )
        timer.daemon = True
        timer.start()
    except Exception as e:
        logger.exception(f"Помилка планування таймера {sig_id}: {e}")


def start_background_checker():
    def loop():
        while True:
            try:
                pending = database.get_pending_signals()
                now = datetime.utcnow()
                for row in pending:
                    sig_id = row["id"]
                    expiration_mins = row["expiration_mins"]
                    timestamp_str = row["timestamp_str"]
                    created_at = parse_dt(timestamp_str)
                    if now >= created_at + timedelta(minutes=expiration_mins):
                        process_signal_expiration(sig_id)
            except Exception as e:
                logger.error(f"⚠️ Помилка перевірки сигналів: {e}")
            time.sleep(20)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()


start_background_checker()


@app.route("/")
def index():
    return "Racio_1bot is active"


@app.route("/webhook", methods=["POST"])
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    try:
        data = request.get_json(force=True)
        if data:
            update = Update.de_json(data, bot)
            dispatcher.process_update(update)
    except Exception as e:
        logger.error(f"⚠️ Помилка обробки вебхука: {e}")
    return "ok", 200


def start(update, context):
    user = update.effective_user
    database.register_user(user.id, user.username)

    keyboard = [
        [KeyboardButton("📊 Аналіз усіх пар"), KeyboardButton("💵 Пари")],
        [
            KeyboardButton("📈 Статистика"),
            KeyboardButton("📋 Логи фільтру"),
        ],
    ]
    update.message.reply_text(
        "Бот Racio_1 готовий до роботи! 🚀 Оберіть дію:",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True),
    )


def show_pairs_menu(chat_id):
    buttons = []
    row = []
    for name in PAIRS_MAP.keys():
        row.append(InlineKeyboardButton(name, callback_data=f"pair_{name}"))
        if len(row) == 3:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)

    bot.send_message(
        chat_id=chat_id,
        text="Оберіть валютну пару для аналізу:",
        reply_markup=InlineKeyboardMarkup(buttons),
    )


def process_single_pair(chat_id, name, ticker, ignore_cooldown=False):
    try:
        session_str, session_code, hour = get_current_session_info()
        current_time = time.time()

        if (
            not ignore_cooldown
            and ticker in last_sent_signals
            and (current_time - last_sent_signals[ticker]) < 180
        ):
            bot.send_message(
                chat_id=chat_id,
                text=f"⏳ Пара {name} на кулдауні (зачекайте 3 хв).",
            )
            return

        ticker_yahoo = YAHOO_PAIRS_MAP.get(name, "")
        df_daily, df_macro, df_mid, df_fast, df_micro = fetch_all_timeframes(
            ticker, ticker_yahoo
        )

        if (
            df_macro is None
            or df_macro.empty
            or df_fast is None
            or df_fast.empty
        ):
            bot.send_message(
                chat_id=chat_id, text=f"⚠️ Не вдалося завантажити {name}."
            )
            return

        ws_price = get_live_price(ticker) or (
            get_live_price(ticker_yahoo) if ticker_yahoo else None
        )

        if df_daily is not None and not df_daily.empty:
            df_daily = analyzer.calculate_indicators(df_daily)
        df_macro = analyzer.calculate_indicators(df_macro)
        df_mid = analyzer.calculate_indicators(df_mid)
        df_fast = analyzer.calculate_indicators(df_fast)
        df_micro = analyzer.calculate_indicators(df_micro)

        global_trend = analyzer.get_trend(df_macro, span_val=200)
        mid_trend = analyzer.get_trend(df_mid, span_val=50)
        pivots = analyzer.calculate_pivots(
            df_daily
            if (df_daily is not None and not df_daily.empty)
            else df_macro
        )

        tf_dict = {"1m": df_micro, "5m": df_fast, "15m": df_mid, "1h": df_macro}
        
        sig_data = analyzer.generate_signal(
            global_trend=global_trend,
            mid_trend=mid_trend,
            df_daily=df_daily,
            tf_dict=tf_dict,
        )

        signal_type = sig_data.get("signal", "CALL")
        signal_score = sig_data.get("score", 60)
        primary_tf = sig_data.get("primary_tf", "5m")

        if signal_type == "NONE":
             return

        rsi = float(sig_data.get("rsi", 50))
        adx = float(sig_data.get("adx", 20))
        bb_width = (
            float(df_fast["bb_width"].iloc[-2])
            if "bb_width" in df_fast.columns and len(df_fast) >= 2
            else 0.001
        )

        divergence_str = str(sig_data.get("divergence", "NONE"))
        volatility_ratio = float(sig_data.get("volatility_ratio", 1.0))
        wick_ratio = float(sig_data.get("wick_ratio", 0.0))
        ema_dist = float(sig_data.get("ema_dist", 0.0))

        calculated_expiration = calculate_dynamic_expiration(
            primary_tf, adx, volatility_ratio
        )

        # Передаємо signal_type для розрізнення напрямку угоди
        is_valid, filter_reason = validate_signal_conditions(
            adx, volatility_ratio, calculated_expiration, rsi=rsi, signal_type=signal_type
        )
        if not is_valid:
            bot.send_message(
                chat_id=chat_id,
                text=f"⏭ <b>Пара {name} пропущена:</b> {html.escape(filter_reason)}",
                parse_mode="HTML",
            )
            return

        current_price = ws_price if ws_price else (
            float(df_fast["close"].iloc[-1]) if not df_fast.empty else 0.0
        )

        is_level_ok, level_reason = check_pivot_level_proximity(
            current_price=current_price,
            signal_type=signal_type,
            pivots=pivots,
            threshold_pct=0.0015,
        )
        if not is_level_ok:
            bot.send_message(
                chat_id=chat_id,
                text=f"⏭ <b>Пара {name} пропущена (Рівень):</b> {html.escape(level_reason)}",
                parse_mode="HTML",
            )
            return

        dist_pivot = (
            (current_price - pivots["P"]) / pivots["P"]
            if pivots["P"] > 0
            else 0.0
        )

        # 🟢 Масштабування ML-ймовірності у відсотки
        raw_prob = ml_filter.predict_signal_probability(
            rsi,
            adx,
            bb_width,
            session_code,
            hour,
            divergence_str,
            dist_pivot,
            volatility_ratio,
            wick_ratio,
            ema_dist,
        )
        win_probability = raw_prob * 100.0 if raw_prob <= 1.0 else raw_prob

        if win_probability < 40.0:
            bot.send_message(
                chat_id=chat_id,
                text=f"⏭ <b>Сигнал для {name} відхилено ML:</b> низька ймовірність (<code>{win_probability:.1f}%</code>)",
                parse_mode="HTML",
            )
            return

        macro_chart = create_chart_image(df_macro, name, tf_label="1h")
        mid_chart = create_chart_image(df_mid, name, tf_label="15m")
        micro_chart = create_chart_image(df_fast, name, tf_label="5m")

        ai_payload = {
            "signal": signal_type,
            "score": signal_score,
            "primary_tf": primary_tf,
            "current_price": current_price,
            "pivot_p": pivots.get("P", 0),
            "pivot_r1": pivots.get("R1", 0),
            "pivot_s1": pivots.get("S1", 0),
            "adx": adx,
            "rsi": rsi,
            "divergence": divergence_str,
            "atr_ratio": volatility_ratio,
            "reason": sig_data.get("reason", "Аналіз параметрів"),
            "ml_prob": win_probability,
            "suggested_exp": calculated_expiration,
        }

        charts_list = [
            img for img in [macro_chart, mid_chart, micro_chart] if img
        ]
        ai_audit = analyze_signal_with_gemini(name, ai_payload, charts_list)

        ai_confidence = int(ai_audit.get("confidence", 0))
        ai_reason = str(ai_audit.get("reason", "Аналіз ШІ"))
        final_expiration = int(
            ai_audit.get("suggested_expiration", calculated_expiration)
        )

        if ai_confidence < 7:
            bot.send_message(
                chat_id=chat_id,
                text=f"⏭ <b>Сигнал для {name} відхилено ШІ ({ai_confidence}/10):</b>\n<i>{html.escape(ai_reason)}</i>",
                parse_mode="HTML",
            )
            return

        last_sent_signals[ticker] = time.time()

        direction_icon = (
            "🟢 КУПІВЛЯ (CALL)"
            if signal_type == "CALL"
            else "🔴 ПРОДАЖ (PUT)"
        )

        msg_text = (
            f"⚡ <b>СИГНАЛ: {name}</b> | {direction_icon}\n"
            f"⏱ <b>Експірація:</b> {final_expiration} хв. | <b>ТФ:</b> {primary_tf}\n"
            f"📍 <b>Ціна входу:</b> <code>{current_price:.5f}</code>\n"
            f"----------------------------------\n"
            f"📈 <b>Тренди (1h / 15m):</b> {global_trend} / {mid_trend}\n"
            f"📊 <b>RSI:</b> {rsi:.1f} | <b>ADX:</b> {adx:.1f} | <b>ATR Ratio:</b> {volatility_ratio:.2f}\n"
            f"🌐 <b>Сесія:</b> {session_str}\n"
            f"----------------------------------\n"
            f"🤖 <b>Аналітика моделей:</b>\n"
            f"• ML-ймовірність: <b>{win_probability:.1f}%</b>\n"
            f"• ШІ-впевненість: <b>{ai_confidence}/10</b>\n"
            f"💡 <b>Обґрунтування:</b> <i>{html.escape(ai_reason)}</i>"
        )

        sent_msg = bot.send_message(
            chat_id=chat_id, text=msg_text, parse_mode="HTML"
        )

        timestamp_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        sig_id = database.save_signal(
            ticker=ticker,
            signal_type=signal_type,
            entry_price=current_price,
            expiration_mins=final_expiration,
            chat_id=chat_id,
            message_id=sent_msg.message_id,
            timestamp_str=timestamp_now,
            message_text=msg_text,
        )

        schedule_signal_timer(sig_id, timestamp_now, final_expiration)

    except Exception as e:
        logger.exception(f"Помилка при обробці пари {name}: {e}")
        bot.send_message(
            chat_id=chat_id,
            text=f"⚠️ Помилка при аналізі пари {name}: {html.escape(str(e))}",
        )


def analyze_all_pairs_async(chat_id):
    """Фонова обробка масового аналізу."""
    logger.info(f"🚀 Запущено фоновий масовий аналіз для chat_id: {chat_id}")
    for name, ticker in PAIRS_MAP.items():
        try:
            process_single_pair(chat_id, name, ticker)
            time.sleep(2)
        except Exception as e:
            logger.error(f"⚠️ Помилка фонового аналізу пари {name}: {e}")
    bot.send_message(
        chat_id=chat_id,
        text="✅ <b>Масовий аналіз усіх пар завершено!</b>",
        parse_mode="HTML",
    )


def handle_callback(update, context):
    query = update.callback_query
    chat_id = query.message.chat_id
    data = query.data

    if data.startswith("pair_"):
        pair_name = data.replace("pair_", "")
        ticker = PAIRS_MAP.get(pair_name)
        if ticker:
            query.answer(f"Аналізуємо {pair_name}...")
            process_single_pair(
                chat_id, pair_name, ticker, ignore_cooldown=True
            )


def handle_message(update, context):
    text = update.message.text
    chat_id = update.effective_chat.id

    if text == "📊 Аналіз усіх пар":
        update.message.reply_text(
            "🔎 <b>Розпочинаю фоновий аналіз усіх пар...</b>\n"
            "Сигнали будуть надходити в чат по мірі перевірки.",
            parse_mode="HTML",
        )
        thread = threading.Thread(
            target=analyze_all_pairs_async, args=(chat_id,), daemon=True
        )
        thread.start()

    elif text == "💵 Пари":
        show_pairs_menu(chat_id)

    elif text == "📈 Статистика":
        stats = database.get_stats()
        total = stats.get("total", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)

        closed_trades = wins + losses
        winrate = (wins / closed_trades * 100.0) if closed_trades > 0 else 0.0

        stat_msg = (
            f"📊 <b>Статистика роботи бота:</b>\n\n"
            f"🎯 Всього згенеровано: <b>{total}</b>\n"
            f"✅ Успішних (WIN): <b>{wins}</b>\n"
            f"❌ Невдалих (LOSS): <b>{losses}</b>\n\n"
            f"📈 <b>Winrate: {winrate:.1f}%</b>"
        )
        update.message.reply_text(stat_msg, parse_mode="HTML")

    elif text == "📋 Логи фільтру":
        update.message.reply_text(
            "📋 Логи розширеного аналізу доступні у консолі управління Render."
        )


dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CallbackQueryHandler(handle_callback))
dispatcher.add_handler(
    MessageHandler(Filters.text & ~Filters.command, handle_message)
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
