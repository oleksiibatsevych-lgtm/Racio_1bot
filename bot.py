import os
import io
import time
import threading
import requests
import logging
import html
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CallbackQueryHandler, CommandHandler, MessageHandler, Filters
from telegram.error import BadRequest, TelegramError

from config import TELEGRAM_TOKEN, PAIRS_MAP, YAHOO_PAIRS_MAP, FINNHUB_TOKEN, FINNHUB_API_KEY
from indicators import AdaptiveTechnicalAnalysis
import database
from ml_model import TradingMLFilter
from ai_advisor import AITradingAdvisor
from charts import create_chart_image
from finnhub_ws import start_finnhub_ws, get_live_price

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

analyzer = AdaptiveTechnicalAnalysis()
ml_filter = TradingMLFilter()
ai_advisor = AITradingAdvisor()

last_sent_signals = {}

database.init_db()
start_finnhub_ws()

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
        "token": token
    }

    try:
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("s") == "ok":
                df = pd.DataFrame({
                    "open": data["o"],
                    "high": data["h"],
                    "low": data["l"],
                    "close": data["c"],
                    "volume": data["v"]
                }, index=pd.to_datetime(data["t"], unit="s"))
                df.dropna(subset=["open", "high", "low", "close"], inplace=True)
                return df
    except Exception as e:
        logger.warning(f"Finnhub REST API error for {symbol}: {e}")

    return pd.DataFrame()

def fetch_yahoo_fallback(ticker, interval="1m", range_period="5d"):
    try:
        # Спосіб 1: Спроба через об'єкт Ticker (краще обходить блокування облачних IP)
        tk = yf.Ticker(ticker)
        df_yf = tk.history(period=range_period, interval=interval, auto_adjust=True)
        
        # Спосіб 2: Резервний через yf.download
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
    resampled = df_1m.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    return resampled

def fetch_all_timeframes(ticker_finnhub, ticker_yahoo=""):
    df_1m = fetch_finnhub_candles(ticker_finnhub, resolution="1", count_candles=600)
    
    if df_1m.empty and ticker_yahoo:
        df_1m = fetch_yahoo_fallback(ticker_yahoo, interval="1m", range_period="5d")

    if df_1m.empty:
        return None, None, None, None, None

    df_5m = resample_candles(df_1m, "5min")
    df_15m = resample_candles(df_1m, "15min")

    df_1h = fetch_finnhub_candles(ticker_finnhub, resolution="60", count_candles=200)
    if df_1h.empty and ticker_yahoo:
        df_1h = fetch_yahoo_fallback(ticker_yahoo, interval="1h", range_period="30d")

    df_daily = fetch_finnhub_candles(ticker_finnhub, resolution="D", count_candles=30)
    if df_daily.empty and ticker_yahoo:
        df_daily = fetch_yahoo_fallback(ticker_yahoo, interval="1d", range_period="60d")

    live_p = get_live_price(ticker_finnhub) or (get_live_price(ticker_yahoo) if ticker_yahoo else None)
    if live_p and not df_1m.empty:
        df_1m.iloc[-1, df_1m.columns.get_loc('close')] = live_p
        if not df_5m.empty:
            df_5m.iloc[-1, df_5m.columns.get_loc('close')] = live_p
        if not df_15m.empty:
            df_15m.iloc[-1, df_15m.columns.get_loc('close')] = live_p

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
    
    session_str = ", ".join(sessions) if sessions else "Тихоокеанська сесія"
    return session_str, session_code, hour

def process_signal_expiration(sig_id):
    try:
        sig_data = database.get_signal_by_id(sig_id)
        if sig_data:
            ticker = sig_data['ticker']
            entry_price = float(sig_data['entry_price'])
            signal_type = sig_data['signal_type']

            exit_price = get_live_price(ticker)
            if not exit_price:
                df_check = fetch_finnhub_candles(ticker, resolution="1", count_candles=5)
                if not df_check.empty:
                    exit_price = float(df_check['close'].iloc[-1])

            if exit_price:
                multiplier = 1000 if "JPY" in ticker else 100000
                raw_pips = (exit_price - entry_price) * multiplier if signal_type == "CALL" else (entry_price - exit_price) * multiplier
                pips = int(round(raw_pips))

                if signal_type == "CALL":
                    result = "WIN" if exit_price > entry_price else ("LOSS" if exit_price < entry_price else "NEUTRAL")
                else:
                    result = "WIN" if exit_price < entry_price else ("LOSS" if exit_price > entry_price else "NEUTRAL")

                database.update_signal_result(sig_id, result, exit_price, pips)

                res_icon = "✅ WIN" if result == "WIN" else ("❌ LOSS" if result == "LOSS" else "➖ NEUTRAL")
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
                            parse_mode="HTML"
                        )
                    except BadRequest:
                        logger.warning(f"⚠️ Повідомлення {sig_data['message_id']} в чаті {sig_data['chat_id']} видалено або недоступне.")
                    except TelegramError as te:
                        logger.warning(f"⚠️ Помилка Telegram API при оновленні сигналу {sig_id}: {te}")
                return

        res_data = database.evaluate_single_signal(sig_id, fetch_yahoo_data_func=None)
        if res_data and res_data.get("chat_id") and res_data.get("message_id"):
            pips_val = int(res_data.get('pips', 0))
            pips_str = f"+{pips_val}" if pips_val > 0 else f"{pips_val}"
            res_result = res_data.get('result', 'NEUTRAL')
            entry_p = float(res_data.get('entry_price', 0.0))
            exit_p = float(res_data.get('exit_price', 0.0))
            
            res_icon = "✅ WIN" if res_result == "WIN" else ("❌ LOSS" if res_result == "LOSS" else "➖ NEUTRAL")

            report_str = (
                f"\n----------------------------------\n"
                f"🏁 <b>Результат:</b> {res_icon} (<code>{pips_str}</code> п.)\n"
                f"📍 Вхід: <code>{entry_p:.5f}</code> ➔ 🏁 Закриття: <code>{exit_p:.5f}</code>"
            )

            orig_txt = res_data.get("message_text", "")
            if orig_txt and "🏁 Результат" not in orig_txt:
                try:
                    bot.edit_message_text(
                        chat_id=res_data["chat_id"], 
                        message_id=res_data["message_id"], 
                        text=f"{orig_txt}\n{report_str}",
                        parse_mode="HTML"
                    )
                except BadRequest:
                    logger.warning(f"⚠️ Повідомлення {res_data['message_id']} в чаті {res_data['chat_id']} видалено або недоступне.")
                except TelegramError as te:
                    logger.warning(f"⚠️ Помилка Telegram API при оновленні сигналу {sig_id}: {te}")
    except Exception as e:
        logger.exception(f"Помилка таймера експірації {sig_id}: {e}")

def schedule_signal_timer(sig_id, timestamp_val, expiration_mins):
    try:
        if isinstance(timestamp_val, datetime):
            signal_time = timestamp_val
        else:
            signal_time = datetime.strptime(str(timestamp_val), "%Y-%m-%d %H:%M:%S")
            
        expiry_time = signal_time + timedelta(minutes=expiration_mins)
        delay = max((expiry_time - datetime.utcnow()).total_seconds(), 1)
        timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
        timer.daemon = True
        timer.start()
    except Exception as e:
        logger.exception(f"Помилка планування таймера {sig_id}: {e}")

def restore_pending_timers():
    pending = database.get_pending_signals()
    for i, row in enumerate(pending):
        sig_id = row[0]
        expiration_mins = row[4]
        timestamp_str = row[5]
        try:
            if isinstance(timestamp_str, datetime):
                signal_time = timestamp_str
            else:
                signal_time = datetime.strptime(str(timestamp_str), "%Y-%m-%d %H:%M:%S")
                
            expiry_time = signal_time + timedelta(minutes=expiration_mins)
            delay = max((expiry_time - datetime.utcnow()).total_seconds(), 2 + (i * 2))
            timer = threading.Timer(delay, process_signal_expiration, args=[sig_id])
            timer.daemon = True
            timer.start()
        except Exception as e:
            logger.warning(f"⚠️ Помилка відновлення таймера {sig_id}: {e}")
    logger.info(f"⏳ Відновлено активних таймерів: {len(pending)}")

restore_pending_timers()

@app.route("/")
def index():
    return "Racio_1bot is running with Finnhub REST & WebSocket!"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        update = Update.de_json(request.get_json(force=True), bot)
        dispatcher.process_update(update)
    except Exception as e:
        logger.error(f"Помилка обробки вебхука: {e}")
    return "ok", 200

def start(update, context):
    user = update.effective_user
    database.register_user(user.id, user.username)
    
    keyboard = [
        [KeyboardButton("📊 Аналіз усіх пар"), KeyboardButton("💵 Пари")],
        [KeyboardButton("📈 Статистика"), KeyboardButton("📋 Логи фільтру")]
    ]
    update.message.reply_text("Бот Racio_1 готовий до роботи! 🚀 Оберіть дію в меню:", reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True))

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
    
    reply_markup = InlineKeyboardMarkup(buttons)
    bot.send_message(chat_id=chat_id, text="Оберіть валютну пару для миттєвого мульти-ТФ аналізу:", reply_markup=reply_markup)

def process_single_pair(chat_id, name, ticker, ignore_cooldown=False):
    try:
        session_str, session_code, hour = get_current_session_info()
        current_time = time.time()
        
        if not ignore_cooldown and ticker in last_sent_signals and (current_time - last_sent_signals[ticker]) < 180:
            bot.send_message(chat_id=chat_id, text=f"⏳ Пара {name} на кулдауні (зачекайте 3 хвилини).")
            return

        ticker_yahoo = YAHOO_PAIRS_MAP.get(name, "")
        df_daily, df_macro, df_mid, df_fast, df_micro = fetch_all_timeframes(ticker, ticker_yahoo)
        
        if df_macro is None or df_macro.empty or df_fast is None or df_fast.empty:
            bot.send_message(chat_id=chat_id, text=f"⚠️ Не вдалося завантажити котирування для {name}. Спробуйте пізніше або іншу пару.")
            return

        ws_price = get_live_price(ticker) or (get_live_price(ticker_yahoo) if ticker_yahoo else None)

        if df_daily is not None and not df_daily.empty:
            df_daily = analyzer.calculate_indicators(df_daily)
        df_macro = analyzer.calculate_indicators(df_macro)
        df_mid = analyzer.calculate_indicators(df_mid)
        df_fast = analyzer.calculate_indicators(df_fast)
        df_micro = analyzer.calculate_indicators(df_micro)

        global_trend = analyzer.get_trend(df_macro, span_val=200)
        mid_trend = analyzer.get_trend(df_mid, span_val=50)
        pivots = analyzer.calculate_pivots(df_daily if (df_daily is not None and not df_daily.empty) else df_macro)
        
        tf_dict = {'1m': df_micro, '5m': df_fast, '15m': df_mid, '1h': df_macro}
        sig_data = analyzer.generate_signal(
            global_trend=global_trend,
            mid_trend=mid_trend,
            df_daily=df_daily,
            tf_dict=tf_dict
        )
        
        signal_type = sig_data.get('signal', 'CALL')
        signal_score = sig_data.get('score', 60)
        primary_tf = sig_data.get('primary_tf', '5m')
        
        rsi = sig_data.get('rsi', 50)
        adx = sig_data.get('adx', 20)
        bb_width = float(df_fast['bb_width'].iloc[-2]) if 'bb_width' in df_fast.columns and len(df_fast) >= 2 else 0.001
        
        pct_b_val = float(sig_data.get('pct_b', 0.5)) if 'pct_b' in sig_data else 0.5

        divergence_str = str(sig_data.get('divergence', 'NONE'))
        volatility_ratio = sig_data.get('volatility_ratio', 1.0)
        wick_ratio = sig_data.get('wick_ratio', 0.0)
        ema_dist = sig_data.get('ema_dist', 0.0)
        
        strategy_name = sig_data.get('strategy', 'Мульти-ТФ Сигнал')
        strategy_priority = sig_data.get('priority', 2)
        calculated_expiration = sig_data.get('expiration_minutes', 5)
        
        current_price = ws_price if ws_price else (float(df_fast['close'].iloc[-1]) if not df_fast.empty else 0.0)
        dist_pivot = (current_price - pivots['P']) / pivots['P'] if pivots['P'] > 0 else 0.0
        
        win_probability = ml_filter.predict_signal_probability(
            rsi, adx, bb_width, session_code, hour, divergence_str, dist_pivot,
            volatility_ratio, wick_ratio, ema_dist
        )
        
        ai_confidence = 7
        ai_reason = "Консультація за індикаторною моделлю"
        optimal_tf_final = primary_tf

        try:
            macro_chart = create_chart_image(df_macro, name, tf_label="1h")
            mid_chart = create_chart_image(df_mid, name, tf_label="15m")
            micro_chart = create_chart_image(df_fast, name, tf_label="5m")

            ai_payload = {
                'signal': signal_type, 'score': signal_score, 'primary_tf': primary_tf,
                'adx': adx, 'global_trend': global_trend, 'mid_trend': mid_trend,
                'reason': sig_data.get('reason'), 'rsi': rsi, 'atr': sig_data.get('atr'),
                'suggested_exp': calculated_expiration, 'strategy': strategy_name,
                'priority': strategy_priority, 'wick_ratio': wick_ratio,
                'ema_dist': ema_dist, 'volatility_ratio': volatility_ratio,
                'divergence': divergence_str
            }

            ai_audit = ai_advisor.evaluate_signal(name, ai_payload, macro_chart, mid_chart, micro_chart) or {}
            ai_confidence = int(ai_audit.get("confidence", 7))
            ai_reason = str(ai_audit.get("reason", "Підтверджено ШІ-консультантом"))
            optimal_tf_final = str(ai_audit.get("optimal_tf", primary_tf))
            
            if ai_audit.get("suggested_expiration"):
                calculated_expiration = int(ai_audit.get("suggested_expiration"))
        except Exception as e:
            logger.warning(f"⚠️ ШІ-сервіси тимчасово офлайн для {name}: {e}")

        last_sent_signals[ticker] = time.time()  
        expiration = calculated_expiration
        
        icon = "🟢" if signal_type == "CALL" else "🔴"
        action_text = "КУПІВЛЯ (CALL)" if signal_type == "CALL" else "ПРОДАЖ (PUT)"
        
        msg_text = (
            f"📊 <b>{html.escape(str(name))} ({html.escape(str(ticker))})</b>\n"
            f"{icon} <b>{action_text}</b> | ⏱ Експірація: <b>{expiration} хв</b>\n"
            f"⚡ <b>Signal Score:</b> <code>{signal_score}/100</code> | 📐 ТФ: <code>{html.escape(str(optimal_tf_final))}</code>\n"
            f"🎯 Стратегія: <code>{html.escape(str(strategy_name))}</code>\n"
            f"----------------------------------\n"
            f"📈 <b>Параметри ринку:</b>\n"
            f"• Ціна входу: <code>{current_price:.5f}</code> {'⚡ (Realtime)' if ws_price else ''}\n"
            f"• Тренди (1h / 15m): <code>{html.escape(str(global_trend))} / {html.escape(str(mid_trend))}</code>\n"
            f"• RSI: <code>{rsi}</code> | ADX: <code>{adx}</code> | %B: <code>{pct_b_val:.2f}</code>\n"
            f"• Дивергенція: <code>{html.escape(str(divergence_str))}</code>\n"
            f"• ATR Ratio: <code>{volatility_ratio}</code> | Відхилення EMA: <code>{ema_dist}%</code>\n"
            f"🌐 Сесія: <code>{html.escape(str(session_str))}</code>\n"
            f"----------------------------------\n"
            f"🤖 <b>Аналітика моделей:</b>\n"
            f"• ML-ймовірність: <code>{round(win_probability * 100, 1)}%</code>\n"
            f"• ШІ-впевненість: <code>{ai_confidence}/10</code>\n"
            f"💡 <b>Обґрунтування:</b> <i>{html.escape(str(sig_data.get('reason', '')))}</i>\n"
            f"🛡 <b>Висновок ШІ:</b> <i>{html.escape(str(ai_reason))}</i>"
        )
        
        sent_msg = bot.send_message(chat_id=chat_id, text=msg_text, parse_mode="HTML")
        
        timestamp_dt = datetime.utcnow()
        timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S")
        
        sig_id = database.save_signal(
            chat_id=chat_id,
            message_id=sent_msg.message_id,
            ticker=ticker,
            signal_type=signal_type,
            entry_price=current_price,
            expiration_mins=expiration,
            timestamp_str=timestamp_str,
            message_text=msg_text
        )

        if sig_id:
            schedule_signal_timer(sig_id, timestamp_dt, expiration)

    except Exception as e:
        logger.exception(f"Помилка обробки сигналу для {name}: {e}")
        bot.send_message(chat_id=chat_id, text=f"❌ Сталася помилка під час аналізу {name}.")
    finally:
        plt.close('all')
        gc.collect()

def process_all_pairs_background(chat_id):
    bot.send_message(chat_id=chat_id, text="🔎 Розпочато сканування всіх доступних пар...")
    for pair_name, pair_ticker in PAIRS_MAP.items():
        try:
            process_single_pair(chat_id, pair_name, pair_ticker, ignore_cooldown=False)
        except Exception as e:
            logger.error(f"Помилка при фоновій обробці {pair_name}: {e}")
        
        plt.close('all')
        gc.collect()
        time.sleep(1.5)

def handle_text_message(update, context):
    text = update.message.text
    chat_id = update.effective_chat.id
    database.register_user(update.effective_user.id, update.effective_user.username)

    if text == "💵 Пари":
        show_pairs_menu(chat_id)
    elif text == "📊 Аналіз усіх пар":
        thread = threading.Thread(target=process_all_pairs_background, args=(chat_id,))
        thread.daemon = True
        thread.start()
    elif text == "📈 Статистика":
        stats_msg = database.get_stats_summary()
        bot.send_message(chat_id=chat_id, text=f"📊 <b>Статистика роботи бота:</b>\n\n{stats_msg}", parse_mode="HTML")
    elif text == "📋 Логи фільтру":
        logs = database.get_filtered_logs(chat_id)
        if logs:
            log_text = html.escape("\n".join(logs[:15]))
            bot.send_message(chat_id=chat_id, text=f"📋 <b>Останні логи:</b>\n\n{log_text}", parse_mode="HTML")
        else:
            bot.send_message(chat_id=chat_id, text="📋 Логи відсутні або застаріли.")

def handle_callback_query(update, context):
    query = update.callback_query
    try:
        query.answer()
    except Exception as e:
        logger.warning(f"Не вдалося відповісти на callback query: {e}")

    if not query or not query.data:
        return

    chat_id = update.effective_chat.id if update.effective_chat else (query.message.chat_id if query.message else None)
    if not chat_id:
        return

    data = query.data
    logger.info(f"📥 Отримано callback_data: '{data}' від chat_id: {chat_id}")

    if data.startswith("pair_"):
        pair_name = data[5:].strip()
        logger.info(f"🔍 Виконується ручний запит для пари: '{pair_name}'")

        if pair_name in PAIRS_MAP:
            bot.send_message(chat_id=chat_id, text=f"⏳ Виконується мульти-ТФ аналіз для пари {pair_name}...")
            
            thread = threading.Thread(
                target=process_single_pair,
                args=(chat_id, pair_name, PAIRS_MAP[pair_name]),
                kwargs={"ignore_cooldown": True},
                daemon=True
            )
            thread.start()
        else:
            logger.error(f"❌ Пару '{pair_name}' не знайдено в PAIRS_MAP!")
            bot.send_message(chat_id=chat_id, text=f"⚠️ Пару {pair_name} не знайдено в конфігурації PAIRS_MAP.")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CallbackQueryHandler(handle_callback_query))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_message))
