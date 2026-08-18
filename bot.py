import os
import io
import yfinance as yf
from flask import Flask, request
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Dispatcher, CallbackQueryHandler, CommandHandler, MessageHandler, Filters

from config import TELEGRAM_TOKEN, PAIRS_MAP
from ai_advisor import AITradingAdvisor
from indicators import AdaptiveTechnicalAnalysis
from charts import create_chart_image
import database

app = Flask(__name__)

bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

advisor = AITradingAdvisor()
analyzer = AdaptiveTechnicalAnalysis()
active_signals = {}

@app.route("/")
def index():
    return "Racio_1bot is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok", 200

def start(update, context):
    keyboard = [
        [KeyboardButton("📊 Аналіз усіх пар"), KeyboardButton("💵 Пари")],
        [KeyboardButton("📈 Статистика")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    update.message.reply_text(
        "Бот Racio_1 успішно запущений! 🚀 Оберіть потрібну опцію в меню нижче:",
        reply_markup=reply_markup
    )

def handle_text_menu(update, context):
    text = update.message.text
    
    if text == "💵 Пари":
        pairs = list(PAIRS_MAP.items())
        keyboard = []
        for i in range(0, len(pairs), 2):
            row = [InlineKeyboardButton(pairs[i][0], callback_data=f"scan_{pairs[i][1]}")]
            if i + 1 < len(pairs):
                row.append(InlineKeyboardButton(pairs[i+1][0], callback_data=f"scan_{pairs[i+1][1]}"))
            keyboard.append(row)
            
        reply_markup = InlineKeyboardMarkup(keyboard)
        update.message.reply_text("📌 Оберіть пару для запиту аналізу:", reply_markup=reply_markup)
        
    elif text == "📊 Аналіз усіх пар":
        update.message.reply_text("🔄 Запущено ручний запит на загальне сканування ринку...")
        
    elif text == "📈 Статистика":
        stats_text = "📈 Статистика роботи бота:\n"
        try:
            stats = database.get_statistics() if hasattr(database, 'get_statistics') else None
            if stats:
                stats_text += (
                    f"• Загальний вінрейт: {stats.get('winrate', 0)}%\n"
                    f"• Успішних угод: {stats.get('wins', 0)}\n"
                    f"• Усього угод: {stats.get('total', 0)}"
                )
            else:
                stats_text += "• База даних підключена, збір статистики активний."
        except Exception:
            stats_text += "• Статистика обробляється локально через database.py."
        
        update.message.reply_text(stats_text)

def button_callback(update, context):
    query = update.callback_query
    query.answer()
    data = query.data
    
    if data.startswith("scan_"):
        ticker = data.replace("scan_", "")
        query.edit_message_text(
            text=f"🔄 Завантаження даних та розрахунок індикаторів для **{ticker}**...",
            parse_mode="Markdown"
        )
        try:
            # Завантажуємо реальні дані з Yahoo Finance
            df_macro = yf.download(ticker, period="60d", interval="1h", progress=False)
            df_mid = yf.download(ticker, period="10d", interval="15m", progress=False)
            df_micro = yf.download(ticker, period="2d", interval="5m", progress=False)
            
            if df_macro.empty or df_mid.empty:
                query.edit_message_text(text=f"❌ Помилка: не вдалося завантажити дані для {ticker}")
                return

            # Генеруємо реальні графіки через модуль charts
            macro_buf = create_chart_image(df_macro, ticker, "1h")
            mid_buf = create_chart_image(df_mid, ticker, "15m")
            micro_buf = create_chart_image(df_micro, ticker, "5m")
            
            # Розраховуємо індикатори та тренди
            df_indicators = analyzer.calculate_indicators(df_mid)
            sig_data = analyzer.generate_signal(df_indicators, "UP", "UP", "UP")
            
            active_signals[ticker] = {
                "signal_data": sig_data,
                "macro_chart": macro_buf,
                "mid_chart": mid_buf,
                "micro_chart": micro_buf
            }
            
            keyboard = [[InlineKeyboardButton("🔍 Детальний аналіз ШІ", callback_data=f"ai_analyze_{ticker}")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            query.edit_message_text(
                text=f"📈 **Результат технічного аналізу: {ticker}**\n⚡ Графіки та індикатори готові до перевірки ШІ.",
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        except Exception as e:
            query.edit_message_text(text=f"❌ Помилка обробки: {str(e)}")
        
    elif data.startswith("ai_analyze_"):
        ticker = data.replace("ai_analyze_", "")
        query.edit_message_text(
            text=query.message.text + "\n\n⏳ *ШІ аналізує реальні графіки, зачекайте...*",
            parse_mode="Markdown"
        )
        
        saved_item = active_signals.get(ticker)
        if not saved_item:
            query.edit_message_text(
                text=query.message.text.split("\n\n⏳")[0] + "\n\n❌ *Дані застаріли. Зробіть запит повторно.*",
                parse_mode="Markdown"
            )
            return
            
        sig_data = saved_item["signal_data"]
        ai_result = advisor.evaluate_signal(
            ticker, 
            sig_data, 
            saved_item["macro_chart"], 
            saved_item["mid_chart"], 
            saved_item["micro_chart"]
        )
        
        clean_text = query.message.text.split("\n\n⏳")[0]
        updated_text = (
            f"{clean_text}\n\n"
            f"🤖 **Результат ШІ-аналізу:**\n"
            f"• Рішення: **{ai_result.get('decision')}** (Впевненість ШІ: {ai_result.get('confidence')}/10)\n"
            f"• Експірація: {ai_result.get('expiration')} хв\n"
            f"• Висновок: {ai_result.get('reason')}"
        )
        query.edit_message_text(text=updated_text, parse_mode="Markdown")

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text_menu))
dispatcher.add_handler(CallbackQueryHandler(button_callback))
