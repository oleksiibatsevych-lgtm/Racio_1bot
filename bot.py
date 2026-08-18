import os
import io
from flask import Flask, request
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Dispatcher, CallbackQueryHandler, CommandHandler
from ai_advisor import AITradingAdvisor

app = Flask(__name__)

# Токен вашого бота
TOKEN = "8921212255:AAE_Ypn6wCLUxVMjcrrd8TgPncuLTYQRnSg"
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot, None, use_context=True)

advisor = AITradingAdvisor()
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
    update.message.reply_text("Бот Racio_1 успішно запущений і готовий до роботи! 🚀 Сигнали надходитимуть автоматично.")

def calculate_ta_confidence(signal_data: dict) -> int:
    score = 0
    
    # 1. Збіг трендів на різних таймфреймах (максимум 4 бали)
    if signal_data.get('global_trend') == signal_data.get('mid_trend') == signal_data.get('local_trend'):
        score += 4
    elif signal_data.get('global_trend') == signal_data.get('mid_trend'):
        score += 2
        
    # 2. RSI (максимум 2 бали)
    rsi = signal_data.get('rsi', 50)
    if 30 < rsi < 70:
        score += 2
        
    # 3. Сила тренду за ADX (максимум 3 бали)
    adx = signal_data.get('adx', 0)
    if adx > 25:
        score += 3
    elif adx > 20:
        score += 1
        
    # 4. Волатильність за ATR (максимум 1 бал)
    if signal_data.get('atr', 0) > 0:
        score += 1
        
    return score

def process_and_send_signal(pair_name: str, signal_data: dict, macro_chart: io.BytesIO, mid_chart: io.BytesIO, micro_chart: io.BytesIO, context, chat_id: int):
    confidence = calculate_ta_confidence(signal_data)
    
    # Публікуємо сигнал миттєво, якщо оцінка 7 або вище
    if confidence >= 7:
        active_signals[pair_name] = {
            "signal_data": signal_data,
            "macro_chart": macro_chart,
            "mid_chart": mid_chart,
            "micro_chart": micro_chart
        }
        
        keyboard = [[InlineKeyboardButton("🔍 Детальний аналіз ШІ", callback_data=f"ai_analyze_{pair_name}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message_text = (
            f"📈 **Сигнал: {pair_name}**\n"
            f"⚡ Напрямок: {signal_data['signal']}\n"
            f"📊 ТА Оцінка: **{confidence}/10**\n"
            f"💡 Причина: {signal_data['reason']}"
        )
        
        context.bot.send_message(
            chat_id=chat_id, 
            text=message_text, 
            reply_markup=reply_markup, 
            parse_mode="Markdown"
        )
    else:
        print(f"Сигнал по {pair_name} пропущено через низьку оцінку ТА: {confidence}/10")

def button_callback(update, context):
    query = update.callback_query
    query.answer()
    
    if query.data.startswith("ai_analyze_"):
        pair_name = query.data.replace("ai_analyze_", "")
        
        query.edit_message_text(
            text=query.message.text + "\n\n⏳ *ШІ аналізує графіки, зачекайте...*",
            parse_mode="Markdown"
        )
        
        saved_item = active_signals.get(pair_name)
        if not saved_item:
            query.edit_message_text(
                text=query.message.text.split("\n\n⏳")[0] + "\n\n❌ *Дані графіків застаріли.*",
                parse_mode="Markdown"
            )
            return
            
        sig_data = saved_item["signal_data"]
        ai_result = advisor.evaluate_signal(
            pair_name, 
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
dispatcher.add_handler(CallbackQueryHandler(button_callback))
