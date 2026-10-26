# -*- coding: utf-8 -*-
import os
import logging
import json

logger = logging.getLogger(__name__)

# Безпечне підключення старого SDK Google Generative AI без конфліктів простору імен
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

class AIAdvisor:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
        if api_key and HAS_GENAI:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel("gemini-1.5-flash")
            self.model_name = "gemini-1.5-flash"
        else:
            self.client = None
            self.model_name = None

    def evaluate_signal(self, pair_name: str, payload: dict, chart_macro, chart_mid, chart_micro) -> dict:
        if not self.client:
            return {
                "decision": "YES", 
                "confidence": 7, 
                "suggested_expiration": payload.get("suggested_exp", 10), 
                "reason": "ШІ вимкнено (відсутній ключ або бібліотека)"
            }

        system_instruction = (
            "Ти — професійний головний трейдер та ризик-менеджер, який спеціалізується виключно на **БІНАРНИХ ОПЦІОНАХ (Binary Options)**.\n"
            "Твоє завдання — оцінити торговий сигнал з погляду точності мікро-входу, відсутності ринкового шуму та адаптивного часу експірації.\n"
            "Відповідь ти зобов'язаний повернути **ВИКЛЮЧНО у форматі валідного JSON** без жодних додаткових символів чи markdown-обгорток (на кшталт ```json ... ```), у такому форматі:\n"
            "{\n"
            '  "decision": "YES" або "NO",\n'
            '  "confidence": ціле число від 1 до 10,\n'
            '  "suggested_expiration": рекомендований час експірації в хвилинах (ціле число від 1 до 60),\n'
            '  "reason": "Коротке обґрунтування українською мовою з акцентом на мікро-імпульс"\n'
            "}"
        )

        prompt = (
            f"Аналіз активу: {pair_name}\n"
            f"Сигнал двигуна: {payload.get('signal')} (Стратегія: {payload.get('strategy_title')})\n"
            f"Поточна ціна: {payload.get('current_price')}\n"
            f"RSI (1M): {payload.get('rsi')} | ADX: {payload.get('adx')} | Базовий час експірації: {payload.get('suggested_exp')} хв\n"
            f"Причина від двигуна: {payload.get('reason')}\n"
        )

        try:
            img_parts = []
            for chart_io in [chart_macro, chart_mid, chart_micro]:
                if chart_io:
                    val = chart_io.getvalue() if hasattr(chart_io, "getvalue") else chart_io
                    img_parts.append({"mime_type": "image/png", "data": val})
            
            full_prompt = [system_instruction, prompt] + img_parts
            response = self.client.generate_content(full_prompt)
            text_res = response.text.strip()

            if text_res.startswith("```"):
                text_res = text_res.split("```")[1]
                if text_res.startswith("json"):
                    text_res = text_res[4:]
                text_res = text_res.strip()

            parsed = json.loads(text_res)
            return {
                "decision": str(parsed.get("decision", "YES")).upper(),
                "confidence": int(parsed.get("confidence", 7)),
                "suggested_expiration": int(parsed.get("suggested_expiration", payload.get("suggested_exp", 10))),
                "reason": str(parsed.get("reason", "ШІ підтвердив вхід"))
            }
        except Exception as e:
            logger.error(f"Помилка ШІ радника: {e}")
            return {
                "decision": "YES",
                "confidence": 6,
                "suggested_expiration": int(payload.get("suggested_exp", 10)),
                "reason": "Помилка відповіді ШІ, прийнято за алгоритмом."
            }

    def evaluate_closed_trade(self, sig_data: dict) -> str:
        return "Ретроспективний аналіз очікує на дані."

ai_advisor_instance = AIAdvisor()
