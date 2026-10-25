# -*- coding: utf-8 -*-
import os
import logging
import json
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class AIAdvisor:
    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.5-flash"
        else:
            self.client = None
            self.model_name = None

    def evaluate_signal(self, pair_name: str, payload: dict, chart_macro, chart_mid, chart_micro) -> dict:
        if not self.client:
            return {
                "decision": "YES", 
                "confidence": 7, 
                "suggested_expiration": payload.get("suggested_exp", 10), 
                "reason": "ШІ вимкнено (відсутній ключ GEMINI_API_KEY)"
            }

        system_instruction = (
            "Ти — професійний головний трейдер та ризик-менеджер, який спеціалізується виключно на **БІНАРНИХ ОПЦІОНАХ (Binary Options)**.\n"
            "Твоє завдання — оцінити торговий сигнал з погляду точності мікро-входу, відсутності ринкового шуму та адаптивного часу експірації.\n"
            "Ти аналізуєш три таймфрейми (1H макро-тренд, 15M середній імпульс, 1M мікро-точку входу та свічкові тіні).\n"
            "Враховуй контекст попередніх помилок по цьому активу, якщо він переданий, щоб уникнути повторних хибних входів.\n"
            "Базовий адаптивний час експірації вже розрахований системою, але ти можеш скоригувати його в межах від 1 до 60 хвилин з урахуванням поточної волатильності.\n"
            "Відповідь ти зобов'язаний повернути **ВИКЛЮЧНО у форматі валідного JSON** без жодних додаткових символів чи markdown-обгорток (на кшталт ```json ... ```), у такому форматі:\n"
            "{\n"
            '  "decision": "YES" або "NO",\n'
            '  "confidence": ціле число від 1 до 10,\n'
            '  "suggested_expiration": рекомендований час експірації в хвилинах (ціле число від 1 до 60),\n'
            '  "reason": "Коротке обґрунтування українською мовою з акцентом на мікро-імпульс, волатильність та свічкові тіні"\n'
            "}"
        )

        prompt = (
            f"Аналіз активу: {pair_name}\n"
            f"Сигнал двигуна: {payload.get('signal')} (Стратегія: {payload.get('strategy_title')})\n"
            f"Поточна ціна: {payload.get('current_price')}\n"
            f"RSI (1M): {payload.get('rsi')} | ADX: {payload.get('adx')} | Базовий адаптивний час експірації: {payload.get('suggested_exp')} хв\n"
            f"Причина від двигуна: {payload.get('reason')}\n"
            f"Останні невдалі угоди по парі (для саморефлексії): {payload.get('recent_mistakes', 'Немає')}\n\n"
            "Проаналізуй прикріплені графіки (1H, 15M, 1M) та прийми рішення щодо доцільності входу й оптимального часу експірації для бінарних опціонів."
        )

        try:
            contents = [prompt]
            for chart_io, label in [(chart_macro, "1H"), (chart_mid, "15M"), (chart_micro, "1M")]:
                if chart_io and isinstance(chart_io, bytes):
                    contents.append(types.Part.from_bytes(data=chart_io, mime_type="image/png"))
                elif chart_io and hasattr(chart_io, "getvalue"):
                    contents.append(types.Part.from_bytes(data=chart_io.getvalue(), mime_type="image/png"))

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                )
            )

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
            logger.error(f"Помилка ШІ радника (evaluate_signal): {e}")
            return {
                "decision": "YES",
                "confidence": 6,
                "suggested_expiration": int(payload.get("suggested_exp", 10)),
                "reason": "Помилка парсингу відповіді ШІ, прийнято за адаптивним алгоритмом."
            }

    def evaluate_closed_trade(self, sig_data: dict) -> str:
        if not self.client or not sig_data:
            return "ШІ вимкнено або даних недостатньо."

        prompt = (
            f"Проведи глибокий ретроспективний аналіз закритої угоди на бінарних опціонах:\n"
            f"Актив: {sig_data.get('ticker')}\n"
            f"Тип сигналу: {sig_data.get('signal_type')}\n"
            f"Ціна входу: {sig_data.get('entry_price')} -> Ціна закриття: {sig_data.get('exit_price')}\n"
            f"Результат: {sig_data.get('result')} ({sig_data.get('pips')} пунктів)\n"
            f"Час експірації: {sig_data.get('expiration_mins')} хв | RSI: {sig_data.get('rsi')} | ADX: {sig_data.get('adx')}\n"
            f"Стратегія: {sig_data.get('strategy')}\n\n"
            "Напиши розгорнутий висновок українською мовою (2-3 речення): чому угода завершилася так (WIN/LOSS), чи була правильно підібрана експірація та які уроки варто врахувати майбутньому алгоритму."
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(temperature=0.3)
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Помилка ретроспективного аналізу: {e}")
            return "Не вдалося сформувати аналіз помилки через збій API."

ai_advisor_instance = AIAdvisor()
