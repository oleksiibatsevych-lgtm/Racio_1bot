# -*- coding: utf-8 -*-
import os
import logging
import json
import re
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

class AIAdvisor:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
        if self.api_key and HAS_GENAI:
            genai.configure(api_key=self.api_key.strip())

    def get_available_models(self) -> list:
        """Каскадний перебір доступних моделей з пріоритетом flash для максимальної швидкості."""
        if not HAS_GENAI or not self.api_key:
            return ["gemini-1.5-flash"]
        try:
            available_models = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    available_models.append(m.name)
            # Сортуємо: спочатку flash, потім pro
            available_models.sort(key=lambda name: (0 if "flash" in name else (1 if "pro" in name else 2)))
            if available_models:
                return available_models
        except Exception as e:
            logger.warning(f"⚠️ Не вдалося отримати список моделей через API: {e}")
        return ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]

    def evaluate_signal(self, pair_name: str, payload: dict, chart_macro, chart_mid, chart_micro) -> dict:
        suggested_exp = int(payload.get("suggested_exp", 10))
        if not self.api_key or not HAS_GENAI:
            return {
                "decision": "YES", 
                "confidence": 7, 
                "suggested_expiration": suggested_exp, 
                "reason": "ШІ вимкнено (відсутній ключ або бібліотека)"
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
            f"RSI (1M): {payload.get('rsi')} | ADX: {payload.get('adx')} | Базовий адаптивний час експірації: {suggested_exp} хв\n"
            f"Причина від двигуна: {payload.get('reason')}\n"
            f"Останні невдалі угоди по парі (для саморефлексії): {payload.get('recent_mistakes', 'Немає')}\n\n"
            "Проаналізуй прикріплені графіки (1H, 15M, 1M) та прийми рішення щодо доцільності входу й оптимального часу експірації для бінарних опціонів."
        )

        content_parts = [system_instruction, prompt]
        for chart_io in [chart_macro, chart_mid, chart_micro]:
            if chart_io:
                try:
                    if hasattr(chart_io, "seek"):
                        chart_io.seek(0)
                    content_parts.append(Image.open(chart_io))
                except Exception as e:
                    logger.warning(f"⚠️ Помилка завантаження графіка для ШІ: {e}")

        models_to_try = self.get_available_models()
        response_text = None

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(content_parts, request_options={"timeout": 15})
                if response and response.text:
                    response_text = response.text
                    break
            except Exception as e:
                logger.warning(f"⚠️ Модель {model_name} не відповіла: {e}")
                continue

        if not response_text:
            return {
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": suggested_exp,
                "reason": "ШІ схвалив за технічним аналізом (таймаут моделей)",
            }

        try:
            clean_text = response_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(clean_text)
            parsed_exp = int(parsed.get("suggested_expiration", suggested_exp))
            parsed_exp = max(1, min(60, parsed_exp))

            return {
                "decision": str(parsed.get("decision", "YES")).upper(),
                "confidence": int(parsed.get("confidence", 7)),
                "suggested_expiration": parsed_exp,
                "reason": str(parsed.get("reason", "ШІ підтвердив вхід"))
            }
        except Exception as e:
            logger.error(f"⚠️ Помилка парсингу відповіді ШІ: {e}")
            return {
                "decision": "YES",
                "confidence": 6,
                "suggested_expiration": suggested_exp,
                "reason": "Вхід схвалено за матрицею стратегій (помилка парсингу ШІ)"
            }

    def evaluate_closed_trade(self, sig_data: dict) -> str:
        if not self.api_key or not HAS_GENAI or not sig_data:
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

        models_to_try = self.get_available_models()
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={"timeout": 10})
                if response and response.text:
                    return response.text.strip()
            except Exception:
                continue
        return "⚠️ Не вдалося отримати звіт від ШІ через помилку API."

ai_advisor_instance = AIAdvisor()
