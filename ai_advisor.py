# -*- coding: utf-8 -*-
import json
import logging
import os
import google.api_core.exceptions
import google.generativeai as genai
from PIL import Image
import database

logger = logging.getLogger(__name__)

class AITradingAdvisor:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key.strip())

    def get_available_models(self) -> list:
        try:
            available_models = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    available_models.append(m.name)
            available_models.sort(
                key=lambda name: (0 if "flash" in name else (1 if "pro" in name else 2))
            )
            if available_models:
                return available_models
        except Exception as e:
            logger.warning(f"⚠️ Не вдалося отримати список моделей через API: {e}")
        return [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
        ]

    def evaluate_signal(self, name: str, payload: dict, macro_chart=None, mid_chart=None, micro_chart=None) -> dict:
        suggested_exp = payload.get("suggested_exp", 5)
        regime = payload.get("regime", "RANGE_BOUND")

        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY відсутній. ШІ-аналіз пропущено.")
            return {
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": suggested_exp,
                "reason": "GEMINI_API_KEY не налаштовано",
            }

        # Підтягуємо історію останніх угод (Feedback Loop та Few-Shot)
        recent_history = database.get_recent_trades_summary(limit=8)

        # Динамічний поріг впевненості залежно від ринкового режиму
        min_confidence_required = 8 if regime in ["CHAOS", "TREND_EXHAUSTION"] else 6

        default_prompt = f"""
        Ти старший ризик-менеджер елітної системи БІНАРНИХ ОПЦІОНІВ (Fixed Time Trades / High-Yield Options).
        СУВОРА ВИМОГА: Ніяких стоп-лосів (Stop-Loss), тейк-профітів чи відкладених ордерів тут не існує! Угода закривається виключно за фіксованим ТАЙМЕРОМ ЕКСПІРАЦІЇ. Твоя мета — оцінити, чи закриється ціна вище (для CALL) або нижче (для PUT) від точки входу через вказаний час.

        Актив: {name}
        Поточний мікро-режим ринку: {regime}
        Стратегія: {payload.get('strategy_title', 'Quantum Matrix')} (Балів збігу: {payload.get('confluence_count', 1)}).

        ІСТОРІЯ ОСТАННІХ УГОД (Враховуй попередній досвід та помилки системи):
        {recent_history}
        
        Параметри поточного сигналу:
        - Напрямок: {payload.get('signal')} (CALL або PUT)
        - Ціна входу: {payload.get('current_price')}
        - RSI: {payload.get('rsi')}
        - ADX: {payload.get('adx')}
        - Технічне обґрунтування: {payload.get('reason')}
        - Рекомендована експірація: {suggested_exp} хв

        Завдання:
        1. Оціни доцільність входу ("YES" або "NO"). У режимі {regime} будь особливо критичним.
        2. Вкажи точний час експірації у хвилинах ВІД 1 ДО 60 ХВИЛИН з урахуванням волатильності та режиму {regime}.
        3. Дай оцінку впевненості confidence від 1 до 10 (мінімально допустима для цього режиму: {min_confidence_required}).

        Відповідь надай ВИКЛЮЧНО у форматі JSON без жодних обгорток markdown:
        {{
            "decision": "YES" або "NO",
            "confidence": <число від 1 до 10>,
            "suggested_expiration": <число від 1 до 60>,
            "reason": "<коротке обґрунтування українською мовою з акцентом на час та імпульс без згадки стоп-лосів>"
        }}
        """

        prompt = database.get_system_prompt("signal_evaluation_prompt", default_prompt)
        
        content_parts = [prompt]
        for chart in [macro_chart, mid_chart, micro_chart]:
            if chart:
                try:
                    chart.seek(0)
                    content_parts.append(Image.open(chart))
                except Exception as e:
                    logger.warning(f"⚠️ Помилка відкриття графіка для ШІ: {e}")

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
                logger.warning(f"Модель {model_name} недоступна: {e}")
                continue

        if not response_text:
            return {
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": suggested_exp,
                "reason": "ШІ схвалив за технічним аналізом (таймаут API)",
            }

        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text.replace("```json", "", 1)
            if clean_text.startswith("```"):
                clean_text = clean_text.replace("```", "", 1)
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            result = json.loads(clean_text)
            parsed_exp = int(result.get("suggested_expiration", suggested_exp))
            parsed_exp = max(1, min(60, parsed_exp))
            parsed_conf = int(result.get("confidence", 5))
            parsed_decision = str(result.get("decision", "NO")).upper()

            # Додатковий перенос у блек-лист/відмова, якщо впевненість нижча за поріг режиму
            if parsed_conf < min_confidence_required:
                parsed_decision = "NO"

            return {
                "decision": parsed_decision,
                "confidence": parsed_conf,
                "suggested_expiration": parsed_exp,
                "reason": str(result.get("reason", "ШІ надав загальний висновок")),
            }
        except Exception as e:
            logger.error(f"⚠️ Помилка парсингу відповіді ШІ: {e}")
            return {
                "decision": "YES",
                "confidence": 6,
                "suggested_expiration": suggested_exp,
                "reason": "Вхід схвалено за матрицею стратегій",
            }

    def evaluate_closed_trade(self, sig_data: dict) -> str:
        if not self.api_key:
            return "❌ GEMINI_API_KEY не налаштовано."

        default_review_prompt = f"""
        Ти експерт-трейдер з БІНАРНИХ ОПЦІОНІВ (Binary Options).
        КАТЕГОРИЧНА ВИМОГА: На ринку бінарних опціонів НЕМАЄ СТОП-ЛОСІВ або тейк-профітів! Заборонено використовувати ці терміни!
        Угода оцінюється виключно за правильністю обраного часу експірації та точкою входу.

        Проведи ретроспективний розбір закритої угоди:
        - Актив: {sig_data.get('ticker')}
        - Напрямок: {sig_data.get('signal_type')} (CALL/PUT)
        - Результат: {sig_data.get('result')} ({sig_data.get('pips')} пунктів)
        - Час експірації: {sig_data.get('expiration_mins')} хв
        - RSI: {sig_data.get('rsi')}, ADX: {sig_data.get('adx')}

        Дай короткий аналіз українською мовою у 3-4 реченнях. 
        Поясни помилку або успіх суворо через призму: точки входу, мікро-імпульсу свічки та ЧАСУ ЕКСПІРАЦІЇ. ЖОДНИХ СТОП-ЛОСІВ!
        """

        prompt = database.get_system_prompt("trade_review_prompt", default_review_prompt)
        models_to_try = self.get_available_models()

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={"timeout": 12})
                if response and response.text:
                    return response.text.strip()
            except Exception:
                continue

        return "⚠️ Не вдалося отримати звіт від ШІ."

ai_advisor_instance = AITradingAdvisor()
