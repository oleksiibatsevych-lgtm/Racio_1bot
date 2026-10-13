# -*- coding: utf-8 -*-
import json
import logging
import os
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
        return [
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp",
        ]

    def evaluate_signal(self, name: str, payload: dict, macro_chart=None, mid_chart=None, micro_chart=None) -> dict:
        suggested_exp = payload.get("suggested_exp", 15)

        if not self.api_key:
            return {
                "decision": "YES",
                "confidence": 7,
                "suggested_expiration": suggested_exp,
                "reason": "GEMINI_API_KEY не налаштовано",
            }

        recent_history = database.get_recent_trades_summary(limit=6)

        default_prompt = f"""
        Ти професійний трейдер та ризик-менеджер. Проаналізуй ринкові дані та графіки для активу {name}.
        Сигнал сформовано за матрицею 45 стратегій: {payload.get('strategy_title')} (Підтверджень: {payload.get('confluence_count', 1)}).

        ІСТОРІЯ УГОД ТА ВИСНОВКИ ШІ (Для самонавчання):
        {recent_history}
        
        Параметри поточного сигналу:
        - Напрямок: {payload.get('signal')}
        - Поточна ціна: {payload.get('current_price')}
        - RSI: {payload.get('rsi')}
        - ADX: {payload.get('adx')}
        - Глобальний тренд (1h): {payload.get('global_trend')}
        - Технічна причина: {payload.get('reason')}
        - Рекомендована експірація: {suggested_exp} хв

        Завдання:
        1. Оціни доцільність входу ("YES" або "NO").
        2. Вкажи динамічний час експірації у хвилинах ВІД 1 ДО 60 ХВИЛИН.
        3. Підтверджуй вхід ("YES" з confidence >= 5), якщо немає критичних ризиків.

        Відповідь надай ВИКЛЮЧНО у форматі JSON без жодних додаткових символів чи markdown:
        {{
            "decision": "YES" або "NO",
            "confidence": <число від 1 до 10>,
            "suggested_expiration": <число від 1 до 60>,
            "reason": "<коротке обґрунтування українською>"
        }}
        """

        prompt = database.get_system_prompt("signal_evaluation_prompt", default_prompt)
        content_parts = [prompt]
        for chart in [macro_chart, mid_chart, micro_chart]:
            if chart:
                try:
                    chart.seek(0)
                    content_parts.append(Image.open(chart))
                except Exception:
                    pass

        models_to_try = self.get_available_models()
        response_text = None

        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(content_parts, request_options={"timeout": 12})
                if response and response.text:
                    response_text = response.text
                    break
            except Exception:
                continue

        if not response_text:
            return {
                "decision": "YES",
                "confidence": 6,
                "suggested_expiration": suggested_exp,
                "reason": "ШІ зайнятий, авто-схвалення за матрицею 45",
            }

        try:
            clean_text = response_text.strip()
            if clean_text.startswith("```json"): clean_text = clean_text[7:]
            if clean_text.startswith("```"): clean_text = clean_text[3:]
            if clean_text.endswith("```"): clean_text = clean_text[:-3]
            clean_text = clean_text.strip()

            result = json.loads(clean_text)
            parsed_exp = int(result.get("suggested_expiration", suggested_exp))
            parsed_exp = max(1, min(60, parsed_exp))

            return {
                "decision": str(result.get("decision", "NO")).upper(),
                "confidence": int(result.get("confidence", 0)),
                "suggested_expiration": parsed_exp,
                "reason": str(result.get("reason", "Аналіз виконано")),
            }
        except Exception as e:
            logger.error(f"Помилка парсингу ШІ: {e}")
            return {
                "decision": "YES",
                "confidence": 6,
                "suggested_expiration": suggested_exp,
                "reason": "Помилка парсингу ШІ, вхід за ТА",
            }

    def evaluate_closed_trade(self, sig_data: dict) -> str:
        if not self.api_key:
            return "❌ GEMINI_API_KEY не налаштовано."

        default_review_prompt = f"""
        Проведи ретроспективний розбір закритої угоди (матриця 45 стратегій):
        - Актив: {sig_data.get('ticker')} | Стратегія: {sig_data.get('strategy')}
        - Напрямок: {sig_data.get('signal_type')} | Вхід: {sig_data.get('entry_price')} | Вихід: {sig_data.get('exit_price')}
        - Результат: {sig_data.get('result')} ({sig_data.get('pips')} п.) | Експірація: {sig_data.get('expiration_mins')} хв
        - RSI: {sig_data.get('rsi')} | ADX: {sig_data.get('adx')}

        Дай короткий професійний аналіз українською: чому так сталося і що врахувати в майбутньому (3-4 речення).
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


def analyze_signal_with_gemini(symbol, payload, chart_images=[]):
    c_macro = chart_images[0] if len(chart_images) > 0 else None
    c_mid = chart_images[1] if len(chart_images) > 1 else None
    c_micro = chart_images[2] if len(chart_images) > 2 else None
    return ai_advisor_instance.evaluate_signal(symbol, payload, c_macro, c_mid, c_micro)
