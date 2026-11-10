# -*- coding: utf-8 -*-
import os
import logging
import json
import re
import time
import threading
from collections import deque
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    genai = None
    HAS_GENAI = False

PREFERRED_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-lite"]
SKIP_MODEL_WORDS = ("image", "tts", "live", "embedding", "aqa", "thinking", "exp", "learnlm", "gemma", "robotics", "audio")


class RateLimiter:
    """Ковзне вікно: не більше max_calls за 60 с (безкоштовний Gemini ≈ 10–15 запитів/хв на модель)."""
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self._calls = deque()
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        now = time.time()
        with self._lock:
            while self._calls and now - self._calls[0] > 60:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True


def classify_error(e: Exception) -> tuple:
    """Повертає (код, людський опис, секунд охолодження моделі)."""
    name, text = type(e).__name__, str(e)
    low = text.lower()
    if "ResourceExhausted" in name or "429" in text or "quota" in low or "rate limit" in low:
        m = re.search(r"retry in ([\d.]+)s", low)
        wait = int(float(m.group(1))) + 1 if m else 60
        return "quota", "ліміт запитів Gemini (429)", max(30, min(wait, 600))
    if "NotFound" in name or "404" in text or "is not found" in low:
        return "notfound", "модель недоступна (404)", 3600
    if "PermissionDenied" in name or "Unauthenticated" in name or "API key" in text or "403" in text or "401" in text:
        return "auth", "проблема з API-ключем (401/403)", 600
    if "DeadlineExceeded" in name or "timeout" in low or "timed out" in low:
        return "timeout", "таймаут відповіді", 20
    if "ServiceUnavailable" in name or "503" in text or "overloaded" in low:
        return "overload", "модель перевантажена (503)", 30
    return "other", f"{name}: {text[:80]}", 30


class AIAdvisor:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY")
        if self.api_key and HAS_GENAI:
            genai.configure(api_key=self.api_key.strip())
        self.rate = RateLimiter(int(os.environ.get("AI_MAX_CALLS_PER_MIN", "8")))
        self.timeout = int(os.environ.get("AI_TIMEOUT", "25"))
        self.fail_mode = os.environ.get("AI_FAIL_MODE", "open").lower()  # open: пропустити без ШІ, closed: відхилити
        self._cooldown = {}      # model -> unix ts, до якого модель не чіпаємо
        self.last_status = "ще не викликався"

    def get_available_models(self) -> list:
        """Пріоритетні текстові/візуальні Gemini-моделі (без image/tts/live/exp), що реально є в акаунті."""
        if not HAS_GENAI or not self.api_key:
            return PREFERRED_MODELS[:1]
        cached = getattr(self, "_models_cache", None)
        if cached and time.time() - cached[0] < 3600:
            return cached[1]
        env_list = [m.strip() for m in os.environ.get("AI_MODELS", "").split(",") if m.strip()]
        wanted = env_list or PREFERRED_MODELS
        try:
            avail = []
            for m in genai.list_models():
                if "generateContent" in m.supported_generation_methods:
                    avail.append(m.name.replace("models/", ""))
            chosen = [m for m in wanted if m in avail]
            if not chosen:
                chosen = [m for m in sorted(avail)
                          if m.startswith("gemini") and ("flash" in m or "pro" in m)
                          and not any(w in m for w in SKIP_MODEL_WORDS)]
                chosen.sort(key=lambda n: (0 if "flash" in n else 1, n))
            chosen = chosen[:3] or wanted
            self._models_cache = (time.time(), chosen)
            return chosen
        except Exception as e:
            logger.warning(f"⚠️ Не вдалося отримати список моделей через API: {e}")
            return wanted

    def evaluate_signal(self, pair_name: str, payload: dict, chart_macro, chart_mid, chart_micro) -> dict:
        suggested_exp = int(payload.get("suggested_exp", 10))

        def unavailable(reason: str) -> dict:
            self.last_status = reason
            logger.warning(f"⚠️ ШІ недоступний: {reason}")
            return {
                "decision": "NO" if self.fail_mode == "closed" else "YES",
                "confidence": 7,
                "suggested_expiration": suggested_exp,
                "ai_available": False,
                "reason": f"ШІ недоступний ({reason}) — оцінку ШІ не застосовано",
            }

        if not self.api_key or not HAS_GENAI:
            return unavailable("відсутній ключ або бібліотека")

        macro_tf, confirm_tf, primary_tf = payload.get("chart_tfs", ("1h", "15m", payload.get("tf", "3m")))
        exp_lo, exp_hi = payload.get("exp_range", (1, 60))

        system_instruction = (
            "Ти — професійний головний трейдер та ризик-менеджер, який спеціалізується виключно на **БІНАРНИХ ОПЦІОНАХ (Binary Options)**.\n"
            "Твоє завдання — оцінити торговий сигнал з погляду точності мікро-входу, відсутності ринкового шуму та адаптивного часу експірації.\n"
            f"Ти аналізуєш три таймфрейми ({macro_tf} макро-тренд, {confirm_tf} підтвердження, {primary_tf} точка входу та свічкові тіні).\n"
            "Враховуй контекст попередніх помилок по цьому активу, якщо він переданий, щоб уникнути повторних хибних входів.\n"
            f"Базовий адаптивний час експірації вже розрахований системою; дозволена корекція лише від {exp_lo} до {exp_hi} хвилин.\n"
            "Відповідь ти зобов'язаний повернути **ВИКЛЮЧНО у форматі валідного JSON** без жодних додаткових символів чи markdown-обгорток (на кшталт ```json ... ```), у такому форматі:\n"
            "{\n"
            '  "decision": "YES" або "NO",\n'
            '  "confidence": ціле число від 1 до 10,\n'
            f'  "suggested_expiration": рекомендований час експірації в хвилинах (ціле число від {exp_lo} до {exp_hi}),\n'
            '  "reason": "Коротке обґрунтування українською мовою з акцентом на мікро-імпульс, волатильність та свічкові тіні"\n'
            "}"
        )

        prompt = (
            f"Аналіз активу: {pair_name}\n"
            f"Сигнал двигуна: {payload.get('signal')} (Стратегія: {payload.get('strategy_title')})\n"
            f"Робочий таймфрейм сигналу: {payload.get('tf', '1m')} | Незалежних підтверджень: {payload.get('confirmations', 'н/д')}\n"
            f"Поточна ціна: {payload.get('current_price')}\n"
            f"RSI ({primary_tf}): {payload.get('rsi')} | ADX: {payload.get('adx')} | Базовий адаптивний час експірації: {suggested_exp} хв\n"
            f"Причина від двигуна: {payload.get('reason')}\n"
            f"Останні невдалі угоди по парі (для саморефлексії): {payload.get('recent_mistakes', 'Немає')}\n\n"
            f"Проаналізуй прикріплені закриті графіки ({macro_tf}, {confirm_tf}, {primary_tf}) та прийми рішення щодо входу та експірації {exp_lo}–{exp_hi} хв."
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

        if not self.rate.try_acquire():
            return unavailable("перевищено локальний ліміт запитів/хв")

        models_to_try = self.get_available_models()
        response_text = None
        errors = []
        deadline = time.time() + self.timeout + 10

        for model_name in models_to_try:
            if time.time() > deadline:
                break
            if self._cooldown.get(model_name, 0) > time.time():
                errors.append(f"{model_name}: охолодження")
                continue
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(content_parts, request_options={"timeout": self.timeout})
                if response and response.text:
                    response_text = response.text
                    self.last_status = f"ОК ({model_name})"
                    break
                errors.append(f"{model_name}: порожня відповідь")
            except Exception as e:
                code, human, cool = classify_error(e)
                self._cooldown[model_name] = time.time() + cool
                errors.append(f"{model_name}: {human}")
                logger.warning(f"⚠️ Модель {model_name} не відповіла — {human}: {str(e)[:200]}")
                if code == "auth":
                    break
                continue

        if not response_text:
            return unavailable("; ".join(errors[:3]) or "жодна модель не відповіла")

        try:
            clean_text = response_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(clean_text)
            if not isinstance(parsed, dict) or "decision" not in parsed or "confidence" not in parsed:
                return unavailable("відсутнє рішення або впевненість")
            decision = str(parsed["decision"]).strip().upper()
            confidence = int(parsed["confidence"])
            if decision not in ("YES", "NO") or not 1 <= confidence <= 10:
                return unavailable("некоректне рішення або впевненість")
            parsed_exp = int(parsed.get("suggested_expiration", suggested_exp))
            parsed_exp = max(exp_lo, min(exp_hi, parsed_exp))

            return {
                "decision": decision,
                "confidence": confidence,
                "suggested_expiration": parsed_exp,
                "ai_available": True,
                "reason": str(parsed.get("reason", "ШІ підтвердив вхід"))
            }
        except Exception as e:
            logger.error(f"⚠️ Помилка парсингу відповіді ШІ: {e}")
            return unavailable("неочікуваний формат відповіді")

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

        if not self.rate.try_acquire():
            return "⚠️ Ліміт запитів ШІ на хвилину вичерпано, спробуйте пізніше."
        reasons = []
        for model_name in self.get_available_models():
            if self._cooldown.get(model_name, 0) > time.time():
                reasons.append(f"{model_name}: охолодження")
                continue
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt, request_options={"timeout": self.timeout})
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                code, human, cool = classify_error(e)
                self._cooldown[model_name] = time.time() + cool
                reasons.append(f"{model_name}: {human}")
        return "⚠️ Не вдалося отримати звіт від ШІ: " + ("; ".join(reasons[:3]) or "помилка API")

ai_advisor_instance = AIAdvisor()
