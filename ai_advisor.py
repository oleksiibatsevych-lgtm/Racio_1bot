# -*- coding: utf-8 -*-
import os
import logging
import re
import time
import threading
from collections import deque

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
            "Напиши стислий огляд українською (2-3 речення). Це лише метадані: немає свічок, новин чи ринкової структури. Не вигадуй причини результату, не оцінюй правильність експірації та не обіцяй прибутковість. Назви обмеження даних."
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
