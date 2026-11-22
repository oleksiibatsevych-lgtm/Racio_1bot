# -*- coding: utf-8 -*-
import logging
import os
import re
import threading
import time
from collections import deque

import requests

from config import normalize_pair_name

logger = logging.getLogger(__name__)

try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    HAS_NEW_GENAI = True
except ImportError:
    google_genai = None
    genai_types = None
    HAS_NEW_GENAI = False

try:
    import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    legacy_genai = None
    HAS_LEGACY_GENAI = False

PREFERRED_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]
SKIP_MODEL_WORDS = ("image", "tts", "live", "embedding", "aqa", "thinking", "exp", "learnlm", "gemma", "robotics", "audio")


class RateLimiter:
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
        self.api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_KEY") or "").strip() or None
        self.openrouter_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip() or None
        self.openrouter_model = (os.environ.get("OPENROUTER_MODEL") or "").strip()
        self.rate = RateLimiter(int(os.environ.get("AI_MAX_CALLS_PER_MIN", "8")))
        self.timeout = int(os.environ.get("AI_TIMEOUT", "25"))
        self._cooldown = {}
        self.last_status = "ще не викликався"
        self._client = None

        if self.api_key:
            if HAS_NEW_GENAI:
                self._client = google_genai.Client(api_key=self.api_key)
            elif HAS_LEGACY_GENAI:
                legacy_genai.configure(api_key=self.api_key)

    def get_available_models(self) -> list:
        if not self.api_key:
            return PREFERRED_MODELS[:1]
        cached = getattr(self, "_models_cache", None)
        if cached and time.time() - cached[0] < 3600:
            return cached[1]
        env_list = [m.strip() for m in os.environ.get("AI_MODELS", "").split(",") if m.strip()]
        wanted = env_list or PREFERRED_MODELS
        try:
            avail = []
            if HAS_NEW_GENAI and self._client is not None:
                for m in self._client.models.list():
                    name = (getattr(m, "name", "") or "").replace("models/", "")
                    actions = getattr(m, "supported_actions", None) or []
                    if name and (not actions or "generateContent" in actions):
                        avail.append(name)
            elif HAS_LEGACY_GENAI:
                for m in legacy_genai.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        avail.append(m.name.replace("models/", ""))
            chosen = [m for m in wanted if m in avail]
            if not chosen and avail:
                chosen = [
                    m for m in sorted(avail)
                    if m.startswith("gemini") and ("flash" in m or "pro" in m)
                    and not any(w in m for w in SKIP_MODEL_WORDS)
                ]
                chosen.sort(key=lambda n: (0 if "3.8" in n else (1 if "flash" in n else 2), n))
            chosen = chosen[:3] or wanted
            self._models_cache = (time.time(), chosen)
            return chosen
        except Exception as e:
            logger.warning(f"⚠️ Не вдалося отримати список моделей Gemini: {e}")
            return wanted

    def _call_openrouter_fallback(self, prompt: str) -> str | None:
        if not self.openrouter_key or not self.openrouter_model:
            return None
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.openrouter_model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 350},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                txt = data["choices"][0]["message"]["content"]
                if txt:
                    self.last_status = f"OK via OpenRouter ({self.openrouter_model})"
                    return txt.strip()
        except Exception as e:
            logger.warning(f"OpenRouter fallback помилка: {e}")
        return None

    def _generate_text(self, prompt: str) -> str:
        if not self.rate.try_acquire():
            self.last_status = "rate_limited_local"
            return "⚠️ Ліміт запитів ШІ на хвилину вичерпано, спробуйте пізніше."

        reasons = []
        if self.api_key and (HAS_NEW_GENAI or HAS_LEGACY_GENAI):
            for model_name in self.get_available_models():
                if self._cooldown.get(model_name, 0) > time.time():
                    reasons.append(f"{model_name}: охолодження")
                    continue
                try:
                    if HAS_NEW_GENAI and self._client is not None:
                        cfg_kwargs = {"max_output_tokens": 360}
                        if hasattr(genai_types, "ThinkingConfig") and model_name.startswith("gemini-3"):
                            cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level="LOW")
                        response = self._client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=genai_types.GenerateContentConfig(**cfg_kwargs),
                        )
                    else:
                        model = legacy_genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt, request_options={"timeout": self.timeout})

                    text = getattr(response, "text", None)
                    if text:
                        self.last_status = f"OK ({model_name})"
                        return text.strip()
                except Exception as e:
                    code, human, cool = classify_error(e)
                    self._cooldown[model_name] = time.time() + cool
                    reasons.append(f"{model_name}: {human}")
                    self.last_status = f"err:{code}:{model_name}"

        or_reply = self._call_openrouter_fallback(prompt)
        if or_reply:
            return or_reply

        return "⚠️ Не вдалося отримати звіт від ШІ: " + ("; ".join(reasons[:3]) or "ШІ не налаштовано")

    def evaluate_closed_trade(self, sig_data: dict) -> str:
        if (not self.api_key and not (self.openrouter_key and self.openrouter_model)) or not sig_data:
            self.last_status = "disabled_or_no_data"
            return "ШІ вимкнено або даних недостатньо."

        pair = normalize_pair_name(sig_data.get("ticker")) or sig_data.get("ticker")
        is_jpy = "JPY" in str(pair)
        pips_raw = sig_data.get("pips")
        entry_p = sig_data.get("entry_price")
        exit_p = sig_data.get("exit_price")
        if entry_p is not None and exit_p is not None:
            mult = 1000.0 if is_jpy else 100000.0
            diff = float(exit_p) - float(entry_p)
            if str(sig_data.get("signal_type", "")).upper() == "PUT":
                diff = -diff
            pts_str = f"{diff * mult:+.1f} пунктів БО ({diff * mult / 10:+.1f} піпсів)"
        elif pips_raw is not None:
            pts_str = f"{float(pips_raw):+.1f} пунктів БО"
        else:
            pts_str = "невідомо"

        context_lines = []
        for label, key in [
            ("CHOP", "chop"),
            ("ADX", "adx"),
            ("RSI", "rsi"),
            ("Сесія", "session"),
            ("HTF 15m/60m", "htf_summary"),
            ("Рівні S/R", "sr_context"),
            ("Траєкторія під час угоди (MFE/MAE)", "price_path_summary"),
            ("Новини", "news_note"),
        ]:
            val = sig_data.get(key)
            if val is not None and str(val).strip():
                context_lines.append(f"{label}: {val}")

        ctx_block = "\n".join(context_lines) if context_lines else "Додаткові ринкові метрики (свічки/S&R/новини) не передано."

        prompt = (
            "Проведи стислий технічний розбір закритої угоди на бінарних опціонах (3–4 речення українською).\n"
            f"Пара: {pair} | Напрямок: {sig_data.get('signal_type')}\n"
            f"Вхід: {entry_p} -> Вихід: {exit_p}\n"
            f"Результат: {sig_data.get('result')} ({pts_str})\n"
            f"Експірація: {sig_data.get('expiration_mins')} хв | Внутрішній код: {sig_data.get('strategy')}\n"
            f"{ctx_block}\n\n"
            "Правила: спирайся виключно на наведені цифри. Якщо немає траєкторії свічок, S&R чи новин, прямо вкажи, яких даних бракує для висновку про таймінг чи рівень. Не вигадуй причини й не обіцяй прибутковість."
        )
        return self._generate_text(prompt)

    def evaluate_journal_batch(self, batch_summary: dict) -> str:
        if (not self.api_key and not (self.openrouter_key and self.openrouter_model)) or not batch_summary:
            return "Недостатньо даних або ШІ вимкнено."
        prompt = (
            "Проаналізуй зведену статистику журналу закритих угод бінарного бота (5–6 коротких пунктів українською).\n"
            f"Зведені дані:\n{batch_summary}\n\n"
            "Виділи: 1) найслабші пари або сесії, 2) ознаки невідповідності режиму CHOP/ADX, "
            "3) практичні рекомендації для налаштування фільтрів без перенавчання."
        )
        return self._generate_text(prompt)


ai_advisor_instance = AIAdvisor()
