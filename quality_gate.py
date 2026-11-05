# -*- coding: utf-8 -*-
"""
Quality gate — прибирає ДУБЛІКАТИ та ПРОВЕДЕНО-ПРОГРАШНІ сегменти, а не «хороші» сигнали.

  1. bar_id            — одна закрита свічка = максимум один сигнал.
  2. cooldown          — пауза на (пара, ТФ) після сигналу.
  3. конфлікт напрямків — протилежний сигнал по тій самій парі у короткому вікні блокується.
  4. експозиція валюти — не більше N однонаправлених ставок на одну валюту у вікні
                         (EUR/USD, EUR/JPY, EUR/GBP CALL = одна й та сама ставка на EUR).
  5. SegmentFilter     — блокує лише сегменти (пара×ТФ×стратегія, година UTC), де статистика
                         ПЕВНО нижче беззбитковості (верхня межа Wilson < порогу). Без даних — пропускає.
"""
import logging
import math
import threading
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def split_pair(pair_name: str):
    base, quote = pair_name.split("/")
    return base, quote


def wilson_bounds(wins: int, total: int, z: float = 1.96):
    """Довірчий інтервал Wilson для вінрейту."""
    if total <= 0:
        return 0.0, 1.0
    p = wins / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


class SignalGate:
    def __init__(self, cooldown_factor: float = 1.5, min_cooldown_min: int = 3,
                 conflict_window_min: int = 3, max_ccy_exposure: int = 3,
                 exposure_window_min: int = 10):
        self.cooldown_factor = cooldown_factor
        self.min_cooldown_min = min_cooldown_min
        self.conflict_window_min = conflict_window_min
        self.max_ccy_exposure = max_ccy_exposure
        self.exposure_window_min = exposure_window_min
        self._lock = threading.Lock()
        self._seen_bars = {}      # bar_id -> ts
        self._last_by_pair_tf = {}  # (pair, tf) -> (ts, direction)
        self._recent = []         # [(ts, pair, direction, exp_end_ts)]

    # ---- внутрішнє ----
    def _cleanup(self, now_ts: float):
        horizon = now_ts - 3600
        self._seen_bars = {k: v for k, v in self._seen_bars.items() if v > horizon}
        keep = now_ts - max(self.exposure_window_min, self.conflict_window_min) * 60
        self._recent = [r for r in self._recent if r[0] > keep or r[3] > now_ts]

    def _ccy_exposure(self, now_ts: float):
        """Скільки відкритих/свіжих ставок у кожному напрямку на кожну валюту."""
        exp = {}
        window = now_ts - self.exposure_window_min * 60
        for ts, pair, direction, end_ts in self._recent:
            if ts < window and end_ts < now_ts:
                continue
            base, quote = split_pair(pair)
            sign = 1 if direction == "CALL" else -1
            exp[(base, sign)] = exp.get((base, sign), 0) + 1
            exp[(quote, -sign)] = exp.get((quote, -sign), 0) + 1
        return exp

    # ---- публічне ----
    def check(self, cand: dict, now_ts: float = None) -> tuple:
        now_ts = now_ts or time.time()
        pair, tf, direction = cand["pair"], cand["tf"], cand["signal"]
        with self._lock:
            self._cleanup(now_ts)

            if cand["bar_id"] in self._seen_bars:
                return False, "Ця свічка вже дала сигнал"

            last = self._last_by_pair_tf.get((pair, tf))
            if last:
                cd = max(self.min_cooldown_min, cand["tf_minutes"] * self.cooldown_factor) * 60
                if now_ts - last[0] < cd:
                    return False, f"Cooldown {pair} {tf}"

            for ts, p, d, _end in self._recent:
                if p == pair and d != direction and now_ts - ts < self.conflict_window_min * 60:
                    return False, f"Конфлікт напрямків по {pair}"

            base, quote = split_pair(pair)
            sign = 1 if direction == "CALL" else -1
            exposure = self._ccy_exposure(now_ts)
            for ccy, s in ((base, sign), (quote, -sign)):
                if exposure.get((ccy, s), 0) >= self.max_ccy_exposure:
                    side = "довга" if s == 1 else "коротка"
                    return False, f"Ліміт експозиції: {ccy} ({side}) вже {exposure[(ccy, s)]} ставок"
        return True, "OK"

    def register(self, cand: dict, expiration_mins: int, now_ts: float = None):
        now_ts = now_ts or time.time()
        with self._lock:
            self._seen_bars[cand["bar_id"]] = now_ts
            self._last_by_pair_tf[(cand["pair"], cand["tf"])] = (now_ts, cand["signal"])
            self._recent.append((now_ts, cand["pair"], cand["signal"], now_ts + expiration_mins * 60))


class SegmentFilter:
    """
    Блокує сегменти, у яких вінрейт СТАТИСТИЧНО значущо нижчий за беззбитковість.
    break_even — вінрейт беззбитковості (для виплати 85% ≈ 0.54).
    """
    def __init__(self, db_module, break_even: float = 0.50, min_trades: int = 40, ttl_sec: int = 1800):
        self.db = db_module
        self.break_even = break_even
        self.min_trades = min_trades
        self.ttl = ttl_sec
        self._ts = 0.0
        self._segments = {}   # (ticker, tf, strategy) -> (wins, total)
        self._hours = {}      # hour -> (wins, total)
        self._lock = threading.Lock()

    def _refresh(self):
        with self._lock:
            if time.time() - self._ts < self.ttl:
                return
            self._ts = time.time()
            try:
                seg = self.db.get_segment_stats()
                self._segments = {(r["ticker"], r["tf"], r["strategy"]): (int(r["wins"]), int(r["total"])) for r in seg}
                hours = {}
                for r in self.db.get_strategy_time_stats().get("hours", []):
                    h = int(r["trade_hour"])
                    w, t = hours.get(h, (0, 0))
                    hours[h] = (w + int(r["wins"]), t + int(r["total"]))
                self._hours = hours
            except Exception as e:
                logger.warning(f"SegmentFilter: не вдалося оновити статистику: {e}")

    def _is_bad(self, wins, total):
        if total < self.min_trades:
            return False
        _lo, hi = wilson_bounds(wins, total)
        return hi < self.break_even

    def check(self, ticker: str, tf: str, strategy: str, hour_utc: int) -> tuple:
        self._refresh()
        w, t = self._segments.get((ticker, tf, strategy), (0, 0))
        if self._is_bad(w, t):
            return False, f"Сегмент {strategy}/{tf} програшний ({w}/{t})"
        w, t = self._hours.get(hour_utc, (0, 0))
        if self._is_bad(w, t):
            return False, f"Година {hour_utc:02d}:00 UTC програшна ({w}/{t})"
        return True, "OK"
