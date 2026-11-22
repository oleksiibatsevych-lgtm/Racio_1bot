# -*- coding: utf-8 -*-
"""
Quality gate — прибирає дублікати, конфлікти напрямків, надмірну валютну експозицію
та статистично програшні сегменти (верхня межа Wilson < break_even).
"""
import logging
import math
import threading
import time

from config import MAX_ACTIVE_PER_CURRENCY

logger = logging.getLogger(__name__)


def split_pair(pair_name: str):
    base, quote = pair_name.split("/")
    return base, quote


def wilson_bounds(wins: int, total: int, z: float = 1.96):
    if total <= 0:
        return 0.0, 1.0
    p = wins / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


class SignalGate:
    def __init__(
        self,
        cooldown_factor: float = 1.5,
        min_cooldown_min: int = 3,
        conflict_window_min: int = 3,
        max_ccy_exposure: int = MAX_ACTIVE_PER_CURRENCY,
        exposure_window_min: int = 10,
    ):
        self.cooldown_factor = cooldown_factor
        self.min_cooldown_min = min_cooldown_min
        self.conflict_window_min = conflict_window_min
        self.max_ccy_exposure = max(1, int(max_ccy_exposure))
        self.exposure_window_min = exposure_window_min
        self._lock = threading.Lock()
        self._seen_bars = {}
        self._last_by_pair_tf = {}
        self._recent = []

    def _cleanup(self, now_ts: float):
        horizon = now_ts - 3600
        self._seen_bars = {k: v for k, v in self._seen_bars.items() if v > horizon}
        keep = now_ts - max(self.exposure_window_min, self.conflict_window_min) * 60
        self._recent = [r for r in self._recent if r[0] > keep or r[3] > now_ts]

    def _ccy_exposure(self, now_ts: float):
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

    def reset(self):
        with self._lock:
            self._seen_bars.clear()
            self._last_by_pair_tf.clear()
            self._recent.clear()

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
    """Блокує сегменти, у яких верхня межа Wilson нижча за точку беззбитковості."""

    def __init__(self, db_module, break_even=0.5405, min_trades: int = 40, ttl_sec: int = 1800):
        self.db = db_module
        self.break_even = break_even
        self.min_trades = min_trades
        self.ttl = ttl_sec
        self._ts = 0.0
        self._segments = {}
        self._hours = {}
        self._lock = threading.Lock()

    def _current_breakeven(self) -> float:
        try:
            val = self.break_even() if callable(self.break_even) else self.break_even
            return float(val)
        except Exception:
            return 0.5405

    def _refresh(self):
        with self._lock:
            now = time.time()
            if now - self._ts < self.ttl:
                return
            try:
                seg = self.db.get_segment_stats()
                new_segments = {(r["ticker"], r["tf"], r["strategy"]): (int(r["wins"]), int(r["total"])) for r in seg}
                hours = {}
                for r in self.db.get_strategy_time_stats().get("hours", []):
                    h = int(r["trade_hour"])
                    w, t = hours.get(h, (0, 0))
                    hours[h] = (w + int(r["wins"]), t + int(r["total"]))
                self._segments = new_segments
                self._hours = hours
                self._ts = now
            except Exception as e:
                self._ts = now - self.ttl + 60.0
                logger.warning(f"SegmentFilter: не вдалося оновити статистику: {e}")

    def _is_bad(self, wins: int, total: int, be: float) -> bool:
        if total < self.min_trades:
            return False
        _lo, hi = wilson_bounds(wins, total)
        return hi < be

    def check(self, ticker: str, tf: str, strategy: str, hour_utc: int) -> tuple:
        self._refresh()
        be = self._current_breakeven()
        with self._lock:
            w, t = self._segments.get((ticker, tf, strategy), (0, 0))
            hw, ht = self._hours.get(hour_utc, (0, 0))
        if self._is_bad(w, t, be):
            return False, f"Сегмент {strategy}/{tf} програшний ({w}/{t})"
        if self._is_bad(hw, ht, be):
            return False, f"Година {hour_utc:02d}:00 UTC програшна ({hw}/{ht})"
        return True, "OK"
