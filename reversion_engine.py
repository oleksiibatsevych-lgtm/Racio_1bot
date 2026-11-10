import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from trading_sessions import session_allowed, session_name

logger = logging.getLogger(__name__)

PRESETS = {
    "REV_RSI14": 60,
    "REV_BB3": 60,
    "REV_RSI7": 45,
}


def is_reversion(cand):
    return cand.get("strategy") in PRESETS


def load_strategy_config(path):
    defaults = dict.fromkeys(PRESETS, True)
    defaults["session_filter"] = False
    if not Path(path).exists():
        return defaults
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        return {key: saved.get(key, value) if isinstance(saved.get(key, value), bool) else value
                for key, value in defaults.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        logger.exception("Помилка читання конфігурації стратегій")
        return defaults


def calculate_indicators(df):
    df = df.copy()
    delta = df["close"].diff()
    for length in (7, 14):
        up = delta.clip(lower=0).ewm(alpha=1 / length, min_periods=length).mean()
        down = (-delta).clip(lower=0).ewm(alpha=1 / length, min_periods=length).mean()
        df[f"rsi{length}"] = 100 * up / (up + down)
    middle = df["close"].rolling(20).mean()
    spread = 3 * df["close"].rolling(20).std(ddof=0)
    df["bb_lower"] = middle - spread
    df["bb_upper"] = middle + spread
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    if len(tr):
        tr.iloc[0] = np.nan
    df["atr"] = tr.ewm(alpha=1 / 14, min_periods=14).mean()
    conditions = {
        "REV_RSI14": (df["rsi14"] < 20, df["rsi14"] > 80),
        "REV_BB3": ((df["close"] < df["bb_lower"]) & (df["rsi14"] < 30),
                    (df["close"] > df["bb_upper"]) & (df["rsi14"] > 70)),
        "REV_RSI7": (df["rsi7"] < 15, df["rsi7"] > 85),
    }
    for name, (call, put) in conditions.items():
        df[name] = np.where(call, 1, np.where(put, -1, 0))
        df.iloc[:200, df.columns.get_loc(name)] = 0
    return df


def candidates_from_indicators(pair_name, closed, now, strategies_status=None, pair_enabled=None):
    if closed is None or len(closed) < 201:
        return []
    now = pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    bar_end = closed.index[-1] + pd.Timedelta(minutes=5)
    age = (now - bar_end).total_seconds()
    if not 0 <= age <= 180:
        return []
    status = strategies_status if strategies_status is not None else {}
    last = closed.iloc[-1]
    for strategy, expiration in PRESETS.items():
        if not status.get(strategy, True) or not last[strategy]:
            continue
        if status.get("session_filter", False) and not session_allowed(strategy, bar_end):
            continue
        if pair_enabled is not None and not pair_enabled(pair_name, strategy):
            continue
        return [dict(pair=pair_name, tf="5m", tf_minutes=5, strategy=strategy,
                     strategy_title=strategy, signal="CALL" if last[strategy] > 0 else "PUT",
                     current_price=float(last["close"]), atr=float(last["atr"]),
                     rsi=float(last["rsi7" if strategy == "REV_RSI7" else "rsi14"]),
                     bar_id=f"{pair_name}|5m|{closed.index[-1]}", bar_end=bar_end,
                     bar_age_sec=age, exp_range=(expiration, expiration), suggested_exp=expiration,
                     session=session_name(bar_end),
                     confirmations=0, confirmation_names=[], regime="REVERSION",
                     reason=f"{strategy}: відбій 5m, фіксована експірація {expiration} хв")]
    return []


def generate_candidates(pair_name, frames, now, strategies_status=None, pair_enabled=None):
    frame = frames.get("5m")
    if frame is None or frame.empty:
        return []
    now = pd.Timestamp(now)
    if now.tzinfo is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    frame = frame.copy()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    closed = frame.loc[frame.index + pd.Timedelta(minutes=5) <= now]
    return candidates_from_indicators(pair_name, calculate_indicators(closed), now,
                                      strategies_status, pair_enabled)


class ReversionGuard:
    def __init__(self):
        self._lock = threading.RLock()
        self._open = {}

    @contextmanager
    def reserve(self, pair, strategy, ticker, has_open_signal):
        with self._lock:
            if self._open.get((pair, strategy), 0) > time.time():
                yield False, "Попередній сигнал цього пресету ще відкритий"
                return
            try:
                pending = has_open_signal(ticker, strategy)
            except Exception:
                logger.exception("Не вдалося перевірити відкриті REV-сигнали")
                yield False, "Не вдалося перевірити відкриті сигнали в БД"
                return
            yield (False, "Попередній сигнал цього пресету ще відкритий") if pending else (True, "OK")

    def register(self, pair, strategy, expiration):
        with self._lock:
            self._open[(pair, strategy)] = time.time() + expiration * 60
