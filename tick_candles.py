# -*- coding: utf-8 -*-
"""Збирач 1-хвилинних свічок із живих тиків Finnhub WebSocket із коректним упорядкуванням."""
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone

import pandas as pd

MAX_BARS = 3000
MAX_FILL_GAP_MIN = 10

_lock = threading.Lock()
_bars = {}


def add_tick(symbol: str, price: float, ts_ms=None):
    if not symbol or price is None:
        return
    try:
        p = float(price)
    except (TypeError, ValueError):
        return
    if p <= 0:
        return
    ts = (float(ts_ms) / 1000.0) if ts_ms is not None else time.time()
    key = int(ts // 60)
    with _lock:
        d = _bars.setdefault(symbol, OrderedDict())
        bar = d.get(key)
        if bar is None:
            prev_last_key = next(reversed(d.keys())) if d else None
            d[key] = [p, p, p, p, 1]
            if prev_last_key is not None and key < prev_last_key:
                _bars[symbol] = OrderedDict(sorted(d.items()))
                d = _bars[symbol]
            while len(d) > MAX_BARS:
                d.popitem(last=False)
        else:
            if p > bar[1]:
                bar[1] = p
            if p < bar[2]:
                bar[2] = p
            bar[3] = p
            bar[4] += 1


def _minute_to_dt(key: int) -> datetime:
    return datetime.fromtimestamp(key * 60, timezone.utc).replace(tzinfo=None)


def get_1m(symbol: str, now_ts: float = None) -> pd.DataFrame:
    if now_ts is None:
        now_ts = time.time()
    now_key = int(now_ts // 60)
    with _lock:
        d = _bars.get(symbol)
        items = sorted(d.items()) if d else []
    if not items:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    rows = {}
    prev_key, prev_close = None, None
    for key, (o, h, l, c, n) in items:
        if prev_key is not None and 1 < key - prev_key <= MAX_FILL_GAP_MIN + 1:
            for k in range(prev_key + 1, key):
                rows[k] = (prev_close, prev_close, prev_close, prev_close, 0)
        rows[key] = (o, h, l, c, n)
        prev_key, prev_close = key, c
    if prev_key is not None and 0 < now_key - prev_key <= MAX_FILL_GAP_MIN:
        for k in range(prev_key + 1, now_key + 1):
            rows[k] = (prev_close, prev_close, prev_close, prev_close, 0)

    keys = sorted(rows)
    return pd.DataFrame(
        [rows[k] for k in keys],
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([_minute_to_dt(k) for k in keys]),
    )


def merge_with_history(
    symbol: str,
    df_hist: pd.DataFrame,
    lag_min: int = 3,
    min_bars_alone: int = 40,
    now_ts: float = None,
) -> pd.DataFrame:
    if now_ts is None:
        now_ts = time.time()
    ticks = get_1m(symbol, now_ts)
    hist_empty = df_hist is None or df_hist.empty

    if ticks.empty:
        return df_hist if not hist_empty else pd.DataFrame()

    if hist_empty:
        return ticks if len(ticks) >= min_bars_alone else pd.DataFrame()

    if not df_hist.index.is_monotonic_increasing:
        df_hist = df_hist.sort_index()
    if df_hist.index.has_duplicates:
        df_hist = df_hist[~df_hist.index.duplicated(keep="last")]

    now_naive = datetime.fromtimestamp(now_ts, timezone.utc).replace(tzinfo=None)
    last_hist = df_hist.index[-1]
    lag = (now_naive - last_hist).total_seconds() / 60.0
    if lag <= lag_min:
        return df_hist

    new_ticks = ticks[ticks.index > last_hist]
    if new_ticks.empty:
        return df_hist

    common = df_hist.index.intersection(ticks.index)[-60:]
    new_ticks = new_ticks.copy()
    if len(common) >= 5:
        offset = float((df_hist.loc[common, "close"] - ticks.loc[common, "close"]).median())
        for col in ("open", "high", "low", "close"):
            new_ticks[col] = new_ticks[col] + offset

    merged = pd.concat([df_hist, new_ticks])
    if merged.index.has_duplicates:
        merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def stats() -> dict:
    with _lock:
        return {s: len(d) for s, d in _bars.items()}
