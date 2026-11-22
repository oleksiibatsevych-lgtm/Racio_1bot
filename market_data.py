# -*- coding: utf-8 -*-
"""Безкоштовні джерела ринкових даних: biquote.io (без ключа) та dukascopy-python (історія 1m)."""
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from config import BIQUOTE_ENABLED, biquote_symbol, normalize_pair_name

logger = logging.getLogger(__name__)

BIQUOTE_BASE = "https://biquote.io/api"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; RacioBot/2.2)"}
_INTERVAL_MAP = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "60m": "1h", "1h": "1h", "4h": "4h", "1d": "1d"}


def parse_biquote_ohlc(payload: dict | list) -> pd.DataFrame:
    bars = payload.get("bars", []) if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    if df.empty or "openTime" not in df.columns:
        return pd.DataFrame()
    if "isOpen" in df.columns:
        df = df[~df["isOpen"].fillna(False).astype(bool)]
    if df.empty:
        return pd.DataFrame()

    df["openTime"] = pd.to_datetime(df["openTime"], utc=True, errors="coerce")
    df = df.dropna(subset=["openTime"])
    if df.empty:
        return pd.DataFrame()

    df = df.set_index("openTime").sort_index()
    df.index = df.index.tz_convert("UTC").tz_localize(None)

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tickVolume" in df.columns:
        df["volume"] = pd.to_numeric(df["tickVolume"], errors="coerce").fillna(0.0)
    elif "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    else:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])
    return df[~df.index.duplicated(keep="last")]


def fetch_biquote_ohlc(pair_name: str, interval: str = "5m", limit: int = 1000, timeout: int = 8) -> pd.DataFrame:
    if not BIQUOTE_ENABLED:
        return pd.DataFrame()
    sym = biquote_symbol(pair_name)
    iv = _INTERVAL_MAP.get(str(interval).lower())
    if not sym or not iv:
        return pd.DataFrame()
    try:
        resp = requests.get(
            f"{BIQUOTE_BASE}/{sym}/ohlc",
            params={"interval": iv, "limit": min(max(int(limit), 1), 1000)},
            headers=_HEADERS,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return pd.DataFrame()
        return parse_biquote_ohlc(resp.json())
    except Exception as e:
        logger.debug(f"biquote OHLC недоступний для {sym} ({iv}): {e}")
        return pd.DataFrame()


def fetch_biquote_quote(pair_name: str, allow_stale: bool = False, timeout: int = 5) -> float | None:
    if not BIQUOTE_ENABLED:
        return None
    sym = biquote_symbol(pair_name)
    if not sym:
        return None
    try:
        resp = requests.get(
            f"{BIQUOTE_BASE}/{sym}",
            params={"allowStale": "true" if allow_stale else "false"},
            headers=_HEADERS,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not isinstance(data, dict):
            return None
        if not allow_stale and (data.get("marketState") == "closed" or data.get("stale")):
            return None
        mid = data.get("mid")
        return float(mid) if mid and float(mid) > 0 else None
    except Exception:
        return None


def fetch_biquote_batch(pairs: list[str], timeout: int = 6) -> dict[str, float]:
    if not BIQUOTE_ENABLED or not pairs:
        return {}
    sym_to_pair = {biquote_symbol(p): normalize_pair_name(p) for p in pairs}
    syms = [s for s in sym_to_pair if s][:100]
    if not syms:
        return {}
    try:
        resp = requests.get(
            f"{BIQUOTE_BASE}/latest",
            params=[("symbols", s) for s in syms],
            headers=_HEADERS,
            timeout=timeout,
        )
        if resp.status_code != 200:
            return {}
        data = resp.json()
        if isinstance(data, dict):
            items = data.items()
        elif isinstance(data, list):
            items = [(str(t.get("symbol") or ""), t) for t in data if isinstance(t, dict)]
        else:
            items = []

        out = {}
        for sym, tick in items:
            if not isinstance(tick, dict):
                continue
            if tick.get("marketState") == "closed" or tick.get("stale"):
                continue
            mid = tick.get("mid")
            if mid and float(mid) > 0 and sym in sym_to_pair:
                out[sym_to_pair[sym]] = float(mid)
        return out
    except Exception:
        return {}


def fetch_dukascopy_1m(pair_name: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    clean = normalize_pair_name(pair_name)
    if not clean:
        return pd.DataFrame()
    s_utc = start_dt.replace(tzinfo=timezone.utc) if start_dt.tzinfo is None else start_dt.astimezone(timezone.utc)
    e_utc = end_dt.replace(tzinfo=timezone.utc) if end_dt.tzinfo is None else end_dt.astimezone(timezone.utc)
    if e_utc <= s_utc:
        return pd.DataFrame()

    df = pd.DataFrame()
    try:
        import dukascopy_python as duka
        candles = list(duka.fetch_candles(
            instrument=clean,
            interval=1,
            time_unit=duka.TimeUnit.MINUTE,
            offer_side=duka.OfferSide.BID,
            start=s_utc,
            end=e_utc,
            validate_instrument_name=False,
        ))
        df = duka.to_dataframe(candles) if candles else pd.DataFrame()
    except Exception:
        try:
            import dukascopy_python.compat as duka_compat
            df = duka_compat.fetch(
                clean,
                duka_compat.INTERVAL_MIN_1,
                duka_compat.OFFER_SIDE_BID,
                s_utc,
                e_utc,
                max_retries=3,
            )
        except Exception as e:
            logger.warning(f"Dukascopy fetch помилка для {clean}: {e}")
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().rename(columns=str.lower)
    if "timestamp" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out = out.set_index(pd.to_datetime(out["timestamp"], utc=True))
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    if "volume" not in out.columns:
        out["volume"] = 0.0
    req = [c for c in ("open", "high", "low", "close") if c in out.columns]
    if len(req) < 4:
        return pd.DataFrame()
    out = out[["open", "high", "low", "close", "volume"]].sort_index().dropna(subset=["open", "high", "low", "close"])
    return out[~out.index.duplicated(keep="last")]


def download_dukascopy_chunked(pair_name: str, days: int = 90) -> pd.DataFrame:
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=max(1, int(days)))
    parts = []
    cur = start_dt
    while cur < end_dt:
        nxt = min(end_dt, cur + timedelta(days=14))
        chunk = fetch_dukascopy_1m(pair_name, cur, nxt)
        if not chunk.empty:
            parts.append(chunk)
        cur = nxt
    if not parts:
        return pd.DataFrame()
    res = pd.concat(parts).sort_index()
    return res[~res.index.duplicated(keep="last")]
