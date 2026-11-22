# -*- coding: utf-8 -*-
"""
Strategy Lab — walk-forward перевірка 57 наявних + 26 нових каузальних стратегій.
Підтримує Yahoo Finance (--source yahoo) та багаторічну 1m-історію Dukascopy (--source dukascopy).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

import reversion_engine as rev
from market_data import download_dukascopy_chunked
from strategy_engine import (
    STRATEGIES_META,
    compute_base_indicators,
    evaluate_strategy_series,
    htf_alignment,
)

PAIRS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "EUR/JPY", "GBP/JPY",
    "AUD/JPY", "CAD/JPY", "CHF/JPY", "EUR/GBP", "EUR/AUD", "EUR/CAD", "EUR/CHF", "GBP/AUD",
    "GBP/CAD", "GBP/CHF", "AUD/CAD", "AUD/CHF", "CAD/CHF",
]
TIMEFRAMES = [1, 3, 5, 15]
EXPIRATIONS = [1, 2, 3, 5, 10, 15, 30]
SPLIT_FRACTION = 0.60
TOP_K = 20
MIN_DISC_TRADES = 150
MIN_CONF_TRADES = 200
CLUSTER_MINUTES = 30


def yahoo_symbol(pair: str) -> str:
    return pair.replace("/", "") + "=X"


def download_and_merge(pairs, data_dir, source: str = "yahoo", days: int = 90):
    import yfinance as yf
    os.makedirs(data_dir, exist_ok=True)
    for p in pairs:
        path = os.path.join(data_dir, p.replace("/", "_") + ".csv")
        try:
            if source == "dukascopy":
                df = download_dukascopy_chunked(p, days=days)
            else:
                raw = yf.Ticker(yahoo_symbol(p)).history(period="7d", interval="1m", auto_adjust=True)
                if raw is None or raw.empty:
                    df = pd.DataFrame()
                else:
                    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in raw.columns]
                    df = raw[cols].rename(columns=str.lower)
                    if "volume" not in df.columns:
                        df["volume"] = 0.0
                    if df.index.tz is not None:
                        df.index = df.index.tz_convert("UTC").tz_localize(None)

            if df is None or df.empty:
                print(f"  {p}: порожньо")
                continue
            if os.path.exists(path):
                old = pd.read_csv(path, index_col=0, parse_dates=True)
                df = pd.concat([old, df])
                df = df[~df.index.duplicated(keep="last")].sort_index()
            df.to_csv(path)
            print(f"  {p}: {len(df)} хв ({df.index[0]} → {df.index[-1]})")
        except Exception as e:
            print(f"  {p}: помилка {e}")


def load_pair(pair, data_dir):
    path = os.path.join(data_dir, pair.replace("/", "_") + ".csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df.rename(columns=str.lower)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[~df.index.duplicated(keep="last")].dropna(subset=["open", "high", "low", "close"])
    return df if len(df) > 500 else None


def minutes_of(index) -> np.ndarray:
    return index.values.astype("datetime64[m]").astype(np.int64)


def _prev(a, k=1):
    out = np.empty_like(a, dtype=float)
    out[:k] = np.nan
    out[k:] = a[:-k]
    return out


def _sig(call, put):
    out = np.zeros(len(call), dtype=np.int8)
    out[call] = 1
    out[put & ~call] = -1
    return out


def _prev_daily_ohlc_arrays(df_index, highs, lows, closes):
    tmp = pd.DataFrame({"h": highs, "l": lows, "c": closes}, index=df_index)
    grp = tmp.groupby(tmp.index.date)
    daily = grp.agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), cnt=("c", "count"))
    daily = daily[daily["cnt"] >= 6].sort_index()
    if daily.empty:
        nan_arr = np.full(len(df_index), np.nan)
        return nan_arr, nan_arr, nan_arr
    valid_dates = np.array(daily.index, dtype="datetime64[D]")
    bar_dates = np.array(df_index.date, dtype="datetime64[D]")
    pos = np.searchsorted(valid_dates, bar_dates, side="left") - 1
    ok = pos >= 0
    pos_c = np.maximum(pos, 0)
    dh = np.where(ok, daily["h"].to_numpy(float)[pos_c], np.nan)
    dl = np.where(ok, daily["l"].to_numpy(float)[pos_c], np.nan)
    dc = np.where(ok, daily["c"].to_numpy(float)[pos_c], np.nan)
    return dh, dl, dc


def _asian_bounds_arrays(df_index, highs, lows):
    tmp = pd.DataFrame({"h": highs, "l": lows}, index=df_index)
    asia = tmp[tmp.index.hour < 7]
    if asia.empty:
        nan_arr = np.full(len(df_index), np.nan)
        return nan_arr, nan_arr
    daily_asia = asia.groupby(asia.index.date).agg(ah=("h", "max"), al=("l", "min"))
    dates = pd.Series(df_index.date, index=df_index)
    ah = dates.map(daily_asia["ah"]).to_numpy(float)
    al = dates.map(daily_asia["al"]).to_numpy(float)
    lon = (df_index.hour >= 7) & (df_index.hour <= 11)
    return np.where(lon, ah, np.nan), np.where(lon, al, np.nan)


class Feat:
    def __init__(self, df, pair_name: str = ""):
        self.pair = pair_name
        self.is_jpy = "JPY" in pair_name
        self.index = df.index
        self.o, self.h, self.l, self.c = (df[k].values.astype(float) for k in ("open", "high", "low", "close"))
        self.v = df["volume"].values.astype(float) if "volume" in df.columns else np.zeros(len(df), dtype=float)
        self.s = pd.Series(self.c)
        self._cache = {}
        self.htf = {}

    def _get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def rsi(self, n):
        def f():
            d = self.s.diff()
            up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
            dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
            return (100 - 100 / (1 + up / (dn + 1e-12))).values
        return self._get(("rsi", n), f)

    def ema(self, n):
        return self._get(("ema", n), lambda: self.s.ewm(span=n, adjust=False).mean().values)

    def sma(self, n):
        return self._get(("sma", n), lambda: self.s.rolling(n).mean().values)

    def std(self, n):
        return self._get(("std", n), lambda: self.s.rolling(n).std().values)

    def atr(self, n=14):
        def f():
            pc = _prev(self.c)
            tr = np.maximum(self.h - self.l, np.maximum(np.abs(self.h - pc), np.abs(self.l - pc)))
            return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values
        return self._get(("atr", n), f)

    def vol_ratio(self):
        def f():
            atr = pd.Series(self.atr(14))
            med = atr.rolling(200, min_periods=50).median()
            return (atr / (med + 1e-12)).values
        return self._get(("vr",), f)

    def chop(self, n=14):
        def f():
            pc = _prev(self.c)
            tr = pd.Series(np.maximum(self.h - self.l, np.maximum(np.abs(self.h - pc), np.abs(self.l - pc))))
            atr_sum = tr.rolling(n).sum()
            hh = pd.Series(self.h).rolling(n).max()
            ll = pd.Series(self.l).rolling(n).min()
            return (100.0 * np.log10((atr_sum / (hh - ll + 1e-12)).clip(lower=1e-9)) / np.log10(n)).values
        return self._get(("chop", n), f)

    def adx(self, n=14):
        def f():
            up = pd.Series(self.h).diff()
            dn = -pd.Series(self.l).diff()
            pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0))
            mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0))
            atr = pd.Series(self.atr(n)) + 1e-12
            pdi = 100.0 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr
            mdi = 100.0 * mdm.ewm(alpha=1 / n, adjust=False).mean() / atr
            dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
            return dx.ewm(alpha=1 / n, adjust=False).mean().values
        return self._get(("adx", n), f)

    def er(self, n=10):
        def f():
            change = pd.Series(self.c).diff(n).abs()
            volatility = pd.Series(self.c).diff().abs().rolling(n).sum()
            return (change / (volatility + 1e-12)).values
        return self._get(("er", n), f)

    def kama(self, n=10, fast=2, slow=30):
        def f():
            er = self.er(n)
            fast_sc = 2.0 / (fast + 1.0)
            slow_sc = 2.0 / (slow + 1.0)
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            out = np.full(len(self.c), np.nan)
            if n < len(self.c):
                out[n] = self.c[n]
                for i in range(n + 1, len(self.c)):
                    k = sc[i] if np.isfinite(sc[i]) else slow_sc ** 2
                    out[i] = out[i - 1] + k * (self.c[i] - out[i - 1])
            return out
        return self._get(("kama", n, fast, slow), f)


def s_rsi_rev(F, n, lo, hi):
    r = F.rsi(n)
    return _sig(r < lo, r > hi)


def s_bb(F, k, mode):
    ma, sd = F.sma(20), F.std(20)
    below, above = F.c < ma - k * sd, F.c > ma + k * sd
    return _sig(below, above) if mode == "fade" else _sig(above, below)


def s_ema_cross(F, a, b):
    diff = F.ema(a) - F.ema(b)
    pd_ = _prev(diff)
    return _sig((diff > 0) & (pd_ <= 0), (diff < 0) & (pd_ >= 0))


def s_pullback(F):
    e21, e50, r = F.ema(21), F.ema(50), F.rsi(7)
    up, dn = (F.c > e50) & (e21 > e50), (F.c < e50) & (e21 < e50)
    return _sig(up & (r < 35), dn & (r > 65))


def s_macd(F):
    macd = F.ema(12) - F.ema(26)
    hist = macd - pd.Series(macd).ewm(span=9, adjust=False).mean().values
    ph = _prev(hist)
    return _sig((hist > 0) & (ph <= 0), (hist < 0) & (ph >= 0))


def s_donchian(F, n):
    hi = pd.Series(F.h).rolling(n).max().shift(1).values
    lo = pd.Series(F.l).rolling(n).min().shift(1).values
    return _sig(F.c > hi, F.c < lo)


def s_stoch(F):
    lo = pd.Series(F.l).rolling(14).min().values
    hi = pd.Series(F.h).rolling(14).max().values
    k = pd.Series((F.c - lo) / (hi - lo + 1e-12) * 100).rolling(3).mean().values
    d = pd.Series(k).rolling(3).mean().values
    pk, pdd = _prev(k), _prev(d)
    return _sig((k > d) & (pk <= pdd) & (k < 30), (k < d) & (pk >= pdd) & (k > 70))


def s_engulf(F):
    po, pc = _prev(F.o), _prev(F.c)
    bull = (pc < po) & (F.c > F.o) & (F.c >= po) & (F.o <= pc)
    bear = (pc > po) & (F.c < F.o) & (F.c <= po) & (F.o >= pc)
    return _sig(bull, bear)


def s_pin(F):
    rng = F.h - F.l + 1e-12
    body = np.abs(F.c - F.o)
    lw = np.minimum(F.o, F.c) - F.l
    uw = F.h - np.maximum(F.o, F.c)
    return _sig((lw > 2 * body) & (lw > 0.6 * rng), (uw > 2 * body) & (uw > 0.6 * rng))


def s_streak(F, n, mode):
    g = (F.c > F.o).astype(int)
    r = (F.c < F.o).astype(int)
    gs = pd.Series(g).rolling(n).sum().values == n
    rs = pd.Series(r).rolling(n).sum().values == n
    return _sig(rs, gs) if mode == "fade" else _sig(gs, rs)


def s_zscore(F, th):
    z = (F.c - F.sma(30)) / (F.std(30) + 1e-12)
    return _sig(z < -th, z > th)


def _htf(F, h):
    d = F.htf.get(h)
    if d is None:
        z = np.zeros(len(F.c))
        return z.astype(np.int8), z + 50
    return d["trend"], d["rsi"]


def s_mtf_pullback(F, h):
    tr, _ = _htf(F, h)
    r = F.rsi(7)
    return _sig((tr == 1) & (r < 35), (tr == -1) & (r > 65))


def s_mtf_cross(F, h):
    tr, _ = _htf(F, h)
    diff = F.ema(5) - F.ema(13)
    pd_ = _prev(diff)
    return _sig((tr == 1) & (diff > 0) & (pd_ <= 0), (tr == -1) & (diff < 0) & (pd_ >= 0))


def s_mtf_break(F, h):
    tr, _ = _htf(F, h)
    hi = pd.Series(F.h).rolling(10).max().shift(1).values
    lo = pd.Series(F.l).rolling(10).min().shift(1).values
    return _sig((tr == 1) & (F.c > hi), (tr == -1) & (F.c < lo))


def s_mtf_range_fade(F, h):
    _, hr = _htf(F, h)
    ma, sd = F.sma(20), F.std(20)
    calm = (hr > 40) & (hr < 60)
    return _sig(calm & (F.c < ma - 2 * sd), calm & (F.c > ma + 2 * sd))


def s_new_turtle_soup(F):
    hh20 = pd.Series(F.h).rolling(20).max().shift(1).values
    ll20 = pd.Series(F.l).rolling(20).min().shift(1).values
    recent_hh3 = pd.Series(F.h).rolling(3).max().shift(1).values
    recent_ll3 = pd.Series(F.l).rolling(3).min().shift(1).values
    call = (ll20 < recent_ll3) & (F.l < ll20) & (F.c > ll20)
    put = (hh20 > recent_hh3) & (F.h > hh20) & (F.c < hh20)
    return _sig(call, put)


def s_new_asian_sweep(F):
    ah, al = _asian_bounds_arrays(F.index, F.h, F.l)
    call = np.isfinite(al) & (F.l < al) & (F.c > al)
    put = np.isfinite(ah) & (F.h > ah) & (F.c < ah)
    return _sig(call, put)


def s_new_pdhl_reject(F):
    pdh, pdl, _ = _prev_daily_ohlc_arrays(F.index, F.h, F.l, F.c)
    rng = F.h - F.l + 1e-12
    lw = np.minimum(F.o, F.c) - F.l
    uw = F.h - np.maximum(F.o, F.c)
    call = np.isfinite(pdl) & (F.l < pdl) & (F.c > pdl) & (lw >= 0.45 * rng)
    put = np.isfinite(pdh) & (F.h > pdh) & (F.c < pdh) & (uw >= 0.45 * rng)
    return _sig(call, put)


def s_new_round_bounce(F):
    step = 0.50 if F.is_jpy else 0.0050
    tol = 0.15 * F.atr(14)
    nearest = np.round(F.c / step) * step
    r = F.rsi(14)
    chop = F.chop(14)
    call = (chop >= 50) & (np.abs(F.l - nearest) <= tol) & (F.c > nearest) & (r < 38)
    put = (chop >= 50) & (np.abs(F.h - nearest) <= tol) & (F.c < nearest) & (r > 62)
    return _sig(call, put)


def s_new_pivot_s1r1(F):
    pdh, pdl, pdc = _prev_daily_ohlc_arrays(F.index, F.h, F.l, F.c)
    pp = (pdh + pdl + pdc) / 3.0
    r1v = 2.0 * pp - pdl
    s1v = 2.0 * pp - pdh
    chop = F.chop(14)
    call = (chop >= 48) & np.isfinite(s1v) & (F.l <= s1v) & (F.c > s1v) & (F.c > F.o)
    put = (chop >= 48) & np.isfinite(r1v) & (F.h >= r1v) & (F.c < r1v) & (F.c < F.o)
    return _sig(call, put)


def s_new_ob_retest(F):
    atr = F.atr(14)
    po, pc = _prev(F.o), _prev(F.c)
    bull_ob_hi = np.full(len(F.c), np.nan)
    bull_ob_lo = np.full(len(F.c), np.nan)
    bear_ob_hi = np.full(len(F.c), np.nan)
    bear_ob_lo = np.full(len(F.c), np.nan)
    last_b_hi = last_b_lo = last_s_hi = last_s_lo = np.nan
    for i in range(2, len(F.c)):
        imp = F.c[i - 1] - F.o[i - 1]
        if F.c[i - 2] < F.o[i - 2] and imp > 1.5 * atr[i - 1]:
            last_b_hi, last_b_lo = F.h[i - 2], F.l[i - 2]
        if F.c[i - 2] > F.o[i - 2] and -imp > 1.5 * atr[i - 1]:
            last_s_hi, last_s_lo = F.h[i - 2], F.l[i - 2]
        bull_ob_hi[i], bull_ob_lo[i] = last_b_hi, last_b_lo
        bear_ob_hi[i], bear_ob_lo[i] = last_s_hi, last_s_lo
    call = np.isfinite(bull_ob_hi) & (F.l <= bull_ob_hi) & (F.c >= bull_ob_lo) & (F.c > F.o) & (pc < po)
    put = np.isfinite(bear_ob_lo) & (F.h >= bear_ob_lo) & (F.c <= bear_ob_hi) & (F.c < F.o) & (pc > po)
    return _sig(call, put)


def s_new_vwap_zscore(F):
    tp = (F.h + F.l + F.c) / 3.0
    w = F.v if np.sum(F.v) > 0 else np.maximum(F.h - F.l, 1e-6)
    df_tmp = pd.DataFrame({"pv": tp * w, "w": w}, index=F.index)
    grp = df_tmp.groupby(df_tmp.index.date)
    vwap = grp["pv"].cumsum().values / (grp["w"].cumsum().values + 1e-12)
    diff = F.c - vwap
    sd = pd.Series(diff).rolling(30, min_periods=15).std().values + 1e-12
    z = diff / sd
    pz = _prev(z)
    return _sig((pz < -2.0) & (z > pz) & (F.c > F.o), (pz > 2.0) & (z < pz) & (F.c < F.o))


def s_new_kama_band(F):
    k = F.kama(10)
    atr = F.atr(14)
    er = F.er(10)
    call = (er < 0.25) & (F.c < k - 2.2 * atr) & (F.rsi(7) < 30)
    put = (er < 0.25) & (F.c > k + 2.2 * atr) & (F.rsi(7) > 70)
    return _sig(call, put)


def s_new_fisher_rev(F):
    hl2 = 0.5 * (F.h + F.l)
    lo = pd.Series(hl2).rolling(10).min().values
    hi = pd.Series(hl2).rolling(10).max().values
    raw = np.clip(2.0 * ((hl2 - lo) / (hi - lo + 1e-12) - 0.5), -0.999, 0.999)
    val = pd.Series(raw).ewm(alpha=0.33, adjust=False).mean().clip(-0.999, 0.999).values
    ft = np.zeros(len(F.c), dtype=float)
    for i in range(1, len(F.c)):
        v = val[i]
        if np.isfinite(v):
            ft[i] = 0.5 * math.log((1.0 + v) / (1.0 - v)) + 0.5 * ft[i - 1]
    pft = _prev(ft)
    return _sig((pft < -2.0) & (ft > pft), (pft > 2.0) & (ft < pft))


def s_new_bb_pctb_fade(F):
    ma, sd = F.sma(20), F.std(20)
    upper, lower = ma + 2.0 * sd, ma - 2.0 * sd
    pctb = (F.c - lower) / (upper - lower + 1e-12)
    bbw = (upper - lower) / (ma + 1e-12)
    med_w = pd.Series(bbw).rolling(50).median().values
    return _sig((bbw <= med_w) & (pctb < 0.0), (bbw <= med_w) & (pctb > 1.0))


def s_new_impulse_exh(F):
    atr = F.atr(14)
    rng = F.h - F.l + 1e-12
    uw = F.h - np.maximum(F.o, F.c)
    lw = np.minimum(F.o, F.c) - F.l
    big = rng > 2.2 * atr
    call = big & (F.c < F.o) & (lw > 0.35 * rng)
    put = big & (F.c > F.o) & (uw > 0.35 * rng)
    return _sig(call, put)


def s_new_zscore_15m(F):
    z = (F.c - F.sma(30)) / (F.std(30) + 1e-12)
    return _sig((z < -2.3) & (F.c > F.o), (z > 2.3) & (F.c < F.o))


def s_new_holy_grail(F):
    e20 = F.ema(20)
    adx = F.adx(14)
    pl, ph = _prev(F.l), _prev(F.h)
    up_pb = (adx > 28) & (F.c > e20) & (pl <= e20) & (F.c > ph)
    dn_pb = (adx > 28) & (F.c < e20) & (ph >= e20) & (F.c < pl)
    return _sig(up_pb, dn_pb)


def s_new_kama_pb(F):
    k = F.kama(10)
    er = F.er(10)
    r2 = F.rsi(2)
    return _sig((er > 0.35) & (F.c > k) & (r2 < 15), (er > 0.35) & (F.c < k) & (r2 > 85))


def s_new_chandelier(F):
    tr, _ = _htf(F, 60)
    atr = F.atr(22)
    hh = pd.Series(F.h).rolling(22).max().values
    ll = pd.Series(F.l).rolling(22).min().values
    long_stop = hh - 3.0 * atr
    short_stop = ll + 3.0 * atr
    pc = _prev(F.c)
    call = (tr == 1) & (F.c > short_stop) & (pc <= _prev(short_stop))
    put = (tr == -1) & (F.c < long_stop) & (pc >= _prev(long_stop))
    return _sig(call, put)


def s_new_donch_mid_pb(F):
    mid = 0.5 * (pd.Series(F.h).rolling(20).max().values + pd.Series(F.l).rolling(20).min().values)
    adx = F.adx(14)
    pmid = _prev(mid, 3)
    call = (adx > 22) & (mid > pmid) & (F.l <= mid) & (F.c > mid)
    put = (adx > 22) & (mid < pmid) & (F.h >= mid) & (F.c < mid)
    return _sig(call, put)


def s_new_stochrsi_htf(F):
    tr15, _ = _htf(F, 15)
    tr60, _ = _htf(F, 60)
    r = pd.Series(F.rsi(14))
    rlo = r.rolling(14).min()
    rhi = r.rolling(14).max()
    k = ((r - rlo) / (rhi - rlo + 1e-12) * 100).rolling(3).mean().values
    d = pd.Series(k).rolling(3).mean().values
    pk, pdd = _prev(k), _prev(d)
    call = (tr15 == 1) & (tr60 == 1) & (k > d) & (pk <= pdd) & (k < 25)
    put = (tr15 == -1) & (tr60 == -1) & (k < d) & (pk >= pdd) & (k > 75)
    return _sig(call, put)


def s_new_123_pb(F):
    atr = F.atr(14)
    exp_up = (_prev(F.c, 2) - _prev(F.o, 2)) > 1.5 * _prev(atr, 2)
    exp_dn = (_prev(F.o, 2) - _prev(F.c, 2)) > 1.5 * _prev(atr, 2)
    pb_up = _prev(F.c, 1) < _prev(F.o, 1)
    pb_dn = _prev(F.c, 1) > _prev(F.o, 1)
    return _sig(exp_up & pb_up & (F.c > _prev(F.h, 1)), exp_dn & pb_dn & (F.c < _prev(F.l, 1)))


def s_new_tick_climax(F):
    vol = F.v if np.sum(F.v) > 0 else np.maximum(F.h - F.l, 1e-6)
    med = pd.Series(vol).rolling(50).median().values + 1e-12
    spike = vol > 2.5 * med
    body = np.abs(F.c - F.o) + 1e-12
    lw = np.minimum(F.o, F.c) - F.l
    uw = F.h - np.maximum(F.o, F.c)
    return _sig(spike & (lw > 2.0 * body), spike & (uw > 2.0 * body))


def s_new_obv_z_div(F):
    vol = F.v if np.sum(F.v) > 0 else np.maximum(F.h - F.l, 1e-6)
    direction = np.sign(F.c - _prev(F.c))
    obv = pd.Series(np.nan_to_num(direction) * vol).cumsum()
    obv_hi10 = obv.rolling(10).max().shift(1).values
    obv_lo10 = obv.rolling(10).min().shift(1).values
    p_hi10 = pd.Series(F.h).rolling(10).max().shift(1).values
    p_lo10 = pd.Series(F.l).rolling(10).min().shift(1).values
    call = (F.l < p_lo10) & (obv.values > obv_lo10) & (F.c > F.o)
    put = (F.h > p_hi10) & (obv.values < obv_hi10) & (F.c < F.o)
    return _sig(call, put)


def s_new_wpr_hook(F):
    hh = pd.Series(F.h).rolling(14).max().values
    ll = pd.Series(F.l).rolling(14).min().values
    wpr = -100.0 * (hh - F.c) / (hh - ll + 1e-12)
    pw = _prev(wpr)
    chop = F.chop(14)
    return _sig((chop >= 50) & (pw < -90) & (wpr >= -85), (chop >= 50) & (pw > -10) & (wpr <= -15))


def s_new_vortex(F):
    pc, ph, pl = _prev(F.c), _prev(F.h), _prev(F.l)
    tr = pd.Series(np.maximum(F.h - F.l, np.maximum(np.abs(F.h - pc), np.abs(F.l - pc))))
    vmp = pd.Series(np.abs(F.h - pl))
    vmm = pd.Series(np.abs(F.l - ph))
    tr14 = tr.rolling(14).sum() + 1e-12
    vip = (vmp.rolling(14).sum() / tr14).values
    vim = (vmm.rolling(14).sum() / tr14).values
    pvip, pvim = _prev(vip), _prev(vim)
    adx = F.adx(14)
    rising = adx > _prev(adx)
    return _sig((adx > 20) & rising & (vip > vim) & (pvip <= pvim), (adx > 20) & rising & (vip < vim) & (pvip >= pvim))


def s_new_regime_switch(F):
    er = F.er(10)
    k = F.kama(10)
    ma, sd = F.sma(20), F.std(20)
    r7 = F.rsi(7)
    trend_call = (er > 0.35) & (F.c > k) & (F.rsi(2) < 15)
    trend_put = (er > 0.35) & (F.c < k) & (F.rsi(2) > 85)
    mr_call = (er < 0.20) & (F.c < ma - 2.0 * sd) & (r7 < 30)
    mr_put = (er < 0.20) & (F.c > ma + 2.0 * sd) & (r7 > 70)
    return _sig(trend_call | mr_call, trend_put | mr_put)


STRATEGIES = []
for n_, lo_, hi_ in ((7, 30, 70), (7, 20, 80), (7, 15, 85), (14, 30, 70), (14, 25, 75), (14, 20, 80)):
    STRATEGIES.append((f"rsi_rev(n={n_},{lo_}/{hi_})", s_rsi_rev, dict(n=n_, lo=lo_, hi=hi_)))
for k_ in (2.0, 2.5, 3.0):
    STRATEGIES.append((f"bb_fade(k={k_})", s_bb, dict(k=k_, mode="fade")))
    STRATEGIES.append((f"bb_break(k={k_})", s_bb, dict(k=k_, mode="cont")))
for a_, b_ in ((9, 21), (5, 13)):
    STRATEGIES.append((f"ema_cross({a_},{b_})", s_ema_cross, dict(a=a_, b=b_)))
STRATEGIES += [
    ("trend_pullback", s_pullback, {}),
    ("macd_flip", s_macd, {}),
    ("stoch_rev", s_stoch, {}),
    ("engulfing_rev", s_engulf, {}),
    ("pin_bar_rev", s_pin, {}),
]
for n_ in (20, 10):
    STRATEGIES.append((f"donchian_break({n_})", s_donchian, dict(n=n_)))
for n_ in (3, 4):
    STRATEGIES.append((f"streak_fade({n_})", s_streak, dict(n=n_, mode="fade")))
    STRATEGIES.append((f"streak_cont({n_})", s_streak, dict(n=n_, mode="cont")))
for th_ in (2.0, 2.5):
    STRATEGIES.append((f"zscore_fade({th_})", s_zscore, dict(th=th_)))
for h_ in (15, 60):
    STRATEGIES.append((f"mtf_pullback(htf={h_}m)", s_mtf_pullback, dict(h=h_)))
    STRATEGIES.append((f"mtf_ema_cross(htf={h_}m)", s_mtf_cross, dict(h=h_)))
    STRATEGIES.append((f"mtf_breakout(htf={h_}m)", s_mtf_break, dict(h=h_)))
    STRATEGIES.append((f"mtf_range_fade(htf={h_}m)", s_mtf_range_fade, dict(h=h_)))

NEW_STRATEGIES = [
    ("NEW_TURTLE_SOUP_5M", s_new_turtle_soup, {}),
    ("NEW_ASIAN_SWEEP_5M", s_new_asian_sweep, {}),
    ("NEW_PDHL_REJECT_15M", s_new_pdhl_reject, {}),
    ("NEW_ROUND_BOUNCE_5M", s_new_round_bounce, {}),
    ("NEW_PIVOT_S1R1_5M", s_new_pivot_s1r1, {}),
    ("NEW_OB_RETEST_5M", s_new_ob_retest, {}),
    ("NEW_VWAP_ZSCORE_5M", s_new_vwap_zscore, {}),
    ("NEW_KAMA_BAND_5M", s_new_kama_band, {}),
    ("NEW_FISHER_REV_5M", s_new_fisher_rev, {}),
    ("NEW_BB_PCTB_FADE_5M", s_new_bb_pctb_fade, {}),
    ("NEW_IMPULSE_EXH_5M", s_new_impulse_exh, {}),
    ("NEW_ZSCORE_15M", s_new_zscore_15m, {}),
    ("NEW_HOLY_GRAIL_5M", s_new_holy_grail, {}),
    ("NEW_KAMA_PB_5M", s_new_kama_pb, {}),
    ("NEW_CHANDELIER_15M", s_new_chandelier, {}),
    ("NEW_DONCH_MID_PB_5M", s_new_donch_mid_pb, {}),
    ("NEW_STOCHRSI_HTF_5M", s_new_stochrsi_htf, {}),
    ("NEW_123_PB_3M", s_new_123_pb, {}),
    ("NEW_TICK_CLIMAX_3M", s_new_tick_climax, {}),
    ("NEW_OBV_Z_DIV_5M", s_new_obv_z_div, {}),
    ("NEW_WPR_HOOK_5M", s_new_wpr_hook, {}),
    ("NEW_VORTEX_15M", s_new_vortex, {}),
    ("NEW_REGIME_SWITCH_5M", s_new_regime_switch, {}),
]
STRATEGIES.extend(NEW_STRATEGIES)

HTFS = (15, 60)


def attach_htf(F, df1, tf, end_min):
    for h in HTFS:
        if h <= tf:
            continue
        dh = df1.resample(f"{h}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(dh) < 60:
            continue
        sh = pd.Series(dh["close"].values.astype(float))
        e21, e50 = sh.ewm(span=21, adjust=False).mean().values, sh.ewm(span=50, adjust=False).mean().values
        c = sh.values
        trend = np.where((e21 > e50) & (c > e50), 1, np.where((e21 < e50) & (c < e50), -1, 0)).astype(np.int8)
        d_ = sh.diff()
        up = d_.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        dn = (-d_.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rsi = (100 - 100 / (1 + up / (dn + 1e-12))).values
        h_end = minutes_of(dh.index) + h
        j = np.searchsorted(h_end, end_min, side="right") - 1
        ok = j >= 0
        jc = np.maximum(j, 0)
        F.htf[h] = dict(trend=np.where(ok, trend[jc], 0).astype(np.int8), rsi=np.where(ok, rsi[jc], 50.0))


N_CONTROLS = 5


def control_signal(n_bars, seed, density=0.04):
    rng = np.random.default_rng(seed)
    mask = rng.random(n_bars) < density
    dirs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_bars)
    return np.where(mask, dirs, 0).astype(np.int8)


def dynamic_exp(F, tf, rule):
    vr = pd.Series(F.vol_ratio())
    hi = vr.rolling(500, min_periods=100).quantile(0.75).values
    lo = vr.rolling(500, min_periods=100).quantile(0.25).values
    v = vr.values
    bars = np.where(v > hi, 1, np.where(v < lo, 3, 2))
    if rule == "dyn_vol_inv":
        bars = 4 - bars
    return (bars * tf).astype(np.int64)


def extract_trades(sig, end_min, m1_min, m1_open, exp, delay=0):
    idx = np.flatnonzero(sig)
    if idx.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int8)
    em = end_min[idx] + delay
    ex = exp[idx] if isinstance(exp, np.ndarray) else np.full(len(idx), int(exp), dtype=np.int64)
    keep, i, n = [], 0, len(em)
    while i < n:
        keep.append(i)
        i = int(np.searchsorted(em, em[i] + ex[i], side="left"))
    keep = np.array(keep)
    em, dirs, ex = em[keep], sig[idx][keep], ex[keep]

    ie = np.searchsorted(m1_min, em)
    ix = np.searchsorted(m1_min, em + ex)
    n1 = len(m1_min)
    ok = (ie < n1) & (ix < n1)
    ie_c, ix_c = np.minimum(ie, n1 - 1), np.minimum(ix, n1 - 1)
    ok &= (m1_min[ie_c] == em) & (m1_min[ix_c] == em + ex)
    em, dirs, ie_c, ix_c = em[ok], dirs[ok], ie_c[ok], ix_c[ok]
    delta = m1_open[ix_c] - m1_open[ie_c]
    nz = delta != 0
    em, dirs, delta = em[nz], dirs[nz], delta[nz]
    win = ((delta > 0) & (dirs == 1)) | ((delta < 0) & (dirs == -1))
    return em, win.astype(np.int8)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def cluster_stats(win, em, breakeven):
    n = len(win)
    if n == 0:
        return dict(n=0, wr=float("nan"), se=float("nan"), z=float("nan"), p=1.0, lo=float("nan"))
    w = win.sum()
    p = w / n
    _, inv = np.unique(em // CLUSTER_MINUTES, return_inverse=True)
    nb = np.bincount(inv).astype(float)
    wb = np.bincount(inv, weights=win.astype(float))
    resid = wb - p * nb
    var = float((resid ** 2).sum()) / (n ** 2)
    se = max(math.sqrt(var), math.sqrt(p * (1 - p) / n) * 0.5, 1e-9)
    z = (p - breakeven) / se
    return dict(n=n, wr=p, se=se, z=z, p=norm_sf(z), lo=p - 1.645 * se)


def collect_basket_strategies(data, store, delay=0):
    closes_5m, opens_1m, mins_1m = {}, {}, {}
    for pair, df1 in data.items():
        d5 = df1.resample("5min", label="left", closed="left").agg({"close": "last"}).dropna()
        if len(d5) >= 120:
            closes_5m[pair] = d5["close"]
            mins_1m[pair] = minutes_of(df1.index)
            opens_1m[pair] = df1["open"].values.astype(float)

    if all(p in closes_5m for p in ("EUR/USD", "GBP/USD", "AUD/USD")):
        df_u = pd.concat([closes_5m["EUR/USD"], closes_5m["GBP/USD"], closes_5m["AUD/USD"]], axis=1, join="inner")
        df_u.columns = ["EU", "GU", "AU"]
        ret = df_u.pct_change(3)
        z_eu = ret["EU"] - ret[["GU", "AU"]].mean(axis=1)
        score = z_eu / (z_eu.rolling(60).std() + 1e-12)
        sig_eu = np.where(score < -2.2, 1, np.where(score > 2.2, -1, 0)).astype(np.int8)
        end_m = minutes_of(df_u.index) + 5
        for exp in (10, 15):
            em, w = extract_trades(sig_eu, end_m, mins_1m["EUR/USD"], opens_1m["EUR/USD"], exp, delay)
            if len(em):
                store.setdefault(("NEW_USD_DIV_5M", 5, str(exp)), []).append((em, w))

    jpy_pairs = [p for p in ("EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY", "USD/JPY") if p in closes_5m]
    if len(jpy_pairs) >= 4:
        df_j = pd.concat([closes_5m[p] for p in jpy_pairs], axis=1, join="inner")
        df_j.columns = jpy_pairs
        r1 = df_j.pct_change(1)
        sync_sig = np.where((r1 > 0.0005).sum(axis=1) >= 3, 1, np.where((r1 < -0.0005).sum(axis=1) >= 3, -1, 0)).astype(np.int8)
        end_m = minutes_of(df_j.index) + 5
        for p in jpy_pairs:
            for exp in (10, 15):
                em, w = extract_trades(sync_sig, end_m, mins_1m[p], opens_1m[p], exp, delay)
                if len(em):
                    store.setdefault(("NEW_JPY_SYNC_5M", 5, str(exp)), []).append((em, w))

    if "EUR/USD" in closes_5m and "USD/CHF" in closes_5m:
        df_c = pd.concat([closes_5m["EUR/USD"], closes_5m["USD/CHF"]], axis=1, join="inner")
        df_c.columns = ["EU", "UC"]
        synth = np.log(df_c["EU"] * df_c["UC"])
        z = (synth - synth.rolling(60).mean()) / (synth.rolling(60).std() + 1e-12)
        sig_eu = np.where(z < -2.2, 1, np.where(z > 2.2, -1, 0)).astype(np.int8)
        end_m = minutes_of(df_c.index) + 5
        for exp in (10, 15):
            em, w = extract_trades(sig_eu, end_m, mins_1m["EUR/USD"], opens_1m["EUR/USD"], exp, delay)
            if len(em):
                store.setdefault(("NEW_CORR_SPREAD_5M", 5, str(exp)), []).append((em, w))


def collect(data, delay=0, verbose=True):
    store = {}
    t0 = time.time()
    for pi, (pair, df1) in enumerate(data.items()):
        m1_min, m1_open = minutes_of(df1.index), df1["open"].values.astype(float)
        htf_frames = {}
        for h in (15, 60):
            dh = df1.resample(f"{h}min", label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}
            ).dropna()
            if len(dh) >= 60:
                htf_frames[h] = dh

        for tf in TIMEFRAMES:
            if tf == 1:
                dft = df1
            else:
                dft = df1.resample(f"{tf}min", label="left", closed="left").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna(subset=["open", "high", "low", "close"])
            if len(dft) < 120:
                continue
            end_min = minutes_of(dft.index) + tf
            F = Feat(dft, pair_name=pair)
            attach_htf(F, df1, tf, end_min)
            exp_variants = [(str(e), e) for e in EXPIRATIONS] + [
                (r, dynamic_exp(F, tf, r)) for r in ("dyn_vol", "dyn_vol_inv")
            ]

            sigs = [(name, fn(F, **kw)) for name, fn, kw in STRATEGIES]

            base_ind = compute_base_indicators(dft)
            for s_key, meta in STRATEGIES_META.items():
                if meta["tf_min"] != tf:
                    continue
                if s_key in rev.PRESETS:
                    sg = evaluate_strategy_series(dft, s_key)
                    sigs.append((f"BOT_{s_key}", sg))
                    continue
                htf_m = meta.get("htf")
                htf_ctx = htf_alignment(dft, htf_frames.get(htf_m), tf, htf_m) if htf_m else None
                for reg_on in (True, False):
                    label = f"BOT_{s_key}{'_REG' if reg_on else '_RAW'}"
                    sg = evaluate_strategy_series(dft, s_key, ind=base_ind, htf_ctx=htf_ctx, regime_filter=reg_on)
                    sigs.append((label, sg))

            sigs += [
                (f"CTRL_random_{k + 1}", control_signal(len(dft), seed=1000 * pi + 10 * tf + k))
                for k in range(N_CONTROLS)
            ]

            for name, sg in sigs:
                if name.startswith("BOT_") and name[4:].split("_REG")[0].split("_RAW")[0] in STRATEGIES_META:
                    key_name = name[4:].replace("_REG", "").replace("_RAW", "")
                    exp_set = list(exp_variants)
                    fixed_exp = int(STRATEGIES_META[key_name]["exp"])
                    if str(fixed_exp) not in {lbl for lbl, _ in exp_set}:
                        exp_set.append((str(fixed_exp), fixed_exp))
                else:
                    exp_set = exp_variants

                for exp_label, exp in exp_set:
                    em, win = extract_trades(sg, end_min, m1_min, m1_open, exp, delay)
                    if len(em):
                        store.setdefault((name, tf, exp_label), []).append((em, win))
        if verbose:
            print(f"  [{pi + 1}/{len(data)}] {pair} готово ({time.time() - t0:.0f}с)", flush=True)

    collect_basket_strategies(data, store, delay=delay)

    merged = {}
    for k, parts in store.items():
        em = np.concatenate([p[0] for p in parts])
        win = np.concatenate([p[1] for p in parts])
        order = np.argsort(em, kind="stable")
        merged[k] = (em[order], win[order])
    return merged


def analyze(merged, payout):
    breakeven = 1.0 / (1.0 + payout)
    all_em = np.concatenate([v[0] for v in merged.values()])
    split = np.quantile(all_em, SPLIT_FRACTION)
    rows = []
    for (name, tf, exp), (em, win) in merged.items():
        d, c = em < split, em >= split
        sd = cluster_stats(win[d], em[d], breakeven)
        sc = cluster_stats(win[c], em[c], breakeven)
        rows.append(dict(
            strategy=name, tf=tf, exp=exp, control=name.startswith("CTRL"),
            d_n=sd["n"], d_wr=sd["wr"], d_z=sd["z"],
            c_n=sc["n"], c_wr=sc["wr"], c_z=sc["z"], c_p=sc["p"], c_lo=sc["lo"],
        ))
    rep = pd.DataFrame(rows)
    return rep, split, breakeven


def prune_config_from_report(rep: pd.DataFrame, config_path: str, threshold: float = 0.45, min_trades: int = 30) -> list[str]:
    if not os.path.exists(config_path):
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    bot_rows = rep[rep["strategy"].str.startswith("BOT_")].copy()
    if bot_rows.empty:
        return []
    bot_rows["base_key"] = bot_rows["strategy"].str.replace(r"^BOT_|_(REG|RAW)$", "", regex=True)
    bot_rows["tot_n"] = bot_rows["d_n"] + bot_rows["c_n"]
    bot_rows["tot_w"] = bot_rows["d_wr"].fillna(0) * bot_rows["d_n"] + bot_rows["c_wr"].fillna(0) * bot_rows["c_n"]
    agg = bot_rows.groupby("base_key")[["tot_w", "tot_n"]].sum()

    disabled = []
    for key, row in agg.iterrows():
        n = float(row["tot_n"])
        if n < min_trades:
            continue
        wr = float(row["tot_w"]) / n
        if key in cfg and wr < threshold and cfg[key]:
            cfg[key] = False
            disabled.append(f"{key} ({wr * 100:.1f}%, n={int(n)})")

    if disabled:
        tmp = config_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, config_path)
    return disabled


def build_summary(rep, split, breakeven, payout, data_info):
    real = rep[~rep.control & (rep.d_n >= MIN_DISC_TRADES)].copy()
    ctrl = rep[rep.control & (rep.d_n >= MIN_DISC_TRADES)].copy()
    n_configs = len(real)
    top = real.sort_values("d_z", ascending=False).head(TOP_K).copy()
    alpha = 0.05 / TOP_K
    top["PASS"] = (top.c_n >= MIN_CONF_TRADES) & (top.c_wr > breakeven) & (top.c_p < alpha)
    lines = []
    A = lines.append
    A("🧪 ЛАБОРАТОРІЯ СТРАТЕГІЙ (57 НАЯВНИХ + 26 НОВИХ КАУЗАЛЬНИХ)")
    A(f"Дані: {data_info}")
    A(f"Беззбиток при виплаті {payout * 100:.0f}%: {breakeven * 100:.2f}%")
    A(f"Перевірено конфігурацій: {n_configs}")
    A(f"Поділ за часом: перші {int(SPLIT_FRACTION * 100)}% — пошук, решта — підтвердження")
    A("")
    if len(ctrl):
        A("Контроль (випадковий напрямок):")
        A(f"  найкращий у пошуку: {ctrl.d_wr.max() * 100:.1f}% → у підтвердженні: "
          f"{ctrl.loc[ctrl.d_wr.idxmax(), 'c_wr'] * 100:.1f}%")
        A(f"  середній вінрейт контролю: пошук {ctrl.d_wr.mean() * 100:.1f}%, підтвердження {ctrl.c_wr.mean() * 100:.1f}%")
    if len(real):
        A(f"Найкращий вінрейт у пошуку: {real.d_wr.max() * 100:.1f}%")
        A(f"Середній вінрейт усіх стратегій у підтвердженні: {real.c_wr.mean() * 100:.2f}%")
    A("")
    A(f"ТОП-{TOP_K} за пошуком → позавибіркове підтвердження:")
    A("стратегія | ТФ | exp | пошук n/WR | підтвердж. n/WR | p | вердикт")
    for _, r in top.iterrows():
        exp_txt = f"{r.exp}хв" if str(r.exp).isdigit() else f"динамічна:{r.exp}"
        A(f"{r.strategy} | {int(r.tf)}m | {exp_txt} | {int(r.d_n)}/{r.d_wr * 100:.1f}% | "
          f"{int(r.c_n)}/{r.c_wr * 100:.1f}% | {r.c_p:.3f} | {'✅ PASS' if r.PASS else '❌'}")
    n_pass = int(top.PASS.sum())
    A("")
    A(f"ПІДСУМОК: пройшли перевірку {n_pass} з {len(top)}.")
    return "\n".join(lines), top[top.PASS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="lab_data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--source", choices=("yahoo", "dukascopy"), default="yahoo")
    ap.add_argument("--days", type=int, default=90, help="кількість днів історії 1m для Dukascopy")
    ap.add_argument("--payout", type=float, default=0.85)
    ap.add_argument("--delay", type=int, default=0, help="затримка входу в хвилинах після закриття свічки")
    ap.add_argument("--out", default=".")
    ap.add_argument("--prune-config", default="", help="шлях до strategies_config.json для авто-вимкнення стратегій нижче порога")
    ap.add_argument("--prune-threshold", type=float, default=0.45)
    args = ap.parse_args()

    if args.download:
        print(f"Завантаження 1m-історії ({args.source})…")
        download_and_merge(PAIRS, args.data_dir, source=args.source, days=args.days)

    data = {p: d for p in PAIRS if (d := load_pair(p, args.data_dir)) is not None}
    if not data:
        print("Немає даних. Запустіть: python3 lab.py --download --source dukascopy --days 90")
        sys.exit(1)
    first = min(d.index[0] for d in data.values())
    last = max(d.index[-1] for d in data.values())
    days = (last - first).total_seconds() / 86400
    info = f"{len(data)} пар, {first:%Y-%m-%d} → {last:%Y-%m-%d} ({days:.1f} дн.)"
    print(info)

    merged = collect(data, delay=args.delay)
    rep, split, breakeven = analyze(merged, args.payout)
    summary, passed = build_summary(rep, split, breakeven, args.payout, info)

    os.makedirs(args.out, exist_ok=True)
    rep.sort_values("d_z", ascending=False).to_csv(os.path.join(args.out, "lab_report.csv"), index=False)
    with open(os.path.join(args.out, "lab_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    with open(os.path.join(args.out, "lab_validated.json"), "w", encoding="utf-8") as f:
        json.dump(
            [dict(strategy=r.strategy, tf=int(r.tf), exp=str(r.exp), conf_wr=float(r.c_wr), conf_n=int(r.c_n))
             for _, r in passed.iterrows()],
            f,
            ensure_ascii=False,
            indent=2,
        )
    if args.prune_config:
        pruned = prune_config_from_report(rep, args.prune_config, threshold=args.prune_threshold)
        print(f"Відсіяно нижче {args.prune_threshold * 100:.0f}%: {pruned or 'нічого'}")
    print("\n" + summary)


if __name__ == "__main__":
    main()
