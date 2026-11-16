# -*- coding: utf-8 -*-
"""Causal Local & Global Support/Resistance engine and Radar trade plan builder."""
import numpy as np
import pandas as pd

from config import normalize_pair_name


def _clean_df(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    out = out.rename(columns=str.lower)
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            return pd.DataFrame()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("UTC").tz_localize(None)
    return out.sort_index().dropna(subset=["open", "high", "low", "close"])


def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < 2:
        return 0.0
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c_prev = df["close"].astype(float).shift(1)
    tr = pd.concat([(h - l).abs(), (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, min_periods=min(period, len(tr)), adjust=False).mean()
    val = float(atr.iloc[-1])
    return val if np.isfinite(val) and val > 0 else float((h - l).median() or 0.0)


def _calc_chop_adx(df: pd.DataFrame, period: int = 14) -> tuple[float | None, float | None]:
    if df is None or len(df) < period * 2 + 5:
        return None, None
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    c_prev = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    hh = h.rolling(period).max()
    ll = l.rolling(period).min()
    span = (hh - ll).replace(0, np.nan)
    chop = 100.0 * np.log10((atr_sum / span).clip(lower=1e-9)) / np.log10(period)

    up_move = h.diff()
    dn_move = -l.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_ema = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0, np.nan)
    pdi = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_ema
    mdi = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_ema
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()

    c_val = float(chop.iloc[-1]) if np.isfinite(chop.iloc[-1]) else None
    a_val = float(adx.iloc[-1]) if np.isfinite(adx.iloc[-1]) else None
    return c_val, a_val


def _calc_vol_ratio(df: pd.DataFrame) -> float:
    if df is None or len(df) < 30:
        return 1.0
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c_prev = df["close"].astype(float).shift(1)
    tr = pd.concat([(h - l).abs(), (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    med = atr.rolling(180, min_periods=30).median().iloc[-1]
    cur = atr.iloc[-1]
    if pd.isna(med) or pd.isna(cur) or med <= 0:
        return 1.0
    return float(max(0.4, min(2.5, cur / med)))


def _causal_swings(df: pd.DataFrame, left: int = 3, right: int = 3, weight: float = 1.0, tag: str = "Swing") -> list[dict]:
    """Суворо каузальні локальні екстремуми: свічка підтверджується лише через right закритих барів."""
    if df is None or len(df) < left + right + 5:
        return []
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    n = len(df)
    levels = []
    for i in range(left, n - right):
        win_h = highs[i - left : i + right + 1]
        win_l = lows[i - left : i + right + 1]
        recency = 0.7 + 0.6 * (i / max(1, n - 1))
        if highs[i] == np.max(win_h) and np.sum(win_h == highs[i]) == 1:
            levels.append({"price": float(highs[i]), "kind": "R", "weight": weight * recency, "tag": tag})
        if lows[i] == np.min(win_l) and np.sum(win_l == lows[i]) == 1:
            levels.append({"price": float(lows[i]), "kind": "S", "weight": weight * recency, "tag": tag})
    return levels


def _cluster_levels(raw_levels: list[dict], tolerance: float, ref_price: float) -> tuple[list[dict], list[dict]]:
    if not raw_levels or tolerance <= 0:
        return [], []
    ordered = sorted(raw_levels, key=lambda x: x["price"])
    clusters = []
    cur = [ordered[0]]
    for item in ordered[1:]:
        mean_p = sum(x["price"] * x["weight"] for x in cur) / sum(x["weight"] for x in cur)
        if abs(item["price"] - mean_p) <= tolerance:
            cur.append(item)
        else:
            clusters.append(cur)
            cur = [item]
    clusters.append(cur)

    supports, resistances = [], []
    for grp in clusters:
        tot_w = sum(x["weight"] for x in grp)
        price = sum(x["price"] * x["weight"] for x in grp) / tot_w
        tags = sorted({x["tag"] for x in grp})
        entry = {
            "price": float(price),
            "strength": round(tot_w + 0.35 * (len(grp) - 1), 2),
            "touches": len(grp),
            "tags": tags,
        }
        if price < ref_price:
            supports.append(entry)
        elif price > ref_price:
            resistances.append(entry)
        else:
            supports.append(entry)
            resistances.append(entry)

    supports.sort(key=lambda x: ref_price - x["price"])
    resistances.sort(key=lambda x: x["price"] - ref_price)
    return supports, resistances


def _daily_pivots_and_pdhl(df_60m: pd.DataFrame, ref_price: float) -> list[dict]:
    if df_60m is None or df_60m.empty:
        return []
    daily = df_60m.resample("1D", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    if len(daily) < 2:
        return []
    prev = daily.iloc[-2]
    ph, pl, pc = float(prev["high"]), float(prev["low"]), float(prev["close"])
    pp = (ph + pl + pc) / 3.0
    r1 = 2.0 * pp - pl
    s1 = 2.0 * pp - ph
    r2 = pp + (ph - pl)
    s2 = pp - (ph - pl)

    out = []
    for p, tag, w in [
        (ph, "PDH", 2.2),
        (pl, "PDL", 2.2),
        (pp, "Pivot", 1.3),
        (r1, "R1", 1.5),
        (s1, "S1", 1.5),
        (r2, "R2", 1.8),
        (s2, "S2", 1.8),
    ]:
        out.append({"price": float(p), "kind": "S" if p <= ref_price else "R", "weight": w, "tag": tag})

    if len(daily) >= 8:
        prev_week = daily.iloc[-8:-1]
        out.append({"price": float(prev_week["high"].max()), "kind": "R", "weight": 2.4, "tag": "7D_High"})
        out.append({"price": float(prev_week["low"].min()), "kind": "S", "weight": 2.4, "tag": "7D_Low"})
    return out


def _round_psych_levels(ref_price: float, is_jpy: bool, atr_h1: float) -> list[dict]:
    step = 0.50 if is_jpy else 0.0050
    base = np.floor(ref_price / step) * step
    levels = []
    for k in range(-3, 4):
        lvl = round(float(base + k * step), 3 if is_jpy else 5)
        if abs(lvl - ref_price) <= max(step * 2.5, atr_h1 * 3.5):
            is_major = abs((lvl / (step * 2)) - round(lvl / (step * 2))) < 1e-6
            levels.append({
                "price": lvl,
                "kind": "S" if lvl <= ref_price else "R",
                "weight": 1.6 if is_major else 1.1,
                "tag": "Psych.00" if is_major else "Psych.50",
            })
    return levels


def _suggest_expiry(scope: str, vol_ratio: float) -> int:
    if scope == "LOCAL":
        if vol_ratio >= 1.25:
            return 5
        if vol_ratio <= 0.85:
            return 10
        return 7
    if vol_ratio >= 1.25:
        return 15
    if vol_ratio <= 0.85:
        return 45
    return 30


def analyze_pair_sr(
    pair_name: str,
    df_5m: pd.DataFrame | None,
    df_15m: pd.DataFrame | None = None,
    df_60m: pd.DataFrame | None = None,
    live_price: float | None = None,
) -> dict:
    """
    Розраховує локальні (M5/M15/Pivot) та глобальні (H1/H4/PDH/PDL/круглі) рівні S/R
    і формує рекомендовану ціну входу, напрямок (CALL/PUT/WAIT) та час експірації.
    """
    clean_pair = normalize_pair_name(pair_name)
    is_jpy = "JPY" in clean_pair
    pt_mult = 1000.0 if is_jpy else 100000.0

    d5 = _clean_df(df_5m)
    if d5.empty:
        return {"ok": False, "pair": clean_pair}

    d15 = _clean_df(df_15m)
    if d15.empty and len(d5) >= 30:
        d15 = d5.resample("15min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()

    d60 = _clean_df(df_60m)
    if d60.empty and len(d5) >= 120:
        d60 = d5.resample("60min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()

    ref_price = float(live_price) if live_price else float(d5["close"].iloc[-1])
    atr_5m = _calc_atr(d5, 14)
    atr_15m = _calc_atr(d15, 14) if not d15.empty else atr_5m * 1.7
    atr_60m = _calc_atr(d60, 14) if not d60.empty else atr_5m * 3.2
    vol_ratio = _calc_vol_ratio(d5)
    chop_5m, adx_5m = _calc_chop_adx(d5, 14)

    local_raw = []
    local_raw.extend(_causal_swings(d5.tail(220), left=3, right=3, weight=1.0, tag="M5"))
    if not d15.empty:
        local_raw.extend(_causal_swings(d15.tail(140), left=3, right=2, weight=1.45, tag="M15"))

    piv_all = _daily_pivots_and_pdhl(d60, ref_price)
    for item in piv_all:
        if item["tag"] in ("Pivot", "S1", "R1"):
            local_raw.append(item)

    loc_sup, loc_res = _cluster_levels(local_raw, max(atr_5m * 0.45, 1e-6), ref_price)

    global_raw = []
    if not d60.empty:
        global_raw.extend(_causal_swings(d60.tail(240), left=3, right=2, weight=2.0, tag="H1"))
        d240 = d60.resample("4h", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}
        ).dropna()
        if len(d240) >= 12:
            global_raw.extend(_causal_swings(d240.tail(120), left=2, right=2, weight=2.6, tag="H4"))
    for item in piv_all:
        if item["tag"] in ("PDH", "PDL", "S2", "R2", "7D_High", "7D_Low"):
            global_raw.append(item)
    global_raw.extend(_round_psych_levels(ref_price, is_jpy, atr_60m))

    glob_sup, glob_res = _cluster_levels(global_raw, max(atr_60m * 0.40, atr_15m * 0.65, 1e-6), ref_price)

    best_loc_s = loc_sup[0] if loc_sup else None
    best_loc_r = loc_res[0] if loc_res else None
    best_glob_s = glob_sup[0] if glob_sup else None
    best_glob_r = glob_res[0] if glob_res else None

    candidates = []
    for scope, lvl, side in [
        ("GLOBAL", best_glob_s, "CALL"),
        ("GLOBAL", best_glob_r, "PUT"),
        ("LOCAL", best_loc_s, "CALL"),
        ("LOCAL", best_loc_r, "PUT"),
    ]:
        if lvl is None:
            continue
        dist = abs(ref_price - lvl["price"])
        dist_atr = dist / max(atr_5m, 1e-9)
        scope_bonus = 1.35 if scope == "GLOBAL" else 1.0
        score = (lvl["strength"] * scope_bonus) / (0.65 + dist_atr)
        candidates.append({
            "scope": scope,
            "level": lvl,
            "side": side,
            "dist": dist,
            "dist_atr": dist_atr,
            "dist_pts": dist * pt_mult,
            "score": score,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)

    is_strong_trend = (adx_5m is not None and adx_5m >= 26.0) and (chop_5m is not None and chop_5m < 44.0)
    trend_dir = "UP" if (len(d15) >= 25 and d15["close"].iloc[-1] > d15["close"].ewm(span=21).mean().iloc[-1]) else "DOWN"

    def _entry_for(lvl_price: float, side: str, scope: str) -> float:
        buf = (0.15 if scope == "GLOBAL" else 0.12) * atr_5m
        raw = (lvl_price + buf) if side == "CALL" else (lvl_price - buf)
        if side == "CALL" and ref_price <= raw:
            raw = ref_price
        if side == "PUT" and ref_price >= raw:
            raw = ref_price
        return round(float(raw), 3 if is_jpy else 5)

    chosen = candidates[0] if candidates else None
    action_dir = "WAIT"
    rec_price = None
    rec_exp = 10
    rec_scope = "LOCAL"
    setup_desc = "Недостатньо рівнів для розрахунку"

    if chosen is not None:
        rec_scope = chosen["scope"]
        rec_exp = _suggest_expiry(rec_scope, vol_ratio)
        lvl_p = chosen["level"]["price"]
        tags_txt = "/".join(chosen["level"]["tags"][:2])

        if (
            is_strong_trend
            and rec_scope == "LOCAL"
            and chosen["dist_atr"] <= 1.2
            and ((trend_dir == "UP" and chosen["side"] == "PUT") or (trend_dir == "DOWN" and chosen["side"] == "CALL"))
        ):
            action_dir = "CALL" if trend_dir == "UP" else "PUT"
            rec_price = round(float(lvl_p), 3 if is_jpy else 5)
            setup_desc = f"Пробій і ретест локального рівня ({tags_txt}) за трендом"
        else:
            rec_price = _entry_for(lvl_p, chosen["side"], rec_scope)
            if chosen["dist_atr"] <= 1.6:
                action_dir = chosen["side"]
                scope_ua = "глобального" if rec_scope == "GLOBAL" else "локального"
                side_ua = "підтримки" if chosen["side"] == "CALL" else "опору"
                setup_desc = f"Відбій від {scope_ua} рівня {side_ua} ({tags_txt})"
            else:
                action_dir = "WAIT"
                scope_ua = "глобального" if rec_scope == "GLOBAL" else "локального"
                setup_desc = f"Ціна посередині каналу ({chosen['dist_pts']:.0f} пт до {scope_ua} {tags_txt}) — чекайте рівень"

    best_call = next((c for c in candidates if c["side"] == "CALL"), None)
    best_put = next((c for c in candidates if c["side"] == "PUT"), None)

    return {
        "ok": True,
        "pair": clean_pair,
        "is_jpy": is_jpy,
        "ref_price": ref_price,
        "atr_5m_pts": round(atr_5m * pt_mult, 1),
        "vol_ratio": vol_ratio,
        "local_support": best_loc_s,
        "local_resistance": best_loc_r,
        "global_support": best_glob_s,
        "global_resistance": best_glob_r,
        "recommended_direction": action_dir,
        "recommended_price": rec_price,
        "recommended_exp": rec_exp,
        "recommended_scope": rec_scope,
        "setup_desc": setup_desc,
        "call_target": {
            "price": _entry_for(best_call["level"]["price"], "CALL", best_call["scope"]),
            "exp": _suggest_expiry(best_call["scope"], vol_ratio),
            "scope": best_call["scope"],
            "tags": best_call["level"]["tags"],
        } if best_call else None,
        "put_target": {
            "price": _entry_for(best_put["level"]["price"], "PUT", best_put["scope"]),
            "exp": _suggest_expiry(best_put["scope"], vol_ratio),
            "scope": best_put["scope"],
            "tags": best_put["level"]["tags"],
        } if best_put else None,
    }
