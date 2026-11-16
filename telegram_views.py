# -*- coding: utf-8 -*-
import html
import math
from datetime import timedelta

from config import AUTO_DISABLE_WR_THRESHOLD, PAYOUT, normalize_pair_name
from reversion_engine import PRESETS
from strategy_engine import STRATEGIES_META, STRATEGY_CATEGORIES, get_strategy_expiration
from trading_sessions import get_tz_info, session_name, utc_datetime

HOLDOUT = {"REV_RSI14": (276, 67.03), "REV_BB3": (351, 65.24), "REV_RSI7": (921, 60.80)}
PERIODS = {"today": "сьогодні", "7": "7 днів", "30": "30 днів", "all": "все"}

SESSION_LABELS = {
    "Overlap": "Overlap (Лондон + Нью-Йорк)",
    "London": "Лондон",
    "NY": "Нью-Йорк",
    "Asia": "Азія",
    "Other": "Міжсесійний перехід",
}


def format_pair_name(ticker: str | None) -> str:
    return normalize_pair_name(ticker) or "—"


def _is_jpy_pair(pair: str | None, price: float | None = None) -> bool:
    clean = format_pair_name(pair)
    if clean and clean != "—":
        return "JPY" in clean
    if price is not None:
        return abs(float(price)) >= 20.0
    return False


def format_price(price: float | None, pair: str | None = None) -> str:
    if price is None:
        return "немає котирування"
    fmt = "{:.3f}" if _is_jpy_pair(pair, price) else "{:.5f}"
    return fmt.format(float(price))


def localized_session_name(value) -> str:
    raw = session_name(value)
    return SESSION_LABELS.get(raw, raw)


def break_even_pct(payout: float = PAYOUT) -> float:
    return 100.0 / (1.0 + float(payout))


def wr_badge(wins: int, total: int, payout: float = PAYOUT, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD) -> str:
    if total <= 0:
        return "⚪"
    wr = wins / total
    if wr * 100.0 >= break_even_pct(payout):
        return "🟢"
    if wr >= float(wr_threshold):
        return "🟡"
    return "🔴"


def progress_bar(percent):
    filled = max(0, min(10, round(percent / 10)))
    return "█" * filled + "░" * (10 - filled)


def _wilson_lower(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.5
    p = wins / total
    denom = 1.0 + (z * z) / total
    center = p + (z * z) / (2.0 * total)
    rad = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)
    return max(0.0, min(1.0, (center - rad) / denom))


def compute_signal_confidence(cand: dict, payout: float = PAYOUT) -> tuple[int, str, str]:
    preset = str(cand.get("strategy") or "")
    sig_dir = str(cand.get("signal") or "CALL").upper()
    score = 50.0

    c_votes = int(cand.get("call_votes") or 0)
    p_votes = int(cand.get("put_votes") or 0)
    total_votes = c_votes + p_votes
    if total_votes > 0:
        win_votes = max(c_votes, p_votes)
        opp_votes = min(c_votes, p_votes)
        c_score = float(cand.get("confluence_score") or win_votes)
        net_ratio = (win_votes - opp_votes) / total_votes
        score += min(16.0, max(0.0, (c_score - 0.8) * 6.5)) * net_ratio
        score -= opp_votes * 3.5
    else:
        prio = float(cand.get("priority") or 80)
        score += 3.0 if prio >= 90 else 1.5

    confs = int(cand.get("confirmations") or 0)
    if confs > 0:
        score += min(7.0, confs * 5.5)

    chop = cand.get("chop")
    adx = cand.get("adx")
    meta = STRATEGIES_META.get(preset, {})
    regime = str(cand.get("regime") or meta.get("regime") or "").upper()
    cat = str(meta.get("category") or "").upper()
    is_reversion_style = (
        regime in ("REVERSION", "RANGE", "SMC", "PA")
        or cat in ("REVERSION", "OSCILLATOR", "SMC", "PA")
        or preset.startswith(("REV_", "LIQ_", "DIV_", "BB_", "CRSI_", "PIVOT_"))
    )

    if is_reversion_style:
        if chop is not None:
            if float(chop) >= 52.0:
                score += 5.0
            elif float(chop) >= 45.0:
                score += 2.0
            elif float(chop) < 38.0:
                score -= 6.0
        if adx is not None:
            if float(adx) < 22.0:
                score += 3.0
            elif float(adx) > 32.0:
                score -= 5.0
    else:
        if adx is not None:
            if float(adx) >= 25.0:
                score += 5.0
            elif float(adx) < 18.0:
                score -= 6.0
        if chop is not None:
            if float(chop) < 45.0:
                score += 4.0
            elif float(chop) >= 58.0:
                score -= 6.0

    rsi = cand.get("rsi")
    if rsi is not None:
        r = float(rsi)
        if is_reversion_style:
            if (sig_dir == "CALL" and r <= 22.0) or (sig_dir == "PUT" and r >= 78.0):
                score += 7.0
            elif (sig_dir == "CALL" and r <= 32.0) or (sig_dir == "PUT" and r >= 68.0):
                score += 4.0
            elif (sig_dir == "CALL" and r >= 65.0) or (sig_dir == "PUT" and r <= 35.0):
                score -= 4.0
        else:
            if (sig_dir == "CALL" and r >= 75.0) or (sig_dir == "PUT" and r <= 25.0):
                score -= 4.0
            elif (sig_dir == "CALL" and 50.0 <= r <= 66.0) or (sig_dir == "PUT" and 34.0 <= r <= 50.0):
                score += 2.5

    be = 1.0 / (1.0 + float(payout))
    hw = cand.get("hist_wins")
    ht = cand.get("hist_total")
    if hw is not None and ht:
        lo = _wilson_lower(int(hw), int(ht))
        score += max(-10.0, min(10.0, (lo - be) * 80.0))
    elif preset in HOLDOUT:
        n_h, wr_h = HOLDOUT[preset]
        lo = _wilson_lower(round(n_h * wr_h / 100.0), n_h)
        score += max(-10.0, min(10.0, (lo - be) * 80.0))

    pct = int(round(max(25.0, min(92.0, score))))
    if pct >= 76:
        return pct, "ВИСОКА", "🔥"
    if pct >= 62:
        return pct, "ДОБРА", "🟢"
    if pct >= 48:
        return pct, "ПОМІРНА", "🟡"
    return pct, "БАЗОВА", "🟠"


def wr_line(title, rows, auto_disabled=False, payout: float = PAYOUT, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD):
    decisive = [r for r in rows if r.get("status") == "CLOSED" and r.get("result") in ("WIN", "LOSS")]
    n = len(decisive)
    wins = sum(r["result"] == "WIN" for r in decisive)
    losses = n - wins
    percent = 100.0 * wins / n if n else 0.0
    ico = wr_badge(wins, n, payout=payout, wr_threshold=wr_threshold)
    suffix = " ⛔" if auto_disabled else ""
    clean_title = html.escape(
        format_pair_name(title)
        if (":" in title or ("_" in title and "/" not in title and title not in STRATEGIES_META))
        else title
    )
    if not n:
        return f"{ico} <b>{clean_title}</b> — немає угод{suffix}"
    return f"{ico} <b>{clean_title}</b> — <b>{percent:.1f}%</b> ({wins}✅ / {losses}❌ · {n}){suffix}"


def signal_card(pair, cand, entry, now, tz_name: str = "Europe/Kyiv", risk_amount: float | None = None):
    _ = risk_amount
    exp = get_strategy_expiration(cand)
    tf = cand.get("tf") or STRATEGIES_META.get(cand.get("strategy", ""), {}).get("tf", "5m")
    expires_utc = utc_datetime(now) + timedelta(minutes=exp)
    tz_label, tz_obj = get_tz_info(tz_name)
    expires_local = expires_utc.astimezone(tz_obj)
    direction = "🟢 CALL ↑" if cand["signal"] == "CALL" else "🔴 PUT ↓"
    clean_pair = format_pair_name(pair)
    price_fmt = format_price(entry, clean_pair)
    exp_mode_label = "Золота середина" if cand.get("exp_mode") == "DYN" else "Фіксована"

    conf_pct, conf_label, conf_ico = compute_signal_confidence(cand)
    conf_bar = progress_bar(conf_pct)

    bar_end = cand.get("bar_end") or now
    sess_txt = localized_session_name(bar_end)

    parts = [
        f"<b>{html.escape(clean_pair)} · {direction}</b>",
        f"⏳ <b>ЕКСПІРАЦІЯ: {exp} ХВИЛИН</b>",
        f"🕒 До <b>{expires_local:%H:%M} {html.escape(tz_label)}</b> ({expires_utc:%H:%M} UTC) · {exp_mode_label}",
        f"{conf_ico} Впевненість: <b>{conf_pct}% ({conf_label})</b> [{conf_bar}]",
    ]

    rsi_part = f" · RSI: {float(cand['rsi']):.1f}" if cand.get("rsi") is not None else ""
    parts.append(f"📍 Вхід: <b>{price_fmt}</b>{rsi_part}")
    parts.append(f"🌐 Сесія: {html.escape(sess_txt)}")

    agreeing = cand.get("agreeing_strategies") or []
    opposing = cand.get("opposing_strategies") or []
    if len(agreeing) > 1 or opposing:
        c_votes = int(cand.get("call_votes") or 0)
        p_votes = int(cand.get("put_votes") or 0)
        c_score = float(cand.get("confluence_score") or 0.0)
        parts.append(f"🧩 Консенсус: <b>{c_score:.2f}</b> ({c_votes}🟢 vs {p_votes}🔴)")

    metrics = []
    if cand.get("chop") is not None:
        metrics.append(f"CHOP {float(cand['chop']):.1f}")
    if cand.get("adx") is not None:
        metrics.append(f"ADX {float(cand['adx']):.1f}")
    metrics.append(f"ТФ {html.escape(str(tf))}")
    parts.append(f"📊 Ринок: {' | '.join(metrics)}")
    return "\n".join(parts)


def settlement_text(
    result,
    exit_price,
    pips,
    pair: str | None = None,
    entry_price: float | None = None,
    signal_type: str | None = None,
):
    icon = {"WIN": "✅", "LOSS": "❌", "NEUTRAL": "➖", "VOID": "⚪"}.get(result, "⚪")
    price_str = format_price(exit_price, pair) if exit_price is not None else "немає котирування"

    if exit_price is None:
        return f"🏁 <b>{icon} {html.escape(str(result))}</b> · Вихід: {price_str} · пункти: —"

    is_jpy = _is_jpy_pair(pair, exit_price)
    pt_mult = 1000.0 if is_jpy else 100000.0

    if entry_price is not None:
        raw_diff = float(exit_price) - float(entry_price)
        if str(signal_type or "").upper() == "PUT":
            raw_diff = -raw_diff
        elif str(signal_type or "").upper() != "CALL" and pips is not None and float(pips) != 0:
            if (raw_diff > 0) != (float(pips) > 0):
                raw_diff = -raw_diff
        pts = raw_diff * pt_mult
    else:
        pts = float(pips or 0.0)

    pts_str = f"{pts:+.1f}" if abs(pts - round(pts)) >= 0.05 else f"{round(pts):+d}"
    classic_pips = pts / 10.0
    return (
        f"🏁 <b>{icon} {html.escape(str(result))}</b> · Вихід: {price_str} · "
        f"пункти: <b>{pts_str} пт</b> ({classic_pips:+.1f} піпс)"
    )


def stats_text(rows, period, auto_disabled_global=None, payout: float = PAYOUT, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD):
    disabled_set = set(auto_disabled_global or ())
    decisive = [r for r in rows if r.get("status") == "CLOSED" and r.get("result") in ("WIN", "LOSS")]
    total = len(decisive)
    wins = sum(r["result"] == "WIN" for r in decisive)
    losses = total - wins
    wr_pct = (100.0 * wins / total) if total else 0.0
    ev_pct = ((wins / total) * float(payout) - (1.0 - wins / total)) * 100.0 if total else 0.0
    be_pct = break_even_pct(payout)

    lines = [f"📊 <b>Статистика · {PERIODS[period]}</b>"]
    if not total:
        lines.append("⚪ <b>Загалом:</b> немає завершених угод")
        lines.append("Угод за цей період немає.")
        return "\n".join(lines)

    lines.append(
        f"{wr_badge(wins, total, payout, wr_threshold)} <b>Загалом:</b> <b>{wr_pct:.1f}%</b> ({wins}✅ / {losses}❌ · {total} угод)"
    )
    lines.append(
        f"<i>Очікування ({int(float(payout) * 100)}% випл.): {ev_pct:+.1f}% · Беззбиток: {be_pct:.1f}% · Авто-вимк: &lt;{float(wr_threshold) * 100:.0f}%</i>"
    )

    lines.append("\n<b>📌 По стратегіях</b>")
    for cat_key, cat_title in STRATEGY_CATEGORIES.items():
        cat_items = []
        for s_key, meta in STRATEGIES_META.items():
            if meta["category"] != cat_key:
                continue
            sub = [r for r in decisive if r.get("strategy") == s_key]
            if not sub:
                continue
            sw = sum(r["result"] == "WIN" for r in sub)
            sn = len(sub)
            cat_items.append((sw / sn, sn, sw, s_key, meta["title"]))
        if not cat_items:
            continue
        cat_items.sort(reverse=True)
        lines.append(f"<i>{html.escape(cat_title)}</i>")
        for wr, sn, sw, s_key, title in cat_items:
            tag = " ⛔" if s_key in disabled_set else ""
            lines.append(
                f"  {wr_badge(sw, sn, payout, wr_threshold)} {html.escape(title)} — <b>{wr * 100:.1f}%</b> ({sw}✅/{sn - sw}❌){tag}"
            )

    unknown_keys = sorted({r.get("strategy") for r in decisive if r.get("strategy") not in STRATEGIES_META})
    for uk in unknown_keys:
        if not uk:
            continue
        sub = [r for r in decisive if r.get("strategy") == uk]
        sw = sum(r["result"] == "WIN" for r in sub)
        sn = len(sub)
        lines.append(f"  {wr_badge(sw, sn, payout, wr_threshold)} {html.escape(uk)} — <b>{sw / sn * 100:.1f}%</b> ({sw}✅/{sn - sw}❌)")

    pair_groups = {}
    for r in decisive:
        p = format_pair_name(r.get("ticker"))
        pair_groups.setdefault(p, []).append(r)

    ranked_pairs = []
    for p, grp in pair_groups.items():
        pw = sum(r["result"] == "WIN" for r in grp)
        pn = len(grp)
        ranked_pairs.append((pw / pn, pw, pn, p))
    ranked_pairs.sort(reverse=True)

    if len(ranked_pairs) <= 8:
        lines.append("\n<b>💱 По валютних парах</b>")
        for wr, pw, pn, p in ranked_pairs:
            lines.append(f"  {wr_badge(pw, pn, payout, wr_threshold)} <b>{html.escape(p)}</b> — <b>{wr * 100:.1f}%</b> ({pw}✅/{pn - pw}❌)")
    else:
        lines.append("\n<b>🏆 Топ-5 пар</b>")
        for wr, pw, pn, p in ranked_pairs[:5]:
            lines.append(f"  {wr_badge(pw, pn, payout, wr_threshold)} <b>{html.escape(p)}</b> — <b>{wr * 100:.1f}%</b> ({pw}✅/{pn - pw}❌)")
        lines.append("\n<b>⚠️ Низ-5 пар</b>")
        for wr, pw, pn, p in ranked_pairs[-5:]:
            lines.append(f"  {wr_badge(pw, pn, payout, wr_threshold)} <b>{html.escape(p)}</b> — <b>{wr * 100:.1f}%</b> ({pw}✅/{pn - pw}❌)")

    return "\n".join(lines)


def stats_matrix_text(live_rows, bt_rows, payout: float = PAYOUT, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD) -> str:
    live_list = live_rows if isinstance(live_rows, (list, tuple)) else []
    bt_list = bt_rows if isinstance(bt_rows, (list, tuple)) else []

    by_pair = {}
    for r in live_list:
        p = format_pair_name(r.get("ticker"))
        s = r.get("strategy") or "?"
        w, t = int(r.get("wins") or 0), int(r.get("total") or 0)
        by_pair.setdefault(p, {}).setdefault(s, {})["live"] = (w, t)

    for r in bt_list:
        p = format_pair_name(r.get("pair_name") or r.get("pair"))
        s = r.get("strategy_key") or r.get("strategy") or "?"
        w, t = int(r.get("wins") or 0), int(r.get("total") or 0)
        by_pair.setdefault(p, {}).setdefault(s, {})["bt"] = (w, t)

    if not by_pair:
        return "📋 <b>Матриця Пара × Стратегія</b>\nПоки що немає статистики живих угод або бектесту."

    lines = ["📋 <b>Матриця Пара × Стратегія (Живі | Бектест)</b>"]
    for pair in sorted(by_pair):
        lines.append(f"\n💱 <b>{html.escape(pair)}</b>")
        items = []
        for s_key, d in by_pair[pair].items():
            lw, lt = d.get("live", (0, 0))
            bw, bt = d.get("bt", (0, 0))
            sort_wr = (lw / lt) if lt else ((bw / bt) if bt else 0.0)
            items.append((sort_wr, s_key, lw, lt, bw, bt))
        items.sort(reverse=True)
        for _, s_key, lw, lt, bw, bt in items:
            parts = []
            if lt:
                parts.append(f"Live {wr_badge(lw, lt, payout, wr_threshold)} {lw / lt * 100:.1f}% ({lw}/{lt})")
            if bt:
                parts.append(f"BT {wr_badge(bw, bt, payout, wr_threshold)} {bw / bt * 100:.1f}% ({bw}/{bt})")
            lines.append(f"  • <code>{html.escape(s_key)}</code>: " + " | ".join(parts))
    return "\n".join(lines)


def auto_disabled_text(items: list, wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD) -> str:
    pct = float(wr_threshold) * 100.0
    if not isinstance(items, (list, tuple)) or not items:
        return (
            f"🛡 <b>Авто-вимкнення (&lt;{pct:.0f}% WR)</b>\n"
            "✅ Жодна стратегія зараз не заблокована."
        )
    lines = [
        f"🚫 <b>Авто-вимкнені стратегії (&lt;{pct:.0f}% WR)</b>",
        "Натисніть на кнопку нижче, щоб вручну ввімкнути стратегію знову:",
    ]
    for it in items:
        scope = "Глобально" if it["scope_type"] == "GLOBAL" else format_pair_name(it["pair_name"])
        wr = float(it.get("win_rate") or 0.0) * 100.0
        lines.append(
            f"⛔ <b>{html.escape(scope)}</b> · <code>{html.escape(it['strategy_key'])}</code> — "
            f"{wr:.1f}% ({it['wins']}/{it['total']}, {html.escape(it.get('source') or '')})"
        )
    return "\n".join(lines)


def active_text(rows, now, live_prices: dict | None = None):
    lines = ["⏳ <b>Активні угоди</b>"]
    if live_prices is not None:
        lines[0] += f" <i>(оновлено {utc_datetime(now):%H:%M:%S} UTC)</i>"
    for r in rows:
        pair = format_pair_name(r.get("ticker"))
        expiry = r.get("expires_at") or utc_datetime(r["created_at"]) + timedelta(minutes=r["expiration_mins"])
        seconds = max(0, int((utc_datetime(expiry) - utc_datetime(now)).total_seconds()))
        status_line = f"{seconds // 60}:{seconds % 60:02d} до експірації" if seconds else "очікуємо результат"
        entry = float(r["entry_price"]) if r.get("entry_price") is not None else None
        live_str = ""
        if live_prices and r.get("ticker") in live_prices and live_prices[r["ticker"]] and entry:
            lp = float(live_prices[r["ticker"]])
            move = lp - entry if r.get("signal_type") == "CALL" else entry - lp
            pts = move * (1000.0 if _is_jpy_pair(pair, lp) else 100000.0)
            ico = "🟢 +" if pts > 0 else ("🔴 " if pts < 0 else "⚪ ")
            p_fmt = format_price(lp, pair)
            live_str = f"\n  📍 Зараз: {p_fmt} · {ico}{round(pts)} пт"
        lines.append(
            f"#{r['id']} {html.escape(pair)} · ⏱ {r['expiration_mins']} хв\n"
            f"  {html.escape(r['signal_type'])} · {status_line}{live_str}"
        )
    if not rows:
        lines.append("Відкритих угод немає.")
    return "\n".join(lines)


def history_text(rows):
    lines = ["📜 <b>Історія · останні 15</b>"]
    for r in rows:
        pair = format_pair_name(r.get("ticker"))
        result = "VOID" if r["status"] == "VOID" else r["result"]
        lines.append(
            f"#{r['id']} {html.escape(pair)} · {html.escape(r.get('signal_type') or '')} ({r.get('expiration_mins') or '?'} хв)\n"
            + settlement_text(
                result,
                float(r["exit_price"]) if r.get("exit_price") is not None else None,
                r.get("pips") or 0,
                pair=pair,
                entry_price=float(r["entry_price"]) if r.get("entry_price") is not None else None,
                signal_type=r.get("signal_type"),
            )
        )
    return "\n".join(lines) if rows else lines[0] + "\nЗавершених угод немає."


def pair_radar_text(
    diag: dict,
    news_status: tuple[bool, str],
    upcoming_news: list,
    live_rows: list,
    bt_rows: list,
    live_price: float | None = None,
    payout: float = PAYOUT,
    wr_threshold: float = AUTO_DISABLE_WR_THRESHOLD,
    sr_plan: dict | None = None,
) -> str:
    pair = diag["pair"]
    price = live_price or diag.get("price")
    p_str = format_price(price, pair) if price else "—"
    chop_s = f"{diag['chop']:.1f}" if diag.get("chop") is not None else "—"
    adx_s = f"{diag['adx']:.1f}" if diag.get("adx") is not None else "—"
    rsi_s = f"{diag['rsi14']:.1f}" if diag.get("rsi14") is not None else "—"
    vr_s = f"{diag['vol_ratio']:.2f}x" if diag.get("vol_ratio") is not None else "—"

    regime_desc = "Флет/діапазон" if (diag.get("chop") or 50) >= 48 else "Трендовий рух"
    news_bad, news_reason = news_status
    news_line = f"⛔ {html.escape(news_reason)}" if news_bad else "🟢 Спокійно (карантин не активний)"
    sess_raw = diag.get("session", "Other")
    sess_disp = SESSION_LABELS.get(sess_raw, sess_raw)

    lines = [
        f"🔍 <b>Радар пари {html.escape(pair)}</b>",
        f"💲 Ціна: <b>{p_str}</b> · Сесія: <b>{html.escape(sess_disp)}</b>",
        f"📊 5m: CHOP <b>{chop_s}</b> ({regime_desc}) · ADX <b>{adx_s}</b> · RSI14 <b>{rsi_s}</b> · Vol <b>{vr_s}</b>",
        f"🧭 Тренд: 15m {diag.get('trend_15m', '—')} | 60m {diag.get('trend_60m', '—')}",
        f"📰 Новини: {news_line}",
        f"⚙️ Стратегій в роботі: <b>{diag['checked']}</b> (глобально вимк: {diag['off_global']}, по парі: {diag['off_pair']}, пауза 1г: {diag.get('muted', 0)})",
    ]

    if sr_plan and sr_plan.get("ok"):
        def _lvl_str(item):
            if not item:
                return "—"
            tags = ",".join(item.get("tags", [])[:2])
            return f"<b>{format_price(item['price'], pair)}</b> ({html.escape(tags)} · x{item['touches']})"

        lines.append("\n📐 <b>Рівні підтримки та опору (S/R):</b>")
        lines.append(f"  🔹 Локальні (M5/M15): Підтр. {_lvl_str(sr_plan.get('local_support'))} | Опір {_lvl_str(sr_plan.get('local_resistance'))}")
        lines.append(f"  🔸 Глобальні (H1/PDH): Підтр. {_lvl_str(sr_plan.get('global_support'))} | Опір {_lvl_str(sr_plan.get('global_resistance'))}")

        rec_dir = sr_plan.get("recommended_direction", "WAIT")
        rec_price = sr_plan.get("recommended_price")
        rec_exp = sr_plan.get("recommended_exp", 10)
        scope_ua = "Глобальний рівень" if sr_plan.get("recommended_scope") == "GLOBAL" else "Локальний рівень"

        if rec_dir in ("CALL", "PUT") and rec_price is not None:
            dir_badge = "🟢 CALL ↑" if rec_dir == "CALL" else "🔴 PUT ↓"
            lines.append(
                f"🎯 <b>Торговий план ({scope_ua}):</b> {dir_badge}\n"
                f"  📍 Реком. ціна входу: <b>{format_price(rec_price, pair)}</b>\n"
                f"  ⏳ <b>ЕКСПІРАЦІЯ: {rec_exp} ХВ</b>\n"
                f"  <i>{html.escape(sr_plan.get('setup_desc', ''))}</i>"
            )
        else:
            lines.append(f"⏳ <b>Сценарій:</b> {html.escape(sr_plan.get('setup_desc', 'Очікування підходу до рівня'))}")
            ct = sr_plan.get("call_target")
            pt = sr_plan.get("put_target")
            if ct:
                lines.append(f"  🟢 CALL від <b>{format_price(ct['price'], pair)}</b> · Експ. <b>{ct['exp']} ХВ</b>")
            if pt:
                lines.append(f"  🔴 PUT від <b>{format_price(pt['price'], pair)}</b> · Експ. <b>{pt['exp']} ХВ</b>")

    if upcoming_news:
        lines.append("\n<b>🗓 Найближчі події (High):</b>")
        for ev in upcoming_news[:3]:
            lines.append(f"  • {ev['dt']:%H:%M} UTC ({ev['country']}): {html.escape(ev['title'])}")

    if live_rows:
        lw = sum(int(r.get("wins") or 0) for r in live_rows)
        lt = sum(int(r.get("total") or 0) for r in live_rows)
        if lt:
            lines.append(f"\n📈 <b>Жива статистика пари:</b> {wr_badge(lw, lt, payout, wr_threshold)} {lw / lt * 100:.1f}% ({lw}✅/{lt - lw}❌)")

    if bt_rows:
        top_bt = sorted(bt_rows, key=lambda r: (float(r.get("win_rate") or 0.0), int(r.get("total") or 0)), reverse=True)[:5]
        lines.append("\n<b>🧪 Топ стратегій за бектестом:</b>")
        for r in top_bt:
            bw, bt = int(r.get("wins") or 0), int(r.get("total") or 0)
            if bt:
                lines.append(f"  {wr_badge(bw, bt, payout, wr_threshold)} <code>{html.escape(r['strategy_key'])}</code> — {bw / bt * 100:.1f}% ({bw}/{bt})")

    return "\n".join(lines)


def strategy_stat_text(strategy_key: str, live_rows: list, bt_rows: list, payout: float = PAYOUT) -> str:
    lw = sum(int(r.get("wins") or 0) for r in live_rows)
    lt = sum(int(r.get("total") or 0) for r in live_rows)
    bw = sum(int(r.get("wins") or 0) for r in bt_rows)
    bt = sum(int(r.get("total") or 0) for r in bt_rows)
    be = break_even_pct(payout)
    l_str = f"{lw / lt * 100:.1f}% ({lw}/{lt})" if lt else "немає даних"
    b_str = f"{bw / bt * 100:.1f}% ({bw}/{bt})" if bt else "немає даних"
    return f"📊 {html.escape(str(strategy_key))}\nLive: {l_str}\nBT: {b_str}\nБеззбиток ({int(payout * 100)}%): {be:.1f}%"
