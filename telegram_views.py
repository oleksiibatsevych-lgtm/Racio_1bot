# -*- coding: utf-8 -*-
import html
from datetime import timedelta

from config import AUTO_DISABLE_WR_THRESHOLD, PAYOUT, normalize_pair_name
from reversion_engine import PRESETS
from strategy_engine import ALL_STRATEGY_KEYS, STRATEGIES_META, STRATEGY_CATEGORIES, get_strategy_expiration
from trading_sessions import KYIV, session_name, utc_datetime

HOLDOUT = {"REV_RSI14": (276, 67.03), "REV_BB3": (351, 65.24), "REV_RSI7": (921, 60.80)}
PERIODS = {"today": "сьогодні", "7": "7 днів", "30": "30 днів", "all": "все"}
BREAK_EVEN_PCT = 100.0 / (1.0 + PAYOUT)


def format_pair_name(ticker: str | None) -> str:
    return normalize_pair_name(ticker) or "—"


def wr_badge(wins: int, total: int) -> str:
    if total <= 0:
        return "⚪"
    wr = wins / total
    if wr * 100.0 >= BREAK_EVEN_PCT:
        return "🟢"
    if wr >= AUTO_DISABLE_WR_THRESHOLD:
        return "🟡"
    return "🔴"


def progress_bar(percent):
    filled = max(0, min(10, round(percent / 10)))
    return "█" * filled + "░" * (10 - filled)


def wr_line(title, rows, auto_disabled=False):
    decisive = [r for r in rows if r.get("status") == "CLOSED" and r.get("result") in ("WIN", "LOSS")]
    n = len(decisive)
    wins = sum(r["result"] == "WIN" for r in decisive)
    losses = n - wins
    percent = 100.0 * wins / n if n else 0.0
    ico = wr_badge(wins, n)
    suffix = " ⛔" if auto_disabled else ""
    clean_title = html.escape(format_pair_name(title) if (":" in title or "_" in title and "/" not in title and title not in STRATEGIES_META) else title)
    if not n:
        return f"{ico} <b>{clean_title}</b> — немає угод{suffix}"
    return f"{ico} <b>{clean_title}</b> — <b>{percent:.1f}%</b> ({wins}✅ / {losses}❌ · {n}){suffix}"


def signal_card(pair, cand, entry, now):
    preset = cand["strategy"]
    exp = get_strategy_expiration(cand)
    tf = cand.get("tf") or STRATEGIES_META.get(preset, {}).get("tf", "5m")
    expires = utc_datetime(now) + timedelta(minutes=exp)
    direction = "🟢 CALL ↑" if cand["signal"] == "CALL" else "🔴 PUT ↓"
    clean_pair = format_pair_name(pair)
    price_fmt = f"{entry:.3f}" if "JPY" in clean_pair else f"{entry:.5f}"

    if preset in HOLDOUT:
        n, wr = HOLDOUT[preset]
        footer = f"📚 Research holdout: {wr:.2f}% · n={n}; не прогноз"
    else:
        meta = STRATEGIES_META.get(preset, {})
        cat_name = STRATEGY_CATEGORIES.get(meta.get("category", ""), "Стратегія")
        extras = []
        if cand.get("chop") is not None:
            extras.append(f"CHOP {cand['chop']:.1f}")
        if cand.get("adx") is not None:
            extras.append(f"ADX {cand['adx']:.1f}")
        extra_str = (" · " + " | ".join(extras)) if extras else ""
        footer = f"📂 {html.escape(cat_name)}{extra_str}\n🧪 Перевірено за правилами стратегії"

    return (
        f"<b>{html.escape(clean_pair)} · {direction}</b>\n"
        f"<code>{html.escape(preset)}</code> · {html.escape(tf)}\n"
        f"⏱ {exp} хв — фіксована · до {expires:%H:%M} UTC / {expires.astimezone(KYIV):%H:%M} Київ\n"
        f"📍 Вхід: <b>{price_fmt}</b> · RSI: {cand['rsi']:.1f}\n"
        f"🌐 Сесія: {session_name(cand['bar_end'])}\n"
        f"{footer}"
    )


def settlement_text(result, exit_price, pips):
    icon = {"WIN": "✅", "LOSS": "❌", "NEUTRAL": "➖", "VOID": "⚪"}[result]
    price = f"{exit_price:.5f}" if exit_price is not None else "немає котирування"
    movement = f"{pips / 10:+.1f}" if exit_price is not None else "—"
    return f"🏁 <b>{icon} {result}</b> · Вихід: {price} · піпси: {movement}"


def stats_text(rows, period, auto_disabled_global=None):
    disabled_set = set(auto_disabled_global or ())
    decisive = [r for r in rows if r.get("status") == "CLOSED" and r.get("result") in ("WIN", "LOSS")]
    total = len(decisive)
    wins = sum(r["result"] == "WIN" for r in decisive)
    losses = total - wins
    wr_pct = (100.0 * wins / total) if total else 0.0
    ev_pct = ((wins / total) * PAYOUT - (1.0 - wins / total)) * 100.0 if total else 0.0

    lines = [f"📊 <b>Статистика · {PERIODS[period]}</b>"]
    if not total:
        lines.append("⚪ <b>Загалом:</b> немає завершених угод")
        lines.append("Угод за цей період немає.")
        return "\n".join(lines)

    lines.append(
        f"{wr_badge(wins, total)} <b>Загалом:</b> <b>{wr_pct:.1f}%</b> ({wins}✅ / {losses}❌ · {total} угод)"
    )
    lines.append(
        f"<i>Очікування ({int(PAYOUT * 100)}% випл.): {ev_pct:+.1f}% · Беззбиток: {BREAK_EVEN_PCT:.1f}% · Авто-вимк: &lt;{AUTO_DISABLE_WR_THRESHOLD * 100:.0f}%</i>"
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
            lines.append(f"  {wr_badge(sw, sn)} {html.escape(title)} — <b>{wr * 100:.1f}%</b> ({sw}✅/{sn - sw}❌){tag}")

    unknown_keys = sorted({r.get("strategy") for r in decisive if r.get("strategy") not in STRATEGIES_META})
    for uk in unknown_keys:
        if not uk:
            continue
        sub = [r for r in decisive if r.get("strategy") == uk]
        sw = sum(r["result"] == "WIN" for r in sub)
        sn = len(sub)
        lines.append(f"  {wr_badge(sw, sn)} {html.escape(uk)} — <b>{sw / sn * 100:.1f}%</b> ({sw}✅/{sn - sw}❌)")

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
            lines.append(f"  {wr_badge(pw, pn)} <b>{html.escape(p)}</b> — <b>{wr * 100:.1f}%</b> ({pw}✅/{pn - pw}❌)")
    else:
        lines.append("\n<b>🏆 Топ-5 пар</b>")
        for wr, pw, pn, p in ranked_pairs[:5]:
            lines.append(f"  {wr_badge(pw, pn)} <b>{html.escape(p)}</b> — <b>{wr * 100:.1f}%</b> ({pw}✅/{pn - pw}❌)")
        lines.append("\n<b>⚠️ Низ-5 пар</b>")
        for wr, pw, pn, p in ranked_pairs[-5:]:
            lines.append(f"  {wr_badge(pw, pn)} <b>{html.escape(p)}</b> — <b>{wr * 100:.1f}%</b> ({pw}✅/{pn - pw}❌)")

    return "\n".join(lines)


def stats_matrix_text(live_rows, bt_rows) -> str:
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
                parts.append(f"Live {wr_badge(lw, lt)} {lw / lt * 100:.1f}% ({lw}/{lt})")
            if bt:
                parts.append(f"BT {wr_badge(bw, bt)} {bw / bt * 100:.1f}% ({bw}/{bt})")
            lines.append(f"  • <code>{html.escape(s_key)}</code>: " + " | ".join(parts))
    return "\n".join(lines)


def auto_disabled_text(items: list) -> str:
    if not isinstance(items, (list, tuple)) or not items:
        return (
            f"🛡 <b>Авто-вимкнення (&lt;{AUTO_DISABLE_WR_THRESHOLD * 100:.0f}% WR)</b>\n"
            "✅ Жодна стратегія зараз не заблокована."
        )
    lines = [
        f"🚫 <b>Авто-вимкнені стратегії (&lt;{AUTO_DISABLE_WR_THRESHOLD * 100:.0f}% WR)</b>",
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


def active_text(rows, now):
    lines = ["⏳ <b>Активні угоди</b>"]
    for r in rows:
        pair = format_pair_name(r.get("ticker"))
        expiry = r.get("expires_at") or utc_datetime(r["created_at"]) + timedelta(minutes=r["expiration_mins"])
        seconds = max(0, int((utc_datetime(expiry) - utc_datetime(now)).total_seconds()))
        lines.append(
            f"#{r['id']} {html.escape(pair)} · {html.escape(r['strategy'] or '')}\n"
            f"{html.escape(r['signal_type'])} · {seconds // 60}:{seconds % 60:02d} до експірації"
            if seconds
            else f"#{r['id']} {html.escape(pair)} · очікуємо результат"
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
            f"#{r['id']} {html.escape(pair)} · {html.escape(r['strategy'] or '')}\n"
            + settlement_text(result, float(r["exit_price"]) if r.get("exit_price") is not None else None, r.get("pips") or 0)
        )
    return "\n".join(lines) if rows else lines[0] + "\nЗавершених угод немає."
