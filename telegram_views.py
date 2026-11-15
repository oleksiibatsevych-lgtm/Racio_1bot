# -*- coding: utf-8 -*-
import html
from datetime import timedelta

from reversion_engine import PRESETS
from strategy_engine import ALL_STRATEGY_KEYS, STRATEGIES_META, STRATEGY_CATEGORIES, get_strategy_expiration
from trading_sessions import KYIV, session_name, utc_datetime

HOLDOUT = {"REV_RSI14": (276, 67.03), "REV_BB3": (351, 65.24), "REV_RSI7": (921, 60.80)}
PERIODS = {"today": "сьогодні", "7": "7 днів", "30": "30 днів", "all": "все"}


def progress_bar(percent):
    filled = max(0, min(10, round(percent / 10)))
    return "█" * filled + "░" * (10 - filled)


def wr_line(title, rows):
    decisive = [r for r in rows if r.get("status") == "CLOSED" and r.get("result") in ("WIN", "LOSS")]
    n = len(decisive)
    wins = sum(r["result"] == "WIN" for r in decisive)
    percent = 100 * wins / n if n else 0
    return f"{html.escape(title)}: {'%.1f%%' % percent if n else '—'} · n={n}\n{progress_bar(percent)}"


def signal_card(pair, cand, entry, now):
    preset = cand["strategy"]
    exp = get_strategy_expiration(cand)
    tf = cand.get("tf") or STRATEGIES_META.get(preset, {}).get("tf", "5m")
    expires = utc_datetime(now) + timedelta(minutes=exp)
    direction = "🟢 CALL ↑" if cand["signal"] == "CALL" else "🔴 PUT ↓"
    price_fmt = f"{entry:.3f}" if "JPY" in pair else f"{entry:.5f}"

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
        footer = f"📂 {html.escape(cat_name)}{extra_str}\n🧪 Нова стратегія · перевірте через /lab та /backtest"

    return (
        f"<b>{html.escape(pair)} · {direction}</b>\n"
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


def stats_text(rows, period):
    lines = [f"📊 <b>Статистика · {PERIODS[period]}</b>", wr_line("Загалом", rows), "\n<b>Пресети</b>"]
    for preset in PRESETS:
        lines.append(wr_line(preset, [r for r in rows if r.get("strategy") == preset]))

    extra_rows = []
    for key in ALL_STRATEGY_KEYS:
        if key in PRESETS:
            continue
        sub = [r for r in rows if r.get("strategy") == key and r.get("status") == "CLOSED" and r.get("result") in ("WIN", "LOSS")]
        if sub:
            extra_rows.append((key, sub))
    if extra_rows:
        extra_rows.sort(key=lambda item: len(item[1]), reverse=True)
        lines.append("\n<b>Нові стратегії (з угодами)</b>")
        for key, sub in extra_rows[:8]:
            lines.append(wr_line(key, sub))

    pairs = sorted({r["ticker"] for r in rows if r.get("status") == "CLOSED" and r.get("result") in ("WIN", "LOSS")})
    groups = [(pair, [r for r in rows if r["ticker"] == pair and r.get("status") == "CLOSED" and r.get("result") in ("WIN", "LOSS")]) for pair in pairs]
    groups.sort(key=lambda item: sum(r["result"] == "WIN" for r in item[1]) / len(item[1]), reverse=True)
    for title, selected in [("Топ пар", groups[:3]), ("Низ пар", groups[-3:])]:
        lines.append(f"\n<b>{title}</b>")
        lines.extend(wr_line(pair, group) for pair, group in selected)
    if not rows:
        lines.append("Угод за цей період немає.")
    return "\n".join(lines)


def active_text(rows, now):
    lines = ["⏳ <b>Активні угоди</b>"]
    for r in rows:
        expiry = r.get("expires_at") or utc_datetime(r["created_at"]) + timedelta(minutes=r["expiration_mins"])
        seconds = max(0, int((utc_datetime(expiry) - utc_datetime(now)).total_seconds()))
        lines.append(
            f"#{r['id']} {html.escape(r['ticker'])} · {html.escape(r['strategy'] or '')}\n"
            f"{html.escape(r['signal_type'])} · {seconds // 60}:{seconds % 60:02d} до експірації"
            if seconds
            else f"#{r['id']} {html.escape(r['ticker'])} · очікуємо результат"
        )
    if not rows:
        lines.append("Відкритих угод немає.")
    return "\n".join(lines)


def history_text(rows):
    lines = ["📜 <b>Історія · останні 15</b>"]
    for r in rows:
        result = "VOID" if r["status"] == "VOID" else r["result"]
        lines.append(
            f"#{r['id']} {html.escape(r['ticker'])} · {html.escape(r['strategy'] or '')}\n"
            + settlement_text(result, float(r["exit_price"]) if r.get("exit_price") is not None else None, r.get("pips") or 0)
        )
    return "\n".join(lines) if rows else lines[0] + "\nЗавершених угод немає."
