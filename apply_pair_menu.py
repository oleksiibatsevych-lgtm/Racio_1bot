"""One-time, idempotent patch: restore a simple pair picker for manual scans.

Run from the repository root with: python3 apply_pair_menu.py
It edits only the existing manual-scan section in bot.py. Review the diff before restart.
"""
from pathlib import Path

path = Path(__file__).with_name("bot.py")
source = path.read_text(encoding="utf-8")

old_scan = '''    if text == "📡 Сигнал зараз":
        status = message.reply_text("📡 Перевіряю 21 пару…")
        def scan():
            count = sum(analyze_single_pair(chat_id, pair, ticker) for pair, ticker in PAIRS_MAP.items())
            bot.edit_message_text(chat_id=chat_id, message_id=status.message_id, text=f"📡 Завершено · сигналів: {count}")
        threading.Thread(target=scan, daemon=True).start()
'''
new_scan = '''    if text == "📡 Сигнал зараз":
        message.reply_text("Оберіть, що перевірити:", reply_markup=manual_scan_keyboard())
'''

keyboard_anchor = '''def stats_keyboard():
    return InlineKeyboardMarkup([[InlineKeyboardButton(title, callback_data="stats|" + period)
                                  for period, title in PERIODS.items()]])
'''
keyboard_replacement = keyboard_anchor + '''\n\ndef manual_scan_keyboard():
    buttons = [[InlineKeyboardButton("🌐 Усі валютні пари", callback_data="scan|ALL")]]
    pairs = list(PAIRS_MAP)
    buttons.extend([[InlineKeyboardButton(pair, callback_data="scan|" + pair)
                     for pair in pairs[i:i + 2]] for i in range(0, len(pairs), 2)])
    return InlineKeyboardMarkup(buttons)
'''

callback_anchor = '''    if action == "stats" and len(parts) == 2 and parts[1] in PERIODS:
        query.edit_message_text(show_stats(chat_id, parts[1]), parse_mode="HTML", reply_markup=stats_keyboard())
'''
callback_replacement = callback_anchor + '''    elif action == "scan" and len(parts) == 2:
        if parts[1] == "ALL":
            selected_pairs = list(PAIRS_MAP.items())
            label = "усі пари"
        elif parts[1] in PAIRS_MAP:
            selected_pairs = [(parts[1], PAIRS_MAP[parts[1]])]
            label = parts[1]
        else:
            query.edit_message_text("Невідома валютна пара.")
            return
        query.edit_message_text(f"🔎 Перевіряю {label}. Результат надішлю окремим повідомленням.")
        def scan_selected_pairs():
            count = sum(analyze_single_pair(chat_id, pair, ticker)
                        for pair, ticker in selected_pairs)
            bot.send_message(chat_id=chat_id, text=f"✅ Перевірку завершено · сигналів: {count}")
        threading.Thread(target=scan_selected_pairs, daemon=True).start()
'''

if "def manual_scan_keyboard():" not in source:
    if keyboard_anchor not in source:
        raise SystemExit("Could not find stats_keyboard() anchor; bot.py left unchanged.")
    source = source.replace(keyboard_anchor, keyboard_replacement, 1)

if 'message.reply_text("Оберіть, що перевірити:", reply_markup=manual_scan_keyboard())' not in source:
    if old_scan not in source:
        raise SystemExit("Could not find existing manual scan block; bot.py left unchanged.")
    source = source.replace(old_scan, new_scan, 1)

if 'elif action == "scan" and len(parts) == 2:' not in source:
    if callback_anchor not in source:
        raise SystemExit("Could not find callback handler anchor; bot.py left unchanged.")
    source = source.replace(callback_anchor, callback_replacement, 1)

path.write_text(source, encoding="utf-8")
print("Manual pair picker installed in bot.py. Review with: git diff -- bot.py")
