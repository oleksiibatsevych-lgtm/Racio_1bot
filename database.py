from datetime import datetime, timedelta

stats_history = []
active_signals = {}
signal_counter = 0

def get_statistics():
    now = datetime.now()
    day_ago = now - timedelta(days=1)
    week_ago = now - timedelta(days=7)

    def process_items(items):
        pair_data = {}
        total_wins = 0
        total_valid = 0

        for item in items:
            p = item["pair"]
            res = item["result"]
            if p not in pair_data:
                pair_data[p] = {"requests": 0, "wins": 0, "losses": 0}
            pair_data[p]["requests"] += 1
            if res == "WIN":
                pair_data[p]["wins"] += 1
                total_wins += 1
                total_valid += 1
            elif res == "LOSS":
                pair_data[p]["losses"] += 1
                total_valid += 1

        result = {}
        for p, d in pair_data.items():
            valid = d["wins"] + d["losses"]
            wr = (d["wins"] / valid * 100) if valid > 0 else 0.0
            result[p] = {
                "requests": d["requests"],
                "wins": d["wins"],
                "losses": d["losses"],
                "winrate": round(wr, 1)
            }

        overall_wr = (total_wins / total_valid * 100) if total_valid > 0 else 0.0
        return result, round(overall_wr, 1)

    day_items = [i for i in stats_history if i["timestamp"] >= day_ago]
    week_items = [i for i in stats_history if i["timestamp"] >= week_ago]
    return process_items(day_items), process_items(week_items), process_items(stats_history)

def format_stats_text(title, data_tuple):
    data, overall_wr = data_tuple
    text = f"📊 *{title}*\n"
    if not data:
        text += "\nЩе немає оцінених угод за цей період."
        return text
    text += f"🏆 **Загальний вінрейт:** `{overall_wr}%`\n\n"
    text += "📋 *По парах*:\n"
    for pair, counts in data.items():
        text += f"🌟 *{pair}* Всього: `{counts['requests']}` | ✅ `{counts['wins']}` ❌ `{counts['losses']}` | Вінрейт: `{counts['winrate']}%`\n"
    return text
