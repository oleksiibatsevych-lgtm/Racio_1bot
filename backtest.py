# -*- coding: utf-8 -*-
import os
import json
import logging
import pandas as pd
import yfinance as yf
from indicators import AdaptiveTechnicalAnalysis
from config import YAHOO_PAIRS_MAP, PAIRS_MAP

logger = logging.getLogger(__name__)
analyzer = AdaptiveTechnicalAnalysis()

PERFORMANCE_CACHE_FILE = "pair_strategy_performance.json"

def clean_strategy_name(title: str) -> str:
    return title.replace("📊 ", "").replace("📉 ", "").replace("⚡ ", "").replace("📈 ", "").replace("📐 ", "").replace("🕯️ ", "").replace("💥 ", "").replace("🚀 ", "").strip()

def clear_backtest_cache():
    """Повністю очищає кеш бектесту та результатів стратегій для тестування нових параметрів."""
    try:
        if os.path.exists(PERFORMANCE_CACHE_FILE):
            os.remove(PERFORMANCE_CACHE_FILE)
            return True, "Кеш бектесту та результатів стратегій успішно видалено."
        return True, "Кеш файл був відсутній, але стан чистий."
    except Exception as e:
        return False, f"Помилка очищення кешу: {e}"

def run_backtest(pair_name: str, period: str = "30d") -> str:
    yahoo_ticker = YAHOO_PAIRS_MAP.get(pair_name, f"{pair_name.replace('/', '')}=X")
    try:
        df = yf.download(tickers=yahoo_ticker, period=period, interval="1h", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return f"❌ Не вдалося завантажити дані для бектесту {pair_name}"
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
        
        if len(df) < 35:
            return f"⚠️ Замало свічок для бектесту {pair_name}"

        total_signals = 0
        wins = 0
        strategy_stats = {}

        for i in range(30, len(df) - 3):
            sub_df = df.iloc[:i]
            analysis = analyzer.analyze_all_timeframes(sub_df, sub_df, sub_df, sub_df, sub_df, sub_df, pair_name=pair_name)
            sig = analysis.get("signal")
            
            if sig in ["CALL", "PUT"]:
                strat_raw = analysis.get("strategy_title", "Matrix")
                strat_name = clean_strategy_name(strat_raw)
                
                if strat_name not in strategy_stats:
                    strategy_stats[strat_name] = {"total": 0, "wins": 0}
                
                strategy_stats[strat_name]["total"] += 1
                total_signals += 1
                
                entry_p = float(sub_df["close"].iloc[-1])
                exit_p = float(df["close"].iloc[i + 3])
                
                if exit_p != entry_p:
                    is_win = (exit_p > entry_p) if sig == "CALL" else (exit_p < entry_p)
                    if is_win:
                        wins += 1
                        strategy_stats[strat_name]["wins"] += 1

        winrate = (wins / total_signals * 100) if total_signals > 0 else 0.0
        
        report_lines = [
            f"📊 <b>Результати бектесту для <code>{pair_name}</code> ({period}):</b>\n",
            f"🎯 Всього сигналів: <b>{total_signals}</b>",
            f"✅ Перемог (WIN): <b>{wins}</b>",
            f"🏆 Загальний вінрейт: <b>{winrate:.1f}%</b>\n",
            "📋 <b>Ефективність стратегій:</b>"
        ]
        
        for s_name, s_data in strategy_stats.items():
            s_tot = s_data["total"]
            s_wins = s_data["wins"]
            s_wr = (s_wins / s_tot * 100) if s_tot > 0 else 0.0
            icon = "🟢" if s_wr >= 53 else "🔴"
            report_lines.append(f"{icon} <b>{s_name}</b>: {s_wins}/{s_tot} ({s_wr:.1f}% WR)")
            
        return "\n".join(report_lines)
    except Exception as e:
        return f"❌ Помилка виконання бектесту: {e}"


def run_backtest_all_detailed(period: str = "30d") -> list:
    """
    Запускає детальний бектест по ВСІХ парах ТА по ВСІХ стратегіях.
    Повертає список відформатованих повідомлень для відправки в Telegram.
    """
    global_total = 0
    global_wins = 0
    global_strat_stats = {}
    pair_detailed_results = {}

    for p_name in PAIRS_MAP.keys():
        try:
            yahoo_ticker = YAHOO_PAIRS_MAP.get(p_name, f"{p_name.replace('/', '')}=X")
            df = yf.download(tickers=yahoo_ticker, period=period, interval="1h", progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df.dropna(subset=["open", "high", "low", "close"], inplace=True)

            if len(df) < 35:
                continue

            pair_tot = 0
            pair_wins = 0
            p_strats = {}

            for i in range(30, len(df) - 3):
                sub_df = df.iloc[:i]
                analysis = analyzer.analyze_all_timeframes(sub_df, sub_df, sub_df, sub_df, sub_df, sub_df, pair_name=p_name)
                sig = analysis.get("signal")

                if sig in ["CALL", "PUT"]:
                    strat_raw = analysis.get("strategy_title", "Matrix")
                    strat_name = clean_strategy_name(strat_raw)

                    if strat_name not in p_strats:
                        p_strats[strat_name] = {"total": 0, "wins": 0}
                    if strat_name not in global_strat_stats:
                        global_strat_stats[strat_name] = {"total": 0, "wins": 0}

                    p_strats[strat_name]["total"] += 1
                    global_strat_stats[strat_name]["total"] += 1
                    pair_tot += 1
                    global_total += 1

                    entry_p = float(sub_df["close"].iloc[-1])
                    exit_p = float(df["close"].iloc[i + 3])

                    if exit_p != entry_p:
                        is_win = (exit_p > entry_p) if sig == "CALL" else (exit_p < entry_p)
                        if is_win:
                            p_strats[strat_name]["wins"] += 1
                            global_strat_stats[strat_name]["wins"] += 1
                            pair_wins += 1
                            global_wins += 1

            p_wr = (pair_wins / pair_tot * 100) if pair_tot > 0 else 0.0
            pair_detailed_results[p_name] = {
                "total": pair_tot,
                "wins": pair_wins,
                "winrate": p_wr,
                "strategies": p_strats
            }
        except Exception as e:
            logger.error(f"Помилка бектесту для {p_name}: {e}")
            continue

    global_wr = (global_wins / global_total * 100) if global_total > 0 else 0.0

    msg_header = [
        f"🌐 <b>ГЛОБАЛЬНИЙ ЗВІТ БЕКТЕСТУ ({period})</b>\n"
        "──────────────────────────────────────",
        f"🎯 Всього сигналів по всіх парах: <b>{global_total}</b>",
        f"✅ Перемог (WIN): <b>{global_wins}</b>",
        f"🏆 Глобальний вінрейт: <b>{global_wr:.1f}%</b>\n",
        "📊 <b>ЗВЕДЕННЯ ПО СТРАТЕГІЯХ (УСІ ПАРИ):</b>"
    ]

    for s_name, s_data in global_strat_stats.items():
        s_tot = s_data["total"]
        s_wins = s_data["wins"]
        s_wr = (s_wins / s_tot * 100) if s_tot > 0 else 0.0
        icon = "🟢" if s_wr >= 53 else ("🟡" if s_wr >= 48 else "🔴")
        msg_header.append(f"{icon} <b>{s_name}</b>: {s_wins}/{s_tot} ({s_wr:.1f}% WR)")

    msg_header.append("\n🔀 <b>РОЗБИВКА ПО ПАРАХ ТА СТРАТЕГІЯХ:</b>\n")

    sorted_pairs = sorted(pair_detailed_results.items(), key=lambda x: x[1]["winrate"], reverse=True)

    messages = []
    current_msg = "\n".join(msg_header) + "\n"

    for p_name, p_data in sorted_pairs:
        p_ico = "🟢" if p_data["winrate"] >= 53 else ("🟡" if p_data["winrate"] >= 48 else "🔴")
        p_block = f"{p_ico} <b>{p_name}</b> — <b>{p_data['winrate']:.1f}% WR</b> ({p_data['wins']}/{p_data['total']})\n"
        for st_n, st_d in p_data["strategies"].items():
            st_wr = (st_d["wins"] / st_d["total"] * 100) if st_d["total"] > 0 else 0.0
            st_ico = "🟢" if st_wr >= 53 else "🔴"
            p_block += f"  └ {st_ico} {st_n}: {st_d['wins']}/{st_d['total']} ({st_wr:.1f}%)\n"
        p_block += "\n"

        if len(current_msg + p_block) > 3500:
            messages.append(current_msg)
            current_msg = p_block
        else:
            current_msg += p_block

    if current_msg:
        messages.append(current_msg)

    return messages
