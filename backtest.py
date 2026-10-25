# -*- coding: utf-8 -*-
import os
import json
import logging
import pandas as pd
import yfinance as yf
from indicators import AdaptiveTechnicalAnalysis
from config import YAHOO_PAIRS_MAP

logger = logging.getLogger(__name__)
analyzer = AdaptiveTechnicalAnalysis()

PERFORMANCE_CACHE_FILE = "pair_strategy_performance.json"

def clean_strategy_name(title: str) -> str:
    return title.replace("📊 ", "").replace("📉 ", "").replace("⚡ ", "").replace("📈 ", "").replace("📐 ", "").replace("🕯️ ", "").replace("💥 ", "").strip()

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
