# -*- coding: utf-8 -*-
import pandas as pd
import yfinance as yf
from indicators import AdaptiveTechnicalAnalysis
from config import YAHOO_PAIRS_MAP

def run_backtest(pair_name="EUR/USD", period="30d"):
    yahoo_ticker = YAHOO_PAIRS_MAP.get(pair_name, "EURUSD=X")
    
    try:
        df = yf.download(tickers=yahoo_ticker, period=period, interval="1h", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return f"❌ Не вдалося завантажити історичні дані для бектесту {pair_name}"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df.dropna(subset=["open", "high", "low", "close"], inplace=True)
    except Exception as e:
        return f"❌ Помилка завантаження історії: {e}"

    if len(df) < 50:
        return f"❌ Замало історичних даних для {pair_name}"

    analyzer = AdaptiveTechnicalAnalysis()
    
    total_signals = 0
    total_wins = 0
    total_losses = 0
    
    strategy_stats = {}
    multiplier = 1000 if "JPY" in yahoo_ticker else 100000

    for i in range(30, len(df) - 3):
        slice_df = df.iloc[:i]
        analysis = analyzer.analyze_all_timeframes(slice_df, slice_df, slice_df, slice_df, slice_df, slice_df)
        
        signal = analysis.get("signal")
        if signal in ["CALL", "PUT"]:
            strat_title = analysis.get("strategy_title", "Стратегія")
            
            entry_price = float(slice_df["close"].iloc[-1])
            exit_price = float(df["close"].iloc[i + 3])
            
            raw_pips = (exit_price - entry_price) * multiplier if signal == "CALL" else (entry_price - exit_price) * multiplier
            pips = int(round(raw_pips))
            
            if pips != 0:
                is_win = pips > 0
                total_signals += 1
                if is_win:
                    total_wins += 1
                else:
                    total_losses += 1
                
                if strat_title not in strategy_stats:
                    strategy_stats[strat_title] = {"wins": 0, "losses": 0, "total": 0}
                
                strategy_stats[strat_title]["total"] += 1
                if is_win:
                    strategy_stats[strat_title]["wins"] += 1
                else:
                    strategy_stats[strat_title]["losses"] += 1

    overall_wr = (total_wins / total_signals * 100) if total_signals > 0 else 0.0

    report = (
        f"📊 <b>Результати бектесту матриці ({pair_name}):</b>\n\n"
        f"• Період: <b>{period} (1H дані)</b>\n"
        f"• Всього сигналів: <b>{total_signals}</b>\n"
        f"• ✅ WIN: <b>{total_wins}</b> | ❌ LOSS: <b>{total_losses}</b>\n"
        f"• 🏆 Загальний вінрейт: <b>{overall_wr:.1f}%</b>\n\n"
        f"📋 <b>ТОП стратегій (за вінрейтом):</b>\n"
    )

    sorted_strategies = sorted(
        strategy_stats.items(),
        key=lambda x: (x[1]["wins"] / x[1]["total"] if x[1]["total"] > 0 else 0),
        reverse=True
    )

    # Виводимо топ-20 найефективніших стратегій, щоб звіт не перевищував ліміт Telegram
    for strat, data in sorted_strategies[:20]:
        wr = (data["wins"] / data["total"] * 100) if data["total"] > 0 else 0.0
        icon = "🟢" if wr >= 53 else ("🟡" if wr >= 48 else "🔴")
        report += f"{icon} <b>{strat}</b>: {data['wins']}/{data['total']} ({wr:.1f}% WR)\n"

    if len(sorted_strategies) > 20:
        report += f"\n<i>...та ще {len(sorted_strategies) - 20} стратегій з меншою активністю.</i>"

    return report
