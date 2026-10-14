# -*- coding: utf-8 -*-
import pandas as pd
import yfinance as yf
from indicators import AdaptiveTechnicalAnalysis
from config import YAHOO_PAIRS_MAP

def run_backtest(pair_name="EUR/USD", period="30d") -> str:
    ticker = YAHOO_PAIRS_MAP.get(pair_name, "EURUSD=X")
    
    df = yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=True)
    if df.empty:
        return f"❌ Не вдалося завантажити історичні дані для {pair_name}."

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    analyzer = AdaptiveTechnicalAnalysis()
    wins, losses = 0, 0

    for i in range(50, len(df) - 3):
        df_slice = df.iloc[:i]
        df_1m = df_slice.resample("1min").ffill().tail(100)
        df_3m = df_slice.resample("3min").ffill().tail(100)
        df_5m = df_slice.resample("5min").ffill().tail(100)
        df_15m = df_slice.resample("15min").ffill().tail(100)

        analysis = analyzer.analyze_all_timeframes(df_slice, df_slice, df_15m, df_5m, df_3m, df_1m)
        signal = analysis.get("signal")

        if signal in ["CALL", "PUT"]:
            entry_price = analysis.get("current_price")
            future_price = df['close'].iloc[i + 2]
            
            if signal == "CALL":
                result = "WIN" if future_price > entry_price else "LOSS"
            else:
                result = "WIN" if future_price < entry_price else "LOSS"
                
            if result == "WIN": wins += 1
            else: losses += 1

    total = wins + losses
    winrate = (wins / total * 100) if total > 0 else 0
    
    report = (
        f"📊 <b>Результати бектесту матриці ({pair_name}):</b>\n\n"
        f"• Період: <b>{period} (1H дані)</b>\n"
        f"• Всього сигналів: <b>{total}</b>\n"
        f"• ✅ WIN: <b>{wins}</b>\n"
        f"• ❌ LOSS: <b>{losses}</b>\n"
        f"• 🏆 Вінрейт: <b>{winrate:.1f}%</b>"
    )
    return report

if __name__ == "__main__":
    print(run_backtest("EUR/USD", "30d"))
