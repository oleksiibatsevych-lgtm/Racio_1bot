import pandas as pd
import pandas_ta as ta
import numpy as np


class AdaptiveTechnicalAnalysis:
    def __init__(self):
        pass

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 30:
            return df

        try:
            df.ta.rsi(length=14, append=True)
            df.ta.adx(length=14, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.ta.atr(length=14, append=True)
            
            df.ta.ema(length=10, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)

            if 'RSI_14' in df.columns:
                df['rsi'] = df['RSI_14']
            else:
                df['rsi'] = 50.0

            if 'ADX_14' in df.columns:
                df['adx'] = df['ADX_14']
            else:
                df['adx'] = 20.0

            bb_cols = [c for c in df.columns if 'BBL' in c]
            if bb_cols:
                df['bb_lower'] = df[bb_cols[0]]
                df['bb_upper'] = df[[c for c in df.columns if 'BBU' in c][0]]
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_lower']

            if 'ATR_14' in df.columns:
                df['atr'] = df['ATR_14']
                df['atr_sma'] = df['atr'].rolling(window=14).mean()
                df['volatility_ratio'] = df['atr'] / df['atr_sma']
                df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0)
            else:
                df['volatility_ratio'] = 1.0

        except Exception as e:
            print(f"Помилка розрахунку індикаторів: {e}")

        return df

    def get_trend(self, df: pd.DataFrame, span_val: int = 50) -> str:
        """Визначення тренду з мертвиною зоною 0.05% проти флетового шуму."""
        if df is None or df.empty or len(df) < span_val:
            return "NEUTRAL"

        close = float(df['close'].iloc[-1])
        try:
            ema_col = f"EMA_{span_val}"
            if ema_col in df.columns:
                ema = float(df[ema_col].iloc[-1])
            else:
                ema = float(df['close'].ewm(span=span_val, adjust=False).mean().iloc[-1])

            threshold = ema * 0.0005  # Мертва зона (0.05%)

            if close > (ema + threshold):
                return "BULLISH"
            elif close < (ema - threshold):
                return "BEARISH"
        except Exception:
            pass
        return "NEUTRAL"

    def calculate_pivots(self, df: pd.DataFrame) -> dict:
        pivots = {"P": 0.0, "R1": 0.0, "R2": 0.0, "R3": 0.0, "S1": 0.0, "S2": 0.0, "S3": 0.0}
        if df is None or len(df) < 2:
            return pivots

        prev_candle = df.iloc[-2]
        high = float(prev_candle['high'])
        low = float(prev_candle['low'])
        close = float(prev_candle['close'])

        p = (high + low + close) / 3
        r1 = (2 * p) - low
        s1 = (2 * p) - high
        r2 = p + (high - low)
        s2 = p - (high - low)
        r3 = high + 2 * (p - low)
        s3 = low - 2 * (high - p)

        return {"P": p, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}

    def check_divergence(self, df: pd.DataFrame) -> str:
        if len(df) < 20 or 'rsi' not in df.columns:
            return "NONE"

        recent_price_low = df['low'].iloc[-10:].min()
        past_price_low = df['low'].iloc[-20:-10].min()
        recent_rsi_low = df['rsi'].iloc[-10:].min()
        past_rsi_low = df['rsi'].iloc[-20:-10].min()

        recent_price_high = df['high'].iloc[-10:].max()
        past_price_high = df['high'].iloc[-20:-10].max()
        recent_rsi_high = df['rsi'].iloc[-10:].max()
        past_rsi_high = df['rsi'].iloc[-20:-10].max()

        if recent_price_low < past_price_low and recent_rsi_low > past_rsi_low:
            return "BULLISH"
        if recent_price_high > past_price_high and recent_rsi_high < past_rsi_high:
            return "BEARISH"

        return "NONE"

    def _evaluate_bounce_signal(self, current_price: float, pivots: dict, rsi: float) -> dict:
        if not pivots or current_price <= 0:
            return {"signal": "NONE", "score": 0, "reason": ""}

        resistances = [pivots.get("R1"), pivots.get("R2"), pivots.get("R3")]
        supports = [pivots.get("S1"), pivots.get("S2"), pivots.get("S3")]

        threshold = 0.0030

        for s_level in supports:
            if s_level and s_level > 0:
                dist_pct = (current_price - s_level) / current_price
                if 0 <= dist_pct <= threshold:
                    score = 75 + (15 if rsi < 45 else 0)
                    return {
                        "signal": "CALL",
                        "score": score,
                        "reason": f"Тест підтримки S ({s_level:.5f}) — зона відскоку вгору",
                    }

        for r_level in resistances:
            if r_level and r_level > 0:
                dist_pct = (r_level - current_price) / current_price
                if 0 <= dist_pct <= threshold:
                    score = 75 + (15 if rsi > 55 else 0)
                    return {
                        "signal": "PUT",
                        "score": score,
                        "reason": f"Тест опору R ({r_level:.5f}) — зона відскоку вниз",
                    }

        return {"signal": "NONE", "score": 0, "reason": ""}

    def generate_signal(
        self, global_trend: str, mid_trend: str, df_daily: pd.DataFrame, tf_dict: dict
    ) -> dict:
        df_fast = tf_dict.get("5m")
        if df_fast is None or df_fast.empty or 'rsi' not in df_fast.columns:
            return {"signal": "NONE", "score": 0, "reason": "Недостатньо даних (5m)"}

        current_price = float(df_fast['close'].iloc[-1])
        rsi = float(df_fast['rsi'].iloc[-1])
        adx = float(df_fast['adx'].iloc[-1]) if 'adx' in df_fast.columns else 20.0
        
        volatility_ratio = 1.0
        if 'volatility_ratio' in df_fast.columns:
            volatility_ratio = float(df_fast['volatility_ratio'].iloc[-1])
            
        divergence = self.check_divergence(df_fast)
        pivots = self.calculate_pivots(df_daily if df_daily is not None else tf_dict.get("1h"))

        # 1. ПРІОРИТЕТ: ВІДСКОК ВІД РІВНІВ
        bounce = self._evaluate_bounce_signal(current_price, pivots, rsi)
        if bounce["signal"] != "NONE":
            return {
                "signal": bounce["signal"],
                "score": bounce["score"],
                "primary_tf": "5m",
                "strategy_type": "BOUNCE",
                "rsi": rsi,
                "adx": adx,
                "divergence": divergence,
                "volatility_ratio": volatility_ratio,
                "wick_ratio": 0.0,
                "ema_dist": 0.0,
                "reason": bounce["reason"],
            }

        # 2. ТРЕНДОВА СТРАТЕГІЯ (з вимовою синхронізації 3-х ТФ: 1h + 15m + 5m)
        local_trend = self.get_trend(df_fast, span_val=10)
        signal = "NONE"
        score = 0
        reason = "Відсутність чіткого трендового сетапу"

        if global_trend == "BULLISH" and mid_trend == "BULLISH" and local_trend == "BULLISH":
            if rsi < 60:  # Вхід на ранній стадії імпульсу
                signal = "CALL"
                score = 70
                reason = "Синхронний бичачий імпульс (1h + 15m + 5m)"
                if divergence == "BULLISH":
                    score += 15

        elif global_trend == "BEARISH" and mid_trend == "BEARISH" and local_trend == "BEARISH":
            if rsi > 40:  # Вхід на ранній стадії імпульсу
                signal = "PUT"
                score = 70
                reason = "Синхронний ведмежий імпульс (1h + 15m + 5m)"
                if divergence == "BEARISH":
                    score += 15

        wick_ratio = 0.0
        ema_dist = 0.0
        try:
            high = float(df_fast['high'].iloc[-1])
            low = float(df_fast['low'].iloc[-1])
            open_p = float(df_fast['open'].iloc[-1])
            close_p = float(df_fast['close'].iloc[-1])
            
            total_len = high - low
            if total_len > 0:
                if signal == "CALL":
                    lower_wick = min(open_p, close_p) - low
                    wick_ratio = lower_wick / total_len
                elif signal == "PUT":
                    upper_wick = high - max(open_p, close_p)
                    wick_ratio = upper_wick / total_len

            if 'EMA_10' in df_fast.columns:
                ema_10 = float(df_fast['EMA_10'].iloc[-1])
                ema_dist = abs(current_price - ema_10) / ema_10
        except Exception:
            pass

        return {
            "signal": signal,
            "score": score,
            "primary_tf": "5m",
            "strategy_type": "TREND",
            "rsi": rsi,
            "adx": adx,
            "divergence": divergence,
            "volatility_ratio": volatility_ratio,
            "wick_ratio": wick_ratio,
            "ema_dist": ema_dist,
            "reason": reason,
        }
