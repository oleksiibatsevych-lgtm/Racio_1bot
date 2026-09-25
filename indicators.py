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
                df['bb_middle'] = df[[c for c in df.columns if 'BBM' in c][0]]
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

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
        if df is None or df.empty or len(df) < span_val:
            return "NEUTRAL"

        close = float(df['close'].iloc[-1])
        try:
            ema_col = f"EMA_{span_val}"
            if ema_col in df.columns:
                ema = float(df[ema_col].iloc[-1])
            else:
                ema = float(df['close'].ewm(span=span_val, adjust=False).mean().iloc[-1])

            threshold = ema * 0.0005  # Мертва зона 0.05%

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

    def generate_signal(
        self, global_trend: str, mid_trend: str, df_daily: pd.DataFrame, tf_dict: dict
    ) -> dict:
        df_1m = tf_dict.get("1m")
        df_3m = tf_dict.get("3m")
        df_5m = tf_dict.get("5m")
        
        if df_5m is None or df_fast is None if False else df_5m.empty or 'rsi' not in df_5m.columns:
            return {"signal": "NONE", "score": 0, "reason": "Недостатньо даних (5m)"}

        current_price = float(df_5m['close'].iloc[-1])
        rsi_5m = float(df_fast.get('rsi', pd.Series([50])).iloc[-1]) if 'df_fast' in locals() else float(df_5m['rsi'].iloc[-1])
        adx_5m = float(df_5m['adx'].iloc[-1]) if 'adx' in df_5m.columns else 20.0
        
        volatility_ratio = float(df_5m['volatility_ratio'].iloc[-1]) if 'volatility_ratio' in df_5m.columns else 1.0

        bb_upper_5m = float(df_5m['bb_upper'].iloc[-1]) if 'bb_upper' in df_5m.columns else 0.0
        bb_lower_5m = float(df_5m['bb_lower'].iloc[-1]) if 'bb_lower' in df_5m.columns else 0.0
        
        divergence = self.check_divergence(df_5m)
        pivots = self.calculate_pivots(df_daily if df_daily is not None else tf_dict.get("1h"))
        local_trend = self.get_trend(df_5m, span_val=10)

        # ----------------------------------------------------
        # ГІЛКА 1: МІКРО-ФЛЕТ (Скальпінг на 1m та 3m)
        # ----------------------------------------------------
        if df_1m is not None and not df_1m.empty and 'rsi' in df_1m.columns:
            rsi_1m = float(df_1m['rsi'].iloc[-1])
            bb_lower_1m = float(df_1m['bb_lower'].iloc[-1]) if 'bb_lower' in df_1m.columns else 0.0
            bb_upper_1m = float(df_1m['bb_upper'].iloc[-1]) if 'bb_upper' in df_1m.columns else 0.0
            
            if bb_lower_1m > 0 and current_price <= bb_lower_1m and rsi_1m <= 32.0:
                return {
                    "signal": "CALL",
                    "score": 80,
                    "primary_tf": "1m",
                    "strategy_type": "BOUNCE",
                    "rsi": rsi_1m, "adx": adx_5m,
                    "divergence": divergence,
                    "volatility_ratio": volatility_ratio,
                    "reason": f"Мікро-флет 1m: Відскок від низу Боллінджера (RSI: {rsi_1m:.1f})",
                    "suggested_exp": 3
                }
            if bb_upper_1m > 0 and current_price >= bb_upper_1m and rsi_1m >= 68.0:
                return {
                    "signal": "PUT",
                    "score": 80,
                    "primary_tf": "1m",
                    "strategy_type": "BOUNCE",
                    "rsi": rsi_1m, "adx": adx_5m,
                    "divergence": divergence,
                    "volatility_ratio": volatility_ratio,
                    "reason": f"Мікро-флет 1m: Відскок від верху Боллінджера (RSI: {rsi_1m:.1f})",
                    "suggested_exp": 3
                }

        # ----------------------------------------------------
        # ГІЛКА 2: СТАНДАРТНИЙ ФЛЕТ НА 5m
        # ----------------------------------------------------
        if adx_5m < 22.0:
            if bb_lower_5m > 0 and current_price <= bb_lower_5m * 1.0005 and rsi_5m <= 40.0:
                return {
                    "signal": "CALL",
                    "score": 75,
                    "primary_tf": "5m",
                    "strategy_type": "BOUNCE",
                    "rsi": rsi_5m, "adx": adx_5m,
                    "divergence": divergence,
                    "volatility_ratio": volatility_ratio,
                    "reason": f"Флет 5m: Відскок від нижньої межі каналу (RSI: {rsi_5m:.1f})",
                    "suggested_exp": 5
                }

            if bb_upper_5m > 0 and current_price >= bb_upper_5m * 0.9995 and rsi_5m >= 60.0:
                return {
                    "signal": "PUT",
                    "score": 75,
                    "primary_tf": "5m",
                    "strategy_type": "BOUNCE",
                    "rsi": rsi_5m, "adx": adx_5m,
                    "divergence": divergence,
                    "volatility_ratio": volatility_ratio,
                    "reason": f"Флет 5m: Відскок від верхньої межі каналу (RSI: {rsi_5m:.1f})",
                    "suggested_exp": 5
                }

            supports = [pivots.get("S1"), pivots.get("S2"), pivots.get("S3")]
            for s_level in supports:
                if s_level and s_level > 0 and abs(current_price - s_level) / current_price <= 0.0012:
                    if rsi_5m <= 45.0:
                        return {
                            "signal": "CALL",
                            "score": 80,
                            "primary_tf": "5m",
                            "strategy_type": "BOUNCE",
                            "rsi": rsi_5m, "adx": adx_5m,
                            "divergence": divergence,
                            "volatility_ratio": volatility_ratio,
                            "reason": f"Флет 5m: Відскок від підтримки Pivot ({s_level:.5f})",
                            "suggested_exp": 5
                        }

            resistances = [pivots.get("R1"), pivots.get("R2"), pivots.get("R3")]
            for r_level in resistances:
                if r_level and r_level > 0 and abs(r_level - current_price) / current_price <= 0.0012:
                    if rsi_5m >= 55.0:
                        return {
                            "signal": "PUT",
                            "score": 80,
                            "primary_tf": "5m",
                            "strategy_type": "BOUNCE",
                            "rsi": rsi_5m, "adx": adx_5m,
                            "divergence": divergence,
                            "volatility_ratio": volatility_ratio,
                            "reason": f"Флет 5m: Відскок від опору Pivot ({r_level:.5f})",
                            "suggested_exp": 5
                        }

        # ----------------------------------------------------
        # ГІЛКА 3: ТРЕНДОВА СТРАТЕГІЯ (1h + 15m + 5m)
        # ----------------------------------------------------
        else:
            if global_trend == "BULLISH" and mid_trend == "BULLISH" and local_trend == "BULLISH":
                if rsi_5m < 62.0:
                    score = 70 + (15 if divergence == "BULLISH" else 0)
                    return {
                        "signal": "CALL",
                        "score": score,
                        "primary_tf": "5m",
                        "strategy_type": "TREND",
                        "rsi": rsi_5m, "adx": adx_5m,
                        "divergence": divergence,
                        "volatility_ratio": volatility_ratio,
                        "reason": f"Тренд ВГОРУ: Синхронізація ТФ (ADX: {adx_5m:.1f})",
                        "suggested_exp": 10
                    }

            if global_trend == "BEARISH" and mid_trend == "BEARISH" and local_trend == "BEARISH":
                if rsi_5m > 38.0:
                    score = 70 + (15 if divergence == "BEARISH" else 0)
                    return {
                        "signal": "PUT",
                        "score": score,
                        "primary_tf": "5m",
                        "strategy_type": "TREND",
                        "rsi": rsi_5m, "adx": adx_5m,
                        "divergence": divergence,
                        "volatility_ratio": volatility_ratio,
                        "reason": f"Тренд ВНИЗ: Синхронізація ТФ (ADX: {adx_5m:.1f})",
                        "suggested_exp": 10
                    }

            if divergence == "BULLISH" and rsi_5m < 50.0:
                return {
                    "signal": "CALL",
                    "score": 75,
                    "primary_tf": "5m",
                    "strategy_type": "TREND",
                    "rsi": rsi_5m, "adx": adx_5m,
                    "divergence": divergence,
                    "volatility_ratio": volatility_ratio,
                    "reason": f"Трендова бычача дивергенція (ADX: {adx_5m:.1f})",
                    "suggested_exp": 10
                }

            if divergence == "BEARISH" and rsi_5m > 50.0:
                return {
                    "signal": "PUT",
                    "score": 75,
                    "primary_tf": "5m",
                    "strategy_type": "TREND",
                    "rsi": rsi_5m, "adx": adx_5m,
                    "divergence": divergence,
                    "volatility_ratio": volatility_ratio,
                    "reason": f"Трендова ведмежа дивергенція (ADX: {adx_5m:.1f})",
                    "suggested_exp": 10
                }

        return {"signal": "NONE", "score": 0, "reason": "Умови не сформовані"}
