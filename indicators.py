import pandas as pd
import numpy as np

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 14:
            return df
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        # Bollinger Bands
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_width'] = df['bb_width'].fillna(0.001)

        # ATR & ADX
        high = df['high']
        low = df['low']
        close = df['close']
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        df['atr'] = atr.fillna(0.0010)

        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(window=14).mean() / (atr + 1e-9)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(window=14).mean() / (atr + 1e-9)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        df['adx'] = dx.rolling(window=14).mean().fillna(20)

        # EMAs
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # Wick Ratio & EMA Distance
        candle_range = (df['high'] - df['low']).replace(0, 0.00001)
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        df['max_wick'] = np.maximum(upper_wick, lower_wick)
        df['wick_ratio'] = (df['max_wick'] / candle_range).fillna(0.0)
        df['upper_wick_ratio'] = (upper_wick / candle_range).fillna(0.0)
        df['lower_wick_ratio'] = (lower_wick / candle_range).fillna(0.0)

        df['ema_dist'] = ((df['close'] - df['ema_20']).abs() / df['close']).fillna(0.0)

        return df

    def get_trend(self, df: pd.DataFrame, span_val=50) -> str:
        if df is None or df.empty or len(df) < span_val:
            return "NEUTRAL"
        ema = df['close'].ewm(span=span_val, adjust=False).mean()
        current_price = df['close'].iloc[-1]
        current_ema = ema.iloc[-1]
        if current_price > current_ema * 1.0005:
            return "BULLISH"
        elif current_price < current_ema * 0.9995:
            return "BEARISH"
        return "NEUTRAL"

    def calculate_pivots(self, df_macro: pd.DataFrame) -> dict:
        if df_macro is None or df_macro.empty or len(df_macro) < 2:
            return {"P": 0, "R1": 0, "S1": 0, "R2": 0, "S2": 0}
        high = float(df_macro['high'].iloc[-2])
        low = float(df_macro['low'].iloc[-2])
        close = float(df_macro['close'].iloc[-2])
        p = (high + low + close) / 3
        r1 = (2 * p) - low
        s1 = (2 * p) - high
        r2 = p + (high - low)
        s2 = p - (high - low)
        return {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2}

    def detect_divergence(self, df: pd.DataFrame) -> str:
        if df is None or df.empty or 'rsi' not in df.columns or len(df) < 35:
            return "NONE"
        
        recent_prices = df['close'].iloc[-30:].values
        recent_rsi = df['rsi'].iloc[-30:].values
        
        price_lower_low = recent_prices[-1] < recent_prices[-15] and recent_prices[-15] < recent_prices[0]
        rsi_higher_low = recent_rsi[-1] > recent_rsi[-15] and recent_rsi[-15] > recent_rsi[0]
        if price_lower_low and rsi_higher_low:
            return "BULLISH_DIV"

        price_higher_high = recent_prices[-1] > recent_prices[-15] and recent_prices[-15] > recent_prices[0]
        rsi_lower_high = recent_rsi[-1] < recent_rsi[-15] and recent_rsi[-15] > recent_rsi[0]
        if price_higher_high and rsi_lower_high:
            return "BEARISH_DIV"

        return "NONE"

    def get_min_dist_to_pivot_or_round(self, price: float, pivots: dict) -> float:
        if price <= 0:
            return 0.0
        
        distances = []
        for p_name, p_val in pivots.items():
            if p_val > 0:
                distances.append(abs(price - p_val) / price)
        
        round_factor = 100 if price > 50 else 10000
        mod_val = (price * round_factor) % 250
        dist_round = min(mod_val, 250 - mod_val) / (price * round_factor)
        distances.append(dist_round)

        return min(distances) if distances else 0.001

    def _check_trend_following(self, df_5m, df_1m, global_trend, adx) -> dict:
        """Стратегія 1: Вхід за трендом на відкотах (з фільтром розвороту свічки M1)"""
        if adx < 25 or global_trend == "NEUTRAL" or df_5m.empty or df_1m.empty:
            return {"signal": "NONE"}
        
        last_1m = df_1m.iloc[-1]
        rsi_1m = float(last_1m.get('rsi', 50))
        close_1m = float(last_1m['close'])
        open_1m = float(last_1m['open'])
        lower_wick_ratio = float(last_1m.get('lower_wick_ratio', 0))
        upper_wick_ratio = float(last_1m.get('upper_wick_ratio', 0))

        if global_trend == 'BULLISH':
            if 48.0 <= rsi_1m <= 65.0 and (close_1m > open_1m or lower_wick_ratio >= 0.25):
                return {
                    "signal": "CALL",
                    "strategy": "TREND_FOLLOWING",
                    "strategy_title": "1. Вхід за трендом (ADX > 25)",
                    "suggested_exp": 5,
                    "reason": f"Підтверджений відкат у бичачому тренді (ADX: {adx:.1f}, RSI 1m: {rsi_1m:.1f})"
                }

        elif global_trend == 'BEARISH':
            if 35.0 <= rsi_1m <= 52.0 and (close_1m < open_1m or upper_wick_ratio >= 0.25):
                return {
                    "signal": "PUT",
                    "strategy": "TREND_FOLLOWING",
                    "strategy_title": "1. Вхід за трендом (ADX > 25)",
                    "suggested_exp": 5,
                    "reason": f"Підтверджений відкат у ведмежому тренді (ADX: {adx:.1f}, RSI 1m: {rsi_1m:.1f})"
                }

        return {"signal": "NONE"}

    def _check_mean_reversion(self, df_5m, df_1m, adx) -> dict:
        """Стратегія 2: Скальпінг у флеті (ADX < 20)"""
        if adx >= 22 or df_1m.empty:
            return {"signal": "NONE"}

        last_1m = df_1m.iloc[-1]
        close_1m = float(last_1m['close'])
        bb_lower_1m = float(last_1m.get('bb_lower', 0))
        bb_upper_1m = float(last_1m.get('bb_upper', 0))
        rsi_1m = float(last_1m.get('rsi', 50))

        if close_1m <= bb_lower_1m or rsi_1m < 35:
            return {
                "signal": "CALL",
                "strategy": "MEAN_REVERSION",
                "strategy_title": "2. Скальпінг у флеті (BB/RSI)",
                "suggested_exp": 3,
                "reason": f"Флет: відскок від нижньої смуги BB (RSI 1m: {rsi_1m:.1f})"
            }
        elif close_1m >= bb_upper_1m or rsi_1m > 65:
            return {
                "signal": "PUT",
                "strategy": "MEAN_REVERSION",
                "strategy_title": "2. Скальпінг у флеті (BB/RSI)",
                "suggested_exp": 3,
                "reason": f"Флет: відскок від верхньої смуги BB (RSI 1m: {rsi_1m:.1f})"
            }

        return {"signal": "NONE"}

    def _check_breakout(self, df_5m, bb_width, adx) -> dict:
        """Стратегія 3: Пробій стиснення волатильності (BB Squeeze)"""
        if bb_width > 0.0020 or df_5m.empty or len(df_5m) < 3:
            return {"signal": "NONE"}

        last_5m = df_5m.iloc[-1]
        prev_5m = df_5m.iloc[-2]
        close_5m = float(last_5m['close'])
        bb_upper = float(last_5m.get('bb_upper', 0))
        bb_lower = float(last_5m.get('bb_lower', 0))

        if close_5m > bb_upper and float(last_5m['close']) > float(prev_5m['high']):
            return {
                "signal": "CALL",
                "strategy": "BREAKOUT",
                "strategy_title": "3. Пробій стиснення волатильності",
                "suggested_exp": 4,
                "reason": f"Імпульсний пробій флету вгору (BB Width: {bb_width:.5f})"
            }
        elif close_5m < bb_lower and float(last_5m['close']) < float(prev_5m['low']):
            return {
                "signal": "PUT",
                "strategy": "BREAKOUT",
                "strategy_title": "3. Пробій стиснення волатильності",
                "suggested_exp": 4,
                "reason": f"Імпульсний пробій флету вниз (BB Width: {bb_width:.5f})"
            }

        return {"signal": "NONE"}

    def _check_divergence(self, df_5m, div) -> dict:
        """Стратегія 4: Розворот за дивергенцією RSI"""
        if div == "NONE" or df_5m.empty:
            return {"signal": "NONE"}

        last_5m = df_5m.iloc[-1]
        rsi_5m = float(last_5m.get('rsi', 50))

        if div == "BULLISH_DIV" and rsi_5m < 45:
            return {
                "signal": "CALL",
                "strategy": "DIVERGENCE",
                "strategy_title": "4. Розворот за дивергенцією RSI",
                "suggested_exp": 12,
                "reason": f"Быча дивергенція на M5 (RSI: {rsi_5m:.1f})"
            }
        elif div == "BEARISH_DIV" and rsi_5m > 55:
            return {
                "signal": "PUT",
                "strategy": "DIVERGENCE",
                "strategy_title": "4. Розворот за дивергенцією RSI",
                "suggested_exp": 12,
                "reason": f"Ведмежа дивергенція на M5 (RSI: {rsi_5m:.1f})"
            }

        return {"signal": "NONE"}

    def _check_pivot_bounce(self, df_5m, df_1m, pivots) -> dict:
        """Стратегія 5: Відскок від Pivot та круглих рівнів"""
        if not pivots or df_5m.empty or df_1m.empty:
            return {"signal": "NONE"}

        close_1m = float(df_1m['close'].iloc[-1])
        rsi_1m = float(df_1m['rsi'].iloc[-1]) if 'rsi' in df_1m.columns else 50
        adx_5m = float(df_5m['adx'].iloc[-1]) if 'adx' in df_5m.columns else 20

        if adx_5m > 30:
            return {"signal": "NONE"}

        s1, s2 = pivots.get("S1", 0), pivots.get("S2", 0)
        r1, r2 = pivots.get("R1", 0), pivots.get("R2", 0)

        near_s = (s1 > 0 and abs(close_1m - s1) / close_1m < 0.0003) or (s2 > 0 and abs(close_1m - s2) / close_1m < 0.0003)
        near_r = (r1 > 0 and abs(close_1m - r1) / close_1m < 0.0003) or (r2 > 0 and abs(close_1m - r2) / close_1m < 0.0003)

        if near_s and rsi_1m < 38:
            return {
                "signal": "CALL",
                "strategy": "PIVOT_BOUNCE",
                "strategy_title": "5. Відскок від Pivot / Круглих рівнів",
                "suggested_exp": 4,
                "reason": f"Тест підтримки S1/S2 з RSI перепроданістю ({rsi_1m:.1f})"
            }
        elif near_r and rsi_1m > 62:
            return {
                "signal": "PUT",
                "strategy": "PIVOT_BOUNCE",
                "strategy_title": "5. Відскок від Pivot / Круглих рівнів",
                "suggested_exp": 4,
                "reason": f"Тест опору R1/R2 з RSI перекупленістю ({rsi_1m:.1f})"
            }

        return {"signal": "NONE"}

    def _check_m1_pinbar(self, df_1m, mid_trend) -> dict:
        """Стратегія 6: M1 Скальпінг за Пінбарами"""
        if df_1m.empty or len(df_1m) < 3 or mid_trend == "NEUTRAL":
            return {"signal": "NONE"}

        last_1m = df_1m.iloc[-1]
        lower_wick_ratio = float(last_1m.get('lower_wick_ratio', 0))
        upper_wick_ratio = float(last_1m.get('upper_wick_ratio', 0))
        rsi_1m = float(last_1m.get('rsi', 50))

        if mid_trend == "BULLISH" and lower_wick_ratio >= 0.55 and rsi_1m < 50:
            return {
                "signal": "CALL",
                "strategy": "M1_PINBAR",
                "strategy_title": "6. M1 Скальпінг за Пінбарами",
                "suggested_exp": 2,
                "reason": f"Бычий пінбар на M1 біля EMA (Нижня тінь: {lower_wick_ratio*100:.0f}%)"
            }
        elif mid_trend == "BEARISH" and upper_wick_ratio >= 0.55 and rsi_1m > 50:
            return {
                "signal": "PUT",
                "strategy": "M1_PINBAR",
                "strategy_title": "6. M1 Скальпінг за Пінбарами",
                "suggested_exp": 2,
                "reason": f"Ведмежий пінбар на M1 біля EMA (Верхня тінь: {upper_wick_ratio*100:.0f}%)"
            }

        return {"signal": "NONE"}

    def _check_hybrid_adaptive(self, df_1m, df_5m, df_3m, global_trend, mid_trend, pivots) -> dict:
        """Стратегія 7: Адаптивний Гібрид"""
        if df_5m.empty or df_1m.empty:
            return {"signal": "NONE"}

        last_5m = df_5m.iloc[-1]
        last_1m = df_1m.iloc[-1]
        rsi_5m = float(last_5m.get('rsi', 50))
        rsi_1m = float(last_1m.get('rsi', 50))
        adx = float(last_5m.get('adx', 20))
        bb_upper = float(last_5m.get('bb_upper', 0))
        bb_lower = float(last_5m.get('bb_lower', 0))
        close_5m = float(last_5m['close'])
        close_1m = float(last_1m['close'])
        div = self.detect_divergence(df_5m)

        r1 = pivots.get('R1', 0) if pivots else 0
        s1 = pivots.get('S1', 0) if pivots else 0

        effective_trend = global_trend if global_trend != 'NEUTRAL' else mid_trend

        if adx < 25:
            if close_1m <= bb_lower or rsi_1m < 46:
                return {
                    "signal": "CALL",
                    "strategy": "HYBRID_ADAPTIVE",
                    "strategy_title": "7. Адаптивний Гібрид",
                    "suggested_exp": 5,
                    "reason": f"Флет відскок знизу (RSI 1m: {rsi_1m:.1f})"
                }
            elif close_1m >= bb_upper or rsi_1m > 54:
                return {
                    "signal": "PUT",
                    "strategy": "HYBRID_ADAPTIVE",
                    "strategy_title": "7. Адаптивний Гібрид",
                    "suggested_exp": 5,
                    "reason": f"Флет відскок зверху (RSI 1m: {rsi_1m:.1f})"
                }
        else:
            if effective_trend == 'BULLISH' and rsi_5m < 72:
                if (bb_lower > 0 and close_5m <= bb_lower * 1.004) or (s1 > 0 and close_5m <= s1 * 1.003) or (rsi_5m < 50) or (div == 'BULLISH_DIV'):
                    return {
                        "signal": "CALL",
                        "strategy": "HYBRID_ADAPTIVE",
                        "strategy_title": "7. Адаптивний Гібрид",
                        "suggested_exp": 5,
                        "reason": f"Тренд вгору (ADX: {adx:.1f}, RSI: {rsi_5m:.1f})"
                    }
            elif effective_trend == 'BEARISH' and rsi_5m > 28:
                if (bb_upper > 0 and close_5m >= bb_upper * 0.996) or (r1 > 0 and close_5m >= r1 * 0.997) or (rsi_5m > 50) or (div == 'BEARISH_DIV'):
                    return {
                        "signal": "PUT",
                        "strategy": "HYBRID_ADAPTIVE",
                        "strategy_title": "7. Адаптивний Гібрид",
                        "suggested_exp": 5,
                        "reason": f"Тренд вниз (ADX: {adx:.1f}, RSI: {rsi_5m:.1f})"
                    }

        if df_3m is not None and not df_3m.empty and len(df_3m) >= 10:
            rsi_3m = float(df_3m.iloc[-1].get('rsi', 50))
            if rsi_3m < 38:
                return {
                    "signal": "CALL",
                    "strategy": "HYBRID_ADAPTIVE",
                    "strategy_title": "7. Адаптивний Гібрид",
                    "suggested_exp": 3,
                    "reason": f"Скальпінг 3M: імпульс перепроданості (RSI 3m: {rsi_3m:.1f})"
                }
            elif rsi_3m > 62:
                return {
                    "signal": "PUT",
                    "strategy": "HYBRID_ADAPTIVE",
                    "strategy_title": "7. Адаптивний Гібрид",
                    "suggested_exp": 3,
                    "reason": f"Скальпінг 3M: імпульс перекупленості (RSI 3m: {rsi_3m:.1f})"
                }

        return {"signal": "NONE"}

    def generate_signal(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, global_trend: str, mid_trend: str, df_macro: pd.DataFrame=None, df_3m: pd.DataFrame=None) -> dict:
        if df_5m is None or df_5m.empty or len(df_5m) < 15 or df_1m is None or df_1m.empty or len(df_1m) < 10:
            return {'signal': 'NONE', 'reason': 'Мало даних', 'suggested_exp': 5, 'strategy': 'NONE'}

        adx = float(df_5m['adx'].iloc[-1]) if 'adx' in df_5m.columns else 20
        bb_width = float(df_5m['bb_width'].iloc[-1]) if 'bb_width' in df_5m.columns else 0.001
        div = self.detect_divergence(df_5m)
        pivots = self.calculate_pivots(df_macro) if df_macro is not None and not df_macro.empty else {}

        strategies = [
            self._check_divergence(df_5m, div),
            self._check_breakout(df_5m, bb_width, adx),
            self._check_trend_following(df_5m, df_1m, global_trend, adx),
            self._check_mean_reversion(df_5m, df_1m, adx),
            self._check_pivot_bounce(df_5m, df_1m, pivots),
            self._check_m1_pinbar(df_1m, mid_trend),
            self._check_hybrid_adaptive(df_1m, df_5m, df_3m, global_trend, mid_trend, pivots)
        ]

        valid_signals = [s for s in strategies if s.get("signal") in ["CALL", "PUT"]]

        if not valid_signals:
            return {'signal': 'NONE', 'reason': 'Жодна з 7 стратегій не знайшла точки входу', 'suggested_exp': 5, 'strategy': 'NONE'}

        best_signal = valid_signals[0]
        best_signal['confluence_count'] = len(valid_signals)
        best_signal['rsi'] = round(float(df_5m['rsi'].iloc[-1]), 1) if 'rsi' in df_5m.columns else 50.0
        best_signal['adx'] = round(adx, 1)
        best_signal['divergence'] = div

        return best_signal

    def analyze_all_timeframes(self, df_daily, df_1h, df_15m, df_5m, df_3m, df_1m) -> dict:
        df_macro = df_1h
        df_mid = df_15m
        df_fast = df_5m
        df_micro = df_1m
        
        global_trend = self.get_trend(df_macro, span_val=200)
        mid_trend = self.get_trend(df_mid, span_val=50)
        pivots = self.calculate_pivots(df_macro)
        
        df_indicators_5m = self.calculate_indicators(df_fast.copy()) if df_fast is not None and not df_fast.empty else pd.DataFrame()
        df_indicators_3m = self.calculate_indicators(df_3m.copy()) if df_3m is not None and not df_3m.empty else pd.DataFrame()
        df_indicators_1m = self.calculate_indicators(df_micro.copy()) if df_micro is not None and not df_micro.empty else pd.DataFrame()
        
        if df_indicators_5m.empty or df_indicators_1m.empty:
            return {"signal": "NONE", "reason": "Недостатньо даних"}

        sig_data = self.generate_signal(df_indicators_1m, df_indicators_5m, global_trend, mid_trend, df_macro, df_indicators_3m)
        
        current_price = float(df_indicators_5m['close'].iloc[-1]) if not df_indicators_5m.empty else 0.0
        atr = float(df_indicators_5m['atr'].iloc[-1]) if not df_indicators_5m.empty and 'atr' in df_indicators_5m.columns else 0.001
        volatility_ratio = (atr / current_price) * 1000 if current_price > 0 else 1.0

        wick_ratio = float(df_indicators_1m['wick_ratio'].iloc[-1]) if not df_indicators_1m.empty and 'wick_ratio' in df_indicators_1m.columns else 0.0
        ema_dist = float(df_indicators_5m['ema_dist'].iloc[-1]) if not df_indicators_5m.empty and 'ema_dist' in df_indicators_5m.columns else 0.0
        dist_pivot = self.get_min_dist_to_pivot_or_round(current_price, pivots)

        res_signal = sig_data.get("signal", "NONE")

        res = {
            "signal": res_signal,
            "strategy": sig_data.get("strategy", "HYBRID_ADAPTIVE"),
            "strategy_title": sig_data.get("strategy_title", "7. Адаптивний Гібрид"),
            "confluence_count": sig_data.get("confluence_count", 1),
            "reason": sig_data.get("reason", "Умови не виконано"),
            "rsi": sig_data.get("rsi", 50),
            "adx": sig_data.get("adx", 20),
            "atr": atr,
            "volatility_ratio": volatility_ratio,
            "atr_ratio": volatility_ratio,
            "divergence": sig_data.get("divergence", "NONE"),
            "suggested_exp": sig_data.get("suggested_exp", 5),
            "global_trend": global_trend,
            "mid_trend": mid_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_indicators_5m['bb_width'].iloc[-1]) if not df_indicators_5m.empty and 'bb_width' in df_indicators_5m.columns else 0.001,
            "wick_ratio": wick_ratio,
            "ema_dist": ema_dist,
            "dist_pivot": dist_pivot,
        }
        return res
