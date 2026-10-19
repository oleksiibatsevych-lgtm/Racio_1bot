# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 14:
            return df
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_width'] = df['bb_width'].fillna(0.001)

        # Z-Score для статистичного повернення до середнього
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - rolling_mean) / (rolling_std + 1e-10)
        df['z_score'] = df['z_score'].fillna(0.0)

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

        df['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low + 1e-9))

        df['donchian_upper'] = df['high'].rolling(window=20).max()
        df['donchian_lower'] = df['low'].rolling(window=20).min()

        candle_range = (df['high'] - df['low']).replace(0, 0.00001)
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        df['max_wick'] = np.maximum(upper_wick, lower_wick)
        df['wick_ratio'] = (df['max_wick'] / candle_range).fillna(0.0)

        return df

    def get_trend(self, df: pd.DataFrame, span_val=50) -> str:
        if df is None or df.empty or len(df) < span_val:
            return "NEUTRAL"
        ema = df['close'].ewm(span=span_val, adjust=False).mean()
        current_price = df['close'].iloc[-1]
        current_ema = ema.iloc[-1]
        if current_price > current_ema * 1.0002:
            return "BULLISH"
        elif current_price < current_ema * 0.9998:
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

    def get_min_dist_to_pivot_or_round(self, price: float, pivots: dict) -> float:
        if price <= 0:
            return 0.0
        distances = [abs(price - p_val) / price for p_val in pivots.values() if p_val > 0]
        round_factor = 100 if price > 50 else 10000
        mod_val = (price * round_factor) % 250
        distances.append(min(mod_val, 250 - mod_val) / (price * round_factor))
        return min(distances) if distances else 0.001

    # --- МУЛЬТИ ТАЙМФРЕЙМОВИЙ РУШІЙ З 4 СТРАТЕГІЯМИ ---
    def generate_master_confluence_signal(self, df_1m, df_3m, df_5m, df_15m, df_1h, global_trend, pivots) -> dict:
        if df_1m is None or df_1m.empty or len(df_1m) < 15 or df_3m is None or df_3m.empty:
            return {'signal': 'NONE', 'reason': 'Мало даних для багатотаймфреймового аналізу', 'suggested_exp': 10, 'strategy': 'NONE'}

        buffer_pct = 0.0015  # Буферна зона допуску для Боллінджера

        # === СТРАТЕГІЯ 1: Z-Score Mean Reversion (Статистичне повернення до середнього) ===
        z_score_1m = float(df_1m['z_score'].iloc[-1]) if 'z_score' in df_1m.columns else 0.0
        if z_score_1m <= -2.1:
            return {
                'signal': 'CALL',
                'strategy': 'Z-Score Mean Reversion',
                'strategy_title': 'Reversal Z-Score CALL',
                'confluence_count': 3,
                'suggested_exp': 5,
                'reason': f"Z-Score Mean Reversion: ціна критично нижче середнього ({z_score_1m:.2f})"
            }
        elif z_score_1m >= 2.1:
            return {
                'signal': 'PUT',
                'strategy': 'Z-Score Mean Reversion',
                'strategy_title': 'Reversal Z-Score PUT',
                'confluence_count': 3,
                'suggested_exp': 5,
                'reason': f"Z-Score Mean Reversion: ціна критично вище середнього ({z_score_1m:.2f})"
            }

        # === СТРАТЕГІЯ 2: Flat Matrix Bounce (Розширені межі BB на 1M, 3M, 5M, 15M) ===
        timeframes_to_check = {
            '1M': df_1m,
            '3M': df_3m,
            '5M': df_5m,
            '15M': df_15m
        }
        adx_15m = float(df_15m['adx'].iloc[-1]) if df_15m is not None and not df_15m.empty and 'adx' in df_15m.columns else 20

        for tf_name, df_tf in timeframes_to_check.items():
            if df_tf is None or df_tf.empty or 'bb_lower' not in df_tf.columns or 'rsi' not in df_tf.columns:
                continue

            last_row = df_tf.iloc[-1]
            tf_price = float(last_row['close'])
            bb_lower = float(last_row['bb_lower'])
            bb_upper = float(last_row['bb_upper'])
            rsi = float(last_row['rsi'])
            adx = float(last_row.get('adx', 20))

            lower_buffer = bb_lower * (1 + buffer_pct)
            upper_buffer = bb_upper * (1 - buffer_pct)

            if adx <= 26 and adx_15m < 30:
                if (tf_price <= lower_buffer or last_row['low'] <= bb_lower) and rsi < 36:
                    exp_time = 3 if tf_name == '1M' else (6 if tf_name == '3M' else 10)
                    return {
                        'signal': 'CALL',
                        'strategy': 'Flat Matrix Bounce',
                        'strategy_title': f'Flat Matrix Bounce CALL [{tf_name}]',
                        'confluence_count': 2,
                        'suggested_exp': exp_time,
                        'reason': f"Флет [{tf_name}]: Відбиття від нижньої межі BB (з буфером), RSI: {rsi:.1f}"
                    }
                elif (tf_price >= upper_buffer or last_row['high'] >= bb_upper) and rsi > 64:
                    exp_time = 3 if tf_name == '1M' else (6 if tf_name == '3M' else 10)
                    return {
                        'signal': 'PUT',
                        'strategy': 'Flat Matrix Bounce',
                        'strategy_title': f'Flat Matrix Bounce PUT [{tf_name}]',
                        'confluence_count': 2,
                        'suggested_exp': exp_time,
                        'reason': f"Флет [{tf_name}]: Відбиття від верхньої межі BB (з буфером), RSI: {rsi:.1f}"
                    }

        # === СТРАТЕГІЯ 3: Multi-Timeframe Scalp / Price Action ===
        rsi_1m = float(df_1m['rsi'].iloc[-1]) if 'rsi' in df_1m.columns else 50
        rsi_3m = float(df_3m['rsi'].iloc[-1]) if 'rsi' in df_3m.columns else 50
        if rsi_1m < 28 and rsi_3m < 32:
            return {
                'signal': 'CALL',
                'strategy': 'Multi-Timeframe Scalp',
                'strategy_title': 'Multi-Timeframe Scalp CALL',
                'confluence_count': 3,
                'suggested_exp': 4,
                'reason': f"Multi-TF Scalp: синхронна перепроданість на 1M ({rsi_1m:.1f}) та 3M ({rsi_3m:.1f})"
            }
        elif rsi_1m > 72 and rsi_3m > 68:
            return {
                'signal': 'PUT',
                'strategy': 'Multi-Timeframe Scalp',
                'strategy_title': 'Multi-Timeframe Scalp PUT',
                'confluence_count': 3,
                'suggested_exp': 4,
                'reason': f"Multi-TF Scalp: синхронна перекупленість на 1M ({rsi_1m:.1f}) та 3M ({rsi_3m:.1f})"
            }

        # === СТРАТЕГІЯ 4: Trend Matrix Breakout / Continuation ===
        trend_1h = self.get_trend(df_1h, 50) if df_1h is not None and not df_1h.empty else global_trend
        trend_15m = self.get_trend(df_15m, 20) if df_15m is not None and not df_15m.empty else "NEUTRAL"
        adx_3m = float(df_3m['adx'].iloc[-1]) if 'adx' in df_3m.columns else 20

        if global_trend == "BULLISH" and (trend_1h == "BULLISH" or trend_15m == "BULLISH") and adx_3m > 22 and rsi_1m < 48:
            return {
                'signal': 'CALL',
                'strategy': 'Trend Breakout',
                'strategy_title': 'Trend Matrix Breakout CALL',
                'confluence_count': 3,
                'suggested_exp': 12,
                'reason': "ТРЕНД (1H-1M синхрон): Імпульс за висхідним трендом"
            }
        elif global_trend == "BEARISH" and (trend_1h == "BEARISH" or trend_15m == "BEARISH") and adx_3m > 22 and rsi_1m > 52:
            return {
                'signal': 'PUT',
                'strategy': 'Trend Breakout',
                'strategy_title': 'Trend Matrix Breakout PUT',
                'confluence_count': 3,
                'suggested_exp': 12,
                'reason': "ТРЕНД (1H-1M синхрон): Імпульс за низхідним трендом"
            }

        return {'signal': 'NONE', 'reason': 'Мультитаймфреймовий баланс нейтральний', 'suggested_exp': 10, 'strategy': 'NONE'}

    def analyze_all_timeframes(self, df_daily, df_1h, df_15m, df_5m, df_3m, df_1m) -> dict:
        global_trend = self.get_trend(df_1h, span_val=200)
        pivots = self.calculate_pivots(df_1h)
        
        df_ind_1m = self.calculate_indicators(df_1m.copy()) if df_1m is not None and not df_1m.empty else pd.DataFrame()
        df_ind_3m = self.calculate_indicators(df_3m.copy()) if df_3m is not None and not df_3m.empty else pd.DataFrame()
        df_ind_5m = self.calculate_indicators(df_5m.copy()) if df_5m is not None and not df_5m.empty else pd.DataFrame()
        df_ind_15m = self.calculate_indicators(df_15m.copy()) if df_15m is not None and not df_15m.empty else pd.DataFrame()
        df_ind_1h = self.calculate_indicators(df_1h.copy()) if df_1h is not None and not df_1h.empty else pd.DataFrame()

        if df_ind_1m.empty or df_ind_3m.empty:
            return {"signal": "NONE", "reason": "Недостатньо даних"}

        sig_data = self.generate_master_confluence_signal(df_ind_1m, df_ind_3m, df_ind_5m, df_ind_15m, df_ind_1h, global_trend, pivots)
        
        current_price = float(df_ind_1m['close'].iloc[-1]) if not df_ind_1m.empty else 0.0
        atr = float(df_ind_1m['atr'].iloc[-1]) if not df_ind_1m.empty and 'atr' in df_ind_1m.columns else 0.001

        return {
            "signal": sig_data.get("signal", "NONE"),
            "strategy": sig_data.get("strategy", "MULTI_TF_DUAL_ENGINE"),
            "strategy_title": sig_data.get("strategy_title", "Multi-TF Dual Engine"),
            "confluence_count": sig_data.get("confluence_count", 1),
            "reason": sig_data.get("reason", "Умови не виконано"),
            "rsi": float(df_ind_1m['rsi'].iloc[-1]) if 'rsi' in df_ind_1m.columns else 50,
            "adx": float(df_ind_3m['adx'].iloc[-1]) if 'adx' in df_ind_3m.columns else 20,
            "atr": atr,
            "volatility_ratio": (atr / current_price) * 1000 if current_price > 0 else 1.0,
            "suggested_exp": sig_data.get("suggested_exp", 10),
            "global_trend": global_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_ind_3m['bb_width'].iloc[-1]) if not df_ind_3m.empty else 0.001,
            "dist_pivot": self.get_min_dist_to_pivot_or_round(current_price, pivots),
        }
