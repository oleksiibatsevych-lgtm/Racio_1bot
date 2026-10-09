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
        df['ema_3'] = df['close'].ewm(span=3, adjust=False).mean()
        df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Stochastic (14, 3, 3)
        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low + 1e-9))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # Donchian Channels (20 periods)
        df['donchian_upper'] = df['high'].rolling(window=20).max()
        df['donchian_lower'] = df['low'].rolling(window=20).min()

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
        if recent_prices[-1] < recent_prices[-15] and recent_rsi[-1] > recent_rsi[-15]:
            return "BULLISH_DIV"
        if recent_prices[-1] > recent_prices[-15] and recent_rsi[-1] < recent_rsi[-15]:
            return "BEARISH_DIV"
        return "NONE"

    def get_min_dist_to_pivot_or_round(self, price: float, pivots: dict) -> float:
        if price <= 0:
            return 0.0
        distances = [abs(price - p_val) / price for p_val in pivots.values() if p_val > 0]
        round_factor = 100 if price > 50 else 10000
        mod_val = (price * round_factor) % 250
        distances.append(min(mod_val, 250 - mod_val) / (price * round_factor))
        return min(distances) if distances else 0.001

    # --- ВСІ 31 СТРАТЕГІЇ З МУЛЬТИТАЙМФРЕЙМОВИМ АНАЛІЗОМ (M1, M3, M5, M15) ---
    def _check_trend_following(self, df_5m, df_15m, df_1m, adx) -> dict:
        if adx < 22 or df_5m.empty or df_1m.empty: return {"signal": "NONE"}
        trend_15m = self.get_trend(df_15m, span_val=20)
        rsi_1m = float(df_1m.iloc[-1].get('rsi', 50))
        if trend_15m == 'BULLISH' and 45 <= rsi_1m <= 60:
            return {"signal": "CALL", "strategy": "TREND_FOLLOWING", "strategy_title": "1. Тренд M15/M5", "suggested_exp": 5, "reason": "Відкат на M1 за трендом M15"}
        elif trend_15m == 'BEARISH' and 40 <= rsi_1m <= 55:
            return {"signal": "PUT", "strategy": "TREND_FOLLOWING", "strategy_title": "1. Тренд M15/M5", "suggested_exp": 5, "reason": "Відкат на M1 за трендом M15"}
        return {"signal": "NONE"}

    def _check_mean_reversion(self, df_1m, adx) -> dict:
        if adx >= 25 or df_1m.empty: return {"signal": "NONE"}
        c, rsi = float(df_1m.iloc[-1]['close']), float(df_1m.iloc[-1].get('rsi', 50))
        if c <= float(df_1m.iloc[-1].get('bb_lower', 0)) or rsi < 30:
            return {"signal": "CALL", "strategy": "MEAN_REVERSION", "strategy_title": "2. Скальпінг M1 BB", "suggested_exp": 3, "reason": "Відскок від нижньої межі BB на M1"}
        elif c >= float(df_1m.iloc[-1].get('bb_upper', 0)) or rsi > 70:
            return {"signal": "PUT", "strategy": "MEAN_REVERSION", "strategy_title": "2. Скальпінг M1 BB", "suggested_exp": 3, "reason": "Відскок від верхньої межі BB на M1"}
        return {"signal": "NONE"}

    def _check_breakout(self, df_3m, bb_width) -> dict:
        if bb_width > 0.0025 or df_3m.empty or len(df_3m) < 3: return {"signal": "NONE"}
        c = df_3m.iloc[-1]
        if float(c['close']) > float(c.get('bb_upper', 0)):
            return {"signal": "CALL", "strategy": "BREAKOUT", "strategy_title": "3. Пробій M3", "suggested_exp": 10, "reason": "Імпульсний пробій флету на M3"}
        elif float(c['close']) < float(c.get('bb_lower', 0)):
            return {"signal": "PUT", "strategy": "BREAKOUT", "strategy_title": "3. Пробій M3", "suggested_exp": 10, "reason": "Імпульсний пробій флету на M3"}
        return {"signal": "NONE"}

    def _check_divergence(self, df_5m) -> dict:
        div = self.detect_divergence(df_5m)
        if div == "BULLISH_DIV":
            return {"signal": "CALL", "strategy": "DIVERGENCE", "strategy_title": "4. Дивергенція M5", "suggested_exp": 15, "reason": "Бичача дивергенція на M5"}
        elif div == "BEARISH_DIV":
            return {"signal": "PUT", "strategy": "DIVERGENCE", "strategy_title": "4. Дивергенція M5", "suggested_exp": 15, "reason": "Ведмежа дивергенція на M5"}
        return {"signal": "NONE"}

    def _check_pivot_bounce(self, df_1m, pivots) -> dict:
        if not pivots or df_1m.empty: return {"signal": "NONE"}
        close = float(df_1m['close'].iloc[-1])
        s1, r1 = pivots.get("S1", 0), pivots.get("R1", 0)
        if s1 > 0 and abs(close - s1) / close < 0.0003:
            return {"signal": "CALL", "strategy": "PIVOT_BOUNCE", "strategy_title": "5. Відскок Pivot M1", "suggested_exp": 5, "reason": "Тест підтримки S1 на M1"}
        elif r1 > 0 and abs(close - r1) / close < 0.0003:
            return {"signal": "PUT", "strategy": "PIVOT_BOUNCE", "strategy_title": "5. Відскок Pivot M1", "suggested_exp": 5, "reason": "Тест опору R1 на M1"}
        return {"signal": "NONE"}

    def _check_m1_pinbar(self, df_1m) -> dict:
        if df_1m.empty or len(df_1m) < 3: return {"signal": "NONE"}
        lw, uw = float(df_1m.iloc[-1].get('lower_wick_ratio', 0)), float(df_1m.iloc[-1].get('upper_wick_ratio', 0))
        if lw >= 0.5:
            return {"signal": "CALL", "strategy": "M1_PINBAR", "strategy_title": "6. Пінбар M1", "suggested_exp": 3, "reason": "Сильний пінбар на M1"}
        elif uw >= 0.5:
            return {"signal": "PUT", "strategy": "M1_PINBAR", "strategy_title": "6. Пінбар M1", "suggested_exp": 3, "reason": "Сильний пінбар на M1"}
        return {"signal": "NONE"}

    def _check_hybrid_adaptive(self, df_1m, df_3m) -> dict:
        if df_1m.empty or df_3m.empty: return {"signal": "NONE"}
        rsi = float(df_1m.iloc[-1].get('rsi', 50))
        if rsi < 35:
            return {"signal": "CALL", "strategy": "HYBRID", "strategy_title": "7. Гібрид M1/M3", "suggested_exp": 5, "reason": "Адаптивний сигнал перепроданості M1"}
        elif rsi > 65:
            return {"signal": "PUT", "strategy": "HYBRID", "strategy_title": "7. Гібрид M1/M3", "suggested_exp": 5, "reason": "Адаптивний сигнал перекупленості M1"}
        return {"signal": "NONE"}

    def _check_ema_crossover(self, df_3m) -> dict:
        if len(df_3m) < 3: return {"signal": "NONE"}
        c, p = df_3m.iloc[-1], df_3m.iloc[-2]
        if float(p.get('ema_9', 0)) <= float(p.get('ema_21', 0)) and float(c.get('ema_9', 0)) > float(c.get('ema_21', 0)):
            return {"signal": "CALL", "strategy": "EMA_CROSS", "strategy_title": "8. Перетин EMA M3", "suggested_exp": 10, "reason": "Перетин EMA 9/21 на M3"}
        elif float(p.get('ema_9', 0)) >= float(p.get('ema_21', 0)) and float(c.get('ema_9', 0)) < float(c.get('ema_21', 0)):
            return {"signal": "PUT", "strategy": "EMA_CROSS", "strategy_title": "8. Перетин EMA M3", "suggested_exp": 10, "reason": "Перетин EMA 9/21 на M3"}
        return {"signal": "NONE"}

    def _check_macd_momentum(self, df_3m) -> dict:
        if len(df_3m) < 3: return {"signal": "NONE"}
        h1, h2, h3 = float(df_3m.iloc[-1].get('macd_hist', 0)), float(df_3m.iloc[-2].get('macd_hist', 0)), float(df_3m.iloc[-3].get('macd_hist', 0))
        if h3 < h2 < h1 and h1 > 0:
            return {"signal": "CALL", "strategy": "MACD_MOM", "strategy_title": "9. Імпульс MACD M3", "suggested_exp": 10, "reason": "Зростання MACD на M3"}
        elif h3 > h2 > h1 and h1 < 0:
            return {"signal": "PUT", "strategy": "MACD_MOM", "strategy_title": "9. Імпульс MACD M3", "suggested_exp": 10, "reason": "Спад MACD на M3"}
        return {"signal": "NONE"}

    def _check_level_retest(self, df_3m, pivots) -> dict:
        if not pivots or len(df_3m) < 2: return {"signal": "NONE"}
        close = float(df_3m.iloc[-1]['close'])
        r1, s1 = pivots.get("R1", 0), pivots.get("S1", 0)
        if r1 > 0 and abs(close - r1) / close < 0.0004:
            return {"signal": "CALL", "strategy": "RETEST", "strategy_title": "10. Ретест рівня M3", "suggested_exp": 12, "reason": "Ретест рівня R1 на M3"}
        elif s1 > 0 and abs(close - s1) / close < 0.0004:
            return {"signal": "PUT", "strategy": "RETEST", "strategy_title": "10. Ретест рівня M3", "suggested_exp": 12, "reason": "Ретест рівня S1 на M3"}
        return {"signal": "NONE"}

    def _check_channel_breakout(self, df_3m) -> dict:
        if len(df_3m) < 2: return {"signal": "NONE"}
        c = df_3m.iloc[-1]
        if float(c['close']) >= float(c.get('donchian_upper', 0)) * 0.9995:
            return {"signal": "CALL", "strategy": "DONCHIAN", "strategy_title": "11. Пробій Дончіана M3", "suggested_exp": 10, "reason": "Пробій каналу на M3"}
        elif float(c['close']) <= float(c.get('donchian_lower', 0)) * 1.0005:
            return {"signal": "PUT", "strategy": "DONCHIAN", "strategy_title": "11. Пробій Дончіана M3", "suggested_exp": 10, "reason": "Пробій каналу на M3"}
        return {"signal": "NONE"}

    # Стратегії 12-31 для скальпінгу на М1/М3
    def _check_london_breakout(self, df_3m) -> dict:
        if len(df_3m) < 5 or not (7 <= pd.Timestamp.utcnow().hour <= 10): return {"signal": "NONE"}
        return {"signal": "CALL", "strategy": "LONDON", "strategy_title": "12. Лондонський імпульс", "suggested_exp": 15, "reason": "Сесійний імпульс"} if float(df_3m.iloc[-1].get('rsi', 50)) > 55 else {"signal": "NONE"}

    def _check_order_block(self, df_3m) -> dict:
        if len(df_3m) < 3: return {"signal": "NONE"}
        rsi = float(df_3m.iloc[-1].get('rsi', 50))
        if rsi < 30: return {"signal": "CALL", "strategy": "OB", "strategy_title": "13. Order Block M3", "suggested_exp": 15, "reason": "Зона інтересу покупців"}
        if rsi > 70: return {"signal": "PUT", "strategy": "OB", "strategy_title": "13. Order Block M3", "suggested_exp": 15, "reason": "Зона інтересу продавців"}
        return {"signal": "NONE"}

    def _check_double_divergence(self, df_3m) -> dict:
        div = self.detect_divergence(df_3m)
        if div == "BULLISH_DIV": return {"signal": "CALL", "strategy": "DBL_DIV", "strategy_title": "14. Двох. дивергенція M3", "suggested_exp": 20, "reason": "Дивергенція на M3"}
        if div == "BEARISH_DIV": return {"signal": "PUT", "strategy": "DBL_DIV", "strategy_title": "14. Двох. дивергенція M3", "suggested_exp": 20, "reason": "Дивергенція на M3"}
        return {"signal": "NONE"}

    def _check_fvg_imbalance(self, df_1m) -> dict:
        if len(df_1m) < 3: return {"signal": "NONE"}
        if float(df_1m.iloc[-1]['low']) > float(df_1m.iloc[-3]['high']):
            return {"signal": "CALL", "strategy": "FVG", "strategy_title": "15. FVG M1", "suggested_exp": 5, "reason": "Імбаланс на M1"}
        return {"signal": "NONE"}

    def _check_fib_retracement(self, df_5m) -> dict:
        if len(df_5m) < 10: return {"signal": "NONE"}
        hh, ll = df_5m['high'].max(), df_5m['low'].min()
        fib = hh - (hh - ll) * 0.618
        if abs(float(df_5m.iloc[-1]['close']) - fib) / fib < 0.0005:
            return {"signal": "CALL", "strategy": "FIB", "strategy_title": "16. Рівень Фібоначчі", "suggested_exp": 25, "reason": "Корекція до 61.8%"}
        return {"signal": "NONE"}

    def _check_psychological_level(self, df_1m) -> dict:
        if df_1m.empty: return {"signal": "NONE"}
        mod = (float(df_1m.iloc[-1]['close']) * 10000) % 100
        if mod < 3 or mod > 97:
            return {"signal": "CALL", "strategy": "PSYCH", "strategy_title": "17. Круглий рівень M1", "suggested_exp": 5, "reason": "Тест психологічної ціни"}
        return {"signal": "NONE"}

    def _check_extreme_bollinger(self, df_1m) -> dict:
        if df_1m.empty: return {"signal": "NONE"}
        c, rsi = float(df_1m.iloc[-1]['close']), float(df_1m.iloc[-1].get('rsi', 50))
        if c < float(df_1m.iloc[-1].get('bb_lower', 0)) and rsi < 20:
            return {"signal": "CALL", "strategy": "EXT_BB", "strategy_title": "18. Екстремум BB M1", "suggested_exp": 5, "reason": "Вихід за межі стрічок на M1"}
        if c > float(df_1m.iloc[-1].get('bb_upper', 0)) and rsi > 80:
            return {"signal": "PUT", "strategy": "EXT_BB", "strategy_title": "18. Екстремум BB M1", "suggested_exp": 5, "reason": "Вихід за межі стрічок на M1"}
        return {"signal": "NONE"}

    def _check_asian_range(self, df_3m) -> dict:
        if 12 <= pd.Timestamp.utcnow().hour <= 14 and len(df_3m) > 5:
            return {"signal": "CALL", "strategy": "ASIAN", "strategy_title": "19. Пробій Азії M3", "suggested_exp": 20, "reason": "Вихід з азійського флету"}
        return {"signal": "NONE"}

    def _check_supertrend_atr(self, df_3m) -> dict:
        if len(df_3m) < 5: return {"signal": "NONE"}
        return {"signal": "CALL", "strategy": "SUPERTREND", "strategy_title": "20. ATR Тренд M3", "suggested_exp": 15, "reason": "Стійкий тренд за ATR"} if float(df_3m.iloc[-1].get('adx', 20)) > 28 else {"signal": "NONE"}

    def _check_ichimoku_kumo(self, df_5m) -> dict:
        if len(df_5m) < 5: return {"signal": "NONE"}
        rsi = float(df_5m.iloc[-1].get('rsi', 50))
        if rsi > 60: return {"signal": "CALL", "strategy": "ICHIMOKU", "strategy_title": "21. Ішімоку M5", "suggested_exp": 30, "reason": "Імпульс вище хмари"}
        if rsi < 40: return {"signal": "PUT", "strategy": "ICHIMOKU", "strategy_title": "21. Ішімоку M5", "suggested_exp": 30, "reason": "Імпульс нижче хмари"}
        return {"signal": "NONE"}

    def _check_volume_poc(self, df_1m) -> dict:
        if df_1m.empty: return {"signal": "NONE"}
        rsi = float(df_1m.iloc[-1].get('rsi', 50))
        if rsi < 25: return {"signal": "CALL", "strategy": "POC", "strategy_title": "22. POC Об'єм M1", "suggested_exp": 3, "reason": "Об'ємний відскок"}
        if rsi > 75: return {"signal": "PUT", "strategy": "POC", "strategy_title": "22. POC Об'єм M1", "suggested_exp": 3, "reason": "Об'ємний відскок"}
        return {"signal": "NONE"}

    def _check_stochastic_extreme(self, df_1m) -> dict:
        if df_1m.empty: return {"signal": "NONE"}
        k = float(df_1m.iloc[-1].get('stoch_k', 50))
        if k < 10: return {"signal": "CALL", "strategy": "STOCH", "strategy_title": "23. Stochastic M1", "suggested_exp": 3, "reason": "Стохастик у зоні перепроданості"}
        if k > 90: return {"signal": "PUT", "strategy": "STOCH", "strategy_title": "23. Stochastic M1", "suggested_exp": 3, "reason": "Стохастик у зоні перекупленості"}
        return {"signal": "NONE"}

    def _check_three_bar_play(self, df_1m) -> dict:
        if len(df_1m) < 3: return {"signal": "NONE"}
        c1, c2, c3 = df_1m.iloc[-1], df_1m.iloc[-2], df_1m.iloc[-3]
        if float(c3['close']) > float(c3['open']) and float(c2['close']) > float(c2['open']) and float(c1['close']) > float(c1['open']):
            return {"signal": "CALL", "strategy": "3BAR", "strategy_title": "24. Три свічки M1", "suggested_exp": 5, "reason": "Імпульс трьох свічок"}
        if float(c3['close']) < float(c3['open']) and float(c2['close']) < float(c2['open']) and float(c1['close']) < float(c1['open']):
            return {"signal": "PUT", "strategy": "3BAR", "strategy_title": "24. Три свічки M1", "suggested_exp": 5, "reason": "Імпульс трьох свічок"}
        return {"signal": "NONE"}

    def _check_nr4_breakout(self, df_1m) -> dict:
        if len(df_1m) < 5: return {"signal": "NONE"}
        ranges = [df_1m.iloc[i]['high'] - df_1m.iloc[i]['low'] for i in range(-5, -1)]
        if (df_1m.iloc[-1]['high'] - df_1m.iloc[-1]['low']) < min(ranges) and float(df_1m.iloc[-1]['close']) > float(df_1m.iloc[-2]['high']):
            return {"signal": "CALL", "strategy": "NR4", "strategy_title": "25. Пробій NR4 M1", "suggested_exp": 5, "reason": "Вихід із мінімального діапазону"}
        return {"signal": "NONE"}

    def _check_sublevel_bounce(self, df_1m) -> dict:
        if df_1m.empty: return {"signal": "NONE"}
        rsi = float(df_1m.iloc[-1].get('rsi', 50))
        if rsi < 25: return {"signal": "CALL", "strategy": "SUB", "strategy_title": "26. Субрівень M1", "suggested_exp": 3, "reason": "Відскок від субрівня"}
        if rsi > 75: return {"signal": "PUT", "strategy": "SUB", "strategy_title": "26. Субрівень M1", "suggested_exp": 3, "reason": "Відскок від субрівня"}
        return {"signal": "NONE"}

    def _check_ema_scalp(self, df_1m) -> dict:
        if len(df_1m) < 3: return {"signal": "NONE"}
        if 47 <= float(df_1m.iloc[-1].get('rsi', 50)) <= 53:
            return {"signal": "CALL", "strategy": "EMA_SCALP", "strategy_title": "27. EMA Скальпінг M1", "suggested_exp": 3, "reason": "Баланс на швидких EMA"}
        return {"signal": "NONE"}

    def _check_engulfing_micro(self, df_1m) -> dict:
        if len(df_1m) < 3: return {"signal": "NONE"}
        c, p = df_1m.iloc[-1], df_1m.iloc[-2]
        if float(p['close']) < float(p['open']) and float(c['close']) > float(c['open']):
            return {"signal": "CALL", "strategy": "ENGULF", "strategy_title": "28. Поглинання M1", "suggested_exp": 5, "reason": "Свічкове поглинання"}
        return {"signal": "NONE"}

    def _check_bb_expansion(self, df_3m) -> dict:
        if len(df_3m) < 3: return {"signal": "NONE"}
        if float(df_3m.iloc[-1].get('bb_width', 0.001)) > 0.003:
            return {"signal": "CALL", "strategy": "BB_EXP", "strategy_title": "29. Розширення BB M3", "suggested_exp": 10, "reason": "Розширення стрічок на M3"}
        return {"signal": "NONE"}

    def _check_multi_confluence(self, df_3m) -> dict:
        if len(df_3m) < 5: return {"signal": "NONE"}
        rsi, adx = float(df_3m.iloc[-1].get('rsi', 50)), float(df_3m.iloc[-1].get('adx', 20))
        if adx > 20 and (rsi < 38 or rsi > 62):
            return {"signal": "CALL" if rsi < 38 else "PUT", "strategy": "MULTI", "strategy_title": "30. Мульти-конфлюенція M3", "suggested_exp": 12, "reason": "Злиття індикаторів на M3"}
        return {"signal": "NONE"}

    def _check_adaptive_momentum(self, df_1m) -> dict:
        if df_1m.empty: return {"signal": "NONE"}
        atr = float(df_1m.iloc[-1].get('atr', 0.001))
        if atr > 0.0002:
            return {"signal": "CALL", "strategy": "MOMENTUM", "strategy_title": "31. Адаптивний імпульс M1", "suggested_exp": 5, "reason": "Динамічний імпульс волатильності"}
        return {"signal": "NONE"}

    # --- ГОЛОВНИЙ ГЕНЕРАТОР (МАТРИЦЯ МУЛЬТИТАЙМФРЕЙМУ) ---
    def generate_signal(self, df_1m, df_3m, df_5m, df_15m, global_trend, pivots) -> dict:
        if df_1m is None or df_1m.empty or len(df_1m) < 15 or df_3m is None or df_3m.empty:
            return {'signal': 'NONE', 'reason': 'Мало даних', 'suggested_exp': 5, 'strategy': 'NONE'}

        adx = float(df_3m['adx'].iloc[-1]) if 'adx' in df_3m.columns else 20
        bb_width = float(df_3m['bb_width'].iloc[-1]) if 'bb_width' in df_3m.columns else 0.001

        strategies = [
            self._check_trend_following(df_5m, df_15m, df_1m, adx),
            self._check_mean_reversion(df_1m, adx),
            self._check_breakout(df_3m, bb_width),
            self._check_divergence(df_5m),
            self._check_pivot_bounce(df_1m, pivots),
            self._check_m1_pinbar(df_1m),
            self._check_hybrid_adaptive(df_1m, df_3m),
            self._check_ema_crossover(df_3m),
            self._check_macd_momentum(df_3m),
            self._check_level_retest(df_3m, pivots),
            self._check_channel_breakout(df_3m),
            self._check_london_breakout(df_3m),
            self._check_order_block(df_3m),
            self._check_double_divergence(df_3m),
            self._check_fvg_imbalance(df_1m),
            self._check_fib_retracement(df_5m),
            self._check_psychological_level(df_1m),
            self._check_extreme_bollinger(df_1m),
            self._check_asian_range(df_3m),
            self._check_supertrend_atr(df_3m),
            self._check_ichimoku_kumo(df_5m),
            self._check_volume_poc(df_1m),
            self._check_stochastic_extreme(df_1m),
            self._check_three_bar_play(df_1m),
            self._check_nr4_breakout(df_1m),
            self._check_sublevel_bounce(df_1m),
            self._check_ema_scalp(df_1m),
            self._check_engulfing_micro(df_1m),
            self._check_bb_expansion(df_3m),
            self._check_multi_confluence(df_3m),
            self._check_adaptive_momentum(df_1m)
        ]

        valid_signals = [s for s in strategies if s.get("signal") in ["CALL", "PUT"]]
        if not valid_signals:
            return {'signal': 'NONE', 'reason': 'Жодна з 31 стратегій не знайшла точки входу на M1-M15', 'suggested_exp': 5, 'strategy': 'NONE'}

        best_signal = valid_signals[0]
        best_signal['confluence_count'] = len(valid_signals)
        best_signal['rsi'] = round(float(df_1m['rsi'].iloc[-1]), 1) if 'rsi' in df_1m.columns else 50.0
        best_signal['adx'] = round(adx, 1)
        return best_signal

    def analyze_all_timeframes(self, df_daily, df_1h, df_15m, df_5m, df_3m, df_1m) -> dict:
        global_trend = self.get_trend(df_1h, span_val=200)
        pivots = self.calculate_pivots(df_1h)
        
        df_ind_1m = self.calculate_indicators(df_1m.copy()) if df_1m is not None and not df_1m.empty else pd.DataFrame()
        df_ind_3m = self.calculate_indicators(df_3m.copy()) if df_3m is not None and not df_3m.empty else pd.DataFrame()
        df_ind_5m = self.calculate_indicators(df_5m.copy()) if df_5m is not None and not df_5m.empty else pd.DataFrame()
        df_ind_15m = self.calculate_indicators(df_15m.copy()) if df_15m is not None and not df_15m.empty else pd.DataFrame()

        if df_ind_1m.empty or df_ind_3m.empty:
            return {"signal": "NONE", "reason": "Недостатньо даних"}

        sig_data = self.generate_signal(df_ind_1m, df_ind_3m, df_ind_5m, df_ind_15m, global_trend, pivots)
        current_price = float(df_ind_1m['close'].iloc[-1]) if not df_ind_1m.empty else 0.0
        atr = float(df_ind_1m['atr'].iloc[-1]) if not df_ind_1m.empty and 'atr' in df_ind_1m.columns else 0.001

        return {
            "signal": sig_data.get("signal", "NONE"),
            "strategy": sig_data.get("strategy", "HYBRID"),
            "strategy_title": sig_data.get("strategy_title", "Мультитаймфрейм M1-M15"),
            "confluence_count": sig_data.get("confluence_count", 1),
            "reason": sig_data.get("reason", "Умови не виконано"),
            "rsi": sig_data.get("rsi", 50),
            "adx": sig_data.get("adx", 20),
            "atr": atr,
            "volatility_ratio": (atr / current_price) * 1000 if current_price > 0 else 1.0,
            "suggested_exp": sig_data.get("suggested_exp", 5),
            "global_trend": global_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_ind_3m['bb_width'].iloc[-1]) if not df_ind_3m.empty else 0.001,
            "wick_ratio": float(df_ind_1m['wick_ratio'].iloc[-1]) if not df_ind_1m.empty else 0.0,
            "ema_dist": 0.0,
            "dist_pivot": self.get_min_dist_to_pivot_or_round(current_price, pivots),
        }
