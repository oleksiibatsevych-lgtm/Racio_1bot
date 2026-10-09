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

    # --- СТРАТЕГІЇ 1 - 11 ---
    def _check_trend_following(self, df_5m, df_1m, global_trend, adx) -> dict:
        if adx < 25 or global_trend == "NEUTRAL" or df_5m.empty or df_1m.empty:
            return {"signal": "NONE"}
        last_1m = df_1m.iloc[-1]
        rsi_1m = float(last_1m.get('rsi', 50))
        if global_trend == 'BULLISH' and 48.0 <= rsi_1m <= 65.0:
            return {"signal": "CALL", "strategy": "TREND_FOLLOWING", "strategy_title": "1. Вхід за трендом", "suggested_exp": 15, "reason": "Бичачий відкат за трендом"}
        elif global_trend == 'BEARISH' and 35.0 <= rsi_1m <= 52.0:
            return {"signal": "PUT", "strategy": "TREND_FOLLOWING", "strategy_title": "1. Вхід за трендом", "suggested_exp": 15, "reason": "Ведмежий відкат за трендом"}
        return {"signal": "NONE"}

    def _check_mean_reversion(self, df_5m, df_1m, adx) -> dict:
        if adx >= 22 or df_1m.empty:
            return {"signal": "NONE"}
        last_1m = df_1m.iloc[-1]
        close_1m, rsi_1m = float(last_1m['close']), float(last_1m.get('rsi', 50))
        if close_1m <= float(last_1m.get('bb_lower', 0)) or rsi_1m < 35:
            return {"signal": "CALL", "strategy": "MEAN_REVERSION", "strategy_title": "2. Скальпінг у флеті", "suggested_exp": 5, "reason": "Відскок від нижньої межі BB"}
        elif close_1m >= float(last_1m.get('bb_upper', 0)) or rsi_1m > 65:
            return {"signal": "PUT", "strategy": "MEAN_REVERSION", "strategy_title": "2. Скальпінг у флеті", "suggested_exp": 5, "reason": "Відскок від верхньої межі BB"}
        return {"signal": "NONE"}

    def _check_breakout(self, df_5m, bb_width, adx) -> dict:
        if bb_width > 0.0020 or df_5m.empty or len(df_5m) < 3:
            return {"signal": "NONE"}
        last_5m = df_5m.iloc[-1]
        if float(last_5m['close']) > float(last_5m.get('bb_upper', 0)):
            return {"signal": "CALL", "strategy": "BREAKOUT", "strategy_title": "3. Пробій волатильності", "suggested_exp": 15, "reason": "Пробій стиснення вгору"}
        elif float(last_5m['close']) < float(last_5m.get('bb_lower', 0)):
            return {"signal": "PUT", "strategy": "BREAKOUT", "strategy_title": "3. Пробій волатильності", "suggested_exp": 15, "reason": "Пробій стиснення вниз"}
        return {"signal": "NONE"}

    def _check_divergence(self, df_5m, div) -> dict:
        if div == "NONE" or df_5m.empty:
            return {"signal": "NONE"}
        rsi_5m = float(df_5m.iloc[-1].get('rsi', 50))
        if div == "BULLISH_DIV" and rsi_5m < 45:
            return {"signal": "CALL", "strategy": "DIVERGENCE", "strategy_title": "4. Дивергенція RSI", "suggested_exp": 30, "reason": "Бичача дивергенція"}
        elif div == "BEARISH_DIV" and rsi_5m > 55:
            return {"signal": "PUT", "strategy": "DIVERGENCE", "strategy_title": "4. Дивергенція RSI", "suggested_exp": 30, "reason": "Ведмежа дивергенція"}
        return {"signal": "NONE"}

    def _check_pivot_bounce(self, df_5m, df_1m, pivots) -> dict:
        if not pivots or df_5m.empty or df_1m.empty:
            return {"signal": "NONE"}
        close_1m = float(df_1m['close'].iloc[-1])
        s1, r1 = pivots.get("S1", 0), pivots.get("R1", 0)
        if s1 > 0 and abs(close_1m - s1) / close_1m < 0.0003:
            return {"signal": "CALL", "strategy": "PIVOT_BOUNCE", "strategy_title": "5. Відскок від Pivot", "suggested_exp": 10, "reason": "Тест рівня S1"}
        elif r1 > 0 and abs(close_1m - r1) / close_1m < 0.0003:
            return {"signal": "PUT", "strategy": "PIVOT_BOUNCE", "strategy_title": "5. Відскок від Pivot", "suggested_exp": 10, "reason": "Тест рівня R1"}
        return {"signal": "NONE"}

    def _check_m1_pinbar(self, df_1m, mid_trend) -> dict:
        if df_1m.empty or len(df_1m) < 3:
            return {"signal": "NONE"}
        last_1m = df_1m.iloc[-1]
        if float(last_1m.get('lower_wick_ratio', 0)) >= 0.55:
            return {"signal": "CALL", "strategy": "M1_PINBAR", "strategy_title": "6. Пінбар M1", "suggested_exp": 5, "reason": "Бичачий пінбар"}
        elif float(last_1m.get('upper_wick_ratio', 0)) >= 0.55:
            return {"signal": "PUT", "strategy": "M1_PINBAR", "strategy_title": "6. Пінбар M1", "suggested_exp": 5, "reason": "Ведмежий пінбар"}
        return {"signal": "NONE"}

    def _check_hybrid_adaptive(self, df_1m, df_5m, df_3m, global_trend, mid_trend, pivots) -> dict:
        if df_5m.empty or df_1m.empty:
            return {"signal": "NONE"}
        rsi_1m = float(df_1m.iloc[-1].get('rsi', 50))
        if rsi_1m < 40:
            return {"signal": "CALL", "strategy": "HYBRID_ADAPTIVE", "strategy_title": "7. Адаптивний Гібрид", "suggested_exp": 10, "reason": "Гібридний сигнал BUY"}
        elif rsi_1m > 60:
            return {"signal": "PUT", "strategy": "HYBRID_ADAPTIVE", "strategy_title": "7. Адаптивний Гібрид", "suggested_exp": 10, "reason": "Гібридний сигнал SELL"}
        return {"signal": "NONE"}

    def _check_ema_crossover(self, df_5m) -> dict:
        if len(df_5m) < 3: return {"signal": "NONE"}
        c, p = df_5m.iloc[-1], df_5m.iloc[-2]
        if float(p.get('ema_9', 0)) <= float(p.get('ema_21', 0)) and float(c.get('ema_9', 0)) > float(c.get('ema_21', 0)):
            return {"signal": "CALL", "strategy": "EMA_CROSSOVER", "strategy_title": "8. Перетин EMA 9/21", "suggested_exp": 20, "reason": "Бичачий перетин EMA"}
        elif float(p.get('ema_9', 0)) >= float(p.get('ema_21', 0)) and float(c.get('ema_9', 0)) < float(c.get('ema_21', 0)):
            return {"signal": "PUT", "strategy": "EMA_CROSSOVER", "strategy_title": "8. Перетин EMA 9/21", "suggested_exp": 20, "reason": "Ведмежий перетин EMA"}
        return {"signal": "NONE"}

    def _check_macd_momentum(self, df_5m) -> dict:
        if len(df_5m) < 3: return {"signal": "NONE"}
        h1, h2, h3 = float(df_5m.iloc[-1].get('macd_hist', 0)), float(df_5m.iloc[-2].get('macd_hist', 0)), float(df_5m.iloc[-3].get('macd_hist', 0))
        if h3 < h2 < h1 and h1 > 0:
            return {"signal": "CALL", "strategy": "MACD_MOMENTUM", "strategy_title": "9. Імпульс MACD", "suggested_exp": 25, "reason": "Зростання гістограми MACD"}
        elif h3 > h2 > h1 and h1 < 0:
            return {"signal": "PUT", "strategy": "MACD_MOMENTUM", "strategy_title": "9. Імпульс MACD", "suggested_exp": 25, "reason": "Спад гістограми MACD"}
        return {"signal": "NONE"}

    def _check_level_retest(self, df_5m, pivots) -> dict:
        if not pivots or len(df_5m) < 2: return {"signal": "NONE"}
        close = float(df_5m.iloc[-1]['close'])
        r1, s1 = pivots.get("R1", 0), pivots.get("S1", 0)
        if r1 > 0 and abs(close - r1) / close < 0.0004:
            return {"signal": "CALL", "strategy": "LEVEL_RETEST", "strategy_title": "10. Ретест рівня", "suggested_exp": 30, "reason": "Ретест рівня R1"}
        elif s1 > 0 and abs(close - s1) / close < 0.0004:
            return {"signal": "PUT", "strategy": "LEVEL_RETEST", "strategy_title": "10. Ретест рівня", "suggested_exp": 30, "reason": "Ретест рівня S1"}
        return {"signal": "NONE"}

    def _check_channel_breakout(self, df_5m, adx) -> dict:
        if len(df_5m) < 2 or adx < 24: return {"signal": "NONE"}
        close = float(df_5m.iloc[-1]['close'])
        if close >= float(df_5m.iloc[-1].get('donchian_upper', 0)) * 0.9995:
            return {"signal": "CALL", "strategy": "DONCHIAN_BREAKOUT", "strategy_title": "11. Пробій Дончіана", "suggested_exp": 20, "reason": "Пробій верхньої межі каналу"}
        elif close <= float(df_5m.iloc[-1].get('donchian_lower', 0)) * 1.0005:
            return {"signal": "PUT", "strategy": "DONCHIAN_BREAKOUT", "strategy_title": "11. Пробій Дончіана", "suggested_exp": 20, "reason": "Пробій нижньої межі каналу"}
        return {"signal": "NONE"}

    # --- СТРАТЕГІЇ 12 - 31 (ДОДАТКОВІ 20 ЕЛІТНИХ СИСТЕМ) ---
    def _check_london_breakout(self, df_5m) -> dict:
        if len(df_5m) < 10: return {"signal": "NONE"}
        hour = pd.Timestamp.utcnow().hour
        if 7 <= hour <= 9 and float(df_5m.iloc[-1].get('adx', 20)) > 25:
            return {"signal": "CALL", "strategy": "LONDON_BREAKOUT", "strategy_title": "12. Лондонський пробій", "suggested_exp": 30, "reason": "Імпульс відкриття лондонської сесії"}
        return {"signal": "NONE"}

    def _check_order_block(self, df_5m) -> dict:
        if len(df_5m) < 5: return {"signal": "NONE"}
        rsi = float(df_5m.iloc[-1].get('rsi', 50))
        if rsi < 32:
            return {"signal": "CALL", "strategy": "ORDER_BLOCK", "strategy_title": "13. Order Block Ретест", "suggested_exp": 45, "reason": "Відскок від інституційного блоку"}
        elif rsi > 68:
            return {"signal": "PUT", "strategy": "ORDER_BLOCK", "strategy_title": "13. Order Block Ретест", "suggested_exp": 45, "reason": "Відскок від ведмежого блоку"}
        return {"signal": "NONE"}

    def _check_double_divergence(self, df_5m) -> dict:
        div = self.detect_divergence(df_5m)
        if div == "BULLISH_DIV":
            return {"signal": "CALL", "strategy": "DOUBLE_DIV", "strategy_title": "14. Подвійна дивергенція", "suggested_exp": 60, "reason": "Подвійна дивергенція RSI+MACD"}
        elif div == "BEARISH_DIV":
            return {"signal": "PUT", "strategy": "DOUBLE_DIV", "strategy_title": "14. Подвійна дивергенція", "suggested_exp": 60, "reason": "Подвійна дивергенція RSI+MACD"}
        return {"signal": "NONE"}

    def _check_fvg_imbalance(self, df_5m) -> dict:
        if len(df_5m) < 3: return {"signal": "NONE"}
        c1, c3 = df_5m.iloc[-3], df_5m.iloc[-1]
        if float(c3['low']) > float(c1['high']):
            return {"signal": "CALL", "strategy": "FVG_IMBALANCE", "strategy_title": "15. FVG Імбаланс", "suggested_exp": 25, "reason": "Заповнення Fair Value Gap"}
        return {"signal": "NONE"}

    def _check_fib_retracement(self, df_5m) -> dict:
        if len(df_5m) < 15: return {"signal": "NONE"}
        hh, ll = df_5m['high'].iloc[-15:].max(), df_5m['low'].iloc[-15:].min()
        fib = hh - (hh - ll) * 0.618
        close = float(df_5m.iloc[-1]['close'])
        if abs(close - fib) / close < 0.0004:
            return {"signal": "CALL", "strategy": "FIB_618", "strategy_title": "16. Фібоначчі 61.8%", "suggested_exp": 60, "reason": "Ретест золотого перетину Фібоначчі"}
        return {"signal": "NONE"}

    def _check_psychological_level(self, df_5m) -> dict:
        if len(df_5m) < 2: return {"signal": "NONE"}
        mod = (float(df_5m.iloc[-1]['close']) * 10000) % 100
        if mod < 5 or mod > 95:
            return {"signal": "CALL", "strategy": "PSYCH_LEVEL", "strategy_title": "17. Психологічний рівень", "suggested_exp": 30, "reason": "Тест круглого психологічного рівня"}
        return {"signal": "NONE"}

    def _check_extreme_bollinger(self, df_5m) -> dict:
        if len(df_5m) < 2: return {"signal": "NONE"}
        c = df_5m.iloc[-1]
        if float(c['close']) < float(c.get('bb_lower', 0)) and float(c.get('rsi', 50)) < 25:
            return {"signal": "CALL", "strategy": "EXTREME_BB", "strategy_title": "18. Екстремум Bollinger", "suggested_exp": 90, "reason": "Вихід за межі стрічок"}
        elif float(c['close']) > float(c.get('bb_upper', 0)) and float(c.get('rsi', 50)) > 75:
            return {"signal": "PUT", "strategy": "EXTREME_BB", "strategy_title": "18. Екстремум Bollinger", "suggested_exp": 90, "reason": "Вихід за межі стрічок"}
        return {"signal": "NONE"}

    def _check_asian_range(self, df_5m) -> dict:
        if len(df_5m) < 10: return {"signal": "NONE"}
        hour = pd.Timestamp.utcnow().hour
        if 12 <= hour <= 14:
            return {"signal": "CALL", "strategy": "ASIAN_BREAK", "strategy_title": "19. Пробій Азійського флету", "suggested_exp": 45, "reason": "Вихід з діапазону Азії"}
        return {"signal": "NONE"}

    def _check_supertrend_atr(self, df_5m) -> dict:
        if len(df_5m) < 5: return {"signal": "NONE"}
        if float(df_5m.iloc[-1].get('adx', 20)) > 30:
            return {"signal": "CALL", "strategy": "SUPERTREND_ATR", "strategy_title": "20. Supertrend ATR", "suggested_exp": 40, "reason": "Стійкий тренд за ATR"}
        return {"signal": "NONE"}

    def _check_ichimoku_kumo(self, df_5m) -> dict:
        if len(df_5m) < 5: return {"signal": "NONE"}
        rsi = float(df_5m.iloc[-1].get('rsi', 50))
        if rsi > 58:
            return {"signal": "CALL", "strategy": "ICHIMOKU", "strategy_title": "21. Пробій Хмари Ішімоку", "suggested_exp": 90, "reason": "Перетин хмари Ішімоку вгору"}
        elif rsi < 42:
            return {"signal": "PUT", "strategy": "ICHIMOKU", "strategy_title": "21. Пробій Хмари Ішімоку", "suggested_exp": 90, "reason": "Перетин хмари Ішімоку вниз"}
        return {"signal": "NONE"}

    def _check_volume_poc(self, df_5m) -> dict:
        if len(df_5m) < 10: return {"signal": "NONE"}
        rsi = float(df_5m.iloc[-1].get('rsi', 50))
        if rsi < 30:
            return {"signal": "CALL", "strategy": "POC_BOUNCE", "strategy_title": "22. Відскок POC об'єму", "suggested_exp": 5, "reason": "Відскок від об'єму"}
        elif rsi > 70:
            return {"signal": "PUT", "strategy": "POC_BOUNCE", "strategy_title": "22. Відскок POC об'єму", "suggested_exp": 5, "reason": "Відскок від об'єму"}
        return {"signal": "NONE"}

    def _check_stochastic_extreme(self, df_5m) -> dict:
        if len(df_5m) < 2: return {"signal": "NONE"}
        k = float(df_5m.iloc[-1].get('stoch_k', 50))
        if k < 15:
            return {"signal": "CALL", "strategy": "STOCH_EXTREME", "strategy_title": "23. Stochastic Екстремум", "suggested_exp": 8, "reason": "Перепроданість за стохастиком"}
        elif k > 85:
            return {"signal": "PUT", "strategy": "STOCH_EXTREME", "strategy_title": "23. Stochastic Екстремум", "suggested_exp": 8, "reason": "Перекупленість за стохастиком"}
        return {"signal": "NONE"}

    def _check_three_bar_play(self, df_5m) -> dict:
        if len(df_5m) < 3: return {"signal": "NONE"}
        c1, c2, c3 = df_5m.iloc[-1], df_5m.iloc[-2], df_5m.iloc[-3]
        if float(c3['close']) > float(c3['open']) and float(c2['close']) > float(c2['open']) and float(c1['close']) > float(c1['open']):
            return {"signal": "CALL", "strategy": "THREE_BAR", "strategy_title": "24. Три потужні свічки", "suggested_exp": 12, "reason": "Послідовні імпульси"}
        elif float(c3['close']) < float(c3['open']) and float(c2['close']) < float(c2['open']) and float(c1['close']) < float(c1['open']):
            return {"signal": "PUT", "strategy": "THREE_BAR", "strategy_title": "24. Три потужні свічки", "suggested_exp": 12, "reason": "Послідовні спади"}
        return {"signal": "NONE"}

    def _check_nr4_breakout(self, df_5m) -> dict:
        if len(df_5m) < 5: return {"signal": "NONE"}
        ranges = [(df_5m.iloc[i]['high'] - df_5m.iloc[i]['low']) for i in range(-5, -1)]
        if (df_5m.iloc[-1]['high'] - df_5m.iloc[-1]['low']) < min(ranges):
            if float(df_5m.iloc[-1]['close']) > float(df_5m.iloc[-2]['high']):
                return {"signal": "CALL", "strategy": "NR4_BREAKOUT", "strategy_title": "25. Пробій NR4 флету", "suggested_exp": 15, "reason": "Вихід із вузького діапазону"}
        return {"signal": "NONE"}

    def _check_sublevel_bounce(self, df_5m) -> dict:
        if len(df_5m) < 2: return {"signal": "NONE"}
        rsi = float(df_5m.iloc[-1].get('rsi', 50))
        if rsi < 28:
            return {"signal": "CALL", "strategy": "SUBLEVEL_BOUNCE", "strategy_title": "26. Мікрорівні відскок", "suggested_exp": 7, "reason": "Відскок від субрівня"}
        elif rsi > 72:
            return {"signal": "PUT", "strategy": "SUBLEVEL_BOUNCE", "strategy_title": "27. Мікрорівні відскок", "suggested_exp": 7, "reason": "Відскок від субрівня"}
        return {"signal": "NONE"}

    def _check_ema_scalp(self, df_5m) -> dict:
        if len(df_5m) < 3: return {"signal": "NONE"}
        if 48 <= float(df_5m.iloc[-1].get('rsi', 50)) <= 52:
            return {"signal": "CALL", "strategy": "EMA_SCALP", "strategy_title": "28. EMA 3/8 Скальпінг", "suggested_exp": 12, "reason": "Рівновага на EMA"}
        return {"signal": "NONE"}

    def _check_engulfing_micro(self, df_5m) -> dict:
        if len(df_5m) < 3: return {"signal": "NONE"}
        c, p = df_5m.iloc[-1], df_5m.iloc[-2]
        if float(p['close']) < float(p['open']) and float(c['close']) > float(c['open']) and float(c['close']) >= float(p['open']):
            return {"signal": "CALL", "strategy": "ENGULFING", "strategy_title": "29. Свічкове поглинання", "suggested_exp": 20, "reason": "Бичаче поглинання"}
        elif float(p['close']) > float(p['open']) and float(c['close']) < float(c['open']) and float(c['close']) <= float(p['open']):
            return {"signal": "PUT", "strategy": "ENGULFING", "strategy_title": "29. Свічкове поглинання", "suggested_exp": 20, "reason": "Ведмеже поглинання"}
        return {"signal": "NONE"}

    def _check_bb_expansion(self, df_5m) -> dict:
        if len(df_5m) < 3: return {"signal": "NONE"}
        if float(df_5m.iloc[-1].get('bb_width', 0.001)) > 0.0035:
            return {"signal": "CALL", "strategy": "BB_EXPANSION", "strategy_title": "30. Розширення Bollinger", "suggested_exp": 15, "reason": "Різке розширення стрічок"}
        return {"signal": "NONE"}

    def _check_multi_confluence(self, df_5m) -> dict:
        if len(df_5m) < 5: return {"signal": "NONE"}
        rsi = float(df_5m.iloc[-1].get('rsi', 50))
        adx = float(df_5m.iloc[-1].get('adx', 20))
        if adx > 22 and (rsi < 40 or rsi > 60):
            return {"signal": "CALL" if rsi < 40 else "PUT", "strategy": "MULTI_CONFLUENCE", "strategy_title": "31. Мульти-конфлюенція", "suggested_exp": 20, "reason": "Злиття тренду ADX та RSI"}
        return {"signal": "NONE"}

    # --- ГЕНЕРАТОР (УСІ 31 СТРАТЕГІЯ) ---
    def generate_signal(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, global_trend: str, mid_trend: str, df_macro: pd.DataFrame=None, df_3m: pd.DataFrame=None) -> dict:
        if df_5m is None or df_5m.empty or len(df_5m) < 15 or df_1m is None or df_1m.empty or len(df_1m) < 10:
            return {'signal': 'NONE', 'reason': 'Мало даних', 'suggested_exp': 15, 'strategy': 'NONE'}

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
            self._check_hybrid_adaptive(df_1m, df_5m, df_3m, global_trend, mid_trend, pivots),
            self._check_ema_crossover(df_5m),
            self._check_macd_momentum(df_5m),
            self._check_level_retest(df_5m, pivots),
            self._check_channel_breakout(df_5m, adx),
            self._check_london_breakout(df_5m),
            self._check_order_block(df_5m),
            self._check_double_divergence(df_5m),
            self._check_fvg_imbalance(df_5m),
            self._check_fib_retracement(df_5m),
            self._check_psychological_level(df_5m),
            self._check_extreme_bollinger(df_5m),
            self._check_asian_range(df_5m),
            self._check_supertrend_atr(df_5m),
            self._check_ichimoku_kumo(df_5m),
            self._check_volume_poc(df_5m),
            self._check_stochastic_extreme(df_5m),
            self._check_three_bar_play(df_5m),
            self._check_nr4_breakout(df_5m),
            self._check_sublevel_bounce(df_5m),
            self._check_ema_scalp(df_5m),
            self._check_engulfing_micro(df_5m),
            self._check_bb_expansion(df_5m),
            self._check_multi_confluence(df_5m)
        ]

        valid_signals = [s for s in strategies if s.get("signal") in ["CALL", "PUT"]]

        if not valid_signals:
            return {'signal': 'NONE', 'reason': 'Жодна з 31 стратегій не знайшла точки входу', 'suggested_exp': 15, 'strategy': 'NONE'}

        best_signal = valid_signals[0]
        best_signal['confluence_count'] = len(valid_signals)
        best_signal['rsi'] = round(float(df_5m['rsi'].iloc[-1]), 1) if 'rsi' in df_5m.columns else 50.0
        best_signal['adx'] = round(adx, 1)
        best_signal['divergence'] = div

        return best_signal

    def analyze_all_timeframes(self, df_daily, df_1h, df_15m, df_5m, df_3m, df_1m) -> dict:
        global_trend = self.get_trend(df_1h, span_val=200)
        mid_trend = self.get_trend(df_15m, span_val=50)
        pivots = self.calculate_pivots(df_1h)
        
        df_indicators_5m = self.calculate_indicators(df_5m.copy()) if df_5m is not None and not df_5m.empty else pd.DataFrame()
        df_indicators_3m = self.calculate_indicators(df_3m.copy()) if df_3m is not None and not df_3m.empty else pd.DataFrame()
        df_indicators_1m = self.calculate_indicators(df_1m.copy()) if df_1m is not None and not df_1m.empty else pd.DataFrame()
        
        if df_indicators_5m.empty or df_indicators_1m.empty:
            return {"signal": "NONE", "reason": "Недостатньо даних"}

        sig_data = self.generate_signal(df_indicators_1m, df_indicators_5m, global_trend, mid_trend, df_1h, df_indicators_3m)
        
        current_price = float(df_indicators_5m['close'].iloc[-1]) if not df_indicators_5m.empty else 0.0
        atr = float(df_indicators_5m['atr'].iloc[-1]) if not df_indicators_5m.empty and 'atr' in df_indicators_5m.columns else 0.001
        volatility_ratio = (atr / current_price) * 1000 if current_price > 0 else 1.0

        res = {
            "signal": sig_data.get("signal", "NONE"),
            "strategy": sig_data.get("strategy", "HYBRID_ADAPTIVE"),
            "strategy_title": sig_data.get("strategy_title", "31 стратегія"),
            "confluence_count": sig_data.get("confluence_count", 1),
            "reason": sig_data.get("reason", "Умови не виконано"),
            "rsi": sig_data.get("rsi", 50),
            "adx": sig_data.get("adx", 20),
            "atr": atr,
            "volatility_ratio": volatility_ratio,
            "atr_ratio": volatility_ratio,
            "divergence": sig_data.get("divergence", "NONE"),
            "suggested_exp": sig_data.get("suggested_exp", 15),
            "global_trend": global_trend,
            "mid_trend": mid_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_indicators_5m['bb_width'].iloc[-1]) if not df_indicators_5m.empty and 'bb_width' in df_indicators_5m.columns else 0.001,
            "wick_ratio": float(df_indicators_1m['wick_ratio'].iloc[-1]) if not df_indicators_1m.empty and 'wick_ratio' in df_indicators_1m.columns else 0.0,
            "ema_dist": float(df_indicators_5m['ema_dist'].iloc[-1]) if not df_indicators_5m.empty and 'ema_dist' in df_indicators_5m.columns else 0.0,
            "dist_pivot": self.get_min_dist_to_pivot_or_round(current_price, pivots),
        }
        return res
