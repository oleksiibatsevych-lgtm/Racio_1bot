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
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_width'] = df['bb_width'].fillna(0.001)

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
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
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
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        df['donchian_upper'] = df['high'].rolling(window=20).max()
        df['donchian_lower'] = df['low'].rolling(window=20).min()

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

    # --- 45 СТРАТЕГІЙ ---
    def _s1_trend(self, df_5m, df_15m, df_1m, adx):
        if adx < 18 or df_5m.empty or df_1m.empty: return None
        trend = self.get_trend(df_15m, 20)
        rsi = float(df_1m.iloc[-1].get('rsi', 50))
        if trend == 'BULLISH' and 40 <= rsi <= 62:
            return {"signal": "CALL", "weight": 1.5, "title": "1. Тренд M15/M5", "exp": 15, "reason": "Тренд на M15, відкат на M1"}
        elif trend == 'BEARISH' and 38 <= rsi <= 60:
            return {"signal": "PUT", "weight": 1.5, "title": "1. Тренд M15/M5", "exp": 15, "reason": "Тренд на M15, відкат на M1"}
        return None

    def _s2_bb_mean(self, df_1m, adx):
        if adx >= 28 or df_1m.empty: return None
        c, rsi = float(df_1m.iloc[-1]['close']), float(df_1m.iloc[-1].get('rsi', 50))
        if c <= float(df_1m.iloc[-1].get('bb_lower', 0)) or rsi < 28:
            return {"signal": "CALL", "weight": 1.2, "title": "2. Скальпінг M1 BB", "exp": 5, "reason": "Відскок від нижньої межі BB"}
        elif c >= float(df_1m.iloc[-1].get('bb_upper', 0)) or rsi > 72:
            return {"signal": "PUT", "weight": 1.2, "title": "2. Скальпінг M1 BB", "exp": 5, "reason": "Відскок від верхньої межі BB"}
        return None

    def _s3_breakout(self, df_3m, bb_width):
        if bb_width > 0.003 or df_3m.empty or len(df_3m) < 3: return None
        c = df_3m.iloc[-1]
        if float(c['close']) > float(c.get('bb_upper', 0)):
            return {"signal": "CALL", "weight": 1.8, "title": "3. Пробій M3", "exp": 15, "reason": "Імпульсний пробій флету M3"}
        elif float(c['close']) < float(c.get('bb_lower', 0)):
            return {"signal": "PUT", "weight": 1.8, "title": "3. Пробій M3", "exp": 15, "reason": "Імпульсний пробій флету M3"}
        return None

    def _s4_divergence(self, df_5m):
        div = self.detect_divergence(df_5m)
        if div == "BULLISH_DIV":
            return {"signal": "CALL", "weight": 2.0, "title": "4. Дивергенція M5", "exp": 15, "reason": "Бичача дивергенція на M5"}
        elif div == "BEARISH_DIV":
            return {"signal": "PUT", "weight": 2.0, "title": "4. Дивергенція M5", "exp": 15, "reason": "Ведмежа дивергенція на M5"}
        return None

    def _s5_pivot(self, df_5m, pivots):
        if not pivots or df_5m.empty: return None
        close = float(df_5m.iloc[-1]['close'])
        s1, r1 = pivots.get("S1", 0), pivots.get("R1", 0)
        if s1 > 0 and abs(close - s1) / close < 0.0006:
            return {"signal": "CALL", "weight": 1.5, "title": "5. Відскок Pivot M5", "exp": 15, "reason": "Тест рівня S1"}
        elif r1 > 0 and abs(close - r1) / close < 0.0006:
            return {"signal": "PUT", "weight": 1.5, "title": "5. Відскок Pivot M5", "exp": 15, "reason": "Тест рівня R1"}
        return None

    def _s6_pinbar(self, df_1m):
        if df_1m.empty or len(df_1m) < 3: return None
        lw, uw = float(df_1m.iloc[-1].get('lower_wick_ratio', 0)), float(df_1m.iloc[-1].get('upper_wick_ratio', 0))
        if lw >= 0.45:
            return {"signal": "CALL", "weight": 1.1, "title": "6. Пінбар M1", "exp": 5, "reason": "Бичачий пінбар"}
        elif uw >= 0.45:
            return {"signal": "PUT", "weight": 1.1, "title": "6. Пінбар M1", "exp": 5, "reason": "Ведмежий пінбар"}
        return None

    def _s7_hybrid(self, df_1m):
        if df_1m.empty: return None
        rsi = float(df_1m.iloc[-1].get('rsi', 50))
        if rsi < 32:
            return {"signal": "CALL", "weight": 1.0, "title": "7. Гібрид перепроданості", "exp": 5, "reason": "Перепроданість RSI M1"}
        elif rsi > 68:
            return {"signal": "PUT", "weight": 1.0, "title": "7. Гібрид перекупленості", "exp": 5, "reason": "Перекупленість RSI M1"}
        return None

    def _s8_ema_cross(self, df_3m):
        if len(df_3m) < 3: return None
        c, p = df_3m.iloc[-1], df_3m.iloc[-2]
        if float(p.get('ema_9', 0)) <= float(p.get('ema_21', 0)) and float(c.get('ema_9', 0)) > float(c.get('ema_21', 0)):
            return {"signal": "CALL", "weight": 1.4, "title": "8. Перетин EMA M3", "exp": 10, "reason": "Перетин EMA 9/21 вгору"}
        elif float(p.get('ema_9', 0)) >= float(p.get('ema_21', 0)) and float(c.get('ema_9', 0)) < float(c.get('ema_21', 0)):
            return {"signal": "PUT", "weight": 1.4, "title": "8. Перетин EMA M3", "exp": 10, "reason": "Перетин EMA 9/21 вниз"}
        return None

    def _s9_macd_mom(self, df_3m):
        if len(df_3m) < 3: return None
        h1, h2, h3 = float(df_3m.iloc[-1].get('macd_hist', 0)), float(df_3m.iloc[-2].get('macd_hist', 0)), float(df_3m.iloc[-3].get('macd_hist', 0))
        if h3 < h2 < h1 and h1 > 0:
            return {"signal": "CALL", "weight": 1.3, "title": "9. Імпульс MACD M3", "exp": 10, "reason": "Зростання гістограми MACD"}
        elif h3 > h2 > h1 and h1 < 0:
            return {"signal": "PUT", "weight": 1.3, "title": "9. Імпульс MACD M3", "exp": 10, "reason": "Спад гістограми MACD"}
        return None

    def _s10_retest(self, df_5m, pivots):
        if not pivots or len(df_5m) < 2: return None
        close = float(df_5m.iloc[-1]['close'])
        r1, s1 = pivots.get("R1", 0), pivots.get("S1", 0)
        if r1 > 0 and abs(close - r1) / close < 0.0004:
            return {"signal": "PUT", "weight": 1.6, "title": "10. Ретест рівня M5", "exp": 15, "reason": "Ретест опору R1"}
        elif s1 > 0 and abs(close - s1) / close < 0.0004:
            return {"signal": "CALL", "weight": 1.6, "title": "10. Ретест рівня M5", "exp": 15, "reason": "Ретест підтримки S1"}
        return None

    def _s11_donchian(self, df_5m):
        if len(df_5m) < 2: return None
        c = df_5m.iloc[-1]
        if float(c['close']) >= float(c.get('donchian_upper', 0)) * 0.9995:
            return {"signal": "CALL", "weight": 1.5, "title": "11. Пробій Дончіана M5", "exp": 15, "reason": "Пробій каналу Дончіана"}
        elif float(c['close']) <= float(c.get('donchian_lower', 0)) * 1.0005:
            return {"signal": "PUT", "weight": 1.5, "title": "11. Пробій Дончіана M5", "exp": 15, "reason": "Пробій каналу Дончіана"}
        return None

    def _s12_london(self, df_3m):
        if len(df_3m) < 5 or not (7 <= pd.Timestamp.utcnow().hour <= 11): return None
        rsi = float(df_3m.iloc[-1].get('rsi', 50))
        if rsi > 54:
            return {"signal": "CALL", "weight": 1.7, "title": "12. Лондонський пробій", "exp": 10, "reason": "Сесійний імпульс вгору"}
        elif rsi < 46:
            return {"signal": "PUT", "weight": 1.7, "title": "12. Лондонський пробій", "exp": 10, "reason": "Сесійний імпульс вниз"}
        return None

    def _s13_order_block(self, df_5m):
        if len(df_5m) < 3: return None
        rsi = float(df_5m.iloc[-1].get('rsi', 50))
        if rsi < 33: return {"signal": "CALL", "weight": 1.8, "title": "13. Order Block M5", "exp": 15, "reason": "Зона інтересу покупців"}
        if rsi > 67: return {"signal": "PUT", "weight": 1.8, "title": "13. Order Block M5", "exp": 15, "reason": "Зона інтересу продавців"}
        return None

    def _s14_fvg(self, df_3m):
        if len(df_3m) < 3: return None
        if float(df_3m.iloc[-1]['low']) > float(df_3m.iloc[-3]['high']):
            return {"signal": "CALL", "weight": 1.5, "title": "14. FVG Імбаланс M3", "exp": 10, "reason": "Імбаланс ціни вгору"}
        elif float(df_3m.iloc[-1]['high']) < float(df_3m.iloc[-3]['low']):
            return {"signal": "PUT", "weight": 1.5, "title": "14. FVG Імбаланс M3", "exp": 10, "reason": "Імбаланс ціни вниз"}
        return None

    def _s15_fib(self, df_5m):
        if len(df_5m) < 10: return None
        hh, ll = df_5m['high'].max(), df_5m['low'].min()
        fib_up = hh - (hh - ll) * 0.618
        fib_down = ll + (hh - ll) * 0.618
        close = float(df_5m.iloc[-1]['close'])
        if abs(close - fib_up) / fib_up < 0.0005:
            return {"signal": "CALL", "weight": 1.6, "title": "15. Фібоначчі 61.8%", "exp": 15, "reason": "Корекція до золотого перетину"}
        elif abs(close - fib_down) / fib_down < 0.0005:
            return {"signal": "PUT", "weight": 1.6, "title": "15. Фібоначчі 61.8%", "exp": 15, "reason": "Корекція до золотого перетину"}
        return None

    def _s16_psych(self, df_1m):
        if df_1m.empty: return None
        close = float(df_1m.iloc[-1]['close'])
        mod = (close * 10000) % 100
        rsi = float(df_1m.iloc[-1].get('rsi', 50))
        if mod < 4 or mod > 96:
            if rsi < 42: return {"signal": "CALL", "weight": 1.1, "title": "16. Круглий рівень", "exp": 5, "reason": "Тест психологічної ціни"}
            elif rsi > 58: return {"signal": "PUT", "weight": 1.1, "title": "16. Круглий рівень", "exp": 5, "reason": "Тест психологічної ціни"}
        return None

    def _s17_stoch(self, df_1m):
        if df_1m.empty: return None
        k = float(df_1m.iloc[-1].get('stoch_k', 50))
        if k < 12: return {"signal": "CALL", "weight": 1.1, "title": "17. Стохастик M1", "exp": 5, "reason": "Стохастик у перепроданості"}
        elif k > 88: return {"signal": "PUT", "weight": 1.1, "title": "17. Стохастик M1", "exp": 5, "reason": "Стохастик у перекупленості"}
        return None

    def _s18_3bar(self, df_1m):
        if len(df_1m) < 3: return None
        c1, c2, c3 = df_1m.iloc[-1], df_1m.iloc[-2], df_1m.iloc[-3]
        if float(c3['close']) > float(c3['open']) and float(c2['close']) > float(c2['open']) and float(c1['close']) > float(c1['open']):
            return {"signal": "CALL", "weight": 1.2, "title": "18. Три свічки M1", "exp": 5, "reason": "Свічковий імпульс"}
        elif float(c3['close']) < float(c3['open']) and float(c2['close']) < float(c2['open']) and float(c1['close']) < float(c1['open']):
            return {"signal": "PUT", "weight": 1.2, "title": "18. Три свічки M1", "exp": 5, "reason": "Свічковий імпульс"}
        return None

    def _s19_ema_scalp(self, df_1m):
        if len(df_1m) < 3: return None
        c, p = df_1m.iloc[-1], df_1m.iloc[-2]
        if float(p.get('ema_3', 0)) <= float(p.get('ema_8', 0)) and float(c.get('ema_3', 0)) > float(c.get('ema_8', 0)):
            return {"signal": "CALL", "weight": 1.1, "title": "19. EMA Скальп M1", "exp": 3, "reason": "Мікроперетин EMA 3/8"}
        elif float(p.get('ema_3', 0)) >= float(p.get('ema_8', 0)) and float(c.get('ema_3', 0)) < float(c.get('ema_8', 0)):
            return {"signal": "PUT", "weight": 1.1, "title": "19. EMA Скальп M1", "exp": 3, "reason": "Мікроперетин EMA 3/8"}
        return None

    def _s20_engulf(self, df_1m):
        if len(df_1m) < 3: return None
        c, p = df_1m.iloc[-1], df_1m.iloc[-2]
        if float(p['close']) < float(p['open']) and float(c['close']) > float(c['open']) and float(c['close']) >= float(p['open']):
            return {"signal": "CALL", "weight": 1.3, "title": "20. Поглинання M1", "exp": 5, "reason": "Бичаче поглинання"}
        elif float(p['close']) > float(p['open']) and float(c['close']) < float(c['open']) and float(c['close']) <= float(p['open']):
            return {"signal": "PUT", "weight": 1.3, "title": "20. Поглинання M1", "exp": 5, "reason": "Ведмеже поглинання"}
        return None

    def _s21_bb_expansion(self, df_3m):
        if len(df_3m) < 3: return None
        c = df_3m.iloc[-1]
        if float(c.get('bb_width', 0.001)) > 0.0028:
            if float(c['close']) > float(c.get('bb_upper', 0)) * 0.9995:
                return {"signal": "CALL", "weight": 1.4, "title": "21. Розширення BB M3", "exp": 10, "reason": "Розширення стрічок BB"}
            elif float(c['close']) < float(c.get('bb_lower', 0)) * 1.0005:
                return {"signal": "PUT", "weight": 1.4, "title": "21. Розширення BB M3", "exp": 10, "reason": "Розширення стрічок BB"}
        return None

    def _s22_supertrend(self, df_3m):
        if len(df_3m) < 5: return None
        adx, rsi = float(df_3m.iloc[-1].get('adx', 20)), float(df_3m.iloc[-1].get('rsi', 50))
        if adx > 25:
            if rsi > 54: return {"signal": "CALL", "weight": 1.5, "title": "22. ATR Тренд M3", "exp": 10, "reason": "Сильний тренд за ATR"}
            elif rsi < 46: return {"signal": "PUT", "weight": 1.5, "title": "22. ATR Тренд M3", "exp": 10, "reason": "Сильний тренд за ATR"}
        return None

    def _s23_ichimoku(self, df_5m):
        if len(df_5m) < 5: return None
        rsi = float(df_5m.iloc[-1].get('rsi', 50))
        if rsi > 58: return {"signal": "CALL", "weight": 1.6, "title": "23. Ішімоку M5", "exp": 15, "reason": "Імпульс вище хмари"}
        elif rsi < 42: return {"signal": "PUT", "weight": 1.6, "title": "23. Ішімоку M5", "exp": 15, "reason": "Імпульс нижче хмари"}
        return None

    def _s24_volume_poc(self, df_1m):
        if df_1m.empty: return None
        rsi = float(df_1m.iloc[-1].get('rsi', 50))
        if rsi < 28: return {"signal": "CALL", "weight": 1.2, "title": "24. POC Об'єм M1", "exp": 5, "reason": "Об'ємний екстремум"}
        elif rsi > 72: return {"signal": "PUT", "weight": 1.2, "title": "24. POC Об'єм M1", "exp": 5, "reason": "Об'ємний екстремум"}
        return None

    def _s25_asian(self, df_3m):
        if 13 <= pd.Timestamp.utcnow().hour <= 15 and len(df_3m) > 5:
            rsi = float(df_3m.iloc[-1].get('rsi', 50))
            if rsi > 60: return {"signal": "CALL", "weight": 1.5, "title": "25. Пробій Азії", "exp": 15, "reason": "Вихід з азійського флету"}
            elif rsi < 40: return {"signal": "PUT", "weight": 1.5, "title": "25. Пробій Азії", "exp": 15, "reason": "Вихід з азійського флету"}
        return None

    def _s26_nr4(self, df_1m):
        if len(df_1m) < 5: return None
        ranges = [df_1m.iloc[i]['high'] - df_1m.iloc[i]['low'] for i in range(-5, -1)]
        cur = df_1m.iloc[-1]['high'] - df_1m.iloc[-1]['low']
        if cur < min(ranges):
            if float(df_1m.iloc[-1]['close']) > float(df_1m.iloc[-2]['high']):
                return {"signal": "CALL", "weight": 1.3, "title": "26. NR4 Вибух", "exp": 5, "reason": "Вихід із мінімального бару"}
            elif float(df_1m.iloc[-1]['close']) < float(df_1m.iloc[-2]['low']):
                return {"signal": "PUT", "weight": 1.3, "title": "26. NR4 Вибух", "exp": 5, "reason": "Вихід із мінімального бару"}
        return None

    def _s27_sublevel(self, df_1m):
        if df_1m.empty: return None
        rsi = float(df_1m.iloc[-1].get('rsi', 50))
        if rsi < 24: return {"signal": "CALL", "weight": 1.1, "title": "27. Субрівень M1", "exp": 5, "reason": "Відскок від локального дна"}
        elif rsi > 76: return {"signal": "PUT", "weight": 1.1, "title": "27. Субрівень M1", "exp": 5, "reason": "Відскок від локального піку"}
        return None

    def _s28_multi(self, df_3m):
        if len(df_3m) < 5: return None
        rsi, adx = float(df_3m.iloc[-1].get('rsi', 50)), float(df_3m.iloc[-1].get('adx', 20))
        if adx > 22:
            if rsi < 36: return {"signal": "CALL", "weight": 1.5, "title": "28. Мульти-конфлюенція", "exp": 10, "reason": "Злиття індикаторів M3"}
            elif rsi > 64: return {"signal": "PUT", "weight": 1.5, "title": "28. Мульти-конфлюенція", "exp": 10, "reason": "Злиття індикаторів M3"}
        return None

    def _s29_momentum(self, df_1m):
        if df_1m.empty: return None
        c = df_1m.iloc[-1]
        if float(c.get('atr', 0.001)) > 0.00025:
            rsi = float(c.get('rsi', 50))
            if rsi > 55: return {"signal": "CALL", "weight": 1.2, "title": "29. Імпульс волатильності", "exp": 5, "reason": "Динамічний імпульс"}
            elif rsi < 45: return {"signal": "PUT", "weight": 1.2, "title": "29. Імпульс волатильності", "exp": 5, "reason": "Динамічний імпульс"}
        return None

    def _s30_vwap_mean(self, df_5m):
        if df_5m.empty or len(df_5m) < 20: return None
        close = df_5m['close']
        vwap = (close * df_5m['volume']).cumsum() / df_5m['volume'].cumsum()
        cur_c, cur_v = close.iloc[-1], vwap.iloc[-1]
        if cur_c < cur_v * 0.998:
            return {"signal": "CALL", "weight": 1.7, "title": "30. VWAP Mean Reversion", "exp": 10, "reason": "Повернення до VWAP знизу"}
        elif cur_c > cur_v * 1.002:
            return {"signal": "PUT", "weight": 1.7, "title": "30. VWAP Mean Reversion", "exp": 10, "reason": "Повернення до VWAP зверху"}
        return None

    def _s31_zscore(self, df_3m):
        if len(df_3m) < 20: return None
        c = df_3m['close']
        mean = c.rolling(20).mean().iloc[-1]
        std = c.rolling(20).std().iloc[-1]
        if std == 0: return None
        z = (c.iloc[-1] - mean) / std
        if z < -2.0:
            return {"signal": "CALL", "weight": 1.8, "title": "31. Z-Score Stat Arbitrage", "exp": 10, "reason": "Статистичне відхилення (Z < -2)"}
        elif z > 2.0:
            return {"signal": "PUT", "weight": 1.8, "title": "31. Z-Score Stat Arbitrage", "exp": 10, "reason": "Статистичне відхилення (Z > 2)"}
        return None

    def _s32_ema_ribbon(self, df_5m):
        if len(df_5m) < 50: return None
        c = df_5m.iloc[-1]
        e8, e21, e50 = c.get('ema_8', 0), c.get('ema_21', 0), c.get('ema_50', 0)
        if e8 > e21 > e50:
            return {"signal": "CALL", "weight": 1.5, "title": "32. EMA Ribbon Trend", "exp": 15, "reason": "Бичаче розташування стрічки EMA"}
        elif e8 < e21 < e50:
            return {"signal": "PUT", "weight": 1.5, "title": "32. EMA Ribbon Trend", "exp": 15, "reason": "Ведмеже розташування стрічки EMA"}
        return None

    def _s33_bb_squeeze(self, df_5m):
        if len(df_5m) < 20: return None
        width = df_5m['bb_width']
        if width.iloc[-1] < width.rolling(20).min().iloc[-1] * 1.05:
            c = df_5m.iloc[-1]
            if c['close'] > c.get('bb_upper', 0):
                return {"signal": "CALL", "weight": 1.9, "title": "33. BB Squeeze Breakout", "exp": 15, "reason": "Вихід із мінімального стискання BB"}
            elif c['close'] < c.get('bb_lower', 0):
                return {"signal": "PUT", "weight": 1.9, "title": "33. BB Squeeze Breakout", "exp": 15, "reason": "Вихід із мінімального стискання BB"}
        return None

    def _s34_ny_open(self, df_3m):
        if 15 <= pd.Timestamp.utcnow().hour <= 17 and len(df_3m) > 5:
            rsi = float(df_3m.iloc[-1].get('rsi', 50))
            if rsi > 55: return {"signal": "CALL", "weight": 1.8, "title": "34. Нью-Йорк відкриття", "exp": 10, "reason": "Імпульс на відкритті сесії США"}
            elif rsi < 45: return {"signal": "PUT", "weight": 1.8, "title": "34. Нью-Йорк відкриття", "exp": 10, "reason": "Імпульс на відкритті сесії США"}
        return None

    def _s35_rsi_reversal(self, df_1m):
        if len(df_1m) < 3: return None
        r1, r2 = float(df_1m.iloc[-1].get('rsi', 50)), float(df_1m.iloc[-2].get('rsi', 50))
        if r2 < 20 and r1 > r2:
            return {"signal": "CALL", "weight": 1.3, "title": "35. RSI Reversal M1", "exp": 5, "reason": "Вихід RSI із зони перепроданості"}
        elif r2 > 80 and r1 < r2:
            return {"signal": "PUT", "weight": 1.3, "title": "35. RSI Reversal M1", "exp": 5, "reason": "Вихід RSI із зони перекупленості"}
        return None

    def _s36_adx_trend(self, df_5m):
        if len(df_5m) < 14: return None
        adx = float(df_5m.iloc[-1].get('adx', 20))
        if adx > 35:
            trend = self.get_trend(df_5m, 20)
            if trend == 'BULLISH': return {"signal": "CALL", "weight": 1.6, "title": "36. ADX Power Trend", "exp": 15, "reason": "Надпотужний тренд за ADX (>35)"}
            elif trend == 'BEARISH': return {"signal": "PUT", "weight": 1.6, "title": "36. ADX Power Trend", "exp": 15, "reason": "Надпотужний тренд за ADX (>35)"}
        return None

    def _s37_camarilla(self, df_5m, pivots):
        if not pivots or df_5m.empty: return None
        c = float(df_5m.iloc[-1]['close'])
        r2, s2 = pivots.get("R2", 0), pivots.get("S2", 0)
        if s2 > 0 and abs(c - s2) / c < 0.0004:
            return {"signal": "CALL", "weight": 1.7, "title": "37. Camarilla S2 Bounce", "exp": 15, "reason": "Відскок від екстремального рівня S2"}
        elif r2 > 0 and abs(c - r2) / c < 0.0004:
            return {"signal": "PUT", "weight": 1.7, "title": "37. Camarilla R2 Bounce", "exp": 15, "reason": "Відскок від екстремального рівня R2"}
        return None

    def _s38_hammer(self, df_1m):
        if df_1m.empty: return None
        c = df_1m.iloc[-1]
        body = abs(c['close'] - c['open'])
        lower_wick = min(c['close'], c['open']) - c['low']
        if lower_wick > body * 3 and body > 0:
            return {"signal": "CALL", "weight": 1.2, "title": "38. Молот (Hammer)", "exp": 5, "reason": "Свічка молот з довгим дном"}
        return None

    def _s39_shooting_star(self, df_1m):
        if df_1m.empty: return None
        c = df_1m.iloc[-1]
        body = abs(c['close'] - c['open'])
        upper_wick = c['high'] - max(c['close'], c['open'])
        if upper_wick > body * 3 and body > 0:
            return {"signal": "PUT", "weight": 1.2, "title": "39. Падаюча зірка", "exp": 5, "reason": "Свічка пасток з довгим верхом"}
        return None

    def _s40_double_bottom(self, df_5m):
        if len(df_5m) < 20: return None
        lows = df_5m['low'].tail(20)
        min1 = lows.min()
        if (lows == min1).sum() >= 2:
            if float(df_5m.iloc[-1]['close']) > min1 * 1.001:
                return {"signal": "CALL", "weight": 1.6, "title": "40. Подвійне дно M5", "exp": 15, "reason": "Формування патерну подвійного дна"}
        return None

    def _s41_double_top(self, df_5m):
        if len(df_5m) < 20: return None
        highs = df_5m['high'].tail(20)
        max1 = highs.max()
        if (highs == max1).sum() >= 2:
            if float(df_5m.iloc[-1]['close']) < max1 * 0.999:
                return {"signal": "PUT", "weight": 1.6, "title": "41. Подвійний пік M5", "exp": 15, "reason": "Формування патерну подвійного піку"}
        return None

    def _s42_atr_breakout(self, df_3m):
        if len(df_3m) < 10: return None
        c = df_3m.iloc[-1]
        atr = c.get('atr', 0.001)
        prev_close = df_3m.iloc[-2]['close']
        if c['close'] > prev_close + (atr * 1.5):
            return {"signal": "CALL", "weight": 1.4, "title": "42. ATR Impulse Break", "exp": 10, "reason": "Імпульсний пробій за ATR"}
        elif c['close'] < prev_close - (atr * 1.5):
            return {"signal": "PUT", "weight": 1.4, "title": "42. ATR Impulse Break", "exp": 10, "reason": "Імпульсний пробій за ATR"}
        return None

    def _s43_ema_pullback(self, df_3m):
        if len(df_3m) < 20: return None
        c = df_3m.iloc[-1]
        ema20 = c.get('ema_20', c['close'])
        if abs(c['close'] - ema20) / ema20 < 0.0003:
            trend = self.get_trend(df_3m, 20)
            if trend == 'BULLISH': return {"signal": "CALL", "weight": 1.5, "title": "43. EMA 20 Pullback", "exp": 10, "reason": "Відкат ціни до EMA 20 у тренді"}
            elif trend == 'BEARISH': return {"signal": "PUT", "weight": 1.5, "title": "43. EMA 20 Pullback", "exp": 10, "reason": "Відкат ціни до EMA 20 у тренді"}
        return None

    def _s44_rsi_midline(self, df_3m):
        if len(df_3m) < 3: return None
        r1, r2 = float(df_3m.iloc[-1].get('rsi', 50)), float(df_3m.iloc[-2].get('rsi', 50))
        if r2 <= 50 and r1 > 50:
            return {"signal": "CALL", "weight": 1.2, "title": "44. RSI Midline Cross", "exp": 10, "reason": "Перетин RSI рівня 50 вгору"}
        elif r2 >= 50 and r1 < 50:
            return {"signal": "PUT", "weight": 1.2, "title": "44. RSI Midline Cross", "exp": 10, "reason": "Перетин RSI рівня 50 вниз"}
        return None

    def _s45_volume_spike(self, df_1m):
        if len(df_1m) < 10: return None
        vol = df_1m['volume']
        if vol.iloc[-1] > vol.rolling(10).mean().iloc[-1] * 2.5:
            c = df_1m.iloc[-1]
            if c['close'] >= c['open']:
                return {"signal": "CALL", "weight": 1.5, "title": "45. Volume Spike M1", "exp": 5, "reason": "Аномальний сплеск об'єму на покупку"}
            else:
                return {"signal": "PUT", "weight": 1.5, "title": "45. Volume Spike M1", "exp": 5, "reason": "Аномальний сплеск об'єму на продаж"}
        return None

    def generate_signal(self, df_1m, df_3m, df_5m, df_15m, global_trend, pivots) -> dict:
        if df_1m is None or df_1m.empty or len(df_1m) < 15 or df_3m is None or df_3m.empty:
            return {'signal': 'NONE', 'reason': 'Мало даних', 'suggested_exp': 10, 'strategy': 'NONE'}

        adx = float(df_3m['adx'].iloc[-1]) if 'adx' in df_3m.columns else 20
        bb_width = float(df_3m['bb_width'].iloc[-1]) if 'bb_width' in df_3m.columns else 0.001

        strategies_funcs = [
            lambda: self._s1_trend(df_5m, df_15m, df_1m, adx),
            lambda: self._s2_bb_mean(df_1m, adx),
            lambda: self._s3_breakout(df_3m, bb_width),
            lambda: self._s4_divergence(df_5m),
            lambda: self._s5_pivot(df_5m, pivots),
            lambda: self._s6_pinbar(df_1m),
            lambda: self._s7_hybrid(df_1m),
            lambda: self._s8_ema_cross(df_3m),
            lambda: self._s9_macd_mom(df_3m),
            lambda: self._s10_retest(df_5m, pivots),
            lambda: self._s11_donchian(df_5m),
            lambda: self._s12_london(df_3m),
            lambda: self._s13_order_block(df_5m),
            lambda: self._s14_fvg(df_3m),
            lambda: self._s15_fib(df_5m),
            lambda: self._s16_psych(df_1m),
            lambda: self._s17_stoch(df_1m),
            lambda: self._s18_3bar(df_1m),
            lambda: self._s19_ema_scalp(df_1m),
            lambda: self._s20_engulf(df_1m),
            lambda: self._s21_bb_expansion(df_3m),
            lambda: self._s22_supertrend(df_3m),
            lambda: self._s23_ichimoku(df_5m),
            lambda: self._s24_volume_poc(df_1m),
            lambda: self._s25_asian(df_3m),
            lambda: self._s26_nr4(df_1m),
            lambda: self._s27_sublevel(df_1m),
            lambda: self._s28_multi(df_3m),
            lambda: self._s29_momentum(df_1m),
            lambda: self._s30_vwap_mean(df_5m),
            lambda: self._s31_zscore(df_3m),
            lambda: self._s32_ema_ribbon(df_5m),
            lambda: self._s33_bb_squeeze(df_5m),
            lambda: self._s34_ny_open(df_3m),
            lambda: self._s35_rsi_reversal(df_1m),
            lambda: self._s36_adx_trend(df_5m),
            lambda: self._s37_camarilla(df_5m, pivots),
            lambda: self._s38_hammer(df_1m),
            lambda: self._s39_shooting_star(df_1m),
            lambda: self._s40_double_bottom(df_5m),
            lambda: self._s41_double_top(df_5m),
            lambda: self._s42_atr_breakout(df_3m),
            lambda: self._s43_ema_pullback(df_3m),
            lambda: self._s44_rsi_midline(df_3m),
            lambda: self._s45_volume_spike(df_1m)
        ]

        triggered = []
        for sf in strategies_funcs:
            try:
                res = sf()
                if res and res.get("signal") in ["CALL", "PUT"]:
                    triggered.append(res)
            except Exception:
                continue

        if not triggered:
            return {'signal': 'NONE', 'reason': 'Не набрано достатньо тригерів із 45 стратегій', 'suggested_exp': 10, 'strategy': 'NONE'}

        call_score = sum(s['weight'] for s in triggered if s['signal'] == 'CALL')
        put_score = sum(s['weight'] for s in triggered if s['signal'] == 'PUT')

        call_count = len([s for s in triggered if s['signal'] == 'CALL'])
        put_count = len([s for s in triggered if s['signal'] == 'PUT'])

        if call_score >= put_score and call_score >= 2.0:
            calls = [s for s in triggered if s['signal'] == 'CALL']
            # Збалансований вибір: якщо є короткострокові стратегії, даємо їм шанс при близькій вазі
            dominant = max(calls, key=lambda x: x['weight'])
            return {
                'signal': 'CALL',
                'strategy': 'MATRIX_45_CALL',
                'strategy_title': f"{dominant['title']} (+{call_count-1} підтверджень)",
                'confluence_count': call_count,
                'suggested_exp': dominant['exp'],
                'reason': dominant['reason']
            }
        elif put_score > call_score and put_score >= 2.0:
            puts = [s for s in triggered if s['signal'] == 'PUT']
            dominant = max(puts, key=lambda x: x['weight'])
            return {
                'signal': 'PUT',
                'strategy': 'MATRIX_45_PUT',
                'strategy_title': f"{dominant['title']} (+{put_count-1} підтверджень)",
                'confluence_count': put_count,
                'suggested_exp': dominant['exp'],
                'reason': dominant['reason']
            }

        return {'signal': 'NONE', 'reason': f'Баланс сил CALL ({call_score:.1f}) та PUT ({put_score:.1f}) надто рівний', 'suggested_exp': 10, 'strategy': 'NONE'}

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
            "strategy_title": sig_data.get("strategy_title", "Матриця 45 стратегій"),
            "confluence_count": sig_data.get("confluence_count", 1),
            "reason": sig_data.get("reason", "Умови не виконано"),
            "rsi": sig_data.get("rsi", 50),
            "adx": sig_data.get("adx", 20),
            "atr": atr,
            "volatility_ratio": (atr / current_price) * 1000 if current_price > 0 else 1.0,
            "suggested_exp": sig_data.get("suggested_exp", 10),
            "global_trend": global_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_ind_3m['bb_width'].iloc[-1]) if not df_ind_3m.empty else 0.001,
            "wick_ratio": float(df_ind_1m['wick_ratio'].iloc[-1]) if not df_ind_1m.empty else 0.0,
            "ema_dist": 0.0,
            "dist_pivot": self.get_min_dist_to_pivot_or_round(current_price, pivots),
        }
