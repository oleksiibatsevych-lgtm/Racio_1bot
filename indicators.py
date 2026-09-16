import pandas as pd
import numpy as np
from datetime import datetime

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df):
        if df is None or df.empty or len(df) < 20:
            return df

        df = df.copy()
        df.columns = [str(col).lower() for col in df.columns]

        if 'close' not in df.columns:
            return df

        # 1. RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        # 2. Bollinger Bands & %B
        sma = df['close'].rolling(window=20).mean()
-       std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_middle'] = sma
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / (sma + 1e-10)).fillna(0.001)
        
        # Розрахунок %B для M1 скальпінгу
        bb_range = df['bb_upper'] - df['bb_lower']
        df['pct_b'] = np.where(bb_range > 0, (df['close'] - df['bb_lower']) / bb_range, 0.5)

        # 3. ATR
        if 'high' in df.columns and 'low' in df.columns:
            high = df['high']
            low = df['low']
            close_prev = df['close'].shift(1)
            tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
            df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean().fillna(0.0010)
        else:
            df['atr'] = 0.0010

        # 4. ADX
        if 'high' in df.columns and 'low' in df.columns and 'atr' in df.columns:
            high_prev = df['high'].shift(1)
            low_prev = df['low'].shift(1)
            up_move = df['high'] - high_prev
            down_move = low_prev - df['low']
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
            minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
            tr_s = df['atr'].ewm(alpha=1/14, adjust=False).mean() + 1e-10
            plus_di = 100 * (plus_dm_s / tr_s)
            minus_di = 100 * (minus_dm_s / tr_s)
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
            df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(20)
        else:
            df['adx'] = 20

        # 5. EMA Віяло (EMA9 / EMA21 / EMA50)
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        return df

    def get_trend(self, df, span_val=50):
        if df.empty or 'close' not in df.columns or len(df) < span_val:
            return "NEUTRAL"
        ema = df['close'].ewm(span=span_val, adjust=False).mean()
        current_price = df['close'].iloc[-2] if len(df) >= 2 else df['close'].iloc[-1]
        current_ema = ema.iloc[-2] if len(df) >= 2 else ema.iloc[-1]
        threshold = current_ema * 0.0003
        
        if current_price - current_ema > threshold:
            return "BULLISH"
        elif current_price - current_ema < -threshold:
            return "BEARISH"
        return "NEUTRAL"

    def calculate_pivots(self, df_daily):
        if df_daily.empty or 'high' not in df_daily.columns or len(df_daily) < 2:
            return {"P": 0, "R1": 0, "S1": 0, "R2": 0, "S2": 0}
        last_day = df_daily.iloc[-2] if len(df_daily) >= 2 else df_daily.iloc[-1]
        high, low, close = float(last_day['high']), float(last_day['low']), float(last_day['close'])
        p = (high + low + close) / 3
        return {"P": p, "R1": (2 * p) - low, "S1": (2 * p) - high, "R2": p + (high - low), "S2": p - (high - low)}

    def detect_divergence(self, df, window=30):
        if df.empty or 'rsi' not in df.columns or 'close' not in df.columns or len(df) < window:
            return "NONE"
        sub = df.tail(window).copy().reset_index(drop=True)
        prices, rsi_vals = sub['close'].values, sub['rsi'].values
        
        low_pivots = [i for i in range(2, len(prices) - 2) if prices[i] <= prices[i-1] and prices[i] <= prices[i-2] and prices[i] <= prices[i+1] and prices[i] <= prices[i+2]]
        if len(low_pivots) >= 2:
            p1, p2 = low_pivots[-2], low_pivots[-1]
            if prices[p2] < prices[p1] and rsi_vals[p2] > rsi_vals[p1] + 1.5:
                return "BULLISH_DIV"

        high_pivots = [i for i in range(2, len(prices) - 2) if prices[i] >= prices[i-1] and prices[i] >= prices[i-2] and prices[i] >= prices[i+1] and prices[i] >= prices[i+2]]
        if len(high_pivots) >= 2:
            p1, p2 = high_pivots[-2], high_pivots[-1]
            if prices[p2] > prices[p1] and rsi_vals[p2] < rsi_vals[p1] - 1.5:
                return "BEARISH_DIV"

        return "NONE"

    def generate_signal(self, df_1m, df_5m, global_trend="NEUTRAL", mid_trend="NEUTRAL", df_daily=None):
        if isinstance(global_trend, pd.DataFrame):
            df_macro = global_trend
            global_trend = self.get_trend(df_macro, span_val=200)
            mid_trend = self.get_trend(df_5m, span_val=50)
            if df_daily is None:
                df_daily = df_macro

        df_1m = self.calculate_indicators(df_1m)
        df_5m = self.calculate_indicators(df_5m)

        if df_5m.empty or len(df_5m) < 20 or df_1m.empty or len(df_1m) < 20:
            return {'signal': 'HOLD', 'reason': 'Мало даних', 'priority': 4, 'strategy': 'HOLD', 'suggested_exp': 5}

        last_5m = df_5m.iloc[-2] if len(df_5m) >= 2 else df_5m.iloc[-1]
        last_1m = df_1m.iloc[-2] if len(df_1m) >= 2 else df_1m.iloc[-1]

        rsi_5m, rsi_1m = float(last_5m.get('rsi', 50)), float(last_1m.get('rsi', 50))
        adx, atr = float(last_5m.get('adx', 20)), float(last_5m.get('atr', 0.001))
        pct_b_1m = float(last_1m.get('pct_b', 0.5))
        close_5m, close_1m = float(last_5m['close']), float(last_1m['close'])
        ema_9, ema_21 = float(last_5m.get('ema_9', close_5m)), float(last_5m.get('ema_21', close_5m))
        
        atr_ma = df_5m['atr'].rolling(20).mean().iloc[-2] if len(df_5m) >= 20 else atr
        volatility_ratio = float(atr / (atr_ma + 1e-10))
        div_5m = self.detect_divergence(df_5m, window=30)
        
        pivots = self.calculate_pivots(df_daily) if df_daily is not None else {}
        r1, s1 = pivots.get('R1', 0), pivots.get('S1', 0)
        r2, s2 = pivots.get('R2', 0), pivots.get('S2', 0)

        # Визначення стану ринку згідно з вашою матрицею
        signal = 'HOLD'
        reason = ''
        strategy_name = 'HOLD'
        priority = 4
        suggested_exp = 5

        # -------------------------------------------------------------
        # ПРІОРИТЕТ 1: Трендовий Імпульс (Hard Veto проти відбоїв)
        # -------------------------------------------------------------
        if adx >= 25 and volatility_ratio >= 1.15 and ema_9 > ema_21:
            signal = 'CALL'
            strategy_name = 'Трендовий Імпульс (BUY)'
            priority = 1
            suggested_exp = 7  # 3-10 хвилин
            reason = f"Вибух волатильності та тренд (ADX: {adx:.1f}, Vol_Ratio: {volatility_ratio:.2f})"
        elif adx >= 25 and volatility_ratio >= 1.15 and ema_9 < ema_21:
            signal = 'PUT'
            strategy_name = 'Трендовий Імпульс (SELL)'
            priority = 1
            suggested_exp = 7
            reason = f"Вибух волатильності та тренд (ADX: {adx:.1f}, Vol_Ratio: {volatility_ratio:.2f})"

        # -------------------------------------------------------------
        # ПРІОРИТЕТ 2: HTF Макро-Тренд / Дивергенції (15-30 хв)
        # -------------------------------------------------------------
        elif div_5m == 'BULLISH_DIV' or global_trend == 'BULLISH' and (s2 > 0 and close_5m <= s2 * 1.002):
            signal = 'CALL'
            strategy_name = 'HTF Макро-Тренд / Розворот'
            priority = 2
            suggested_exp = 20  # 15, 20 або 30 хв
            reason = f"Макро-сигнал / Дивергенція біля S2/R2"
        elif div_5m == 'BEARISH_DIV' or global_trend == 'BEARISH' and (r2 > 0 and close_5m >= r2 * 0.998):
            signal = 'PUT'
            strategy_name = 'HTF Макро-Тренд / Розворот'
            priority = 2
            suggested_exp = 20
            reason = f"Макро-сигнал / Дивергенція біля S2/R2"

        # -------------------------------------------------------------
        # ПРІОРИТЕТ 3: M5 Конфлюентність (Відбиття від рівнів, 3-7 хв)
        # -------------------------------------------------------------
        elif adx < 25 and ((s1 > 0 and close_5m <= s1 * 1.002) or rsi_5m <= 40):
            signal = 'CALL'
            strategy_name = 'M5 Конфлюентність'
            priority = 3
            suggested_exp = 5  # 3-7 хвилин
            reason = f"Відбиття від рівня S1 / RSI перепроданість ({rsi_5m:.1f})"
        elif adx < 25 and ((r1 > 0 and close_5m >= r1 * 0.998) or rsi_5m >= 60):
            signal = 'PUT'
            strategy_name = 'M5 Конфлюентність'
            priority = 3
            suggested_exp = 5
            reason = f"Відбиття від рівня R1 / RSI перекупленість ({rsi_5m:.1f})"

        # -------------------------------------------------------------
        # ПРІОРИТЕТ 4: M1 Скальпінг (Мікрофлет / Торкання меж, 1-3 хв)
        # -------------------------------------------------------------
        elif pct_b_1m <= 0.15 and rsi_1m <= 32:
            signal = 'CALL'
            strategy_name = 'M1 Скальпінг (Відбиття меж)'
            priority = 4
            suggested_exp = 2  # 1-3 хвилини
            reason = f"Торкання нижньої межі %B ({pct_b_1m:.2f}) та RSI M1 ({rsi_1m:.1f})"
        elif pct_b_1m >= 0.85 and rsi_1m >= 68:
            signal = 'PUT'
            strategy_name = 'M1 Скальпінг (Відбиття меж)'
            priority = 4
            suggested_exp = 2
            reason = f"Торкання верхньої межі %B ({pct_b_1m:.2f}) та RSI M1 ({rsi_1m:.1f})"

        return {
            'signal': signal,
            'rsi': round(rsi_5m, 1),
            'adx': round(adx, 1),
            'atr': atr,
            'divergence': div_5m,
            'suggested_exp': suggested_exp,
            'expiration_minutes': suggested_exp,
            'strategy': strategy_name,
            'priority': priority,
            'reason': reason if reason else 'Очікування ринкових умов',
            'volatility_ratio': round(volatility_ratio, 3),
            'wick_ratio': 0.0,
            'ema_dist': 0.0
        }
