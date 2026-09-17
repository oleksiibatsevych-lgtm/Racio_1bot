import pandas as pd
import numpy as np
from datetime import datetime

class AdaptiveTechnicalAnalysis:
    
    def get_rma(self, series, period):
        """Згладжування Вайлдера (Running Moving Average) для точного ATR та ADX"""
        return series.ewm(alpha=1/period, adjust=False).mean()

    def calculate_indicators(self, df, rsi_period=14, bb_period=20, ema_periods=(9, 21, 50)):
        if df is None or df.empty or len(df) < max(bb_period, max(ema_periods)) + 10:
            return df

        df = df.copy()
        df.columns = [str(col).lower() for col in df.columns]

        if 'close' not in df.columns:
            return df

        # 1. RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = self.get_rma(gain, rsi_period)
        avg_loss = self.get_rma(loss, rsi_period)
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        # 2. Bollinger Bands & %B
        sma = df['close'].rolling(window=bb_period).mean()
        std = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_middle'] = sma
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / (sma + 1e-10)).fillna(0.001)
        bb_range = df['bb_upper'] - df['bb_lower']
        df['pct_b'] = np.where(bb_range > 0, (df['close'] - df['bb_lower']) / bb_range, 0.5)

        # 3. ATR (за методом Вайлдера)
        if 'high' in df.columns and 'low' in df.columns:
            high, low, close_prev = df['high'], df['low'], df['close'].shift(1)
            tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
            df['atr'] = self.get_rma(tr, 14).fillna(0.0010)
        else:
            df['atr'] = 0.0010

        # 4. ADX (за методом Вайлдера)
        if 'high' in df.columns and 'low' in df.columns and 'atr' in df.columns:
            up_move = df['high'] - df['high'].shift(1)
            down_move = df['low'].shift(1) - df['low']
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
            
            plus_di = 100 * (self.get_rma(pd.Series(plus_dm, index=df.index), 14) / (df['atr'] + 1e-10))
            minus_di = 100 * (self.get_rma(pd.Series(minus_dm, index=df.index), 14) / (df['atr'] + 1e-10))
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
            df['adx'] = self.get_rma(dx, 14).fillna(20)
        else:
            df['adx'] = 20

        # 5. EMA Віяло
        for span in ema_periods:
            df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
            
        # Обчислення відстані до ключової EMA
        ema_main = f'ema_{ema_periods[-1]}'
        df['ema_dist'] = ((df['close'] - df[ema_main]) / (df[ema_main] + 1e-10)) * 100

        # 6. MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 7. Об'єм (VWAP - зважена за об'ємом ціна)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap_20'] = (typical_price * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
        else:
            df['vwap_20'] = df['close']

        # 8. Wick Ratio (Аналіз тіней свічки)
        if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
            candle_range = df['high'] - df['low'] + 1e-10
            df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
            df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range
        else:
            df['upper_wick_ratio'] = 0.0
            df['lower_wick_ratio'] = 0.0
            
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
        if df.empty or 'rsi' not in df.columns or 'macd' not in df.columns or len(df) < window:
            return "NONE"
        sub = df.tail(window).copy().reset_index(drop=True)
        prices, rsi_vals, macd_vals = sub['close'].values, sub['rsi'].values, sub['macd'].values
        
        low_pivots = [i for i in range(2, len(prices) - 2) if prices[i] <= prices[i-1] and prices[i] <= prices[i-2] and prices[i] <= prices[i+1] and prices[i] <= prices[i+2]]
        high_pivots = [i for i in range(2, len(prices) - 2) if prices[i] >= prices[i-1] and prices[i] >= prices[i-2] and prices[i] >= prices[i+1] and prices[i] >= prices[i+2]]
        
        if len(low_pivots) >= 2:
            p1, p2 = low_pivots[-2], low_pivots[-1]
            if prices[p2] < prices[p1] and (rsi_vals[p2] > rsi_vals[p1] + 1.5 or macd_vals[p2] > macd_vals[p1]):
                return "BULLISH_DIV"
            if prices[p2] > prices[p1] and rsi_vals[p2] < rsi_vals[p1] - 1.5:
                return "HIDDEN_BULLISH_DIV"

        if len(high_pivots) >= 2:
            p1, p2 = high_pivots[-2], high_pivots[-1]
            if prices[p2] > prices[p1] and (rsi_vals[p2] < rsi_vals[p1] - 1.5 or macd_vals[p2] < macd_vals[p1]):
                return "BEARISH_DIV"
            if prices[p2] < prices[p1] and rsi_vals[p2] > rsi_vals[p1] + 1.5:
                return "HIDDEN_BEARISH_DIV"

        return "NONE"

    def generate_signal(self, df_1m, df_5m, global_trend="NEUTRAL", mid_trend="NEUTRAL", df_daily=None):
        if isinstance(global_trend, pd.DataFrame):
            df_macro = global_trend
            global_trend = self.get_trend(df_macro, span_val=200)
            mid_trend = self.get_trend(df_5m, span_val=50)
            if df_daily is None:
                df_daily = df_macro

        df_1m = self.calculate_indicators(df_1m, rsi_period=9)
        df_5m = self.calculate_indicators(df_5m, rsi_period=14)

        if df_5m.empty or len(df_5m) < 20 or df_1m.empty or len(df_1m) < 20:
            return {'signal': 'HOLD', 'reason': 'Мало даних', 'priority': 4, 'strategy': 'HOLD', 'suggested_exp': 5}

        last_5m = df_5m.iloc[-2] if len(df_5m) >= 2 else df_5m.iloc[-1]
        last_1m = df_1m.iloc[-2] if len(df_1m) >= 2 else df_1m.iloc[-1]

        rsi_5m, rsi_1m = float(last_5m.get('rsi', 50)), float(last_1m.get('rsi', 50))
        adx, atr = float(last_5m.get('adx', 20)), float(last_5m.get('atr', 0.001))
        pct_b_1m = float(last_1m.get('pct_b', 0.5))
        close_5m, close_1m = float(last_5m['close']), float(last_1m['close'])
        
        ema_9, ema_21 = float(last_5m.get('ema_9', close_5m)), float(last_5m.get('ema_21', close_5m))
        ema_dist_val = float(last_5m.get('ema_dist', 0.0))
        vwap_20 = float(last_5m.get('vwap_20', close_5m))
        macd_hist = float(last_5m.get('macd_hist', 0.0))
        
        atr_ma = df_5m['atr'].rolling(20).mean().iloc[-2] if len(df_5m) >= 20 else atr
        volatility_ratio = float(atr / (atr_ma + 1e-10))
        div_5m = self.detect_divergence(df_5m, window=30)
        
        pivots = self.calculate_pivots(df_daily) if df_daily is not None else {}
        r1, s1 = pivots.get('R1', 0), pivots.get('S1', 0)
        r2, s2 = pivots.get('R2', 0), pivots.get('S2', 0)

        signal, reason, strategy_name = 'HOLD', '', 'HOLD'
        priority, suggested_exp = 4, 5
        wick_ratio_final = 0.0

        # ПРІОРИТЕТ 1: Трендовий Імпульс
        if adx >= 25 and volatility_ratio >= 1.15 and ema_9 > ema_21 and close_5m > vwap_20 and macd_hist > 0:
            signal = 'CALL'
            strategy_name = 'Трендовий Імпульс (BUY)'
            priority = 1
            suggested_exp = int(7)
            wick_ratio_final = float(last_5m.get('lower_wick_ratio', 0.0))
            reason = f"Волатильність + Об'єм (ADX: {adx:.1f}, Ціна вище VWAP, MACD+)"
        elif adx >= 25 and volatility_ratio >= 1.15 and ema_9 < ema_21 and close_5m < vwap_20 and macd_hist < 0:
            signal = 'PUT'
            strategy_name = 'Трендовий Імпульс (SELL)'
            priority = 1
            suggested_exp = int(7)
            wick_ratio_final = float(last_5m.get('upper_wick_ratio', 0.0))
            reason = f"Волатильність + Об'єм (ADX: {adx:.1f}, Ціна нижче VWAP, MACD-)"

        # ПРІОРИТЕТ 2: HTF Макро-Тренд / Дивергенції
        elif 'BULLISH_DIV' in div_5m or (global_trend == 'BULLISH' and s2 > 0 and close_5m <= s2 * 1.002):
            signal = 'CALL'
            strategy_name = 'HTF Макро / Розворот'
            priority = 2
            suggested_exp = int(20)
            wick_ratio_final = float(last_5m.get('lower_wick_ratio', 0.0))
            reason = f"Макро-сигнал / Дивергенція ({div_5m})"
        elif 'BEARISH_DIV' in div_5m or (global_trend == 'BEARISH' and r2 > 0 and close_5m >= r2 * 0.998):
            signal = 'PUT'
            strategy_name = 'HTF Макро / Розворот'
            priority = 2
            suggested_exp = int(20)
            wick_ratio_final = float(last_5m.get('upper_wick_ratio', 0.0))
            reason = f"Макро-сигнал / Дивергенція ({div_5m})"

        # ПРІОРИТЕТ 3: M5 Конфлюентність
        elif adx < 25 and ((s1 > 0 and close_5m <= s1 * 1.002) or rsi_5m <= 40):
            signal = 'CALL'
            strategy_name = 'M5 Конфлюентність'
            priority = 3
            suggested_exp = int(5)
            wick_ratio_final = float(last_5m.get('lower_wick_ratio', 0.0))
            reason = f"S1 / RSI перепроданість ({rsi_5m:.1f})"
        elif adx < 25 and ((r1 > 0 and close_5m >= r1 * 0.998) or rsi_5m >= 60):
            signal = 'PUT'
            strategy_name = 'M5 Конфлюентність'
            priority = 3
            suggested_exp = int(5)
            wick_ratio_final = float(last_5m.get('upper_wick_ratio', 0.0))
            reason = f"R1 / RSI перекупленість ({rsi_5m:.1f})"

        # ПРІОРИТЕТ 4: M1 Скальпінг
        elif pct_b_1m <= 0.15 and rsi_1m <= 32:
            signal = 'CALL'
            strategy_name = 'M1 Скальпінг (Відбиття)'
            priority = 4
            suggested_exp = int(2)
            wick_ratio_final = float(last_1m.get('lower_wick_ratio', 0.0))
            reason = f"%B ({pct_b_1m:.2f}) та RSI M1 ({rsi_1m:.1f})"
        elif pct_b_1m >= 0.85 and rsi_1m >= 68:
            signal = 'PUT'
            strategy_name = 'M1 Скальпінг (Відбиття)'
            priority = 4
            suggested_exp = int(2)
            wick_ratio_final = float(last_1m.get('upper_wick_ratio', 0.0))
            reason = f"%B ({pct_b_1m:.2f}) та RSI M1 ({rsi_1m:.1f})"

        # --- ЗАХИСНИЙ ФІЛЬТР ТРЕНДУ (Фільтрація контррендових угод) ---
        if global_trend == 'BULLISH' and signal == 'PUT':
            # Якщо тренд висхідний, а бот хоче продати — скасовуємо або переводимо в HOLD
            signal = 'HOLD'
            reason = f"Фільтр тренду: відхилено PUT проти BULLISH тренду (RSI був {rsi_5m:.1f})"
            strategy_name = 'HOLD (Фільтр тренду)'
        elif global_trend == 'BEARISH' and signal == 'CALL':
            # Якщо тренд спадний, а бот хоче купити — скасовуємо
            signal = 'HOLD'
            reason = f"Фільтр тренду: відхилено CALL проти BEARISH тренду (RSI був {rsi_5m:.1f})"
            strategy_name = 'HOLD (Фільтр тренду)'

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
            'wick_ratio': round(wick_ratio_final, 3),
            'ema_dist': round(ema_dist_val, 3)
        }
