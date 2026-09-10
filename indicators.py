import pandas as pd
import numpy as np

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df):
        if df.empty or len(df) < 20:
            return df

        df = df.copy()

        # 1. RSI з використанням згладжування Уайлдера (RMA)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        # 2. Смуги Боллінджера
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_middle'] = sma
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma + 1e-10)
        df['bb_width'] = df['bb_width'].fillna(0.001)

        # 3. ATR зі згладжуванням Уайлдера
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean().fillna(0.0010)

        # 4. ADX зі згладжуванням Уайлдера
        high_prev = high.shift(1)
        low_prev = low.shift(1)
        
        up_move = high - high_prev
        down_move = low_prev - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        plus_dm_series = pd.Series(plus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
        minus_dm_series = pd.Series(minus_dm, index=df.index).ewm(alpha=1/14, adjust=False).mean()
        tr_smoothed = tr.ewm(alpha=1/14, adjust=False).mean() + 1e-10
        
        plus_di = 100 * (plus_dm_series / tr_smoothed)
        minus_di = 100 * (minus_dm_series / tr_smoothed)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        df['adx'] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(20)

        # 5. Експоненційні ковзні середні
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        return df

    def get_trend(self, df, span_val=50):
        if df.empty or len(df) < span_val:
            return "NEUTRAL"
        
        ema = df['close'].ewm(span=span_val, adjust=False).mean()
        current_price = df['close'].iloc[-2] if len(df) >= 2 else df['close'].iloc[-1]
        current_ema = ema.iloc[-2] if len(df) >= 2 else ema.iloc[-1]
        
        # Адаптивний поріг тренду на основі ATR або динамічного відхилення
        if 'atr' in df.columns and not pd.isna(df['atr'].iloc[-1]):
            atr_val = df['atr'].iloc[-1]
            threshold = max(atr_val * 0.5, current_ema * 0.0003)
        else:
            threshold = current_ema * 0.0005
            
        diff = current_price - current_ema
        if diff > threshold:
            return "BULLISH"
        elif diff < -threshold:
            return "BEARISH"
        return "NEUTRAL"

    def calculate_pivots(self, df_daily):
        """
        Розрахунок рівнів Pivot на основі денних свічок (df_daily).
        """
        if df_daily.empty or len(df_daily) < 2:
            return {"P": 0, "R1": 0, "S1": 0, "R2": 0, "S2": 0}
            
        last_day = df_daily.iloc[-2] if len(df_daily) >= 2 else df_daily.iloc[-1]
        high = float(last_day['high'])
        low = float(last_day['low'])
        close = float(last_day['close'])
        
        p = (high + low + close) / 3
        r1 = (2 * p) - low
        s1 = (2 * p) - high
        r2 = p + (high - low)
        s2 = p - (high - low)
        return {"P": p, "R1": r1, "S1": s1, "R2": r2, "S2": s2}

    def detect_divergence(self, df, window=30):
        """
        Пошук дивергенції на основі екстремумів (Swing High / Swing Low).
        """
        if df.empty or 'rsi' not in df.columns or len(df) < window:
            return "NONE"
            
        sub = df.tail(window).copy().reset_index(drop=True)
        prices = sub['close'].values
        rsi_vals = sub['rsi'].values
        
        # Локальні мінімуми для бичачої дивергенції
        low_pivots = []
        for i in range(2, len(prices) - 2):
            if prices[i] <= prices[i-1] and prices[i] <= prices[i-2] and prices[i] <= prices[i+1] and prices[i] <= prices[i+2]:
                low_pivots.append((i, prices[i], rsi_vals[i]))
                
        if len(low_pivots) >= 2:
            p1, p2 = low_pivots[-2], low_pivots[-1]
            if p2[1] < p1[1] and p2[2] > p1[2] + 1.5:
                return "BULLISH_DIV"

        # Локальні максимуми для ведмежої дивергенції
        high_pivots = []
        for i in range(2, len(prices) - 2):
            if prices[i] >= prices[i-1] and prices[i] >= prices[i-2] and prices[i] >= prices[i+1] and prices[i] >= prices[i+2]:
                high_pivots.append((i, prices[i], rsi_vals[i]))
                
        if len(high_pivots) >= 2:
            p1, p2 = high_pivots[-2], high_pivots[-1]
            if p2[1] > p1[1] and p2[2] < p1[2] - 1.5:
                return "BEARISH_DIV"

        return "NONE"

    def detect_channel_pattern(self, df, window=30):
        if len(df) < window:
            return "UNKNOWN", 0.0
            
        recent = df.tail(window).copy()
        highs = recent['high'].values
        lows = recent['low'].values
        x = np.arange(len(recent))
        
        slope_high, _ = np.polyfit(x, highs, 1)
        slope_low, _ = np.polyfit(x, lows, 1)
        
        avg_price = recent['close'].mean()
        norm_high = (slope_high * window) / (avg_price + 1e-10) * 100
        norm_low = (slope_low * window) / (avg_price + 1e-10) * 100
        
        if abs(norm_high) < 0.2 and abs(norm_low) < 0.2:
            return "HORIZONTAL_FLAT", 0.0
        elif norm_high > 0.15 and norm_low > 0.15:
            return "ASCENDING_CHANNEL", (norm_high + norm_low) / 2
        elif norm_high < -0.15 and norm_low < -0.15:
            return "DESCENDING_CHANNEL", (norm_high + norm_low) / 2
        else:
            return "EXPANDING_OR_WEDGE", (norm_high + norm_low) / 2

    def is_candle_too_wide(self, df, threshold_multiplier=2.3):
        if len(df) < 15:
            return False
        target_candle = df.iloc[-2]
        total_range = target_candle['high'] - target_candle['low']
        recent_ranges = (df['high'] - df['low']).iloc[-16:-2].mean()
        if total_range > (recent_ranges * threshold_multiplier):
            return True
        return False

    def calculate_dynamic_expiration(self, df_5m, df_1m, signal, adx, atr, rsi):
        """
        Розрахунок динамічної експірації на основі швидкості ринку (ATR, ADX, відхилення RSI).
        """
        try:
            if df_5m.empty or len(df_5m) < 20:
                return 5

            close = float(df_5m['close'].iloc[-2]) if len(df_5m) >= 2 else float(df_5m['close'].iloc[-1])
            if close <= 0:
                return 5

            atr_ma = df_5m['atr'].rolling(20).mean().iloc[-2] if len(df_5m) >= 2 else atr
            volatility_ratio = (atr / (atr_ma + 1e-10))

            if adx >= 25:
                # Трендовий режим: експірація пропорційна імпульсу тренду
                base_mins = 5.0
                time_factor = (30.0 / max(adx, 15.0)) * (1.0 / max(volatility_ratio, 0.5))
                calculated_mins = round(base_mins * time_factor)
                expiration = int(np.clip(calculated_mins, 2, 30))
            else:
                # Флетовий режим: експірація залежить від ступеня перекупленості/перепроданості RSI
                rsi_dev = abs(rsi - 50.0)
                base_mins = 4.0 + (15.0 / (rsi_dev + 1.0))
                time_factor = 1.0 / max(volatility_ratio, 0.6)
                calculated_mins = round(base_mins * time_factor)
                expiration = int(np.clip(calculated_mins, 2, 25))

            return expiration
        except Exception as e:
            return 5

    def generate_signal(self, df_1m, df_5m, global_trend, mid_trend, df_daily=None):
        """
        Генерація сигналів по повністю закритих свічках (iloc[-2]) для усунення перемальовування.
        """
        if df_5m.empty or len(df_5m) < 20 or df_1m.empty or len(df_1m) < 20:
            return {'signal': 'HOLD', 'reason': 'Мало даних', 'suggested_exp': 5, 'rsi': 50, 'adx': 20, 'atr': 0.001}

        last_5m = df_5m.iloc[-2] if len(df_5m) >= 2 else df_5m.iloc[-1]
        last_1m = df_1m.iloc[-2] if len(df_1m) >= 2 else df_1m.iloc[-1]
        
        rsi_5m = float(last_5m.get('rsi', 50))
        rsi_1m = float(last_1m.get('rsi', 50))
        adx = float(last_5m.get('adx', 20))
        atr = float(last_5m.get('atr', 0.001))
        bb_upper = float(last_5m.get('bb_upper', 0))
        bb_lower = float(last_5m.get('bb_lower', 0))
        close_5m = float(last_5m['close'])
        close_1m = float(last_1m['close'])
        div = self.detect_divergence(df_5m, window=30)

        pivots = self.calculate_pivots(df_daily) if df_daily is not None and not df_daily.empty else {}
        r1 = pivots.get('R1', 0)
        s1 = pivots.get('S1', 0)

        signal = 'HOLD'
        reason_parts = []

        effective_trend = global_trend if global_trend != 'NEUTRAL' else mid_trend

        if adx < 21:
            if close_1m <= bb_lower or rsi_1m < 35:
                signal = 'CALL'
                reason_parts.append("Флет: відскок знизу")
                if rsi_1m < 35: reason_parts.append(f"RSI 1m ({rsi_1m:.1f})")
            elif close_1m >= bb_upper or rsi_1m > 65:
                signal = 'PUT'
                reason_parts.append("Флет: відскок зверху")
                if rsi_1m > 65: reason_parts.append(f"RSI 1m ({rsi_1m:.1f})")
        else:
            if effective_trend == 'BULLISH':
                if (bb_lower > 0 and close_5m <= bb_lower * 1.003) or (s1 > 0 and close_5m <= s1 * 1.002) or rsi_5m < 55 or div == 'BULLISH_DIV':
                    signal = 'CALL'
                    reason_parts.append(f"Тренд вгору (ADX: {adx:.1f})")
                    if bb_lower > 0 and close_5m <= bb_lower * 1.003: reason_parts.append("Відбій від Bollinger Lower")
                    if s1 > 0 and close_5m <= s1 * 1.002: reason_parts.append("Відбій від Pivot S1")
                    if rsi_5m < 55: reason_parts.append(f"Відкат RSI ({rsi_5m:.1f})")
                    if div == 'BULLISH_DIV': reason_parts.append("Бичача дивергенція")
            elif effective_trend == 'BEARISH':
                if (bb_upper > 0 and close_5m >= bb_upper * 0.997) or (r1 > 0 and close_5m >= r1 * 0.998) or rsi_5m > 45 or div == 'BEARISH_DIV':
                    signal = 'PUT'
                    reason_parts.append(f"Тренд вниз (ADX: {adx:.1f})")
                    if bb_upper > 0 and close_5m >= bb_upper * 0.997: reason_parts.append("Відбій від Bollinger Upper (Опір)")
                    if r1 > 0 and close_5m >= r1 * 0.998: reason_parts.append("Відбій від Pivot R1 (Опір)")
                    if rsi_5m > 45: reason_parts.append(f"Відкат RSI ({rsi_5m:.1f})")
                    if div == 'BEARISH_DIV': reason_parts.append("Ведмежа дивергенція")

        if signal != 'HOLD':
            channel_type, _ = self.detect_channel_pattern(df_5m, window=30)
            is_wide = self.is_candle_too_wide(df_5m, threshold_multiplier=2.3)

            if is_wide:
                if (effective_trend == 'BULLISH' and signal == 'CALL') or \
                   (effective_trend == 'BEARISH' and signal == 'PUT'):
                    reason_parts.append("Імпульсне підтвердження за трендом")
                else:
                    signal = 'HOLD'
                    reason_parts = ["Пропущено: Широка свічка проти тренду"]
            else:
                if channel_type == "ASCENDING_CHANNEL" and signal == 'PUT':
                    if effective_trend != 'BEARISH':
                        signal = 'HOLD'
                        reason_parts = ["Пропущено: Продаж проти висхідного каналу"]
                elif channel_type == "DESCENDING_CHANNEL" and signal == 'CALL':
                    if effective_trend != 'BULLISH':
                        signal = 'HOLD'
                        reason_parts = ["Пропущено: Купівля проти низхідного каналу"]

        suggested_exp = self.calculate_dynamic_expiration(df_5m, df_1m, signal, adx, atr, rsi_5m)

        reason = " + ".join(reason_parts) if reason_parts else "Умови не виконано"
        return {
            'signal': signal,
            'rsi': round(rsi_5m, 1),
            'adx': round(adx, 1),
            'atr': atr,
            'divergence': div,
            'suggested_exp': suggested_exp,
            'reason': reason
        }
