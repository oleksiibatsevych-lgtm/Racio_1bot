import pandas as pd
import numpy as np
from datetime import datetime

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df):
        if df.empty or len(df) < 20:
            return df

        df = df.copy()

        # 1. RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)

        # 2. Bollinger Bands + %B Oscillator
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_middle'] = sma
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / (sma + 1e-10)).fillna(0.001)
        df['bb_width_ma'] = df['bb_width'].rolling(window=20).mean().fillna(df['bb_width'])
        
        # Відносне положення ціни в каналі (%B) від 0.0 до 1.0
        df['percent_b'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)).fillna(0.5)

        # 3. ATR
        high = df['high']
        low = df['low']
        close_prev = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean().fillna(0.0010)

        # 4. ADX
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

        # 5. EMA
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # 6. Volume
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean().fillna(df['volume'])
        
        return df

    def get_trend(self, df, span_val=50):
        if df.empty or len(df) < span_val:
            return "NEUTRAL"
        
        ema = df['close'].ewm(span=span_val, adjust=False).mean()
        current_price = df['close'].iloc[-2] if len(df) >= 2 else df['close'].iloc[-1]
        current_ema = ema.iloc[-2] if len(df) >= 2 else ema.iloc[-1]
        
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
        if df_daily.empty or len(df_daily) < 2:
            return {"P": 0, "R1": 0, "S1": 0, "R2": 0, "S2": 0}
            
        last_day = df_daily.iloc[-2] if len(df_daily) >= 2 else df_daily.iloc[-1]
        high, low, close = float(last_day['high']), float(last_day['low']), float(last_day['close'])
        
        p = (high + low + close) / 3
        return {"P": p, "R1": (2 * p) - low, "S1": (2 * p) - high, "R2": p + (high - low), "S2": p - (high - low)}

    def detect_divergence(self, df, window=35):
        if df.empty or 'rsi' not in df.columns or len(df) < window:
            return "NONE"
            
        sub = df.tail(window).copy().reset_index(drop=True)
        prices = sub['close'].values
        rsi_vals = sub['rsi'].values
        
        low_pivots = []
        for i in range(2, len(prices) - 2):
            if prices[i] <= prices[i-1] and prices[i] <= prices[i-2] and prices[i] <= prices[i+1] and prices[i] <= prices[i+2]:
                low_pivots.append((i, prices[i], rsi_vals[i]))
                
        if len(low_pivots) >= 2:
            p1, p2 = low_pivots[-2], low_pivots[-1]
            if p2[1] < p1[1] and p2[2] > p1[2] + 1.5:
                return "BULLISH_DIV"

        high_pivots = []
        for i in range(2, len(prices) - 2):
            if prices[i] >= prices[i-1] and prices[i] >= prices[i-2] and prices[i] >= prices[i+1] and prices[i] >= prices[i+2]:
                high_pivots.append((i, prices[i], rsi_vals[i]))
                
        if len(high_pivots) >= 2:
            p1, p2 = high_pivots[-2], high_pivots[-1]
            if p2[1] > p1[1] and p2[2] < p1[2] - 1.5:
                return "BEARISH_DIV"

        return "NONE"

    def get_wick_ratio(self, candle, direction):
        high, low = float(candle['high']), float(candle['low'])
        open_p, close_p = float(candle['open']), float(candle['close'])
        total_range = high - low
        if total_range == 0:
            return 0.0
            
        if direction == 'CALL':
            lower_wick = min(open_p, close_p) - low
            return lower_wick / total_range
        else:
            upper_wick = high - max(open_p, close_p)
            return upper_wick / total_range

    def calculate_dynamic_expiration(self, df_5m, df_1m, signal, adx, atr, rsi, divergence="NONE", trend_aligned=False, is_pivot_rejection=False):
        try:
            now_utc = datetime.utcnow()
            minute = now_utc.minute
            
            mins_to_5m = 5 - (minute % 5)
            mins_to_15m = 15 - (minute % 15)
            
            if mins_to_5m < 2: mins_to_5m += 5
            if mins_to_15m < 3: mins_to_15m += 15

            use_15m_clock = (divergence != "NONE" or (trend_aligned and adx >= 25) or is_pivot_rejection)
            expiration = mins_to_15m if use_15m_clock else mins_to_5m
            return int(np.clip(expiration, 2, 15))
        except Exception:
            return 5

    def generate_signal(self, df_1m, df_5m, global_trend, mid_trend, df_daily=None):
        if df_5m.empty or len(df_5m) < 20 or df_1m.empty or len(df_1m) < 20:
            return {
                'signal': 'HOLD', 'reason': 'Мало даних', 'suggested_exp': 5, 
                'rsi': 50, 'adx': 20, 'atr': 0.001, 'volatility_ratio': 1.0, 
                'wick_ratio': 0.0, 'ema_dist': 0.0
            }

        # 1. Конфлікт трендів 1h та 15m
        if global_trend != mid_trend and global_trend != 'NEUTRAL' and mid_trend != 'NEUTRAL':
            return {
                'signal': 'HOLD', 'reason': 'Конфлікт трендів між 1h та 15m', 'suggested_exp': 5,
                'rsi': 50, 'adx': 20, 'atr': 0.001, 'volatility_ratio': 1.0,
                'wick_ratio': 0.0, 'ema_dist': 0.0
            }

        last_5m = df_5m.iloc[-2] if len(df_5m) >= 2 else df_5m.iloc[-1]
        last_1m = df_1m.iloc[-2] if len(df_1m) >= 2 else df_1m.iloc[-1]
        
        rsi_5m, rsi_1m = float(last_5m.get('rsi', 50)), float(last_1m.get('rsi', 50))
        adx, atr = float(last_5m.get('adx', 20)), float(last_5m.get('atr', 0.001))
        close_5m, close_1m = float(last_5m['close']), float(last_1m['close'])
        pct_b_1m = float(last_1m.get('percent_b', 0.5))
        pct_b_5m = float(last_5m.get('percent_b', 0.5))
        ema_50 = float(last_5m.get('ema_50', close_5m))
        
        atr_ma = df_5m['atr'].rolling(20).mean().iloc[-2] if len(df_5m) >= 20 else atr
        volatility_ratio = float(atr / (atr_ma + 1e-10))

        if volatility_ratio < 0.6 or volatility_ratio > 2.5:
            return {
                'signal': 'HOLD', 'reason': 'Аномальна волатильність', 'suggested_exp': 5,
                'rsi': round(rsi_5m, 1), 'adx': round(adx, 1), 'atr': atr, 'volatility_ratio': round(volatility_ratio, 3),
                'wick_ratio': 0.0, 'ema_dist': 0.0
            }

        vol_1m = float(last_1m.get('volume', 0))
        vol_ma_1m = float(df_1m['volume_ma'].iloc[-2]) if 'volume_ma' in df_1m.columns and len(df_1m) >= 2 else 1.0
        volume_ratio = vol_1m / (vol_ma_1m + 1e-10) if vol_ma_1m > 0 else 1.0

        ema_dist = float((close_5m - ema_50) / (ema_50 + 1e-10))
        div = self.detect_divergence(df_5m, window=35)

        pivots = self.calculate_pivots(df_daily) if df_daily is not None and not df_daily.empty else {}
        r1, s1 = pivots.get('R1', 0), pivots.get('S1', 0)

        effective_trend = global_trend if global_trend != 'NEUTRAL' else mid_trend
        call_score, put_score = 0, 0
        call_reasons, put_reasons = [], []
        is_pivot_rejection = False

        # ==========================================
        # 🟢 ОЦІНКА УМОВ ДЛЯ КУПІВЛІ (CALL)
        # ==========================================
        # Точний відбій від нижньої межі BB (%B <= 0.20)
        if pct_b_1m <= 0.20 or pct_b_5m <= 0.20:
            call_score += 30
            call_reasons.append("Відбій від BB Lower (%B <= 0.20)")

        if s1 > 0 and close_5m <= s1 * 1.0015:
            call_score += 20
            call_reasons.append("Підтримка Pivot S1")
            is_pivot_rejection = True

        if div == 'BULLISH_DIV':
            call_score += 35
            call_reasons.append("🎯 Бичача дивергенція")

        if rsi_1m < 38 and rsi_5m < 48:
            call_score += 20
            call_reasons.append(f"Перепроданість RSI ({rsi_1m:.1f})")

        call_wick = self.get_wick_ratio(last_1m, 'CALL')
        if call_wick >= 0.40:
            call_score += 25 if volume_ratio >= 1.2 else 15
            call_reasons.append(f"Ґніт CALL ({round(call_wick*100)}%)")

        if effective_trend == 'BULLISH':
            call_score += 15
            call_reasons.append("За трендом B")

        # ==========================================
        # 🔴 ОЦІНКА УМОВ ДЛЯ ПРОДАЖУ (PUT)
        # ==========================================
        # Точний відбій від верхньої межі BB (%B >= 0.80)
        if pct_b_1m >= 0.80 or pct_b_5m >= 0.80:
            put_score += 30
            put_reasons.append("Відбій від BB Upper (%B >= 0.80)")

        if r1 > 0 and close_5m >= r1 * 0.9985:
            put_score += 20
            put_reasons.append("Опір Pivot R1")
            is_pivot_rejection = True

        if div == 'BEARISH_DIV':
            put_score += 35
            put_reasons.append("🎯 Ведмежа дивергенція")

        if rsi_1m > 62 and rsi_5m > 52:
            put_score += 20
            put_reasons.append(f"Перекупленість RSI ({rsi_1m:.1f})")

        put_wick = self.get_wick_ratio(last_1m, 'PUT')
        if put_wick >= 0.40:
            put_score += 25 if volume_ratio >= 1.2 else 15
            put_reasons.append(f"Ґніт PUT ({round(put_wick*100)}%)")

        if effective_trend == 'BEARISH':
            put_score += 15
            put_reasons.append("За трендом S")

        # ==========================================
        # 🛡 СУВОРІ ФІЛЬТРИ ВЗАЄМОВИКЛЮЧЕННЯ (HARD VETO)
        # ==========================================
        # CALL БЛОКУЄТЬСЯ, якщо ціна у верхній половині каналу (%B > 0.50) або RSI > 50
        if pct_b_1m > 0.50 or rsi_5m >= 52 or rsi_1m >= 55:
            call_score = 0
            call_reasons = []

        # PUT БЛОКУЄТЬСЯ, якщо ціна у нижній половині каналу (%B < 0.50) або RSI < 50
        if pct_b_1m < 0.50 or rsi_5m <= 48 or rsi_1m <= 45:
            put_score = 0
            put_reasons = []

        threshold = 60
        signal = 'HOLD'
        reason = 'Умови не виконано / Блокування суперечностей індикаторів'
        wick_ratio = 0.0

        if call_score >= threshold and call_score > put_score:
            signal = 'CALL'
            reason = " + ".join(call_reasons)
            wick_ratio = call_wick
        elif put_score >= threshold and put_score > call_score:
            signal = 'PUT'
            reason = " + ".join(put_reasons)
            wick_ratio = put_wick

        suggested_exp = self.calculate_dynamic_expiration(
            df_5m, df_1m, signal, adx, atr, rsi_5m, 
            divergence=div, trend_aligned=(effective_trend == global_trend),
            is_pivot_rejection=is_pivot_rejection
        )

        return {
            'signal': signal, 'rsi': round(rsi_5m, 1), 'adx': round(adx, 1), 'atr': atr,
            'divergence': div, 'suggested_exp': suggested_exp, 'reason': reason,
            'volatility_ratio': round(volatility_ratio, 3), 'wick_ratio': round(wick_ratio, 3),
            'ema_dist': round(ema_dist, 5)
        }
