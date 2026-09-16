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

        # 2. Bollinger Bands
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_middle'] = sma
        df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / (sma + 1e-10)).fillna(0.001)
        df['bb_width_ma'] = df['bb_width'].rolling(window=20).mean().fillna(df['bb_width'])

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

        # 6. Volume MA
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

        # 1. Жорстке блокування при прямому конфлікті трендів 1h та 15m
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
        bb_upper, bb_lower = float(last_5m.get('bb_upper', 0)), float(last_5m.get('bb_lower', 0))
        close_5m, close_1m = float(last_5m['close']), float(last_1m['close'])
        ema_50 = float(last_5m.get('ema_50', close_5m))
        
        # Волатильність та об'єм
        atr_ma = df_5m['atr'].rolling(20).mean().iloc[-2] if len(df_5m) >= 20 else atr
        volatility_ratio = float(atr / (atr_ma + 1e-10))

        # 2. Фільтр критичної волатильності (мертвий флет або новинний шторм)
        if volatility_ratio < 0.6:
            return {
                'signal': 'HOLD', 'reason': 'Аномально низька волатильність (мертвий ринок)', 'suggested_exp': 5,
                'rsi': round(rsi_5m, 1), 'adx': round(adx, 1), 'atr': atr, 'volatility_ratio': round(volatility_ratio, 3),
                'wick_ratio': 0.0, 'ema_dist': 0.0
            }
        elif volatility_ratio > 2.5:
            return {
                'signal': 'HOLD', 'reason': 'Аномально висока волатильність (новинний сплеск)', 'suggested_exp': 5,
                'rsi': round(rsi_5m, 1), 'adx': round(adx, 1), 'atr': atr, 'volatility_ratio': round(volatility_ratio, 3),
                'wick_ratio': 0.0, 'ema_dist': 0.0
            }

        vol_1m = float(last_1m.get('volume', 0))
        vol_ma_1m = float(df_1m['volume_ma'].iloc[-2]) if 'volume_ma' in df_1m.columns and len(df_1m) >= 2 else 1.0
        volume_ratio = vol_1m / (vol_ma_1m + 1e-10) if vol_ma_1m > 0 else 1.0

        # Перевірка розширення Bollinger Bands
        bb_w_5m = float(last_5m.get('bb_width', 0.001))
        bb_w_ma = float(df_5m['bb_width_ma'].iloc[-2]) if 'bb_width_ma' in df_5m.columns and len(df_5m) >= 2 else bb_w_5m
        is_bb_expanding = (bb_w_5m > bb_w_ma * 1.25) and (adx > 25)

        ema_dist = float((close_5m - ema_50) / (ema_50 + 1e-10))
        div = self.detect_divergence(df_5m, window=35)

        pivots = self.calculate_pivots(df_daily) if df_daily is not None and not df_daily.empty else {}
        r1, s1 = pivots.get('R1', 0), pivots.get('S1', 0)

        effective_trend = global_trend if global_trend != 'NEUTRAL' else mid_trend
        call_score, put_score = 0, 0
        call_reasons, put_reasons = [], []
        is_pivot_rejection = False

        # 3. Bollinger Bands відскок з захистом від розширення
        if bb_lower > 0 and close_1m <= bb_lower * 1.001:
            if not is_bb_expanding:
                call_score += 25
                call_reasons.append("Відбій від BB Lower")
            else:
                call_reasons.append("⚠️ BB розширюється (ігнор відбою)")

        if bb_upper > 0 and close_1m >= bb_upper * 0.999:
            if not is_bb_expanding:
                put_score += 25
                put_reasons.append("Відбій від BB Upper")
            else:
                put_reasons.append("⚠️ BB розширюється (ігнор відбою)")

        # Pivot підтримка/опір
        if s1 > 0 and close_5m <= s1 * 1.002:
            call_score += 20
            call_reasons.append("Підтримка Pivot S1")
            is_pivot_rejection = True

        if r1 > 0 and close_5m >= r1 * 0.998:
            put_score += 20
            put_reasons.append("Опір Pivot R1")
            is_pivot_rejection = True

        # Дивергенція
        if div == 'BULLISH_DIV':
            call_score += 35
            call_reasons.append("🎯 Бичача дивергенція (+35)")
        elif div == 'BEARISH_DIV':
            put_score += 35
            put_reasons.append("🎯 Ведмежа дивергенція (+35)")

        # 4. Рівні RSI та Жорсткі обмеження для тренду
        # CALL заборонений при перекупленості (RSI 5m >= 65)
        if (rsi_1m < 35 or rsi_5m < 40) and rsi_5m < 65:
            call_score += 20
            call_reasons.append(f"Перепроданість RSI ({rsi_1m:.1f})")

        # PUT заборонений при перепроданості (RSI 5m <= 35)
        if (rsi_1m > 65 or rsi_5m > 60) and rsi_5m > 35:
            put_score += 20
            put_reasons.append(f"Перекупленість RSI ({rsi_1m:.1f})")

        # 5. Ґніт відскоку + Підтвердження об'ємом (Volume confirmation)
        call_wick = self.get_wick_ratio(last_1m, 'CALL')
        put_wick = self.get_wick_ratio(last_1m, 'PUT')

        if call_wick >= 0.40:
            pts = 15
            if volume_ratio >= 1.2:
                pts += 10
                call_reasons.append(f"Ґніт відскоку CALL ({round(call_wick*100)}%) + Vol x{round(volume_ratio,1)}")
            else:
                call_reasons.append(f"Ґніт відскоку CALL ({round(call_wick*100)}%)")
            call_score += pts

        if put_wick >= 0.40:
            pts = 15
            if volume_ratio >= 1.2:
                pts += 10
                put_reasons.append(f"Ґніт відскоку PUT ({round(put_wick*100)}%) + Vol x{round(volume_ratio,1)}")
            else:
                put_reasons.append(f"Ґніт відскоку PUT ({round(put_wick*100)}%)")
            put_score += pts

        # 6. Врахування тренду та штраф за контртренд
        if effective_trend == 'BULLISH':
            call_score += 15
            call_reasons.append("За трендом B")
            if div != 'BEARISH_DIV' and not is_pivot_rejection:
                put_score -= 30  # Жорсткий штраф проти сильного тренду
        elif effective_trend == 'BEARISH':
            put_score += 15
            put_reasons.append("За трендом S")
            if div != 'BULLISH_DIV' and not is_pivot_rejection:
                call_score -= 30  # Жорсткий штраф проти сильного тренду

        threshold_call = 50 if div == 'BULLISH_DIV' else 60
        threshold_put = 50 if div == 'BEARISH_DIV' else 60

        if adx > 35:
            if div == 'BULLISH_DIV' and global_trend == 'BEARISH': call_score -= 20
            elif div == 'BEARISH_DIV' and global_trend == 'BULLISH': put_score -= 20

        signal = 'HOLD'
        reason = 'Недостатньо конфлюентності'
        wick_ratio = 0.0

        if call_score >= threshold_call and call_score > put_score:
            signal = 'CALL'
            reason = " + ".join(call_reasons)
            wick_ratio = call_wick
        elif put_score >= threshold_put and put_score > call_score:
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
