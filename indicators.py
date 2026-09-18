import pandas as pd
import numpy as np
from datetime import datetime

class AdaptiveTechnicalAnalysis:
    
    def get_rma(self, series, period):
        """Згладжування Вайлдера (Running Moving Average) для точного ATR та ADX"""
        return series.ewm(alpha=1/period, adjust=False).mean()

    def calculate_indicators(self, df, rsi_period=14, bb_period=20, ema_periods=(9, 21, 50)):
        """Обчислення технічних індикаторів для довільного таймфрейму"""
        if df is None or df.empty or len(df) < max(bb_period, max(ema_periods)) + 5:
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
            
        ema_main = f'ema_{ema_periods[-1]}'
        df['ema_dist'] = ((df['close'] - df[ema_main]) / (df[ema_main] + 1e-10)) * 100

        # 6. MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 7. Стохастик (Stochastic Oscillator 14, 3, 3)
        if 'high' in df.columns and 'low' in df.columns:
            lowest_low = df['low'].rolling(window=14).min()
            highest_high = df['high'].rolling(window=14).max()
            df['stoch_k'] = ((df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)) * 100
            df['stoch_k'] = df['stoch_k'].fillna(50)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean().fillna(50)
        else:
            df['stoch_k'] = 50
            df['stoch_d'] = 50

        # 8. Об'єм (VWAP)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap_20'] = (typical_price * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)
        else:
            df['vwap_20'] = df['close']

        # 9. Wick Ratio
        if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
            candle_range = df['high'] - df['low'] + 1e-10
            df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / candle_range
            df['lower_wick_ratio'] = (df[['open', 'close']].min(axis=1) - df['low']) / candle_range
        else:
            df['upper_wick_ratio'] = 0.0
            df['lower_wick_ratio'] = 0.0
            
        return df

    def get_trend(self, df, span_val=50):
        """Визначення напрямку тренду за заданим періодом EMA"""
        if df is None or df.empty or 'close' not in df.columns or len(df) < span_val:
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
        """Розрахунок рівнів Півот на основі денних даних"""
        if df_daily is None or df_daily.empty or 'high' not in df_daily.columns or len(df_daily) < 2:
            return {"P": 0, "R1": 0, "S1": 0, "R2": 0, "S2": 0}
        last_day = df_daily.iloc[-2] if len(df_daily) >= 2 else df_daily.iloc[-1]
        high, low, close = float(last_day['high']), float(last_day['low']), float(last_day['close'])
        p = (high + low + close) / 3
        return {"P": p, "R1": (2 * p) - low, "S1": (2 * p) - high, "R2": p + (high - low), "S2": p - (high - low)}

    def detect_divergence(self, df, window=25):
        """Детекція бичачої чи ведмежої дивергенції"""
        if df is None or df.empty or 'rsi' not in df.columns or 'macd' not in df.columns or len(df) < window:
            return "NONE"
        sub = df.tail(window).copy().reset_index(drop=True)
        prices, rsi_vals, macd_vals = sub['close'].values, sub['rsi'].values, sub['macd'].values
        
        low_pivots = [i for i in range(2, len(prices) - 2) if prices[i] <= prices[i-1] and prices[i] <= prices[i-2] and prices[i] <= prices[i+1] and prices[i] <= prices[i+2]]
        high_pivots = [i for i in range(2, len(prices) - 2) if prices[i] >= prices[i-1] and prices[i] >= prices[i-2] and prices[i] >= prices[i+1] and prices[i] >= prices[i+2]]
        
        if len(low_pivots) >= 2:
            p1, p2 = low_pivots[-2], low_pivots[-1]
            if prices[p2] < prices[p1] and (rsi_vals[p2] > rsi_vals[p1] + 1.5 or macd_vals[p2] > macd_vals[p1]):
                return "BULLISH_DIV"

        if len(high_pivots) >= 2:
            p1, p2 = high_pivots[-2], high_pivots[-1]
            if prices[p2] > prices[p1] and (rsi_vals[p2] < rsi_vals[p1] - 1.5 or macd_vals[p2] < macd_vals[p1]):
                return "BEARISH_DIV"

        return "NONE"

    def score_single_timeframe(self, df_tf, tf_name, global_trend="NEUTRAL", pivots=None):
        """
        Оцінює один таймфрейм із динамічною вагою. 
        У тренді пріоритет за трендовими формаціями, у флеті - за відбиттям від рівнів.
        """
        if df_tf is None or df_tf.empty or len(df_tf) < 20:
            return {'call_score': 0, 'put_score': 0, 'reasons_call': [], 'reasons_put': []}

        df_tf = self.calculate_indicators(df_tf, rsi_period=9 if tf_name == '1m' else 14)
        last = df_tf.iloc[-2] if len(df_tf) >= 2 else df_tf.iloc[-1]

        rsi = float(last.get('rsi', 50))
        stoch_k = float(last.get('stoch_k', 50))
        stoch_d = float(last.get('stoch_d', 50))
        adx = float(last.get('adx', 20))
        atr = float(last.get('atr', 0.001))
        pct_b = float(last.get('pct_b', 0.5))
        close = float(last['close'])
        ema_9 = float(last.get('ema_9', close))
        ema_21 = float(last.get('ema_21', close))
        vwap_20 = float(last.get('vwap_20', close))
        macd_hist = float(last.get('macd_hist', 0.0))
        lower_wick = float(last.get('lower_wick_ratio', 0.0))
        upper_wick = float(last.get('upper_wick_ratio', 0.0))
        
        div = self.detect_divergence(df_tf, window=25)

        call_score = 0
        put_score = 0
        reasons_call = []
        reasons_put = []

        # Визначення фази ринку
        is_trending = adx >= 25
        is_strong_trend = adx >= 35

        # 1. ТРЕНД ТА СТРУКТУРА
        trend_base_weight = 20 if is_trending else 10
        
        if ema_9 > ema_21:
            call_score += trend_base_weight
            reasons_call.append(f"Локальний висхідний тренд EMA ({tf_name})")
        elif ema_9 < ema_21:
            put_score += trend_base_weight
            reasons_put.append(f"Локальний низхідний тренд EMA ({tf_name})")

        if ema_9 > ema_21 and global_trend == "BULLISH":
            call_score += 25
            reasons_call.append("Синхронізація макро та мікро трендів 📈")
        elif ema_9 < ema_21 and global_trend == "BEARISH":
            put_score += 25
            reasons_put.append("Синхронізація макро та мікро трендів 📉")

        if close > vwap_20:
            call_score += 10
            reasons_call.append("Ціна вище VWAP")
        elif close < vwap_20:
            put_score += 10
            reasons_put.append("Ціна нижче VWAP")

        if macd_hist > 0:
            call_score += 10
        elif macd_hist < 0:
            put_score += 10

        # 2. ОСЦИЛЯТОРИ ТА ВІДКАТИ
        osc_weight_deep = 15 if is_trending else 25
        
        if rsi <= 30:
            if not (is_strong_trend and global_trend == "BEARISH"):
                call_score += osc_weight_deep
                reasons_call.append(f"Перепроданість RSI ({rsi:.1f})")
        elif rsi >= 70:
            if not (is_strong_trend and global_trend == "BULLISH"):
                put_score += osc_weight_deep
                reasons_put.append(f"Перекупленість RSI ({rsi:.1f})")

        if stoch_k <= 20 and stoch_k > stoch_d:
            call_score += 15
            reasons_call.append("Висхідний перетин Stochastic")
        elif stoch_k >= 80 and stoch_k < stoch_d:
            put_score += 15
            reasons_put.append("Низхідний перетин Stochastic")

        # 3. ВОЛАТИЛЬНІСТЬ ТА ФЛЕТОВІ МЕЖІ
        bb_weight = 10 if is_trending else 25
        if pct_b <= 0.15:
            if not (is_strong_trend and global_trend == "BEARISH"):
                call_score += bb_weight
                reasons_call.append(f"Відбиття від нижньої межі BB (%B: {pct_b:.2f})")
        elif pct_b >= 0.85:
            if not (is_strong_trend and global_trend == "BULLISH"):
                put_score += bb_weight
                reasons_put.append(f"Відбиття від верхньої межі BB (%B: {pct_b:.2f})")

        if lower_wick >= 0.4:
            call_score += 15
            reasons_call.append(f"Пінбар/Відкупна тінь ({lower_wick:.2f})")
        elif upper_wick >= 0.4:
            put_score += 15
            reasons_put.append(f"Пінбар/Тінь продажів ({upper_wick:.2f})")

        # 4. ДИВЕРГЕНЦІЯ ТА РІВНІ PIVOT
        if div == 'BULLISH_DIV':
            call_score += 30
            reasons_call.append(f"Бича дивергенція на {tf_name}")
        elif div == 'BEARISH_DIV':
            put_score += 30
            reasons_put.append(f"Ведмежа дивергенція на {tf_name}")

        pivot_weight = 10 if is_trending else 20
        if pivots:
            s1, s2 = pivots.get('S1', 0), pivots.get('S2', 0)
            r1, r2 = pivots.get('R1', 0), pivots.get('R2', 0)
            if (s1 > 0 and close <= s1 * 1.002 and close >= s1 * 0.998) or (s2 > 0 and close <= s2 * 1.002 and close >= s2 * 0.998):
                call_score += pivot_weight
                reasons_call.append("Тест підтримки Pivot S1/S2")
            elif (r1 > 0 and close >= r1 * 0.998 and close <= r1 * 1.002) or (r2 > 0 and close >= r2 * 0.998 and close <= r2 * 1.002):
                put_score += pivot_weight
                reasons_put.append("Тест опору Pivot R1/R2")

        return {
            'tf_name': tf_name,
            'call_score': call_score,
            'put_score': put_score,
            'rsi': rsi,
            'adx': adx,
            'atr': atr,
            'pct_b': pct_b,
            'divergence': div,
            'lower_wick': lower_wick,
            'upper_wick': upper_wick,
            'reasons_call': reasons_call,
            'reasons_put': reasons_put
        }

    def generate_signal(self, df_1m=None, df_5m=None, global_trend="NEUTRAL", mid_trend="NEUTRAL", df_daily=None, tf_dict=None):
        """
        Глибокий мульти-таймфреймовий аналіз. Формує сигнал CALL/PUT
        із точним розрахунком часу експірації та підсумковим балом.
        """
        frames = {}
        if isinstance(tf_dict, dict):
            frames = tf_dict
        else:
            if df_1m is not None and not df_1m.empty:
                frames['1m'] = df_1m
            if df_5m is not None and not df_5m.empty:
                frames['5m'] = df_5m

        if isinstance(global_trend, pd.DataFrame):
            df_macro = global_trend
            global_trend = self.get_trend(df_macro, span_val=200)
            if df_daily is None:
                df_daily = df_macro

        pivots = self.calculate_pivots(df_daily) if df_daily is not None else {}

        if not frames:
            return {
                'signal': 'CALL',
                'score': 50,
                'confidence_level': 'LOW_DATA',
                'suggested_exp': 5,
                'expiration_minutes': 5,
                'strategy': 'Базовий аналіз',
                'priority': 4,
                'reason': 'Недостатньо історичних даних для аналізу'
            }

        evaluated_tfs = []
        for tf_name, df_tf in frames.items():
            tf_res = self.score_single_timeframe(df_tf, tf_name, global_trend=global_trend, pivots=pivots)
            evaluated_tfs.append(tf_res)

        total_call = sum(t['call_score'] for t in evaluated_tfs)
        total_put = sum(t['put_score'] for t in evaluated_tfs)

        if total_call >= total_put:
            direction = 'CALL'
            raw_score = total_call
            best_tf_res = max(evaluated_tfs, key=lambda x: x['call_score'])
            selected_reasons = best_tf_res['reasons_call']
            wick_ratio_final = best_tf_res['lower_wick']
        else:
            direction = 'PUT'
            raw_score = total_put
            best_tf_res = max(evaluated_tfs, key=lambda x: x['put_score'])
            selected_reasons = best_tf_res['reasons_put']
            wick_ratio_final = best_tf_res['upper_wick']

        max_possible = max(len(evaluated_tfs) * 120, 100)
        final_score = int(min(100, max(45, (raw_score / max_possible) * 100 + 30)))

        primary_tf = best_tf_res['tf_name']
        atr = best_tf_res['atr']
        adx = best_tf_res['adx']
        rsi = best_tf_res['rsi']
        div = best_tf_res['divergence']

        if primary_tf == '1m':
            base_exp = 3
        elif primary_tf in ['3m', '5m']:
            base_exp = 5
        elif primary_tf == '15m':
            base_exp = 15
        else:
            base_exp = 5

        atr_ma = frames[primary_tf]['atr'].rolling(20).mean().iloc[-2] if len(frames[primary_tf]) >= 20 else atr
        volatility_ratio = float(atr / (atr_ma + 1e-10))

        if volatility_ratio > 1.35 or adx > 32:
            suggested_exp = max(2, base_exp - 1)
        elif volatility_ratio < 0.75:
            suggested_exp = base_exp + 2
        else:
            suggested_exp = base_exp

        if final_score >= 75:
            confidence_level = "HIGH"
            strategy_name = f"Синхронний тренд {direction}"
        elif final_score >= 60:
            confidence_level = "MEDIUM"
            strategy_name = f"Локальний імпульс {primary_tf}"
        else:
            confidence_level = "LOW_RISK"
            strategy_name = f"Канальний відклик (Флет)"

        reason_str = ", ".join(selected_reasons[:3]) if selected_reasons else f"Пріоритет напрямку {direction}"

        return {
            'signal': direction,
            'score': final_score,
            'confidence_level': confidence_level,
            'rsi': round(rsi, 1),
            'adx': round(adx, 1),
            'atr': atr,
            'divergence': div,
            'suggested_exp': suggested_exp,
            'expiration_minutes': suggested_exp,
            'primary_tf': primary_tf,
            'strategy': strategy_name,
            'priority': 1 if final_score >= 75 else (2 if final_score >= 60 else 3),
            'reason': reason_str,
            'volatility_ratio': round(volatility_ratio, 3),
            'wick_ratio': round(wick_ratio_final, 3),
            'ema_dist': round(float(frames[primary_tf].get('ema_dist', pd.Series([0])).iloc[-1]), 3)
        }
