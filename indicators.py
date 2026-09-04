import pandas as pd
import numpy as np

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df):
        if df.empty or len(df) < 14:
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
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        df['atr'] = atr.fillna(0.0010)

        plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / (atr + 1e-9)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / (atr + 1e-9)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
        df['adx'] = dx.rolling(window=14).mean().fillna(20)

        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        return df

    def get_trend(self, df, span_val=50):
        if df.empty or len(df) < span_val:
            return "NEUTRAL"
        ema = df['close'].ewm(span=span_val, adjust=False).mean()
        current_price = df['close'].iloc[-1]
        current_ema = ema.iloc[-1]
        if current_price > current_ema * 1.001:
            return "BULLISH"
        elif current_price < current_ema * 0.999:
            return "BEARISH"
        return "NEUTRAL"

    def calculate_pivots(self, df_macro):
        if df_macro.empty or len(df_macro) < 2:
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

    def detect_divergence(self, df):
        if df.empty or 'rsi' not in df.columns or len(df) < 35:
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

    def generate_signal(self, df_1m, df_5m, global_trend, mid_trend):
        if df_5m.empty or len(df_5m) < 15 or df_1m.empty or len(df_1m) < 10:
            return {'signal': 'HOLD', 'reason': 'Мало даних', 'suggested_exp': 10}

        last_5m = df_5m.iloc[-1]
        last_1m = df_1m.iloc[-1]
        
        rsi_5m = float(last_5m.get('rsi', 50))
        rsi_1m = float(last_1m.get('rsi', 50))
        adx = float(last_5m.get('adx', 20))
        atr = float(last_5m.get('atr', 0.001))
        bb_upper = float(last_5m.get('bb_upper', 0))
        bb_lower = float(last_5m.get('bb_lower', 0))
        close_5m = float(last_5m['close'])
        close_1m = float(last_1m['close'])
        div = self.detect_divergence(df_5m)

        signal = 'HOLD'
        reason_parts = []
        expiration = 10

        if adx < 22:
            if close_1m <= bb_lower or rsi_1m < 32:
                signal = 'CALL'
                expiration = 3
                reason_parts.append("Флет M1/M5 (відскок знизу)")
                if close_1m <= bb_lower: reason_parts.append("Пробій нижньої BB")
                if rsi_1m < 32: reason_parts.append(f"RSI 1m ({rsi_1m:.1f})")
            elif close_1m >= bb_upper or rsi_1m > 68:
                signal = 'PUT'
                expiration = 3
                reason_parts.append("Флет M1/M5 (відскок зверху)")
                if close_1m >= bb_upper: reason_parts.append("Пробій верхньої BB")
                if rsi_1m > 68: reason_parts.append(f"RSI 1m ({rsi_1m:.1f})")
        else:
            effective_trend = global_trend if global_trend != 'NEUTRAL' else mid_trend
            if effective_trend == 'BULLISH':
                if rsi_5m < 50 or div == 'BULLISH_DIV':
                    signal = 'CALL'
                    expiration = 5
                    reason_parts.append("Тренд вгору (1h/15m)")
                    if rsi_5m < 50: reason_parts.append(f"RSI відкат ({rsi_5m:.1f})")
                    if div == 'BULLISH_DIV': reason_parts.append("Бичача дивергенція")
            elif effective_trend == 'BEARISH':
                if rsi_5m > 50 or div == 'BEARISH_DIV':
                    signal = 'PUT'
                    expiration = 5
                    reason_parts.append("Тренд вниз (1h/15m)")
                    if rsi_5m > 50: reason_parts.append(f"RSI відкат ({rsi_5m:.1f})")
                    if div == 'BEARISH_DIV': reason_parts.append("Ведмежа дивергенція")

        if signal != 'HOLD':
            ema_1m = float(df_1m['ema_10'].iloc[-1]) if 'ema_10' in df_1m.columns else close_1m
            if signal == 'CALL' and close_1m < ema_1m * 0.995 and adx >= 22:
                signal = 'HOLD'
            elif signal == 'PUT' and close_1m > ema_1m * 1.005 and adx >= 22:
                signal = 'HOLD'
            else:
                reason_parts.append("1m фільтр пройдено")

        reason = " + ".join(reason_parts) if reason_parts else "Умови не виконано"
        return {
            'signal': signal,
            'rsi': round(rsi_5m, 1),
            'adx': round(adx, 1),
            'atr': atr,
            'divergence': div,
            'suggested_exp': expiration,
            'reason': reason
        }
