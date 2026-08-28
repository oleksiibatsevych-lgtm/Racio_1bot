import pandas as pd
import numpy as np

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df):
        if df.empty or len(df) < 14:
            return df
        
        # RSI 14
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        # Bollinger Bands та Z-Score
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_width'] = df['bb_width'].fillna(0.001)
        
        # Z-Score для відхилень і меж
        df['z_score'] = (df['close'] - sma) / (std + 1e-9)
        df['z_score'] = df['z_score'].fillna(0.0)

        # ADX та ATR
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
        if df.empty or 'rsi' not in df.columns or len(df) < 15:
            return "NONE"
        
        recent_prices = df['close'].iloc[-10:].values
        recent_rsi = df['rsi'].iloc[-10:].values
        
        price_lower_low = recent_prices[-1] < recent_prices[-5] and recent_prices[-5] < recent_prices[0]
        rsi_higher_low = recent_rsi[-1] > recent_rsi[-5] and recent_rsi[-5] > recent_rsi[0]
        if price_lower_low and rsi_higher_low:
            return "BULLISH_DIV"

        price_higher_high = recent_prices[-1] > recent_prices[-5] and recent_prices[-5] > recent_prices[0]
        rsi_lower_high = recent_rsi[-1] < recent_rsi[-5] and recent_rsi[-5] > recent_rsi[0]
        if price_higher_high and rsi_lower_high:
            return "BEARISH_DIV"

        return "NONE"

    def generate_signal(self, df, global_trend, mid_trend, ticker):
        if df.empty or len(df) < 15:
            return {'signal': 'HOLD', 'reason': 'Мало даних'}

        last_row = df.iloc[-1]
        rsi = float(last_row.get('rsi', 50))
        adx = float(last_row.get('adx', 20))
        atr = float(last_row.get('atr', 0.001))
        z_score = float(last_row.get('z_score', 0.0))
        bb_upper = float(last_row.get('bb_upper', 0))
        bb_lower = float(last_row.get('bb_lower', 0))
        close_price = float(last_row['close'])
        div = self.detect_divergence(df)

        signal = 'HOLD'
        reason_parts = []

        # РЕЖИМ 1: ФЛЕТ / КАНАЛ (ADX < 20) — робота від меж та Z-Score як гнучка точка входу
        if adx < 20:
            if close_price <= bb_lower * 1.005 or rsi < 42 or z_score <= -1.2:
                signal = 'CALL'
                reason_parts.append("Флет/Канал (відскок)")
                if z_score <= -1.2: reason_parts.append(f"Z-Score вхід ({z_score:.2f})")
                if close_price <= bb_lower * 1.005: reason_parts.append("Нижня межа BB")
                if rsi < 42: reason_parts.append(f"RSI ({rsi:.1f})")
            elif close_price >= bb_upper * 0.995 or rsi > 58 or z_score >= 1.2:
                signal = 'PUT'
                reason_parts.append("Флет/Канал (відскок)")
                if z_score >= 1.2: reason_parts.append(f"Z-Score вхід ({z_score:.2f})")
                if close_price >= bb_upper * 0.995: reason_parts.append("Верхня межа BB")
                if rsi > 58: reason_parts.append(f"RSI ({rsi:.1f})")

        # РЕЖИМ 2: ТРЕНД (ADX >= 20) — трендова торгівля з відкатом через RSI, дивергенції або Z-Score
        else:
            if global_trend == 'BULLISH' and mid_trend in ['BULLISH', 'NEUTRAL']:
                if rsi < 48 or div == 'BULLISH_DIV' or z_score < -1.0:
                    signal = 'CALL'
                    reason_parts.append("Тренд вгору")
                    if rsi < 48: reason_parts.append(f"RSI відкат ({rsi:.1f})")
                    if div == 'BULLISH_DIV': reason_parts.append("Бичача дивергенція")
                    if z_score < -1.0: reason_parts.append(f"Z-Score точка входу ({z_score:.2f})")
            elif global_trend == 'BEARISH' and mid_trend in ['BEARISH', 'NEUTRAL']:
                if rsi > 52 or div == 'BEARISH_DIV' or z_score > 1.0:
                    signal = 'PUT'
                    reason_parts.append("Тренд вниз")
                    if rsi > 52: reason_parts.append(f"RSI відкат ({rsi:.1f})")
                    if div == 'BEARISH_DIV': reason_parts.append("Ведмежа дивергенція")
                    if z_score > 1.0: reason_parts.append(f"Z-Score точка входу ({z_score:.2f})")

        reason = " + ".join(reason_parts) if reason_parts else "Умови не виконано"
        return {
            'signal': signal,
            'rsi': round(rsi, 1),
            'adx': round(adx, 1),
            'atr': atr,
            'z_score': round(z_score, 2),
            'divergence': div,
            'reason': reason
        }
