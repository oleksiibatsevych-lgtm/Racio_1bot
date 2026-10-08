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

        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        return df

    def get_trend(self, df: pd.DataFrame, span_val=50) -> str:
        if df is None or df.empty or len(df) < span_val:
            return "NEUTRAL"
        ema = df['close'].ewm(span=span_val, adjust=False).mean()
        current_price = df['close'].iloc[-1]
        current_ema = ema.iloc[-1]
        if current_price > current_ema * 1.001:
            return "BULLISH"
        elif current_price < current_ema * 0.999:
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

    def calculate_flexible_expiration(self, adx: float, bb_width: float, div: str, global_trend: str, mid_trend: str) -> int:
        """
        Динамічний розрахунок експірації в діапазоні від 1 до 60 хвилин.
        """
        if bb_width > 0.005 and adx < 18:
            exp = int(np.clip(round(bb_width * 500), 1, 4))
        elif adx < 25:
            exp = 5
        elif 25 <= adx < 35:
            exp = 10 if global_trend == mid_trend else 7
        elif 35 <= adx < 45:
            exp = 15
        else:
            exp = 30

        if div != "NONE":
            exp = min(60, exp + 15)

        if global_trend != "NEUTRAL" and global_trend == mid_trend and adx > 30:
            exp = min(60, int(exp * 1.5))

        return int(max(1, min(60, exp)))

    def generate_signal(self, df_1m: pd.DataFrame, df_5m: pd.DataFrame, global_trend: str, mid_trend: str, df_macro: pd.DataFrame=None, df_3m: pd.DataFrame=None) -> dict:
        if df_5m is None or df_5m.empty or len(df_5m) < 15 or df_1m is None or df_1m.empty or len(df_1m) < 10:
            return {'signal': 'HOLD', 'reason': 'Мало даних', 'suggested_exp': 5}

        last_5m = df_5m.iloc[-1]
        last_1m = df_1m.iloc[-1]
        
        rsi_5m = float(last_5m.get('rsi', 50))
        rsi_1m = float(last_1m.get('rsi', 50))
        adx = float(last_5m.get('adx', 20))
        atr = float(last_5m.get('atr', 0.001))
        bb_upper = float(last_5m.get('bb_upper', 0))
        bb_lower = float(last_5m.get('bb_lower', 0))
        bb_width = float(last_5m.get('bb_width', 0.001))
        close_5m = float(last_5m['close'])
        close_1m = float(last_1m['close'])
        div = self.detect_divergence(df_5m)

        pivots = self.calculate_pivots(df_macro) if df_macro is not None and not df_macro.empty else {}
        r1 = pivots.get('R1', 0)
        s1 = pivots.get('S1', 0)

        signal = 'HOLD'
        reason_parts = []
        
        expiration = self.calculate_flexible_expiration(adx, bb_width, div, global_trend, mid_trend)
        effective_trend = global_trend if global_trend != 'NEUTRAL' else mid_trend

        # Пом'якшені умови RSI для збільшення кількості сигналів
        if adx < 25:
            if close_1m <= bb_lower or rsi_1m < 46:
                signal = 'CALL'
                reason_parts.append("Флет: відскок знизу")
                reason_parts.append(f"RSI 1m ({rsi_1m:.1f})")
            elif close_1m >= bb_upper or rsi_1m > 54:
                signal = 'PUT'
                reason_parts.append("Флет: відскок зверху")
                reason_parts.append(f"RSI 1m ({rsi_1m:.1f})")
        else:
            if effective_trend == 'BULLISH' and rsi_5m < 72:
                if (bb_lower > 0 and close_5m <= bb_lower * 1.004) or (s1 > 0 and close_5m <= s1 * 1.003) or (rsi_5m < 50) or (div == 'BULLISH_DIV'):
                    signal = 'CALL'
                    reason_parts.append(f"Тренд вгору (ADX: {adx:.1f})")
                    if rsi_5m < 50: reason_parts.append(f"Корекція RSI ({rsi_5m:.1f})")
                    if div == 'BULLISH_DIV': reason_parts.append("Бичача дивергенція")
            elif effective_trend == 'BEARISH' and rsi_5m > 28:
                if (bb_upper > 0 and close_5m >= bb_upper * 0.996) or (r1 > 0 and close_5m >= r1 * 0.997) or (rsi_5m > 50) or (div == 'BEARISH_DIV'):
                    signal = 'PUT'
                    reason_parts.append(f"Тренд вниз (ADX: {adx:.1f})")
                    if rsi_5m > 50: reason_parts.append(f"Корекція RSI ({rsi_5m:.1f})")
                    if div == 'BEARISH_DIV': reason_parts.append("Ведмежа дивергенція")

        # Перевірка 3M таймфрейму, якщо на 5M сигнал відсутній
        if signal == 'HOLD' and df_3m is not None and not df_3m.empty and len(df_3m) >= 10:
            last_3m = df_3m.iloc[-1]
            rsi_3m = float(last_3m.get('rsi', 50))
            if rsi_3m < 38:
                signal = 'CALL'
                reason_parts.append("Скальпінг 3M: імпульс перепроданості")
                expiration = max(1, min(60, int(expiration // 1.5)))
            elif rsi_3m > 62:
                signal = 'PUT'
                reason_parts.append("Скальпінг 3M: імпульс перекупленості")
                expiration = max(1, min(60, int(expiration // 1.5)))

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

    def analyze_all_timeframes(self, df_daily, df_1h, df_15m, df_5m, df_3m, df_1m) -> dict:
        df_macro = df_1h
        df_mid = df_15m
        df_fast = df_5m
        df_micro = df_1m
        
        global_trend = self.get_trend(df_macro, span_val=200)
        mid_trend = self.get_trend(df_mid, span_val=50)
        pivots = self.calculate_pivots(df_macro)
        
        df_indicators_5m = self.calculate_indicators(df_fast.copy()) if df_fast is not None and not df_fast.empty else pd.DataFrame()
        df_indicators_3m = self.calculate_indicators(df_3m.copy()) if df_3m is not None and not df_3m.empty else pd.DataFrame()
        df_indicators_1m = self.calculate_indicators(df_micro.copy()) if df_micro is not None and not df_micro.empty else pd.DataFrame()
        
        if df_indicators_5m.empty or df_indicators_1m.empty:
            return {"signal": "NONE", "reason": "Недостатньо даних"}

        sig_data = self.generate_signal(df_indicators_1m, df_indicators_5m, global_trend, mid_trend, df_macro, df_indicators_3m)
        
        current_price = float(df_indicators_5m['close'].iloc[-1]) if not df_indicators_5m.empty else 0.0
        atr = float(df_indicators_5m['atr'].iloc[-1]) if not df_indicators_5m.empty and 'atr' in df_indicators_5m.columns else 0.001
        volatility_ratio = (atr / current_price) * 1000 if current_price > 0 else 1.0

        res_signal = sig_data.get("signal", "HOLD")
        if res_signal == "HOLD":
            res_signal = "NONE"

        res = {
            "signal": res_signal,
            "reason": sig_data.get("reason", "Умови не виконано"),
            "rsi": sig_data.get("rsi", 50),
            "adx": sig_data.get("adx", 20),
            "atr": atr,
            "volatility_ratio": volatility_ratio,
            "atr_ratio": volatility_ratio,
            "divergence": sig_data.get("divergence", "NONE"),
            "suggested_exp": sig_data.get("suggested_exp", 5),
            "global_trend": global_trend,
            "mid_trend": mid_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_indicators_5m['bb_width'].iloc[-1]) if not df_indicators_5m.empty and 'bb_width' in df_indicators_5m.columns else 0.001,
        }
        return res
