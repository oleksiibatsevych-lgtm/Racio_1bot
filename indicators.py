# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from regime_detector import detect_market_regime
from adaptive_optimizer import get_strategy_adaptive_weight

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 14:
            return df
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)

        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_width'] = df['bb_width'].fillna(0.001)

        df['z_score'] = (df['close'] - sma) / (std + 1e-10)
        df['z_score'] = df['z_score'].fillna(0.0)

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
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        lowest_low = df['low'].rolling(window=14).min()
        highest_high = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low + 1e-9))
        df['stoch_k'] = df['stoch_k'].fillna(50)

        candle_range = (df['high'] - df['low']).replace(0, 0.00001)
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        df['max_wick'] = np.maximum(upper_wick, lower_wick)
        df['wick_ratio'] = (df['max_wick'] / candle_range).fillna(0.0)

        return df

    def get_trend(self, df: pd.DataFrame, span_val=50) -> str:
        if df is None or df.empty or len(df) < span_val:
            return "NEUTRAL"
        ema = df['close'].ewm(span=span_val, adjust=False).mean()
        current_price = df['close'].iloc[-1]
        current_ema = ema.iloc[-1]
        if current_price > current_ema * 1.0001:
            return "BULLISH"
        elif current_price < current_ema * 0.9999:
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

    def get_min_dist_to_pivot_or_round(self, price: float, pivots: dict) -> float:
        if price <= 0:
            return 0.0
        distances = [abs(price - p_val) / price for p_val in pivots.values() if p_val > 0]
        round_factor = 100 if price > 50 else 10000
        mod_val = (price * round_factor) % 250
        distances.append(min(mod_val, 250 - mod_val) / (price * round_factor))
        return min(distances) if distances else 0.001

    def generate_master_confluence_signal(self, df_1m, df_3m, df_5m, df_15m, df_1h, global_trend, pivots, pair_name="EUR/USD") -> dict:
        if df_1m is None or df_1m.empty or len(df_1m) < 15 or df_3m is None or df_3m.empty:
            return {'signal': 'NONE', 'reason': 'Мало даних для аналізу', 'suggested_exp': 10, 'strategy': 'NONE', 'signal_type_engine': 'trend_multi'}

        regime = detect_market_regime(df_1m, df_3m, df_15m)
        
        call_score = 0.0
        put_score = 0.0
        reasons = []

        last_1m = df_1m.iloc[-1]
        rsi_1m = float(last_1m.get('rsi', 50))
        z_score = float(last_1m.get('z_score', 0))
        stoch_k = float(last_1m.get('stoch_k', 50))
        wick_ratio = float(last_1m.get('wick_ratio', 0))

        # Адаптивна вага для стратегій
        w_zscore = get_strategy_adaptive_weight(pair_name, "Z-Score Mean Reversion")
        w_bb = get_strategy_adaptive_weight(pair_name, "Flat Matrix Bounce")
        w_trend = get_strategy_adaptive_weight(pair_name, "Trend Breakout")

        # 1. Z-Score Mean Reversion
        if z_score <= -1.9:
            call_score += 2.5 * w_zscore
            reasons.append(f"Z-Score перепроданість ({z_score:.2f})")
        elif z_score >= 1.9:
            put_score += 2.5 * w_zscore
            reasons.append(f"Z-Score перекупленість ({z_score:.2f})")

        # 2. Bollinger / Flat Bounce
        if last_1m['close'] <= last_1m['bb_lower'] and rsi_1m < 36:
            call_score += 2.2 * w_bb
            reasons.append(f"Відбиття від BB Lower (RSI: {rsi_1m:.1f})")
        elif last_1m['close'] >= last_1m['bb_upper'] and rsi_1m > 64:
            put_score += 2.2 * w_bb
            reasons.append(f"Відбиття від BB Upper (RSI: {rsi_1m:.1f})")

        # 3. Trend Alignment
        if regime == "STRONG_TREND":
            if global_trend == "BULLISH" and rsi_1m < 49:
                call_score += 2.8 * w_trend
                reasons.append("Сильний висхідний тренд + імпульс")
            elif global_trend == "BEARISH" and rsi_1m > 51:
                put_score += 2.8 * w_trend
                reasons.append("Сильний низхідний тренд + імпульс")

        # 4. Stochastic & Price Action
        if stoch_k < 18 and wick_ratio > 0.45:
            call_score += 1.8
            reasons.append(f"Stoch перепроданість + тінь свічки ({wick_ratio:.2f})")
        elif stoch_k > 82 and wick_ratio > 0.45:
            put_score += 1.8
            reasons.append(f"Stoch перекупленість + тінь свічки ({wick_ratio:.2f})")

        threshold = 3.0
        if call_score >= threshold and call_score > put_score:
            return {
                'signal': 'CALL',
                'strategy': 'Quantum Confluence Matrix',
                'strategy_title': f'Квантовий сигнал CALL [{regime}]',
                'signal_type_engine': 'trend_multi',
                'confluence_count': int(round(call_score)),
                'suggested_exp': 10,
                'regime': regime,
                'reason': " | ".join(reasons)
            }
        elif put_score >= threshold and put_score > call_score:
            return {
                'signal': 'PUT',
                'strategy': 'Quantum Confluence Matrix',
                'strategy_title': f'Квантовий сигнал PUT [{regime}]',
                'signal_type_engine': 'trend_multi',
                'confluence_count': int(round(put_score)),
                'suggested_exp': 10,
                'regime': regime,
                'reason': " | ".join(reasons)
            }

        return {'signal': 'NONE', 'reason': f'Режим: {regime}. Бали нижче елітного порогу', 'suggested_exp': 10, 'strategy': 'NONE', 'signal_type_engine': 'trend_multi', 'regime': regime}

    def analyze_all_timeframes(self, df_daily, df_1h, df_15m, df_5m, df_3m, df_1m, pair_name="EUR/USD") -> dict:
        global_trend = self.get_trend(df_1h, span_val=200)
        pivots = self.calculate_pivots(df_1h)
        
        df_ind_1m = self.calculate_indicators(df_1m.copy()) if df_1m is not None and not df_1m.empty else pd.DataFrame()
        df_ind_3m = self.calculate_indicators(df_3m.copy()) if df_3m is not None and not df_3m.empty else pd.DataFrame()
        df_ind_5m = self.calculate_indicators(df_5m.copy()) if df_5m is not None and not df_5m.empty else pd.DataFrame()
        df_ind_15m = self.calculate_indicators(df_15m.copy()) if df_15m is not None and not df_15m.empty else pd.DataFrame()
        df_ind_1h = self.calculate_indicators(df_1h.copy()) if df_1h is not None and not df_1h.empty else pd.DataFrame()

        if df_ind_1m.empty or df_ind_3m.empty:
            return {"signal": "NONE", "reason": "Недостатньо даних"}

        sig_data = self.generate_master_confluence_signal(df_ind_1m, df_ind_3m, df_ind_5m, df_ind_15m, df_ind_1h, global_trend, pivots, pair_name)
        
        current_price = float(df_ind_1m['close'].iloc[-1]) if not df_ind_1m.empty else 0.0
        atr = float(df_ind_1m['atr'].iloc[-1]) if not df_ind_1m.empty and 'atr' in df_ind_1m.columns else 0.001

        return {
            "signal": sig_data.get("signal", "NONE"),
            "strategy": sig_data.get("strategy", "QUANTUM_MATRIX"),
            "strategy_title": sig_data.get("strategy_title", "Quantum Matrix"),
            "signal_type_engine": sig_data.get("signal_type_engine", "trend_multi"),
            "confluence_count": sig_data.get("confluence_count", 1),
            "reason": sig_data.get("reason", "Умови не виконано"),
            "rsi": float(df_ind_1m['rsi'].iloc[-1]) if 'rsi' in df_ind_1m.columns else 50,
            "adx": float(df_ind_3m['adx'].iloc[-1]) if 'adx' in df_ind_3m.columns else 20,
            "atr": atr,
            "volatility_ratio": (atr / current_price) * 1000 if current_price > 0 else 1.0,
            "suggested_exp": sig_data.get("suggested_exp", 10),
            "global_trend": global_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_ind_3m['bb_width'].iloc[-1]) if not df_ind_3m.empty else 0.001,
            "dist_pivot": self.get_min_dist_to_pivot_or_round(current_price, pivots),
            "regime": sig_data.get("regime", "RANGE_BOUND"),
        }
