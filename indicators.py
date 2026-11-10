# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from regime_detector import detect_market_regime
from adaptive_optimizer import (
    get_strategy_adaptive_weight, 
    get_adaptive_rsi_limits,
    calculate_dynamic_threshold,
    calculate_smart_expiration
)
import database

class AdaptiveTechnicalAnalysis:
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or len(df) < 14:
            return df
        
        df = df.copy()
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
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma + 1e-10)
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

        # Обчислення VWAP (Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['volume'] if ('volume' in df.columns and df['volume'].sum() > 0) else pd.Series(1.0, index=df.index)
        # VWAP скидається щодня (UTC), інакше це просто «середнє за весь датасет»
        try:
            session_key = pd.Series(df.index.date, index=df.index)
            pv = typical_price * vol
            df['vwap'] = pv.groupby(session_key).cumsum() / (vol.groupby(session_key).cumsum() + 1e-10)
        except Exception:
            df['vwap'] = (typical_price * vol).cumsum() / (vol.cumsum() + 1e-10)

        # Обчислення Каналів Кельтнера (Keltner Channels)
        df['keltner_mid'] = df['ema_20']
        df['keltner_upper'] = df['keltner_mid'] + (2.0 * df['atr'])
        df['keltner_lower'] = df['keltner_mid'] - (2.0 * df['atr'])

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

    def detect_rsi_divergence(self, df: pd.DataFrame) -> str:
        """
        Детектор бичачої та ведмежої дивергенції за останні 15 свічок.
        """
        if df is None or len(df) < 15 or 'rsi' not in df.columns:
            return "NONE"
        
        try:
            p_curr_low = df['low'].iloc[-1]
            p_prev_low = df['low'].iloc[-12:-2].min()
            rsi_curr_low = df['rsi'].iloc[-1]
            rsi_prev_low = df['rsi'].iloc[-12:-2].min()

            p_curr_high = df['high'].iloc[-1]
            p_prev_high = df['high'].iloc[-12:-2].max()
            rsi_curr_high = df['rsi'].iloc[-1]
            rsi_prev_high = df['rsi'].iloc[-12:-2].max()

            # Бичача дивергенція: ціна робить нижчий мінімум, а RSI робить вищий мінімум
            if p_curr_low < p_prev_low and rsi_curr_low > rsi_prev_low and rsi_curr_low < 45:
                return "BULLISH"

            # Ведмежа дивергенція: ціна робить вищий максимум, а RSI робить нижчий максимум
            if p_curr_high > p_prev_high and rsi_curr_high < rsi_prev_high and rsi_curr_high > 55:
                return "BEARISH"
        except Exception:
            pass

        return "NONE"

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

    def generate_master_confluence_signal(
        self, df_1m, df_3m, df_5m, df_15m, df_1h, global_trend, pivots, pair_name="EUR/USD", strategies_status=None
    ) -> dict:
        if df_1m is None or df_1m.empty or len(df_1m) < 15 or df_3m is None or df_3m.empty:
            return {'signal': 'NONE', 'reason': 'Мало даних для аналізу', 'suggested_exp': 10, 'strategy': 'NONE', 'signal_type_engine': 'trend_multi'}

        if strategies_status is None:
            strategies_status = {}

        regime = detect_market_regime(df_1m, df_3m, df_15m)
        last_1m = df_1m.iloc[-1]
        
        current_price = float(last_1m['close'])
        curr_atr = float(last_1m.get('atr', 0.001))
        base_atr = float(df_1m['atr'].mean()) if 'atr' in df_1m.columns else curr_atr
        vol_ratio = (curr_atr / base_atr) if base_atr > 0 else 1.0

        lower_rsi, upper_rsi = get_adaptive_rsi_limits(regime, curr_atr, base_atr)
        dyn_threshold = calculate_dynamic_threshold(regime, vol_ratio, base_threshold=2.3)

        call_score = 0.0
        put_score = 0.0
        call_reasons, put_reasons = [], []
        call_hits, put_hits = [], []

        rsi_1m = float(last_1m.get('rsi', 50))
        z_score = float(last_1m.get('z_score', 0))
        stoch_k = float(last_1m.get('stoch_k', 50))
        wick_ratio = float(last_1m.get('wick_ratio', 0))
        adx = float(last_1m.get('adx', 20))
        macd_hist = float(last_1m.get('macd_hist', 0))
        vwap_val = float(last_1m.get('vwap', current_price))
        ema_20_val = float(last_1m.get('ema_20', current_price))
        
        divergence = self.detect_rsi_divergence(df_1m)

        w_zscore = get_strategy_adaptive_weight(pair_name, "Z-Score Mean Reversion")
        w_bb = get_strategy_adaptive_weight(pair_name, "Flat Matrix Bounce")
        w_trend = get_strategy_adaptive_weight(pair_name, "TREND_IMPULSE")

        # 1. КЛАСТЕР: TREND_IMPULSE (Виключно за трендом на відкатах до EMA/VWAP)
        is_trend_active = strategies_status.get("trend_multi", True) and database.get_pair_strategy_status(pair_name, "trend_multi")
        if is_trend_active and regime != "CHAOS":
            mid_trend = self.get_trend(df_15m, span_val=50)
            confirm_trend = self.get_trend(df_3m, span_val=50)
            previous = df_1m.iloc[-2]
            previous_ema = float(previous.get('ema_20', previous['close']))
            previous_vwap = float(previous.get('vwap', previous['close']))
            if global_trend == mid_trend == confirm_trend == "BULLISH":
                # Відкат до EMA20 або VWAP під час висхідного тренду
                if (previous['low'] <= previous_ema * 1.0005 or previous['low'] <= previous_vwap * 1.0005) and current_price > last_1m['open'] and current_price > ema_20_val and rsi_1m > 38 and rsi_1m < upper_rsi:
                    call_score += 2.5 * w_trend
                    call_reasons.append("Trend Pullback Continuation (BULLISH EMA/VWAP)")
                    call_hits.append("TREND_IMPULSE")
            elif global_trend == mid_trend == confirm_trend == "BEARISH":
                # Відкат до EMA20 або VWAP під час спадного тренду
                if (previous['high'] >= previous_ema * 0.9995 or previous['high'] >= previous_vwap * 0.9995) and current_price < last_1m['open'] and current_price < ema_20_val and rsi_1m < 62 and rsi_1m > lower_rsi:
                    put_score += 2.5 * w_trend
                    put_reasons.append("Trend Pullback Continuation (BEARISH EMA/VWAP)")
                    put_hits.append("TREND_IMPULSE")

        # 2. КЛАСТЕР: RANGE_CORRIDOR (Флетові коридори + RSI Divergence)
        is_range_active = strategies_status.get("flat_corridor", True) and database.get_pair_strategy_status(pair_name, "flat_corridor")
        if is_range_active and regime in ["RANGE_BOUND", "TREND_EXHAUSTION"]:
            if (last_1m['close'] <= last_1m['bb_lower'] or z_score <= -1.8) and (rsi_1m < lower_rsi or divergence == "BULLISH"):
                call_score += 2.4 * w_bb
                call_reasons.append(f"Range Corridor Bounce (BB Lower / RSI Div: {divergence})")
                call_hits.append("RANGE_CORRIDOR")
            elif (last_1m['close'] >= last_1m['bb_upper'] or z_score >= 1.8) and (rsi_1m > upper_rsi or divergence == "BEARISH"):
                put_score += 2.4 * w_bb
                put_reasons.append(f"Range Corridor Bounce (BB Upper / RSI Div: {divergence})")
                put_hits.append("RANGE_CORRIDOR")

        # 3. КЛАСТЕР: PRICE_LEVELS (Price Action & Pivot + Liquidity Sweep у CHAOS)
        is_levels_active = strategies_status.get("price_action", strategies_status.get("market_levels", True)) and database.get_pair_strategy_status(pair_name, "price_action")
        if is_levels_active:
            if regime == "CHAOS":
                # У режимі CHAOS дозволено ВИКЛЮЧНО Liquidity Sweep з довгою тінню
                candle_range = float(last_1m['high'] - last_1m['low'])
                lower_wick = min(last_1m['open'], last_1m['close']) - last_1m['low']
                upper_wick = last_1m['high'] - max(last_1m['open'], last_1m['close'])
                if candle_range > 0:
                    if last_1m['close'] > last_1m['open'] and lower_wick / candle_range > 0.45:
                        call_score += 2.4
                        call_reasons.append(f"CHAOS Liquidity Sweep (Нижня тінь: {lower_wick / candle_range:.2f})")
                        call_hits.append("PRICE_LEVELS")
                    elif last_1m['close'] < last_1m['open'] and upper_wick / candle_range > 0.45:
                        put_score += 2.4
                        put_reasons.append(f"CHAOS Liquidity Sweep (Верхня тінь: {upper_wick / candle_range:.2f})")
                        put_hits.append("PRICE_LEVELS")
            else:
                dist_piv = self.get_min_dist_to_pivot_or_round(current_price, pivots)
                if dist_piv < 0.0003 and wick_ratio > 0.35:
                    if rsi_1m < 48:
                        call_score += 1.8
                        call_reasons.append("Pivot Level Reaction + Wick (CALL)")
                        call_hits.append("PRICE_LEVELS")
                    elif rsi_1m > 52:
                        put_score += 1.8
                        put_reasons.append("Pivot Level Reaction + Wick (PUT)")
                        put_hits.append("PRICE_LEVELS")

        # 4. КЛАСТЕР: MICRO_BREAKOUT (VWAP Bounce & SQUEEZE з фільтром Candle Close)
        is_breakout_active = strategies_status.get("oscillators", strategies_status.get("volume_volatility", True)) and database.get_pair_strategy_status(pair_name, "oscillators")
        if is_breakout_active:
            if regime == "SQUEEZE":
                # Фільтр підтвердження імпульсної свічки поглинання (Candle Close Filter)
                candle_body = abs(last_1m['close'] - last_1m['open'])
                if candle_body > curr_atr * 0.7:
                    if last_1m['close'] > last_1m['bb_upper'] and macd_hist > 0:
                        call_score += 2.2
                        call_reasons.append("SQUEEZE Confirmed Breakout (CALL)")
                        call_hits.append("MICRO_BREAKOUT")
                    elif last_1m['close'] < last_1m['bb_lower'] and macd_hist < 0:
                        put_score += 2.2
                        put_reasons.append("SQUEEZE Confirmed Breakout (PUT)")
                        put_hits.append("MICRO_BREAKOUT")
            else:
                # VWAP + Stochastic Micro-bounce
                if stoch_k < 20 and current_price > vwap_val and macd_hist > 0:
                    call_score += 1.8
                    call_reasons.append("VWAP Micro-Bounce (CALL)")
                    call_hits.append("MICRO_BREAKOUT")
                elif stoch_k > 80 and current_price < vwap_val and macd_hist < 0:
                    put_score += 1.8
                    put_reasons.append("VWAP Micro-Bounce (PUT)")
                    put_hits.append("MICRO_BREAKOUT")

        base_suggested_exp = 3 if regime in ["SQUEEZE", "TREND_EXHAUSTION"] else 5
        smart_exp = calculate_smart_expiration(base_suggested_exp, regime, vol_ratio)

        if call_score >= dyn_threshold and call_score > put_score:
            return {
                'signal': 'CALL',
                'strategy': call_hits[0],
                'strategy_title': f'Квантовий CALL [{regime}]',
                'signal_type_engine': call_hits[0],
                'confluence_count': int(round(call_score)),
                'suggested_exp': smart_exp,
                'regime': regime,
                'dyn_threshold': dyn_threshold,
                'reason': " | ".join(call_reasons)
            }
        elif put_score >= dyn_threshold and put_score > call_score:
            return {
                'signal': 'PUT',
                'strategy': put_hits[0],
                'strategy_title': f'Квантовий PUT [{regime}]',
                'signal_type_engine': put_hits[0],
                'confluence_count': int(round(put_score)),
                'suggested_exp': smart_exp,
                'regime': regime,
                'dyn_threshold': dyn_threshold,
                'reason': " | ".join(put_reasons)
            }

        return {
            'signal': 'NONE', 
            'reason': f'Режим: {regime}. Бали нижче динамічного порогу ({dyn_threshold:.1f})', 
            'suggested_exp': smart_exp, 
            'strategy': 'NONE', 
            'signal_type_engine': 'trend_multi', 
            'regime': regime
        }

    def analyze_all_timeframes(self, df_daily, df_1h, df_15m, df_5m, df_3m, df_1m, pair_name="EUR/USD", strategies_status=None) -> dict:
        global_trend = self.get_trend(df_1h, span_val=200)
        pivots = self.calculate_pivots(df_daily if df_daily is not None and not df_daily.empty and len(df_daily) >= 2 else df_1h)
        
        df_ind_1m = self.calculate_indicators(df_1m) if df_1m is not None and not df_1m.empty else pd.DataFrame()
        df_ind_3m = self.calculate_indicators(df_3m) if df_3m is not None and not df_3m.empty else pd.DataFrame()
        df_ind_5m = self.calculate_indicators(df_5m) if df_5m is not None and not df_5m.empty else pd.DataFrame()
        df_ind_15m = self.calculate_indicators(df_15m) if df_15m is not None and not df_15m.empty else pd.DataFrame()
        df_ind_1h = self.calculate_indicators(df_1h) if df_1h is not None and not df_1h.empty else pd.DataFrame()

        if df_ind_1m.empty or df_ind_3m.empty:
            return {"signal": "NONE", "reason": "Недостатньо даних"}

        sig_data = self.generate_master_confluence_signal(
            df_ind_1m, df_ind_3m, df_ind_5m, df_ind_15m, df_ind_1h, global_trend, pivots, pair_name, strategies_status
        )
        
        current_price = float(df_ind_1m['close'].iloc[-1]) if not df_ind_1m.empty else 0.0
        atr = float(df_ind_1m['atr'].iloc[-1]) if not df_ind_1m.empty and 'atr' in df_ind_1m.columns else 0.001
        atr_mean = float(df_ind_1m['atr'].mean()) if 'atr' in df_ind_1m.columns else atr
        vol_ratio_rel = (atr / atr_mean) if atr_mean > 0 else 1.0
        ema_20_last = float(df_ind_1m['ema_20'].iloc[-1]) if 'ema_20' in df_ind_1m.columns else current_price
        ema_dist = ((current_price - ema_20_last) / current_price) if current_price > 0 else 0.0
        wick_ratio_last = float(df_ind_1m['wick_ratio'].iloc[-1]) if 'wick_ratio' in df_ind_1m.columns else 0.0
        divergence_last = self.detect_rsi_divergence(df_ind_1m)

        return {
            "divergence": divergence_last,
            "wick_ratio": wick_ratio_last,
            "ema_dist": ema_dist,
            "vol_ratio_rel": vol_ratio_rel,
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
            "suggested_exp": sig_data.get("suggested_exp", 5),
            "global_trend": global_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_ind_3m['bb_width'].iloc[-1]) if not df_ind_3m.empty else 0.001,
            "dist_pivot": self.get_min_dist_to_pivot_or_round(current_price, pivots),
            "regime": sig_data.get("regime", "RANGE_BOUND"),
            "dyn_threshold": sig_data.get("dyn_threshold", 2.3)
        }
