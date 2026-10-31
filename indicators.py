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

    def generate_master_confluence_signal(
        self, df_1m, df_3m, df_5m, df_15m, df_1h, global_trend, pivots, pair_name="EUR/USD", strategies_status=None
    ) -> dict:
        if df_1m is None or df_1m.empty or len(df_1m) < 15 or df_3m is None or df_3m.empty:
            return {'signal': 'NONE', 'reason': 'Мало даних для аналізу', 'suggested_exp': 10, 'strategy': 'NONE', 'signal_type_engine': 'trend_multi', 'is_shadow': False}

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
        reasons = []
        active_strategies_hit = []

        rsi_1m = float(last_1m.get('rsi', 50))
        z_score = float(last_1m.get('z_score', 0))
        stoch_k = float(last_1m.get('stoch_k', 50))
        wick_ratio = float(last_1m.get('wick_ratio', 0))
        adx = float(last_1m.get('adx', 20))
        macd_hist = float(last_1m.get('macd_hist', 0))

        w_zscore = get_strategy_adaptive_weight(pair_name, "Z-Score Mean Reversion")
        w_bb = get_strategy_adaptive_weight(pair_name, "Flat Matrix Bounce")
        w_trend = get_strategy_adaptive_weight(pair_name, "Trend Breakout")

        # 1. TREND_IMPULSE (Мульти-тренд)
        is_trend_active = strategies_status.get("trend_multi", True) and database.get_pair_strategy_status(pair_name, "trend_multi")
        if is_trend_active:
            if global_trend == "BULLISH":
                call_score += 2.2 * w_trend
                reasons.append("Глобальний тренд BULLISH (1H)")
                if adx > 22 and rsi_1m > 45 and rsi_1m < upper_rsi:
                    call_score += 1.2
                    reasons.append("Імпульсне продовження тренду")
                    active_strategies_hit.append("TREND_IMPULSE")
            elif global_trend == "BEARISH":
                put_score += 2.2 * w_trend
                reasons.append("Глобальний тренд BEARISH (1H)")
                if adx > 22 and rsi_1m < 55 and rsi_1m > lower_rsi:
                    put_score += 1.2
                    reasons.append("Імпульсне продовження тренду")
                    active_strategies_hit.append("TREND_IMPULSE")

        # 2. RANGE_CORRIDOR (Флетові коридори)
        is_range_active = strategies_status.get("flat_corridor", True) and database.get_pair_strategy_status(pair_name, "flat_corridor")
        if is_range_active:
            if last_1m['close'] <= last_1m['bb_lower'] and rsi_1m < lower_rsi:
                call_score += 2.0 * w_bb
                reasons.append(f"Відбій BB Lower (RSI: {rsi_1m:.1f})")
                active_strategies_hit.append("RANGE_CORRIDOR")
            elif last_1m['close'] >= last_1m['bb_upper'] and rsi_1m > upper_rsi:
                put_score += 2.0 * w_bb
                reasons.append(f"Відбій BB Upper (RSI: {rsi_1m:.1f})")
                active_strategies_hit.append("RANGE_CORRIDOR")

            if z_score <= -1.7:
                call_score += 1.8 * w_zscore
                reasons.append(f"Z-Score перепроданість ({z_score:.2f})")
                active_strategies_hit.append("RANGE_CORRIDOR")
            elif z_score >= 1.7:
                put_score += 1.8 * w_zscore
                reasons.append(f"Z-Score перекупленість ({z_score:.2f})")
                active_strategies_hit.append("RANGE_CORRIDOR")

        # 3. PRICE_LEVELS (Price Action & Pivot)
        is_levels_active = (strategies_status.get("price_action", True) and database.get_pair_strategy_status(pair_name, "price_action")) or \
                           (strategies_status.get("market_levels", True) and database.get_pair_strategy_status(pair_name, "market_levels"))
        if is_levels_active:
            dist_piv = self.get_min_dist_to_pivot_or_round(current_price, pivots)
            if dist_piv < 0.0003:
                if rsi_1m < 45:
                    call_score += 1.5
                    reasons.append("Реакція на Pivot/Круглий рівень (CALL)")
                    active_strategies_hit.append("PRICE_LEVELS")
                elif rsi_1m > 55:
                    put_score += 1.5
                    reasons.append("Реакція на Pivot/Круглий рівень (PUT)")
                    active_strategies_hit.append("PRICE_LEVELS")

            if wick_ratio > 0.40:
                if last_1m['close'] > last_1m['open']:
                    call_score += 1.3
                    reasons.append(f"Price Action: Нижня тінь ({wick_ratio:.2f})")
                    active_strategies_hit.append("PRICE_LEVELS")
                else:
                    put_score += 1.3
                    reasons.append(f"Price Action: Верхня тінь ({wick_ratio:.2f})")
                    active_strategies_hit.append("PRICE_LEVELS")

        # 4. MICRO_BREAKOUT (Волатильні пробої)
        is_breakout_active = (strategies_status.get("volume_volatility", True) and database.get_pair_strategy_status(pair_name, "volume_volatility")) or \
                             (strategies_status.get("oscillators", True) and database.get_pair_strategy_status(pair_name, "oscillators"))
        if is_breakout_active:
            if stoch_k < 18 and macd_hist > 0:
                call_score += 1.6
                reasons.append("Stoch екстремум + MACD конвергенція")
                active_strategies_hit.append("MICRO_BREAKOUT")
            elif stoch_k > 82 and macd_hist < 0:
                put_score += 1.6
                reasons.append("Stoch екстремум + MACD дивергенція")
                active_strategies_hit.append("MICRO_BREAKOUT")

        base_suggested_exp = 5 if regime in ["SQUEEZE", "TREND_EXHAUSTION"] else 10
        smart_exp = calculate_smart_expiration(base_suggested_exp, regime, vol_ratio)
        dominant_strategy = active_strategies_hit[0] if active_strategies_hit else "QUANTUM_MATRIX"

        # ПЕРЕВІРКА НА SHADOW MODE (якщо стратегія вимкнена користувачем, але дає сигнал)
        # Визначаємо чи вимкнена глобально чи по парах конкретна стратегія
        is_any_target_active = True
        if call_score >= dyn_threshold or put_score >= dyn_threshold:
            # Перевіримо чи активна хоча б одна спрацьована стратегія
            pass

        if call_score >= dyn_threshold and call_score > put_score:
            return {
                'signal': 'CALL',
                'strategy': dominant_strategy,
                'strategy_title': f'Квантовий CALL [{regime}]',
                'signal_type_engine': dominant_strategy,
                'confluence_count': int(round(call_score)),
                'suggested_exp': smart_exp,
                'regime': regime,
                'dyn_threshold': dyn_threshold,
                'reason': " | ".join(reasons)
            }
        elif put_score >= dyn_threshold and put_score > call_score:
            return {
                'signal': 'PUT',
                'strategy': dominant_strategy,
                'strategy_title': f'Квантовий PUT [{regime}]',
                'signal_type_engine': dominant_strategy,
                'confluence_count': int(round(put_score)),
                'suggested_exp': smart_exp,
                'regime': regime,
                'dyn_threshold': dyn_threshold,
                'reason': " | ".join(reasons)
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
        pivots = self.calculate_pivots(df_1h)
        
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
            "suggested_exp": sig_data.get("suggested_exp", 5),
            "global_trend": global_trend,
            "pivots": pivots,
            "current_price": current_price,
            "bb_width": float(df_ind_3m['bb_width'].iloc[-1]) if not df_ind_3m.empty else 0.001,
            "dist_pivot": self.get_min_dist_to_pivot_or_round(current_price, pivots),
            "regime": sig_data.get("regime", "RANGE_BOUND"),
            "dyn_threshold": sig_data.get("dyn_threshold", 2.3)
        }
