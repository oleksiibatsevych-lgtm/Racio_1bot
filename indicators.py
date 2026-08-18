import numpy as np
import pandas as pd

class AdaptiveTechnicalAnalysis:
    def __init__(self):
        self.rsi_window = 14

    def calculate_adx(self, df: pd.DataFrame, window=14) -> pd.Series:
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(span=window, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=window, adjust=False).mean() / (atr + 1e-9)
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=window, adjust=False).mean() / (atr + 1e-9)

        dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
        adx = dx.ewm(span=window, adjust=False).mean()
        return adx

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        res_df = df.copy()
        if isinstance(res_df.columns, pd.MultiIndex):
            res_df.columns = res_df.columns.get_level_values(0)
        res_df.columns = [str(c).lower() for c in res_df.columns]

        delta = res_df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0.0)
        avg_gain = gain.ewm(com=self.rsi_window - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=self.rsi_window - 1, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        res_df['rsi'] = 100 - (100 / (1 + rs))

        res_df['ema_10'] = res_df['close'].ewm(span=10, adjust=False).mean()
        res_df['ema_20'] = res_df['close'].ewm(span=20, adjust=False).mean()

        bb_window = 20
        res_df['bb_middle'] = res_df['close'].rolling(window=bb_window).mean()
        bb_std = res_df['close'].rolling(window=bb_window).std()
        res_df['bb_upper'] = res_df['bb_middle'] + (bb_std * 2)
        res_df['bb_lower'] = res_df['bb_middle'] - (bb_std * 2)
        
        # Ширина смуг Боллінджера для фільтрації мертвого ринку
        res_df['bb_width'] = (res_df['bb_upper'] - res_df['bb_lower']) / res_df['bb_middle']

        res_df['local_support'] = res_df['low'].rolling(window=15).min()
        res_df['local_resistance'] = res_df['high'].rolling(window=15).max()

        tr1 = res_df['high'] - res_df['low']
        tr2 = (res_df['high'] - res_df['close'].shift(1)).abs()
        tr3 = (res_df['low'] - res_df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        res_df['atr'] = tr.ewm(span=14, adjust=False).mean()

        res_df['adx'] = self.calculate_adx(res_df, window=14)
        return res_df

    def get_trend(self, df: pd.DataFrame, span_val=200) -> str:
        if df is None or df.empty:
            return "NEUTRAL"
        g_df = df.copy()
        g_df['ema'] = g_df['close'].ewm(span=span_val, adjust=False).mean()
        last_close = g_df['close'].iloc[-1]
        last_ema = g_df['ema'].iloc[-1]
        return "UP" if last_close > last_ema else "DOWN"

    def generate_signal(self, df_5m: pd.DataFrame, global_trend: str, mid_trend: str, asset_name: str) -> dict:
        default = {
            "signal": "HOLD", "rsi": None, "atr": None, "adx": None,
            "reason": "No setup", "global_trend": global_trend,
            "mid_trend": mid_trend, "local_trend": "NEUTRAL"
        }
        if len(df_5m) < 25:
            return default

        last = df_5m.iloc[-1]
        c = last['close']
        l_sup = last['local_support']
        l_res = last['local_resistance']
        rsi = last['rsi']
        atr = last['atr']
        adx = last['adx']
        ema10 = last['ema_10']
        bb_lower = last['bb_lower']
        bb_upper = last['bb_upper']
        bb_width = last.get('bb_width', 0.01)

        # 1. Фільтр надто вузького флету (шуму) за допомогою Боллінджера
        min_width = 0.0005 if "JPY" in asset_name.upper() else 0.001
        if bb_width < min_width:
            return {
                "signal": "HOLD", "rsi": round(float(rsi), 2), "atr": round(float(atr), 5), "adx": round(float(adx), 2),
                "reason": "Низька волатильність (Флет Боллінджера)",
                "global_trend": global_trend, "mid_trend": mid_trend, "local_trend": "FLAT"
            }

        local_trend = "UP" if c > ema10 else "DOWN"
        is_flat = adx < 18

        if is_flat:
            if c <= bb_lower or (abs(c - l_sup) / c < 0.005 and rsi < 42):
                return {
                    "signal": "CALL", "rsi": round(float(rsi), 2), "atr": round(float(atr), 5), "adx": round(float(adx), 2),
                    "reason": f"Флєт (ADX: {round(adx, 1)}) + Відскок від межі BB/Підтримки (RSI: {round(rsi, 1)})",
                    "global_trend": global_trend, "mid_trend": mid_trend, "local_trend": local_trend
                }
            elif c >= bb_upper or (abs(c - l_res) / c < 0.005 and rsi > 58):
                return {
                    "signal": "PUT", "rsi": round(float(rsi), 2), "atr": round(float(atr), 5), "adx": round(float(adx), 2),
                    "reason": f"Флєт (ADX: {round(adx, 1)}) + Відскок від межі BB/Опору (RSI: {round(rsi, 1)})",
                    "global_trend": global_trend, "mid_trend": mid_trend, "local_trend": local_trend
                }
        else:
            dist_sup = abs(c - l_sup) / c
            dist_res = abs(c - l_res) / c

            # Адаптивні зони та перевірка тренду для збереження кількості й покращення вінрейту
            if (global_trend == "UP" or mid_trend == "UP") and local_trend == "UP":
                if rsi < 52 or dist_sup < 0.012 or c > ema10:
                    return {
                        "signal": "CALL", "rsi": round(float(rsi), 2), "atr": round(float(atr), 5), "adx": round(float(adx), 2),
                        "reason": f"Тренд ВГОРУ (ADX: {round(adx, 1)}) + Адаптивний RSI/Імпульс",
                        "global_trend": global_trend, "mid_trend": mid_trend, "local_trend": local_trend
                    }

            if (global_trend == "DOWN" or mid_trend == "DOWN") and local_trend == "DOWN":
                if rsi > 48 or dist_res < 0.012 or c < ema10:
                    return {
                        "signal": "PUT", "rsi": round(float(rsi), 2), "atr": round(float(atr), 5), "adx": round(float(adx), 2),
                        "reason": f"Тренд ВНИЗ (ADX: {round(adx, 1)}) + Адаптивний RSI/Імпульс",
                        "global_trend": global_trend, "mid_trend": mid_trend, "local_trend": local_trend
                    }

        return default
