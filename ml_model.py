import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import database

class TradingMLFilter:
    def __init__(self, model_path="ml_model.pkl"):
        self.model_path = model_path
        self.model = self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def save_model(self):
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
        except Exception:
            pass

    def extract_features(self, rsi, adx, bb_width, session_code=1, hour=12, divergence="NONE", dist_pivot=0.0,
                         volatility_ratio=1.0, wick_ratio=0.0, ema_dist=0.0):
        div_encoded = 1 if divergence and str(divergence) != "NONE" else 0
        return [[
            float(rsi), float(adx), float(bb_width), int(session_code), int(hour), 
            int(div_encoded), float(dist_pivot), float(volatility_ratio), float(wick_ratio), float(ema_dist)
        ]]

    def predict_signal_probability(self, rsi, adx, bb_width, session_code=1, hour=12, divergence="NONE", dist_pivot=0.0,
                                   volatility_ratio=1.0, wick_ratio=0.0, ema_dist=0.0):
        if self.model is None:
            base = 0.58
            if rsi < 35 or rsi > 65: base += 0.06
            if adx > 25: base += 0.05
            if divergence and str(divergence) != "NONE": base += 0.08
            if wick_ratio >= 0.40: base += 0.05
            return min(round(base, 2), 0.95)
        try:
            X = self.extract_features(rsi, adx, bb_width, session_code, hour, divergence, dist_pivot, volatility_ratio, wick_ratio, ema_dist)
            proba = self.model.predict_proba(X)[0][1]
            return float(proba)
        except Exception:
            return 0.62

    def train_model(self):
        try:
            with database.get_connection() as conn:
                query = """
                    SELECT rsi, adx, bb_width, 
                           COALESCE(session_code, 1) as session_code, 
                           COALESCE(hour, 12) as hour, 
                           COALESCE(divergence, 'NONE') as divergence, 
                           COALESCE(dist_pivot, 0.0) as dist_pivot,
                           COALESCE(volatility_ratio, 1.0) as volatility_ratio,
                           COALESCE(wick_ratio, 0.0) as wick_ratio,
                           COALESCE(ema_dist, 0.0) as ema_dist,
                           result 
                    FROM signals 
                    WHERE result IS NOT NULL AND result != 'NEUTRAL'
                """
                df = pd.read_sql(query, conn)

            if len(df) < 10:
                return False, "⚠️ Замало завершених угод для навчання ШІ (мінімум 10)."

            df['target'] = df['result'].apply(lambda x: 1 if x == 'WIN' else 0)
            df['div_encoded'] = df['divergence'].apply(lambda x: 1 if x and str(x) != "NONE" else 0)

            X = np.column_stack([
                df['rsi'].fillna(50).astype(float),
                df['adx'].fillna(20).astype(float),
                df['bb_width'].fillna(0.001).astype(float),
                df['session_code'].fillna(1).astype(int),
                df['hour'].fillna(12).astype(int),
                df['div_encoded'].astype(int),
                df['dist_pivot'].fillna(0.0).astype(float),
                df['volatility_ratio'].fillna(1.0).astype(float),
                df['wick_ratio'].fillna(0.0).astype(float),
                df['ema_dist'].fillna(0.0).astype(float)
            ])
            y = df['target'].values

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.save_model()
            return True, f"✅ ШІ успішно перенавчено на реальній базі з {len(df)} угод!"
        except Exception as e:
            return False, f"❌ Помилка навчання моделі: {e}"
