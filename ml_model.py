# -*- coding: utf-8 -*-
import os
import pickle
import logging

import numpy as np
import pandas as pd

import database

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # xgboost відсутній — використаємо sklearn
    XGBClassifier = None
    HAS_XGB = False

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

FEATURE_NAMES = [
    "rsi", "adx", "bb_width", "session_code", "hour", "div_encoded", "dist_pivot",
    "volatility_ratio", "wick_ratio", "ema_dist", "tf_minutes", "confirmations",
]
MODEL_VERSION = 3
MIN_TRAIN_ROWS = 60       # менше — модель не навчається, фільтр пропускає все
MIN_TEST_AUC = 0.53       # модель вмикається лише якщо на відкладеній вибірці є реальна перевага


def _new_estimator():
    if HAS_XGB:
        return XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, random_state=42,
                             eval_metric="logloss")
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120,
                                          min_samples_leaf=20, random_state=42)


class TradingMLFilter:
    """
    Чесний ML-фільтр:
      * ознаки логуються в БД при збереженні сигналу (див. bot.save_signal);
      * навчання лише на рядках, де ознаки реально записані (bb_width IS NOT NULL);
      * time-split валідація: остання 25% угод — тест. Модель активується лише при AUC >= 0.53;
      * поки моделі немає — predict_proba повертає None-подібне нейтральне значення (фільтр не блокує).
    """

    def __init__(self, model_path="ml_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.meta = {"trained": False, "auc": None, "rows": 0, "version": MODEL_VERSION}
        self._load()

    # ---- збереження ----
    def _load(self):
        if not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, "rb") as f:
                blob = pickle.load(f)
            if isinstance(blob, dict) and blob.get("version") == MODEL_VERSION:
                self.model, self.meta = blob["model"], blob["meta"]
        except Exception as e:
            logger.warning(f"ML: не вдалося завантажити модель: {e}")

    def load_model(self):  # зворотна сумісність
        self._load()
        return self.model

    def save_model(self):
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump({"version": MODEL_VERSION, "model": self.model, "meta": self.meta}, f)
        except Exception as e:
            logger.warning(f"ML: не вдалося зберегти модель: {e}")

    @property
    def is_active(self) -> bool:
        return self.model is not None

    # ---- ознаки ----
    @staticmethod
    def vectorize(f: dict) -> np.ndarray:
        div = f.get("divergence", "NONE")
        div_enc = 1 if div and str(div) != "NONE" else 0
        return np.array([[
            float(f.get("rsi", 50)), float(f.get("adx", 20)), float(f.get("bb_width", 0.001)),
            int(f.get("session_code", 1)), int(f.get("hour", 12)), div_enc,
            float(f.get("dist_pivot", 0.0)), float(f.get("volatility_ratio", 1.0)),
            float(f.get("wick_ratio", 0.0)), float(f.get("ema_dist", 0.0)),
            float(f.get("tf_minutes", 1)), float(f.get("confirmations", 0)),
        ]], dtype=float)

    def extract_features(self, rsi, adx, bb_width, session_code=1, hour=12, divergence="NONE", dist_pivot=0.0,
                         volatility_ratio=1.0, wick_ratio=0.0, ema_dist=0.0, tf_minutes=1, confirmations=0):
        return self.vectorize(dict(rsi=rsi, adx=adx, bb_width=bb_width, session_code=session_code, hour=hour,
                                   divergence=divergence, dist_pivot=dist_pivot, volatility_ratio=volatility_ratio,
                                   wick_ratio=wick_ratio, ema_dist=ema_dist, tf_minutes=tf_minutes,
                                   confirmations=confirmations))

    # ---- прогноз ----
    def predict_proba(self, features: dict) -> float:
        """
        Повертає ймовірність WIN у відсотках. Якщо моделі ще немає (мало даних / слабка валідація) —
        нейтральні 55.0: фільтр не блокує, але й не вдає, що має інформацію.
        """
        if self.model is None or not isinstance(features, dict):
            return 55.0
        try:
            p = float(self.model.predict_proba(self.vectorize(features))[0][1])
            return round(p * 100.0, 1)
        except Exception as e:
            logger.warning(f"ML: помилка прогнозу: {e}")
            return 55.0

    def predict_signal_probability(self, **kw):  # зворотна сумісність
        return self.predict_proba(kw) / 100.0

    def evaluate_ml_signal(self, features_dict: dict, htf_trend_aligned: bool = False) -> dict:
        prob = self.predict_proba(features_dict)
        required = 54.0 if htf_trend_aligned else 58.0
        if not self.is_active:
            return {"should_trade": True, "probability": prob, "threshold": 0.0}
        return {"should_trade": prob >= required, "probability": round(prob, 2), "threshold": required}

    # ---- навчання ----
    def train_model(self):
        conn = None
        try:
            conn = database.get_connection()
            query = """
                SELECT rsi, adx, bb_width,
                       COALESCE(session_code, 1) AS session_code,
                       COALESCE(hour, 12) AS hour,
                       COALESCE(divergence, 'NONE') AS divergence,
                       COALESCE(dist_pivot, 0.0) AS dist_pivot,
                       COALESCE(volatility_ratio, 1.0) AS volatility_ratio,
                       COALESCE(wick_ratio, 0.0) AS wick_ratio,
                       COALESCE(ema_dist, 0.0) AS ema_dist,
                       COALESCE(NULLIF(regexp_replace(primary_tf, '[^0-9]', '', 'g'), '')::int, 1) AS tf_minutes,
                       COALESCE(score, 0) AS confirmations,
                       result
                FROM signals
                WHERE result IN ('WIN', 'LOSS') AND bb_width IS NOT NULL
                ORDER BY id ASC
            """
            df = pd.read_sql(query, conn)
        except Exception as e:
            return False, f"❌ Помилка завантаження даних з БД: {e}"
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
        return self.train_from_frame(df)

    def train_from_frame(self, df: pd.DataFrame):
        try:
            if len(df) < MIN_TRAIN_ROWS:
                return False, (f"⚠️ Замало угод із записаними ознаками ({len(df)}/{MIN_TRAIN_ROWS}). "
                               f"Фільтр поки пропускає все, дані накопичуються.")
            df = df.copy()
            df["div_encoded"] = df["divergence"].apply(lambda x: 1 if x and str(x) != "NONE" else 0)
            X = df[[c for c in FEATURE_NAMES]].astype(float).fillna(0.0).values
            y = (df["result"] == "WIN").astype(int).values

            split = int(len(df) * 0.75)
            X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                return False, "⚠️ У вибірці лише один клас результатів — навчання неможливе."

            est = _new_estimator()
            est.fit(X_tr, y_tr)
            auc = float(roc_auc_score(y_te, est.predict_proba(X_te)[:, 1]))

            if auc < MIN_TEST_AUC:
                self.model = None
                self.meta = {"trained": False, "auc": round(auc, 3), "rows": len(df), "version": MODEL_VERSION}
                return False, (f"⚠️ Модель не пройшла валідацію (AUC на тесті {auc:.3f} < {MIN_TEST_AUC}). "
                               f"Фільтр лишається нейтральним — це чесніше, ніж вдавати прогноз.")

            final = _new_estimator()
            final.fit(X, y)
            self.model = final
            self.meta = {"trained": True, "auc": round(auc, 3), "rows": len(df), "version": MODEL_VERSION}
            self.save_model()
            backend = "XGBoost" if HAS_XGB else "sklearn-GB"
            return True, f"✅ {backend} навчено на {len(df)} угодах. AUC на відкладеній вибірці: {auc:.3f}"
        except Exception as e:
            return False, f"❌ Помилка навчання моделі: {e}"

    def status_line(self) -> str:
        if self.is_active:
            return f"ML активна (AUC {self.meta.get('auc')}, угод {self.meta.get('rows')})"
        return "ML нейтральна (недостатньо даних/слабка валідація)"


def evaluate_ml_signal(model_obj, features_dict: dict, htf_trend_aligned: bool = False) -> dict:
    if isinstance(model_obj, TradingMLFilter):
        return model_obj.evaluate_ml_signal(features_dict, htf_trend_aligned)
    return {"should_trade": True, "probability": 55.0, "threshold": 0.0}
