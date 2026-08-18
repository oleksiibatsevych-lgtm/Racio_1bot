import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "trading_model.pkl"
DB_NAME = "trading_stats.db"

class TradingMLFilter:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
            except:
                self.model = None

    def train_model(self):
        if not os.path.exists(DB_NAME):
            return False, "База даних відсутня."

        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(
            "SELECT rsi, adx, bb_width, result FROM signals WHERE status = 'COMPLETED'", conn
        )
        conn.close()

        if len(df) < 30:
            return False, f"⚠️ Недостатньо даних для навчання (накопичено: {len(df)}/30)"

        df['target'] = df['result'].apply(lambda x: 1 if x == 'WIN' else 0)
        df = df.dropna()
        if len(df) < 20:
            return False, "⚠️ Замало валідних даних після очищення."

        X = df[['rsi', 'adx', 'bb_width']]
        y = df['target']

        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        self.model.fit(X, y)

        joblib.dump(self.model, MODEL_FILE)
        return True, f"✅ Модель успішно навчено на {len(df)} угодах!"

    def predict_signal_probability(self, rsi, adx, bb_width) -> float:
        if self.model is None:
            return 1.0  
        try:
            X_new = pd.DataFrame([[rsi, adx, bb_width]], columns=['rsi', 'adx', 'bb_width'])
            proba = self.model.predict_proba(X_new)[0][1]
            return float(proba)
        except Exception as e:
            print(f"Помилка предикту ML: {e}")
            return 1.0

    def generate_strategy_report(self) -> str:
        if self.model is None:
            if os.path.exists(MODEL_FILE):
                self.load_model()
            else:
                return "⚠️ Модель ще не навчена. Потрібно мінімум 30 завершених угод."

        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(
            "SELECT rsi, adx, bb_width, result FROM signals WHERE status = 'COMPLETED'", conn
        )
        conn.close()

        if len(df) < 30:
            return f"📊 Статистики замало для аналізу. Потрібно мінімум 30 завершених угод (є: {len(df)})."

        importances = self.model.feature_importances_
        features = ['RSI', 'ADX (Сила тренду)', 'Ширина Боллінджера']
        sorted_idx = np.argsort(importances)[::-1]

        df['is_win'] = df['result'].apply(lambda x: 1 if x == 'WIN' else 0)
        avg_loss_adx = df[df['is_win'] == 0]['adx'].mean()
        avg_win_adx = df[df['is_win'] == 1]['adx'].mean()
        
        avg_loss_rsi = df[df['is_win'] == 0]['rsi'].mean()
        avg_win_rsi = df[df['is_win'] == 1]['rsi'].mean()

        report = (
            f"🧠 **Аналітичний звіт та поради від ШІ-моделі:**\n\n"
            f"📌 **Вплив індикаторів на результат угод:**\n"
            f"1. {features[sorted_idx[0]]} (Важливість: {round(importances[sorted_idx[0]] * 100, 1)}%)\n"
            f"2. {features[sorted_idx[1]]} (Важливість: {round(importances[sorted_idx[1]] * 100, 1)}%)\n"
            f"3. {features[sorted_idx[2]]} (Важливість: {round(importances[sorted_idx[2]] * 100, 1)}%)\n\n"
            f"🔍 **Порівняльний аналіз (WIN vs LOSS):**\n"
            f"• Середній ADX у виграшних: **{round(avg_win_adx, 1)}** | у збиткових: **{round(avg_loss_adx, 1)}**\n"
            f"• Середній RSI у виграшних: **{round(avg_win_rsi, 1)}** | у збиткових: **{round(avg_loss_rsi, 1)}**\n\n"
            f"💡 **Рекомендація з коригування стратегії:**\n"
        )

        if avg_loss_adx < avg_win_adx:
            report += "• Збиткові угоди частіше трапляються при низькому ADX. Варто підняти мінімальний поріг ADX для трендових входів."
        else:
            report += "• Ринковий шум та екстремальні значення RSI часто призводять до мінусів. Варто посилити фільтрацію."

        return report
