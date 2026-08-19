import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

MODEL_FILE = "trading_model.pkl"
DB_NAME = "trading_stats.db"

class TradingMLFilter:
    def __init__(self):
        self.model = None
        self.init_db()  # Автоматичне створення таблиці з усіма необхідними полями
        self.load_model()

    def init_db(self):
        """Створює таблицю signals з полями для автоперевірки, якщо її ще немає"""
        conn = sqlite3.connect(DB_NAME)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                pair TEXT,
                action TEXT,
                entry_price REAL,
                duration_min INTEGER,
                rsi REAL,
                adx REAL,
                bb_width REAL,
                result TEXT,
                status TEXT
            )
        """)
        conn.commit()
        conn.close()

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
            except:
                self.model = None

    def get_data_for_training(self):
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query(
            "SELECT rsi, adx, bb_width, timestamp, result FROM signals WHERE status = 'COMPLETED'", conn
        )
        conn.close()
        return df

    def train_model(self):
        df = self.get_data_for_training()
        if df.empty or 'timestamp' not in df.columns:
            return False, "База даних порожня або не містить необхідних даних."
        
        # Витягуємо годину з timestamp для аналізу часових закономірностей
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['target'] = df['result'].apply(lambda x: 1 if x == 'WIN' else 0)
        df = df.dropna()

        # Суворий поріг: 200 угод для об'єктивної статистики
        if len(df) < 200:
            return False, f"⚠️ Недостатньо даних для об'єктивного навчання (накопичено: {len(df)}/200)"

        X = df[['rsi', 'adx', 'bb_width', 'hour']]
        y = df['target']

        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
        self.model.fit(X, y)

        joblib.dump(self.model, MODEL_FILE)
        return True, f"✅ Модель успішно перенавчена на {len(df)} угодах з урахуванням часу!"

    def predict_signal_probability(self, rsi, adx, bb_width) -> float:
        conn = sqlite3.connect(DB_NAME)
        count = conn.execute("SELECT count(*) FROM signals WHERE status = 'COMPLETED'").fetchone()[0]
        conn.close()
        
        # Якщо угод менше 200 — вимикаємо фільтрацію (пропускаємо все для збору бази)
        if count < 200 or self.model is None:
            return 1.0  
            
        try:
            current_hour = datetime.now().hour
            X_new = pd.DataFrame([[rsi, adx, bb_width, current_hour]], 
                                 columns=['rsi', 'adx', 'bb_width', 'hour'])
            proba = self.model.predict_proba(X_new)[0][1]
            return float(proba)
        except Exception as e:
            print(f"Помилка предикту ML: {e}")
            return 1.0

    def generate_strategy_report(self) -> str:
        conn = sqlite3.connect(DB_NAME)
        count = conn.execute("SELECT count(*) FROM signals WHERE status = 'COMPLETED'").fetchone()[0]
        conn.close()

        if count < 200 or self.model is None:
            return f"📊 Статистики замало для аналізу. Потрібно мінімум 200 завершених угод (накопичено: {count}/200)."

        df = self.get_data_for_training()
        if len(df) < 200:
            return f"📊 Статистики замало для аналізу (є: {len(df)}/200)."

        try:
            importances = self.model.feature_importances_
            features = ['RSI', 'ADX (Сила тренду)', 'Ширина Боллінджера', 'Година доби']
            sorted_idx = np.argsort(importances)[::-1]

            df['is_win'] = df['result'].apply(lambda x: 1 if x == 'WIN' else 0)
            avg_loss_adx = df[df['is_win'] == 0]['adx'].mean()
            avg_win_adx = df[df['is_win'] == 1]['adx'].mean()
            
            avg_loss_rsi = df[df['is_win'] == 0]['rsi'].mean()
            avg_win_rsi = df[df['is_win'] == 1]['rsi'].mean()

            report = (
                f"🧠 **Аналітичний звіт та поради від ШІ-моделі (200+ угод):**\n\n"
                f"📌 **Вплив факторів на результат угод:**\n"
                f"1. {features[sorted_idx[0]]} (Важливість: {round(importances[sorted_idx[0]] * 100, 1)}%)\n"
                f"2. {features[sorted_idx[1]]} (Важливість: {round(importances[sorted_idx[1]] * 100, 1)}%)\n"
                f"3. {features[sorted_idx[2]]} (Важливість: {round(importances[sorted_idx[2]] * 100, 1)}%)\n"
                f"4. {features[sorted_idx[3]]} (Важливість: {round(importances[sorted_idx[3]] * 100, 1)}%)\n\n"
                f"🔍 **Порівняльний аналіз (WIN vs LOSS):**\n"
                f"• Середній ADX у виграшних: **{round(avg_win_adx, 1)}** | у збиткових: **{round(avg_loss_adx, 1)}**\n"
                f"• Середній RSI у виграшних: **{round(avg_win_rsi, 1)}** | у збиткових: **{round(avg_loss_rsi, 1)}**\n\n"
                f"💡 **Рекомендація з коригування стратегії:**\n"
            )

            if avg_loss_adx < avg_win_adx:
                report += "• Збиткові угоди частіше трапляються при низькому ADX. Варто підняти мінімальний поріг ADX."
            else:
                report += "• Ринковий шум та екстремальні значення RSI впливають на мінуси. Враховуйте часові сесії."

            return report
        except Exception as e:
            return f"❌ Помилка генерації звіту: {e}"
