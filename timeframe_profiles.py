# -*- coding: utf-8 -*-
"""
Профілі таймфреймів: кожен профіль — окремий НЕЗАЛЕЖНИЙ потік сигналів зі своїм
робочим таймфреймом, підтверджуючим ТФ, діапазоном експірації та параметрами якості.

Ідея: замість того щоб тримати кількість сигналів «послабленням» фільтрів на одному ТФ,
ми множимо кількість потоків (пара × ТФ × експірація) і при цьому тримаємо
у кожному потоці суворіші умови входу (закрита свічка, мін. 2 підтвердження).
"""

PROFILES = {
    "1m": {
        "minutes": 1,
        "primary": "1m",
        "confirm": "3m",
        "mid": "15m",
        "base_exp": 2,
        "exp_range": (1, 3),
        "min_confirmations": 2,
        "ml_min_auto": 53.0,
        "ml_min_manual": 44.0,
    },
    "3m": {
        "minutes": 3,
        "primary": "3m",
        "confirm": "5m",
        "mid": "15m",
        "base_exp": 3,
        "exp_range": (3, 6),
        "min_confirmations": 2,
        "ml_min_auto": 53.0,
        "ml_min_manual": 44.0,
    },
    "5m": {
        "minutes": 5,
        "primary": "5m",
        "confirm": "15m",
        "mid": "1h",
        "base_exp": 5,
        "exp_range": (5, 15),
        "min_confirmations": 2,
        "ml_min_auto": 53.0,
        "ml_min_manual": 44.0,
    },
    "15m": {
        "minutes": 15,
        "primary": "15m",
        "confirm": "1h",
        "mid": "1h",
        "base_exp": 15,
        "exp_range": (15, 30),
        "min_confirmations": 2,
        "ml_min_auto": 53.0,
        "ml_min_manual": 44.0,
    },
}

# Які профілі активні (можна вимикати/вмикати без зміни коду)
ACTIVE_PROFILES = ["1m", "3m", "5m", "15m"]

# Стратегії, що працюють на відбій (mean reversion) — для них інші правила підтверджень
REVERSAL_STRATEGIES = {"RANGE_CORRIDOR", "PRICE_LEVELS"}

TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "1d": 1440}
