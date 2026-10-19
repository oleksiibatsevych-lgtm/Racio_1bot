# -*- coding: utf-8 -*-
import os
import json

PERFORMANCE_CACHE_FILE = "pair_strategy_performance.json"

def get_strategy_adaptive_weight(pair_name: str, strategy_name: str) -> float:
    """
    Повертає динамічний ваговий коефіцієнт стратегії на основі історії бектесту/успіху.
    Дозволяє приглушувати збиткові сетапи і підсилювати прибуткові.
    """
    if os.path.exists(PERFORMANCE_CACHE_FILE):
        try:
            with open(PERFORMANCE_CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
                if pair_name in cache and strategy_name in cache[pair_name]:
                    wr = cache[pair_name][strategy_name].get("wr", 50.0)
                    if wr >= 75.0:
                        return 1.35
                    elif wr >= 60.0:
                        return 1.15
                    elif wr < 45.0:
                        return 0.4
        except Exception:
            pass
    return 1.0

def calculate_smart_expiration(base_exp: int, regime: str, volatility_ratio: float) -> int:
    """
    Розраховує точний час експірації від 1 до 60 хвилин залежно від ринкового режиму.
    """
    exp = base_exp
    if regime == "STRONG_TREND":
        exp = int(exp * 1.25)
    elif regime == "SQUEEZE":
        exp = int(exp * 0.75)
    elif regime == "RANGE_BOUND":
        exp = int(exp * 1.0)
    elif regime == "TREND_EXHAUSTION":
        exp = int(exp * 1.1)
    elif regime == "CHAOS":
        exp = 3

    if volatility_ratio > 1.4:
        exp = int(exp * 0.85)
    elif volatility_ratio < 0.7:
        exp = int(exp * 1.2)

    return max(1, min(60, int(exp)))
