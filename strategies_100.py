# -*- coding: utf-8 -*-
"""100 унікальних суворо каузальних стратегій (7 категорій) для єдиного реєстру Racio_1bot."""
import math
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

EXTENDED_CATEGORIES = {
    "QUANT": "🧬 Квантові та Мікроструктура",
}

EXTENDED_META = {
    # --- MEAN (20) ---
    "MR_VWAP_Z_5M": {"title": "VWAP Z-Score 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 88, "weight": 1.15},
    "MR_VWAP_Z_15M": {"title": "VWAP Z-Score 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "RANGE", "priority": 90, "weight": 1.20},
    "MR_KAMA_BND_5M": {"title": "KAMA Band 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 87, "weight": 1.15},
    "MR_KAMA_BND_15M": {"title": "KAMA Band 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "RANGE", "priority": 89, "weight": 1.20},
    "MR_FISH_5M": {"title": "Fisher Transform 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 10, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.15},
    "MR_FISH_15M": {"title": "Fisher Transform 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "RANGE", "priority": 88, "weight": 1.20},
    "MR_INV_FISH_5M": {"title": "Inv Fisher RSI 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 85, "weight": 1.10},
    "MR_PCTB_5M": {"title": "BB %B Squeeze 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.10},
    "MR_Z30_15M": {"title": "Z-Score 30 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "RANGE", "priority": 89, "weight": 1.20},
    "MR_Z50_5M": {"title": "Z-Score 50 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 85, "weight": 1.10},
    "MR_SMTH_5M": {"title": "SuperSmoother 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.15},
    "MR_LAG_RSI_5M": {"title": "Laguerre RSI 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 88, "weight": 1.15},
    "MR_LAG_RSI_15M": {"title": "Laguerre RSI 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "RANGE", "priority": 90, "weight": 1.20},
    "MR_CAM_H3L3_5M": {"title": "Camarilla L3/H3 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 87, "weight": 1.15},
    "MR_CAM_H4L4_15M": {"title": "Camarilla L4/H4 15m", "category": "MEAN", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "RANGE", "priority": 89, "weight": 1.20},
    "MR_CMO_5M": {"title": "Chande CMO 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 84, "weight": 1.05},
    "MR_RSI2_MA_5M": {"title": "Connors RSI2 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 10, "htf": None, "regime": "RANGE", "priority": 85, "weight": 1.10},
    "MR_ENV_ATR_5M": {"title": "EMA ATR Env 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.10},
    "MR_RVI_REV_5M": {"title": "RVI Hook 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 84, "weight": 1.05},
    "MR_MID_CH_5M": {"title": "Donchian40 Fade 5m", "category": "MEAN", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 85, "weight": 1.10},
    # --- TREND (18) ---
    "TR_HOLY_5M": {"title": "Holy Grail 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 90, "weight": 1.20},
    "TR_HOLY_15M": {"title": "Holy Grail 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 92, "weight": 1.25},
    "TR_KAMA_PB_5M": {"title": "KAMA Pullback 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 88, "weight": 1.15},
    "TR_KAMA_PB_15M": {"title": "KAMA Pullback 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 90, "weight": 1.20},
    "TR_CHAND_15M": {"title": "Chandelier 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": 60, "regime": "TREND", "priority": 89, "weight": 1.20},
    "TR_DON_MID_5M": {"title": "Donchian Mid PB 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 86, "weight": 1.10},
    "TR_HMA_5M": {"title": "Hull MA Flip 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": 60, "regime": "TREND", "priority": 87, "weight": 1.15},
    "TR_HMA_15M": {"title": "Hull MA Flip 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": 60, "regime": "TREND", "priority": 90, "weight": 1.20},
    "TR_VORTEX_5M": {"title": "Vortex ADX 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 86, "weight": 1.15},
    "TR_VORTEX_15M": {"title": "Vortex ADX 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 89, "weight": 1.20},
    "TR_STC_5M": {"title": "Schaff Cycle 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 87, "weight": 1.15},
    "TR_STC_15M": {"title": "Schaff Cycle 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 90, "weight": 1.20},
    "TR_ELDER_5M": {"title": "Elder Impulse 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 86, "weight": 1.10},
    "TR_ELDER_3M": {"title": "Triple Screen 3m", "category": "TREND", "tf": "3m", "tf_min": 3, "exp": 9, "htf": 15, "regime": "TREND", "priority": 83, "weight": 1.05},
    "TR_STOCHR_5M": {"title": "StochRSI HTF 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": 60, "regime": "TREND", "priority": 88, "weight": 1.15},
    "TR_AROON_15M": {"title": "Aroon Cross 15m", "category": "TREND", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "TREND", "priority": 87, "weight": 1.15},
    "TR_VIDYA_5M": {"title": "VIDYA Pullback 5m", "category": "TREND", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 86, "weight": 1.10},
    "TR_123_PB_3M": {"title": "1-2-3 Pullback 3m", "category": "TREND", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "TREND", "priority": 82, "weight": 1.05},
    # --- VOL (15) ---
    "VL_ASIA_SWP_5M": {"title": "Asia Sweep 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 92, "weight": 1.25},
    "VL_NY_SWP_5M": {"title": "London Sweep NY 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 91, "weight": 1.20},
    "VL_IMP_EXH_5M": {"title": "Impulse Exhaust 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 10, "htf": None, "regime": "PA", "priority": 88, "weight": 1.15},
    "VL_IMP_EXH_15M": {"title": "Impulse Exhaust 15m", "category": "VOL", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "PA", "priority": 90, "weight": 1.20},
    "VL_NR4_BRK_5M": {"title": "NR4 Breakout 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 85, "weight": 1.10},
    "VL_NR7_BRK_15M": {"title": "NR7 Breakout 15m", "category": "VOL", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "BREAKOUT", "priority": 88, "weight": 1.15},
    "VL_ID_NR4_5M": {"title": "ID+NR4 Break 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 87, "weight": 1.15},
    "VL_BB_EXP_5M": {"title": "BB Expansion 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 86, "weight": 1.10},
    "VL_ATR_EXP_3M": {"title": "ATR Surge 3m", "category": "VOL", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "BREAKOUT", "priority": 81, "weight": 1.00},
    "VL_GAP_FILL_5M": {"title": "Micro Gap Fade 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 84, "weight": 1.05},
    "VL_ORB_NY30_5M": {"title": "ORB NY 30m 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 90, "weight": 1.20},
    "VL_ORB_LN30_5M": {"title": "ORB Lon 30m 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 90, "weight": 1.20},
    "VL_CLIMAX_3M": {"title": "Tick Climax 3m", "category": "VOL", "tf": "3m", "tf_min": 3, "exp": 6, "htf": None, "regime": "PA", "priority": 82, "weight": 1.05},
    "VL_CLIMAX_5M": {"title": "Tick Climax 5m", "category": "VOL", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 87, "weight": 1.15},
    "VL_KCP_BRK_15M": {"title": "Keltner Break 15m", "category": "VOL", "tf": "15m", "tf_min": 15, "exp": 30, "htf": 60, "regime": "BREAKOUT", "priority": 88, "weight": 1.15},
    # --- DIV (15) ---
    "DV_OBV_5M": {"title": "OBV Divergence 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 88, "weight": 1.15},
    "DV_OBV_15M": {"title": "OBV Divergence 15m", "category": "DIV", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "PA", "priority": 90, "weight": 1.20},
    "DV_WPR_HK_5M": {"title": "WPR Hook 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 10, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.10},
    "DV_WPR_DIV_5M": {"title": "WPR Div 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 88, "weight": 1.15},
    "DV_CCI_DIV_5M": {"title": "CCI Div 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 89, "weight": 1.15},
    "DV_CCI_DIV_15M": {"title": "CCI Div 15m", "category": "DIV", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "PA", "priority": 91, "weight": 1.20},
    "DV_STOCH_DIV_5M": {"title": "Stoch Div 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 88, "weight": 1.15},
    "DV_MFI_DIV_5M": {"title": "MFI Div 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 87, "weight": 1.15},
    "DV_FORCE_5M": {"title": "Force Index 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 85, "weight": 1.10},
    "DV_TRIX_5M": {"title": "TRIX Cross 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 85, "weight": 1.10},
    "DV_PPO_DIV_15M": {"title": "PPO Div 15m", "category": "DIV", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "PA", "priority": 89, "weight": 1.20},
    "DV_UO_5M": {"title": "Ultimate Osc 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.10},
    "DV_CMO_DIV_5M": {"title": "CMO Div 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 87, "weight": 1.15},
    "DV_RSI7_DIV_3M": {"title": "RSI7 Div 3m", "category": "DIV", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "PA", "priority": 82, "weight": 1.05},
    "DV_DPO_REV_5M": {"title": "Detrended DPO 5m", "category": "DIV", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 84, "weight": 1.05},
    # --- SMC (12) ---
    "SM_TURTLE_5M": {"title": "Turtle Soup 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "SMC", "priority": 90, "weight": 1.20},
    "SM_TURTLE_15M": {"title": "Turtle Soup 15m", "category": "SMC", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "SMC", "priority": 92, "weight": 1.25},
    "SM_PDHL_15M": {"title": "PDH/PDL Reject 15m", "category": "SMC", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "SMC", "priority": 93, "weight": 1.25},
    "SM_PDHL_5M": {"title": "PDH/PDL Reject 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "SMC", "priority": 91, "weight": 1.20},
    "SM_OB_RET_5M": {"title": "Order Block 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "SMC", "priority": 89, "weight": 1.20},
    "SM_OB_RET_15M": {"title": "Order Block 15m", "category": "SMC", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "SMC", "priority": 91, "weight": 1.25},
    "SM_CHOCH_5M": {"title": "CHoCH Retest 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "SMC", "priority": 88, "weight": 1.15},
    "SM_EQHL_5M": {"title": "EQH/EQL Sweep 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "SMC", "priority": 89, "weight": 1.15},
    "SM_STOP_HNT_3M": {"title": "Stop Hunt 3m", "category": "SMC", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "SMC", "priority": 84, "weight": 1.05},
    "SM_FVG_15M": {"title": "FVG Impulse 15m", "category": "SMC", "tf": "15m", "tf_min": 15, "exp": 30, "htf": 60, "regime": "SMC", "priority": 91, "weight": 1.25},
    "SM_ROUND_5M": {"title": "Round .00/.50 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 87, "weight": 1.15},
    "SM_PIV_S1R1_5M": {"title": "Pivot S1/R1 5m", "category": "SMC", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.10},
    # --- PA (10) ---
    "PA_OOPS_5M": {"title": "Williams Oops 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 86, "weight": 1.10},
    "PA_SMASH_5M": {"title": "Smash Day 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 87, "weight": 1.15},
    "PA_HARAMI_5M": {"title": "Harami BB 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 85, "weight": 1.10},
    "PA_TWEZ_5M": {"title": "Tweezers S/R 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 86, "weight": 1.10},
    "PA_MOR_EVE_5M": {"title": "Morning/Eve Star 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 87, "weight": 1.15},
    "PA_MARU_5M": {"title": "Marubozu Trend 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 84, "weight": 1.05},
    "PA_RAIL_5M": {"title": "Railroad Tracks 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 86, "weight": 1.10},
    "PA_KICK_5M": {"title": "Kicker Pattern 5m", "category": "PA", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "PA", "priority": 85, "weight": 1.10},
    "PA_PIN_15M": {"title": "Pin Bar BB 15m", "category": "PA", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "PA", "priority": 89, "weight": 1.20},
    "PA_3BR_REV_3M": {"title": "3-Bar Micro Rev 3m", "category": "PA", "tf": "3m", "tf_min": 3, "exp": 9, "htf": None, "regime": "PA", "priority": 81, "weight": 1.00},
    # --- QUANT (10) ---
    "QN_REG_SW_5M": {"title": "ER Regime Switch 5m", "category": "QUANT", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "ANY", "priority": 90, "weight": 1.20},
    "QN_REG_SW_15M": {"title": "ER Regime Sw 15m", "category": "QUANT", "tf": "15m", "tf_min": 15, "exp": 30, "htf": None, "regime": "ANY", "priority": 92, "weight": 1.25},
    "QN_HURST_MR_5M": {"title": "Variance Ratio MR 5m", "category": "QUANT", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 88, "weight": 1.15},
    "QN_ENTROPY_5M": {"title": "Sign Entropy MR 5m", "category": "QUANT", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.10},
    "QN_SKEW_REV_5M": {"title": "Return Skew Rev 5m", "category": "QUANT", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 85, "weight": 1.10},
    "QN_KURT_EXH_5M": {"title": "Kurtosis Tail 5m", "category": "QUANT", "tf": "5m", "tf_min": 5, "exp": 10, "htf": None, "regime": "RANGE", "priority": 86, "weight": 1.15},
    "QN_BODY_EFF_5M": {"title": "Body Efficiency 5m", "category": "QUANT", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "BREAKOUT", "priority": 85, "weight": 1.10},
    "QN_TICK_SPD_3M": {"title": "Tick Surge 3m", "category": "QUANT", "tf": "3m", "tf_min": 3, "exp": 6, "htf": None, "regime": "PA", "priority": 82, "weight": 1.05},
    "QN_VWAP_TR_5M": {"title": "VWAP Trend PB 5m", "category": "QUANT", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "TREND", "priority": 87, "weight": 1.15},
    "QN_MULT_Z_5M": {"title": "Multi-Z Blend 5m", "category": "QUANT", "tf": "5m", "tf_min": 5, "exp": 15, "htf": None, "regime": "RANGE", "priority": 89, "weight": 1.20},
}


def _prev(a, k=1):
    out = np.full_like(a, np.nan, dtype=float)
    if len(a) > k:
        out[k:] = a[:-k]
    return out


def _sig(call, put):
    c, p = np.asarray(call, dtype=bool), np.asarray(put, dtype=bool)
    out = np.zeros(len(c), dtype=np.int8)
    out[c] = 1
    out[p & ~c] = -1
    return out


def _wma(a, n):
    n = max(1, int(n))
    out = np.full(len(a), np.nan, dtype=float)
    if len(a) < n:
        return out
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    out[n - 1:] = np.einsum("ij,j->i", sliding_window_view(a, n), w)
    return out


def _hma(a, n):
    n = max(2, int(n))
    raw = 2.0 * _wma(a, max(1, n // 2)) - _wma(a, n)
    sq = max(1, int(round(math.sqrt(n))))
    out = np.full(len(a), np.nan, dtype=float)
    start = n - 1
    if len(a) - start >= sq:
        out[start:] = _wma(raw[start:], sq)
    return out


def _rsi(c, n):
    d = pd.Series(c).diff()
    up = d.clip(lower=0.0).ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    dn = (-d.clip(upper=0.0)).ewm(alpha=1.0 / n, min_periods=n, adjust=False).mean()
    out = np.where((up + dn) > 1e-12, 100.0 * up / (up + dn + 1e-12), 50.0)
    out[:n] = np.nan
    return out


def _ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().to_numpy(float)


def _sma(a, n):
    return pd.Series(a).rolling(n, min_periods=n).mean().to_numpy(float)


def _std(a, n):
    return pd.Series(a).rolling(n, min_periods=n).std(ddof=0).to_numpy(float)


def _cmo(c, n=14):
    d = pd.Series(c).diff()
    up = d.clip(lower=0.0).rolling(n, min_periods=n).sum()
    dn = (-d.clip(upper=0.0)).rolling(n, min_periods=n).sum()
    return (100.0 * (up - dn) / (up + dn + 1e-12)).to_numpy(float)


def _er(c, n=10):
    s = pd.Series(c)
    return (s.diff(n).abs() / (s.diff().abs().rolling(n, min_periods=n).sum() + 1e-12)).to_numpy(float)


def _kama(c, n=10, fast=2, slow=30):
    er = _er(c, n)
    fsc, ssc = 2.0 / (fast + 1.0), 2.0 / (slow + 1.0)
    sc = (er * (fsc - ssc) + ssc) ** 2
    out = np.full(len(c), np.nan, dtype=float)
    if n < len(c):
        out[n] = c[n]
        for i in range(n + 1, len(c)):
            k = sc[i] if np.isfinite(sc[i]) else ssc ** 2
            out[i] = out[i - 1] + k * (c[i] - out[i - 1])
    return out


def _vidya(c, n=14):
    cmo = np.abs(_cmo(c, n)) / 100.0
    alpha = 2.0 / (n + 1.0)
    out = np.full(len(c), np.nan, dtype=float)
    if n < len(c):
        out[n] = c[n]
        for i in range(n + 1, len(c)):
            out[i] = out[i - 1] + alpha * (cmo[i] if np.isfinite(cmo[i]) else 0.2) * (c[i] - out[i - 1])
    return out


def _supersmoother(c, period=15):
    a1 = math.exp(-1.414 * math.pi / period)
    b1 = 2.0 * a1 * math.cos(1.414 * math.pi / period)
    c2, c3 = b1, -(a1 * a1)
    c1 = 1.0 - c2 - c3
    out = np.copy(c).astype(float)
    for i in range(2, len(c)):
        out[i] = c1 * 0.5 * (c[i] + c[i - 1]) + c2 * out[i - 1] + c3 * out[i - 2]
    return out


def _laguerre(c, gamma=0.7):
    n = len(c)
    l0 = np.zeros(n)
    l1 = np.zeros(n)
    l2 = np.zeros(n)
    l3 = np.zeros(n)
    out = np.full(n, 0.5, dtype=float)
    for i in range(1, n):
        l0[i] = (1.0 - gamma) * c[i] + gamma * l0[i - 1]
        l1[i] = -gamma * l0[i] + l0[i - 1] + gamma * l1[i - 1]
        l2[i] = -gamma * l1[i] + l1[i - 1] + gamma * l2[i - 1]
        l3[i] = -gamma * l2[i] + l2[i - 1] + gamma * l3[i - 1]
        cu = max(0.0, l0[i] - l1[i]) + max(0.0, l1[i] - l2[i]) + max(0.0, l2[i] - l3[i])
        cd = max(0.0, l1[i] - l0[i]) + max(0.0, l2[i] - l1[i]) + max(0.0, l3[i] - l2[i])
        out[i] = cu / (cu + cd) if (cu + cd) > 1e-12 else 0.5
    return out


def _fisher(h, l, n=10):
    hl2 = 0.5 * (h + l)
    lo = pd.Series(hl2).rolling(n, min_periods=n).min().to_numpy(float)
    hi = pd.Series(hl2).rolling(n, min_periods=n).max().to_numpy(float)
    raw = np.clip(2.0 * ((hl2 - lo) / (hi - lo + 1e-12) - 0.5), -0.999, 0.999)
    val = pd.Series(raw).ewm(alpha=0.33, adjust=False).mean().clip(-0.999, 0.999).to_numpy(float)
    ft = np.zeros(len(h), dtype=float)
    for i in range(1, len(h)):
        if np.isfinite(val[i]):
            ft[i] = 0.5 * math.log((1.0 + val[i]) / (1.0 - val[i])) + 0.5 * ft[i - 1]
    return ft


def _stc(c, fast=23, slow=50, cycle=10, factor=0.5):
    macd = pd.Series(c).ewm(span=fast, adjust=False).mean() - pd.Series(c).ewm(span=slow, adjust=False).mean()
    lo1, hi1 = macd.rolling(cycle, min_periods=cycle).min(), macd.rolling(cycle, min_periods=cycle).max()
    k1 = np.where((hi1 - lo1) > 1e-12, 100.0 * (macd - lo1) / (hi1 - lo1 + 1e-12), 50.0)
    d1 = np.zeros(len(c), dtype=float)
    for i in range(1, len(c)):
        v = k1[i] if np.isfinite(k1[i]) else d1[i - 1]
        d1[i] = d1[i - 1] + factor * (v - d1[i - 1])
    d1s = pd.Series(d1)
    lo2, hi2 = d1s.rolling(cycle, min_periods=cycle).min(), d1s.rolling(cycle, min_periods=cycle).max()
    k2 = np.where((hi2 - lo2) > 1e-12, 100.0 * (d1s - lo2) / (hi2 - lo2 + 1e-12), 50.0)
    stc = np.zeros(len(c), dtype=float)
    for i in range(1, len(c)):
        v = k2[i] if np.isfinite(k2[i]) else stc[i - 1]
        stc[i] = stc[i - 1] + factor * (v - stc[i - 1])
    return stc


def _prev_daily(idx, h, l, c, min_bars=6):
    tmp = pd.DataFrame({"h": h, "l": l, "c": c}, index=idx)
    daily = tmp.groupby(tmp.index.date).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), cnt=("c", "count"))
    daily = daily[daily["cnt"] >= min_bars].sort_index()
    if daily.empty:
        na = np.full(len(idx), np.nan)
        return na, na, na
    vd = np.array(daily.index, dtype="datetime64[D]")
    bd = np.array(pd.DatetimeIndex(idx).date, dtype="datetime64[D]")
    pos = np.searchsorted(vd, bd, side="left") - 1
    ok = pos >= 0
    pc = np.maximum(pos, 0)
    return (
        np.where(ok, daily["h"].to_numpy(float)[pc], np.nan),
        np.where(ok, daily["l"].to_numpy(float)[pc], np.nan),
        np.where(ok, daily["c"].to_numpy(float)[pc], np.nan),
    )


def _session_bounds(idx, h, l, sh, eh, tsh, teh, min_bars=3):
    tmp = pd.DataFrame({"h": h, "l": l}, index=idx)
    win = tmp[(tmp.index.hour >= sh) & (tmp.index.hour < eh)]
    if win.empty:
        na = np.full(len(idx), np.nan)
        return na, na
    grp = win.groupby(win.index.date).agg(rh=("h", "max"), rl=("l", "min"), cnt=("h", "count"))
    grp = grp[grp["cnt"] >= min_bars]
    dates = pd.Series(pd.DatetimeIndex(idx).date, index=idx)
    rh = dates.map(grp["rh"]).to_numpy(float)
    rl = dates.map(grp["rl"]).to_numpy(float)
    act = (pd.DatetimeIndex(idx).hour >= tsh) & (pd.DatetimeIndex(idx).hour < teh)
    return np.where(act, rh, np.nan), np.where(act, rl, np.nan)


def _div_swing(ind, osc, left=5, right=2, lo_max=None, hi_min=None):
    h, l, o, c = ind["h"], ind["l"], ind["o"], ind["c"]
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    lps, hps = [], []
    for i in range(left + right, n):
        p = i - right
        if np.isnan(osc[p]) or np.isnan(osc[i]) or np.isnan(osc[i - 1]):
            continue
        wl = l[p - left : p + right + 1]
        if l[p] == np.min(wl) and np.sum(wl == l[p]) == 1:
            lps.append((p, l[p], osc[p]))
            if len(lps) >= 2:
                pp, pl, po = lps[-2]
                if 4 <= (p - pp) <= 45 and l[p] < pl and osc[p] > po:
                    if (lo_max is None or (osc[p] <= lo_max and po <= lo_max)) and osc[i] > osc[i - 1] and c[i] > o[i]:
                        out[i] = 1
        wh = h[p - left : p + right + 1]
        if h[p] == np.max(wh) and np.sum(wh == h[p]) == 1:
            hps.append((p, h[p], osc[p]))
            if len(hps) >= 2:
                pp, ph, po = hps[-2]
                if 4 <= (p - pp) <= 45 and h[p] > ph and osc[p] < po:
                    if (hi_min is None or (osc[p] >= hi_min and po >= hi_min)) and osc[i] < osc[i - 1] and c[i] < o[i]:
                        if out[i] == 0:
                            out[i] = -1
    return out


def _orb_30m(idx, o, h, l, c, atr, start_h, start_m, end_h):
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    idx_u = pd.DatetimeIndex(idx)
    if idx_u.tz is not None:
        idx_u = idx_u.tz_convert("UTC").tz_localize(None)
    dates, hours, mins = idx_u.date, idx_u.hour, idx_u.minute
    start_tot = start_h * 60 + start_m
    end_or_tot = start_tot + 30
    end_tr_tot = end_h * 60
    ranges = {}
    for i in range(n):
        tot = hours[i] * 60 + mins[i]
        if start_tot <= tot < end_or_tot:
            d = dates[i]
            cur = ranges.get(d)
            if cur is None:
                ranges[d] = [h[i], l[i], 1]
            else:
                cur[0] = max(cur[0], h[i])
                cur[1] = min(cur[1], l[i])
                cur[2] += 1
    trig = set()
    for i in range(1, n):
        tot = hours[i] * 60 + mins[i]
        if not (end_or_tot <= tot < end_tr_tot):
            continue
        d = dates[i]
        rng = ranges.get(d)
        if not rng or rng[2] < 2 or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        orh, orl, _ = rng
        if not (0.6 * atr[i] <= (orh - orl) <= 8.0 * atr[i]):
            continue
        if (d, 1) not in trig and c[i] > orh and c[i - 1] <= orh and c[i] > o[i]:
            out[i] = 1
            trig.add((d, 1))
        elif (d, -1) not in trig and c[i] < orl and c[i - 1] >= orl and c[i] < o[i]:
            out[i] = -1
            trig.add((d, -1))
    return out


def eval_extended_strategy(df: pd.DataFrame, key: str, ind: dict, htf_trend: np.ndarray) -> np.ndarray:
    o, h, l, c = ind["o"], ind["h"], ind["l"], ind["c"]
    v = df["volume"].to_numpy(float) if "volume" in df.columns else np.zeros(len(df), dtype=float)
    n = len(c)
    atr14, chop, adx = ind["atr14"], ind["chop"], ind["adx"]
    sma20, std20 = ind["sma20"], ind["std20"]
    bb_u, bb_l = sma20 + 2.0 * std20, sma20 - 2.0 * std20
    ema21, ema50 = ind["ema21"], ind["ema50"]
    rsi7, rsi14 = ind["rsi7"], ind["rsi14"]
    w = v if np.sum(v) > 0 else np.maximum(h - l, 1e-6)
    is_jpy = float(np.nanmedian(c)) > 20.0

    # --- MEAN (20) ---
    if key in ("MR_VWAP_Z_5M", "MR_VWAP_Z_15M"):
        tp = (h + l + c) / 3.0
        dtmp = pd.DataFrame({"pv": tp * w, "w": w}, index=ind["index"])
        grp = dtmp.groupby(dtmp.index.date)
        vwap = grp["pv"].cumsum().to_numpy() / (grp["w"].cumsum().to_numpy() + 1e-12)
        diff = c - vwap
        z = diff / (_std(diff, 30) + 1e-12)
        pz = _prev(z)
        th = 2.0 if key.endswith("5M") else 2.1
        return _sig((pz < -th) & (z > pz) & (c > o), (pz > th) & (z < pz) & (c < o))
    elif key in ("MR_KAMA_BND_5M", "MR_KAMA_BND_15M"):
        k, er = _kama(c, 10), _er(c, 10)
        return _sig((er < 0.30) & (c < k - 1.7 * atr14) & (rsi7 < 35), (er < 0.30) & (c > k + 1.7 * atr14) & (rsi7 > 65))
    elif key in ("MR_FISH_5M", "MR_FISH_15M"):
        ft = _fisher(h, l, 10)
        pft = _prev(ft)
        return _sig((pft < -2.0) & (ft > pft) & (c > o), (pft > 2.0) & (ft < pft) & (c < o))
    elif key == "MR_INV_FISH_5M":
        v1 = 0.1 * (_rsi(c, 5) - 50.0)
        s = _wma(np.nan_to_num(v1), 5)
        ift = (np.exp(2.0 * s) - 1.0) / (np.exp(2.0 * s) + 1.0)
        pift = _prev(ift)
        return _sig((pift < -0.5) & (ift >= -0.5), (pift > 0.5) & (ift <= 0.5))
    elif key == "MR_PCTB_5M":
        pctb = (c - bb_l) / (bb_u - bb_l + 1e-12)
        bbw = (bb_u - bb_l) / (sma20 + 1e-12)
        medw = pd.Series(bbw).rolling(50, min_periods=20).median().to_numpy(float)
        return _sig((bbw <= medw) & (pctb < 0.0) & (c > o), (bbw <= medw) & (pctb > 1.0) & (c < o))
    elif key == "MR_Z30_15M":
        z = (c - _sma(c, 30)) / (_std(c, 30) + 1e-12)
        return _sig((z < -2.3) & (c > o), (z > 2.3) & (c < o))
    elif key == "MR_Z50_5M":
        z = (c - _sma(c, 50)) / (_std(c, 50) + 1e-12)
        return _sig((chop >= 48) & (z < -2.0) & (c > o), (chop >= 48) & (z > 2.0) & (c < o))
    elif key == "MR_SMTH_5M":
        sm = _supersmoother(c, 15)
        return _sig((c < sm - 1.9 * atr14) & (rsi7 < 32), (c > sm + 1.9 * atr14) & (rsi7 > 68))
    elif key in ("MR_LAG_RSI_5M", "MR_LAG_RSI_15M"):
        lr = _laguerre(c, 0.7)
        plr = _prev(lr)
        return _sig((plr < 0.15) & (lr >= 0.15) & (c > o), (plr > 0.85) & (lr <= 0.85) & (c < o))
    elif key == "MR_CAM_H3L3_5M":
        dh, dl, dc = _prev_daily(ind["index"], h, l, c)
        rng_d = dh - dl
        l3 = dc - rng_d * 1.1 / 4.0
        h3 = dc + rng_d * 1.1 / 4.0
        return _sig((chop >= 48) & np.isfinite(l3) & (l <= l3) & (c > l3) & (rsi14 < 40), (chop >= 48) & np.isfinite(h3) & (h >= h3) & (c < h3) & (rsi14 > 60))
    elif key == "MR_CAM_H4L4_15M":
        dh, dl, dc = _prev_daily(ind["index"], h, l, c)
        rng_d = dh - dl
        l4 = dc - rng_d * 1.1 / 2.0
        h4 = dc + rng_d * 1.1 / 2.0
        return _sig(np.isfinite(l4) & (l <= l4) & (c > l4) & (c > o), np.isfinite(h4) & (h >= h4) & (c < h4) & (c < o))
    elif key == "MR_CMO_5M":
        cmo = _cmo(c, 14)
        pcmo = _prev(cmo)
        return _sig((chop >= 48) & (pcmo < -50) & (cmo >= -50), (chop >= 48) & (pcmo > 50) & (cmo <= 50))
    elif key == "MR_RSI2_MA_5M":
        r2 = _rsi(c, 2)
        return _sig((chop >= 50) & (r2 < 5.0) & (c > o), (chop >= 50) & (r2 > 95.0) & (c < o))
    elif key == "MR_ENV_ATR_5M":
        rng_b = np.maximum(h - l, 1e-12)
        lw = (np.minimum(o, c) - l) / rng_b
        uw = (h - np.maximum(o, c)) / rng_b
        return _sig((c < ema21 - 1.9 * atr14) & (lw >= 0.30), (c > ema21 + 1.9 * atr14) & (uw >= 0.30))
    elif key == "MR_RVI_REV_5M":
        rng_b = np.maximum(h - l, 1e-12)
        num = _sma(c - o, 4) / (_sma(rng_b, 4) + 1e-12)
        rvi = _sma(num, 10)
        prvi = _prev(rvi)
        return _sig((adx < 22) & (prvi < -0.35) & (rvi > prvi) & (c > o), (adx < 22) & (prvi > 0.35) & (rvi < prvi) & (c < o))
    elif key == "MR_MID_CH_5M":
        hi40 = pd.Series(h).rolling(40, min_periods=40).max().shift(1).to_numpy(float)
        lo40 = pd.Series(l).rolling(40, min_periods=40).min().shift(1).to_numpy(float)
        return _sig((chop >= 52) & (l <= lo40) & (c > lo40) & (c > o), (chop >= 52) & (h >= hi40) & (c < hi40) & (c < o))

    # --- TREND (18) ---
    elif key in ("TR_HOLY_5M", "TR_HOLY_15M"):
        pl, ph = _prev(l), _prev(h)
        return _sig((adx > 28) & (c > ema21) & (pl <= ema21) & (c > ph), (adx > 28) & (c < ema21) & (ph >= ema21) & (c < pl))
    elif key in ("TR_KAMA_PB_5M", "TR_KAMA_PB_15M"):
        k, er = _kama(c, 10), _er(c, 10)
        r2 = _rsi(c, 2)
        return _sig((er > 0.35) & (c > k) & (r2 < 15), (er > 0.35) & (c < k) & (r2 > 85))
    elif key == "TR_CHAND_15M":
        hh = pd.Series(h).rolling(22, min_periods=22).max().to_numpy(float)
        ll = pd.Series(l).rolling(22, min_periods=22).min().to_numpy(float)
        ls, ss = hh - 3.0 * atr14, ll + 3.0 * atr14
        pc = _prev(c)
        return _sig((htf_trend >= 0) & (c > ss) & (pc <= _prev(ss)), (htf_trend <= 0) & (c < ls) & (pc >= _prev(ls)))
    elif key == "TR_DON_MID_5M":
        mid = 0.5 * (pd.Series(h).rolling(20, min_periods=20).max().to_numpy(float) + pd.Series(l).rolling(20, min_periods=20).min().to_numpy(float))
        pm = _prev(mid, 3)
        return _sig((adx > 22) & (mid > pm) & (l <= mid) & (c > mid), (adx > 22) & (mid < pm) & (h >= mid) & (c < mid))
    elif key in ("TR_HMA_5M", "TR_HMA_15M"):
        hm = _hma(c, 21)
        phm, phm2 = _prev(hm, 1), _prev(hm, 2)
        return _sig((htf_trend >= 0) & (c > ema50) & (hm > phm) & (phm <= phm2), (htf_trend <= 0) & (c < ema50) & (hm < phm) & (phm >= phm2))
    elif key in ("TR_VORTEX_5M", "TR_VORTEX_15M"):
        pc, ph, pl = _prev(c), _prev(h), _prev(l)
        tr_s = pd.Series(np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc))))
        tr14 = tr_s.rolling(14, min_periods=14).sum() + 1e-12
        vip = (pd.Series(np.abs(h - pl)).rolling(14, min_periods=14).sum() / tr14).to_numpy(float)
        vim = (pd.Series(np.abs(l - ph)).rolling(14, min_periods=14).sum() / tr14).to_numpy(float)
        rising = adx > _prev(adx)
        return _sig((adx > 20) & rising & (vip > vim) & (_prev(vip) <= _prev(vim)), (adx > 20) & rising & (vip < vim) & (_prev(vip) >= _prev(vim)))
    elif key in ("TR_STC_5M", "TR_STC_15M"):
        st = _stc(c)
        pst = _prev(st)
        return _sig((pst < 25) & (st >= 25) & (c > o), (pst > 75) & (st <= 75) & (c < o))
    elif key == "TR_ELDER_5M":
        e13 = _ema(c, 13)
        pe13, pmh = _prev(e13), _prev(ind["macd_hist"])
        pe13_2, pmh2 = _prev(e13, 2), _prev(ind["macd_hist"], 2)
        bull_now = (e13 > pe13) & (ind["macd_hist"] > pmh)
        bear_now = (e13 < pe13) & (ind["macd_hist"] < pmh)
        was_bull = (pe13 > pe13_2) & (pmh > pmh2)
        was_bear = (pe13 < pe13_2) & (pmh < pmh2)
        return _sig(bull_now & (~was_bull) & (c > ema50), bear_now & (~was_bear) & (c < ema50))
    elif key == "TR_ELDER_3M":
        fi2 = _ema((c - _prev(c)) * w, 2)
        pfi = _prev(fi2)
        return _sig((htf_trend == 1) & (pfi < 0) & (fi2 > 0), (htf_trend == -1) & (pfi > 0) & (fi2 < 0))
    elif key == "TR_STOCHR_5M":
        rs = pd.Series(rsi14)
        rlo, rhi = rs.rolling(14, min_periods=14).min(), rs.rolling(14, min_periods=14).max()
        k = (100.0 * (rs - rlo) / (rhi - rlo + 1e-12)).rolling(3).mean().to_numpy(float)
        d = _sma(k, 3)
        return _sig((htf_trend == 1) & (k > d) & (_prev(k) <= _prev(d)) & (k < 25), (htf_trend == -1) & (k < d) & (_prev(k) >= _prev(d)) & (k > 75))
    elif key == "TR_AROON_15M":
        a_up = np.full(n, np.nan)
        a_dn = np.full(n, np.nan)
        if n >= 26:
            wh, wl = sliding_window_view(h, 26), sliding_window_view(l, 26)
            a_up[25:] = 100.0 * np.argmax(wh, axis=1) / 25.0
            a_dn[25:] = 100.0 * np.argmin(wl, axis=1) / 25.0
        return _sig((a_up > 70) & (a_dn < 30) & (_prev(a_up) <= _prev(a_dn)) & (a_up > a_dn), (a_dn > 70) & (a_up < 30) & (_prev(a_dn) <= _prev(a_up)) & (a_dn > a_up))
    elif key == "TR_VIDYA_5M":
        vd = _vidya(c, 14)
        pvd = _prev(vd)
        return _sig((vd > pvd) & (l <= vd) & (c > vd) & (c > o), (vd < pvd) & (h >= vd) & (c < vd) & (c < o))
    elif key == "TR_123_PB_3M":
        exp_u = (_prev(c, 2) - _prev(o, 2)) > 1.1 * _prev(atr14, 2)
        exp_d = (_prev(o, 2) - _prev(c, 2)) > 1.1 * _prev(atr14, 2)
        return _sig(exp_u & (_prev(c, 1) < _prev(o, 1)) & (c > _prev(h, 1)), exp_d & (_prev(c, 1) > _prev(o, 1)) & (c < _prev(l, 1)))

    # --- VOL (15) ---
    elif key == "VL_ASIA_SWP_5M":
        ah, al = _session_bounds(ind["index"], h, l, 0, 7, 7, 11)
        return _sig(np.isfinite(al) & (l < al) & (c > al) & (c > o), np.isfinite(ah) & (h > ah) & (c < ah) & (c < o))
    elif key == "VL_NY_SWP_5M":
        lh, ll = _session_bounds(ind["index"], h, l, 8, 12, 13, 16)
        return _sig(np.isfinite(ll) & (l < ll) & (c > ll) & (c > o), np.isfinite(lh) & (h > lh) & (c < lh) & (c < o))
    elif key in ("VL_IMP_EXH_5M", "VL_IMP_EXH_15M"):
        rng = np.maximum(h - l, 1e-12)
        lw = np.minimum(o, c) - l
        uw = h - np.maximum(o, c)
        big = rng > 2.0 * atr14
        return _sig(big & (c < o) & (lw > 0.35 * rng), big & (c > o) & (uw > 0.35 * rng))
    elif key == "VL_NR4_BRK_5M":
        rng = h - l
        nr4 = _prev(rng, 1) <= pd.Series(rng).shift(2).rolling(3, min_periods=3).min().to_numpy(float)
        return _sig(nr4 & (c > _prev(h)) & (c > o), nr4 & (c < _prev(l)) & (c < o))
    elif key == "VL_NR7_BRK_15M":
        rng = h - l
        nr7 = _prev(rng, 1) <= pd.Series(rng).shift(2).rolling(6, min_periods=6).min().to_numpy(float)
        return _sig(nr7 & (c > _prev(h)) & (c > o), nr7 & (c < _prev(l)) & (c < o))
    elif key == "VL_ID_NR4_5M":
        rng = h - l
        inside = (_prev(h, 1) <= _prev(h, 2)) & (_prev(l, 1) >= _prev(l, 2))
        nr4 = _prev(rng, 1) <= pd.Series(rng).shift(2).rolling(3, min_periods=3).min().to_numpy(float)
        return _sig(inside & nr4 & (c > _prev(h)), inside & nr4 & (c < _prev(l)))
    elif key == "VL_BB_EXP_5M":
        bbw = (bb_u - bb_l) / (sma20 + 1e-12)
        lo20 = pd.Series(bbw).rolling(20, min_periods=20).min().shift(1).to_numpy(float)
        return _sig((_prev(bbw) <= _prev(lo20) * 1.05) & (bbw > lo20 * 1.15) & (c > bb_u), (_prev(bbw) <= _prev(lo20) * 1.05) & (bbw > lo20 * 1.15) & (c < bb_l))
    elif key == "VL_ATR_EXP_3M":
        pc = _prev(c)
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        tr[0] = h[0] - l[0]
        atr5 = pd.Series(tr).ewm(alpha=0.2, min_periods=5, adjust=False).mean().to_numpy(float)
        ratio = atr5 / (ind["atr20"] + 1e-12)
        rng = np.maximum(h - l, 1e-12)
        body = np.abs(c - o)
        return _sig((ratio > 1.20) & (body > 0.55 * rng) & (c > o), (ratio > 1.20) & (body > 0.55 * rng) & (c < o))
    elif key == "VL_GAP_FILL_5M":
        pc = _prev(c)
        gap = o - pc
        return _sig((gap < -0.15 * atr14) & (c > o) & (c < pc), (gap > 0.15 * atr14) & (c < o) & (c > pc))
    elif key == "VL_ORB_NY30_5M":
        return _orb_30m(ind["index"], o, h, l, c, atr14, 14, 30, 18)
    elif key == "VL_ORB_LN30_5M":
        return _orb_30m(ind["index"], o, h, l, c, atr14, 8, 0, 12)
    elif key in ("VL_CLIMAX_3M", "VL_CLIMAX_5M"):
        med = pd.Series(w).rolling(40, min_periods=20).median().to_numpy(float) + 1e-12
        spike = w > 1.8 * med
        body = np.abs(c - o) + 1e-12
        return _sig(spike & ((np.minimum(o, c) - l) > 1.5 * body), spike & ((h - np.maximum(o, c)) > 1.5 * body))
    elif key == "VL_KCP_BRK_15M":
        ku, kl = ema21 + 1.5 * atr14, ema21 - 1.5 * atr14
        return _sig((htf_trend >= 0) & (adx > 24) & (c > ku) & (_prev(c) <= _prev(ku)), (htf_trend <= 0) & (adx > 24) & (c < kl) & (_prev(c) >= _prev(kl)))

    # --- DIV (15) ---
    elif key in ("DV_OBV_5M", "DV_OBV_15M"):
        obv = pd.Series(np.nan_to_num(np.sign(c - _prev(c))) * w).cumsum()
        ohi, olo = obv.rolling(10, min_periods=10).max().shift(1).to_numpy(float), obv.rolling(10, min_periods=10).min().shift(1).to_numpy(float)
        phi = pd.Series(h).rolling(10, min_periods=10).max().shift(1).to_numpy(float)
        plo = pd.Series(l).rolling(10, min_periods=10).min().shift(1).to_numpy(float)
        return _sig((l < plo) & (obv.to_numpy(float) > olo) & (c > o), (h > phi) & (obv.to_numpy(float) < ohi) & (c < o))
    elif key == "DV_WPR_HK_5M":
        pw = _prev(ind["wpr14"])
        return _sig((chop >= 50) & (pw < -90) & (ind["wpr14"] >= -85), (chop >= 50) & (pw > -10) & (ind["wpr14"] <= -15))
    elif key == "DV_WPR_DIV_5M":
        return _div_swing(ind, ind["wpr14"], lo_max=-65.0, hi_min=-35.0)
    elif key in ("DV_CCI_DIV_5M", "DV_CCI_DIV_15M"):
        return _div_swing(ind, ind["cci20"], lo_max=-90.0, hi_min=90.0)
    elif key == "DV_STOCH_DIV_5M":
        return _div_swing(ind, ind["stoch_k"], lo_max=35.0, hi_min=65.0)
    elif key == "DV_MFI_DIV_5M":
        tp = (h + l + c) / 3.0
        rm = tp * w
        dtp = tp - _prev(tp)
        pmf = pd.Series(np.where(dtp > 0, rm, 0.0)).rolling(14, min_periods=14).sum()
        nmf = pd.Series(np.where(dtp < 0, rm, 0.0)).rolling(14, min_periods=14).sum()
        mfi = (100.0 - 100.0 / (1.0 + pmf / (nmf + 1e-12))).to_numpy(float)
        return _div_swing(ind, mfi, lo_max=40.0, hi_min=60.0)
    elif key == "DV_FORCE_5M":
        fi13 = _ema((c - _prev(c)) * w, 13)
        pfi = _prev(fi13)
        return _sig((pfi < 0) & (fi13 > 0) & (c > ema50), (pfi > 0) & (fi13 < 0) & (c < ema50))
    elif key == "DV_TRIX_5M":
        e3 = _ema(_ema(_ema(np.log(np.maximum(c, 1e-9)), 12), 12), 12)
        trix = (e3 - _prev(e3)) / (np.abs(_prev(e3)) + 1e-12) * 10000.0
        sig_t = _sma(trix, 9)
        return _sig((trix > sig_t) & (_prev(trix) <= _prev(sig_t)) & (c > ema50), (trix < sig_t) & (_prev(trix) >= _prev(sig_t)) & (c < ema50))
    elif key == "DV_PPO_DIV_15M":
        ppo = (_ema(c, 12) - _ema(c, 26)) / (_ema(c, 26) + 1e-12) * 100.0
        return _div_swing(ind, ppo - _ema(ppo, 9))
    elif key == "DV_UO_5M":
        pc = _prev(c)
        bp = c - np.minimum(l, pc)
        tr_u = np.maximum(h, pc) - np.minimum(l, pc)
        a7 = pd.Series(bp).rolling(7).sum() / (pd.Series(tr_u).rolling(7).sum() + 1e-12)
        a14 = pd.Series(bp).rolling(14).sum() / (pd.Series(tr_u).rolling(14).sum() + 1e-12)
        a28 = pd.Series(bp).rolling(28).sum() / (pd.Series(tr_u).rolling(28).sum() + 1e-12)
        uo = (100.0 * (4.0 * a7 + 2.0 * a14 + a28) / 7.0).to_numpy(float)
        puo = _prev(uo)
        return _sig((puo < 30) & (uo >= 30) & (c > o), (puo > 70) & (uo <= 70) & (c < o))
    elif key == "DV_CMO_DIV_5M":
        return _div_swing(ind, _cmo(c, 14), lo_max=-25.0, hi_min=25.0)
    elif key == "DV_RSI7_DIV_3M":
        return _div_swing(ind, rsi7, lo_max=35.0, hi_min=65.0)
    elif key == "DV_DPO_REV_5M":
        dpo = c - _sma(c, 20)
        pdpo = _prev(dpo)
        return _sig((chop >= 48) & (pdpo < -1.5 * atr14) & (dpo > pdpo) & (c > o), (chop >= 48) & (pdpo > 1.5 * atr14) & (dpo < pdpo) & (c < o))

    # --- SMC (12) ---
    elif key in ("SM_TURTLE_5M", "SM_TURTLE_15M"):
        hh20 = pd.Series(h).rolling(20, min_periods=20).max().shift(1).to_numpy(float)
        ll20 = pd.Series(l).rolling(20, min_periods=20).min().shift(1).to_numpy(float)
        r_hh3 = pd.Series(h).rolling(3, min_periods=3).max().shift(1).to_numpy(float)
        r_ll3 = pd.Series(l).rolling(3, min_periods=3).min().shift(1).to_numpy(float)
        return _sig((ll20 < r_ll3) & (l < ll20) & (c > ll20), (hh20 > r_hh3) & (h > hh20) & (c < hh20))
    elif key in ("SM_PDHL_15M", "SM_PDHL_5M"):
        pdh, pdl, _ = _prev_daily(ind["index"], h, l, c)
        rng = np.maximum(h - l, 1e-12)
        return _sig(np.isfinite(pdl) & (l < pdl) & (c > pdl) & ((np.minimum(o, c) - l) >= 0.45 * rng), np.isfinite(pdh) & (h > pdh) & (c < pdh) & ((h - np.maximum(o, c)) >= 0.45 * rng))
    elif key in ("SM_OB_RET_5M", "SM_OB_RET_15M"):
        po, pc = _prev(o), _prev(c)
        bh = np.full(n, np.nan)
        bl = np.full(n, np.nan)
        sh = np.full(n, np.nan)
        sl = np.full(n, np.nan)
        lbh = lbl = lsh = lsl = np.nan
        for i in range(2, n):
            imp = c[i - 1] - o[i - 1]
            if c[i - 2] < o[i - 2] and imp > 1.2 * atr14[i - 1]:
                lbh, lbl = h[i - 2], l[i - 2]
            if c[i - 2] > o[i - 2] and -imp > 1.2 * atr14[i - 1]:
                lsh, lsl = h[i - 2], l[i - 2]
            bh[i], bl[i], sh[i], sl[i] = lbh, lbl, lsh, lsl
        return _sig(np.isfinite(bh) & (l <= bh) & (c >= bl) & (c > o) & (pc < po), np.isfinite(sl) & (h >= sl) & (c <= sh) & (c < o) & (pc > po))
    elif key == "SM_CHOCH_5M":
        sh12 = pd.Series(h).rolling(12, min_periods=12).max().shift(1).to_numpy(float)
        sl12 = pd.Series(l).rolling(12, min_periods=12).min().shift(1).to_numpy(float)
        return _sig((c > _prev(sh12)) & (_prev(c) <= _prev(sh12)) & (c > ema21), (c < _prev(sl12)) & (_prev(c) >= _prev(sl12)) & (c < ema21))
    elif key == "SM_EQHL_5M":
        eqh = np.abs(_prev(h, 2) - _prev(h, 6)) <= 0.15 * atr14
        eql = np.abs(_prev(l, 2) - _prev(l, 6)) <= 0.15 * atr14
        lvl_h = np.maximum(_prev(h, 2), _prev(h, 6))
        lvl_l = np.minimum(_prev(l, 2), _prev(l, 6))
        return _sig(eql & (l < lvl_l) & (c > lvl_l) & (c > o), eqh & (h > lvl_h) & (c < lvl_h) & (c < o))
    elif key == "SM_STOP_HNT_3M":
        hi10 = pd.Series(h).rolling(10, min_periods=10).max().shift(1).to_numpy(float)
        lo10 = pd.Series(l).rolling(10, min_periods=10).min().shift(1).to_numpy(float)
        return _sig(((lo10 - l) > 0.3 * atr14) & (c > lo10) & ((h - l) > 1.2 * atr14), ((h - hi10) > 0.3 * atr14) & (c < hi10) & ((h - l) > 1.2 * atr14))
    elif key == "SM_FVG_15M":
        out = np.zeros(n, dtype=np.int8)
        for i in range(25, n):
            if not np.isfinite(atr14[i]) or atr14[i] <= 0:
                continue
            for k in range(2, 10):
                m = i - k
                if m < 2 or not np.isfinite(atr14[m - 1]):
                    continue
                if (l[m] - h[m - 2]) >= 0.25 * atr14[m - 1] and htf_trend[i] >= 0:
                    if np.all(c[m + 1 : i] >= h[m - 2]) and l[i] <= l[m] and c[i] > h[m - 2] and c[i] > o[i]:
                        out[i] = 1
                        break
                if (l[m - 2] - h[m]) >= 0.25 * atr14[m - 1] and htf_trend[i] <= 0:
                    if np.all(c[m + 1 : i] <= l[m - 2]) and h[i] >= h[m] and c[i] < l[m - 2] and c[i] < o[i]:
                        out[i] = -1
                        break
        return out
    elif key == "SM_ROUND_5M":
        step = 0.50 if is_jpy else 0.0050
        nearest = np.round(c / step) * step
        tol = 0.15 * atr14
        return _sig((chop >= 50) & (np.abs(l - nearest) <= tol) & (c > nearest) & (rsi14 < 38), (chop >= 50) & (np.abs(h - nearest) <= tol) & (c < nearest) & (rsi14 > 62))
    elif key == "SM_PIV_S1R1_5M":
        pdh, pdl, pdc = _prev_daily(ind["index"], h, l, c)
        pp = (pdh + pdl + pdc) / 3.0
        r1v, s1v = 2.0 * pp - pdl, 2.0 * pp - pdh
        return _sig((chop >= 48) & np.isfinite(s1v) & (l <= s1v) & (c > s1v) & (c > o), (chop >= 48) & np.isfinite(r1v) & (h >= r1v) & (c < r1v) & (c < o))

    # --- PA (10) ---
    elif key == "PA_OOPS_5M":
        return _sig((o < _prev(l) + 0.05 * atr14) & (c > _prev(c)) & (c > o), (o > _prev(h) - 0.05 * atr14) & (c < _prev(c)) & (c < o))
    elif key == "PA_SMASH_5M":
        lo5 = pd.Series(l).rolling(5, min_periods=5).min().shift(2).to_numpy(float)
        hi5 = pd.Series(h).rolling(5, min_periods=5).max().shift(2).to_numpy(float)
        return _sig((_prev(c) < lo5) & (c > _prev(h)) & (c > o), (_prev(c) > hi5) & (c < _prev(l)) & (c < o))
    elif key == "PA_HARAMI_5M":
        inside = (h <= _prev(h)) & (l >= _prev(l))
        return _sig(inside & (_prev(c) < _prev(o)) & (c > o) & (l <= bb_l) & (rsi14 < 40), inside & (_prev(c) > _prev(o)) & (c < o) & (h >= bb_u) & (rsi14 > 60))
    elif key == "PA_TWEZ_5M":
        lo15 = pd.Series(l).rolling(15, min_periods=15).min().shift(1).to_numpy(float)
        hi15 = pd.Series(h).rolling(15, min_periods=15).max().shift(1).to_numpy(float)
        return _sig((np.abs(l - _prev(l)) <= 0.08 * atr14) & (l <= lo15 * 1.0005) & (c > o) & (_prev(c) < _prev(o)), (np.abs(h - _prev(h)) <= 0.08 * atr14) & (h >= hi15 * 0.9995) & (c < o) & (_prev(c) > _prev(o)))
    elif key == "PA_MOR_EVE_5M":
        b2 = np.abs(_prev(c, 2) - _prev(o, 2)) > 0.6 * atr14
        small1 = np.abs(_prev(c, 1) - _prev(o, 1)) < 0.3 * atr14
        mid2 = 0.5 * (_prev(o, 2) + _prev(c, 2))
        return _sig(b2 & (_prev(c, 2) < _prev(o, 2)) & small1 & (c > o) & (c > mid2) & (_prev(l, 1) <= bb_l), b2 & (_prev(c, 2) > _prev(o, 2)) & small1 & (c < o) & (c < mid2) & (_prev(h, 1) >= bb_u))
    elif key == "PA_MARU_5M":
        rng = np.maximum(h - l, 1e-12)
        body = np.abs(c - o)
        return _sig((body >= 0.80 * rng) & (rng > 1.3 * atr14) & (c > o) & (c > ema50), (body >= 0.80 * rng) & (rng > 1.3 * atr14) & (c < o) & (c < ema50))
    elif key == "PA_RAIL_5M":
        pb = np.abs(_prev(c) - _prev(o))
        cb = np.abs(c - o)
        eq = (pb > 0.6 * atr14) & (cb > 0.6 * atr14) & (np.abs(pb - cb) < 0.35 * np.maximum(pb, cb))
        return _sig(eq & (_prev(c) < _prev(o)) & (c > o) & (l <= bb_l + 0.3 * atr14), eq & (_prev(c) > _prev(o)) & (c < o) & (h >= bb_u - 0.3 * atr14))
    elif key == "PA_KICK_5M":
        return _sig((_prev(c) < _prev(o)) & (o >= _prev(c)) & (c > _prev(o)) & ((c - o) > 0.6 * atr14), (_prev(c) > _prev(o)) & (o <= _prev(c)) & (c < _prev(o)) & ((o - c) > 0.6 * atr14))
    elif key == "PA_PIN_15M":
        rng = np.maximum(h - l, 1e-12)
        body = np.abs(c - o)
        lw = np.minimum(o, c) - l
        uw = h - np.maximum(o, c)
        return _sig((lw > 2.2 * body) & (lw > 0.58 * rng) & (l < bb_l) & (c > bb_l), (uw > 2.2 * body) & (uw > 0.58 * rng) & (h > bb_u) & (c < bb_u))
    elif key == "PA_3BR_REV_3M":
        r3 = (_prev(c, 3) < _prev(o, 3)) & (_prev(c, 2) < _prev(o, 2)) & (_prev(c, 1) < _prev(o, 1))
        g3 = (_prev(c, 3) > _prev(o, 3)) & (_prev(c, 2) > _prev(o, 2)) & (_prev(c, 1) > _prev(o, 1))
        return _sig(r3 & (c > _prev(h, 1)) & (c > o), g3 & (c < _prev(l, 1)) & (c < o))

    # --- QUANT (10) ---
    elif key in ("QN_REG_SW_5M", "QN_REG_SW_15M"):
        er, k = _er(c, 10), _kama(c, 10)
        r2 = _rsi(c, 2)
        return _sig(((er > 0.35) & (c > k) & (r2 < 15)) | ((er < 0.20) & (c < bb_l) & (rsi7 < 30)), ((er > 0.35) & (c < k) & (r2 > 85)) | ((er < 0.20) & (c > bb_u) & (rsi7 > 70)))
    elif key == "QN_HURST_MR_5M":
        r1 = pd.Series(c).pct_change(1)
        r4 = pd.Series(c).pct_change(4)
        vr_h = (r4.rolling(40, min_periods=20).var() / (4.0 * r1.rolling(40, min_periods=20).var() + 1e-12)).to_numpy(float)
        return _sig((vr_h < 0.85) & (c < bb_l) & (rsi7 < 35), (vr_h < 0.85) & (c > bb_u) & (rsi7 > 65))
    elif key == "QN_ENTROPY_5M":
        sgn = np.sign(c - _prev(c))
        flips = pd.Series((sgn != _prev(sgn)).astype(float)).rolling(20, min_periods=20).mean().to_numpy(float)
        return _sig((flips > 0.55) & (rsi7 < 28) & (c > o), (flips > 0.55) & (rsi7 > 72) & (c < o))
    elif key == "QN_SKEW_REV_5M":
        sk = pd.Series(c).pct_change().rolling(20, min_periods=20).skew().to_numpy(float)
        return _sig((sk < -1.2) & (c > o) & (rsi14 < 40), (sk > 1.2) & (c < o) & (rsi14 > 60))
    elif key == "QN_KURT_EXH_5M":
        kt = pd.Series(c).pct_change().rolling(25, min_periods=25).kurt().to_numpy(float)
        return _sig((kt > 2.0) & (c < bb_l) & (c > o), (kt > 2.0) & (c > bb_u) & (c < o))
    elif key == "QN_BODY_EFF_5M":
        eff = np.abs(c - o) / (np.maximum(h - l, 1e-12))
        peff = pd.Series(eff).shift(1).rolling(5, min_periods=5).mean().to_numpy(float)
        return _sig((peff < 0.35) & (eff > 0.75) & (c > o) & (c > _prev(h)), (peff < 0.35) & (eff > 0.75) & (c < o) & (c < _prev(l)))
    elif key == "QN_TICK_SPD_3M":
        med = pd.Series(w).rolling(40, min_periods=20).median().to_numpy(float) + 1e-12
        return _sig((w > 1.8 * med) & ((c - o) > 0.5 * atr14) & (c > ema21), (w > 1.8 * med) & ((o - c) > 0.5 * atr14) & (c < ema21))
    elif key == "QN_VWAP_TR_5M":
        tp = (h + l + c) / 3.0
        dtmp = pd.DataFrame({"pv": tp * w, "w": w}, index=ind["index"])
        grp = dtmp.groupby(dtmp.index.date)
        vwap = grp["pv"].cumsum().to_numpy() / (grp["w"].cumsum().to_numpy() + 1e-12)
        return _sig((adx > 24) & (_prev(l) <= _prev(vwap)) & (c > vwap) & (c > o) & (c > ema50), (adx > 24) & (_prev(h) >= _prev(vwap)) & (c < vwap) & (c < o) & (c < ema50))
    elif key == "QN_MULT_Z_5M":
        z10 = (c - _sma(c, 10)) / (_std(c, 10) + 1e-12)
        z20 = (c - sma20) / (std20 + 1e-12)
        z50 = (c - _sma(c, 50)) / (_std(c, 50) + 1e-12)
        mz = (z10 + z20 + z50) / 3.0
        return _sig((mz < -2.0) & (c > o), (mz > 2.0) & (c < o))

    return np.zeros(n, dtype=np.int8)
