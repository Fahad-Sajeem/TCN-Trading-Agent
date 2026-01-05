# -*- coding: utf-8 -*-
"""Configuration file for TCN trading model"""

import os

# Data fetching configuration
BASE_URL = "https://api.binance.us/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"
DATA_DAYS = 180

# Technical indicators
ATR_PERIOD = 14
RSI_PERIOD = 14
MAX_RESISTANCE_AGE = 200  # candles

# Labeling configuration
K = 10                 # prediction horizon (candles)
ATR_BREAK = 1.0        # breakout tolerance
ATR_CONTINUE = 1.0     # continuation threshold

# Feature columns
FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "rsi", "atr",
    "dist_to_resistance",
    "resistance_age",
    "resistance_touches"
]

# Label columns
LABEL_COLS = [
    "breakout",
    "retest",
    "continuation",
    "move_size"
]

# Model configuration
LOOKBACK = 200
NUM_CHANNELS = [64] * 7
KERNEL_SIZE = 3
DROPOUT = 0.1

# Training configuration
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 3e-4
TRAIN_SPLIT = 0.8
POS_WEIGHT = 5.0
GRAD_CLIP = 1.0

# File paths (adjust these for your environment)
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "BTCUSDT_5m_6months.csv")
RESISTANCE_DATA_PATH = os.path.join(DATA_DIR, "resistance.csv")
LABELS_PATH = os.path.join(DATA_DIR, "labels.csv")
X_PATH = os.path.join(DATA_DIR, "X.npy")
Y_PATH = os.path.join(DATA_DIR, "y.npy")
MODEL_PATH = os.path.join(DATA_DIR, "trading_tcn.pth")

