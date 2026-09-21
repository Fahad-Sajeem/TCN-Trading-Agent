# -*- coding: utf-8 -*-
"""Build sliding-window training arrays from indicators + labels"""

import numpy as np
import pandas as pd
from config import (
    RESISTANCE_DATA_PATH, LABELS_PATH, X_PATH, Y_PATH,
    FEATURE_COLS, LABEL_COLS, LOOKBACK
)


def create_sliding_windows(df, feature_cols, label_cols, lookback=LOOKBACK):
    """Slice a labeled indicator dataframe into (X, y) lookback windows"""
    X, y = [], []

    feature_array = df[feature_cols].values
    label_array = df[label_cols].values

    for t in range(lookback - 1, len(df)):
        if np.isnan(label_array[t]).any():
            continue

        x_window = feature_array[t - lookback + 1: t + 1]
        y_target = label_array[t]

        X.append(x_window)
        y.append(y_target)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y


def preprocess_data():
    """Join indicators with labels and build sliding-window arrays"""
    df = pd.read_csv(RESISTANCE_DATA_PATH)
    labels_df = pd.read_csv(LABELS_PATH)

    df = df.reset_index(drop=True)
    labels_df["t"] = labels_df["t"].astype(int)

    labels_aligned = labels_df.set_index("t")[LABEL_COLS]
    df = df.join(labels_aligned, how="left")

    X, y = create_sliding_windows(df, FEATURE_COLS, LABEL_COLS, lookback=LOOKBACK)

    np.save(X_PATH, X)
    np.save(Y_PATH, y)

    return X, y


if __name__ == "__main__":
    X, y = preprocess_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
