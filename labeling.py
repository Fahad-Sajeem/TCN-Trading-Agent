# -*- coding: utf-8 -*-
"""Label creation for training data"""

import numpy as np
import pandas as pd
from config import K, ATR_BREAK, ATR_CONTINUE


def label_breakout(df, t, resistance, atr):
    """Label if breakout occurs within K candles"""
    future_close = df['close'].iloc[t+1 : t+K+1]
    future_open  = df['open'].iloc[t+1 : t+K+1]

    return int(
        (future_close > resistance + 1.0 * atr).any()
        and (future_open < resistance).any()
    )


def label_retest(df, t, resistance, atr):
    """Label if retest occurs within K candles"""
    future_low = df['low'].iloc[t+1 : t+K+1]
    return int(future_low.min() <= resistance + ATR_BREAK * atr)


def label_continuation(df, t, resistance, atr):
    """Label if continuation occurs in next K candles"""
    future_high = df['high'].iloc[t+K+1 : t+2*K+1]
    return int(future_high.max() > resistance + ATR_CONTINUE * atr)


def label_move_size(df, t, resistance, atr):
    """Label move size in ATR units"""
    future_high = df['high'].iloc[t+1 : t+2*K+1]
    move = (future_high.max() - resistance) / atr
    return np.clip(move, -5.0, 5.0)


def create_labels(df):
    """Create labels for all valid timesteps"""
    labels = []

    for t in range(len(df)):

        # Ensure future data exists
        if t + 2*K >= len(df):
            continue

        resistance = df['resistance_price'].iloc[t]
        atr = df['atr'].iloc[t]

        # Skip if resistance or ATR is missing
        if pd.isna(resistance) or pd.isna(atr) or atr == 0:
            continue

        breakout = label_breakout(df, t, resistance, atr)

        if breakout:
            retest = label_retest(df, t, resistance, atr)
        else:
            retest = 0

        if retest:
            continuation = label_continuation(df, t, resistance, atr)
        else:
            continuation = 0

        move_size = label_move_size(df, t, resistance, atr)

        labels.append({
            "t": t,
            "breakout": breakout,
            "retest": retest,
            "continuation": continuation,
            "move_size": move_size
        })

    return pd.DataFrame(labels)


if __name__ == "__main__":
    from config import RESISTANCE_DATA_PATH, LABELS_PATH
    
    df = pd.read_csv(RESISTANCE_DATA_PATH)
    labels_df = create_labels(df)
    labels_df.to_csv(LABELS_PATH, index=False)
    print(f"Labels created. Shape: {labels_df.shape}")
    print(labels_df.head())

