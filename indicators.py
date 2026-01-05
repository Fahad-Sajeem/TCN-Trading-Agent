# -*- coding: utf-8 -*-
"""Technical indicators and resistance detection"""

import numpy as np
import pandas as pd
from config import ATR_PERIOD, RSI_PERIOD, MAX_RESISTANCE_AGE


def compute_atr(df, period=ATR_PERIOD):
    """Compute Average True Range"""
    high = df['high']
    low = df['low']
    close = df['close']

    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr


def compute_rsi(df, period=RSI_PERIOD):
    """Compute Relative Strength Index"""
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detect_swing_highs(df):
    """Detect swing highs in the price data"""
    df['is_swing_high'] = False

    for i in range(1, len(df) - 1):
        if df.loc[i, 'high'] > df.loc[i-1, 'high'] and df.loc[i, 'high'] > df.loc[i+1, 'high']:
            df.loc[i, 'is_swing_high'] = True

    return df


def find_resistances(df):
    """Find resistance levels from swing highs"""
    resistances = []

    for i in range(len(df)):
        if df.loc[i, 'is_swing_high']:
            resistances.append({
                "price": df.loc[i, 'high'],
                "index": i,
                "touches": 1,
                "last_touch": i
            })

    return resistances


def nearest_resistance(idx, close_price, resistances, max_age=MAX_RESISTANCE_AGE):
    """Find nearest resistance level"""
    valid = []
    for r in resistances:
        if r["price"] > close_price and (idx - r["index"]) <= max_age:
            valid.append(r)

    if not valid:
        return None

    return min(valid, key=lambda r: r["price"])


def compute_resistance_features(df, resistances):
    """Compute resistance-related features"""
    df['dist_to_resistance'] = np.nan
    df['resistance_age'] = np.nan
    df['resistance_touches'] = np.nan
    df['resistance_price'] = np.nan

    for i in range(len(df)):
        r = nearest_resistance(i, df.loc[i, 'close'], resistances)

        if r is None:
            # No resistance overhead
            df.loc[i, 'resistance_price'] = np.nan
            df.loc[i, 'dist_to_resistance'] = 5.0
            df.loc[i, 'resistance_age'] = 1.0
            df.loc[i, 'resistance_touches'] = 0.0
        else:
            atr = df.loc[i, 'atr']

            df.loc[i, 'resistance_price'] = r['price']
            df.loc[i, 'dist_to_resistance'] = (df.loc[i, 'close'] - r['price']) / (atr + 1e-6)
            df.loc[i, 'resistance_age'] = (i - r['index']) / MAX_RESISTANCE_AGE
            df.loc[i, 'resistance_touches'] = min(r['touches'] / 5, 1.0)

    return df


def add_indicators(df):
    """Add all technical indicators to dataframe"""
    df['atr'] = compute_atr(df)
    df['rsi'] = compute_rsi(df)
    
    df = detect_swing_highs(df)
    resistances = find_resistances(df)
    df = compute_resistance_features(df, resistances)
    
    return df


if __name__ == "__main__":
    import sys
    from config import RAW_DATA_PATH, RESISTANCE_DATA_PATH
    
    df = pd.read_csv(RAW_DATA_PATH)
    df = add_indicators(df)
    df.to_csv(RESISTANCE_DATA_PATH, index=False)
    print(f"Indicators computed and saved. Shape: {df.shape}")

