# -*- coding: utf-8 -*-
"""Data fetching from Binance API"""

import os
import time
import requests
import pandas as pd
from config import BASE_URL, SYMBOL, INTERVAL


def get_klines(symbol, interval, start_time, end_time):
    """Fetch klines data from Binance API"""
    all_klines = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }

        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if not data:
            break

        all_klines.extend(data)

        # move start_time forward
        start_time = data[-1][0] + 1

        time.sleep(0.2)  # avoid rate limits

    return all_klines


def fetch_and_save_data(symbol=SYMBOL, interval=INTERVAL, start_time=None, end_time=None, 
                        output_path=None):
    """Fetch data and save to CSV"""
    from utils import get_time_range
    from config import RAW_DATA_PATH
    
    if start_time is None or end_time is None:
        start_time, end_time = get_time_range()
    
    if output_path is None:
        output_path = RAW_DATA_PATH
    
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    print(f"Fetching data for {symbol} ({interval})...")
    raw_data = get_klines(symbol, interval, start_time, end_time)
    
    if not raw_data:
        print("Warning: No data fetched from API")
        return None
    
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ]
    
    df = pd.DataFrame(raw_data, columns=columns)
    
    # Convert types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    # Sort & reset index
    df = df.sort_values("open_time").reset_index(drop=True)
    
    # Save to file
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
    
    return df


if __name__ == "__main__":
    df = fetch_and_save_data()
    if df is not None:
        print(f"\nData fetch complete. Shape: {df.shape}")
    else:
        print("\nFailed to fetch data.")

