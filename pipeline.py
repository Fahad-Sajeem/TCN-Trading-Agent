# -*- coding: utf-8 -*-
"""Main pipeline script to run the entire workflow"""

import os
from data_fetcher import fetch_and_save_data
from indicators import add_indicators
import pandas as pd
from labeling import create_labels
from preprocessing import preprocess_data
from train import train_model
from config import (
    RAW_DATA_PATH, RESISTANCE_DATA_PATH, LABELS_PATH, 
    X_PATH, Y_PATH, MODEL_PATH
)


def run_full_pipeline():
    """Run the complete pipeline from data fetching to training"""
    
    print("=" * 60)
    print("Step 1: Fetching data from Binance API")
    print("=" * 60)
    df = fetch_and_save_data()
    print(f"✓ Data fetched and saved. Shape: {df.shape}\n")
    
    print("=" * 60)
    print("Step 2: Computing technical indicators")
    print("=" * 60)
    df = pd.read_csv(RAW_DATA_PATH)
    df = add_indicators(df)
    df.to_csv(RESISTANCE_DATA_PATH, index=False)
    print(f"✓ Indicators computed and saved. Shape: {df.shape}\n")
    
    print("=" * 60)
    print("Step 3: Creating labels")
    print("=" * 60)
    labels_df = create_labels(df)
    labels_df.to_csv(LABELS_PATH, index=False)
    print(f"✓ Labels created. Shape: {labels_df.shape}\n")
    
    print("=" * 60)
    print("Step 4: Preprocessing data")
    print("=" * 60)
    X, y = preprocess_data()
    print(f"✓ Data preprocessed. X shape: {X.shape}, y shape: {y.shape}\n")
    
    print("=" * 60)
    print("Step 5: Training model")
    print("=" * 60)
    train_model()
    print("✓ Model trained and saved.\n")
    
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()

