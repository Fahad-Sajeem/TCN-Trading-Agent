# -*- coding: utf-8 -*-
"""WebSocket-based real-time prediction using Binance 5-minute candles"""

import json
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import torch

try:
    import websocket
except ImportError:
    raise ImportError(
        "websocket-client is required. Install it with: pip install websocket-client"
    )

from config import (
    FEATURE_COLS, LOOKBACK, MODEL_PATH, SYMBOL, INTERVAL,
    ATR_PERIOD, RSI_PERIOD, MAX_RESISTANCE_AGE, X_PATH
)
from indicators import (
    compute_atr, compute_rsi, detect_swing_highs,
    find_resistances, nearest_resistance, compute_resistance_features
)
from model import TradingTCN


def compute_normalization_stats():
    """
    Compute mean and std from training data for feature normalization.
    This matches the normalization used during training.
    """
    if not os.path.exists(X_PATH):
        print(f"⚠️  Warning: Training data {X_PATH} not found. Using default normalization.")
        return None, None
    
    # Load training data
    X = np.load(X_PATH)  # Shape: (samples, time, features)
    
    # Reshape to (samples * time, features) to compute per-feature statistics
    X_flat = X.reshape(-1, X.shape[-1])  # (samples * time, features)
    
    # Compute mean and std for each feature
    feature_means = np.nanmean(X_flat, axis=0)
    feature_stds = np.nanstd(X_flat, axis=0)
    
    # Avoid division by zero
    feature_stds = np.where(feature_stds < 1e-6, 1.0, feature_stds)
    
    print(f"✅ Normalization stats computed from training data:")
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:20s} - Mean: {feature_means[i]:10.2f}, Std: {feature_stds[i]:10.2f}")
    
    return feature_means, feature_stds


class BinanceWebSocketPredictor:
    """Real-time prediction using Binance WebSocket"""
    
    def __init__(self, symbol=SYMBOL, interval=INTERVAL, model_path=MODEL_PATH):
        self.symbol = symbol.lower()
        self.interval = interval
        self.model_path = model_path
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradingTCN(num_features=len(FEATURE_COLS))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load normalization statistics from training data
        self.feature_means, self.feature_stds = compute_normalization_stats()
        
        # Data buffer - maintain last LOOKBACK candles
        self.candle_buffer = deque(maxlen=LOOKBACK + 50)  # Extra buffer for indicators
        self.df = None
        
        # Resistance tracking
        self.resistances = []
        
        # WebSocket
        self.ws = None
        self.running = False
        
        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Using device: {self.device}")
    
    def _initialize_dataframe(self):
        """Initialize dataframe from buffer"""
        if len(self.candle_buffer) < 50:  # Need some data for indicators
            return None
        
        data = list(self.candle_buffer)
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        # Convert types
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df = df.sort_values('open_time').reset_index(drop=True)
        return df
    
    def _update_indicators(self):
        """Update technical indicators and resistance features"""
        if self.df is None or len(self.df) < 50:
            return
        
        # Compute indicators
        self.df['atr'] = compute_atr(self.df, period=ATR_PERIOD)
        self.df['rsi'] = compute_rsi(self.df, period=RSI_PERIOD)
        
        # Update swing highs and resistances
        self.df = detect_swing_highs(self.df)
        self.resistances = find_resistances(self.df)
        
        # Update resistance features
        self.df = compute_resistance_features(self.df, self.resistances)
    
    def _prepare_model_input(self):
        """Prepare the last LOOKBACK candles for model prediction with normalization"""
        if self.df is None or len(self.df) < LOOKBACK:
            return None
        
        # Get last LOOKBACK rows
        df_window = self.df.tail(LOOKBACK).copy()
        
        # Check if we have all required features
        if df_window[FEATURE_COLS].isna().any().any():
            return None
        
        # Extract features in correct order
        feature_array = df_window[FEATURE_COLS].values  # (LOOKBACK, num_features)
        
        # FIX 1: Apply same normalization as training (z-score normalization)
        if self.feature_means is not None and self.feature_stds is not None:
            # Normalize: (x - mean) / std
            feature_array = (feature_array - self.feature_means) / (self.feature_stds + 1e-8)
        else:
            # Fallback: per-feature normalization if training stats not available
            feature_means = np.nanmean(feature_array, axis=0, keepdims=True)
            feature_stds = np.nanstd(feature_array, axis=0, keepdims=True)
            feature_stds = np.where(feature_stds < 1e-6, 1.0, feature_stds)
            feature_array = (feature_array - feature_means) / feature_stds
        
        # FIX 3: Clip extreme values for safety (prevent extreme logits)
        feature_array = np.clip(feature_array, -5.0, 5.0)
        
        # Debug: Print normalized feature statistics
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0
        
        if self._debug_count % 10 == 0:  # Print every 10 predictions
            print(f"\n[DEBUG] Normalized Feature Statistics:")
            print(f"  Overall - Mean: {feature_array.mean():.3f}, Std: {feature_array.std():.3f}, "
                  f"Min: {feature_array.min():.3f}, Max: {feature_array.max():.3f}")
            if feature_array.mean() > 1.0 or feature_array.std() > 2.0:
                print(f"  ⚠️  WARNING: Normalization may be incorrect!")
        
        # Transpose to (num_features, LOOKBACK) as model expects
        feature_array = feature_array.T  # (num_features, LOOKBACK)
        
        # Add batch dimension: (1, num_features, LOOKBACK)
        feature_array = feature_array[np.newaxis, :, :]
        
        # Convert to tensor
        X_tensor = torch.tensor(feature_array, dtype=torch.float32).to(self.device)
        
        # Verify no NaN or Inf
        if torch.isnan(X_tensor).any() or torch.isinf(X_tensor).any():
            print("⚠️  ERROR: Input contains NaN or Inf values!")
            return None
        
        return X_tensor
    
    def _make_prediction(self, X_tensor):
        """Make prediction using the model"""
        with torch.no_grad():
            breakout_logits, retest_logits, cont_logits, move_pred = self.model(X_tensor)
        
        # Get raw logits for debugging
        breakout_logit_val = breakout_logits.item()
        retest_logit_val = retest_logits.item()
        cont_logit_val = cont_logits.item()
        
        # FIX 4: Convert logits to probabilities (classification heads → sigmoid)
        breakout_prob = torch.sigmoid(breakout_logits).item()
        retest_prob = torch.sigmoid(retest_logits).item()
        cont_prob = torch.sigmoid(cont_logits).item()
        
        # Move size is raw linear output (can be negative)
        # Apply softplus if you want only positive values, otherwise keep as is
        move_size = move_pred.item()
        
        return {
            'breakout_prob': breakout_prob,
            'retest_prob': retest_prob,
            'continuation_prob': cont_prob,
            'move_size_atr': move_size,
            'breakout_logit': breakout_logit_val,  # For debugging
            'retest_logit': retest_logit_val,      # For debugging
            'cont_logit': cont_logit_val,          # For debugging
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Binance kline stream format
            if 'k' in data:
                kline = data['k']
                
                # Only process closed candles (is_closed = True)
                if kline['x']:  # x = is_closed
                    candle = [
                        int(kline['t']),      # open_time
                        float(kline['o']),    # open
                        float(kline['h']),    # high
                        float(kline['l']),    # low
                        float(kline['c']),    # close
                        float(kline['v']),    # volume
                        int(kline['T']),      # close_time
                        float(kline['q']),    # quote_asset_volume
                        int(kline['n']),      # number_of_trades
                        float(kline['V']),    # taker_buy_base_volume
                        float(kline['Q']),    # taker_buy_quote_volume
                        kline.get('B', '')    # ignore
                    ]
                    
                    self.candle_buffer.append(candle)
                    
                    # Update dataframe
                    self.df = self._initialize_dataframe()
                    
                    if self.df is not None:
                        # Update indicators
                        self._update_indicators()
                        
                        # Prepare input and make prediction
                        X_tensor = self._prepare_model_input()
                        
                        if X_tensor is not None:
                            # Quick self-check: Verify normalization
                            input_mean = X_tensor.mean().item()
                            input_std = X_tensor.std().item()
                            input_min = X_tensor.min().item()
                            input_max = X_tensor.max().item()
                            
                            prediction = self._make_prediction(X_tensor)
                            
                            # Get current price info
                            current_price = float(kline['c'])
                            current_time = pd.to_datetime(int(kline['t']), unit='ms')
                            
                            # Print prediction with debug info
                            print(f"\n{'='*60}")
                            print(f"Time: {current_time} | Price: ${current_price:.2f}")
                            print(f"{'='*60}")
                            print(f"Breakout Probability:     {prediction['breakout_prob']:.3f} (logit: {prediction['breakout_logit']:.3f})")
                            print(f"Retest Probability:       {prediction['retest_prob']:.3f} (logit: {prediction['retest_logit']:.3f})")
                            print(f"Continuation Probability: {prediction['continuation_prob']:.3f} (logit: {prediction['cont_logit']:.3f})")
                            print(f"Move Size (ATR):          {prediction['move_size_atr']:.3f}")
                            print(f"{'='*60}")
                            
                            # Quick self-check output
                            is_healthy = (abs(input_mean) < 1.0 and 0.5 < input_std < 2.0 and 
                                         input_min > -5.5 and input_max < 5.5)
                            status = "✅ Healthy" if is_healthy else "⚠️  Check normalization"
                            print(f"Input Stats - Mean: {input_mean:.3f}, Std: {input_std:.3f}, "
                                  f"Min: {input_min:.3f}, Max: {input_max:.3f} {status}")
                            print(f"{'='*60}\n")
                            
                            # Call custom callback if provided
                            if hasattr(self, 'on_prediction'):
                                self.on_prediction(prediction, current_price, current_time)
        
        except Exception as e:
            print(f"Error processing message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print("WebSocket connection closed")
        self.running = False
    
    def _on_open(self, ws):
        """Handle WebSocket open"""
        print(f"✅ WebSocket connected for {self.symbol.upper()} {self.interval}")
        print("Waiting for candle data...")
    
    def start(self):
        """Start WebSocket connection and prediction"""
        # Binance WebSocket stream URL
        stream_name = f"{self.symbol}@kline_{self.interval}"
        ws_url = f"wss://stream.binance.com:9443/ws/{stream_name}"
        
        print(f"Connecting to Binance WebSocket: {stream_name}")
        
        # Create WebSocket connection
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        
        self.running = True
        
        # Run WebSocket in a separate thread
        def run_ws():
            self.ws.run_forever()
        
        ws_thread = threading.Thread(target=run_ws, daemon=True)
        ws_thread.start()
        
        # Wait for connection
        time.sleep(2)
        
        return self
    
    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
        print("WebSocket connection stopped")
    
    def initialize_with_historical(self, num_candles=300):
        """Initialize buffer with historical data before starting WebSocket"""
        from data_fetcher import get_klines
        import datetime as dt
        
        print(f"Fetching {num_candles} historical candles...")
        
        end_time = int(dt.datetime.utcnow().timestamp() * 1000)
        start_time = end_time - (num_candles * 5 * 60 * 1000)  # 5 min intervals
        
        raw_data = get_klines(self.symbol.upper(), self.interval, start_time, end_time)
        
        if raw_data:
            # Add all historical candles to buffer
            for kline in raw_data:
                candle = [
                    int(kline[0]),      # open_time
                    float(kline[1]),    # open
                    float(kline[2]),    # high
                    float(kline[3]),    # low
                    float(kline[4]),    # close
                    float(kline[5]),    # volume
                    int(kline[6]),      # close_time
                    float(kline[7]),    # quote_asset_volume
                    int(kline[8]),      # number_of_trades
                    float(kline[9]),    # taker_buy_base_volume
                    float(kline[10]),   # taker_buy_quote_volume
                    kline[11]           # ignore
                ]
                self.candle_buffer.append(candle)
            
            # Initialize dataframe
            self.df = self._initialize_dataframe()
            if self.df is not None:
                self._update_indicators()
                print(f"✅ Initialized with {len(self.df)} historical candles")
            else:
                print("⚠️  Not enough historical data for initialization")
        else:
            print("⚠️  Could not fetch historical data")


def predict_with_websocket(symbol=SYMBOL, interval=INTERVAL, model_path=MODEL_PATH, 
                           use_historical=True, num_historical=300):
    """
    Main function to start WebSocket prediction
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT')
        interval: Candle interval (e.g., '5m')
        model_path: Path to trained model
        use_historical: Whether to initialize with historical data
        num_historical: Number of historical candles to fetch
    """
    predictor = BinanceWebSocketPredictor(symbol=symbol, interval=interval, 
                                          model_path=model_path)
    
    if use_historical:
        predictor.initialize_with_historical(num_candles=num_historical)
    
    # Start WebSocket
    predictor.start()
    
    # Keep running
    try:
        while predictor.running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping WebSocket...")
        predictor.stop()


def simple_predict(symbol="BTCUSDT", interval="5m"):
    """
    Simple function to start WebSocket prediction with default settings
    
    Args:
        symbol: Trading symbol (default: "BTCUSDT")
        interval: Candle interval (default: "5m")
    
    Example:
        >>> simple_predict("BTCUSDT", "5m")
    """
    predict_with_websocket(
        symbol=symbol,
        interval=interval,
        use_historical=True,
        num_historical=300
    )


if __name__ == "__main__":
    # Example usage
    simple_predict("BTCUSDT", "5m")

