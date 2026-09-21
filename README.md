# TCN Trading Agent

A Temporal Convolutional Network (TCN) that watches BTC/USDT 5-minute candles and predicts, for each candle, the probability of a **resistance breakout**, a **retest** of that breakout, a **continuation** move, and the likely **move size** (in ATR units). It's a research/learning project, not a trading bot that places orders — it only produces predictions.

> ⚠️ Educational project. Not financial advice. The model has not been backtested for profitability; treat its output as a signal to study, not a signal to trade on.

## How it works

1. **Fetch** — pull 6 months of 5-minute BTCUSDT candles from the public Binance API (`data_fetcher.py`). No API key needed; it's a public market-data endpoint.
2. **Indicators** — compute ATR, RSI, swing highs, and distance/age/touch-count to the nearest resistance level (`indicators.py`).
3. **Labels** — for every candle, look K candles into the future and label whether a breakout happened, whether it got retested, whether it continued, and how big the move was (`labeling.py`).
4. **Windows** — slice the labeled series into overlapping lookback windows (200 candles) for the model to train on (`preprocessing.py` — see [Known gaps](#known-gaps)).
5. **Train** — a 7-layer causal, dilated TCN (`model.py`) with four output heads (breakout / retest / continuation / move-size), trained with BCE + MSE loss (`train.py`).
6. **Predict live** — `websocket_predictor.py` connects to Binance's public WebSocket kline stream, maintains a rolling buffer of candles, recomputes indicators in real time, and prints predictions as each 5-minute candle closes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch pandas numpy requests websocket-client
```

There's no `requirements.txt` yet — the packages above are everything the code imports.

## Usage

Run the pipeline end-to-end:

```bash
python pipeline.py
```

Or step by step:

```bash
python data_fetcher.py     # fetch & save 6 months of BTCUSDT 5m candles -> data/
python indicators.py       # compute ATR/RSI/resistance features
python labeling.py         # generate breakout/retest/continuation/move-size labels
python train.py            # train the TCN and save data/trading_tcn.pth
```

Live prediction against the real-time Binance stream, using the trained model:

```bash
python websocket_predictor.py
```

## Model

`TradingTCN` (`model.py`) is a stack of 7 dilated causal `TemporalBlock`s (dilation doubling each layer: 1, 2, 4, ... 64), each with residual connections. The final timestep's features feed four linear heads:

| Head | Output | Loss |
|---|---|---|
| Breakout | probability | BCE (pos-weighted) |
| Retest | probability | BCE (pos-weighted) |
| Continuation | probability | BCE (pos-weighted) |
| Move size | ATR-scaled magnitude | MSE |

All hyperparameters (lookback, channels, learning rate, thresholds, feature/label columns) live in `config.py`.

## Known gaps

- `pipeline.py` imports a `preprocessing.py` module (for building the sliding-window `X`/`y` arrays) that isn't in the repo yet — running the full pipeline as-is will fail at that step. The windowing logic it needs exists inline in `tcn_model_training.py` (`create_sliding_windows`); it just hasn't been extracted into its own module.
- `tcn_model_training.py` is the original Colab notebook this project grew out of (hardcoded Google Drive paths, no error handling). Kept for reference; `data_fetcher.py` / `indicators.py` / `labeling.py` / `train.py` are the maintained, modular versions of the same logic.
- No test suite yet.

## Disclaimer

This predicts probabilities from historical resistance/breakout patterns on one symbol and interval. It does not size positions, place orders, or manage risk, and it has not been validated against realistic trading costs or out-of-sample data. Do not trade real money on its output.
