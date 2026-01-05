# -*- coding: utf-8 -*-
"""Training script for TCN model"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os

from model import TradingTCN
from config import (
    X_PATH, Y_PATH, MODEL_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    TRAIN_SPLIT, POS_WEIGHT, GRAD_CLIP
)


def prepare_data():
    """Load and prepare data for training"""
    # Ensure the data directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    # Convert NumPy → Torch
    X = np.transpose(X, (0, 2, 1))
    mask = np.isfinite(X).all(axis=(1,2)) & np.isfinite(y).all(axis=1)
    X = X[mask]
    y = y[mask]

    print("Breakout rate:", y[:, 0].mean())
    print("Retest rate:", y[:, 1].mean())
    print("Continuation rate:", y[:, 2].mean())
    print("Move size mean/std:", y[:, 3].mean(), y[:, 3].std())

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Time-based Train / Validation Split
    split_idx = int(len(X_tensor) * TRAIN_SPLIT)

    X_train = X_tensor[:split_idx]
    y_train = y_tensor[:split_idx]

    X_val = X_tensor[split_idx:]
    y_val = y_tensor[split_idx:]

    print("Train samples:", len(X_train))
    print("Val samples:", len(X_val))

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader


def train_model():
    """Main training function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = prepare_data()

    # Get feature count from transposed data (after transpose: samples, features, time)
    # Or use config value directly
    from config import FEATURE_COLS
    model = TradingTCN(num_features=len(FEATURE_COLS))
    model.to(device)

    # Loss Functions
    pos_weight = torch.tensor([POS_WEIGHT], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mse = nn.MSELoss()  # for move_size

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # Forward pass
            breakout_logits, retest_logits, cont_logits, move_pred = model(xb)

            # Loss per head
            loss_breakout = bce(breakout_logits.squeeze(), yb[:, 0])
            loss_retest   = bce(retest_logits.squeeze(),   yb[:, 1])
            loss_cont     = bce(cont_logits.squeeze(),     yb[:, 2])
            loss_move     = mse(move_pred.squeeze(),       yb[:, 3])

            # Total loss
            loss = (
                loss_breakout +
                loss_retest +
                loss_cont +
                0.01 * loss_move
            )

            # Backprop
            loss.backward()

            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                breakout_logits, retest_logits, cont_logits, move_pred = model(xb)

                loss = (
                    bce(breakout_logits.squeeze(), yb[:, 0]) +
                    bce(retest_logits.squeeze(),   yb[:, 1]) +
                    bce(cont_logits.squeeze(),     yb[:, 2]) +
                    0.01 * mse(move_pred.squeeze(), yb[:, 3])
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    # Save Model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()

