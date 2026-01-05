# -*- coding: utf-8 -*-
"""TCN Model Architecture"""

import torch
import torch.nn as nn
from config import NUM_CHANNELS, KERNEL_SIZE, DROPOUT


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Match channels for residual
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # causal trim
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]  # causal trim
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TradingTCN(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            num_channels=NUM_CHANNELS,
            kernel_size=KERNEL_SIZE,
            dropout=DROPOUT
        )

        self.breakout_head = nn.Linear(64, 1)
        self.retest_head = nn.Linear(64, 1)
        self.cont_head = nn.Linear(64, 1)
        self.move_head = nn.Linear(64, 1)

    def forward(self, x):
        """
        x shape: (batch, features, time)
        """

        y = self.tcn(x)          # (batch, 64, time)
        last = y[:, :, -1]       # last timestep (batch, 64)

        breakout = self.breakout_head(last)       # logits
        retest = self.retest_head(last)
        continuation = self.cont_head(last)
        move_size = self.move_head(last)

        return breakout, retest, continuation, move_size

