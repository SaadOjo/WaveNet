from __future__ import annotations

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        return self.relu(out)


def make_selfresnet(mode: str = "binary") -> nn.Sequential:
    layers = [
        nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        ResidualBlock(16, 16),
        ResidualBlock(16, 32, stride=2),
        ResidualBlock(32, 32),
        ResidualBlock(32, 64, stride=2),
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
    ]
    if mode.upper() == "CIFAR":
        layers.append(nn.Linear(128, 10))
    else:
        layers.append(nn.Linear(128, 1))
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)
