from __future__ import annotations

import torch
import torch.nn as nn


def make_wavenet(mode: str = "binary") -> nn.Sequential:
    layers = [
        nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(2, 2), nn.Dropout(0.2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2, 2), nn.Dropout(0.3),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 128), nn.BatchNorm1d(128), nn.ReLU(),
        nn.Dropout(0.4),
    ]
    if mode.upper() == "CIFAR":
        layers.append(nn.Linear(128, 10))
    else:
        layers.append(nn.Linear(128, 1))
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def load_matching_weights(model: nn.Module, state_dict: dict) -> nn.Module:
    current_state = model.state_dict()
    filtered = {
        key: value
        for key, value in state_dict.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    current_state.update(filtered)
    model.load_state_dict(current_state)
    return model
