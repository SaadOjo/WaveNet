from __future__ import annotations

import torch
import torch.nn as nn


def _final_head(hidden_features: int, mode: str = "binary"):
    if mode.upper() == "CIFAR":
        return [nn.Linear(hidden_features, 10)]
    return [nn.Linear(hidden_features, 1), nn.Sigmoid()]


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
        *_final_head(128, mode),
    ]
    return nn.Sequential(*layers)


def make_small_wavenet_v3(mode: str = "binary") -> nn.Sequential:
    layers = [
        nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2, 2), nn.Dropout(0.2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2, 2), nn.Dropout(0.2),
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Dropout(0.2),
        nn.Flatten(),
        nn.Linear(128 * 5 * 5, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Dropout(0.4),
        *_final_head(256, mode),
    ]
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
