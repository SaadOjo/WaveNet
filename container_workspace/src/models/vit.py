from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


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


def get_vit_binary_model(dropout: float = 0.5, unlock_last_block: bool = True, state_dict: dict | None = None):
    model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    if unlock_last_block:
        for param in model.encoder.layers[-1].parameters():
            param.requires_grad = True
    num_ftrs = model.heads.head.in_features
    model.heads = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid(),
    )
    if state_dict is not None:
        model = load_matching_weights(model, state_dict)
    for param in model.heads.parameters():
        param.requires_grad = True
    return model
