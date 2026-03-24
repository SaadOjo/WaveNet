from __future__ import annotations

import torch.nn as nn
import torchvision.models as tv_models


def get_vit_binary_model(dropout: float = 0.5, unlock_last_block: bool = True):
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
    for param in model.heads.parameters():
        param.requires_grad = True
    return model
