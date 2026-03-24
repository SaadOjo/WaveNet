from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def _convert_first_conv_to_single_channel(model):
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
    return model


def get_resnext50_binary_model(single_channel: bool = True, unlock_last_block: bool = False, dropout: float = 0.0):
    model = tv_models.resnext50_32x4d(weights=tv_models.ResNeXt50_32X4D_Weights.DEFAULT)
    if single_channel:
        model = _convert_first_conv_to_single_channel(model)

    for param in model.parameters():
        param.requires_grad = False

    if unlock_last_block:
        for param in model.layer4[2].parameters():
            param.requires_grad = True

    num_ftrs = model.fc.in_features
    head_layers = []
    if dropout > 0:
        head_layers.append(nn.Dropout(p=dropout))
    head_layers.extend([nn.Linear(num_ftrs, 1), nn.Sigmoid()])
    model.fc = nn.Sequential(*head_layers)
    for param in model.fc.parameters():
        param.requires_grad = True
    if single_channel:
        for param in model.conv1.parameters():
            param.requires_grad = True
    return model


def get_resnext101_binary_model(single_channel: bool = True):
    model = tv_models.resnext101_32x8d(weights=tv_models.ResNeXt101_32X8D_Weights.DEFAULT)
    if single_channel:
        model = _convert_first_conv_to_single_channel(model)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    for param in model.fc.parameters():
        param.requires_grad = True
    if single_channel:
        for param in model.conv1.parameters():
            param.requires_grad = True
    return model


class SimpleSignalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        return self.fc(x)


def get_resnext50_v2_model():
    return SimpleSignalCNN()
