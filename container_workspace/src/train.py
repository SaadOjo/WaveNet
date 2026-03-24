from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from src.evaluate import evaluate_binary_model, evaluate_multiclass_model


def train_binary_model(
    model,
    train_loader,
    device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    step_size: int,
    gamma: float,
    val_loader=None,
    patience: int | None = None,
    label_smoothing: float = 0.0,
    optimizer_trainable_only: bool = False,
    wandb_run=None,
):
    model = model.to(device)
    criterion = nn.BCELoss()
    params = filter(lambda p: p.requires_grad, model.parameters()) if optimizer_trainable_only else model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = -1.0
    epochs_without_improvement = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float().unsqueeze(1)
            if label_smoothing > 0:
                y_target = y * (1 - label_smoothing) + (label_smoothing / 2)
            else:
                y_target = y

            outputs = model(x)
            loss = criterion(outputs, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = (outputs > 0.5).float()
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train/loss": running_loss / max(total_train, 1),
            "train/accuracy": correct_train / max(total_train, 1),
            "lr": current_lr,
        }

        if val_loader is not None:
            metrics = evaluate_binary_model(model, val_loader, device)
            epoch_log.update(
                {
                    "val/accuracy": metrics["Accuracy"],
                    "val/precision": metrics["Precision"],
                    "val/recall": metrics["Recall"],
                    "val/f1": metrics["F1-score"],
                    "val/roc_auc": metrics["ROC-AUC score"],
                }
            )
            accuracy = metrics["Accuracy"]
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                if wandb_run is not None:
                    wandb_run.summary["best_accuracy"] = best_acc
                    wandb_run.summary["best_epoch"] = epoch
                    wandb_run.summary["best_precision"] = metrics["Precision"]
                    wandb_run.summary["best_recall"] = metrics["Recall"]
                    wandb_run.summary["best_f1"] = metrics["F1-score"]
                    wandb_run.summary["best_roc_auc"] = metrics["ROC-AUC score"]
            else:
                epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                if wandb_run is not None:
                    wandb_run.log(epoch_log)
                break

        if wandb_run is not None:
            wandb_run.log(epoch_log)

    if val_loader is not None:
        model.load_state_dict(best_model_wts)
    return model, model.state_dict()


def train_multiclass_model(
    model,
    train_loader,
    device,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    step_size: int,
    gamma: float,
    val_loader=None,
    patience: int | None = None,
    wandb_run=None,
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = -1.0
    epochs_without_improvement = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train/loss": running_loss / max(total_train, 1),
            "train/accuracy": correct_train / max(total_train, 1),
            "lr": current_lr,
        }

        if val_loader is not None:
            metrics = evaluate_multiclass_model(model, val_loader, device)
            epoch_log.update(
                {
                    "val/accuracy": metrics["Accuracy"],
                    "val/precision": metrics["Precision"],
                    "val/recall": metrics["Recall"],
                    "val/f1": metrics["F1-score"],
                    "val/roc_auc": metrics["ROC-AUC score"],
                }
            )
            accuracy = metrics["Accuracy"]
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                if wandb_run is not None:
                    wandb_run.summary["best_accuracy"] = best_acc
                    wandb_run.summary["best_epoch"] = epoch
            else:
                epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                if wandb_run is not None:
                    wandb_run.log(epoch_log)
                break

        if wandb_run is not None:
            wandb_run.log(epoch_log)

    if val_loader is not None:
        model.load_state_dict(best_model_wts)
    return model, model.state_dict()
