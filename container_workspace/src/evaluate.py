from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_binary_model(model, dataloader, device, threshold: float = 0.5) -> dict:
    model.eval()
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y_true = y.to(device).float().view(-1)
            outputs = model(x)
            probs = outputs.view(-1).float()
            preds = (probs >= threshold).float()
            y_true_all.append(y_true.cpu())
            y_pred_all.append(preds.cpu())
            y_proba_all.append(probs.cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    y_proba = torch.cat(y_proba_all).numpy()

    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC score": auc,
    }


def evaluate_multiclass_model(model, dataloader, device) -> dict:
    model.eval()
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y_true = y.to(device).long()
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            y_true_all.append(y_true.cpu())
            y_pred_all.append(preds.cpu())
            y_proba_all.append(probs.cpu())

    y_true = torch.cat(y_true_all).numpy()
    y_pred = torch.cat(y_pred_all).numpy()
    y_proba = torch.cat(y_proba_all).numpy()

    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr")
    except ValueError:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC score": auc,
    }


def make_result_dict(data_name: str, model_name: str, metrics: dict) -> dict:
    data_id = data_name[4:6] if data_name.startswith("data") else data_name
    return {
        "data_id": data_id,
        "data_name": data_name,
        "model": model_name,
        **metrics,
    }
