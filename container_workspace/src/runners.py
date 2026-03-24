from __future__ import annotations

from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data import list_dataset_names, load_ts_train_test, prepare_flattened_binary_data
from src.datasets import CIFAR10GrayDataset, load_or_build_cwt_dataset
from src.evaluate import evaluate_binary_model, evaluate_multiclass_model, make_result_dict
from src.io import append_results, create_run_dir, save_run_summary
from src.train import train_binary_model, train_multiclass_model
from src.wandb_utils import init_wandb_run


def _resolve_device(gpu_id: int | None = None):
    if torch.cuda.is_available():
        if gpu_id is None:
            return torch.device("cuda")
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def run_classical_experiment(*, model_builder, model_name: str, data_root: Path, results_dir: Path, runs_root: Path, result_stem: str, target_ids=None):
    dataset_names = list_dataset_names(data_root, target_ids)
    results = []
    for data_name in dataset_names:
        X_train, y_train, X_test, y_test = prepare_flattened_binary_data(data_root, data_name)
        model = model_builder()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = float("nan")
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC score": auc,
        }
        results.append(make_result_dict(data_name, model_name, metrics))

    run_dir = create_run_dir(result_stem, runs_root)
    config = {"model": model_name, "target_ids": list(target_ids or []), "datasets": dataset_names}
    save_run_summary(run_dir, results, config)
    append_results(results_dir, result_stem, results)
    return results, run_dir


def run_basic_cwt_experiment(
    *,
    model_builder,
    model_name: str,
    data_root: Path,
    results_dir: Path,
    runs_root: Path,
    result_stem: str,
    target_ids=None,
    batch_size: int = 32,
    wavelet: str = "morl",
    scales=None,
    epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    step_size: int = 8,
    gamma: float = 0.3,
    norm_type: str = "none",
    colorize: bool = False,
    resize_to=None,
    patience: int | None = None,
    label_smoothing: float = 0.0,
    optimizer_trainable_only: bool = False,
    cache_dir: Path | None = None,
    rebuild_cache: bool = False,
    use_cache: bool = True,
    model_config: dict | None = None,
    wandb_enabled: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    wandb_notes: str | None = None,
    wandb_mode: str | None = None,
    gpu_id: int | None = None,
):
    if scales is None:
        scales = list(range(1, 21))
    if cache_dir is None:
        cache_dir = data_root / "cache"
    model_config = model_config or {}

    dataset_names = list_dataset_names(data_root, target_ids)
    results = []
    device = _resolve_device(gpu_id)

    base_config = {
        "model": model_name,
        "target_ids": list(target_ids or []),
        "datasets": dataset_names,
        "batch_size": batch_size,
        "wavelet": wavelet,
        "scales": list(scales),
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "step_size": step_size,
        "gamma": gamma,
        "norm_type": norm_type,
        "colorize": colorize,
        "resize_to": resize_to,
        "patience": patience,
        "label_smoothing": label_smoothing,
        "optimizer_trainable_only": optimizer_trainable_only,
        "cache_dir": str(cache_dir),
        "use_cache": use_cache,
        "rebuild_cache": rebuild_cache,
        "device": str(device),
        "gpu_id": gpu_id,
        "model_config": model_config,
        "wandb_enabled": wandb_enabled,
        "wandb_project": wandb_project,
        "wandb_group": wandb_group,
        "wandb_tags": wandb_tags,
    }

    for data_name in dataset_names:
        x_train, y_train, x_test, y_test = load_ts_train_test(data_root, data_name)
        train_dataset, train_cache_path, train_cache_hit = load_or_build_cwt_dataset(
            x_df=x_train,
            y_list=y_train,
            cache_dir=cache_dir,
            dataset_name=data_name,
            split_name="train",
            wavelet=wavelet,
            scales=scales,
            new_size=resize_to,
            norm_type=norm_type,
            colorize=colorize,
            rebuild_cache=rebuild_cache,
            use_cache=use_cache,
        )
        val_dataset, val_cache_path, val_cache_hit = load_or_build_cwt_dataset(
            x_df=x_test,
            y_list=y_test,
            cache_dir=cache_dir,
            dataset_name=data_name,
            split_name="test",
            wavelet=wavelet,
            scales=scales,
            new_size=resize_to,
            norm_type=norm_type,
            colorize=colorize,
            rebuild_cache=rebuild_cache,
            use_cache=use_cache,
        )

        print(
            f"[{data_name}] cache train={'hit' if train_cache_hit else 'build'} ({train_cache_path.name}), "
            f"test={'hit' if val_cache_hit else 'build'} ({val_cache_path.name})"
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        run_config = {
            **base_config,
            "dataset_name": data_name,
            "data_id": data_name[4:6] if data_name.startswith("data") else data_name,
            "train_size": len(train_dataset),
            "test_size": len(val_dataset),
            "train_cache_path": str(train_cache_path),
            "test_cache_path": str(val_cache_path),
            "train_cache_hit": train_cache_hit,
            "test_cache_hit": val_cache_hit,
        }

        wandb_run = None
        if wandb_enabled:
            wandb_run = init_wandb_run(
                config=run_config,
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group or result_stem,
                tags=(wandb_tags or []) + [result_stem, data_name],
                notes=wandb_notes,
                mode=wandb_mode,
                run_name=f"{result_stem}-{data_name}",
            )

        try:
            model = model_builder()
            model, _ = train_binary_model(
                model,
                train_loader,
                device,
                epochs=epochs,
                lr=lr,
                weight_decay=weight_decay,
                step_size=step_size,
                gamma=gamma,
                val_loader=val_loader if patience is not None else None,
                patience=patience,
                label_smoothing=label_smoothing,
                optimizer_trainable_only=optimizer_trainable_only,
                wandb_run=wandb_run,
            )
            metrics = evaluate_binary_model(model, val_loader, device)
            result = make_result_dict(data_name, model_name, metrics)
            results.append(result)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "final/accuracy": metrics["Accuracy"],
                        "final/precision": metrics["Precision"],
                        "final/recall": metrics["Recall"],
                        "final/f1": metrics["F1-score"],
                        "final/roc_auc": metrics["ROC-AUC score"],
                    }
                )
                wandb_run.summary["result"] = result
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    run_dir = create_run_dir(result_stem, runs_root)
    save_run_summary(run_dir, results, base_config)
    append_results(results_dir, result_stem, results)
    return results, run_dir


def pretrain_on_cifar(*, model_builder, model_name: str, weights_path: Path, results_dir: Path, runs_root: Path, result_stem: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = CIFAR10GrayDataset(train=True, img_size=(20, 20))
    val_dataset = CIFAR10GrayDataset(train=False, img_size=(20, 20))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = model_builder(mode="CIFAR")
    model, state_dict = train_multiclass_model(
        model,
        train_loader,
        device,
        epochs=150,
        lr=1e-3,
        weight_decay=1e-4,
        step_size=8,
        gamma=0.3,
        val_loader=val_loader,
    )
    metrics = evaluate_multiclass_model(model, val_loader, device)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, weights_path)

    results = [make_result_dict("CIFAR-10", model_name, metrics)]
    run_dir = create_run_dir(result_stem, runs_root)
    save_run_summary(run_dir, results, {"model": model_name, "weights_path": str(weights_path)})
    append_results(results_dir, result_stem, results)
    return state_dict, results, run_dir
