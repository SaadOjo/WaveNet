from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io import project_root
from src.models.vit import get_vit_binary_model
from src.runners import pretrain_binary_cwt, run_basic_cwt_experiment

PRETRAIN_NOTES = "ViT_B_16: ImageNet init, then churn pretraining on dataset 20. Only last encoder block + head trainable."
TL_NOTES = "ViT_B_16 transfer run initialized from ImageNet+dataset20 checkpoint. Only last encoder block + head trainable."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ids", nargs="*", default=None)
    parser.add_argument("--pretrain-target-ids", nargs="*", default=["20"])
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--retrain-base", action="store_true")
    parser.add_argument("--pretrain-only", action="store_true")
    parser.add_argument("--result-stem", default="wavelet_vit_tl20")
    parser.add_argument("--model-name", default="ViT_B_16_TL20")
    parser.add_argument("--pretrain-result-stem", default="wavelet_vit_pretrain20")
    parser.add_argument("--pretrain-model-name", default="ViT_B_16_Pretrain20")
    parser.add_argument("--wandb-project", default="wavenet-project")
    parser.add_argument("--wandb-group", default="wavelet_vit_tl20")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-notes", default=TL_NOTES)
    parser.add_argument("--pretrain-wandb-group", default="wavelet_vit_pretrain20")
    parser.add_argument("--pretrain-wandb-notes", default=PRETRAIN_NOTES)
    args = parser.parse_args()

    root = project_root()
    dropout = 0.5
    unlock_last_block = True
    weights_path = Path(args.weights_path) if args.weights_path else root / "weights" / "base_weights_vit_tl20.pth"

    base_weights = None
    if weights_path.exists() and not args.retrain_base:
        base_weights = torch.load(weights_path, map_location="cpu")
    else:
        base_weights, _, _ = pretrain_binary_cwt(
            model_builder=lambda state_dict=None: get_vit_binary_model(
                dropout=dropout,
                unlock_last_block=unlock_last_block,
                state_dict=state_dict,
            ),
            model_name=args.pretrain_model_name,
            data_root=root / "data",
            results_dir=root / "results",
            runs_root=root / "runs",
            result_stem=args.pretrain_result_stem,
            weights_path=weights_path,
            target_ids=args.pretrain_target_ids,
            rebuild_cache=args.rebuild_cache,
            use_cache=not args.no_cache,
            batch_size=16,
            epochs=100,
            lr=1e-4,
            weight_decay=1e-4,
            step_size=8,
            gamma=0.3,
            resize_to=(224, 224),
            colorize=True,
            norm_type="none",
            patience=20,
            label_smoothing=0.1,
            optimizer_trainable_only=True,
            model_config={
                "backbone": "vit_b_16",
                "dropout": dropout,
                "unlock_last_block": unlock_last_block,
                "pretrained_weights": "ViT_B_16_Weights.DEFAULT",
                "phase": "pretrain20",
            },
            wandb_enabled=not args.no_wandb,
            gpu_id=args.gpu_id,
            wandb_project=args.wandb_project,
            wandb_group=args.pretrain_wandb_group,
            wandb_mode=args.wandb_mode,
            wandb_tags=["vit", "wavelet", "pretrain20", "tl20"],
            wandb_notes=args.pretrain_wandb_notes,
        )

    if args.pretrain_only:
        print(f"Base weights ready at {weights_path}")
        return

    results, run_dir = run_basic_cwt_experiment(
        model_builder=lambda: get_vit_binary_model(
            dropout=dropout,
            unlock_last_block=unlock_last_block,
            state_dict=base_weights,
        ),
        model_name=args.model_name,
        data_root=root / "data",
        results_dir=root / "results",
        runs_root=root / "runs",
        result_stem=args.result_stem,
        target_ids=args.target_ids,
        rebuild_cache=args.rebuild_cache,
        use_cache=not args.no_cache,
        batch_size=16,
        epochs=100,
        lr=1e-4,
        weight_decay=1e-4,
        step_size=8,
        gamma=0.3,
        resize_to=(224, 224),
        colorize=True,
        norm_type="none",
        patience=20,
        label_smoothing=0.1,
        optimizer_trainable_only=True,
        model_config={
            "backbone": "vit_b_16",
            "dropout": dropout,
            "unlock_last_block": unlock_last_block,
            "pretrained_weights": "ViT_B_16_Weights.DEFAULT",
            "weights_path": str(weights_path),
            "phase": "transfer_from_20",
        },
        wandb_enabled=not args.no_wandb,
        gpu_id=args.gpu_id,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_tags=["vit", "wavelet", "transfer-learning", "tl20"],
        wandb_notes=args.wandb_notes,
    )
    print(f"Saved {len(results)} results to {run_dir}")


if __name__ == "__main__":
    main()
