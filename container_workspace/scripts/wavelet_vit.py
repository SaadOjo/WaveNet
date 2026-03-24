from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vit import get_vit_binary_model
from src.io import project_root
from src.runners import run_basic_cwt_experiment


DEFAULT_NOTES = "ViT_B_16 with ImageNet pre-trained weights. Only encoder.layers[-1] + classification head unlocked."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ids", nargs="*", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--wandb-project", default="wavenet-project")
    parser.add_argument("--wandb-group", default="wavelet_vit")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-notes", default=DEFAULT_NOTES)
    args = parser.parse_args()

    root = project_root()
    dropout = 0.5
    unlock_last_block = True
    results, run_dir = run_basic_cwt_experiment(
        model_builder=lambda: get_vit_binary_model(dropout=dropout, unlock_last_block=unlock_last_block),
        model_name="ViT_B_16_TransferLearning",
        data_root=root / "data",
        results_dir=root / "results",
        runs_root=root / "runs",
        result_stem="wavelet_vit",
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
        },
        wandb_enabled=not args.no_wandb,
        gpu_id=args.gpu_id,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_tags=["vit", "wavelet", "transfer-learning"],
        wandb_notes=args.wandb_notes,
    )
    print(f"Saved {len(results)} results to {run_dir}")


if __name__ == "__main__":
    main()
