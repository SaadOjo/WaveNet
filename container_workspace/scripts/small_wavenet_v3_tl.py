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
from src.models.cnn import load_matching_weights, make_small_wavenet_v3
from src.runners import pretrain_binary_cwt, run_basic_cwt_experiment

DEFAULT_PRETRAIN_NOTES = "SmallWaveNetV3 binary pretraining on churn data before transfer runs."
DEFAULT_TL_NOTES = "SmallWaveNetV3 transfer learning run initialized from churn-pretrained base weights."


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
    parser.add_argument("--result-stem", default="small_wavenet_v3_tl")
    parser.add_argument("--model-name", default="SmallWaveNetV3_TL")
    parser.add_argument("--pretrain-result-stem", default="small_wavenet_v3_pretrain")
    parser.add_argument("--pretrain-model-name", default="SmallWaveNetV3_Base")
    parser.add_argument("--wandb-project", default="wavenet-project")
    parser.add_argument("--wandb-group", default="small_wavenet_v3_tl")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-notes", default=DEFAULT_TL_NOTES)
    parser.add_argument("--pretrain-wandb-group", default="small_wavenet_v3_pretrain")
    parser.add_argument("--pretrain-wandb-notes", default=DEFAULT_PRETRAIN_NOTES)
    args = parser.parse_args()

    root = project_root()
    weights_path = Path(args.weights_path) if args.weights_path else root / "weights" / "base_weights_small_wavenet_v3.pth"

    base_weights = None
    if weights_path.exists() and not args.retrain_base:
        base_weights = torch.load(weights_path, map_location="cpu")
    else:
        base_weights, _, _ = pretrain_binary_cwt(
            model_builder=lambda state_dict=None: load_matching_weights(make_small_wavenet_v3(mode="binary"), state_dict)
            if state_dict is not None
            else make_small_wavenet_v3(mode="binary"),
            model_name=args.pretrain_model_name,
            data_root=root / "data",
            results_dir=root / "results",
            runs_root=root / "runs",
            result_stem=args.pretrain_result_stem,
            weights_path=weights_path,
            target_ids=args.pretrain_target_ids,
            rebuild_cache=args.rebuild_cache,
            use_cache=not args.no_cache,
            batch_size=64,
            epochs=150,
            lr=1e-3,
            weight_decay=1e-4,
            step_size=8,
            gamma=0.3,
            norm_type="none",
            colorize=False,
            resize_to=None,
            patience=40,
            label_smoothing=0.0,
            optimizer_trainable_only=False,
            model_config={
                "variant": "v3",
                "phase": "binary_pretrain",
                "input_shape": [1, 20, 20],
                "conv_blocks": [[32, 32], [64, 64], [128, 128]],
                "fc": [256, 1],
                "transfer_learning": False,
                "pretrain_target_ids": args.pretrain_target_ids,
            },
            wandb_enabled=not args.no_wandb,
            gpu_id=args.gpu_id,
            wandb_project=args.wandb_project,
            wandb_group=args.pretrain_wandb_group,
            wandb_mode=args.wandb_mode,
            wandb_tags=["small-wavenet", "v3", "pretrain", "binary-churn"],
            wandb_notes=args.pretrain_wandb_notes,
        )

    if args.pretrain_only:
        print(f"Base weights ready at {weights_path}")
        return

    def model_builder():
        model = make_small_wavenet_v3(mode="binary")
        return load_matching_weights(model, base_weights)

    results, run_dir = run_basic_cwt_experiment(
        model_builder=model_builder,
        model_name=args.model_name,
        data_root=root / "data",
        results_dir=root / "results",
        runs_root=root / "runs",
        result_stem=args.result_stem,
        target_ids=args.target_ids,
        rebuild_cache=args.rebuild_cache,
        use_cache=not args.no_cache,
        batch_size=64,
        epochs=150,
        lr=1e-3,
        weight_decay=1e-4,
        step_size=8,
        gamma=0.3,
        norm_type="none",
        colorize=False,
        resize_to=None,
        patience=40,
        label_smoothing=0.0,
        optimizer_trainable_only=False,
        model_config={
            "variant": "v3",
            "phase": "transfer_run",
            "input_shape": [1, 20, 20],
            "conv_blocks": [[32, 32], [64, 64], [128, 128]],
            "fc": [256, 1],
            "transfer_learning": True,
            "weights_path": str(weights_path),
            "pretrain_target_ids": args.pretrain_target_ids,
        },
        wandb_enabled=not args.no_wandb,
        gpu_id=args.gpu_id,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_tags=["small-wavenet", "v3", "wavelet", "tl", "binary-churn"],
        wandb_notes=args.wandb_notes,
    )
    print(f"Saved {len(results)} results to {run_dir}")


if __name__ == "__main__":
    main()
