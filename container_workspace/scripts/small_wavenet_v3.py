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

from src.io import project_root
from src.models.cnn import make_small_wavenet_v3
from src.runners import run_basic_cwt_experiment

DEFAULT_NOTES = "SmallWaveNetV3 from legacy notebook v3 architecture, trained without transfer learning."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ids", nargs="*", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--result-stem", default="small_wavenet_v3")
    parser.add_argument("--model-name", default="SmallWaveNetV3")
    parser.add_argument("--wandb-project", default="wavenet-project")
    parser.add_argument("--wandb-group", default="small_wavenet_v3")
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--wandb-notes", default=DEFAULT_NOTES)
    args = parser.parse_args()

    root = project_root()
    results, run_dir = run_basic_cwt_experiment(
        model_builder=lambda: make_small_wavenet_v3(mode="binary"),
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
            "input_shape": [1, 20, 20],
            "conv_blocks": [[32, 32], [64, 64], [128, 128]],
            "fc": [256, 1],
            "transfer_learning": False,
        },
        wandb_enabled=not args.no_wandb,
        gpu_id=args.gpu_id,
        wandb_project=args.wandb_project,
        wandb_group=args.wandb_group,
        wandb_mode=args.wandb_mode,
        wandb_tags=["small-wavenet", "v3", "wavelet", "no-tl"],
        wandb_notes=args.wandb_notes,
    )
    print(f"Saved {len(results)} results to {run_dir}")


if __name__ == "__main__":
    main()
