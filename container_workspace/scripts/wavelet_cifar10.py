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

from src.models.cnn import load_matching_weights, make_wavenet
from src.io import project_root
from src.runners import pretrain_on_cifar, run_basic_cwt_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ids", nargs="*", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--weights-path", default=None)
    parser.add_argument("--retrain-base", action="store_true")
    parser.add_argument("--pretrain-only", action="store_true")
    args = parser.parse_args()

    root = project_root()
    weights_path = Path(args.weights_path) if args.weights_path else root / "weights" / "base_weights_cifar.pth"

    if weights_path.exists() and not args.retrain_base:
        base_weights = torch.load(weights_path, map_location="cpu")
    else:
        base_weights, _, _ = pretrain_on_cifar(
            model_builder=make_wavenet,
            model_name="WaveNet_Base_CIFAR10",
            weights_path=weights_path,
            results_dir=root / "results",
            runs_root=root / "runs",
            result_stem="wavelet_cifar10_pretrain",
        )

    if args.pretrain_only:
        print(f"Base weights ready at {weights_path}")
        return

    def model_builder():
        model = make_wavenet(mode="binary")
        return load_matching_weights(model, base_weights)

    results, run_dir = run_basic_cwt_experiment(
        model_builder=model_builder,
        model_name="WaveNet_TL_CIFAR",
        data_root=root / "data",
        results_dir=root / "results",
        runs_root=root / "runs",
        result_stem="wavelet_cifar10",
        target_ids=args.target_ids,
        rebuild_cache=args.rebuild_cache,
        use_cache=not args.no_cache,
        batch_size=32,
        epochs=150,
        lr=1e-3,
        weight_decay=1e-4,
        step_size=8,
        gamma=0.3,
    )
    print(f"Saved {len(results)} results to {run_dir}")


if __name__ == "__main__":
    main()
