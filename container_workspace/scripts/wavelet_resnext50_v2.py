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

from src.models.resnext import get_resnext50_v2_model
from src.io import project_root
from src.runners import run_basic_cwt_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ids", nargs="*", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    root = project_root()
    results, run_dir = run_basic_cwt_experiment(
        model_builder=get_resnext50_v2_model,
        model_name="ResNeXt50_TransferLearning-v2",
        data_root=root / "data",
        results_dir=root / "results",
        runs_root=root / "runs",
        result_stem="wavelet_resnext50_v2",
        target_ids=args.target_ids,
        rebuild_cache=args.rebuild_cache,
        use_cache=not args.no_cache,
        batch_size=16,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        step_size=8,
        gamma=0.3,
        resize_to=(224, 224),
        colorize=True,
        patience=20,
        label_smoothing=0.1,
    )
    print(f"Saved {len(results)} results to {run_dir}")


if __name__ == "__main__":
    main()
