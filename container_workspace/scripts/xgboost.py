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

from src.models.classical import get_xgboost_model
from src.io import project_root
from src.runners import run_classical_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ids", nargs="*", default=None)
    args = parser.parse_args()

    root = project_root()
    results, run_dir = run_classical_experiment(
        model_builder=get_xgboost_model,
        model_name="XGBoost",
        data_root=root / "data",
        results_dir=root / "results",
        runs_root=root / "runs",
        result_stem="xgboost",
        target_ids=args.target_ids,
    )
    print(f"Saved {len(results)} results to {run_dir}")


if __name__ == "__main__":
    main()
