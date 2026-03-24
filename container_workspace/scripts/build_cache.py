from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import list_dataset_names, load_ts_train_test
from src.datasets import load_or_build_cwt_dataset
from src.io import project_root


def main():
    parser = argparse.ArgumentParser(description="Build raw CWT cache without training models.")
    parser.add_argument("--target-ids", nargs="*", default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    root = project_root()
    data_root = root / "data"
    cache_dir = data_root / "cache"
    dataset_names = list_dataset_names(data_root, args.target_ids)

    for data_name in dataset_names:
        print(f"building raw cache for {data_name}")
        x_train, y_train, x_test, y_test = load_ts_train_test(data_root, data_name)
        _, train_cache_path, train_hit = load_or_build_cwt_dataset(
            x_df=x_train,
            y_list=y_train,
            cache_dir=cache_dir,
            dataset_name=data_name,
            split_name="train",
            wavelet="morl",
            scales=list(range(1, 21)),
            rebuild_cache=args.rebuild_cache,
            use_cache=True,
        )
        _, test_cache_path, test_hit = load_or_build_cwt_dataset(
            x_df=x_test,
            y_list=y_test,
            cache_dir=cache_dir,
            dataset_name=data_name,
            split_name="test",
            wavelet="morl",
            scales=list(range(1, 21)),
            rebuild_cache=args.rebuild_cache,
            use_cache=True,
        )
        print(
            f"  train={'hit' if train_hit else 'built'}: {train_cache_path.name}\n"
            f"  test ={'hit' if test_hit else 'built'}: {test_cache_path.name}"
        )

    print("\nRaw cache build complete.")


if __name__ == "__main__":
    main()
