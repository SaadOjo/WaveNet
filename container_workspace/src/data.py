from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe


def dataset_id_from_name(data_name: str) -> str:
    return data_name[4:6]


def list_dataset_names(data_root: Path, target_ids: Iterable[str] | None = None) -> list[str]:
    target_ids = set(target_ids or [])
    names = [
        path.name
        for path in sorted(data_root.iterdir())
        if path.is_dir() and path.name.startswith("data")
    ]
    if not target_ids:
        return names

    filtered = []
    for name in names:
        for target_id in target_ids:
            if name == f"data{target_id}" or name.startswith(f"data{target_id}_"):
                filtered.append(name)
                break
    return filtered


def get_ts_paths(data_root: Path, data_name: str, problem_name: str = "TokenItaly_vers0") -> tuple[Path, Path]:
    dataset_dir = data_root / data_name / problem_name
    train_path = dataset_dir / f"{problem_name}_TRAIN.ts"
    test_path = dataset_dir / f"{problem_name}_TEST.ts"
    return train_path, test_path


def load_ts_train_test(data_root: Path, data_name: str, problem_name: str = "TokenItaly_vers0"):
    train_path, test_path = get_ts_paths(data_root, data_name, problem_name)
    x_train, y_train = load_from_tsfile_to_dataframe(str(train_path))
    x_test, y_test = load_from_tsfile_to_dataframe(str(test_path))
    return x_train, y_train, x_test, y_test


def expand_and_label(x_df: pd.DataFrame, y_array) -> pd.DataFrame:
    expanded_df = pd.DataFrame(x_df["dim_0"].tolist())
    expanded_df.columns = [f"week{str(i).zfill(2)}" for i in range(expanded_df.shape[1])]
    expanded_df["label"] = y_array
    return expanded_df


def prepare_flattened_binary_data(data_root: Path, data_name: str, problem_name: str = "TokenItaly_vers0"):
    x_train, y_train, x_test, y_test = load_ts_train_test(data_root, data_name, problem_name)
    train_expanded = expand_and_label(x_train, y_train)
    test_expanded = expand_and_label(x_test, y_test)

    X_train = train_expanded.drop(columns=["label"])
    y_train = train_expanded["label"].astype(int)
    X_test = test_expanded.drop(columns=["label"])
    y_test = test_expanded["label"].astype(int)
    return X_train, y_train, X_test, y_test
