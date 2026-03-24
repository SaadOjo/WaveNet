from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def create_run_dir(script_name: str, runs_root: Path | None = None) -> Path:
    if runs_root is None:
        runs_root = project_root() / "runs"
    run_dir = runs_root / script_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_run_summary(run_dir: Path, results: list[dict], config: dict):
    (run_dir / "config.json").write_text(json.dumps(_to_jsonable(config), indent=2))
    (run_dir / "results.json").write_text(json.dumps(_to_jsonable(results), indent=2))
    pd.DataFrame(results).to_csv(run_dir / "results.csv", index=False)


def append_results(results_dir: Path, stem: str, new_results: list[dict]):
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{stem}.json"
    csv_path = results_dir / f"{stem}.csv"

    if json_path.exists():
        existing = json.loads(json_path.read_text())
    else:
        existing = []
    existing.extend(new_results)
    json_path.write_text(json.dumps(_to_jsonable(existing), indent=2))
    pd.DataFrame(existing).to_csv(csv_path, index=False)
