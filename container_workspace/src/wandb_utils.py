from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.io import project_root

_DEFAULT_WANDB_API_KEY = "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"
_DEFAULT_PROJECT = "wavenet-project"


def _to_jsonable(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def ensure_wandb_env(api_key: str | None = None, mode: str | None = None):
    os.environ.setdefault("WANDB_API_KEY", api_key or _DEFAULT_WANDB_API_KEY)
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DIR", str(project_root() / "results" / "wandb"))
    if mode:
        os.environ["WANDB_MODE"] = mode


def init_wandb_run(
    *,
    config: dict,
    run_name: str,
    project: str | None = None,
    entity: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    notes: str | None = None,
    mode: str | None = None,
    api_key: str | None = None,
):
    ensure_wandb_env(api_key=api_key, mode=mode)
    import wandb

    return wandb.init(
        project=project or _DEFAULT_PROJECT,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        notes=notes,
        config=_to_jsonable(config),
        dir=os.environ.get("WANDB_DIR"),
    )
