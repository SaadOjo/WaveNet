# Session Context

This file is a handoff so work can continue later from a local terminal session.

## Working directory

`/home/ortaboy/Projects/Churn_Prediction/Churn_Wavelet_Refactor`

## What changed in this session

### Repository cleanup
Removed from `container_workspace/`:
- `for_paper_ver1/`
- `for_paper_ver2/`
- `for_paper_ver3/`
- `for_paper_ver4/`
- `for_paper_ver5/`
- `start_jupyter.sh`

Remaining focus area:
- `container_workspace/wavelet_experimentation/`

### Container workflow refactor
Added:
- `compose.yaml`
- `Makefile`

Updated:
- `start_docker.sh`
- `container_operations.sh`

The repo now uses a **persistent Docker dev container** instead of ad hoc manual docker commands.

## Intended workflow

From inside this directory:

```bash
make build
make up
make shell
make ps
make logs
make down
```

Run commands inside the running container with:

```bash
make run CMD="python path/to/script.py --arg value"
```

## Current container design

- Service name: `app`
- Image: `churn-wavelet-refactor:latest`
- Container name: `churn-wavelet-refactor`
- Mounted workspace: `./container_workspace -> /workspace`
- GPU enabled through Docker Compose NVIDIA device reservation
- Container stays alive with `sleep infinity`

## Goal of the refactor

Move away from notebook-driven execution and toward:
- scripts for experiment runs
- configs for parameters
- notebooks only for exploration/analysis

Primary target for refactor:
- `container_workspace/wavelet_experimentation/`

## Suggested next step

Inspect and reorganize:
- notebooks
- results files
- W&B logs
- reusable code candidates

Then decide how to split into something like:
- `src/`
- `scripts/`
- `configs/`
- `runs/`

## Kitty note
This file does **not** preserve the chat itself. It is only a local handoff/context file so the work can be resumed from a Kitty terminal.
