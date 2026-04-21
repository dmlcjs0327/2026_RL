# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

- `UAV_RL_PretrainAdapt_Proposal.md` (top-level) — **Canonical research-direction document** (2026-04~). Single source of truth for research scope, methodology, experimental design, risk management. When questions about "what is this project" / "what are we doing" come up, read this first. Written in Korean.
- `environment/` — Aerial Gym simulator (fork of `ntnu-arl/aerial_gym_simulator` v2.0.0) + RL training code. This is where all experiment work happens. Upstream vs local-additions diff is tracked in `environment/docs/upstream_diff.md`.
- `environment2/` — **ignored working copy / scratch duplicate** (Flightmare 0.0.5 legacy). Do not edit here unless explicitly asked.
- `paper/` — **(deprecated)** Previous research direction's LaTeX manuscript workspace. Files are deleted from the index (uncommitted); the directory is effectively empty and the proposal above has replaced it. Do not resurrect paper/ unless the user explicitly asks — when writing/paper tasks come up, work from `UAV_RL_PretrainAdapt_Proposal.md` instead.

The **target venue** for the current research direction is **Drones (MDPI, IF 4.8)** — not RA-L or T-RO. Drone models target real-hardware constraints (Jetson Orin Nano onboard, X500-class quadrotor via Aerial Gym v2's official Sim2Real path using Holybro X500 v2 + Pixhawk Jetson Baseboard).

## Development environment

- Conda env **`aerialgym`** (Python 3.8), built on NVIDIA Isaac Gym + Warp.
- Before running anything under `environment/`:
  ```bash
  conda activate aerialgym
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
  # Only if Vulkan / rgbImage buffer error 999:
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
  ```
- Package install: `pip install -e .` from `environment/` (defined in `environment/setup.py`, depends on `isaacgym`, `warp-lang==1.0.0`, `numpy==1.23`, `rl-games`, `sample-factory`, `pytorch3d`).
- Isaac Gym quirk: `gymutil.py` line 337 should be patched `parse_args()` → `parse_known_args()`.

## RL training entry points

**Current research direction = PPO-only via Sample Factory** (proposal §2.5 scope). The Sample Factory integration with custom `pretrain_adapt_task` is specified by the proposal's W1~W11 work items (§4.9.2) and is built out during M1. Until that pipeline lands, only the upstream-derived `rl_games` runner is wired up.

### `rl_games` runner — upstream PPO baseline (currently the only working entry)

```bash
cd environment/aerial_gym/rl_training/rl_games
python runner.py --task=navigation_task --num_envs=256 --headless=True --use_warp=True
```

- **Must be run from `rl_games/` directory.** The runner writes checkpoints to `rl_games/runs/<experiment>/nn/*.pth` relative to the CWD.
- Replay: same command with `--num_envs=1 --headless=False --play --checkpoint=./runs/<exp>/nn/<file>.pth`.
- YAMLs in the same folder (`ppo_*.yaml`).

### Legacy runners (retained, NOT the current research direction)

The following paths exist from the prior goal-conditioned sparse-reward direction and should **not** be used for new experiments under the pretrain-adapt research scope. They are kept for code archaeology and possible cross-reference:

- `environment/aerial_gym/rl_training/our_algorithms/` — self-hosted modular off-policy runner (TD3/SAC/DDPG variants, HER, TDM, CRL, SSL pretraining). Drivers: `run_all_experiments.py`, `run_baseline_vs_proposed.py`, `run_corridor_baselines.py`, `run_her_ablation.py`, `run_matrix.py`, `run_proposed_only.py`, `run_sparse_test.py`.
- `environment/aerial_gym/rl_training/offpolicy/` — predecessor of `our_algorithms/`, SAC/TD3/DDPG/HER/TDM/CRL configs + `train.py`.

If the user explicitly asks to rerun a legacy off-policy experiment, use these; otherwise default new RL work to Sample Factory.

## Registries — how tasks and envs get wired up

Two import-time registration points are the source of truth for what you can pass as `--task` and `env_name`. If you add a task or env and it isn't listed below, it won't resolve.

- **Task registry** — `environment/aerial_gym/task/__init__.py` registers every task name (`position_setpoint_task`, `position_setpoint_gc_task`, `navigation_task`, sim2real variants, etc.) onto `task_registry`. A task name is the pairing of a `*Task` class and a `task_config` module from `config/task_config/`.
- **Env registry** — `environment/aerial_gym/env_manager/__init__.py` registers every `env_name` (`empty_env`, `env_with_obstacles`, `forest_env`, plus custom ones). The custom envs for this project's experiments are in `environment/aerial_gym/our_environment/` (`gc_empty_env`, `corridor_env`, `docking_env`) — this directory exists specifically so paper experiments don't modify the upstream env configs.

To switch the scene under an existing task, edit `env_name` in the task's config file (`config/task_config/<task>_config.py`). To add a new task or env, add the class **and** register it in the matching `__init__.py`.

## Current research task — `navigation_task` + custom `pretrain_adapt_task`

The primary task family for the current direction follows Aerial Gym's `navigation_task` standard I/O:
- **Obs**: 81-d = 17-d state (goal unit vec 3 + distance 1 + euler 3 + lin_vel 3 + ang_vel 3 + prev_action 4) + 64-d frozen β-VAE latent from 270×480 depth camera.
- **Action**: 3-d velocity command, consumed by `lmf2_velocity_control` controller.
- **Robot**: `lmf2` (camera-equipped planar quadrotor).
- **4 downstream tasks** (proposal §5.3): D1 Dense Forest, D2 Corridor, D3 Precision Stationkeeping (0.3 m + velocity stop + 5 s dwell, custom success logic W5), D4 Dynamic Obstacles (W6).

The `pretrain_adapt_task` class and supporting modules (custom Policy with 2-branch input + GRU, custom `LoRALinear`, `build_policy(mode=...)` factory) are specified in proposal §4.9 W1~W11.

**Legacy task for reference only**: `position_setpoint_gc_task` (class `PositionSetpointGCTask`, config `position_setpoint_gc_task_config.py`) belongs to the prior research direction (goal-conditioned sparse-reward + HER/TDM/CRL). Do not use for new pretrain-adapt experiments.

## Running analysis / replay

- TensorBoard: `tensorboard --logdir environment/runs/` (subdirs: `rl_games/runs/`, legacy `our_algorithms/`).
- Env sanity check without training: `python environment/scripts/env_api_test.py` from the `environment/` root.
- Legacy log-analysis scripts (pre-proposal): `our_algorithms/analyze_learning_logs.py`, `our_algorithms/deep_log_analysis.py`.

## Project conventions

- Docs and commit-review memos are written in **Korean**. Keep that language for `UAV_RL_PretrainAdapt_Proposal.md` and `environment/docs/` unless the file is already English.
- **Git**: when the user asks "깃허브에 올려줘" / "push to GitHub", push to the `origin` remote (`git push origin main` → `github.com/dmlcjs0327/2026_RL.git`). Commit messages should be in English. The legacy `shvd` remote (github.com/dmlcjs0327/SHVD.git) still exists but is no longer the primary target. This is defined in `.cursor/rules/github-push.mdc`.
- When staging a commit for the user, only stage files intentionally changed for the task at hand. The working tree carries pre-existing uncommitted state (e.g. `paper/*` deletions, pre-existing environment/aerial_gym edits) from prior sessions — do not drag those into your commits unless the user explicitly asks.
