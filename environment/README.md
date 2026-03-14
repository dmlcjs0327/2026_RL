[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# SHVD — Aerial Gym Simulator

High-fidelity physics-based simulator for training Micro Aerial Vehicle (MAV) platforms (e.g. multirotors) with learning-based methods. Built on [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym).

This repository extends [Aerial Gym Simulator](https://github.com/ntnu-arl/aerial_gym_simulator) for **goal-conditioned reinforcement learning** experiments (TD3/SAC, HER, TDM, CRL, and related baselines) on sparse-reward autonomous drone navigation.

---

## Features

- **Modular design** — Custom environments, robots, sensors, tasks, and controllers; parameters adjustable at runtime.
- **GPU physics** — [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/download) for large-scale multirotor simulation.
- **Parallel geometric controllers** on GPU for controlling many vehicles at once.
- **Custom rendering** ([NVIDIA Warp](https://nvidia.github.io/warp/)) — depth, segmentation, and custom sensors (e.g. LiDAR).
- **RL training** — Scripts and examples for training control and navigation policies.
- **Goal-conditioned task** — `position_setpoint_gc_task` with random goal sampling, sparse reward, and success/collision/timeout logging for off-policy RL.

---

## Documentation

- **[USAGE.md](USAGE.md)** — 사용법·로컬 디렉터리 구조·실행 진입점·태스크/설정 위치 요약 (우선 참고).

Environment-side documentation is organized under **`docs/`** by purpose:

| Directory | Description |
|-----------|-------------|
| [docs/setup/](docs/setup/) | Environment spec (Ubuntu, conda, hardware), terminal commands for running training and replay. |
| [docs/analysis/](docs/analysis/) | Code analysis, test results, and official Aerial Gym docs summary. |
| [docs/reference/](docs/reference/) | Original Aerial Gym reference docs (Getting Started, RL Training, etc.). |

Paper-side planning documents and reference papers now live under **[`../paper/`](../paper/)**.

See **[docs/README.md](docs/README.md)** for a full index and quick links.

---

## Quick start (Ubuntu 22.04, conda)

```bash
conda activate aerialgym   # Python 3.8
cd environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True
```

Goal-conditioned task (random goals, sparse reward):

```bash
python runner.py --task=position_setpoint_gc_task --num_envs=256 --headless=True --use_warp=True
```

See [docs/setup/terminal_commands.md](docs/setup/terminal_commands.md) for full command reference and [docs/setup/environment_spec.md](docs/setup/environment_spec.md) for environment details.

---

## License

BSD-3-Clause. See [../LICENSE](../LICENSE).
