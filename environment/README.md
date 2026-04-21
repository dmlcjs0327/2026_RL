[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# Environment — Aerial Gym Simulator (fork)

High-fidelity physics-based simulator for training Micro Aerial Vehicle (MAV) platforms with learning-based methods. Built on [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym).

This is a fork of [Aerial Gym Simulator v2.0.0](https://github.com/ntnu-arl/aerial_gym_simulator) customized for the **UAV Pretrain-then-Adapt research direction** — see the top-level [`UAV_RL_PretrainAdapt_Proposal.md`](../UAV_RL_PretrainAdapt_Proposal.md) for the research framing and [`docs/upstream_diff.md`](docs/upstream_diff.md) for the exact additions over upstream.

---

## Research use

Primary path for the current research direction:

- **Task family**: Aerial Gym `navigation_task` standard I/O (81-d obs = 17-d state + 64-d VAE latent, 3-d velocity command) + custom `pretrain_adapt_task` (in progress, proposal §4.9 W1).
- **Robot**: `lmf2` (camera-equipped planar quadrotor).
- **Controller**: `lmf2_velocity_control`.
- **RL library**: Sample Factory 2.1.x (recurrent PPO).
- **Policy**: ~280K params, Jetson Orin Nano 온보드 추론 대상.
- **Adaptation strategies under comparison**: Full Fine-Tuning, Input-Branch-Frozen, LoRA (Linear-only default).

Legacy paths retained for reference but **not** the main focus of the current direction:

- `aerial_gym/rl_training/our_algorithms/` — 이전 goal-conditioned sparse-reward 연구에서 사용한 자체 off-policy 러너 (TD3/SAC/DDPG/HER/TDM/CRL + SSL pretraining).
- `aerial_gym/rl_training/offpolicy/` — 위 `our_algorithms/` 의 전신.
- `aerial_gym/task/position_setpoint_gc_task/` — 이전 방향의 goal-conditioned 태스크.

현재 연구 방향은 PPO only (제안서 §2.5 scope)이므로 위 legacy 모듈은 신규 실험 대상이 아니다.

---

## Features

- **Modular design** — Custom environments, robots, sensors, tasks, and controllers; parameters adjustable at runtime.
- **GPU physics** — [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym/download) for large-scale multirotor simulation (~4096 parallel envs with Warp backend).
- **Parallel geometric controllers** on GPU (Lee rate/attitude/vel/acc/pos, fully-actuated for ROV).
- **Custom rendering** via [NVIDIA Warp](https://nvidia.github.io/warp/) — depth, segmentation, and custom sensors (e.g. LiDAR).
- **Domain randomization** — controller gains, IMU noise/bias, external disturbance force/torque.
- **Sim-to-real aids** — PX4 integration examples (Aerial Gym v2 official Sim2Real guide uses Holybro X500 v2 + Pixhawk Jetson Baseboard).

---

## Documentation

- **[USAGE.md](USAGE.md)** — 사용법·로컬 디렉터리 구조·실행 진입점·태스크/설정 위치 요약 (우선 참고).

Environment-side documentation is organized under **`docs/`** by purpose:

| Directory / File | Description |
|-----------|-------------|
| [docs/setup/](docs/setup/) | Environment spec (Ubuntu, conda, hardware), terminal commands for running training and replay. |
| [docs/analysis/](docs/analysis/) | Code analysis, test results, and official Aerial Gym docs summary. |
| [docs/reference/](docs/reference/) | Original Aerial Gym reference docs (Getting Started, RL Training, etc.). |
| [docs/upstream_diff.md](docs/upstream_diff.md) | Upstream(`ntnu-arl/aerial_gym_simulator` v2.0.0) 대비 추가·수정된 파일의 canonical list + 재검증 명령. |

Research-direction document (top-level): [`../UAV_RL_PretrainAdapt_Proposal.md`](../UAV_RL_PretrainAdapt_Proposal.md).

See **[docs/README.md](docs/README.md)** for a full index and quick links.

---

## Quick start (Ubuntu 22.04, conda)

```bash
conda activate aerialgym   # Python 3.8
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

Aerial Gym navigation baseline (upstream 경로, VAE latent + velocity command):

```bash
cd environment/aerial_gym/rl_training/rl_games
python runner.py --task=navigation_task --num_envs=256 --headless=True --use_warp=True
```

Environment API sanity check (학습 없이 obs/action 차원·env step 동작 확인):

```bash
python scripts/env_api_test.py
```

Sample Factory 기반 `pretrain_adapt_task` 파이프라인은 제안서 §4.9의 W1~W11 구현이 완료된 시점(M1)에 진입점이 확정된다.

See [docs/setup/terminal_commands.md](docs/setup/terminal_commands.md) for full command reference and [docs/setup/environment_spec.md](docs/setup/environment_spec.md) for environment details.

---

## License

BSD-3-Clause. See [../LICENSE](../LICENSE).
