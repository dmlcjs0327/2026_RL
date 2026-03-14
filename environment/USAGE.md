# Aerial Gym (Isaac Gym 기반) — 사용법 및 디렉터리 참조

이 문서는 **현재 로컬 `environment` 디렉터리 구조**를 기준으로 작성되었습니다.  
다른 환경에서 개발 환경을 복원하거나, 정보를 빠르게 찾을 때 우선 이 파일을 참고하세요.

- **공식 문서**: [Aerial Gym Simulator](https://ntnu-arl.github.io/aerial_gym_simulator/)
- **기반**: [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)

---

## 1. 디렉터리 구조 (로컬 기준)

```
environment/
├── USAGE.md                 # 이 문서 — 사용법·디렉터리 참조
├── README.md                # 프로젝트 개요·Quick start
├── requirements.txt        # Python 의존성
├── setup.py                 # aerial_gym 패키지 설치
├── pyproject.toml           # black 등 도구 설정
│
├── aerial_gym/              # 메인 패키지
│   ├── config/              # 시뮬레이션·태스크·로봇·센서 설정
│   │   ├── task_config/     # ★ 태스크별 설정 (env_name, robot_name, reward 등)
│   │   ├── robot_config/    # 로봇(URDF 매핑, 컨트롤러 등)
│   │   ├── env_config/      # 환경 에셋·경계
│   │   ├── sensor_config/   # 카메라·LiDAR·IMU
│   │   ├── controller_config/
│   │   ├── sim_config/
│   │   └── asset_config/
│   │
│   ├── task/                # RL 태스크 구현
│   │   ├── position_setpoint_task/        # 위치 setpoint (dense reward)
│   │   ├── position_setpoint_gc_task/     # Goal-conditioned (sparse, 랜덤 목표)
│   │   ├── navigation_task/
│   │   ├── custom_task/                   # 커스텀 태스크 템플릿
│   │   └── ... (sim2real, morphy 등)
│   │
│   ├── rl_training/         # RL 학습 진입점
│   │   ├── rl_games/        # PPO 등 — runner.py 실행 위치
│   │   │   ├── runner.py    # ★ 학습·재생 메인 진입점
│   │   │   ├── *.yaml       # rl-games 알고리즘 설정 (task 이름은 config 내 env_name)
│   │   │   └── runs/        # 학습 로그·체크포인트 (nn/*.pth)
│   │   └── offpolicy/       # SAC, TD3, DDPG, HER, TDM, CRL 등
│   │       ├── train.py     # 오프폴리시 학습 스크립트 (별도 진입점)
│   │       └── configs/     # *.yaml — 알고리즘별 설정
│   │
│   ├── nn/                  # 신경망 (예: depth CNN)
│   │   ├── __init__.py
│   │   └── depth_cnn.py
│   ├── env_manager/         # SimBuilder, 에셋 로드
│   ├── robots/             # 로봇 클래스
│   ├── control/            # 컨트롤러 (lee_attitude_control 등)
│   ├── registry/           # task_registry, sim_registry 등
│   ├── sim/                # Isaac Gym 시뮬레이션 래퍼
│   ├── sensors/            # Warp 기반 센서
│   ├── assets/
│   └── utils/
│
├── resources/              # URDF·메시
│   ├── robots/             # quad, x500, octarotor, BlueROV 등
│   └── models/environment_assets/  # trees, walls, objects
│
├── scripts/                # 유틸 스크립트
│   ├── env_api_test.py     # reset/step API 테스트
│   └── train_position_setpoint_all_algorithms.sh  # 여러 알고리즘 일괄 학습
│
└── docs/                   # 문서
    ├── README.md            # 문서 인덱스
    ├── setup/               # 환경 스펙·터미널 명령어
    │   ├── environment_spec.md
    │   └── terminal_commands.md
    ├── analysis/            # 코드 분석·테스트 결과
    └── reference/           # 공식 문서 참조 (Getting Started, RL Training 등)
```

---

## 2. 실행 진입점 요약

| 목적 | 실행 위치 | 명령/파일 |
|------|-----------|-----------|
| **학습·재생 (rl-games 통합)** | `environment/aerial_gym/rl_training/rl_games` | `python runner.py --file=<yaml> --task=<task_name> ...` |
| **오프폴리시 학습 (SAC/TD3 등)** | `environment/aerial_gym/rl_training/offpolicy` | `python train.py --config configs/<config>.yaml` (또는 runner.py로 오프폴리시 yaml 지정) |
| **환경 API 테스트** | `environment` 루트 | `python scripts/env_api_test.py` |
| **일괄 학습 스크립트** | 셸 | `scripts/train_position_setpoint_all_algorithms.sh` |

---

## 3. 태스크 이름과 설정 파일

태스크는 `aerial_gym/task/__init__.py`에서 `task_registry`에 등록됩니다.

| 태스크 이름 | 설정 파일 (config/task_config/) | 용도 |
|-------------|----------------------------------|------|
| `position_setpoint_task` | `position_setpoint_task_config.py` | 위치 setpoint, dense reward |
| `position_setpoint_gc_task` | `position_setpoint_gc_task_config.py` | Goal-conditioned, sparse, 랜덤 목표 |
| `navigation_task` | `navigation_task_config.py` | 네비게이션 (심상 등) |
| 기타 | `position_setpoint_task_sim2real_config.py` 등 | Sim2Real·PX4 등 |

**환경/로봇 변경**: 해당 태스크의 `task_config` 클래스에서  
`env_name` (`empty_env` / `env_with_obstacles` / `forest_env` 등),  
`robot_name`, `controller_name` 등을 수정합니다.

---

## 4. rl_games runner.py 사용법

**실행 디렉터리**: 반드시 `environment/aerial_gym/rl_training/rl_games` 에서 실행.

```bash
cd environment/aerial_gym/rl_training/rl_games
conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
# 필요 시: export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### 학습

```bash
# position_setpoint_task (dense) — PPO
python runner.py --file=ppo_position_setpoint_dense.yaml --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True

# position_setpoint_gc_task (goal-conditioned)
python runner.py --file=ppo_aerial_quad_gc.yaml --task=position_setpoint_gc_task --num_envs=256 --headless=True --use_warp=True

# 오프폴리시 (SAC/TD3 등): --file에 offpolicy config yaml 지정
python runner.py --file=../offpolicy/configs/sac_position_setpoint_dense.yaml --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True
```

### 재생 (체크포인트)

```bash
python runner.py --file=ppo_position_setpoint_dense.yaml --task=position_setpoint_task --num_envs=1 --headless=False --play --checkpoint=./runs/<실험폴더>/nn/<체크포인트>.pth
```

- 체크포인트는 `rl_games/runs/` 아래 실험 폴더의 `nn/*.pth` 에 저장됩니다.

---

## 5. rl_games YAML 설정 파일 (로컬)

| 파일 | 태스크 (env_name) | 비고 |
|------|-------------------|------|
| `ppo_position_setpoint_dense.yaml` | position_setpoint_task | PPO dense |
| `ppo_aerial_quad.yaml` | (설정 내 참조) | PPO quad |
| `ppo_aerial_quad_gc.yaml` | position_setpoint_gc_task | PPO goal-conditioned |
| `ppo_aerial_quad_gc_smoke.yaml` | smoke 테스트용 | |
| `sac_aerial_quad.yaml` | SAC | |

---

## 6. 오프폴리시 설정 (offpolicy/configs/)

**position_setpoint_task (dense)**  
`ddpg_position_setpoint_dense.yaml`, `td3_position_setpoint_dense.yaml`, `sac_position_setpoint_dense.yaml`,  
`ddpg_ours_*`, `td3_ours_*`, `sac_ours_*`, `*_her_*`, `tdm_position_setpoint_dense.yaml`, `crl_position_setpoint_dense.yaml` 등.

**position_setpoint_gc_task (goal-conditioned)**  
`ddpg_gc.yaml`, `td3_gc.yaml`, `sac_gc.yaml`, `*_ours_gc.yaml`, `*_her_gc.yaml`, `tdm_gc.yaml`, `crl_gc.yaml` 등.

- runner.py로 실행할 때: `--file=../offpolicy/configs/<이름>.yaml` 과 `--task=position_setpoint_task` 또는 `--task=position_setpoint_gc_task` 조합.

---

## 7. 환경(env_name) 전환

태스크 설정 파일(예: `aerial_gym/config/task_config/position_setpoint_task_config.py`)에서:

```python
env_name = "empty_env"           # 기본 빈 환경
# env_name = "env_with_obstacles"
# env_name = "forest_env"
```

수정 후 runner.py를 같은 `--task` 로 다시 실행하면 해당 환경이 적용됩니다.

---

## 8. 문서·환경 스펙 빠른 링크

- **환경 스펙 (OS, Conda, GPU)**: [docs/setup/environment_spec.md](docs/setup/environment_spec.md)
- **터미널 명령어 정리**: [docs/setup/terminal_commands.md](docs/setup/terminal_commands.md)
- **문서 인덱스**: [docs/README.md](docs/README.md)
- **공식 사이트**: [ntnu-arl.github.io/aerial_gym_simulator](https://ntnu-arl.github.io/aerial_gym_simulator/)
  - Installation: [Getting Started](https://ntnu-arl.github.io/aerial_gym_simulator/2_getting_started/)
  - RL Training: [RL Training](https://ntnu-arl.github.io/aerial_gym_simulator/6_rl_training/)
  - Customization: [Customization](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/)

---

## 9. 설치 요약 (공식 문서 기준)

1. Conda 환경: `conda create -n aerialgym python=3.8` 후 PyTorch(CUDA), Isaac Gym 설치.
2. Isaac Gym의 `gymutil.py` 337행: `parse_args()` → `parse_known_args()` 로 수정 권장.
3. `environment` 루트에서: `pip install -e .` (또는 requirements.txt 반영).
4. `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` 설정 (및 필요 시 `VK_ICD_FILENAMES`).

자세한 명령은 [docs/setup/terminal_commands.md](docs/setup/terminal_commands.md) 참고.

---

*이 문서는 로컬 `environment` 디렉터리와 [Aerial Gym 공식 문서](https://ntnu-arl.github.io/aerial_gym_simulator/)를 기준으로 작성되었습니다. 디렉터리 구조나 파일 이름이 바뀌면 이 문서를 우선 갱신하는 것을 권장합니다.*
