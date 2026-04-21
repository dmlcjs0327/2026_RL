# Aerial Gym upstream 대비 추가/수정 사항

이 문서는 `environment/` 디렉터리가 upstream [ntnu-arl/aerial_gym_simulator](https://github.com/ntnu-arl/aerial_gym_simulator) 대비 어떤 부분이 추가·수정되어 있는지 기록한다. 논문 실험 코드가 upstream fork 위에 쌓여 있기 때문에, 병합·리베이스·디버깅 시 어디까지가 "우리 코드"인지 빠르게 판별하기 위함이다.

## 기준점

- **Upstream repo**: https://github.com/ntnu-arl/aerial_gym_simulator
- **Upstream 버전**: v2.0.0 (로컬 `setup.py`와 동일 버전)
- **비교 시점 upstream HEAD**: `7f4986b7bb897b279c58ec82c3ba4123a680bbdd` (*"Updated to have same urdf for various asset types"*)
- **비교 일자**: 2026-04-21

## 재검증 방법

```bash
# 1. upstream clone (얕은 복제로 충분)
git clone --depth 1 https://github.com/ntnu-arl/aerial_gym_simulator.git /tmp/aerial_gym_upstream

# 2. 파일 단위 diff (pycache·runs·egg-info 제외)
diff -rq /tmp/aerial_gym_upstream/ /home/user/aerial_gym_simulator/environment/ \
  | grep -v __pycache__ | grep -v "\.egg-info" | grep -v "/runs"

# 3. 이 문서와 대조하여 "신규 항목" 또는 "사라진 항목"이 없는지 확인
```

`diff -rq` 출력 포맷:
- `/tmp/aerial_gym_upstream/…에만:` → upstream 에만 존재 (삭제되었거나 우리가 빼둔 파일)
- `/home/user/aerial_gym_simulator/environment/…에만:` → **우리가 추가한 파일/디렉터리**
- `파일 X와(과) Y이(가) 다릅니다` → **우리가 수정한 파일**

---

## 추가된 디렉터리 (upstream에 없음)

| 경로 | 역할 | 규모 |
|---|---|---|
| `aerial_gym/our_environment/` | 논문 실험용 커스텀 환경(`gc_empty_env`, `corridor_env`, `docking_env`) + 에셋 레지스트리. upstream의 `config/env_config`를 건드리지 않기 위한 분리 | 4개 env + `assets.py`, 총 **351 LOC** |
| `aerial_gym/task/position_setpoint_gc_task/` | goal-conditioned sparse-reward 태스크 (HER/TDM/CRL 학습용). 이 논문의 메인 태스크 | **645 LOC** (`position_setpoint_gc_task.py` 642) |
| `aerial_gym/rl_training/our_algorithms/` | 자체 모듈형 off-policy 러너 (TD3/SAC/DDPG + HER + SSL pretraining + 실험 드라이버). `runner.py`, `run_all_experiments.py`, `run_baseline_vs_proposed.py`, `run_corridor_baselines.py`, `run_her_ablation.py`, `run_matrix.py`, `run_proposed_only.py`, `run_sparse_test.py`, `configs/` 28개, `agents/analysis/common/networks/pretraining/replay/scripts/docs` 서브패키지 | 총 **~4360 LOC** |
| `aerial_gym/rl_training/offpolicy/` | 구형 SAC/TD3/DDPG/HER/TDM/CRL 훈련 스크립트 (`train.py` 1538 LOC) + `configs/` 20개. `our_algorithms/` 이전의 예전 경로 — 신규 작업은 `our_algorithms` 권장 | ~1540 LOC + 20 configs |
| `aerial_gym/nn/` | `depth_cnn.py` (depth 관측용 공용 CNN 블록) | 50 LOC |
| `scripts/` | `env_api_test.py` (환경 sanity check), `train_position_setpoint_all_algorithms.sh` | 139 LOC |
| `docs/analysis/`, `docs/reference/`, `docs/setup/`, `docs/overrides/`, `docs/images/`, `docs/gifs/`, `docs/README.md` | 논문/실험용 한국어 문서와 리소스. upstream `docs/`는 upstream 기능 문서(2~9번 md) 위주라 교집합이 거의 없음 | — |

## 추가된 단일 파일

| 경로 | 역할 |
|---|---|
| `aerial_gym/config/task_config/position_setpoint_gc_task_config.py` | GC 태스크 설정 (72 LOC) |
| `aerial_gym/config/sensor_config/camera_config/depth_camera_lowres_config.py` | 경량 depth 카메라 설정 (15 LOC) |
| `aerial_gym/config/sensor_config/lidar_config/lowres_lidar_config.py` | 경량 LiDAR 설정 (20 LOC) |
| `aerial_gym/rl_training/rl_games/ppo_aerial_quad_gc.yaml` | GC 태스크용 PPO 설정 (71 LOC) |
| `aerial_gym/rl_training/rl_games/ppo_aerial_quad_gc_smoke.yaml` | GC smoke test 설정 (70 LOC) |
| `aerial_gym/rl_training/rl_games/ppo_position_setpoint_dense.yaml` | dense-reward baseline PPO (70 LOC) |
| `aerial_gym/rl_training/rl_games/sac_aerial_quad.yaml` | SAC baseline 설정 (60 LOC) |
| `USAGE.md` | 프로젝트 사용법 메모 (upstream에 없음) |

## 수정된 파일 (upstream 대비)

`diff_lines`는 `diff` 출력의 `<`/`>` 라인 수 합. 대략적 변경 규모 지표.

| 경로 | upstream LOC | local LOC | diff_lines | 주 변경 목적 |
|---|---|---|---|---|
| `aerial_gym/rl_training/rl_games/runner.py` | 341 | 598 | **307** | GC 태스크·커스텀 env 연동, play/checkpoint 경로 확장 |
| `aerial_gym/task/position_setpoint_task/position_setpoint_task.py` | 282 | 385 | **141** | 논문용 보상·로깅 수정 |
| `aerial_gym/assets/warp_asset.py` | 136 | 179 | 45 | 커스텀 에셋 로딩 지원 |
| `aerial_gym/config/robot_config/base_quad_config.py` | 229 | 248 | 19 | 외란/게인 랜덤화 파라미터 조정 |
| `aerial_gym/task/__init__.py` | 113 | 125 | 12 | `position_setpoint_gc_task` 등록 |
| `aerial_gym/config/robot_config/x500_config.py` | 175 | 184 | 11 | 실기체 모델 파라미터 튜닝 |
| `aerial_gym/config/task_config/position_setpoint_task_config.py` | 30 | 33 | 11 | baseline env 전환 |
| `aerial_gym/env_manager/asset_manager.py` | 71 | 74 | 11 | our_environment 에셋 로딩 |
| `aerial_gym/env_manager/__init__.py` | 14 | 20 | 6 | `gc_empty_env`/`corridor_env`/`docking_env` 등록 |
| `aerial_gym/robots/__init__.py` | 62 | 65 | 5 | (경미) 로봇 등록 엔트리 조정 |
| `aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml` | 71 | 72 | 5 | 파라미터 조정 |
| `aerial_gym/env_manager/IGE_viewer_control.py` | 296 | 298 | 2 | (경미) |

## 상위 수준 파일 차이

- `setup.py`: description이 `"Isaac Gym environments for Aerial Robots (SHVD)"`로 변경 (upstream은 `SHVD` 접미사 없음)
- `README.md`: 프로젝트 고유 내용으로 교체
- `mkdocs.yml`: 문서 네비게이션 커스터마이즈
- `docs/` 전면 교체: upstream의 `2_getting_started.md`~`9_sim2real.md`, `index.md` 제거되고 한국어 문서 구조로 대체

## upstream에서만 존재 (우리 쪽에서 빠진 것)

- `LICENSE`, `.github/`, `.gitignore` — upstream 저장소 메타. 우리는 상위 저장소 루트의 것을 사용하므로 문제 없음
- `docs/2_getting_started.md` ~ `docs/9_sim2real.md`, `docs/index.md` — upstream 문서
- `aerial_gym/rl_training/rl_games/ppo_aerial_quad_navigation.yaml` — upstream에는 있으나 로컬에 없음 (navigation 태스크 설정). 필요 시 upstream에서 가져올 것.

## 주의

- 파일 시스템상 `__pycache__/`, `runs/`, `*.egg-info/`는 diff 대상에서 제외한다.
- 이 문서의 수치(LOC, diff_lines)는 2026-04-21 기준이며, 이후 변경에 따라 재측정 필요.
- upstream HEAD가 앞으로 이동하면 "수정된 파일"이 "새로 추가된 upstream 파일과 충돌"로 바뀔 수 있다. 리베이스 전 반드시 이 문서를 먼저 갱신할 것.
