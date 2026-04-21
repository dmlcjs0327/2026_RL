# UAV RL: Pretrain-then-Adapt Research Repository

> **Research direction (2026-04~)**: NLP/CV에서 검증된 *Pretrain-then-Adapt* 패러다임을 소형(~280K param) UAV 항법 정책에 이식하고, **Full Fine-Tuning / Input-Branch-Frozen / LoRA** 세 적응 전략을 동일 backbone·동일 예산 하에서 체계적으로 비교한다.

**Repository**: [dmlcjs0327/2026_RL](https://github.com/dmlcjs0327/2026_RL)
**Steering document**: [`UAV_RL_PretrainAdapt_Proposal.md`](UAV_RL_PretrainAdapt_Proposal.md) — 연구 범위·방법론·실험 설계·리스크 관리의 단일 출처 (single source of truth)
**Target venue**: *Drones* (MDPI, IF 4.8)
**Timeline**: 4 months post thesis submission

---

## 연구 개요

본 저장소는 **UAV 항법을 위한 강화학습 정책 사전학습-적응 패러다임의 첫 체계적 비교 연구**를 위한 구현·실험 공간이다.

**Core research question** (제안서 §2.4):
> Jetson Orin Nano 급 소형 UAV 항법 정책에서, 총 환경 상호작용 예산을 명시적으로 통제한 공정 비교 하에, **Full-FT / Input-Branch-Frozen / LoRA** 세 적응 전략은 (i) 샘플 효율, (ii) 최종 성능, (iii) 수렴 안정성, (iv) 하이퍼파라미터 민감도, (v) 사전학습 능력 보존도 면에서 어떻게 다르며, 그 차이는 다운스트림 과업 특성에 따라 어떻게 변하는가?

**Simulator**: [Aerial Gym Simulator v2.0.0](https://github.com/ntnu-arl/aerial_gym_simulator) (Warp 백엔드)
**RL library**: Sample Factory 2.1.x (recurrent PPO)
**Base policy**: ~280K params, FP16 ~0.56 MB — Jetson Orin Nano 온보드 추론 대상
**Downstream tasks (4)**: Dense Forest (D1), Corridor (D2), Precision Stationkeeping (D3), Dynamic Obstacles (D4)

세부는 [`UAV_RL_PretrainAdapt_Proposal.md`](UAV_RL_PretrainAdapt_Proposal.md) 참조.

---

## 저장소 구조

```
aerial_gym_simulator/
├── UAV_RL_PretrainAdapt_Proposal.md   # ★ 연구 계획서 (steering document)
├── README.md                          # 이 문서
├── CLAUDE.md                          # Claude Code용 저장소 가이드
├── LICENSE
│
├── environment/                       # Aerial Gym 시뮬레이터 + 학습 코드 (primary workspace)
│   ├── aerial_gym/                    # 시뮬레이터 패키지
│   ├── docs/                          # 환경 문서 (setup / analysis / reference / upstream_diff)
│   ├── resources/                     # URDF·메쉬·환경 에셋
│   ├── scripts/                       # env sanity check, 학습 보조 스크립트
│   ├── runs/                          # 학습 로그·체크포인트 (gitignored)
│   ├── setup.py, pyproject.toml, requirements.txt, mkdocs.yml
│   ├── README.md                      # 환경 사이드 README
│   └── USAGE.md                       # 디렉터리·실행 진입점 레퍼런스
│
├── environment2/                      # Flightmare 0.0.5 (legacy, scratch duplicate)
└── paper/                             # (deprecated) 이전 연구 방향의 LaTeX 작업 공간
                                       #  — 현재는 UAV_RL_PretrainAdapt_Proposal.md 가 대체
```

`environment2/`는 legacy Flightmare 사본으로, 현재 연구에서는 사용하지 않는다 (자세한 비교는 conversations 내 simulator survey 참조).

`paper/`는 이전 goal-conditioned sparse-reward 연구 방향의 LaTeX 원고·참고 논문 저장소였으나, 현재 연구 방향 전환(2026-04)에 따라 deprecated 상태이다. 논문 작성 재개 시 구조는 재설계 예정.

---

## `environment/` 내부

```
environment/aerial_gym/
├── config/          # task / robot / sensor / controller / sim / asset / env 설정
├── task/            # RL 태스크 (navigation_task, position_setpoint_task 등)
├── env_manager/     # 환경·에셋 관리 (Isaac Gym + Warp 백엔드 전환)
├── robots/          # 기체 모델 (lmf2, base_quad, x500, tinyprop, morphy 등)
├── sensors/         # 카메라·LiDAR·IMU (Warp/IsaacGym 양 백엔드)
├── control/         # 기하학적 컨트롤러 (Lee rate/attitude/vel/acc/pos, fully-actuated)
├── sim/             # 시뮬레이터 빌더
├── sim2real/        # sim-to-real 자산
├── rl_training/
│   ├── rl_games/    # PPO baseline (upstream 기본)
│   ├── our_algorithms/   # (legacy) 이전 방향의 off-policy 모듈 (TD3/SAC/HER/TDM/CRL)
│   └── offpolicy/        # (legacy) 이전 방향의 off-policy configs
├── our_environment/ # 커스텀 envs (gc_empty_env, corridor_env, docking_env)
└── nn/ · utils/ · assets/ · registry/ · examples/
```

**현재 연구 방향은 PPO-only (Sample Factory)** — `our_algorithms/`·`offpolicy/`는 이전 sparse-reward off-policy 실험에서 사용한 코드로, 저장소에 보존하지만 **본 연구의 main path가 아니다**. 제안서 §2.5 scope 참조.

Sample Factory 기반 `pretrain_adapt_task` 구현은 제안서 §4.9의 W1~W11 work items로 명세되어 있으며, M1(2개월) 단계에서 통합된다.

---

## Upstream 대비 추가 사항

`environment/`는 [ntnu-arl/aerial_gym_simulator v2.0.0](https://github.com/ntnu-arl/aerial_gym_simulator) fork 위에 프로젝트 코드가 얹혀 있다. 어디까지가 upstream이고 어디부터가 우리 추가분인지는 **[`environment/docs/upstream_diff.md`](environment/docs/upstream_diff.md)** 에 기록되어 있다 (재검증 `diff -rq` 명령 포함).

---

## 빠른 시작

현재 통합 중인 Sample Factory 기반 파이프라인은 제안서 M1 종료 시점에 진입점이 확정된다. 그때까지는 upstream 기반 PPO baseline 으로 시뮬레이터 환경만 검증 가능:

```bash
conda activate aerialgym
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
cd environment/aerial_gym/rl_training/rl_games
python runner.py --task=navigation_task --num_envs=256 --headless=True --use_warp=True
```

환경 sanity check (학습 없이 obs·action 차원과 env step 동작만 확인):

```bash
python environment/scripts/env_api_test.py
```

학습 로그 모니터링:

```bash
tensorboard --logdir environment/runs/
```

세부 setup·실행 명령은 [`environment/docs/setup/`](environment/docs/setup/) 참조.

---

## 문서 위치

- **연구 계획**: [`UAV_RL_PretrainAdapt_Proposal.md`](UAV_RL_PretrainAdapt_Proposal.md) — 범위, 방법론, 실험, 리스크, 기여 전부
- **환경 사이드 사용법**: [`environment/README.md`](environment/README.md), [`environment/USAGE.md`](environment/USAGE.md)
- **Setup·터미널 명령**: [`environment/docs/setup/terminal_commands.md`](environment/docs/setup/terminal_commands.md), [`environment/docs/setup/environment_spec.md`](environment/docs/setup/environment_spec.md)
- **코드 분석·테스트 기록**: [`environment/docs/analysis/`](environment/docs/analysis/)
- **Aerial Gym 원본 문서**: [`environment/docs/reference/`](environment/docs/reference/)
- **Upstream fork 차이**: [`environment/docs/upstream_diff.md`](environment/docs/upstream_diff.md)

---

## 재현성

- 의존성 고정: Ubuntu 22.04 LTS, Python 3.8, PyTorch 2.0.1+cu118, Isaac Gym Preview 4, Warp 1.0.0, Sample Factory 2.1.x, CUDA 11.8 (제안서 §4.9.8)
- 연구 산출물(raw learning curves, 218+ runs, 분석 스크립트)은 논문 accept 시점에 공개
- Seed 명시 (Main 5 seeds: 1, 7, 42, 123, 2024), 음성 결과 모두 보고 (제안서 §8.4)
