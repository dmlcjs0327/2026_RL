# Aerial Gym 단계별 테스트 결과 목차

논문용 baseline 및 custom task 준비를 위한 단계별 테스트 결과 문서입니다.  
**참고**: 사용자 환경(Isaac Gym 설치됨)에서는 학습·재생 명령이 정상 실행됨.  
예: `python runner.py --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True`,  
재생: `--play --checkpoint=./runs/gen_ppo_10-05-40-05/nn/last_gen_ppo_ep_400_rew__7625.1924_.pth`

| 단계 | 문서 | 내용 |
|------|------|------|
| 1 | [test_results_01_code_structure.md](test_results_01_code_structure.md) | 코드 구조: task 클래스, env/robot/controller 레지스트리, 관측/행동/에피소드 길이, reward/collision/timeout 위치, 핵심 질문 답변 |
| 2 | [test_results_02_log_structure.md](test_results_02_log_structure.md) | 로그 구조: `runs/<실험>/nn/*.pth` 체크포인트, 재생 명령 예시 반영 |
| 3 | [test_results_03_env_switch.md](test_results_03_env_switch.md) | 환경 전환 테스트 empty/env_with_obstacles/forest (체크리스트) |
| 4 | [test_results_04_parallel_scaling.md](test_results_04_parallel_scaling.md) | 병렬성 한계 측정 128~2048 (기록 항목 표) |
| 5 | [test_results_05_replay_protocol.md](test_results_05_replay_protocol.md) | Replay/시각 확인 프로토콜 (num_envs=8, --play, checkpoint 경로) |
| 6 | [test_results_06_task_internals.md](test_results_06_task_internals.md) | Task 내부: observation / action / reward / termination 논문용 점검 |
| 7 | [test_results_07_custom_task_decision.md](test_results_07_custom_task_decision.md) | Custom task 필요 여부 결정 → **필요** |
| 8 | [test_results_08_env_api_test.md](test_results_08_env_api_test.md) | Off-policy 환경 API 테스트 스크립트 설명 및 예상 결과 |

**스크립트**

- `environment/scripts/env_api_test.py` — reset/step shape, terminated/truncated, info, episode 경계 확인 (Isaac Gym 설치 후 실행).

**환경·스펙**

- [environment_spec.md](../setup/environment_spec.md) — 개발 머신 PC 스펙, Ubuntu 22.04 듀얼 부팅 등 환경 정리.

**종합 분석**

- [aerial_gym_code_analysis.md](aerial_gym_code_analysis.md) — 전체 코드베이스 분석 요약.
- [aerial_gym_official_docs_summary.md](aerial_gym_official_docs_summary.md) — [공식 문서](https://ntnu-arl.github.io/aerial_gym_simulator/) 면밀 검토 요약 (설치, 시뮬 구성, 태스크/환경 차이, RL 훈련, 센서, Sim2Real, FAQ).
