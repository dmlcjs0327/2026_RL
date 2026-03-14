# 3단계: 환경 전환 테스트

**목표**: 같은 task에서 env_name만 바꿔 empty_env / env_with_obstacles / forest_env 동작 확인.

---

## 1. 환경 전환 방법

- **설정 파일**: `environment/aerial_gym/config/task_config/position_setpoint_task_config.py`
- **필드**: `env_name = "empty_env"` → `"env_with_obstacles"` 또는 `"forest_env"`로 변경.
- **등록**: `environment/aerial_gym/env_manager/__init__.py`에서 세 환경 모두 등록됨.

---

## 2. 실행 여부

- **사용자 환경**에서는 Isaac Gym이 설치되어 있어 학습/재생이 가능함. 아래는 **환경 전환 시 확인할 체크리스트**입니다.

---

## 3. 테스트 순서 및 확인 항목 (실행 시)

| 순서 | env_name | 최소 테스트 | 확인할 것 |
|------|----------|-------------|-----------|
| 1 | empty_env | num_envs=8, headless=False, play | reset 에러 없음, observation/action 차원 동일 |
| 2 | empty_env | num_envs=128, headless=True, 짧은 학습 | 안정 동작 |
| 3 | env_with_obstacles | num_envs=8, headless=False, play | reset 에러 없음, collision 발생 여부 |
| 4 | env_with_obstacles | num_envs=128, headless=True, 짧은 학습 | 동일 |
| 5 | forest_env | num_envs=8, headless=False, play | reset 에러 없음, 속도 저하 정도 |
| 6 | forest_env | num_envs=128, headless=True, 짧은 학습 | forest에서 iteration 속도 |

- **공통**: observation_space_dim=13, action_space_dim=4 유지되는지, 같은 task 코드가 그대로 동작하는지.

---

## 4. 결론

- **코드 구조상** env_name만 바꾸면 세 환경 전환이 가능함.
- **실제 동작**은 Isaac Gym 설치 후 위 순서대로 실행해 확인 필요.
