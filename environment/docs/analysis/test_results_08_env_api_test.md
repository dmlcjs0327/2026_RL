# 8단계: 환경 API 테스트 스크립트

**목표**: Aerial Gym이 off-policy learner에 필요한 정보를 주는지 확인 (reset/step shape, terminated/truncated, info, episode 경계).

---

## 1. 스크립트 위치 및 실행 방법

- **파일**: `environment/scripts/env_api_test.py`
- **실행** (Isaac Gym 설치 후):
  ```bash
  cd /home/user/aerial_gym_simulator/environment
  python scripts/env_api_test.py
  ```
- **의존성**: isaacgym, torch, aerial_gym (프로젝트 루트를 PYTHONPATH에 두거나 스크립트가 자동으로 추가).

---

## 2. 확인 항목 (스크립트가 검사하는 것)

| 항목 | 내용 |
|------|------|
| reset() 결과 shape | 반환값이 (obs_dict, rewards, terminations, truncations, infos) 5-tuple, obs_dict["observations"] shape (num_envs, 13) |
| step(action) 결과 shape | 동일 5-tuple, rewards/terminations/truncations 각 (num_envs,) |
| terminated/truncated 분리 | terminations와 truncations가 별도 텐서로 제공됨 |
| info에 success/collision | position_setpoint_task는 infos={} → 현재 없음 (custom task에서 추가 필요) |
| dict observation vs flat | dict, 키 "observations"에 (num_envs, 13) 텐서 |
| batch env episode 경계 | terminations \| truncations로 done 판별 가능, 리셋된 env는 다음 step에서 새 에피소드 |

---

## 3. 실행 결과 (예상)

- **Isaac Gym 미설치 시**: `SKIP: isaacgym not installed.` 출력 후 종료.
- **설치 후**: reset/step shape, term/trunc 분리, infos 키 등이 출력됨.  
  HER/TDM을 위해 같은 episode 내 trajectory를 알려면 `done`이 True인 인덱스를 기록해 episode 경계로 사용하면 됨.

---

## 4. 결론

- **Off-policy transition generator**로 사용 가능: (s, a, s', r, done) 구성 가능.
- **Episode 경계**: terminated/truncated로 명확함.  
  **보강 필요**: success/collision을 infos에 넣으려면 custom task 또는 wrapper 수정 필요.
