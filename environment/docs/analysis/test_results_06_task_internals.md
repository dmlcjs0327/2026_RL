# 6단계: Task 내부 논문용 점검

**목표**: observation / action / reward / termination이 논문(TD3, SAC, HER, sparse reward)에 맞는지 확인.

---

## 6-1. Observation

- **현재 compact state**: 13차원 (position_setpoint_task).
  - `[0:3]`: 목표 대비 위치 (target - robot_position) — **목표 상대 벡터 있음**.
  - `[3:7]`: 로봇 쿼터니언 (방향).
  - `[7:10]`: body 선속도.
  - `[10:13]`: body 각속도.
- **위치/속도/목표 상대 벡터**: 포함됨.
- **Collision / obstacle summary**: **관측에는 없음**. collision은 termination으로만 전달됨. obstacle 정보는 13차원에 포함되지 않음 (필요 시 task 확장).

---

## 6-2. Action

- **현재**: Lee attitude control — **`[thrust, roll, pitch, yaw_rate]`** (vehicle frame, -1~1).
- **수준**: navigation-level continuous control에 가까움 (직접 motor command가 아님).  
  TD3/SAC baseline 구현에 적합한 수준으로 판단됨.

---

## 6-3. Reward

- **현재**: **Dense** — 거리/자세/각속도 기반 연속 보상 + crash 시 -20.
- **논문**: Sparse reward + collision penalty가 핵심이므로,
  - **선택 1**: 기본 reward를 그대로 두고 추가 실험만 sparse로.
  - **선택 2**: **sparse + collision penalty**로 바꾸는 것을 권장 (목표 도달 시 +1, 그 외 0, collision 시 음수 등).  
  수정 위치: `position_setpoint_task.py`의 `compute_reward()` 및 호출부.

---

## 6-4. Termination

- **Success 종료**: **없음**. position_setpoint_task에는 목표 도달 시 에피소드 종료/성공 플래그가 없음.
- **Collision 종료**: 있음 — `crashes`(contact force > threshold) → terminations → 리셋.
- **Timeout 종료**: 있음 — `sim_steps > episode_len_steps` → truncations → 리셋.
- **논문 지표**: success / collision / timeout을 **분리**하려면 task에서 **success 조건**을 정의하고 `infos["success"]`, `infos["timeouts"]`, `infos["crashes"]` 등으로 넘겨야 함 (navigation_task 참고).

---

## 결론 (6단계)

- **관측**: 목표 상대 벡터 포함, collision/obstacle은 없음 — 필요 시 확장.
- **액션**: navigation-level에 적합.
- **보상**: 현재 dense → 논문용으로 sparse + collision penalty 전환 권장.
- **종료**: success 정의 추가 및 infos 분리 필요.
- **기본 position_setpoint_task를 그대로 쓸지 vs custom task 포크**:  
  **Custom task 포크 권장** — goal 랜덤 샘플링, sparse reward, success/collision/timeout infos를 깔끔히 넣기 위해.
