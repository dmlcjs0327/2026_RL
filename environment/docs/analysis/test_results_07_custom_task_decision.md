# 7단계: Custom Task 필요 여부 결정

**결론**: **Custom task 필요.**

---

## 1. 판단 근거

| 조건 | position_setpoint_task 현황 | custom 필요 |
|------|-----------------------------|-------------|
| goal-conditioned observation이 기본으로 없음 | goal은 있으나 **고정 (0,0,0)** | ✓ 목표 샘플링 필요 |
| hindsight relabeling용 episode 정보 부족 | infos 비어 있음, success/crash 구분 없음 | ✓ |
| sparse reward가 기본이 아님 | Dense reward | ✓ |
| collision/success가 info로 깔끔히 안 나옴 | success 없음, collision은 terminations만 | ✓ |
| action interface 변경 필요 여부 | 현재로도 가능 | - |

---

## 2. 추천

- **기본 position_setpoint_task를 fork**하여 새 task 생성.
- **이름 예**: `position_setpoint_gc_task`, `navigation_goal_task` 등.
- **포함할 내용**:
  - reset/reset_idx에서 **목표 랜덤 샘플링** (env bounds 내).
  - **Sparse reward** + collision penalty.
  - **Success** 조건 (목표 근접 시) 및 `infos["success"]`, `infos["crashes"]`, `infos["timeouts"]`.
  - (선택) episode 경계/길이 등 HER/TDM용 메타 정보.

문서에서도 task 정의를 중심으로 RL framework를 붙이는 구조이므로, task를 새로 만드는 것이 구조상 자연스럽습니다.
