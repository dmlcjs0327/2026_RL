# Aerial Gym 코드 분석 요약

목표: **goal-conditioned off-policy RL (TD3/SAC, HER, TDM, CRL, Ours)** 실험을 위해 시뮬레이터 아키텍처를 완전히 이해하고, transition `(s, a, s', r, done)` 접근 가능 여부를 확인한다.

---

## 1. Task 아키텍처

### 1.1 `position_setpoint_task` 구현 클래스

- **클래스**: `aerial_gym.task.position_setpoint_task.position_setpoint_task.PositionSetpointTask`
- **위치**: `environment/aerial_gym/task/position_setpoint_task/position_setpoint_task.py`
- **상속**: `BaseTask` (`environment/aerial_gym/task/base_task.py`)

### 1.2 태스크 등록 및 실행 경로

- **태스크 레지스트리**: `environment/aerial_gym/task/__init__.py`에서 `task_registry`에 `"position_setpoint_task"` → `PositionSetpointTask` + `position_setpoint_task_config` 등록.
- **RL 실행기**: `environment/aerial_gym/rl_training/rl_games/runner.py`
  - `env_configurations.register("position_setpoint_task", { "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task", **kwargs), ... })`
  - `--task=position_setpoint_task`로 태스크 선택.
- **실행 예**:
  ```bash
  python runner.py --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True
  python runner.py --task=position_setpoint_task --num_envs=8 --headless=False --play --checkpoint=<path>.pth
  ```
  (실행 시 작업 디렉터리는 `environment/aerial_gym/rl_training/rl_games/` 기준.)

### 1.3 태스크 설정 (task config)

- **파일**: `environment/aerial_gym/config/task_config/position_setpoint_task_config.py`
- **주요 필드**:
  - `sim_name`, `env_name`, `robot_name`, `controller_name`: 시뮬/환경/로봇/컨트롤러 선택
  - `num_envs`, `use_warp`, `headless`, `device`
  - `observation_space_dim = 13`, `action_space_dim = 4`
  - `episode_len_steps = 500`
  - `reward_parameters`: `pos_error_*`, `dist_reward_coefficient`, `crash_penalty` 등 (현재 보상 식은 코드 내 하드코딩으로 일부만 사용)

### 1.4 시뮬 빌드 흐름

- **SimBuilder**: `environment/aerial_gym/sim/sim_builder.py`  
  `build_env(sim_name, env_name, robot_name, controller_name, ...)` → `EnvManager` 인스턴스 생성.
- **EnvManager**: `environment/aerial_gym/env_manager/env_manager.py`
  - `env_config_registry.make_env(env_name)`, `sim_config_registry.make_sim(sim_name)` 등으로 설정 로드
  - `populate_env()` → `create_sim()` → IsaacGymEnv / WarpEnv, AssetLoader, RobotManagerIGE, ObstacleManager 생성
  - `prepare_sim()` 후 `global_tensor_dict`에 모든 상태/관측 텐서가 올라감

---

## 2. 환경 설정 (Environment Configuration)

### 2.1 환경 종류 및 등록

- **등록 위치**: `environment/aerial_gym/env_manager/__init__.py`
  - `empty_env` → `EmptyEnvCfg`
  - `env_with_obstacles` → `EnvWithObstaclesCfg`
  - `forest_env` → `ForestEnvCfg`
- **설정 파일**:
  - `environment/aerial_gym/config/env_config/empty_env.py`
  - `environment/aerial_gym/config/env_config/env_with_obstacles.py`
  - `environment/aerial_gym/config/env_config/forest_env.py`

### 2.2 환경별 특징

| 환경 | num_envs 기본값 | env_spacing | 장애물/에셋 | bounds (예시) |
|------|-----------------|-------------|-------------|----------------|
| **empty_env** | 3 | 1.0 | 없음 (`include_asset_type={}`) | lower/upper ±env_spacing |
| **env_with_obstacles** | 64 | 5.0 | panels, objects, walls (left/right/back/front/top/bottom) | 예: x∈[-2,10], y∈[-4,4], z∈[-3,3] |
| **forest_env** | 64 | 5.0 | trees, objects, bottom_wall | 예: x,y∈[-5,5], z∈[-1,3] |

### 2.3 에셋 정의

- **에셋 설정**: `environment/aerial_gym/config/asset_config/env_object_config.py`
  - `panel_asset_params`, `tree_asset_params`, `object_asset_params`, `tile_asset_params`, `thin_asset_params`
  - `left_wall`, `right_wall`, `back_wall`, `front_wall`, `bottom_wall`, `top_wall`
- 각 환경 설정의 `env_config.include_asset_type`으로 사용할 에셋 타입을 켜고, `asset_type_to_dict_map`으로 파라미터 클래스와 연결.

### 2.4 환경 전환 방법

- **태스크 설정에서 env 이름 지정**:  
  `position_setpoint_task_config.py`의 `env_name = "empty_env"`를 `"env_with_obstacles"` 또는 `"forest_env"`로 변경하면 해당 환경이 로드됨.
- **네비게이션 태스크**: `navigation_task_config.py`는 기본 `env_name = "env_with_obstacles"`.

---

## 3. 관측/액션 공간 (Task Interface)

### 3.1 관측 (Observation)

- **구성**: `PositionSetpointTask.process_obs_for_task()` (position_setpoint_task.py 194–203)
  - `task_obs["observations"]` shape: `(num_envs, 13)`
  - **인덱스별 의미**:
    - `[0:3]`: 목표 대비 위치 오차 (월드) — `target_position - robot_position`
    - `[3:7]`: 로봇 방향 쿼터니언 `robot_orientation`
    - `[7:10]`: 로봇 body 선속도 `robot_body_linvel`
    - `[10:13]`: 로봇 body 각속도 `robot_body_angvel`
- **데이터 소스**: `obs_dict = self.sim_env.get_obs()` → `EnvManager.get_obs()` → `global_tensor_dict` 참조.  
  로봇 상태는 `IGE_env_manager.py`에서 `robot_state_tensor` 슬라이스로 채워짐 (`robot_position`, `robot_orientation`, `robot_linvel`, `robot_angvel` 등). body 프레임 속도는 로봇 매니저/컨트롤러에서 계산되어 `robot_body_linvel`, `robot_body_angvel`로 제공.

### 3.2 목표(Setpoint) 생성

- **position_setpoint_task**:
  - `target_position`: `(num_envs, 3)` 텐서.
  - **현재 구현**: `reset()` / `reset_idx()`에서 `self.target_position[:, 0:3] = 0.0`으로 고정 (목표가 항상 원점).
  - 따라서 **goal-conditioned 실험을 위해서는** 이 부분을 수정해 랜덤 목표 또는 비율 기반 샘플링(아래 navigation_task 참고)으로 바꿔야 함.
- **navigation_task** (참고):
  - `reset_idx()`에서 `env_bounds_min/max`와 `target_min_ratio`, `target_max_ratio`로 `target_position[env_ids]`를 랜덤 샘플링 (`torch_interpolate_ratio`).  
  → 목표 생성 패턴을 position_setpoint_task에 이식 가능.

### 3.3 액션 공간

- **차원**: `action_space_dim = 4` (config), `Box(low=-1.0, high=1.0, shape=(4,))`.
- **컨트롤러**: 기본 `lee_attitude_control` (config: `environment/aerial_gym/config/controller_config/lee_controller_config.py`).
- **의미** (Lee attitude control):  
  **`[thrust, roll, pitch, yaw_rate]`** in vehicle frame, 스케일 -1~1.
  - 내부적으로 position/velocity/attitude 제어기로 변환 후, wrench → control allocator → 각 모터 thrust로 적용.
- **정리**: 액션은 **thrust + 자세/요요율 명령** 형태이며, body rate만 쓰는 형태로 바꾸려면 다른 컨트롤러 설정 또는 래퍼가 필요함.

---

## 4. 보상 정의 (Reward)

### 4.1 구현 위치

- **파일**: `environment/aerial_gym/task/position_setpoint_task/position_setpoint_task.py`
- **함수**: `compute_reward()` (JIT 스크립트, 245–281행), 호출은 `compute_rewards_and_crashes(obs_dict)` (205–230행).

### 4.2 현재 보상 구성 (dense)

- `pos_error_vehicle_frame`: 목표 대비 위치 오차 (vehicle frame).
- **항목**:
  - 거리 기반: `dist = norm(pos_error)`, `pos_reward`, `dist_reward = (20 - dist)/40`
  - 자세: up 벡터 틸트 → `up_reward`
  - 각속도: `ang_vel_reward`
  - **충돌**: `crashes > 0`이면 `total_reward = -20` (고정 패널티)
  - 추가로 `dist > 8`이면 해당 env를 crash로 마킹 (목표 이탈로 인한 실패로 처리).
- **특징**: 거리/자세/각속도에 대한 **dense reward**이며, config의 `reward_parameters`는 부분적으로만 반영됨.

### 4.3 논문용 수정 방향 (sparse + collision)

- **sparse**: 목표 도달 시에만 양의 보상 (예: 거리 < threshold일 때만 +1).
- **collision penalty**: 현재처럼 `crashes`일 때 음의 보상 유지 (크기 조절 가능).
- **수정 위치**: `compute_reward()` 및 호출부, 필요 시 `reward_parameters`를 더 활용하도록 확장.

---

## 5. 종료 조건 (Episode Termination)

### 5.1 두 가지 종료 유형

- **termination (done)**: 주로 **충돌**.
  - `self.terminations`는 `obs_dict["crashes"]`와 동일 참조.
  - `compute_rewards_and_crashes()` 내부에서 `crashes` 텐서를 in-place로 갱신 (거리 > 8이면 crash로 설정 등).
- **truncation**: **시간 초과**.
  - `self.truncations[:] = (sim_steps > episode_len_steps)` (position_setpoint_task 172–174행).
  - `episode_len_steps = 500` (config).

### 5.2 충돌 판정

- **물리 충돌**: `EnvManager.compute_observations()` (env_manager.py 346–349행)
  - `collision_tensor += (norm(robot_contact_force_tensor, dim=1) > collision_force_threshold)`
  - `robot_contact_force_tensor`: IGE에서 `global_contact_force_tensor`의 로봇 첫 rigid body 슬라이스 (IGE_env_manager 342–344행).
- **환경별 threshold**:  
  `empty_env`: 0.01 N, `env_with_obstacles`: 0.05 N, `forest_env`: 0.005 N.
- **리셋**: `reset_on_collision == True`이면 `post_reward_calculation_step()` → `reset_terminated_and_truncated_envs()`에서 collision 또는 truncation된 env만 `reset_idx(env_ids)` 호출.

### 5.3 목표 도달 성공

- **position_setpoint_task**: 현재 **목표 도달 시 종료** 로직 없음.  
  성공 판별을 하려면 `process_obs_for_task()` 또는 `step()` 쪽에서 `norm(target_position - robot_position) < threshold`로 success 플래그를 만들고, `infos`에 넣어주는 수정이 필요함.
- **navigation_task**: truncation 시점에 `norm(target - robot_position) < 1.0`이면 success로 기록하고 `infos["successes"]`, `infos["timeouts"]`, `infos["crashes"]` 제공.  
  → 같은 패턴을 position_setpoint_task에 적용 가능.

---

## 6. 병렬 시뮬레이션 설계

### 6.1 num_envs 처리

- **진입점**: runner의 `--num_envs` 또는 yaml `num_actors` / `env_config.num_envs`가 `task_config.num_envs`를 덮어씀 (`runner.py` `update_config()`).
- **SimBuilder.build_env(num_envs=...)** → EnvManager → `cfg.env.num_envs`로 전달되어, Isaac Gym에서 `num_envs`개 환경이 한 번에 생성됨.
- 모든 상태/관측/보상/termination/truncation은 `(num_envs,)` 또는 `(num_envs, dim)` 형태의 GPU 텐서.

### 6.2 리셋

- **전체**: `reset()` → `reset_idx(torch.arange(num_envs))`.
- **일부만**: `post_reward_calculation_step()`에서 termination/truncation된 env id만 수집해 `reset_idx(envs_to_reset)` 호출.
- **reset_idx (EnvManager)**: IGE_env.reset_idx, asset_manager.reset_idx, (use_warp이면) warp_env.reset_idx, robot_manager.reset_idx, write_to_sim, **sim_steps[env_ids] = 0**.

### 6.3 에피소드 ID / 길이 추적

- **sim_steps**: `(num_envs,)` int32 텐서로, 각 env별 현재 스텝 수.  
  리셋된 env만 0으로 초기화되므로, **에피소드 길이와 done/truncation 추적에 사용 가능**.
- **에피소드 경계**:  
  `terminations` 또는 `truncations`가 True인 시점이 에피소드 끝; 다음 스텝에서 해당 env는 이미 리셋된 상태이므로, off-policy 버퍼에 저장할 때 `done = terminations | truncations`로 처리하면 됨.

### 6.4 HER / TDM / step-wise hindsight

- **transition 수집**:  
  매 스텝 `(obs, action, next_obs, reward, done)`을 버퍼에 넣을 수 있음.  
  `obs` = `task_obs["observations"]`, `reward` = `rewards`, `done` = `terminations | truncations`.  
  `next_obs`는 다음 스텝에서 `process_obs_for_task()` 후의 `task_obs["observations"]` (또는 리셋된 env는 reset 후 첫 obs).
- **goal 조건화**:  
  현재 obs에 goal이 이미 포함되어 있음 (0:3이 목표 대비 위치).  
  HER를 쓰려면 목표를 랜덤 샘플링하도록 `target_position`을 바꾸고, replay 시 다른 goal으로 대체할 수 있도록 obs/버퍼 구조를 설계하면 됨.
- **env별 독립성**:  
  각 env가 독립적으로 리셋되므로, 에피소드 단위 샘플링과 trajectory 단위 샘플링 모두 가능.

---

## 7. 로깅 및 체크포인트

### 7.1 rl_games (runner.py)

- **디렉터리**: `runner.py`에서 `os.makedirs("nn", exist_ok=True)`, `os.makedirs("runs", exist_ok=True)` 생성.  
  실행 디렉터리가 `environment/aerial_gym/rl_training/rl_games/`이면 `nn/`, `runs/`는 그 아래에 생성됨.
- **체크포인트**: rl_games 라이브러리 기본 동작에 따름 (보통 `nn/` 또는 config에 지정된 경로에 체크포인트 저장).
- **학습 곡선**: TensorBoard 등은 rl_games 설정에 따라 `runs/` 또는 비슷한 경로 사용 가능.

### 7.2 cleanrl 예제

- `ppo_continuous_action.py`: `SummaryWriter(f"runs/{run_name}")`, 모델 저장 `runs/{run_name}/latest_model.pth`.
- **reward / episode 통계**: SummaryWriter에 기록하는 부분을 참고해, success rate, collision rate, episode length 등을 동일하게 추가 가능.

### 7.3 접근 요약

- **보상 곡선**: rl_games/cleanrl의 로거 설정에서 확인.
- **에피소드 통계**: navigation_task처럼 `infos["successes"]`, `infos["crashes"]`, `infos["timeouts"]`를 추가한 뒤 로거로 기록하면 됨.
- **체크포인트**: `--checkpoint=<path>.pth`로 재생/평가에 사용.

---

## 8. Off-policy RL 호환성 (Transition 생성)

### 8.1 표준 transition

- **한 스텝 흐름**:  
  `step(actions)` → `sim_env.step(actions)` (physics + `compute_observations()` → collision_tensor 갱신) → `compute_rewards_and_crashes(obs_dict)` → `rewards`, `terminations` 갱신 → truncation 갱신 → `post_reward_calculation_step()` (리셋) → `get_return_tuple()` (process_obs_for_task 포함).
- **반환**: `(task_obs, rewards, terminations, truncations, infos)`.
- **추출**:  
  - `state` = 현재 `task_obs["observations"]` (또는 reset 직후면 reset 후 obs).  
  - `action` = 이번 스텝에 넣은 `actions`.  
  - `next_state` = 다음 스텝에서 받은 `task_obs["observations"]` (리셋된 env는 새 에피소드 첫 obs).  
  - `reward` = `rewards`, `done` = `terminations | truncations`.
- **결론**: **(s, a, s', r, done)** 을 매 스텝 쉽게 얻을 수 있어, **replay buffer, HER, TDM, CRL, Ours** 등 off-policy/goal-conditioned 알고리즘에 그대로 사용 가능.

### 8.2 주의사항

- **return_state_before_reset**:  
  True이면 리셋 전 상태를 반환하고, False이면 리셋 후 상태를 반환.  
  Off-policy에서는 보통 **False**로 두고, 리셋된 env의 transition에서 `done=True`로 저장하고 다음 스텝의 obs를 `s'`로 쓰면 됨.
- **목표 변경**:  
  goal-conditioned 실험을 위해 `reset_idx()`/`reset()`에서 `target_position`을 환경 bounds 안에서 랜덤 샘플링하도록 수정 필요.

---

## 9. 환경 스트레스 테스트 (권장)

- **명령 예**:  
  `--num_envs=128`, `256`, `512`, `1024`, `2048` 등으로 변경해 보며 실행.
- **확인 항목**:  
  GPU 메모리 사용량, 시뮬레이션 속도(step/s), 안정성(오류/크래시 여부).  
  `EnvManager.log_memory_use()`로 GPU 메모리 로그 가능.
- **실행 위치**:  
  `environment/aerial_gym/rl_training/rl_games/`에서 `python runner.py --task=position_setpoint_task --num_envs=<N> --headless=True --use_warp=True`.

---

## 10. 요약 체크리스트

| 항목 | 상태 | 비고 |
|------|------|------|
| position_setpoint_task 구현 클래스 | ✓ | PositionSetpointTask, position_setpoint_task.py |
| 태스크/환경 전환 | ✓ | task_config.env_name, env_config_registry |
| 관측 13차원 | ✓ | goal error(3) + quat(4) + body_linvel(3) + body_angvel(3) |
| 액션 4차원 | ✓ | [thrust, roll, pitch, yaw_rate] (Lee attitude) |
| 보상 구현 위치 | ✓ | compute_reward in position_setpoint_task.py, dense → sparse 전환 가능 |
| 목표 생성 | ⚠ | 현재 고정 (0,0,0); 랜덤 목표로 확장 필요 |
| 종료: 충돌 | ✓ | collision_force_tensor > threshold → terminations |
| 종료: 시간 | ✓ | sim_steps > episode_len_steps → truncations |
| 목표 도달 성공 | ⚠ | position_setpoint_task에는 없음; navigation_task 패턴 이식 가능 |
| 병렬 env / 리셋 | ✓ | reset_idx(env_ids), sim_steps 추적 |
| (s,a,s',r,done) 접근 | ✓ | step 반환값과 obs_dict로 바로 구성 가능 |
| 로깅/체크포인트 | ✓ | nn/, runs/, --checkpoint |

**결론**: Aerial Gym은 **off-policy goal-conditioned RL 실험**에 적합하다.  
목표 샘플링과 보상/성공 판정만 논문 설정에 맞게 수정하면, TD3/SAC, HER, TDM, CRL, Ours 구현을 시뮬레이터 위에 올리기 좋은 구조다.
