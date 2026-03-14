# 1단계: 코드 구조 파악 결과

**확인 일시**: 테스트 실행 시점  
**대상**: runner.py, task_config, env_config, robot_config, controller_config, task 정의, runs/ 폴더

---

## 1. 확인 대상 파일/폴더 위치

| 대상 | 실제 경로 |
|------|-----------|
| runner.py | `environment/aerial_gym/rl_training/rl_games/runner.py` |
| position_setpoint_task_config | `environment/aerial_gym/config/task_config/position_setpoint_task_config.py` |
| env_config | `environment/aerial_gym/config/env_config/` (empty_env.py, env_with_obstacles.py, forest_env.py 등) |
| base_quad_config | `environment/aerial_gym/config/robot_config/base_quad_config.py` |
| controller_config | `environment/aerial_gym/config/controller_config/` (lee_controller_config.py 등) |
| task 정의 | `environment/aerial_gym/task/position_setpoint_task/position_setpoint_task.py` |
| runs/ | `environment/aerial_gym/rl_training/rl_games/runs/` (runner가 생성), 프로젝트 루트에도 생성 가능 |

---

## 2. position_setpoint_task가 부르는 Task 클래스

- **레지스트리**: `environment/aerial_gym/task/__init__.py`에서  
  `task_registry.register("position_setpoint_task", PositionSetpointTask, position_setpoint_task_config)`
- **실제 클래스**: `aerial_gym.task.position_setpoint_task.position_setpoint_task.PositionSetpointTask`
- **runner 연결**: `runner.py`에서  
  `env_configurations.register("position_setpoint_task", { "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task", **kwargs), ... })`  
  → `--task=position_setpoint_task` 시 위 클래스 + config 사용

---

## 3. env_name / robot_name / controller_name 레지스트리 연결

- **env_name**:  
  - **태스크 config**: `position_setpoint_task_config.py`의 `env_name = "empty_env"`  
  - **연결**: SimBuilder.build_env() → EnvManager → `env_config_registry.make_env(env_name)`  
  - **등록**: `environment/aerial_gym/env_manager/__init__.py`에서  
    `env_config_registry.register("empty_env", EmptyEnvCfg)` 등
- **robot_name**:  
  - **태스크 config**: `robot_name = "base_quadrotor"`  
  - **연결**: EnvManager.create_sim() → `robot_registry.make_robot(robot_name, ...)` (robot_registry는 robot_registry에서 로드)
- **controller_name**:  
  - **태스크 config**: `controller_name = "lee_attitude_control"`  
  - **연결**: RobotManager 생성 시 controller_registry에서 컨트롤러 로드

---

## 4. 관측/행동 차원 및 에피소드 길이

| 항목 | 값 | 위치 |
|------|-----|------|
| observation_space_dim | 13 | position_setpoint_task_config.py |
| action_space_dim | 4 | position_setpoint_task_config.py |
| episode_len_steps | 500 | position_setpoint_task_config.py |

---

## 5. Reward 관련 파라미터

- **config**: `position_setpoint_task_config.py`의 `reward_parameters`  
  - pos_error_gain1/exp1, pos_error_gain2/exp2, dist_reward_coefficient, max_dist, action_diff_penalty_gain, absolute_action_reward_gain, **crash_penalty: -100**
- **실제 사용**: `position_setpoint_task.py`의 `compute_reward()`는 **하드코딩**된 수식 사용 (거리/자세/각속도 보상, crash 시 -20). config의 reward_parameters는 일부만 반영 가능한 구조.

---

## 6. Collision / Timeout 처리 위치

- **Collision (termination)**  
  - **판정**: `env_manager/env_manager.py`의 `compute_observations()`  
    `collision_tensor += (norm(robot_contact_force_tensor, dim=1) > collision_force_threshold)`  
  - **전달**: `obs_dict["crashes"]` = `collision_tensor`, task에서 `self.terminations = self.obs_dict["crashes"]`  
  - **리셋**: `post_reward_calculation_step()` → `reset_terminated_and_truncated_envs()`에서 `reset_on_collision`이면 collision된 env 리셋
- **Timeout (truncation)**  
  - **판정**: `position_setpoint_task.py` step() 내  
    `self.truncations[:] = (self.sim_env.sim_steps > self.task_config.episode_len_steps)`  
  - **리셋**: 동일하게 `post_reward_calculation_step()`에서 truncation된 env 리셋

---

## 7. 핵심 질문 답변

| 질문 | 답변 |
|------|------|
| **Reward는 어디서 계산되는가?** | `environment/aerial_gym/task/position_setpoint_task/position_setpoint_task.py`의 `compute_reward()` (JIT). 호출은 `compute_rewards_and_crashes(obs_dict)`에서 수행. |
| **Success 조건은 어디서 정의되는가?** | **position_setpoint_task에는 success 조건 없음.** `self.infos = {}`로 비어 있음. navigation_task에는 truncation 시점에 `norm(target - robot_position) < 1.0`이면 success를 infos에 넣음. |
| **Collision 정보는 관측으로 들어오는가, info에서만 주는가?** | **관측에는 직접 안 들어감.** 13차원 obs는 [goal error(3), quat(4), body_linvel(3), body_angvel(3)]. collision은 **terminations**로 반환되며, task_obs에 "collisions" 키가 있으나 rl_games wrapper는 "observations"만 넘김. infos는 비어 있음. |
| **Goal setpoint는 어디서 생성되는가?** | **position_setpoint_task**: `reset()` / `reset_idx()`에서 `self.target_position[:, 0:3] = 0.0`으로 **고정 (원점)**. 랜덤 목표 생성 없음. |

---

## 8. 결론

- **레지스트리**: env / robot / controller는 각 config의 이름으로 레지스트리와 연결됨.
- **보상/종료**: reward와 collision/truncation 처리 위치는 명확함.
- **Success / Goal**: 논문용으로는 **success 정의 추가** 및 **goal 랜덤 샘플링**이 필요함.
