# Aerial Gym Simulator 공식 문서 요약

출처: [Aerial Gym Simulator 공식 문서](https://ntnu-arl.github.io/aerial_gym_simulator/)

이 문서는 https://ntnu-arl.github.io/aerial_gym_simulator/ 사이트 내 모든 섹션을 검토해 핵심 정보만 정리한 요약입니다. 논문용 baseline 및 custom task 설계 시 참고용으로 사용하세요.

---

## 1. 개요 및 특징

- **정의**: MAV(마이크로 항공체)를 위한 **고충실도 물리 기반 시뮬레이터**. NVIDIA **Isaac Gym** 위에 구축.
- **용도**: 학습 기반 방법으로 다중로터 비행·장애물 환경 항법 훈련.
- **라이선스**: BSD-3-Clause, 오픈소스.
- **주요 특징** ([공식 홈](https://ntnu-arl.github.io/aerial_gym_simulator/)):
  - **모듈형·확장 가능**: custom 환경, 로봇, 센서, **태스크**, 컨트롤러를 쉽게 추가·변경.
  - **GPU 병렬 기하 컨트롤러**: 수천 대 다중로터 동시 제어.
  - **NVIDIA Warp 기반 커스텀 렌더링**: depth/segmentation, LiDAR 등 ray-casting 센서.
  - **태스크 정의 중심**: 동일 환경으로 여러 RL 태스크(포지션 setpoint, trajectory tracking, navigation 등)를 **해석만 바꿔** 사용.
- **Isaac Lab / Isaac Sim**: 지원 예정으로 개발 중.

---

## 2. 설치 및 실행 (Getting Started)

- **문서**: [2_getting_started](https://ntnu-arl.github.io/aerial_gym_simulator/2_getting_started/)
- **설치 순서**: conda 환경 → PyTorch/CUDA → Isaac Gym (`python/setup`) → **gymutil.py 수정** (`parse_args()` → `parse_known_args()`) → urdfpy 소스 설치 → Aerial Gym `pip install -e .`
- **테스트**: `python3 position_control_example.py` (examples 폴더).
- **예제**:
  - **Position control**: SimBuilder로 `empty_env`, `lee_position_control`, `num_envs=64` 등 생성 후 action으로 setpoint 전달.
  - **RL 인터페이스**: `task_registry.make_task("position_setpoint_task")` → `reset()`, `step(actions)` 반복.

---

## 3. 시뮬레이션 구성 요소 (Simulation Components)

- **문서**: [4_simulation_components](https://ntnu-arl.github.io/aerial_gym_simulator/4_simulation_components/)

### 3.1 레지스트리

- **sim**, **env**, **robot**, **controller**, **task** 등에 대한 **이름–설정 매핑**.
- 런타임에 새 설정을 등록해 코드에서 이름으로 선택 가능.

### 3.2 시뮬 파라미터

- **PhysX** 기본. `dt`, `gravity`, `substeps`, `physx` (threads, solver, contact 등) 설정.
- config에서 변경·등록 후 시뮬 재시작 시 적용.

### 3.3 에셋 (Assets)

- **config/asset_config**. URDF 기반, `BaseAssetParams`: 개수, 폴더, 위치/자세 비율, density, damping, collision, force sensor, segmentation 등.
- 환경별로 에셋 타입을 켜고 `asset_type_to_dict_map`으로 파라미터 클래스 연결.

### 3.4 환경 (Environments)

- **config/env_config**. 로봇 주변 물리 세계(에셋, bounds, collision threshold 등) 정의.
- **EmptyEnvCfg**: `include_asset_type={}`, bounds는 `env_spacing`으로.
- **EnvWithObstaclesCfg**: panels, objects, walls 등 `include_asset_type` + `asset_type_to_dict_map`.
- **collision_force_threshold**, **reset_on_collision** 등으로 충돌 시 리셋 제어.

### 3.5 태스크 (Tasks)

- **환경과의 차이** ([문서](https://ntnu-arl.github.io/aerial_gym_simulator/4_simulation_components/)):
  - **Environment**: 로봇, 물리 주변, 에셋, 센서 파라미터 등 “무엇이 시뮬에 있는가”.
  - **Task**: 시뮬 세계를 **어떻게 해석해 RL 목표로 쓸지** (보상, 관측, 종료 조건). 같은 환경으로 setpoint / trajectory / navigation / perch 등 여러 태스크 가능.
- **config/task_config**: `task_config` 클래스로 `sim_name`, `env_name`, `robot_name`, `controller_name`, `num_envs`, `observation_space_dim`, `action_space_dim`, `episode_len_steps`, `reward_parameters` 등 정의.
- **PositionSetpointTask** 예시:
  - `step()`: action 변환 → `sim_env.step()` → `compute_rewards_and_crashes()` → truncation 판단 → `post_reward_calculation_step()` → `(task_obs, rewards, terminations, truncations, infos)` 반환.
  - **관측**: `process_obs_for_task()`에서 `target_position - robot_position`, `robot_orientation`, `robot_body_linvel`, `robot_body_angvel` (문서 기준 0:3, 3:7, 7:10, 10:13).
  - **Gymnasium API** 준수 (terminated/truncated 분리).

---

## 4. 로봇과 컨트롤러 (Robots and Controllers)

- **문서**: [3_robots_and_controllers](https://ntnu-arl.github.io/aerial_gym_simulator/3_robots_and_controllers/)

### 4.1 로봇

- **Underactuated Quadrotor** (기본), **Fully-actuated Octarotor**, **Arbitrary Configuration** (8모터 등) 예시 제공.
- 질량·관성은 URDF에서 설정. 컨트롤러 gain은 로봇에 맞게 튜닝.

### 4.2 컨트롤러

- **Lee 등 기하 제어** (Position, Velocity, Acceleration, Attitude, Body-rate 등) GPU 병렬화.
- **Attitude-Thrust / Body-rate-Thrust**: \(e_R\), \(e_\Omega\) 기반 desired moment; thrust는 직접 입력.
- **Position/Velocity/Acceleration**: desired force·orientation 계산 후 내부적으로 attitude 제어로 연결.
- **Motor allocation**: allocation matrix로 wrench → 모터 힘. pseudo-inverse 사용.
- **Motor model**: 1차 시스템, time constant·thrust clamp·thrust rate clamp.
- **Drag**: body frame에서 선형·이차 항 (속도·각속도).

### 4.3 position_setpoint_task에서의 컨트롤

- 문서·config 기준: `lee_attitude_control` 사용 시 **command = [thrust, roll, pitch, yaw_rate]** (vehicle frame, -1~1 스케일).

---

## 5. 커스터마이징 (Customization)

- **문서**: [5_customization](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/)

- **Custom physics**: sim_registry에 custom sim params 등록.
- **Custom environments**: `include_asset_type`, `asset_type_to_dict_map`, bounds 등으로 새 env 클래스 정의 후 env_registry 등록.
- **Custom controllers**: controller_registry에 등록 (예: FullyActuatedController).
- **Custom robots**: robot config에서 allocation matrix, motor model 등 정의 후 robot_registry.
- **Custom tasks**: `BaseTask` 상속, `task_obs`, `reset`/`reset_idx`/`step`, `compute_reward` 구현 후 **task_registry.register_task("custom_task", CustomTask)** (또는 task/__init__.py에 등록).
- **Custom sensors**: LiDAR/Camera 설정 파라미터 (height, width, FOV, range, noise 등) 변경 또는 새 센서 구현.

---

## 6. RL 훈련 (RL Training)

- **문서**: [6_rl_training](https://ntnu-arl.github.io/aerial_gym_simulator/6_rl_training/)

### 6.1 태스크 정의와 통합

- **Task**가 시뮬 전체를 인스턴스화하고, RL 프레임워크는 Task만 래핑하면 됨. “Minimalistic integration”으로 다른 RL 라이브러리 추가 용이.

### 6.2 Position-control 정책 훈련

- **설정**: `config/task_config/position_setpoint_task_config.py`에서 `env_name` (`empty_env` / `env_with_obstacles` / `forest_env`), `robot_name`, `controller_name` 등 선택.
- **명령** (rl_games):
  - 학습: `python3 runner.py --task=position_setpoint_task --num_envs=8192 --headless=True --use_warp=True`
  - 재생: `python3 runner.py --task=position_setpoint_task --num_envs=16 --headless=False --use_warp=True --checkpoint=<path>.pth --play`
- **학습 시간**: 단일 RTX 3090 기준 **1분 미만** (문서 기준).

### 6.3 Navigation 정책 (depth 이미지)

- DCE(Deep Collision Encoding) 기반 정책 예시. 로봇 config에서 **camera 활성화** 필요.
- `ppo_aerial_quad_navigation.yaml`, `runner.py --file=./ppo_aerial_quad_navigation.yaml --num_envs=1024 --headless=True` 등. 학습 약 1시간 (RTX 3090).

### 6.4 rl-games / Sample Factory / CleanRL

- **rl-games**: `ExtractObsWrapper`로 `observations["observations"]`만 반환, `terminated|truncated` → dones. `AERIALRLGPUEnv`가 env_creator로 Task 래핑.
- **Sample Factory**: Task를 Dict observation 등으로 래핑하는 VecEnv 예시.
- **CleanRL**: Task를 그대로 `task_registry.make_task()`로 사용 가능 (RecordEpisodeStatisticsTorch 등 래퍼만 추가).

---

## 7. 센서 및 렌더링 (Sensors and Rendering)

- **문서**: [8_sensors_and_rendering](https://ntnu-arl.github.io/aerial_gym_simulator/8_sensors_and_rendering/)

- **활성화**: 로봇 config의 `sensor_config`에서 `enable_camera`, `enable_lidar`, `enable_imu` 및 해당 config 클래스 지정.
- **Warp 센서**: depth, range, pointcloud, segmentation, surface normal. FOV, resolution, range, noise, randomize_placement 등 설정 가능.
- **LiDAR**: ray-casting, range/segmentation 이미지, pointcloud (world/sensor frame 선택).
- **Segmentation**: mesh vertex “velocity” 필드에 semantic 인코딩 후 ray hit에서 읽는 방식.
- **IMU**: 가속도·각속도, bias random walk·노이즈 모델 (VN-100 등 참고).
- **Warp vs Isaac Gym 카메라**: 동적 환경은 Isaac Gym 필요; 속도·커스터마이징은 Warp 권장. LiDAR/커스텀 센서는 Warp 구현 사용.

---

## 8. Sim2Real

- **문서**: [9_sim2real](https://ntnu-arl.github.io/aerial_gym_simulator/9_sim2real/)

- **PX4 연동**: `position_setpoint_task_sim2real_px4` 태스크, x500 로봇, Holybro X500 v2 예시.
- **보상/액션 한계/모터 계수/시정수/무게/관성** 등 플랫폼 맞춤 튜닝 안내.
- **변환**: PyTorch → TFLite Micro (resources/conversion), PX4 포크·빌드 절차 설명.

---

## 9. FAQ 및 트러블슈팅

- **문서**: [7_FAQ_and_troubleshooting](https://ntnu-arl.github.io/aerial_gym_simulator/7_FAQ_and_troubleshooting/)

- **Environment vs Task**: 환경 = “무엇이 있는가”, 태스크 = “그걸로 어떤 RL 목표를 달성할 것인가”.
- **로봇 초기 pose**: `min_init_state` / `max_init_state` (비율·roll/pitch/yaw 범위).
- **센서 pose 랜덤**: `randomize_placement` (Warp에서 권장).
- **Viewer 빈 화면**: NVIDIA 드라이버, `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES` 확인.
- **rgbImage buffer error 999**: Vulkan `VK_ICD_FILENAMES` 설정.
- **에셋 충돌 무시**: quaternion 정규화 및 `[qx,qy,qz,qw]` 형식 확인.
- **urdfpy**: pip 대신 **소스 설치** 권장 (실린더 메시 버그 회피).

---

## 10. 인용 (Citing)

- **시뮬레이터**: Kulkarni et al., IEEE RA-L 2025, “Aerial Gym Simulator: A Framework for Highly Parallelized Simulation of Aerial Robots”.
- **Navigation 정책 (DCE)**: Kulkarni & Alexis, ICRA 2024, “Reinforcement Learning for Collision-free Flight Exploiting Deep Collision Encoding”.

---

## 11. 우리 분석 문서와의 대응

| 공식 문서 내용 | 우리 코드/분석 문서 |
|----------------|---------------------|
| Task = 보상/관측/종료 해석 | [aerial_gym_code_analysis.md](aerial_gym_code_analysis.md) §3–5, [test_results_06_task_internals.md](test_results_06_task_internals.md) |
| env_name / empty / obstacles / forest | [test_results_03_env_switch.md](test_results_03_env_switch.md), position_setpoint_task_config.py |
| observation 0:3, 3:7, 7:10, 10:13 | position_setpoint_task.process_obs_for_task(), §3.1 |
| rl_games 학습/재생 명령 | [test_results_02_log_structure.md](test_results_02_log_structure.md), [test_results_05_replay_protocol.md](test_results_05_replay_protocol.md) |
| Custom task 등록 | [test_results_07_custom_task_decision.md](test_results_07_custom_task_decision.md), [5_customization](https://ntnu-arl.github.io/aerial_gym_simulator/5_customization/#custom-tasks) |
| terminated/truncated 분리 | env_manager, position_setpoint_task.step(), §5 |

---

이 요약은 공식 사이트를 “면밀히 검토”한 결과를 한곳에 모은 것이며, 상세·최신 내용은 반드시 [ntnu-arl.github.io/aerial_gym_simulator](https://ntnu-arl.github.io/aerial_gym_simulator/) 를 참고하세요.
