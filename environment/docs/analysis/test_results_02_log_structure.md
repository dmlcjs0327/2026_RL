# 2단계: 짧은 학습 실행 및 로그 구조 확인

**목표**: 로그 저장 위치, 체크포인트 주기, 터미널 메트릭, replay용 체크포인트 경로 파악.

---

## 1. 실행 환경 및 실행 여부

- **실행 명령**:  
  `python runner.py --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True`
- **작업 디렉터리**: `environment/aerial_gym/rl_training/rl_games/` (또는 runner.py가 있는 디렉터리)
- **실행 결과**: **사용자 환경에서는 정상 실행됨** (Isaac Gym 설치된 환경에서 위 명령 실행 가능).  
  에이전트 실행 환경에서는 isaacgym 미설치로 실행하지 못했으나, 동일 명령으로 로그/체크포인트 구조 확인 가능.

---

## 2. 로그 파일 저장 위치

- **runner.py** (303–305행):  
  `os.makedirs("nn", exist_ok=True)`, `os.makedirs("runs", exist_ok=True)`  
  → 실행 시 **현재 작업 디렉터리**에 `nn/`, `runs/` 생성.
- **실제 구조** (사용자 환경 기준):  
  - runs 아래에 실험별 디렉터리 생성 (예: `runs/gen_ppo_10-05-40-05/`).  
  - 체크포인트는 해당 실험 폴더 안의 `nn/` 아래 저장 (예: `runs/gen_ppo_10-05-40-05/nn/`).  
  - 파일명 예: `last_gen_ppo_ep_400_rew__7625.1924_.pth` (epoch, reward 등이 들어간 형식).

---

## 3. 체크포인트 저장 주기

- **ppo_aerial_quad.yaml**: `save_best_after: 10` (10 epoch 이후부터 best 모델 저장).
- **실제 저장 주기/이름**: rl_games 알고리즘 코드에 따름. 일반적으로 주기적 저장 + best 유지.

---

## 4. 터미널 메트릭

- **ExtractObsWrapper**: step 시 `observations["observations"]`, `rewards`, `dones`, `infos` 반환.  
  터미널에 찍히는 메트릭은 **rl_games**가 epoch/rollout 통계를 출력하는 부분에 따름 (예: reward, length, FPS 등).
- **position_setpoint_task**: `infos = {}`이므로 **episode length, success, collision** 등은 현재 기본 task에서 infos로 나오지 않음.  
  논문용 로그 수집을 위해서는 task에서 `infos["episode_length"]`, `infos["success"]`, `infos["collision"]` 등을 채우고, rl_games 로거가 이를 수집하도록 해야 함.

---

## 5. Replay용 체크포인트 경로 형식

- **재생 명령** (사용자 환경에서 동작 확인됨):  
  `python runner.py --task=position_setpoint_task --num_envs=8 --headless=False --play --checkpoint=./runs/gen_ppo_10-05-40-05/nn/last_gen_ppo_ep_400_rew__7625.1924_.pth`
- **체크포인트 경로**: `./runs/<실험폴더>/nn/<체크포인트파일>.pth`  
  - 실험 폴더 이름은 config의 `name`(예: gen_ppo)과 타임스탬프 등으로 생성됨.

---

## 6. 결론

- **사용자 환경**에서 학습·재생 명령이 정상 동작하며, 체크포인트는 `runs/<실험>/nn/*.pth` 형식으로 저장됨.
- **논문용 로그 수집**:  
  - 기본 position_setpoint_task는 **infos가 비어 있어** episode length/success/collision 등을 그대로 쓰기 어렵고,  
  - **custom task 또는 wrapper**에서 해당 필드를 채우고, rl_games/TensorBoard에 넘기는 확장이 필요함.
