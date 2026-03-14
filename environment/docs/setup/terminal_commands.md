# 실험환경 구축 — 터미널에 입력할 명령어

아나콘다(conda) 환경 활성화부터 실행까지 **맨 처음부터** 순서대로 정리했습니다.  
아래 명령어를 **위에서부터** 터미널에 복사해 실행하세요.

**본인 환경 요약**: [environment_spec.md](environment_spec.md) — Ubuntu 22.04 듀얼 부팅, i9-13900KF / RTX 3090 / 64GB RAM. Conda 가상환경 **aerialgym** (Python 3.8) 사용 중.

---

## 0. 맨 처음: 터미널 열기 및 Conda 환경 활성화

### 0-1. 터미널 열기

- **본 환경**: **Ubuntu 22.04** 네이티브(듀얼 부팅)이므로, Ubuntu에서 터미널(또는 VS Code 터미널)을 연다.
- (참고) Windows 사용 시: WSL2 Ubuntu 터미널 또는 Anaconda Prompt.

### 0-2. Conda(아나콘다) 환경 활성화

Aerial Gym + Isaac Gym을 설치해 둔 **conda 환경**을 활성화합니다.  
**본 환경**: `aerialgym` (Python 3.8).

```bash
conda activate aerialgym
```

### 0-3. Isaac Gym용 환경 변수 (선택, 오류 시 설정)

Isaac Gym을 쓰려면 아래 변수가 필요할 수 있습니다.  
**한 번만 설정**하거나, `~/.bashrc` / `~/.zshrc` 에 넣어 두면 다음부터는 자동 적용됩니다.

```bash
# Conda 환경의 lib 경로 (본인 conda 경로에 맞게 수정)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# Vulkan 관련 오류(rgbImage buffer error 999)가 나면 추가
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

- Windows에서 Anaconda Prompt를 쓰는 경우: `%CONDA_PREFIX%\Library\bin` 등을 PATH에 넣는 방식으로 맞춰 주세요. (Isaac Gym 공식 문서 참고.)

### 0-4. 환경 루트로 이동 (선택)

어디서 실행하느냐에 따라 필요할 때만 사용합니다.

```bash
cd /home/user/aerial_gym_simulator/environment
```

- **학습/재생**은 아래처럼 **runner가 있는 디렉터리**로 이동한 뒤 실행합니다.

---

## 1. Phase 1-1: 학습·재생 스모크 테스트

### 1-1. runner가 있는 디렉터리로 이동

학습/재생은 **항상 이 디렉터리**에서 실행합니다.

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
```

### 1-2. 학습 실행 (1~2분 후 Ctrl+C로 중단해도 됨)

```bash
python runner.py --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True
```

- **확인**: 실행 후 `runs/` 폴더 아래에 새 실험 폴더와 `nn/*.pth` 파일이 생기는지 확인.

### 1-3. 재생 (체크포인트가 있을 때)

```bash
python runner.py --task=position_setpoint_task --num_envs=1 --headless=False --play --checkpoint=./runs/gen_ppo_10-09-14-57/nn/last_gen_ppo_ep_400_rew__11442.322_.pth
```

- **확인**: 창이 뜨고 드론이 움직이면 정상. 체크포인트 경로는 본인 환경에 맞게 수정.

---

## 2. Phase 1-2: 환경 3종 전환 테스트

먼저 **config에서 env_name만 바꾼 뒤** 아래 명령으로 실행합니다.

- **수정 파일**: `environment/aerial_gym/config/task_config/position_setpoint_task_config.py`  
- **수정 내용**: 8번째 줄 `env_name = "empty_env"` 를 아래 중 하나로 바꿔 저장.

### 2-1. empty_env (기본값이면 그대로 실행)

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_task --num_envs=8 --headless=False --use_warp=True
```

### 2-2. env_with_obstacles

- `position_setpoint_task_config.py`에서 `env_name = "env_with_obstacles"` 로 변경 후:

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_task --num_envs=8 --headless=False --use_warp=True
```

### 2-3. forest_env

- `position_setpoint_task_config.py`에서 `env_name = "forest_env"` 로 변경 후:

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_task --num_envs=8 --headless=False --use_warp=True
```

- 테스트 후 **다음 단계를 위해** `env_name = "empty_env"` 로 되돌려 두어도 됩니다.

---

## 3. Phase 1-3: Off-policy API 테스트

환경 **루트**에서 실행합니다.

```bash
cd /home/user/aerial_gym_simulator/environment
python scripts/env_api_test.py
```

- **확인**: `reset()` / `step()` 결과 shape, `terminations` / `truncations`, `infos` 키가 출력되는지 확인. (Isaac Gym 미설치 환경이면 `SKIP` 메시지가 나옵니다.)

---

## 4. Phase 2 이후: Goal-Conditioned Custom Task 실행

Custom task 등록이 끝난 뒤에는 아래처럼 **새 task 이름**으로 학습·재생할 수 있습니다.

### 학습 (예: empty_env, 256 env)

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_gc_task --num_envs=256 --headless=True --use_warp=True
```

### 재생

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_gc_task --num_envs=8 --headless=False --play --checkpoint=./runs/<실험폴더>/nn/<체크포인트>.pth
```

---

## 5. num_envs 스케일링 테스트 (Phase 3)

같은 작업 디렉터리에서 `num_envs`만 바꿔 가며 실행해 보기.

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_task --num_envs=128 --headless=True --use_warp=True
```

```bash
python runner.py --task=position_setpoint_task --num_envs=512 --headless=True --use_warp=True
```

```bash
python runner.py --task=position_setpoint_task --num_envs=1024 --headless=True --use_warp=True
```

- 각각 짧게 돌린 뒤 GPU 메모리(`nvidia-smi`), iteration 속도, 중간에 죽는지 확인.

---

## 한 번에 복사해서 쓸 수 있는 예시 (맨 처음 실행)

아래를 **한 블록씩** 터미널에 붙여 넣고 실행하면, 환경 활성화부터 학습 1회 실행까지 한 번에 진행할 수 있습니다.

**1) Conda 환경 활성화** (환경 이름은 본인에 맞게 수정)

```bash
conda activate aerialgym
```

**2) Isaac Gym 라이브러리 경로** (이미 .bashrc 등에 있으면 생략 가능)

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

**3) runner 디렉터리로 이동 후 학습 실행**

```bash
cd /home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/rl_games
python runner.py --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True
```

- **본 환경**: Ubuntu 22.04 네이티브이므로 위 경로(`/home/user/aerial_gym_simulator/environment/...`) 그대로 사용. PC 스펙·OS 정리: [environment_spec.md](environment_spec.md).
