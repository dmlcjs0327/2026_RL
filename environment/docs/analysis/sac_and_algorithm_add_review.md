# SAC 사용 가능 여부 및 신규 알고리즘 추가 메뉴얼 검토

공식 문서 [Aerial Gym Simulator](https://ntnu-arl.github.io/aerial_gym_simulator/) 및 프로젝트 코드를 기준으로 정리함.

---

## 1. SAC 사용 가능 여부 (결론: **가능**)

### 1.1 rl_games 측 구현

- **알고리즘 등록**: `rl_games/torch_runner.py`에서 `algo_factory.register_builder('sac', ...)` 로 **SAC**가 등록되어 있음.
- **모델/네트워크**: `model.name: soft_actor_critic`, `network.name: soft_actor_critic` 사용. Replay buffer, actor/critic/alpha 업데이트 등 SAC 로직 구현됨 (`sac_agent.py`).
- **환경 인터페이스**: SAC는 `vec_env.step(actions)` → `(obs, rewards, dones, infos)`, `vec_env.reset()` → `obs` 를 기대함.  
  본 프로젝트의 `ExtractObsWrapper` + `AERIALRLGPUEnv`는 위와 동일한 형식으로 반환하므로 **호환됨**.
- **관측/행동 공간**: `get_env_info()`로 `observation_space`, `action_space` (Box 연속) 전달. SAC는 연속 제어용이므로 **position_setpoint_task** (연속 obs/action)와 맞음.

### 1.2 본 프로젝트에서 빠져 있던 것

- **SAC 전용 YAML 설정이 없음**: 기본 `--file` 이 `ppo_aerial_quad.yaml` 이며, 해당 파일은 `algo.name: a2c_continuous` (PPO)만 사용.
- **runner.py**는 `--file` 로 지정한 YAML의 `algo.name` 을 그대로 쓰므로, **SAC용 YAML만 추가하면** 같은 runner로 SAC 학습 가능.

### 1.3 정리

- **SAC는 rl_games에 구현되어 있고**, 현재 프로젝트의 env wrapper·task와 **인터페이스 호환**됨.
- **SAC를 “사용할 수 있는지”에 대한 답: 예.**  
  프로젝트에 **SAC용 설정 파일**(예: `sac_aerial_quad.yaml`)을 추가한 뒤,  
  `python runner.py --file=sac_aerial_quad.yaml --task=position_setpoint_task ...` 로 실행하면 됨.
- 재생(play) 시에는 rl_games의 `SACPlayer`가 사용되므로, SAC로 학습한 체크포인트는 `--play` 시에도 동일한 설정 파일로 불러오면 됨.

---

## 2. 공식 문서의 “신규 알고리즘 추가” 관련 내용

[Aerial Gym Simulator – RL Training](https://ntnu-arl.github.io/aerial_gym_simulator/) 의 **“RL Training”** 섹션을 보면:

- **제공 예제**: rl-games, Sample Factory, CleanRL **사례**만 설명되어 있음.
- **알고리즘 목록/선택**: PPO 위주로만 안내되어 있고, **SAC/DDPG 등 다른 알고리즘**을 쓰는 방법은 **문서에 없음**.
- **“Adding your own RL frameworks”** (문서 내 해당 절):
  - 내용: *“Kindly refer to the existing implementations for sample-factory, rl-games and CleanRL for reference. You can add your own RL frameworks and contribute via pull requests to this repository.”*
  - 즉, **“기존 구현(rl-games, Sample Factory, CleanRL)을 참고해서 넣을 수 있다”**는 수준이며,  
    **“신규 알고리즘을 추가하는 단계별 메뉴얼”**이나 **“SAC를 쓰는 방법”** 같은 구체적 절은 **없음**.

따라서:

- **“이 프로젝트에 신규 알고리즘(SAC 등)을 추가할 수 있는지”**  
  → 코드 구조상 **가능** (rl_games에 이미 SAC가 있으므로, 설정만 추가하면 됨).
- **“그 방법에 대한 공식 메뉴얼이 있는지”**  
  → **없음.**  
  공식 문서에는 “기존 RL 예제를 참고해 자신만의 프레임워크를 추가할 수 있다”는 안내만 있고,  
  알고리즘별(PPO/SAC 등) 선택 방법이나 “새 알고리즘 추가” 단계별 가이드는 없음.

---

## 3. 본 프로젝트에서 SAC를 쓰는 방법 (실무 요약)

프로젝트에 **SAC용 설정 파일** `environment/aerial_gym/rl_training/rl_games/sac_aerial_quad.yaml` 이 추가되어 있음.

1. **설정 파일**  
   - `algo.name: sac`, `model.name: soft_actor_critic`, `network.name: soft_actor_critic`  
   - SAC 전용: `batch_size`, `replay_buffer_size`, `num_steps_per_episode`, `actor_lr`, `critic_lr`, `gamma`, `init_alpha`, `learnable_temperature`, `num_warmup_frames` 등 (rl_games `sac_agent.py` 및 예제 config 기준).
2. **실행**  
   - 학습:  
     `python runner.py --file=sac_aerial_quad.yaml --task=position_setpoint_task --num_envs=4096 --headless --use_warp`  
   - 재생:  
     `python runner.py --file=sac_aerial_quad.yaml --task=position_setpoint_task --play --checkpoint=<path>`  
3. **env_name**  
   - runner가 `--task` 로 전달한 값을 config의 `env_name` 에 넣으므로, `--task=position_setpoint_task` (또는 `position_setpoint_gc_task`)를 주면 해당 task가 SAC에 연결됨.

이렇게 하면 **별도 알고리즘 구현 없이**, 기존 rl_games SAC를 그대로 사용할 수 있음.
