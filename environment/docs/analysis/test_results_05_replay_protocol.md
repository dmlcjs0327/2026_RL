# 5단계: Replay / 시각 확인 프로토콜

**목표**: 학습 성공/실패를 정량+정성으로 보기 위한 고정 평가 프로토콜.

---

## 1. Replay 표준 설정

- **학습** (작업 디렉터리: `environment/aerial_gym/rl_training/rl_games/`):  
  `python runner.py --task=position_setpoint_task --num_envs=256 --headless=True --use_warp=True`
- **재생** (동작 확인된 예):  
  `python runner.py --task=position_setpoint_task --num_envs=8 --headless=False --play --checkpoint=./runs/gen_ppo_10-05-40-05/nn/last_gen_ppo_ep_400_rew__7625.1924_.pth`
- **선행 조건**: 학습으로 생성된 체크포인트 파일 (또는 위와 같은 경로의 .pth).

---

## 2. 실행 여부

- **사용자 환경**에서 재생 명령 동작 확인됨 (예: `--checkpoint=./runs/gen_ppo_10-05-40-05/nn/last_gen_ppo_ep_400_rew__7625.1924_.pth`).

---

## 3. 각 환경에서 확인할 것 (실행 시)

- 목표점으로 실제 수렴하는지  
- obstacle 환경에서 회피/충돌 패턴  
- forest 환경에서 이상 진동 여부  
- controller saturation 유무  

---

## 4. 결론

- 프로토콜은 위 명령과 확인 항목으로 고정 가능.
- Isaac Gym 설치 및 학습 완료 후 동일 설정으로 replay 실행해 정량/정성 평가하면 됨.
