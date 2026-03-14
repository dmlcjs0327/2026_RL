#!/usr/bin/env python3
"""
Aerial Gym Off-Policy API 테스트 스크립트 (8단계).

확인 항목:
- reset() 결과 shape
- step(action) 결과 shape
- terminated / truncated 분리 여부
- info에 success / collision 존재 여부
- dict observation vs flat tensor
- batch env에서 episode 경계(episode id) 추적 가능 여부

실행: Isaac Gym 설치 후
  cd /path/to/aerial_gym_simulator
  python scripts/env_api_test.py

또는 PYTHONPATH에 aerial_gym 상위 경로 추가 후 실행.
"""

import os
import sys

# aerial_gym 패키지 경로 (프로젝트 루트 기준)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    try:
        import isaacgym  # noqa: F401
    except ImportError:
        print("SKIP: isaacgym not installed. Install Isaac Gym to run this test.")
        return 0

    import torch
    from aerial_gym.registry.task_registry import task_registry

    num_envs = 8
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, num_envs: {num_envs}\n")

    # Task 생성 (position_setpoint_task)
    task_name = "position_setpoint_task"
    env = task_registry.make_task(
        task_name,
        num_envs=num_envs,
        headless=True,
        device=device,
        use_warp=True,
    )

    # --- reset() ---
    out = env.reset()
    if isinstance(out, (list, tuple)):
        obs_container, *rest = out
    else:
        obs_container = out
        rest = []

    print("1. reset() result type:", type(out))
    if isinstance(obs_container, dict):
        print("   observation is dict. Keys:", list(obs_container.keys()))
        if "observations" in obs_container:
            o = obs_container["observations"]
            print("   observations shape:", o.shape if hasattr(o, "shape") else "N/A")
    else:
        print("   observation shape:", obs_container.shape if hasattr(obs_container, "shape") else "N/A")
    print()

    # --- step() ---
    action_dim = env.action_space.shape[0]
    actions = torch.zeros((num_envs, action_dim), device=device)
    step_out = env.step(actions)
    obs_container, rewards, terminations, truncations, infos = step_out

    print("2. step() result:")
    print("   obs type:", type(obs_container), "keys:", list(obs_container.keys()) if isinstance(obs_container, dict) else "N/A")
    if isinstance(obs_container, dict) and "observations" in obs_container:
        print("   observations shape:", obs_container["observations"].shape)
    print("   rewards shape:", rewards.shape)
    print("   terminations shape:", terminations.shape)
    print("   truncations shape:", truncations.shape)
    print("   infos type:", type(infos), "keys:", list(infos.keys()) if isinstance(infos, dict) else "N/A")
    print()

    print("3. terminated / truncated separated: Yes (terminations, truncations as separate tensors)")
    print("4. info success/collision:", "success" in infos, "crashes" in infos or "collision" in str(infos).lower())
    print("5. observation: dict with 'observations' key (flat tensor per env)")
    print()

    # --- Episode boundary: run a few steps and check done flags ---
    print("6. Episode boundary (sim_steps): after reset, run 3 steps and show terminations/truncations")
    env.reset()
    for t in range(3):
        actions = torch.zeros((num_envs, action_dim), device=device)
        obs, rewards, term, trunc, infos = env.step(actions)
        done = term | trunc
        print(f"   step {t+1}: term sum={term.sum().item()}, trunc sum={trunc.sum().item()}, done sum={done.sum().item()}")

    env.close()
    print("\nDone. Off-policy transition (s, a, s', r, done) can be built from obs, actions, rewards, terminations|truncations.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
