"""
Goal-conditioned position setpoint task config.
- Random goal sampling in env bounds.
- Sparse reward + collision penalty.
- success_threshold for success condition (for infos).
"""
import torch


class task_config:
    seed = 1
    sim_name = "base_sim"
    env_name = "empty_env"
    robot_name = "base_quadrotor"
    controller_name = "lee_attitude_control"
    args = {}
    num_envs = 4096
    use_warp = False
    headless = False
    device = "cuda:0"
    observation_space_dim = 13
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 500
    return_state_before_reset = False

    # Goal-conditioned / sparse reward
    success_threshold = 0.5  # [m] distance to goal to count as success
    success_reward = 1.0
    collision_penalty = -10.0  # sparse: only on collision
    # target sampling: ratio in [target_min_ratio, target_max_ratio] of env bounds
    target_min_ratio = [0.2, 0.2, 0.2]
    target_max_ratio = [0.8, 0.8, 0.8]

    reward_parameters = {
        "pos_error_gain1": [2.0, 2.0, 2.0],
        "pos_error_exp1": [1 / 3.5, 1 / 3.5, 1 / 3.5],
        "pos_error_gain2": [2.0, 2.0, 2.0],
        "pos_error_exp2": [2.0, 2.0, 2.0],
        "dist_reward_coefficient": 7.5,
        "max_dist": 15.0,
        "action_diff_penalty_gain": [1.0, 1.0, 1.0],
        "absolute_action_reward_gain": [2.0, 2.0, 2.0],
        "crash_penalty": -100,
    }
