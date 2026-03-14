"""
Goal-conditioned position setpoint task for off-policy RL (TD3/SAC, HER, etc.).
- Random goal sampling within env bounds on reset.
- Sparse reward: success_reward at goal, collision_penalty on crash, 0 otherwise.
- infos: success, crashes, timeouts for logging and evaluation.
"""
from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import (
    quat_apply_inverse,
    quat_axis,
    torch_rand_float_tensor,
    torch_interpolate_ratio,
)
from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("position_setpoint_gc_task")


class PositionSetpointGCTask(BaseTask):
    """Goal-conditioned position setpoint: random goal, sparse reward, success/crash/timeout in infos."""

    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for goal-conditioned position setpoint task.")
        logger.info(
            "Sim: {}, Env: {}, Robot: {}, Controller: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)
        self.counter = 0

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )
        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)

        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {
                "observations": Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
                "observation": Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
                "achieved_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        self.num_envs = self.sim_env.num_envs

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "observation": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "achieved_goal": torch.zeros(
                (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
            ),
            "desired_goal": torch.zeros(
                (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        self.sim_env.delete_env()

    def _sample_target_positions(self, env_ids):
        """Sample target_position[env_ids] within env bounds using config ratio range."""
        target_ratio = torch_rand_float_tensor(
            self.target_min_ratio[env_ids], self.target_max_ratio[env_ids]
        )
        self.target_position[env_ids] = torch_interpolate_ratio(
            self.obs_dict["env_bounds_min"][env_ids],
            self.obs_dict["env_bounds_max"][env_ids],
            target_ratio,
        )

    def reset(self):
        self._sample_target_positions(torch.arange(self.sim_env.num_envs, device=self.device))
        self.infos = {}
        self.sim_env.reset()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        if len(env_ids) > 0:
            self._sample_target_positions(env_ids)
        self.sim_env.reset_idx(env_ids)
        return

    def render(self):
        return None

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.actions = actions

        self.sim_env.step(actions=self.actions)
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        dones = torch.logical_or(self.terminations > 0, self.truncations > 0)
        terminal_obs = self.build_observation_tensor()
        terminal_achieved_goal = self.obs_dict["robot_position"].clone()
        terminal_desired_goal = self.target_position.clone()

        # infos for logging / evaluation (HER, metrics)
        dist_to_goal = torch.norm(
            self.target_position - self.obs_dict["robot_position"], dim=1
        )
        success = (dist_to_goal < self.task_config.success_threshold).float()
        self.infos = {
            "success": success,
            "crashes": self.terminations.float().clone(),
            "timeouts": self.truncations.float().clone(),
            "desired_goal": terminal_desired_goal,
            "achieved_goal": terminal_achieved_goal,
            "terminal_observation": terminal_obs,
            "terminal_achieved_goal": terminal_achieved_goal,
            "terminal_desired_goal": terminal_desired_goal,
            "done": dones.float(),
        }

        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)

        return self.get_return_tuple()

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        self.task_obs["observations"][:] = self.build_observation_tensor()
        self.task_obs["observation"][:] = self.task_obs["observations"]
        self.task_obs["achieved_goal"][:] = self.obs_dict["robot_position"]
        self.task_obs["desired_goal"][:] = self.target_position
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def build_observation_tensor(self):
        obs = torch.zeros(
            (self.sim_env.num_envs, self.task_config.observation_space_dim),
            device=self.device,
            requires_grad=False,
        )
        obs[:, 0:3] = self.target_position - self.obs_dict["robot_position"]
        obs[:, 3:7] = self.obs_dict["robot_orientation"]
        obs[:, 7:10] = self.obs_dict["robot_body_linvel"]
        obs[:, 10:13] = self.obs_dict["robot_body_angvel"]
        return obs

    def compute_reward_from_goals(self, achieved_goal, desired_goal, crashes):
        dist = torch.norm(desired_goal - achieved_goal, dim=1)
        reward = torch.where(
            dist < self.task_config.success_threshold,
            self.task_config.success_reward * torch.ones_like(dist, device=self.device),
            torch.zeros_like(dist, device=self.device),
        )
        reward = torch.where(
            crashes > 0.0,
            self.task_config.collision_penalty * torch.ones_like(reward),
            reward,
        )
        return reward

    def recompute_reward(self, relabeled_next_obs, achieved_goal, desired_goal, crashes):
        _ = relabeled_next_obs
        return self.compute_reward_from_goals(achieved_goal, desired_goal, crashes)

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        angular_velocity = obs_dict["robot_body_angvel"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        return compute_reward_sparse(
            pos_error_vehicle_frame,
            obs_dict["crashes"],
            self.task_config.success_threshold,
            self.task_config.success_reward,
            self.task_config.collision_penalty,
        )


@torch.jit.script
def compute_reward_sparse(
    pos_error,
    crashes,
    success_threshold,
    success_reward,
    collision_penalty,
):
    # type: (Tensor, Tensor, float, float, float) -> Tuple[Tensor, Tensor]
    dist = torch.norm(pos_error, dim=1)
    # Sparse: success_reward only when within threshold, else 0
    reward = torch.where(
        dist < success_threshold,
        success_reward * torch.ones_like(dist, device=pos_error.device),
        torch.zeros_like(dist, device=pos_error.device),
    )
    # Mark as crash if contact-based crash or out-of-range; apply penalty (in-place on crashes)
    crashes[:] = torch.where(dist > 8.0, torch.ones_like(crashes), crashes)
    reward[:] = torch.where(crashes > 0.0, collision_penalty * torch.ones_like(reward), reward)
    return reward, crashes
