import inspect
import numpy as np
import os
import yaml
from collections import deque


import isaacgym


from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.helpers import parse_arguments

import gym
from gym import spaces
from argparse import Namespace

from rl_games.common import env_configurations, vecenv

import torch
import distutils

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class CommonMetricTensorboardWriter:
    TAG_MAP = {
        "rewards/step": "eval/return_raw/step",
        "rewards/time": "eval/return_raw/time",
        "shaped_rewards/step": "eval/return_scaled/step",
        "shaped_rewards/time": "eval/return_scaled/time",
        "episode_lengths/step": "eval/episode_length/step",
        "episode_lengths/time": "eval/episode_length/time",
    }

    PROGRESS_TAG_MAP = {
        "rewards/step": "eval/return_raw/progress",
        "shaped_rewards/step": "eval/return_scaled/progress",
        "episode_lengths/step": "eval/episode_length/progress",
    }

    def __init__(self, writer, algo):
        self.writer = writer
        self.algo = algo

    def _max_training_steps(self):
        if getattr(self.algo, "max_frames", -1) and self.algo.max_frames > 0:
            return float(self.algo.max_frames)
        if getattr(self.algo, "max_epochs", -1) and self.algo.max_epochs > 0:
            return float(self.algo.batch_size_envs * self.algo.max_epochs)
        return None

    def _progress_index(self, current_step):
        max_steps = self._max_training_steps()
        if max_steps is None:
            return None
        progress = min(1.0, max(0.0, float(current_step) / max_steps))
        return int(round(progress * 1000.0))

    def add_scalar(self, tag, value, step, *args, **kwargs):
        if tag.startswith("eval/"):
            self.writer.add_scalar(tag, value, step, *args, **kwargs)
            return
        mapped_tag = self.TAG_MAP.get(tag)
        if mapped_tag is None:
            return
        self.writer.add_scalar(mapped_tag, value, step, *args, **kwargs)
        progress_tag = self.PROGRESS_TAG_MAP.get(tag)
        progress_index = self._progress_index(step)
        if progress_tag is not None and progress_index is not None:
            self.writer.add_scalar(progress_tag, value, progress_index, *args, **kwargs)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def __getattr__(self, item):
        return getattr(self.writer, item)


class CommonMetricObserver:
    def __init__(self, window_size=256):
        self.window_size = window_size

    def before_init(self, base_name, config, experiment_name):
        return

    def after_init(self, algo):
        self.algo = algo
        self.writer = algo.writer
        self.success_history = deque(maxlen=self.window_size)
        self.crash_history = deque(maxlen=self.window_size)
        self.timeout_history = deque(maxlen=self.window_size)

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict) or len(done_indices) == 0:
            return
        required_keys = ("success", "crashes", "timeouts")
        if not all(key in infos for key in required_keys):
            return

        done_env_indices = (done_indices[:: self.algo.num_agents] // self.algo.num_agents).view(-1)
        for env_idx in done_env_indices.tolist():
            self.success_history.append(float(infos["success"][env_idx].item()))
            self.crash_history.append(float(infos["crashes"][env_idx].item()))
            self.timeout_history.append(float(infos["timeouts"][env_idx].item()))

    def after_steps(self):
        return

    def after_clear_stats(self):
        self.success_history.clear()
        self.crash_history.clear()
        self.timeout_history.clear()

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.writer is None or len(self.success_history) == 0:
            return

        success_rate = sum(self.success_history) / len(self.success_history)
        crash_rate = sum(self.crash_history) / len(self.crash_history)
        timeout_rate = sum(self.timeout_history) / len(self.timeout_history)
        max_steps = None
        if getattr(self.algo, "max_frames", -1) and self.algo.max_frames > 0:
            max_steps = float(self.algo.max_frames)
        elif getattr(self.algo, "max_epochs", -1) and self.algo.max_epochs > 0:
            max_steps = float(self.algo.batch_size_envs * self.algo.max_epochs)
        progress_index = None
        if max_steps is not None:
            progress_index = int(round(min(1.0, max(0.0, float(frame) / max_steps)) * 1000.0))

        self.writer.add_scalar("eval/success_rate/step", success_rate, frame)
        self.writer.add_scalar("eval/success_rate/time", success_rate, total_time)
        self.writer.add_scalar("eval/crash_rate/step", crash_rate, frame)
        self.writer.add_scalar("eval/crash_rate/time", crash_rate, total_time)
        self.writer.add_scalar("eval/timeout_rate/step", timeout_rate, frame)
        self.writer.add_scalar("eval/timeout_rate/time", timeout_rate, total_time)
        if progress_index is not None:
            self.writer.add_scalar("eval/success_rate/progress", success_rate, progress_index)
            self.writer.add_scalar("eval/crash_rate/progress", crash_rate, progress_index)
            self.writer.add_scalar("eval/timeout_rate/progress", timeout_rate, progress_index)


def _install_common_metric_tensorboard_patch():
    import rl_games.common.a2c_common as a2c_common

    if getattr(a2c_common, "_common_metric_patch_installed", False):
        return

    original_init = a2c_common.A2CBase.__init__

    def patched_init(self, base_name, params):
        original_init(self, base_name, params)
        if self.writer is not None and not isinstance(self.writer, CommonMetricTensorboardWriter):
            self.writer = CommonMetricTensorboardWriter(self.writer, self)

    def patched_write_stats(
        self,
        total_time,
        epoch_num,
        step_time,
        play_time,
        update_time,
        a_losses,
        c_losses,
        entropies,
        kls,
        last_lr,
        lr_mul,
        frame,
        scaled_time,
        scaled_play_time,
        curr_frames,
    ):
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    a2c_common.A2CBase.__init__ = patched_init
    a2c_common.A2CBase.write_stats = patched_write_stats
    a2c_common._common_metric_patch_installed = True


def _install_training_log_patch():
    """Replace rl_games fps-only log with mean reward and mean episode length (and epoch/frames)."""
    import rl_games.common.a2c_common as a2c_common
    _original_print_statistics = a2c_common.print_statistics

    def _custom_print_statistics(
        print_stats, curr_frames, step_time, step_inference_time, total_time,
        epoch_num, max_epochs, frame, max_frames,
    ):
        if not print_stats:
            return
        # Get caller's self (the agent) to read game_rewards / game_lengths
        try:
            caller_frame = inspect.currentframe().f_back
            agent = caller_frame.f_locals.get("self") if caller_frame else None
        except Exception:
            agent = None
        mean_reward = None
        mean_length = None
        if agent is not None and getattr(agent, "game_rewards", None):
            if getattr(agent.game_rewards, "current_size", 0) > 0:
                try:
                    mean_reward = agent.game_rewards.get_mean()
                    mean_reward = mean_reward[0] if hasattr(mean_reward, "__getitem__") else float(mean_reward)
                except Exception:
                    pass
            if getattr(agent, "game_lengths", None) and getattr(agent.game_lengths, "current_size", 0) > 0:
                try:
                    mean_length = float(agent.game_lengths.get_mean())
                except Exception:
                    pass
        # Prefer reward/ep_len; fallback to original fps line
        if mean_reward is not None:
            ep_str = f"  mean_ep_len: {mean_length:.1f}" if mean_length is not None else ""
            epoch_str = f"{epoch_num:.0f}" if max_epochs == -1 else f"{epoch_num:.0f}/{max_epochs:.0f}"
            frame_str = f"{frame:.0f}" if max_frames == -1 else f"{frame:.0f}/{max_frames:.0f}"
            print(f"epoch: {epoch_str}  frames: {frame_str}  mean_reward: {mean_reward:.4f}{ep_str}")
        else:
            _original_print_statistics(
                print_stats, curr_frames, step_time, step_inference_time, total_time,
                epoch_num, max_epochs, frame, max_frames,
            )

    a2c_common.print_statistics = _custom_print_statistics
# import warnings
# warnings.filterwarnings("error")


class ExtractObsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        observations, *_ = super().reset(**kwargs)
        return observations["observations"]

    def step(self, action):
        observations, rewards, terminated, truncated, infos = super().step(action)

        dones = torch.where(
            terminated | truncated,
            torch.ones_like(terminated),
            torch.zeros_like(terminated),
        )

        return (
            observations["observations"],
            rewards,
            dones,
            infos,
        )


class AERIALRLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]["env_creator"](**kwargs)
        self.env = ExtractObsWrapper(self.env)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()

    def reset_done(self):
        return self.env.reset_done()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info["action_space"] = spaces.Box(
            -np.ones(self.env.task_config.action_space_dim),
            np.ones(self.env.task_config.action_space_dim),
        )
        info["observation_space"] = spaces.Box(
            np.ones(self.env.task_config.observation_space_dim) * -np.Inf,
            np.ones(self.env.task_config.observation_space_dim) * np.Inf,
        )
        print(info["action_space"], info["observation_space"])
        return info


env_configurations.register(
    "position_setpoint_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("position_setpoint_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real_px4",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real_px4", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_acceleration_sim2real",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_acceleration_sim2real", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "navigation_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task("navigation_task", **kwargs),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_gc_task",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_gc_task", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_reconfigurable",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_reconfigurable", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_morphy",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_morphy", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

env_configurations.register(
    "position_setpoint_task_sim2real_end_to_end",
    {
        "env_creator": lambda **kwargs: task_registry.make_task(
            "position_setpoint_task_sim2real_end_to_end", **kwargs
        ),
        "vecenv_type": "AERIAL-RLGPU",
    },
)

vecenv.register(
    "AERIAL-RLGPU",
    lambda config_name, num_actors, **kwargs: AERIALRLGPUEnv(config_name, num_actors, **kwargs),
)


def get_args():
    from isaacgym import gymutil

    custom_parameters = [
        {
            "name": "--seed",
            "type": int,
            "default": 0,
            "required": False,
            "help": "Random seed, if larger than 0 will overwrite the value in yaml config.",
        },
        {
            "name": "--tf",
            "required": False,
            "help": "run tensorflow runner",
            "action": "store_true",
        },
        {
            "name": "--train",
            "required": False,
            "help": "train network",
            "action": "store_true",
        },
        {
            "name": "--play",
            "required": False,
            "help": "play(test) network",
            "action": "store_true",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "required": False,
            "help": "path to checkpoint",
        },
        {
            "name": "--file",
            "type": str,
            "default": "ppo_aerial_quad.yaml",
            "required": False,
            "help": "path to config",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": "1024",
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--sigma",
            "type": float,
            "required": False,
            "help": "sets new sigma value in case if 'fixed_sigma: True' in yaml config",
        },
        {
            "name": "--track",
            "action": "store_true",
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "rl_games",
            "help": "the wandb's project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "the entity (team) of wandb's project",
        },
        {
            "name": "--task",
            "type": str,
            "default": "navigation_task",
            "help": "Override task from config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--headless",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "False",
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--use_warp",
            "type": lambda x: bool(distutils.util.strtobool(x)),
            "default": "True",
            "help": "Choose whether to use warp or Isaac Gym rendeing pipeline.",
        },
        {
            "name": "--total_env_steps",
            "type": int,
            "required": False,
            "help": "Override total environment steps for off-policy configs.",
        },
    ]

    # parse arguments
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def update_config(config, args):

    if args["task"] is not None:
        config["params"]["config"]["env_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["params"]["config"]["name"] = args["experiment_name"]
    config["params"]["config"]["env_config"]["headless"] = args["headless"]
    config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
    config["params"]["config"]["env_config"]["use_warp"] = args["use_warp"]
    if args["num_envs"] > 0:
        config["params"]["config"]["num_actors"] = args["num_envs"]
        # config['params']['config']['num_envs'] = args['num_envs']
        config["params"]["config"]["env_config"]["num_envs"] = args["num_envs"]
        # When num_envs is small, batch_size = num_actors * horizon_length can be smaller
        # than minibatch_size; rl_games requires batch_size % minibatch_size == 0.
        horizon_length = config["params"]["config"].get("horizon_length", 32)
        batch_size = args["num_envs"] * horizon_length
        minibatch_size = config["params"]["config"].get("minibatch_size", 8192)
        if minibatch_size > batch_size or (batch_size % minibatch_size != 0):
            config["params"]["config"]["minibatch_size"] = batch_size
    if args["seed"] > 0:
        config["params"]["seed"] = args["seed"]
        config["params"]["config"]["env_config"]["seed"] = args["seed"]

    config["params"]["config"]["player"] = {"use_vecenv": True}
    return config


def update_offpolicy_config(config, args):
    if args["task"] is not None:
        config["env"]["task_name"] = args["task"]
    if args["experiment_name"] is not None:
        config["experiment"]["name"] = args["experiment_name"]
    if args["num_envs"] > 0:
        config["train"]["num_envs"] = args["num_envs"]
    config["train"]["headless"] = args["headless"]
    config["train"]["use_warp"] = args["use_warp"]
    if args.get("total_env_steps") is not None:
        config["train"]["total_env_steps"] = args["total_env_steps"]
    if args["seed"] > 0:
        config["experiment"]["seed"] = args["seed"]
    return config


if __name__ == "__main__":
    os.makedirs("nn", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    args = vars(get_args())

    config_name = args["file"]

    print("Loading config: ", config_name)
    with open(config_name, "r") as stream:
        config = yaml.safe_load(stream)
        if "params" in config:
            config = update_config(config, args)

            from rl_games.torch_runner import Runner

            _install_common_metric_tensorboard_patch()
            runner = Runner(algo_observer=CommonMetricObserver())
            try:
                runner.load(config)
            except yaml.YAMLError as exc:
                print(exc)

            _install_training_log_patch()

            rank = int(os.getenv("LOCAL_RANK", "0"))
            if args["track"] and rank == 0:
                import wandb

                wandb.init(
                    project=args["wandb_project_name"],
                    entity=args["wandb_entity"],
                    sync_tensorboard=True,
                    config=config,
                    monitor_gym=True,
                    save_code=True,
                )
            runner.run(args)

            if args["track"] and rank == 0:
                wandb.finish()
        else:
            from aerial_gym.rl_training.offpolicy.train import run_experiment

            config = update_offpolicy_config(config, args)
            run_experiment(
                config,
                checkpoint=args.get("checkpoint"),
                play=args.get("play", False),
            )
