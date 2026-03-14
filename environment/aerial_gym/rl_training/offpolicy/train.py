import argparse
import copy
import os
import random
import time
from collections import deque

import aerial_gym  # noqa: F401
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from aerial_gym.registry.task_registry import task_registry


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dir(path):
    os.makedirs(path, exist_ok=True)


def build_mlp(input_dim, hidden_sizes, output_dim, activation_name="relu", output_activation=None):
    activation_cls = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }[activation_name]
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_cls())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation == "tanh":
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def build_activation(activation_name="relu"):
    return {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
    }[activation_name]()


def relabel_observation(obs, achieved_goal, desired_goal):
    relabeled = obs.clone()
    relabeled[..., 0:3] = desired_goal - achieved_goal
    return relabeled


def build_goal_batch(task_obs):
    observation_key = "observation" if "observation" in task_obs else "observations"
    return {
        "observation": task_obs[observation_key].clone(),
        "achieved_goal": task_obs["achieved_goal"].clone(),
        "desired_goal": task_obs["desired_goal"].clone(),
    }


def build_state_features(batch):
    return torch.cat([batch["achieved_goal"], batch["observation"][:, 3:]], dim=-1)


class DeterministicActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, action_dim, activation, output_activation="tanh")

    def forward(self, obs):
        return self.net(obs)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, log_std_bounds):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, action_dim * 2, activation)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (log_std + 1.0)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True), torch.tanh(mu)


class TwinQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.q1 = build_mlp(obs_dim + action_dim, hidden_sizes, 1, activation)
        self.q2 = build_mlp(obs_dim + action_dim, hidden_sizes, 1, activation)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


class SingleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation):
        super().__init__()
        self.q = build_mlp(obs_dim + action_dim, hidden_sizes, 1, activation)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q(x)


class SingleQAuxCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, aux_hidden_dim=32):
        super().__init__()
        activation_cls = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }[activation]
        input_dim = obs_dim + action_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_cls())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.q_head = nn.Linear(prev_dim, 1)
        self.aux_head = nn.Sequential(
            nn.Linear(prev_dim, aux_hidden_dim),
            activation_cls(),
            nn.Linear(aux_hidden_dim, 1),
        )

    def forward(self, obs, action):
        latent = self.backbone(torch.cat([obs, action], dim=-1))
        return self.q_head(latent), self.aux_head(latent)

    def q_only(self, obs, action):
        latent = self.backbone(torch.cat([obs, action], dim=-1))
        return self.q_head(latent)


class TwinQAuxCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, aux_hidden_dim=32):
        super().__init__()
        activation_cls = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
        }[activation]
        input_dim = obs_dim + action_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_cls())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.q1_head = nn.Linear(prev_dim, 1)
        self.q2_head = nn.Linear(prev_dim, 1)
        self.aux_head = nn.Sequential(
            nn.Linear(prev_dim, aux_hidden_dim),
            activation_cls(),
            nn.Linear(aux_hidden_dim, 1),
        )

    def forward(self, obs, action):
        latent = self.backbone(torch.cat([obs, action], dim=-1))
        return self.q1_head(latent), self.q2_head(latent), self.aux_head(latent)

    def q1_only(self, obs, action):
        latent = self.backbone(torch.cat([obs, action], dim=-1))
        return self.q1_head(latent)


class ContrastiveCritic(nn.Module):
    def __init__(self, state_dim, action_dim, goal_dim, hidden_sizes, latent_dim, activation):
        super().__init__()
        self.phi = build_mlp(state_dim + action_dim, hidden_sizes, latent_dim, activation)
        self.psi = build_mlp(goal_dim, hidden_sizes, latent_dim, activation)

    def encode_sa(self, state, action):
        return F.normalize(self.phi(torch.cat([state, action], dim=-1)), dim=-1)

    def encode_goal(self, goal):
        return F.normalize(self.psi(goal), dim=-1)

    def q_value(self, state, action, goal):
        phi = self.encode_sa(state, action)
        psi = self.encode_goal(goal)
        return (phi * psi).sum(dim=-1, keepdim=True)


class FlatReplayBuffer:
    def __init__(self, obs_dim, goal_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.obs = torch.zeros((capacity, obs_dim), device=device)
        self.next_obs = torch.zeros((capacity, obs_dim), device=device)
        self.achieved_goal = torch.zeros((capacity, goal_dim), device=device)
        self.next_achieved_goal = torch.zeros((capacity, goal_dim), device=device)
        self.desired_goal = torch.zeros((capacity, goal_dim), device=device)
        self.action = torch.zeros((capacity, action_dim), device=device)
        self.reward = torch.zeros((capacity, 1), device=device)
        self.done = torch.zeros((capacity, 1), device=device)
        self.crash = torch.zeros((capacity, 1), device=device)

    def __len__(self):
        return self.size

    def add_batch(self, batch):
        batch_size = batch["obs"].shape[0]
        indices = (torch.arange(batch_size, device=self.device) + self.ptr) % self.capacity
        self.obs[indices] = batch["obs"]
        self.next_obs[indices] = batch["next_obs"]
        self.achieved_goal[indices] = batch["achieved_goal"]
        self.next_achieved_goal[indices] = batch["next_achieved_goal"]
        self.desired_goal[indices] = batch["desired_goal"]
        self.action[indices] = batch["action"]
        self.reward[indices] = batch["reward"]
        self.done[indices] = batch["done"]
        self.crash[indices] = batch["crash"]
        self.ptr = int((self.ptr + batch_size) % self.capacity)
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "obs": self.obs[indices],
            "next_obs": self.next_obs[indices],
            "achieved_goal": self.achieved_goal[indices],
            "next_achieved_goal": self.next_achieved_goal[indices],
            "desired_goal": self.desired_goal[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "done": self.done[indices],
            "crash": self.crash[indices],
        }


class EpisodeReplayManager:
    def __init__(
        self,
        num_envs,
        flat_buffer,
        her_cfg,
        recompute_reward_fn,
        reward_scale,
        device,
        max_completed_episodes=2048,
    ):
        self.num_envs = num_envs
        self.flat_buffer = flat_buffer
        self.her_cfg = her_cfg
        self.recompute_reward_fn = recompute_reward_fn
        self.reward_scale = reward_scale
        self.device = device
        self.current_episodes = [[] for _ in range(num_envs)]
        self.completed_episodes = deque(maxlen=max_completed_episodes)
        self.latest_metrics = deque(maxlen=256)
        self.total_completed_episodes = 0

    def add_step(self, obs_batch, actions, rewards, next_obs_batch, dones, infos):
        num_envs = obs_batch["observation"].shape[0]
        done_mask = dones.view(-1) > 0
        for env_idx in range(num_envs):
            transition = {
                "obs": obs_batch["observation"][env_idx].clone(),
                "achieved_goal": obs_batch["achieved_goal"][env_idx].clone(),
                "desired_goal": obs_batch["desired_goal"][env_idx].clone(),
                "action": actions[env_idx].clone(),
                "raw_reward": rewards[env_idx].clone().view(1),
                "reward": (rewards[env_idx].clone().view(1) * self.reward_scale),
                "next_obs": next_obs_batch["observation"][env_idx].clone(),
                "next_achieved_goal": next_obs_batch["achieved_goal"][env_idx].clone(),
                "done": dones[env_idx].clone().view(1),
                "crash": infos["crashes"][env_idx : env_idx + 1].clone(),
                "success": infos["success"][env_idx : env_idx + 1].clone(),
                "timeout": infos["timeouts"][env_idx : env_idx + 1].clone(),
            }
            self.current_episodes[env_idx].append(transition)
            if done_mask[env_idx]:
                self.flush_episode(env_idx)

    def flush_episode(self, env_idx):
        episode_transitions = self.current_episodes[env_idx]
        if not episode_transitions:
            return
        episode = {}
        for key in episode_transitions[0].keys():
            episode[key] = torch.stack([step[key] for step in episode_transitions], dim=0)
        self.flat_buffer.add_batch(
            {
                "obs": episode["obs"],
                "next_obs": episode["next_obs"],
                "achieved_goal": episode["achieved_goal"],
                "next_achieved_goal": episode["next_achieved_goal"],
                "desired_goal": episode["desired_goal"],
                "action": episode["action"],
                "reward": episode["reward"],
                "done": episode["done"],
                "crash": episode["crash"],
            }
        )
        if self.her_cfg["enabled"]:
            self._add_her_transitions(episode)
        self.completed_episodes.append(episode)
        self.total_completed_episodes += 1
        self.latest_metrics.append(
            {
                "episode_return_raw": float(episode["raw_reward"].sum().item()),
                "episode_return_shaped": float(episode["reward"].sum().item()),
                "episode_length": int(episode["reward"].shape[0]),
                "success": float(episode["success"].max().item()),
                "crash": float(episode["crash"].max().item()),
                "timeout": float(episode["timeout"].max().item()),
            }
        )
        self.current_episodes[env_idx] = []

    def _add_her_transitions(self, episode):
        strategy = self.her_cfg["strategy"]
        k = self.her_cfg["k"]
        length = episode["obs"].shape[0]
        relabeled_batches = []
        for t in range(length):
            for _ in range(k):
                if strategy == "future":
                    goal_idx = random.randint(t, length - 1)
                    new_goal = episode["next_achieved_goal"][goal_idx]
                elif strategy == "final":
                    new_goal = episode["next_achieved_goal"][-1]
                elif strategy == "episode":
                    goal_idx = random.randint(0, length - 1)
                    new_goal = episode["next_achieved_goal"][goal_idx]
                else:
                    raise ValueError(f"Unsupported HER strategy: {strategy}")

                relabeled_obs = relabel_observation(
                    episode["obs"][t : t + 1],
                    episode["achieved_goal"][t : t + 1],
                    new_goal.unsqueeze(0),
                )
                relabeled_next_obs = relabel_observation(
                    episode["next_obs"][t : t + 1],
                    episode["next_achieved_goal"][t : t + 1],
                    new_goal.unsqueeze(0),
                )
                relabeled_reward = self.recompute_reward_fn(
                    relabeled_next_obs,
                    episode["next_achieved_goal"][t : t + 1],
                    new_goal.unsqueeze(0),
                    episode["crash"][t : t + 1].view(-1),
                ).view(1, 1) * self.reward_scale
                relabeled_batches.append(
                    {
                        "obs": relabeled_obs,
                        "next_obs": relabeled_next_obs,
                        "achieved_goal": episode["achieved_goal"][t : t + 1],
                        "next_achieved_goal": episode["next_achieved_goal"][t : t + 1],
                        "desired_goal": new_goal.unsqueeze(0),
                        "action": episode["action"][t : t + 1],
                        "reward": relabeled_reward,
                        "done": episode["done"][t : t + 1],
                        "crash": episode["crash"][t : t + 1],
                    }
                )
        if relabeled_batches:
            merged = {}
            for key in relabeled_batches[0].keys():
                merged[key] = torch.cat([batch[key] for batch in relabeled_batches], dim=0)
            self.flat_buffer.add_batch(merged)

    def _sample_episode(self, min_length=1):
        valid = [episode for episode in self.completed_episodes if episode["obs"].shape[0] >= min_length]
        if not valid:
            return None
        return random.choice(valid)

    def sample_future_batch(self, batch_size):
        samples = []
        while len(samples) < batch_size:
            episode = self._sample_episode(min_length=1)
            if episode is None:
                break
            length = episode["obs"].shape[0]
            t = random.randint(0, length - 1)
            future_idx = random.randint(t, length - 1)
            samples.append((episode, t, future_idx))
        return samples

    def sample_stepwise_batch(self, batch_size):
        relabeled_batches = []
        while len(relabeled_batches) < batch_size:
            episode = self._sample_episode(min_length=1)
            if episode is None:
                break
            length = episode["obs"].shape[0]
            t = random.randint(0, length - 1)
            future_idx = random.randint(t, length - 1)
            new_goal = episode["next_achieved_goal"][future_idx]
            relabeled_obs = relabel_observation(
                episode["obs"][t : t + 1],
                episode["achieved_goal"][t : t + 1],
                new_goal.unsqueeze(0),
            )
            relabeled_next_obs = relabel_observation(
                episode["next_obs"][t : t + 1],
                episode["next_achieved_goal"][t : t + 1],
                new_goal.unsqueeze(0),
            )
            relabeled_reward = self.recompute_reward_fn(
                relabeled_next_obs,
                episode["next_achieved_goal"][t : t + 1],
                new_goal.unsqueeze(0),
                episode["crash"][t : t + 1].view(-1),
            ).view(1, 1) * self.reward_scale
            relabeled_batches.append(
                {
                    "obs": relabeled_obs,
                    "next_obs": relabeled_next_obs,
                    "action": episode["action"][t : t + 1],
                    "reward": relabeled_reward,
                    "done": torch.tensor(
                        [[1.0 if future_idx == t else 0.0]], device=self.device, dtype=torch.float32
                    ),
                    "horizon": torch.tensor(
                        [[float(future_idx - t + 1)]], device=self.device, dtype=torch.float32
                    ),
                    "goal": new_goal.unsqueeze(0),
                }
            )
        if not relabeled_batches:
            return {}
        merged = {}
        for key in relabeled_batches[0].keys():
            merged[key] = torch.cat([batch[key] for batch in relabeled_batches], dim=0)
        return merged

    def sample_tdm_batch(self, batch_size, max_horizon):
        samples = []
        while len(samples) < batch_size:
            episode = self._sample_episode(min_length=1)
            if episode is None:
                break
            length = episode["obs"].shape[0]
            t = random.randint(0, length - 1)
            horizon = random.randint(0, max_horizon)
            samples.append((episode, t, horizon))
        return samples


class TD3Agent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, cfg, device):
        self.device = device
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.policy_delay = cfg["policy_delay"]
        self.policy_noise = cfg["policy_noise"]
        self.noise_clip = cfg["noise_clip"]
        self.action_noise = cfg["exploration_noise"]
        self.update_step = 0

        self.actor = DeterministicActor(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TwinQCritic(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action = self.actor(obs)
            if not deterministic:
                action = action + self.action_noise * torch.randn_like(action)
            return action.clamp(-1.0, 1.0)

    def update(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(batch["action"]) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(batch["next_obs"]) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(batch["next_obs"], next_action)
            target_q = batch["reward"] + (1.0 - batch["done"]) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(batch["obs"], batch["action"])
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.zeros(1, device=self.device)
        if self.update_step % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(batch["obs"], self.actor(batch["obs"])).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.update_step += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, online, target):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_step = state["update_step"]


class DDPGAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, cfg, device):
        self.device = device
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.action_noise = cfg["exploration_noise"]
        self.actor = DeterministicActor(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = SingleQCritic(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action = self.actor(obs)
            if not deterministic:
                action = action + self.action_noise * torch.randn_like(action)
            return action.clamp(-1.0, 1.0)

    def update(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_action = self.actor_target(batch["next_obs"]).clamp(-1.0, 1.0)
            target_q = batch["reward"] + (1.0 - batch["done"]) * self.gamma * self.critic_target(
                batch["next_obs"], next_action
            )

        current_q = self.critic(batch["obs"], batch["action"])
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(batch["obs"], self.actor(batch["obs"])).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, online, target):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])


class SACAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, cfg, device):
        self.device = device
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.target_entropy = -float(action_dim)
        self.min_alpha = cfg.get("min_alpha", 1e-4)

        self.actor = GaussianActor(
            obs_dim, action_dim, hidden_sizes, activation, cfg["log_std_bounds"]
        ).to(device)
        self.critic = TwinQCritic(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.log_alpha = torch.tensor([cfg["init_alpha"]], device=device).log().requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg["alpha_lr"])

    @property
    def alpha(self):
        return self.log_alpha.exp().clamp(min=self.min_alpha)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            if deterministic:
                _, _, mean_action = self.actor.sample(obs)
                return mean_action
            action, _, _ = self.actor.sample(obs)
            return action.clamp(-1.0, 1.0)

    def update(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(batch["next_obs"])
            target_q1, target_q2 = self.critic_target(batch["next_obs"], next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_q = batch["reward"] + (1.0 - batch["done"]) * self.gamma * target_v

        current_q1, current_q2 = self.critic(batch["obs"], batch["action"])
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        action, log_prob, _ = self.actor.sample(batch["obs"])
        actor_q1, actor_q2 = self.critic(batch["obs"], action)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(actor_q1, actor_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic, self.critic_target)
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

    def _soft_update(self, online, target):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha = state["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_optimizer.param_groups[0]["lr"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])


class DDPGOursAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, ddpg_cfg, ours_cfg, device):
        self.device = device
        self.gamma = ddpg_cfg["gamma"]
        self.tau = ddpg_cfg["tau"]
        self.action_noise = ddpg_cfg["exploration_noise"]
        self.aux_weight = ours_cfg["aux_weight"]
        self.horizon_scale = ours_cfg["horizon_scale"]
        self.actor = DeterministicActor(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = SingleQAuxCritic(
            obs_dim, action_dim, hidden_sizes, activation, aux_hidden_dim=ours_cfg["aux_hidden_dim"]
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=ddpg_cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=ddpg_cfg["critic_lr"])

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action = self.actor(obs)
            if not deterministic:
                action = action + self.action_noise * torch.randn_like(action)
            return action.clamp(-1.0, 1.0)

    def update(self, episode_manager, batch_size):
        batch = episode_manager.sample_stepwise_batch(batch_size)
        if not batch:
            return {}
        with torch.no_grad():
            next_action = self.actor_target(batch["next_obs"]).clamp(-1.0, 1.0)
            target_q, _ = self.critic_target(batch["next_obs"], next_action)
            target_q = batch["reward"] + (1.0 - batch["done"]) * self.gamma * target_q

        current_q, aux_pred = self.critic(batch["obs"], batch["action"])
        q_loss = F.mse_loss(current_q, target_q)
        aux_target = batch["horizon"] / max(1.0, float(self.horizon_scale))
        aux_loss = F.mse_loss(aux_pred, aux_target)
        critic_loss = q_loss + self.aux_weight * aux_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic.q_only(batch["obs"], self.actor(batch["obs"])).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        return {
            "critic_loss": float(critic_loss.item()),
            "q_loss": float(q_loss.item()),
            "aux_loss": float(aux_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, online, target):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])


class TDMAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, cfg, device):
        self.device = device
        self.gamma = cfg["gamma"]
        self.tau = cfg["tau"]
        self.policy_delay = cfg["policy_delay"]
        self.exploration_noise = cfg["exploration_noise"]
        self.max_horizon = cfg["max_horizon"]
        self.policy_horizon = min(cfg.get("policy_horizon", self.max_horizon), self.max_horizon)
        augmented_obs_dim = obs_dim + 1
        self.actor = DeterministicActor(augmented_obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TwinQCritic(augmented_obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.update_step = 0

    def _augment(self, obs, tau_value):
        return torch.cat([obs, tau_value], dim=-1)

    def rollout_tau(self, batch_size):
        tau_value = self.policy_horizon / max(1, self.max_horizon)
        return torch.full((batch_size, 1), tau_value, device=self.device)

    def act(self, obs, tau_value, deterministic=False):
        augmented_obs = self._augment(obs, tau_value)
        with torch.no_grad():
            action = self.actor(augmented_obs)
            if not deterministic:
                action = action + self.exploration_noise * torch.randn_like(action)
            return action.clamp(-1.0, 1.0)

    def update(self, episode_manager, batch_size):
        samples = episode_manager.sample_tdm_batch(batch_size, self.max_horizon)
        if len(samples) < batch_size:
            return {}

        obs_batch = []
        action_batch = []
        tau_batch = []
        target_batch = []
        for episode, t, horizon in samples:
            goal = episode["desired_goal"][t : t + 1]
            obs = relabel_observation(
                episode["obs"][t : t + 1], episode["achieved_goal"][t : t + 1], goal
            )
            next_obs = relabel_observation(
                episode["next_obs"][t : t + 1],
                episode["next_achieved_goal"][t : t + 1],
                goal,
            )
            tau = torch.tensor([[horizon / max(1, self.max_horizon)]], device=self.device)
            next_tau_value = max(horizon - 1, 0)
            next_tau = torch.tensor([[next_tau_value / max(1, self.max_horizon)]], device=self.device)
            if horizon == 0:
                # TDM anchor: regress immediate post-action distance to the task goal.
                target = -torch.norm(
                    episode["next_achieved_goal"][t : t + 1] - goal, dim=-1, keepdim=True
                )
            else:
                with torch.no_grad():
                    next_action = self.actor_target(self._augment(next_obs, next_tau))
                    target_q1, target_q2 = self.critic_target(self._augment(next_obs, next_tau), next_action)
                    target = self.gamma * torch.min(target_q1, target_q2)
            obs_batch.append(obs)
            action_batch.append(episode["action"][t : t + 1])
            tau_batch.append(tau)
            target_batch.append(target)

        obs_tensor = torch.cat(obs_batch, dim=0)
        action_tensor = torch.cat(action_batch, dim=0)
        tau_tensor = torch.cat(tau_batch, dim=0)
        target_tensor = torch.cat(target_batch, dim=0)
        critic_input = self._augment(obs_tensor, tau_tensor)
        q1, q2 = self.critic(critic_input, action_tensor)
        critic_loss = F.mse_loss(q1, target_tensor) + F.mse_loss(q2, target_tensor)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.zeros(1, device=self.device)
        if self.update_step % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(critic_input, self.actor(critic_input)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.update_step += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, online, target):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_step = state["update_step"]


class TD3OursAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, td3_cfg, ours_cfg, device):
        self.device = device
        self.gamma = td3_cfg["gamma"]
        self.tau = td3_cfg["tau"]
        self.policy_delay = td3_cfg["policy_delay"]
        self.policy_noise = td3_cfg["policy_noise"]
        self.noise_clip = td3_cfg["noise_clip"]
        self.action_noise = td3_cfg["exploration_noise"]
        self.aux_weight = ours_cfg["aux_weight"]
        self.horizon_scale = ours_cfg["horizon_scale"]
        self.update_step = 0

        self.actor = DeterministicActor(obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = TwinQAuxCritic(
            obs_dim, action_dim, hidden_sizes, activation, aux_hidden_dim=ours_cfg["aux_hidden_dim"]
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=td3_cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=td3_cfg["critic_lr"])

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action = self.actor(obs)
            if not deterministic:
                action = action + self.action_noise * torch.randn_like(action)
            return action.clamp(-1.0, 1.0)

    def update(self, episode_manager, batch_size):
        batch = episode_manager.sample_stepwise_batch(batch_size)
        if not batch:
            return {}
        with torch.no_grad():
            noise = (torch.randn_like(batch["action"]) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(batch["next_obs"]) + noise).clamp(-1.0, 1.0)
            target_q1, target_q2, _ = self.critic_target(batch["next_obs"], next_action)
            target_q = batch["reward"] + (1.0 - batch["done"]) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2, aux_pred = self.critic(batch["obs"], batch["action"])
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        aux_target = batch["horizon"] / max(1.0, float(self.horizon_scale))
        aux_loss = F.mse_loss(aux_pred, aux_target)
        critic_loss = q_loss + self.aux_weight * aux_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.zeros(1, device=self.device)
        if self.update_step % self.policy_delay == 0:
            actor_loss = -self.critic.q1_only(batch["obs"], self.actor(batch["obs"])).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.update_step += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "q_loss": float(q_loss.item()),
            "aux_loss": float(aux_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, online, target):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "update_step": self.update_step,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.update_step = state["update_step"]


class CRLAgent:
    def __init__(self, goal_obs_dim, state_dim, action_dim, hidden_sizes, activation, cfg, device):
        self.device = device
        self.temperature = cfg["temperature"]
        self.exploration_noise = cfg["exploration_noise"]
        self.actor = DeterministicActor(goal_obs_dim, action_dim, hidden_sizes, activation).to(device)
        self.critic = ContrastiveCritic(
            state_dim,
            action_dim,
            cfg["goal_dim"],
            hidden_sizes,
            cfg["latent_dim"],
            activation,
        ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])

    def act(self, goal_obs, deterministic=False):
        with torch.no_grad():
            action = self.actor(goal_obs)
            if not deterministic:
                action = action + self.exploration_noise * torch.randn_like(action)
            return action.clamp(-1.0, 1.0)

    def update(self, episode_manager, batch_size):
        samples = episode_manager.sample_future_batch(batch_size)
        if len(samples) < batch_size:
            return {}

        state_batch = []
        goal_obs_batch = []
        action_batch = []
        goal_batch = []
        for episode, t, future_idx in samples:
            state_obs = {
                "observation": episode["obs"][t : t + 1],
                "achieved_goal": episode["achieved_goal"][t : t + 1],
                "desired_goal": episode["desired_goal"][t : t + 1],
            }
            state_batch.append(build_state_features(state_obs))
            goal_obs_batch.append(episode["obs"][t : t + 1])
            action_batch.append(episode["action"][t : t + 1])
            goal_batch.append(episode["next_achieved_goal"][future_idx : future_idx + 1])

        state_tensor = torch.cat(state_batch, dim=0)
        goal_obs_tensor = torch.cat(goal_obs_batch, dim=0)
        action_tensor = torch.cat(action_batch, dim=0)
        goal_tensor = torch.cat(goal_batch, dim=0)

        phi = self.critic.encode_sa(state_tensor, action_tensor)
        psi = self.critic.encode_goal(goal_tensor)
        logits = (phi @ psi.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=self.device)
        critic_loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_action = self.actor(goal_obs_tensor)
        actor_loss = -self.critic.q_value(state_tensor, actor_action, goal_tensor).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])


class SACOursAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes, activation, sac_cfg, ours_cfg, device):
        self.device = device
        self.gamma = sac_cfg["gamma"]
        self.tau = sac_cfg["tau"]
        self.target_entropy = -float(action_dim)
        self.min_alpha = sac_cfg.get("min_alpha", 1e-4)
        self.aux_weight = ours_cfg["aux_weight"]
        self.horizon_scale = ours_cfg["horizon_scale"]

        self.actor = GaussianActor(
            obs_dim, action_dim, hidden_sizes, activation, sac_cfg["log_std_bounds"]
        ).to(device)
        self.critic = TwinQAuxCritic(
            obs_dim, action_dim, hidden_sizes, activation, aux_hidden_dim=ours_cfg["aux_hidden_dim"]
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=sac_cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=sac_cfg["critic_lr"])
        self.log_alpha = torch.tensor([sac_cfg["init_alpha"]], device=device).log().requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=sac_cfg["alpha_lr"])

    @property
    def alpha(self):
        return self.log_alpha.exp().clamp(min=self.min_alpha)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            if deterministic:
                _, _, mean_action = self.actor.sample(obs)
                return mean_action
            action, _, _ = self.actor.sample(obs)
            return action.clamp(-1.0, 1.0)

    def update(self, episode_manager, batch_size):
        batch = episode_manager.sample_stepwise_batch(batch_size)
        if not batch:
            return {}
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(batch["next_obs"])
            target_q1, target_q2, _ = self.critic_target(batch["next_obs"], next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_q = batch["reward"] + (1.0 - batch["done"]) * self.gamma * target_v

        current_q1, current_q2, aux_pred = self.critic(batch["obs"], batch["action"])
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        aux_target = batch["horizon"] / max(1.0, float(self.horizon_scale))
        aux_loss = F.mse_loss(aux_pred, aux_target)
        critic_loss = q_loss + self.aux_weight * aux_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        action, log_prob, _ = self.actor.sample(batch["obs"])
        actor_q1, actor_q2, _ = self.critic(batch["obs"], action)
        actor_loss = (self.alpha.detach() * log_prob - torch.min(actor_q1, actor_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic, self.critic_target)
        return {
            "critic_loss": float(critic_loss.item()),
            "q_loss": float(q_loss.item()),
            "aux_loss": float(aux_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": float(self.alpha.item()),
            "alpha_loss": float(alpha_loss.item()),
        }

    def _soft_update(self, online, target):
        for online_param, target_param in zip(online.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(self.tau * online_param.data)

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().clone(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.log_alpha = state["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_optimizer.param_groups[0]["lr"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])


def make_agent(cfg, obs_dim, action_dim, device):
    hidden_sizes = cfg["network"]["hidden_sizes"]
    activation = cfg["network"]["activation"]
    goal_obs_dim = obs_dim
    state_dim = cfg["env"]["goal_dim"] + (obs_dim - cfg["env"]["goal_dim"])
    algo_name = cfg["algo"]["name"]
    if "ours" in cfg:
        cfg["ours"]["horizon_scale"] = cfg["env"]["episode_len_steps"]
    if algo_name == "td3":
        return TD3Agent(obs_dim, action_dim, hidden_sizes, activation, cfg["td3"], device)
    if algo_name == "ddpg":
        return DDPGAgent(obs_dim, action_dim, hidden_sizes, activation, cfg["ddpg"], device)
    if algo_name == "sac":
        return SACAgent(obs_dim, action_dim, hidden_sizes, activation, cfg["sac"], device)
    if algo_name == "tdm":
        return TDMAgent(obs_dim, action_dim, hidden_sizes, activation, cfg["tdm"], device)
    if algo_name == "crl":
        crl_cfg = cfg["crl"]
        crl_cfg["goal_dim"] = cfg["env"]["goal_dim"]
        return CRLAgent(goal_obs_dim, state_dim, action_dim, hidden_sizes, activation, crl_cfg, device)
    if algo_name == "ddpg_ours":
        return DDPGOursAgent(obs_dim, action_dim, hidden_sizes, activation, cfg["ddpg"], cfg["ours"], device)
    if algo_name == "td3_ours":
        return TD3OursAgent(obs_dim, action_dim, hidden_sizes, activation, cfg["td3"], cfg["ours"], device)
    if algo_name == "sac_ours":
        return SACOursAgent(obs_dim, action_dim, hidden_sizes, activation, cfg["sac"], cfg["ours"], device)
    raise ValueError(f"Unsupported algorithm: {algo_name}")


def extract_next_batch(task_obs, infos, dones):
    next_batch = build_goal_batch(task_obs)
    done_mask = dones.view(-1) > 0
    if done_mask.any() and "terminal_observation" in infos:
        next_batch["observation"][done_mask] = infos["terminal_observation"][done_mask]
        next_batch["achieved_goal"][done_mask] = infos["terminal_achieved_goal"][done_mask]
        next_batch["desired_goal"][done_mask] = infos["terminal_desired_goal"][done_mask]
    return next_batch


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def merge_args(cfg, args):
    if args.experiment_name:
        cfg["experiment"]["name"] = args.experiment_name
    if args.task:
        cfg["env"]["task_name"] = args.task
    if args.num_envs is not None:
        cfg["train"]["num_envs"] = args.num_envs
    if args.headless is not None:
        cfg["train"]["headless"] = args.headless
    if args.use_warp is not None:
        cfg["train"]["use_warp"] = args.use_warp
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.total_env_steps is not None:
        cfg["train"]["total_env_steps"] = args.total_env_steps
    return cfg


def build_env(cfg):
    task_name = cfg["env"]["task_name"]
    task_class = task_registry.get_task_class(task_name)
    task_config = task_registry.get_task_config(task_name)
    task_config.device = cfg["train"]["device"]
    task_config.env_name = cfg["env"]["env_name"]
    task_config.robot_name = cfg["env"]["robot_name"]
    task_config.controller_name = cfg["env"]["controller_name"]
    if "observation_dim" in cfg["env"]:
        task_config.observation_space_dim = cfg["env"]["observation_dim"]
    task_config.action_space_dim = cfg["env"]["action_dim"]
    task_config.episode_len_steps = cfg["env"]["episode_len_steps"]
    if hasattr(task_config, "use_depth_obs") and "use_depth_obs" in cfg["env"]:
        task_config.use_depth_obs = cfg["env"]["use_depth_obs"]
    if hasattr(task_config, "depth_cnn_latent_dim") and "depth_cnn_latent_dim" in cfg["env"]:
        task_config.depth_cnn_latent_dim = cfg["env"]["depth_cnn_latent_dim"]
    if hasattr(task_config, "success_threshold"):
        task_config.success_threshold = cfg["env"]["success_threshold"]
        task_config.success_reward = cfg["env"]["success_reward"]
        task_config.collision_penalty = cfg["env"]["collision_penalty"]
    return task_class(
        task_config,
        seed=cfg["experiment"]["seed"],
        num_envs=cfg["train"]["num_envs"],
        headless=cfg["train"]["headless"],
        device=cfg["train"]["device"],
        use_warp=cfg["train"]["use_warp"],
    )


def save_checkpoint(agent, cfg, step, output_dir):
    checkpoint_path = os.path.join(output_dir, f"{cfg['experiment']['name']}_step_{step}.pt")
    torch.save({"agent": agent.state_dict(), "config": cfg, "step": step}, checkpoint_path)
    return checkpoint_path


def progress_index(current_step, total_steps):
    progress = min(1.0, max(0.0, float(current_step) / max(1, total_steps)))
    return int(round(progress * 1000.0))


def log_eval_metrics(writer, metrics, total_env_steps, elapsed_time, total_budget):
    scalar_time = int(elapsed_time)
    progress_step = progress_index(total_env_steps, total_budget)

    writer.add_scalar("eval/return_raw/step", metrics["mean_return_raw"], total_env_steps)
    writer.add_scalar("eval/return_raw/time", metrics["mean_return_raw"], scalar_time)
    writer.add_scalar("eval/return_raw/progress", metrics["mean_return_raw"], progress_step)

    writer.add_scalar("eval/return_scaled/step", metrics["mean_return_shaped"], total_env_steps)
    writer.add_scalar("eval/return_scaled/time", metrics["mean_return_shaped"], scalar_time)
    writer.add_scalar("eval/return_scaled/progress", metrics["mean_return_shaped"], progress_step)

    writer.add_scalar("eval/episode_length/step", metrics["mean_length"], total_env_steps)
    writer.add_scalar("eval/episode_length/time", metrics["mean_length"], scalar_time)
    writer.add_scalar("eval/episode_length/progress", metrics["mean_length"], progress_step)

    writer.add_scalar("eval/success_rate/step", metrics["success_rate"], total_env_steps)
    writer.add_scalar("eval/success_rate/time", metrics["success_rate"], scalar_time)
    writer.add_scalar("eval/success_rate/progress", metrics["success_rate"], progress_step)

    writer.add_scalar("eval/crash_rate/step", metrics["crash_rate"], total_env_steps)
    writer.add_scalar("eval/crash_rate/time", metrics["crash_rate"], scalar_time)
    writer.add_scalar("eval/crash_rate/progress", metrics["crash_rate"], progress_step)

    writer.add_scalar("eval/timeout_rate/step", metrics["timeout_rate"], total_env_steps)
    writer.add_scalar("eval/timeout_rate/time", metrics["timeout_rate"], scalar_time)
    writer.add_scalar("eval/timeout_rate/progress", metrics["timeout_rate"], progress_step)
    writer.flush()


def log_training_losses(writer, losses, total_env_steps, elapsed_time, total_budget):
    if not losses:
        return
    scalar_time = int(elapsed_time)
    progress_step = progress_index(total_env_steps, total_budget)
    for key, value in losses.items():
        writer.add_scalar(f"train/{key}/step", value, total_env_steps)
        writer.add_scalar(f"train/{key}/time", value, scalar_time)
        writer.add_scalar(f"train/{key}/progress", value, progress_step)


def summarize_latest_metrics(episode_manager):
    if not episode_manager.latest_metrics:
        return None

    mean_return_raw = sum(m["episode_return_raw"] for m in episode_manager.latest_metrics) / len(
        episode_manager.latest_metrics
    )
    mean_return_shaped = sum(m["episode_return_shaped"] for m in episode_manager.latest_metrics) / len(
        episode_manager.latest_metrics
    )
    mean_length = sum(m["episode_length"] for m in episode_manager.latest_metrics) / len(
        episode_manager.latest_metrics
    )
    success_rate = sum(m["success"] for m in episode_manager.latest_metrics) / len(
        episode_manager.latest_metrics
    )
    crash_rate = sum(m["crash"] for m in episode_manager.latest_metrics) / len(
        episode_manager.latest_metrics
    )
    timeout_rate = sum(m["timeout"] for m in episode_manager.latest_metrics) / len(
        episode_manager.latest_metrics
    )

    return {
        "mean_return_raw": mean_return_raw,
        "mean_return_shaped": mean_return_shaped,
        "mean_length": mean_length,
        "success_rate": success_rate,
        "crash_rate": crash_rate,
        "timeout_rate": timeout_rate,
    }


def run_experiment(cfg, checkpoint=None, play=False):
    seed = cfg["experiment"]["seed"]
    set_seed(seed)
    device = torch.device(cfg["train"]["device"])

    env = build_env(cfg)
    cfg["env"]["observation_dim"] = int(env.task_config.observation_space_dim)
    action_dim = int(env.task_config.action_space_dim)
    agent = make_agent(cfg, cfg["env"]["observation_dim"], action_dim, device)

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        agent.load_state_dict(state["agent"])

    if play:
        obs_dict, _, _, _, _ = env.reset()
        obs_batch = build_goal_batch(obs_dict)
        while True:
            if cfg["algo"]["name"] == "tdm":
                tau_value = agent.rollout_tau(cfg["train"]["num_envs"])
                actions = agent.act(obs_batch["observation"], tau_value, deterministic=True)
            else:
                actions = agent.act(obs_batch["observation"], deterministic=True)
            next_obs_dict, rewards, terminated, truncated, infos = env.step(actions)
            dones = torch.logical_or(terminated > 0, truncated > 0).float().unsqueeze(-1)
            obs_batch = build_goal_batch(next_obs_dict)
            _ = rewards, infos

    reward_scale = float(cfg["train"].get("reward_scale", 1.0))
    replay_buffer = FlatReplayBuffer(
        obs_dim=cfg["env"]["observation_dim"],
        goal_dim=cfg["env"]["goal_dim"],
        action_dim=cfg["env"]["action_dim"],
        capacity=cfg["train"]["replay_buffer_size"],
        device=device,
    )
    episode_manager = EpisodeReplayManager(
        num_envs=cfg["train"]["num_envs"],
        flat_buffer=replay_buffer,
        her_cfg=cfg["her"],
        recompute_reward_fn=env.recompute_reward,
        reward_scale=reward_scale,
        device=device,
        max_completed_episodes=cfg["train"]["max_completed_episodes"],
    )

    output_dir = os.path.join(
        "/home/user/aerial_gym_simulator/environment/runs/offpolicy", cfg["experiment"]["name"]
    )
    make_dir(output_dir)
    writer = SummaryWriter(output_dir)

    task_obs, _, _, _, _ = env.reset()
    obs_batch = build_goal_batch(task_obs)
    total_env_steps = 0
    start_time = time.time()
    log_every_env_steps = int(cfg["train"].get("log_every_env_steps", 8192))
    last_log_env_steps = 0
    last_checkpoint_step = 0

    while total_env_steps < cfg["train"]["total_env_steps"]:
        if total_env_steps < cfg["train"]["warmup_env_steps"]:
            actions = torch.rand((cfg["train"]["num_envs"], action_dim), device=device) * 2.0 - 1.0
        elif cfg["algo"]["name"] == "tdm":
            tau_value = agent.rollout_tau(cfg["train"]["num_envs"])
            actions = agent.act(obs_batch["observation"], tau_value, deterministic=False)
        else:
            actions = agent.act(obs_batch["observation"], deterministic=False)

        next_task_obs, rewards, terminated, truncated, infos = env.step(actions)
        dones = torch.logical_or(terminated > 0, truncated > 0).float().unsqueeze(-1)
        next_batch_for_buffer = extract_next_batch(next_task_obs, infos, dones)

        episode_manager.add_step(obs_batch, actions, rewards.unsqueeze(-1), next_batch_for_buffer, dones, infos)
        obs_batch = build_goal_batch(next_task_obs)
        total_env_steps += cfg["train"]["num_envs"]

        losses = {}
        if len(replay_buffer) >= cfg["train"]["batch_size"]:
            updates_per_step = cfg["train"]["gradient_steps_per_iter"]
            for _ in range(updates_per_step):
                losses = agent.update(
                    episode_manager if cfg["algo"]["name"] in {"tdm", "crl", "ddpg_ours", "td3_ours", "sac_ours"} else replay_buffer,
                    cfg["train"]["batch_size"],
                )
        if total_env_steps - last_log_env_steps >= log_every_env_steps:
            metrics = summarize_latest_metrics(episode_manager)
            if metrics is not None:
                elapsed_time = time.time() - start_time
                log_eval_metrics(
                    writer,
                    metrics,
                    total_env_steps,
                    elapsed_time,
                    cfg["train"]["total_env_steps"],
                )
                print(
                    f"step={total_env_steps} mean_reward={metrics['mean_return_shaped']:.3f} "
                    f"raw_reward={metrics['mean_return_raw']:.3f} mean_ep_len={metrics['mean_length']:.1f} "
                    f"success_rate={metrics['success_rate']:.3f} crash_rate={metrics['crash_rate']:.3f}"
                )
            if losses:
                elapsed_time = time.time() - start_time
                log_training_losses(
                    writer,
                    losses,
                    total_env_steps,
                    elapsed_time,
                    cfg["train"]["total_env_steps"],
                )
            last_log_env_steps = total_env_steps

        if total_env_steps - last_checkpoint_step >= cfg["train"]["save_every_env_steps"]:
            checkpoint_path = save_checkpoint(agent, cfg, total_env_steps, output_dir)
            print(f"saved checkpoint: {checkpoint_path}")
            last_checkpoint_step = total_env_steps

    if last_log_env_steps != total_env_steps:
        metrics = summarize_latest_metrics(episode_manager)
        if metrics is not None:
            elapsed_time = time.time() - start_time
            log_eval_metrics(
                writer,
                metrics,
                total_env_steps,
                elapsed_time,
                cfg["train"]["total_env_steps"],
            )

    final_checkpoint = save_checkpoint(agent, cfg, total_env_steps, output_dir)
    print(f"training finished, final checkpoint: {final_checkpoint}")
    writer.flush()
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Goal-conditioned off-policy baselines for Aerial Gym.")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/user/aerial_gym_simulator/environment/aerial_gym/rl_training/offpolicy/configs/td3_gc.yaml",
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--headless", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--use_warp", type=lambda x: x.lower() == "true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total_env_steps", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--play", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = merge_args(load_config(args.config), args)
    run_experiment(cfg, checkpoint=args.checkpoint, play=args.play)


if __name__ == "__main__":
    main()
