"""Camera-based CNN PPO chooses every joint target increment and gripper action.

The environment integrates tanh(raw_action) * joint_increment_scale into its
joint targets. The actor and critic receive RGB camera frame stacks and joint
proprioception. Exact object poses and contact/reward flags stay in the
environment for physics and scoring. There is no nominal movement path, scheduled grasp sequence,
or learned correction around a separate task controller in this module.
Source authored for a Colab notebook; it has not been executed or validated.
"""
from __future__ import annotations

import copy
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import threading
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from environment import (TASK_ID, REWARD_GAMMA, OBSERVATION_DIM, ACTION_DIM,
                         IMAGE_SHAPE, OBSERVATION_SPEC)


DEFAULT_CONFIG = {
    "seed": 7,
    "num_envs": 2,
    "horizon": 64,
    "updates": 1000,
    "epochs": 3,
    "minibatch_size": 128,
    "learning_rate": 3e-4,
    "gamma": REWARD_GAMMA,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
    "eval_every": 100,
    "eval_seed": 2026,
    "eval_episodes": 1,
    "evaluate_before_training": True,
    "evaluate_on_stop": False,
    "anneal_lr": False,
    "save_every": 10,
}

ATTRIBUTION = {
    "method": "direct joint-target-increment PPO",
    "action_semantics": "six raw Normal actions mapped by environment tanh to joint-target increments",
    "movement_source": "all joint and gripper movement commands are selected by the policy",
    "policy_observations": "RGB camera frame stacks plus joint proprioception; no privileged state in actor or critic",
    "policy_type": "cnn_rgb_proprio_ppo_v1",
    "observation_spec": OBSERVATION_SPEC,
    "episode_metrics": "success refers to one physically grasped, lifted and released ball resting on the flat plate",
    "task_id": TASK_ID,
    "algorithm_reference": "https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py",
    "execution_status": "source authored without execution or validation",
}


POLICY_TYPE = "cnn_rgb_proprio_ppo_v1"
POLICY_ARCHITECTURE = {
    "encoder": "Conv16(k8,s4)-Conv32(k4,s2)-Conv32(k3,s1)-FC128",
    "fusion": "RGB features plus joint proprioception -> FC64",
    "actor": "FC64-Tanh-FC6 Normal mean",
    "critic": "FC64-Tanh-FC1",
    "pixel_normalization": "uint8 to float32 divided by 255 inside the encoder",
    "privileged_critic": False,
}


def batch_observation(observation, device="cpu"):
    """Add a batch dimension to one CPU environment observation for inference."""
    if not isinstance(observation, dict) or set(observation) != {"image", "proprio"}:
        raise ValueError("Expected an observation dictionary with image and proprio.")
    image = torch.as_tensor(observation["image"], device=device)
    proprio = torch.as_tensor(observation["proprio"], dtype=torch.float32, device=device)
    if image.dtype != torch.uint8 or tuple(image.shape) != tuple(IMAGE_SHAPE):
        raise ValueError("The single-environment camera pixel shape or dtype does not match.")
    if tuple(proprio.shape) != (OBSERVATION_DIM,):
        raise ValueError("The single-environment proprioception shape does not match.")
    _require_finite("proprioception", proprio)
    return {"image": image.unsqueeze(0), "proprio": proprio.unsqueeze(0)}


def _select_observation(observation, index):
    return {key: value[index] for key, value in observation.items()}


class ActorCritic(nn.Module):
    """Image-based PPO actor and critic with joint proprioception.

    Both heads consume the same image encoder. Neither head receives exact
    ball, plate, end-effector positions or simulator contact/reward flags.
    """

    def __init__(self, observation_dim=OBSERVATION_DIM, action_dim=ACTION_DIM,
                 hidden_dim=64, image_shape=IMAGE_SHAPE):
        super().__init__()
        self.observation_dim = self.obs_dim = int(observation_dim)
        self.action_dim, self.hidden_dim = int(action_dim), int(hidden_dim)
        self.image_shape = tuple(int(value) for value in image_shape)
        if self.observation_dim != OBSERVATION_DIM or self.image_shape != tuple(IMAGE_SHAPE):
            raise ValueError("The vision policy must match the environment observation specification.")
        channels, height, width = self.image_shape
        if (height, width) != (64, 64):
            raise ValueError("This compact CNN expects 64 by 64 camera frames.")
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 4 * 4, 128), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + self.observation_dim, self.hidden_dim), nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.3))
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.)

    def _features(self, observation):
        image = observation["image"]
        proprio = observation["proprio"]
        if image.dtype != torch.uint8:
            raise TypeError("Camera observations must be uint8 pixels, not simulator state.")
        pixels = image.to(dtype=torch.float32).div(255.)
        visual = self.encoder(pixels)
        return self.fusion(torch.cat((visual, proprio), dim=-1))

    def forward(self, observation):
        return self.mean_action(observation)

    def act(self, observation, action=None, deterministic=False):
        features = self._features(observation)
        mean = self.actor(features)
        distribution = Normal(mean, self.log_std.clamp(-4., 1.).exp().expand_as(mean))
        if action is None:
            action = mean if deterministic else distribution.sample()
        # Store the exact raw Normal action; the environment applies tanh.
        return (action, distribution.log_prob(action).sum(-1),
                distribution.entropy().sum(-1), self.critic(features).squeeze(-1))

    def mean_action(self, observation):
        return self.actor(self._features(observation))

    deterministic_action = mean_action

    def value(self, observation):
        return self.critic(self._features(observation)).squeeze(-1)


def compute_gae(rewards, dones, values, next_value, gamma=.995, gae_lambda=.95):
    """Time-limit bootstrap is already included in the supplied rewards."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(next_value)
    for step in reversed(range(rewards.shape[0])):
        following = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        alive = (~dones[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * following * alive - values[step]
        gae = delta + gamma * gae_lambda * alive * gae
        advantages[step] = gae
    return advantages, advantages + values


def _require_finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or infinity.")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("A metric contains NaN or infinity.")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _versions():
    result = {"python": platform.python_version()}
    for name in ("torch", "numpy", "mujoco", "mujoco-warp", "warp-lang"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result


class PPORunner:
    """Background training with frozen CPU snapshots for a separate viewer.

    make_env() runs inside the learner thread so the MJWarp environment can
    adopt that thread's current CUDA stream. Its step() returns observations,
    reward, terminated, truncated and info. info contains a whole-batch
    final_obs dictionary before autoreset plus completed-only episode_returns,
    episode_success and optional episode_filled_count arrays.

    evaluator(cpu_policy, mode='learned'|'random', seed=...) is optional.
    It compares the initial policy and subsequent policies under a fixed seed.
    A loaded checkpoint warm-starts policy/optimizer; simulator state, partial
    rollouts, environment RNG and global RNG are restarted.
    """

    def __init__(self, config=None, *, make_env, evaluator=None,
                 output_dir="runs/so101_direct_rl", device="cuda:0",
                 observation_dim=None, action_dim=6, metadata=None):
        self.config = dict(DEFAULT_CONFIG)
        self.config.update(dict(config or {}))
        self._validate_config()
        self.make_env, self.evaluator = make_env, evaluator
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = _json_safe({**dict(metadata or {}), "task_id": TASK_ID,
                                    "policy_type": POLICY_TYPE,
                                    "observation_spec": OBSERVATION_SPEC,
                                    "architecture": POLICY_ARCHITECTURE})
        self.versions = _versions()
        self.observation_dim = observation_dim or self.config.get("observation_dim", OBSERVATION_DIM)
        self.action_dim = int(action_dim)
        self.lock = threading.RLock()
        self._save_lock = threading.Lock()
        self.stop_event, self.ready_event = threading.Event(), threading.Event()
        self.thread = self.policy = self.optimizer = None
        self.initial_weights = self.latest_weights = None
        self.status, self.error = "준비됨", None
        self.history, self.evaluations = [], []
        self.baseline = self.final_evaluation = None
        self.snapshot_version = self.initial_snapshot_version = 0
        self.completed_updates = self.total_steps = 0
        self.run_start_updates = self.run_start_steps = 0
        self._training = False
        self._pending_optimizer = None
        self.warm_start = None
        if self.observation_dim is not None:
            self._initialize_policy(int(self.observation_dim), self.action_dim)

    def _validate_config(self):
        for key in ("num_envs", "horizon", "updates", "epochs", "minibatch_size", "eval_every", "save_every"):
            value = self.config[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        for key in ("learning_rate", "max_grad_norm"):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f"{key} must be positive and finite.")
        for key in ("gamma", "gae_lambda"):
            if not 0 <= self.config[key] <= 1:
                raise ValueError(f"{key} must lie in [0, 1].")
        if not math.isclose(self.config["gamma"], REWARD_GAMMA, rel_tol=0., abs_tol=1e-12):
            raise ValueError(f"gamma must match the reward potential discount ({REWARD_GAMMA}).")

    def _initialize_policy(self, observation_dim, action_dim):
        if (int(observation_dim), int(action_dim)) != (OBSERVATION_DIM, ACTION_DIM):
            raise ValueError("Vision policy dimensions must match the current environment.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config["seed"])
            self.policy = ActorCritic(observation_dim, action_dim)
        self.observation_dim, self.action_dim = int(observation_dim), int(action_dim)
        self.initial_weights = self._cpu_weights()
        self.latest_weights = copy.deepcopy(self.initial_weights)
        self.ready_event.set()

    def _cpu_weights(self):
        if self.policy is None:
            raise RuntimeError("Pass observation_dim or wait for the environment to initialize.")
        return {key: value.detach().cpu().clone() for key, value in self.policy.state_dict().items()}

    def snapshot(self, initial=False):
        with self.lock:
            weights = self.initial_weights if initial else self.latest_weights
            if weights is None:
                raise RuntimeError("The policy is not ready.")
            version = self.initial_snapshot_version if initial else self.snapshot_version
            return copy.deepcopy(weights), version

    def load_checkpoint(self, path, restore_optimizer=True):
        """Warm-start a compatible camera policy; scalar-state policies cannot be reused."""
        if self._training or self.thread is not None:
            raise RuntimeError("Load a checkpoint before start().")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 4 or payload.get("policy_type") != POLICY_TYPE:
            raise ValueError(
                "This notebook requires a format-4 RGB-camera CNN checkpoint. "
                "Older numeric-state or six-ball checkpoints are incompatible; start a new policy."
            )
        if payload.get("task_id") != TASK_ID:
            raise ValueError("Checkpoint task does not match the camera-based one-ball flat-plate task.")
        if _json_safe(payload.get("observation_spec")) != _json_safe(OBSERVATION_SPEC):
            raise ValueError("Checkpoint camera names, frame stack, image timing or proprioception differ.")
        if payload.get("architecture") != POLICY_ARCHITECTURE:
            raise ValueError("Checkpoint vision architecture does not match this policy.")
        if payload.get("action_semantics") != ATTRIBUTION["action_semantics"]:
            raise ValueError("This is not a direct joint-increment checkpoint from this notebook.")
        obs_dim, act_dim = int(payload["observation_dim"]), int(payload["action_dim"])
        if (obs_dim, act_dim) != (OBSERVATION_DIM, ACTION_DIM):
            raise ValueError("Checkpoint dimensions do not match the current vision environment.")
        if self.observation_dim is not None and (obs_dim, act_dim) != (self.observation_dim, self.action_dim):
            raise ValueError("Checkpoint and environment dimensions differ.")
        if self.policy is None:
            self._initialize_policy(obs_dim, act_dim)
        self.policy.load_state_dict(payload["policy"])
        with self.lock:
            self.initial_weights = self._cpu_weights()
            self.latest_weights = copy.deepcopy(self.initial_weights)
            self.completed_updates = int(payload.get("training_updates", 0))
            self.total_steps = int(payload.get("steps", 0))
            self.snapshot_version = int(payload.get("snapshot_version", self.completed_updates))
            self.initial_snapshot_version = self.snapshot_version
            self._pending_optimizer = payload.get("optimizer") if restore_optimizer else None
            self.warm_start = {"path": str(path), "optimizer_requested": bool(restore_optimizer),
                               "previous_updates": self.completed_updates, "previous_steps": self.total_steps,
                               "scope": "vision policy and optional optimizer; simulator, rollout and RNG restart"}
            self.status = "저장된 정책 불러옴"
        return self

    def _publish(self):
        weights = self._cpu_weights()
        for name, value in weights.items():
            _require_finite(f"policy weight {name}", value)
        with self.lock:
            self.latest_weights = weights
            self.snapshot_version += 1

    def start(self):
        with self.lock:
            if self.thread is not None or self._training:
                raise RuntimeError("Create a new runner for another training run.")
            self.thread = threading.Thread(target=self._run_guarded, name="SO101-direct-PPO", daemon=True)
            self.thread.start()
        return self

    def stop(self, wait=False, timeout=10.):
        self.stop_event.set()
        if wait:
            self.join(timeout)

    def join(self, timeout=None):
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout)
        return self.status

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def _run_guarded(self):
        try:
            self.train()
        except Exception:
            with self.lock:
                self.error, self.status = traceback.format_exc(), "오류"
            self.ready_event.set()
            (self.output_dir / "error.txt").write_text(self.error, encoding="utf-8")
            print(self.error)

    def _observation(self, value, count):
        if not isinstance(value, dict) or set(value) != {"image", "proprio"}:
            raise ValueError("Vision observations must contain exactly image and proprio.")
        image = torch.as_tensor(value["image"], device=self.device)
        proprio = torch.as_tensor(value["proprio"], dtype=torch.float32, device=self.device)
        if image.dtype != torch.uint8:
            raise TypeError("Camera pixels must be uint8 to preserve the rollout memory budget.")
        if tuple(image.shape) != (count, *IMAGE_SHAPE):
            raise ValueError("The batched camera frame stack shape does not match the policy.")
        if tuple(proprio.shape) != (count, self.observation_dim):
            raise ValueError("The batched proprioception shape does not match the policy.")
        _require_finite("proprioception", proprio)
        return {"image": image, "proprio": proprio}

    def _evaluate(self, update, initial=False):
        if self.evaluator is None:
            return None
        old_status, self.status = self.status, "평가 중"
        weights, version = self.snapshot()
        cpu_policy = ActorCritic(self.observation_dim, self.action_dim).eval()
        cpu_policy.load_state_dict(weights)
        results = {}
        with torch.inference_mode():
            for mode in ("learned", "random"):
                results[mode] = _json_safe(self.evaluator(cpu_policy, mode=mode, seed=self.config["eval_seed"]))
        record = {"update": update, "steps": self.total_steps, "snapshot_version": version,
                  "seed": self.config["eval_seed"], "policy_kind": "before this run" if initial else "PPO policy",
                  "results": results}
        with self.lock:
            self.evaluations.append(record)
            self.final_evaluation = dict(results, snapshot_version=version, update=update, seed=self.config["eval_seed"])
            if initial:
                self.baseline = copy.deepcopy(results)
        self.status = old_status
        return record

    def train(self):
        with self.lock:
            if self._training:
                raise RuntimeError("Concurrent training is not supported.")
            self._training = True
        env = None
        try:
            cfg = self.config
            torch.manual_seed(cfg["seed"])
            random.seed(cfg["seed"])
            np.random.seed(cfg["seed"])
            self.status = "환경 준비 중"
            env = self.make_env()
            nenv = int(env.num_envs)
            if nenv != cfg["num_envs"]:
                raise ValueError("Environment count and training configuration differ.")
            if self.policy is None:
                self._initialize_policy(int(env.observation_dim), int(env.action_dim))
            elif (int(env.observation_dim), int(env.action_dim)) != (self.observation_dim, self.action_dim):
                raise ValueError("Environment and policy dimensions differ.")
            self.policy.to(self.device)
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg["learning_rate"], eps=1e-5)
            if self._pending_optimizer is not None:
                self.optimizer.load_state_dict(self._pending_optimizer)
                for state in self.optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(self.device)
                for group in self.optimizer.param_groups:
                    group["lr"] = cfg["learning_rate"]
                self._pending_optimizer = None
            self.run_start_updates, self.run_start_steps = self.completed_updates, self.total_steps
            if cfg["evaluate_before_training"]:
                self._evaluate(self.completed_updates, initial=True)
            self.save("before_training.pt", initial=True)
            horizon, batch_size = cfg["horizon"], cfg["horizon"] * nenv
            obs_buf = {
                "image": torch.empty((horizon, nenv, *IMAGE_SHAPE), dtype=torch.uint8, device=self.device),
                "proprio": torch.empty((horizon, nenv, self.observation_dim), device=self.device),
            }
            action_buf = torch.empty((horizon, nenv, self.action_dim), device=self.device)
            log_buf = torch.empty((horizon, nenv), device=self.device)
            rewards, values = torch.empty_like(log_buf), torch.empty_like(log_buf)
            dones = torch.empty((horizon, nenv), dtype=torch.bool, device=self.device)
            obs = self._observation(env.observation(), nenv)
            recent_returns, recent_success, recent_filled = [], [], []
            recent_lifted, recent_released = [], []
            began = time.perf_counter()
            self.status = "학습 중"
            for local_update in range(1, cfg["updates"] + 1):
                if self.stop_event.is_set():
                    break
                partial = False
                for step in range(horizon):
                    if self.stop_event.is_set():
                        partial = True
                        break
                    for key in obs_buf:
                        obs_buf[key][step].copy_(obs[key])
                    with torch.no_grad():
                        action, log_prob, _, value = self.policy.act(obs)
                        _require_finite("sampled action", action)
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        obs = self._observation(next_obs, nenv)
                        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
                        truncated = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)
                        _require_finite("reward", reward)
                        timeout = truncated & ~terminated
                        if bool(timeout.any()):
                            final_obs = self._observation(info["final_obs"], nenv)
                            reward = reward.clone()
                            reward[timeout] += cfg["gamma"] * self.policy.value(_select_observation(final_obs, timeout))
                    action_buf[step].copy_(action)
                    log_buf[step].copy_(log_prob)
                    values[step].copy_(value)
                    rewards[step].copy_(reward)
                    dones[step].copy_(terminated | truncated)
                    done_count = int((terminated | truncated).sum().item())
                    finished_returns = torch.as_tensor(info["episode_returns"]).detach().cpu()
                    finished_success = torch.as_tensor(info["episode_success"]).detach().cpu()
                    if finished_returns.shape != (done_count,) or finished_success.shape != (done_count,):
                        raise ValueError("Episode metrics must contain only completed whole-task episodes.")
                    _require_finite("episode returns", finished_returns)
                    recent_returns.extend(finished_returns.tolist())
                    recent_success.extend(finished_success.bool().tolist())
                    recent_returns, recent_success = recent_returns[-100:], recent_success[-100:]
                    if "episode_filled_count" in info:
                        finished_filled = torch.as_tensor(info["episode_filled_count"]).detach().cpu()
                        if finished_filled.shape != (done_count,):
                            raise ValueError("Filled counts must contain completed episodes only.")
                        recent_filled.extend(finished_filled.tolist())
                        recent_filled = recent_filled[-100:]
                    for key, recent in (("episode_lifted", recent_lifted),
                                        ("episode_released", recent_released)):
                        finished = torch.as_tensor(info[key]).detach().cpu()
                        if finished.shape != (done_count,):
                            raise ValueError(f"{key} must contain completed episodes only.")
                        recent.extend(finished.bool().tolist())
                        del recent[:-100]
                    self.total_steps += nenv
                if partial:
                    break
                with torch.no_grad():
                    next_value = self.policy.value(obs)
                    advantages, returns = compute_gae(rewards, dones, values, next_value, cfg["gamma"], cfg["gae_lambda"])
                    _require_finite("advantages", advantages)
                b_obs = {
                    "image": obs_buf["image"].reshape(-1, *IMAGE_SHAPE),
                    "proprio": obs_buf["proprio"].reshape(-1, self.observation_dim),
                }
                b_actions = action_buf.reshape(-1, self.action_dim)
                b_log_prob, b_returns, b_adv = log_buf.flatten(), returns.flatten(), advantages.flatten()
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std(unbiased=False) + 1e-8)
                if cfg["anneal_lr"]:
                    self.optimizer.param_groups[0]["lr"] = cfg["learning_rate"] * (1. - (local_update - 1) / cfg["updates"])
                losses, policy_losses, value_losses, entropies, kls, clipfracs = [], [], [], [], [], []
                for _ in range(cfg["epochs"]):
                    epoch_kls = []
                    permutation = torch.randperm(batch_size, device=self.device)
                    for start in range(0, batch_size, cfg["minibatch_size"]):
                        idx = permutation[start:start + cfg["minibatch_size"]]
                        _, log_prob, entropy, value = self.policy.act(_select_observation(b_obs, idx), b_actions[idx])
                        log_ratio = log_prob - b_log_prob[idx]
                        ratio = log_ratio.exp()
                        policy_loss = torch.maximum(-b_adv[idx] * ratio,
                            -b_adv[idx] * ratio.clamp(1 - cfg["clip_coef"], 1 + cfg["clip_coef"])).mean()
                        value_loss = .5 * (value - b_returns[idx]).square().mean()
                        loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy.mean()
                        _require_finite("PPO loss", loss)
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.policy.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
                        self.optimizer.step()
                        kl = ((ratio - 1) - log_ratio).detach().mean().item()
                        epoch_kls.append(kl)
                        losses.append(loss.detach().item())
                        policy_losses.append(policy_loss.detach().item())
                        value_losses.append(value_loss.detach().item())
                        entropies.append(entropy.detach().mean().item())
                        kls.append(kl)
                        clipfracs.append(((ratio - 1).abs() > cfg["clip_coef"]).float().detach().mean().item())
                    if cfg["target_kl"] is not None and np.mean(epoch_kls) > cfg["target_kl"]:
                        break
                self.completed_updates = self.run_start_updates + local_update
                self._publish()
                row = {
                    "update": self.completed_updates, "run_update": local_update,
                    "steps": self.total_steps, "snapshot_version": self.snapshot_version,
                    "train_mean_return": float(np.mean(recent_returns)) if recent_returns else None,
                    "train_success_rate": float(np.mean(recent_success)) if recent_success else None,
                    "train_mean_filled_count": float(np.mean(recent_filled)) if recent_filled else None,
                    "train_lift_rate": float(np.mean(recent_lifted)) if recent_lifted else None,
                    "train_release_rate": float(np.mean(recent_released)) if recent_released else None,
                    "completed_episodes_window": len(recent_returns),
                    "ppo_loss": float(np.mean(losses)), "policy_loss": float(np.mean(policy_losses)),
                    "value_loss": float(np.mean(value_losses)), "entropy": float(np.mean(entropies)),
                    "approx_kl": float(np.mean(kls)), "clip_fraction": float(np.mean(clipfracs)),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "steps_per_second": (self.total_steps - self.run_start_steps) / max(time.perf_counter() - began, 1e-6),
                    "evaluation": None,
                }
                if local_update % cfg["eval_every"] == 0 or local_update == cfg["updates"]:
                    record = self._evaluate(self.completed_updates)
                    row["evaluation"] = record["results"] if record else None
                with self.lock:
                    self.history.append(_json_safe(row))
                if local_update % cfg["save_every"] == 0 or local_update == cfg["updates"]:
                    self.save("latest.pt")
            if self.stop_event.is_set() and cfg["evaluate_on_stop"] and self.completed_updates > self.run_start_updates:
                self._evaluate(self.completed_updates)
            self.status = "중지됨" if self.stop_event.is_set() else "완료"
            self.save("latest.pt")
            return self.final_evaluation
        finally:
            if env is not None and hasattr(env, "close"):
                env.close()
            with self.lock:
                self._training = False

    def save(self, filename="latest.pt", initial=False):
        with self._save_lock:
            weights, version = self.snapshot(initial=initial)
            with self.lock:
                history, evaluations = copy.deepcopy(self.history), copy.deepcopy(self.evaluations)
                payload = {
                    "format_version": 4, "task_id": TASK_ID,
                    "policy_type": POLICY_TYPE,
                    "observation_spec": _json_safe(OBSERVATION_SPEC),
                    "architecture": copy.deepcopy(POLICY_ARCHITECTURE),
                    "action_semantics": ATTRIBUTION["action_semantics"],
                    "policy": weights,
                    "optimizer": copy.deepcopy(self.optimizer.state_dict()) if self.optimizer is not None and not initial else None,
                    "initial_policy": copy.deepcopy(self.initial_weights),
                    "observation_dim": self.observation_dim, "action_dim": self.action_dim,
                    "hidden_dim": 64, "image_shape": list(IMAGE_SHAPE), "config": _json_safe(self.config),
                    "snapshot_version": version, "training_updates": self.completed_updates,
                    "steps": self.total_steps, "status": self.status,
                    "baseline": copy.deepcopy(self.baseline), "final_evaluation": copy.deepcopy(self.final_evaluation),
                    "history": history, "evaluations": evaluations,
                    "attribution": ATTRIBUTION, "metadata": self.metadata, "versions": self.versions,
                    "warm_start": self.warm_start,
                    "checkpoint_scope": "policy and optimizer warm-start; simulator, rollout, and RNG restart",
                }
            path = self.output_dir / filename
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)
            for name, value in (("config.json", payload["config"]), ("history.json", history),
                                ("evaluations.json", evaluations),
                                ("provenance.json", {"attribution": ATTRIBUTION, "metadata": self.metadata,
                                                     "versions": self.versions, "warm_start": self.warm_start,
                                                     "policy_type": POLICY_TYPE,
                                                     "observation_spec": OBSERVATION_SPEC,
                                                     "architecture": POLICY_ARCHITECTURE})): 
                destination = self.output_dir / name
                pending = destination.with_suffix(destination.suffix + ".tmp")
                pending.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                pending.replace(destination)
            if history:
                destination = self.output_dir / "learning_curve.csv"
                pending = destination.with_suffix(".csv.tmp")
                with pending.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                    writer.writeheader()
                    for row in history:
                        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                                         for key, value in row.items()})
                pending.replace(destination)
            return path
