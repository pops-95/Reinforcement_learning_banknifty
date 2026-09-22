"""Masked one-step DQN / optional Double DQN for the shared trading environment.

Uses SB3's replay, target-network updates, normalization, save/load and CUDA
support. Masks apply to warmup, epsilon exploration, greedy actions AND TD
targets. A plain SB3 DQN would not respect this environment's action masks.
The current website uses one vectorized environment; enforce that invariant.
"""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer


NEXT_MASK = "dqn_next_action_mask"


class NextActionMask(gym.Wrapper):
    """Capture next-state masks before DummyVecEnv automatically resets a day."""

    def action_masks(self):
        return self.env.unwrapped.action_masks()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info[NEXT_MASK] = np.asarray(self.action_masks(), dtype=bool).copy()
        return obs, reward, terminated, truncated, info


class MaskReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError("Masked trading DQN currently supports one environment")
        self.next_action_masks = np.zeros(
            (self.buffer_size, self.n_envs, self.action_space.n), dtype=bool
        )

    def add(self, obs, next_obs, action, reward, done, infos):
        masks = np.asarray([info[NEXT_MASK] for info in infos], dtype=bool)
        if masks.shape != (self.n_envs, self.action_space.n) or not masks.any(axis=1).all():
            raise ValueError("Replay transition needs a nonempty next-state action mask")
        self.next_action_masks[self.pos] = masks
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        # With n_envs == 1 the parent's sampled environment index is always zero.
        sample = super()._get_samples(batch_inds, env=env)
        return SimpleNamespace(
            **sample._asdict(),
            next_action_masks=self.to_torch(self.next_action_masks[batch_inds, 0]),
        )


def masked_next_values(online, target, masks, double_dqn):
    """Invalid next actions must never contribute to a Bellman target."""
    if not masks.any(dim=1).all():
        raise ValueError("Empty next-state action mask")
    if double_dqn:
        actions = online.masked_fill(~masks, -th.inf).argmax(dim=1, keepdim=True)
        return target.gather(1, actions)
    return target.masked_fill(~masks, -th.inf).max(dim=1, keepdim=True).values


class BankNiftyDQN(DQN):
    def __init__(self, *args, double_dqn_enabled=True, **kwargs):
        self.double_dqn_enabled = bool(double_dqn_enabled)
        kwargs["replay_buffer_class"] = MaskReplayBuffer
        super().__init__(*args, **kwargs)

    def predict(self, observation, state=None, episode_start=None,
                deterministic=False, action_masks=None):
        if action_masks is None:
            raise ValueError("BankNiftyDQN.predict requires the observation's action_masks")
        self.policy.set_training_mode(False)
        tensor, vectorized = self.policy.obs_to_tensor(observation)
        masks = np.asarray(action_masks, dtype=bool).reshape(-1, self.action_space.n)
        if len(masks) != tensor.shape[0] or not masks.any(axis=1).all():
            raise ValueError("Action masks must match the observation batch and allow an action")
        with th.no_grad():
            q = self.q_net(tensor)
            valid = th.as_tensor(masks, device=q.device)
            actions = q.masked_fill(~valid, -th.inf).argmax(dim=1).cpu().numpy()
        if not deterministic:
            for index, mask in enumerate(masks):
                if np.random.random() < self.exploration_rate:
                    actions[index] = np.random.choice(np.flatnonzero(mask))
        return (actions if vectorized else actions.reshape(())), state

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        masks = np.asarray(self.env.env_method("action_masks"), dtype=bool)
        if masks.shape != (n_envs, self.action_space.n) or not masks.any(axis=1).all():
            raise ValueError("Environment returned invalid DQN action masks")
        if self.num_timesteps < learning_starts:
            actions = np.asarray([np.random.choice(np.flatnonzero(m)) for m in masks])
        else:
            actions, _ = self.predict(self._last_obs, deterministic=False, action_masks=masks)
        return actions, actions.copy()

    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses, q_values = [], []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                target_q = self.q_net_target(data.next_observations)
                online_q = self.q_net(data.next_observations) if self.double_dqn_enabled else target_q
                next_values = masked_next_values(online_q, target_q, data.next_action_masks,
                                                 self.double_dqn_enabled)
                target = data.rewards + (1.0 - data.dones) * self.gamma * next_values
            current = self.q_net(data.observations).gather(1, data.actions.long())
            loss = th.nn.functional.smooth_l1_loss(current, target)
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
            losses.append(float(loss.item()))
            q_values.append(float(current.detach().mean().item()))
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if losses:
            self.logger.record("train/loss", float(np.mean(losses)))
            self.logger.record("train/mean_q", float(np.mean(q_values)))
        self.logger.record("train/exploration_rate", self.exploration_rate)
        self.logger.record("train/replay_size", self.replay_buffer.size())


def build_dqn(env, cfg, device, seed, tensorboard_log=None):
    return BankNiftyDQN(
        "MlpPolicy", env, learning_rate=cfg["learning_rate"], gamma=cfg["gamma"],
        batch_size=cfg["batch_size"], max_grad_norm=cfg["max_grad_norm"],
        buffer_size=cfg["dqn_buffer_size"], learning_starts=cfg["dqn_learning_starts"],
        train_freq=cfg["dqn_train_freq"], gradient_steps=cfg["dqn_gradient_steps"],
        target_update_interval=cfg["dqn_target_update_interval"], tau=cfg["dqn_tau"],
        exploration_fraction=cfg["dqn_exploration_fraction"],
        exploration_initial_eps=cfg["dqn_initial_epsilon"],
        exploration_final_eps=cfg["dqn_final_epsilon"],
        double_dqn_enabled=cfg["dqn_double_enabled"],
        policy_kwargs=dict(activation_fn=th.nn.ReLU,
                           net_arch=[cfg["pi_layer1"], cfg["pi_layer2"], cfg["pi_layer3"]]),
        device=device, seed=seed, verbose=1, tensorboard_log=tensorboard_log,
    )
