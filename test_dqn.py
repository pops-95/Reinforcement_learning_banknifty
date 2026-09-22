"""User-run DQN checks, including a short CPU synthetic training/save-load test.

No historical data or web server is needed. Not executed during implementation.
"""
import json
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from banknifty_dqn import BankNiftyDQN, MaskReplayBuffer, NextActionMask, NEXT_MASK, masked_next_values


class TinyMarket(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, (2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.holding, self.t, self.done = False, 0, False
        return np.array([0., 0.], dtype=np.float32), {}

    def action_masks(self):
        if self.done:
            return np.array([True, False, False, False, False])
        return np.array([False, False, False, True, True] if self.holding
                        else [True, True, True, False, False])

    def step(self, action):
        assert self.action_masks()[int(action)], "DQN selected an invalid trading action"
        reward = 1.0 if action == 4 else -.01
        if action in (1, 2):
            self.holding = True
        elif action == 4:
            self.holding = False
        self.t += 1
        self.done = self.t == 8
        return np.array([float(self.holding), self.t], dtype=np.float32), reward, self.done, False, {}


class DQNTests(unittest.TestCase):
    def test_bellman_targets_ignore_invalid_actions_and_double_decouples_selection(self):
        online = th.tensor([[999., 2., 4., 888., 777.]])
        target = th.tensor([[999., 20., 10., 888., 777.]])
        masks = th.tensor([[False, True, True, False, False]])
        self.assertEqual(masked_next_values(online, target, masks, True).item(), 10.)
        self.assertEqual(masked_next_values(online, target, masks, False).item(), 20.)

    def test_replay_masks_remain_aligned_when_ring_buffer_wraps(self):
        buffer = MaskReplayBuffer(2, TinyMarket.observation_space, TinyMarket.action_space, device="cpu")
        for index in range(3):
            mask = np.eye(5, dtype=bool)[index]
            buffer.add(np.array([[index, 0.]], np.float32), np.array([[index+1, 0.]], np.float32),
                       np.array([index]), np.array([0.]), np.array([False]), [{NEXT_MASK: mask}])
        sample = buffer._get_samples(np.array([0, 1]))
        np.testing.assert_array_equal(sample.observations[:, 0].numpy(), [2, 1])
        np.testing.assert_array_equal(sample.next_action_masks.numpy(), np.eye(5, dtype=bool)[[2, 1]])
        with self.assertRaises(KeyError):
            buffer.add(np.zeros((1, 2)), np.zeros((1, 2)), np.array([0]), np.array([0.]),
                       np.array([False]), [{}])

    def test_terminal_mask_captured_before_vector_auto_reset(self):
        env = NextActionMask(TinyMarket())
        env.reset()
        for _ in range(8):
            _, _, done, _, info = env.step(0)
        self.assertTrue(done)
        np.testing.assert_array_equal(info[NEXT_MASK], [True, False, False, False, False])

    def test_warmup_exploration_training_and_save_load(self):
        env = VecNormalize(DummyVecEnv([lambda: Monitor(NextActionMask(TinyMarket()))]),
                           norm_obs=True, norm_reward=False)
        self.addCleanup(env.close)
        model = BankNiftyDQN("MlpPolicy", env, buffer_size=128, learning_starts=8,
                            train_freq=4, gradient_steps=1, batch_size=8,
                            target_update_interval=16, policy_kwargs=dict(net_arch=[16, 16]),
                            device="cpu", seed=42)
        model.market_algorithm = "dqn"
        model.learn(total_timesteps=64)
        self.assertGreater(model._n_updates, 0)
        self.assertTrue(all(th.isfinite(p).all() for p in model.policy.parameters()))
        obs = env.reset()
        masks = np.asarray(env.env_method("action_masks"))
        for eps in (0., 1.):
            model.exploration_rate = eps
            for _ in range(20):
                action, _ = model.predict(obs, action_masks=masks)
                self.assertTrue(masks[0, int(action[0])])
        with self.assertRaises(ValueError):
            model.predict(obs)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model.save(root / "model.zip")
            env.save(root / "normalization.pkl")
            from banknifty_rl_web import load_trading_model
            restored = load_trading_model(root / "model.zip")
            expected, _ = model.predict(obs, deterministic=True, action_masks=masks)
            actual, _ = restored.predict(obs, deterministic=True, action_masks=masks)
            np.testing.assert_array_equal(actual, expected)
            restored_env = VecNormalize.load(root / "normalization.pkl",
                                             DummyVecEnv([lambda: Monitor(NextActionMask(TinyMarket()))]))
            self.addCleanup(restored_env.close)
            restored_env.training = False
            restored_env.norm_reward = False
            np.testing.assert_allclose(restored_env.obs_rms.mean, env.obs_rms.mean)

    def test_both_presets_load_and_old_settings_default_to_ppo(self):
        import banknifty_rl_web as web
        client = web.app.test_client()
        self.assertEqual(web.parse_training_config({})["algorithm"], "ppo")
        for algorithm in ("ppo", "dqn"):
            path = Path(__file__).parent / f"settings/banknifty_{algorithm}_trade_quality.json"
            payload = json.loads(path.read_text())
            response = client.post("/api/settings/validate", json=payload)
            self.assertEqual(response.status_code, 200, response.json)
            cfg = response.json["settings"]
            self.assertEqual(cfg["training"]["algorithm"], algorithm)
            self.assertFalse(cfg["training"]["daily_targets_enabled"])
            for key in ("daily_profit_limit_enabled", "daily_loss_limit_enabled", "close_on_daily_limit_enabled"):
                self.assertFalse(cfg["environment"][key])
        for payload in ({"algorithm": "sac"}, {"dqn_tau": 0}, {"dqn_gradient_steps": 0},
                        {"dqn_final_epsilon": 1.1}, {"algorithm": 1}):
            with self.assertRaises(ValueError):
                web.parse_training_config(payload)

    def test_shared_banknifty_environment_and_web_action_path(self):
        from test_autonomous_rl import fixture
        from banknifty_env import BankNiftyEnv
        from banknifty_rl_web import _policy_action
        with tempfile.TemporaryDirectory() as temp:
            paths, _, _ = fixture(Path(temp))
            market = BankNiftyEnv(**paths, split="validation", random_day=False,
                                 reference_features_enabled=False)
            vec = VecNormalize(DummyVecEnv([lambda: Monitor(NextActionMask(market))]),
                               norm_reward=False)
            try:
                model = BankNiftyDQN("MlpPolicy", vec, buffer_size=32, learning_starts=4,
                                    train_freq=4, gradient_steps=1, batch_size=4,
                                    policy_kwargs=dict(net_arch=[8]), device="cpu", seed=42)
                model.market_algorithm = "dqn"
                invalid = []

                def record(local, _global):
                    invalid.extend(info["invalid_action"] for info in local["infos"])
                    return True

                model.learn(16, callback=record)
                self.assertEqual(len(invalid), 16)
                self.assertFalse(any(invalid))
                vec.training = False
                obs, _ = market.reset()
                action = _policy_action(model, vec, market, obs)
                self.assertTrue(market.action_masks()[action])
                self.assertIsNone(market._action_probability)
            finally:
                vec.close()


if __name__ == "__main__":
    unittest.main()
