"""User-run synthetic regressions. No training job or real trading data required."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from banknifty_env import BankNiftyEnv, POSITION_FEATURES, BUY_CE, EXIT, WAIT
from banknifty_rl_web import (
    TRAIN_DEFAULTS, LiveCallback, exploration_feedback, parse_training_config,
    validation_candidate_key, validation_target_results, validate_settings_file,
)
from test_autonomous_rl import fixture


class OpportunityEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths, self.frame, self.candles = fixture(self.root)

    def env(self, split="validation", random_day=False, **overrides):
        cfg = dict(reference_features_enabled=False, transaction_costs_enabled=False,
                   slippage_pct=0., observation_history_bars=4)
        cfg.update(overrides)
        env = BankNiftyEnv(**self.paths, split=split, random_day=random_day, **cfg)
        self.addCleanup(env.close)
        env.reset(seed=42)
        return env

    def test_history_shape_padding_causality_and_reset(self):
        self.frame["signal"] = np.arange(len(self.frame)) + 1.
        self.frame.to_parquet(self.paths["observations_file"], index=False)
        env = self.env()
        obs, _ = env.reset()
        width = len(env.feature_columns)
        self.assertEqual(obs.shape, (4 * width + 4 + len(POSITION_FEATURES),))
        np.testing.assert_array_equal(obs[:4 * width].reshape(4, width)[:, 0], [0, 0, 0, 1])
        np.testing.assert_array_equal(obs[4 * width:4 * width + 4], [0, 0, 0, 1])
        env.step(WAIT)
        before = env._get_observation()
        env.day_df.loc[2:, "signal"] = 999999.
        np.testing.assert_array_equal(before, env._get_observation())
        obs, _ = env.reset()
        np.testing.assert_array_equal(obs[:4 * width].reshape(4, width)[:, 0], [0, 0, 0, 1])

    def test_clock_gap_is_zero_padded_not_compressed(self):
        self.frame["signal"] = np.arange(len(self.frame)) + 1.
        self.frame = self.frame.drop(index=1)
        self.frame.to_parquet(self.paths["observations_file"], index=False)
        env = self.env()
        env.step_idx = 1  # 10:02, with 10:01 absent.
        history = env._market_history()
        width = len(env.feature_columns)
        np.testing.assert_array_equal(history[:4 * width].reshape(4, width)[:, 0], [0, 1, 0, 3])
        np.testing.assert_array_equal(history[-4:], [0, 1, 0, 1])

    def test_default_shape_stays_compatible(self):
        env = self.env(observation_history_bars=1)
        self.assertEqual(env._get_observation().shape,
                         (len(env.feature_columns) + len(POSITION_FEATURES),))

    def test_random_start_only_changes_training(self):
        validation = self.frame.copy()
        train = self.frame.copy()
        train["split"] = "train"
        train["timestamp"] -= pd.Timedelta(days=1)
        train_candles = self.candles.copy()
        train_candles["timestamp"] -= pd.Timedelta(days=1)
        pd.concat([train, validation]).to_parquet(self.paths["observations_file"], index=False)
        pd.concat([train_candles, self.candles]).to_parquet(
            self.root / "banknifty_options_2026-06-30.parquet", index=False)
        kwargs = dict(random_start_enabled=True, random_start_probability=1., random_start_max_minutes=4)
        train_env = self.env(split="train", random_day=True, **kwargs)
        starts = []
        for seed in range(12):
            _, info = train_env.reset(seed=seed)
            starts.append(train_env.step_idx)
            self.assertEqual(info["partial_session"], train_env.step_idx != 0)
        self.assertGreater(len(set(starts)), 1)
        self.assertTrue(all(0 <= index <= 4 for index in starts))
        eval_env = self.env(random_day=True, **kwargs)
        for seed in range(3):
            _, info = eval_env.reset(seed=seed)
            self.assertEqual(eval_env.step_idx, 0)
            self.assertFalse(info["partial_session"])

    def test_flat_cutoff_ends_episode_but_not_open_trade(self):
        kwargs = dict(last_entry_time="10:02", end_inactive_episode_enabled=True)
        env = self.env(**kwargs)
        self.assertFalse(env.step(WAIT)[2])
        self.assertTrue(env.step(WAIT)[2])
        holding = self.env(**kwargs)
        self.assertFalse(holding.step(BUY_CE)[2])
        self.assertIsNotNone(holding.position)
        # Do not liquidate or silently discard a position at the entry cutoff.
        from banknifty_env import HOLD
        self.assertFalse(holding.step(HOLD)[2])
        self.assertIsNotNone(holding.position)
        self.assertTrue(holding.step(EXIT)[2])
        self.assertEqual(len(holding.trade_log), 1)

    def test_zero_reward_wait_and_eligible_decision_flags(self):
        env = self.env(first_entry_time="10:02")
        _, reward, done, _, info = env.step(WAIT)
        self.assertEqual(reward, 0.)
        self.assertFalse(done)
        self.assertFalse(info["entry_available"])
        self.assertFalse(info["voluntary_wait"])
        # Entry masks use the next execution timestamp, so 10:01 can enter at 10:02.
        _, reward, _, _, info = env.step(WAIT)
        self.assertEqual(reward, 0.)
        self.assertTrue(info["entry_available"])
        self.assertTrue(info["voluntary_wait"])


class OpportunityTrainingTests(unittest.TestCase):
    def cfg(self, **overrides):
        return dict(TRAIN_DEFAULTS, adaptive_entropy_enabled=True, ent_coef=.02,
                    **overrides)

    def test_entropy_feedback_excludes_forced_wait_and_requires_sample(self):
        cfg = self.cfg()
        normal = exploration_feedback(cfg, .5, 1000, 500, 500)
        collapsed = exploration_feedback(cfg, .5, 1000, 999, 1)
        self.assertFalse(normal["wait_collapse_warning"])
        self.assertTrue(collapsed["wait_collapse_warning"])
        self.assertGreater(collapsed["entropy_coefficient"], normal["entropy_coefficient"])
        self.assertLessEqual(collapsed["entropy_coefficient"], cfg["entropy_max_coef"])
        no_opportunity = exploration_feedback(cfg, .5, 0, 0, 0)
        self.assertIsNone(no_opportunity["voluntary_wait_fraction"])
        self.assertFalse(no_opportunity["wait_collapse_warning"])
        self.assertFalse(exploration_feedback(cfg, .5, 127, 127, 0)["wait_collapse_warning"])

    def test_periodic_validation_at_next_rollout_start_only(self):
        cfg = self.cfg(validation_interval_steps=200)
        validate = Mock()
        live_env = object()
        model = SimpleNamespace(get_env=lambda: live_env, market_algorithm="ppo", ent_coef=.02,
                                logger=Mock())
        callback = LiveCallback(SimpleNamespace(lock=threading.RLock()),
                                training_cfg=cfg, seed_target=1000, validate_checkpoint=validate)
        callback.model = model
        callback._capture_optimizer = Mock()
        callback.num_timesteps = 100
        callback._on_rollout_start()
        validate.assert_not_called()
        callback.num_timesteps = 200
        callback._on_rollout_end()
        validate.assert_not_called()
        callback._on_rollout_start()
        validate.assert_called_once_with(model, live_env, 200)
        callback._on_rollout_start()
        self.assertEqual(validate.call_count, 1)

    def test_profit_first_selection_does_not_prefer_high_wr_losing_policy(self):
        cfg = self.cfg(profit_first_selection_enabled=True, min_trades_per_day=1.)
        common = dict(total_trades=150, total_trading_days=100)
        good = dict(common, profit_factor=1.6, average_R=.2, win_rate=60.)
        bad = dict(common, profit_factor=.8, average_R=-.1, win_rate=80.)
        candidates = [dict(targets=validation_target_results(good, cfg), score=.1),
                      dict(targets=validation_target_results(bad, cfg), score=10.)]
        self.assertIs(max(candidates, key=lambda c: validation_candidate_key(c, cfg)), candidates[0])
        self.assertFalse(candidates[0]["targets"]["targets_met"])

    def test_new_preset_roundtrip_and_controls(self):
        path = Path(__file__).parent / "settings" / "banknifty_ppo_opportunity_v3.json"
        loaded = validate_settings_file(json.loads(path.read_text()))["settings"]
        self.assertEqual(validate_settings_file(loaded)["settings"], loaded)
        env, train = loaded["environment"], loaded["training"]
        self.assertEqual(env["observation_history_bars"], 4)
        self.assertTrue(env["learned_exit_enabled"])
        self.assertFalse(env["daily_profit_limit_enabled"])
        self.assertFalse(env["daily_loss_limit_enabled"])
        self.assertFalse(env["loss_aversion_enabled"])
        self.assertTrue(train["cuda_enabled"])
        self.assertTrue(train["adaptive_entropy_enabled"])
        self.assertFalse(train["daily_targets_enabled"])

    def test_invalid_new_settings_rejected(self):
        for values in ({"validation_interval_steps": -1}, {"wait_collapse_threshold": 1.1},
                       {"wait_monitor_min_decisions": 0},
                       {"adaptive_entropy_enabled": True, "ent_coef": 0.},
                       {"environment": {"observation_history_bars": 0}},
                       {"environment": {"random_start_probability": 2.}}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_training_config(values)


if __name__ == "__main__":
    unittest.main()
