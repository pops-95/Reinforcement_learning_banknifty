"""Small synthetic checks; no training or historical backtest."""
import tempfile
import unittest

from banknifty_rl_web import app, parse_env_config  # Load torch before parquet worker threads on Windows.
import numpy as np
import pandas as pd

from banknifty_env import BankNiftyEnv, validate_config
from reference_strategy import ignition_features, select_contract, structural_stop
from test_learned_exits import synthetic_files


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = synthetic_files(self.temp.name)

    def env(self, **settings):
        env = BankNiftyEnv(**self.paths, split="validation", **settings)
        env.reset(seed=7)
        return env

    def enter(self, env):
        for _ in range(env.reference_window):
            env.step(0)
        self.assertEqual(env.action_masks().tolist(), [True, True, False])
        env.step(1)
        self.assertIsNotNone(env.position)

    def test_warmup_mask_wait_and_fixed_contract(self):
        env = self.env()
        self.assertEqual(env.action_masks().tolist(), [True, False, False])
        self.enter(env)
        symbol = env.position["symbol"]
        env.day_df.loc[env.step_idx:, "reference_symbol"] = "OTHER"
        env.step(0)
        self.assertEqual(env.position["symbol"], symbol)

    def test_target_uses_index_points_and_next_open(self):
        env = self.env(reference_target_underlying_points=10.)
        self.enter(env)
        entry = env.position["spot_entry"]
        env.day_df.loc[env.step_idx, "close"] = entry+11
        next_ts = env.day_df.iloc[env.step_idx+1].timestamp
        key = (env.position["symbol"], next_ts)
        env.option_lookup[key] = env.option_lookup[key]._replace(open=123., high=900., low=1.)
        env.step(0)
        trade = env.trade_log[-1]
        self.assertEqual(trade["exit_reason"], "UNDERLYING_TARGET")
        self.assertEqual(trade["exit_time"], next_ts)
        self.assertAlmostEqual(trade["exit_price"], 123*.999)

    def test_structural_stop_overrides_minimum_hold(self):
        env = self.env()
        self.enter(env)
        env.day_df.loc[env.step_idx, "low"] = env.position["structural_stop"]-1
        env.step(0)
        self.assertEqual(env.trade_log[-1]["exit_reason"], "STRUCTURAL_STOP")

    def test_decay_and_missing_next_quote_remain_pending(self):
        env = self.env()
        self.enter(env)
        env.day_df.loc[env.step_idx, ["reference_bull", "reference_bear"]] = [.3, .7]
        key = (env.position["symbol"], env.day_df.iloc[env.step_idx+1].timestamp)
        del env.option_lookup[key]
        env.step(0)
        self.assertTrue(env.position["pending_exit"])
        env.step(0)
        self.assertEqual(env.trade_log[-1]["exit_reason"], "MOMENTUM_DECAY")

    def test_features_and_structural_stop_are_prefix_causal(self):
        env = self.env()
        bars, options = env.day_df, env.reference_options
        full = ignition_features(bars, options)
        prefix = ignition_features(bars.iloc[:35], options.loc[options.timestamp <= bars.timestamp.iloc[34]])
        np.testing.assert_allclose(full.iloc[:35], prefix, equal_nan=True)
        history = pd.DataFrame({"high": [12, 11, 10, 11, 12, 13], "low": [10, 9, 8, 9, 10, 11]})
        self.assertEqual(structural_stop(history, "CE", 12., buffer_fraction=0), 8.)

    def test_contract_selection_completed_liquidity(self):
        snapshot = pd.DataFrame([
            dict(groww_symbol="ATM", option_type="CE", strike=50000, close=100, volume=10, oi=100),
            dict(groww_symbol="ITM", option_type="CE", strike=49900, close=120, volume=50, oi=100),
            dict(groww_symbol="FAR", option_type="CE", strike=49800, close=150, volume=9999, oi=100)])
        self.assertEqual(select_contract(snapshot, "CE", 50010), "ITM")

    def test_config_and_website(self):
        for config in ({"reference_ignition_min": 1.1}, {"reference_window": 3},
                       {"reference_strike_spacing": 0}, {"reference_pivot_right": 0}):
            with self.assertRaises(ValueError):
                validate_config(config)
        self.assertEqual(parse_env_config({"reference_target_underlying_points": 90})["reference_target_underlying_points"], 90)
        response = app.test_client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"reference_ignition_min", response.data)


if __name__ == "__main__":
    unittest.main()
