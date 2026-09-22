"""Focused regressions for exit-learning defaults and validation selection."""
import unittest

import pandas as pd

from banknifty_env import BankNiftyEnv, validate_config
from banknifty_rl_web import (
    TRAIN_DEFAULTS, ENV_VERSION, CANDIDATE_TRAIN_DEFAULTS, validate_settings_file,
    validation_selection_score, validation_target_results,
)


class TrainingTargetTests(unittest.TestCase):
    def test_settings_round_trip_preserves_custom_values(self):
        payload = dict(format="banknifty_training_settings", version=1,
                       environment_version=ENV_VERSION,
                       candidate_training=dict(CANDIDATE_TRAIN_DEFAULTS),
                       training=dict(TRAIN_DEFAULTS, seed=123, target_average_r=-0.1),
                       environment=validate_config({"terminal_win_min_points": 15.0,
                                                    "path_efficiency_gate_enabled": True}))
        result = validate_settings_file(payload)
        self.assertEqual(result["settings"], payload)
        self.assertEqual(result["defaulted_fields"], [])

    def test_settings_reject_invalid_types_and_report_missing_fields(self):
        payload = dict(format="banknifty_training_settings", version=1,
                       environment_version=ENV_VERSION, training={}, environment={})
        result = validate_settings_file(payload)
        self.assertIn("training.seed", result["defaulted_fields"])
        for invalid in (True, 1.5, "42", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_settings_file(dict(payload, training={"seed": invalid}))
        with self.assertRaises(ValueError):
            validate_settings_file(dict(payload, environment={"unknown_setting": 1}))

    def test_exit_learning_defaults_keep_protective_exits(self):
        cfg = validate_config({})
        self.assertFalse(cfg["path_efficiency_gate_enabled"])
        self.assertTrue(cfg["learned_exit_enabled"])
        for key in ("risk_reward_target_enabled", "underlying_target_enabled",
                    "momentum_decay_exit_enabled"):
            self.assertFalse(cfg[key])
        for key in ("emergency_stop_enabled", "structural_stop_enabled",
                    "max_hold_enabled", "square_off_enabled"):
            self.assertTrue(cfg[key])

    def test_path_window_requires_at_least_two_bars(self):
        with self.assertRaises(ValueError):
            validate_config({"path_efficiency_window": 1})

    def test_target_assessment_requires_sample_and_defined_pf(self):
        metrics = dict(total_trades=50, win_rate=70.0, profit_factor=1.5,
                       average_R=0.1, max_drawdown_R=2.0)
        self.assertTrue(validation_target_results(metrics)["targets_met"])
        self.assertFalse(validation_target_results(
            dict(metrics, total_trades=49))["targets_met"])
        self.assertFalse(validation_target_results(
            dict(metrics, profit_factor=None, win_rate=100))["targets_met"])
        self.assertFalse(validation_target_results(
            dict(metrics, profit_factor=1.49))["targets_met"])

    def test_zero_trades_cannot_outscore_active_candidate(self):
        active = dict(total_trades=50, win_rate=40, profit_factor=0.8,
                      average_R=-0.1, max_drawdown_R=5)
        self.assertLess(validation_selection_score({}),
                        validation_selection_score(active))

    def test_entry_diagnostics_identify_extension_block_despite_momentum(self):
        env = BankNiftyEnv.__new__(BankNiftyEnv)
        for key, value in validate_config({}).items():
            setattr(env, key, value)
        row = dict(return_3m=0.001, return_5m=0.001,
                   atr_ratio_20=1.0, close=102.0, ema_9=101.0,
                   ema_20=100.0, ema_50=99.0, reference_underlying_atr=1.0)
        allowed, details = env._entry_gate_details(row, "CE", {})
        self.assertFalse(allowed)
        self.assertTrue(details["checks"]["momentum_3m"])
        self.assertTrue(details["checks"]["momentum_5m"])
        self.assertTrue(details["checks"]["trend_alignment"])
        self.assertFalse(details["checks"]["extension"])
        env.extension_gate_enabled = False
        self.assertTrue(env._entry_gate_details(row, "CE", {})[0])

    def test_giveback_penalty_works_without_hold_bonus_and_respects_cap(self):
        # Exercise the reward method independently of datasets or model training.
        env = BankNiftyEnv.__new__(BankNiftyEnv)
        for key, value in validate_config({"hold_shaping_enabled": False}).items():
            setattr(env, key, value)
        timestamp = pd.Timestamp("2026-06-01 10:00")
        env.day_df = pd.DataFrame([dict(timestamp=timestamp)])
        env.step_idx = 0
        env.position = dict(pending_exit=False, last_quote_time=timestamp,
                            high_since_entry=160, entry_price=100,
                            initial_risk_points=30, shaping_used=0.0, shaping_r=0.0)
        env._liquidation_mark = lambda position: 0.5
        self.assertLess(env._shape_hold(), 0)
        env.position["shaping_used"] = env.max_shaping_r_per_trade
        self.assertEqual(env._shape_hold(), 0)


if __name__ == "__main__":
    unittest.main()
