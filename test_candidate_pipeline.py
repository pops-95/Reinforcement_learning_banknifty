"""Synthetic candidate regressions. These tests do not train a policy."""
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from banknifty_env import CandidateBankNiftyEnv, BankNiftyEnv, BUY_CE, HOLD, SKIP, TAKE
from build_candidate_dataset import candidate_environment_config, config_fingerprint
from reference_strategy import CANDIDATE_VALUES, CANDIDATE_FEATURES, candidate_observations
from train_candidate_model import parse_candidate_training, good_labels, replay_candidates


def candidate_fixture():
    rows = []
    for i, timestamp in enumerate(pd.date_range("2026-06-01 10:00", periods=5, freq="min")):
        trade = dict(side="CE", option_pnl_points=30.0, R_return=1.0,
                     stale_exit=False, regime="TREND_UP|NORMAL_VOL|NON_EXPIRY")
        rows.append({**dict.fromkeys(CANDIDATE_VALUES, 0.0),
                     "candidate_id": f"candidate_{i}", "timestamp": timestamp,
                     "split": "validation", "side": "CE", "filled": True,
                     "realized_R": 1.0, "stale_exit": False,
                     "next_decision_time": timestamp+pd.Timedelta(minutes=3),
                     "trade_json": json.dumps(trade)})
    return pd.DataFrame(rows)


class CandidateEnvironmentTests(unittest.TestCase):
    def test_observations_do_not_include_labels_or_future_exit_times(self):
        frame = candidate_fixture()
        changed = frame.copy()
        changed["realized_R"] = -1000.0
        changed["MFE_R"] = 99999.0
        changed["filled"] = False
        changed["next_decision_time"] = pd.Timestamp("2030-01-01")
        np.testing.assert_array_equal(candidate_observations(frame), candidate_observations(changed))
        self.assertEqual(candidate_observations(frame).shape[1], len(CANDIDATE_FEATURES))

    def test_skip_advances_one_event_and_take_enforces_occupancy(self):
        env = CandidateBankNiftyEnv(candidate_fixture(), "validation", False)
        env.reset()
        _, reward, done, _, info = env.step(SKIP)
        self.assertEqual(reward, 0)
        self.assertEqual(env.index, 1)
        self.assertFalse(done)
        _, reward, done, _, info = env.step(TAKE)
        self.assertEqual(reward, 1)
        self.assertEqual(env.index, 4)
        self.assertEqual(info["unavailable_candidates"], 2)
        self.assertEqual(info["taken_candidates"], 1)
        self.assertFalse(done)
        _, _, done, _, _ = env.step(SKIP)
        self.assertTrue(done)

    def test_failed_fill_remains_a_candidate_and_is_not_a_win(self):
        frame = candidate_fixture()
        frame.loc[0, "filled"] = False
        env = CandidateBankNiftyEnv(frame, "validation", False)
        env.reset()
        _, reward, _, _, info = env.step(TAKE)
        self.assertEqual(reward, 0)
        self.assertEqual(env.index, 1)
        self.assertEqual(info["rejected_candidates"], 1)
        self.assertIsNone(info["trade"])

    def test_replay_reports_unavailable_events_separately(self):
        frame = candidate_fixture()
        report = replay_candidates(frame, "validation", lambda obs: TAKE,
                                   parse_candidate_training(), {"split_days": {"validation": 2}})
        self.assertEqual(report["metrics"]["total_trades"], 2)
        self.assertEqual(report["metrics"]["total_trading_days"], 2)
        self.assertEqual(report["metrics"]["trades_per_day"], 1.0)
        counts = report["candidates"]
        self.assertEqual(counts["reference_candidates"], 5)
        self.assertEqual(counts["taken_candidates"], 2)
        self.assertEqual(counts["unavailable_candidates"], 3)
        self.assertEqual(counts["candidate_recall"], 40.0)

    def test_target_labels_and_configuration_identity(self):
        frame = candidate_fixture()
        frame.loc[0, "realized_R"] = 0.0
        frame.loc[1, "stale_exit"] = True
        cfg = parse_candidate_training({"good_trade_r": 1.0})
        self.assertEqual(good_labels(frame, cfg).tolist(), [False, False, True, True, True])
        first = candidate_environment_config()
        second = candidate_environment_config({"risk_reward_target_r": 2.0})
        self.assertNotEqual(config_fingerprint(first), config_fingerprint(second))
        self.assertFalse(first["learned_exit_enabled"])
        self.assertFalse(first["entry_penalty_enabled"])

    def test_shared_simulator_matches_minute_execution(self):
        # Same exact historical candles/config, one deterministic entry and exit.
        from test_learned_exits import synthetic_files
        with tempfile.TemporaryDirectory() as directory:
            paths = synthetic_files(directory, count=30, days=1)
            cfg = candidate_environment_config(dict(
                structural_stop_enabled=False, momentum_decay_exit_enabled=False,
                underlying_target_enabled=False, transaction_costs_enabled=False,
                dynamic_slippage_enabled=False, slippage_pct=0.0,
                min_stop_points=10.0, premium_stop_enabled=False, use_atr_stop=False,
                max_hold_minutes=3, min_hold_minutes=3,
            ))
            envs = [BankNiftyEnv(**paths, split="validation", random_day=False, **cfg) for _ in range(2)]
            try:
                for env in envs:
                    env.reset()
                    env.day_df["reference_side"] = "CE"
                    env.day_df["reference_symbol"] = "CE"
                labeled = envs[0].simulate_reference_candidate(0)
                envs[1].step(BUY_CE)
                while envs[1].position is not None:
                    envs[1].step(HOLD)
                actual = envs[1].trade_log[-1]
                self.assertEqual(labeled["exit_reason"], actual["exit_reason"])
                self.assertEqual(labeled["exit_time"], actual["exit_time"])
                self.assertEqual(labeled["R_return"], actual["R_return"])
            finally:
                for env in envs:
                    env.close()


if __name__ == "__main__":
    unittest.main()
