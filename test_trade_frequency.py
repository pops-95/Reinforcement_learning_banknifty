"""Validation-frequency regressions; no training or market data required."""
import json
from pathlib import Path
import unittest

from banknifty_rl_web import (
    parse_training_config, validate_settings_file, validation_selection_score,
    validation_candidate_key, validation_target_results,
)
from train_candidate_model import selection_key, trade_frequency_result


class TradeFrequencyTests(unittest.TestCase):
    def setUp(self):
        self.targets = dict(min_trades_per_day=2.0, min_validation_trades=100,
                            target_win_rate_pct=70.0, target_profit_factor=1.5,
                            target_average_r=0.1)
        self.metrics = dict(total_trades=200, total_trading_days=100,
                            win_rate=70.0, profit_factor=1.5,
                            average_R=0.1, max_drawdown_R=2.0)

    def test_boundary_uses_counts_instead_of_rounded_display(self):
        self.assertTrue(validation_target_results(self.metrics, self.targets)["targets_met"])
        below = dict(self.metrics, total_trades=199, trades_per_day=2.0)
        result = validation_target_results(below, self.targets)
        self.assertFalse(result["checks"]["trades_per_day"])
        self.assertFalse(result["targets_met"])
        self.assertEqual(result["trade_frequency"]["actual"], 1.99)

    def test_missing_days_and_partial_evaluation_cannot_pass(self):
        for days in (0, None):
            metrics = dict(self.metrics)
            if days is None:
                metrics.pop("total_trading_days")
            else:
                metrics["total_trading_days"] = days
            with self.subTest(days=days):
                self.assertFalse(validation_target_results(metrics, self.targets)["targets_met"])
        self.assertFalse(validation_target_results(
            self.metrics, self.targets, complete=False)["targets_met"])

    def test_older_settings_leave_frequency_disabled(self):
        targets = dict(self.targets)
        targets.pop("min_trades_per_day")
        metrics = dict(self.metrics)
        metrics.pop("total_trading_days")
        self.assertTrue(validation_target_results(metrics, targets)["targets_met"])
        self.assertEqual(parse_training_config({})["min_trades_per_day"], 0.0)

    def test_candidate_selection_prefers_frequency_eligible_result(self):
        # Both miss win rate; only the lower-expectancy result meets frequency.
        sparse = dict(metrics=dict(self.metrics, total_trading_days=200,
                                   win_rate=60.0, average_R=0.5))
        active = dict(metrics=dict(self.metrics, win_rate=60.0))
        self.assertGreater(selection_key(active, self.targets),
                           selection_key(sparse, self.targets))
        self.assertFalse(sparse["targets"]["checks"]["trades_per_day"])
        self.assertFalse(active["targets"]["targets_met"])

    def test_legacy_score_penalizes_lower_frequency(self):
        sparse = dict(self.metrics, total_trading_days=200)
        self.assertLess(validation_selection_score(sparse, self.targets),
                        validation_selection_score(self.metrics, self.targets))

    def test_legacy_seed_selection_prefers_minimum_and_preserves_disabled_order(self):
        active_metrics = dict(self.metrics, win_rate=60.0)
        sparse_metrics = dict(active_metrics, total_trading_days=200)
        active = dict(targets=validation_target_results(active_metrics, self.targets), score=0.1)
        sparse = dict(targets=validation_target_results(sparse_metrics, self.targets), score=0.5)
        self.assertIs(max([sparse, active], key=lambda c: validation_candidate_key(c, self.targets)), active)
        disabled = dict(self.targets, min_trades_per_day=0.0)
        self.assertIs(max([sparse, active], key=lambda c: validation_candidate_key(c, disabled)), sparse)

    def test_enabled_frequency_requires_all_evaluated_days(self):
        # Two trades on one active day plus one zero-trade day averages one/day.
        result = trade_frequency_result(dict(total_trades=2, total_trading_days=2), 2.0)
        self.assertEqual(result["actual"], 1.0)
        self.assertFalse(result["meets_minimum"])

    def test_invalid_frequency_rejected(self):
        for value in (-1, float("nan"), float("inf"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_training_config({"min_trades_per_day": value})

    def test_v16_settings_preserve_frequency_on_load_and_export(self):
        path = Path(__file__).parent / "settings" / "banknifty_settings_v1.6_min2_trades.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = validate_settings_file(payload)["settings"]
        # New optional controls default safely without changing saved values.
        for section in ("environment", "training"):
            self.assertEqual({key: loaded[section][key] for key in payload[section]},
                             payload[section])
        exported = json.loads(json.dumps(loaded))
        self.assertEqual(validate_settings_file(exported)["settings"], loaded)


if __name__ == "__main__":
    unittest.main()
