"""User-run regressions for autonomous trading; synthetic data, no long training."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from banknifty_env import BankNiftyEnv, BUY_CE, BUY_PE, HOLD, EXIT, WAIT, POSITION_FEATURES
from prepare_rl_data import option_features, build
from rl_data import load_dataset, validate_splits


def fixture(root):
    times = pd.date_range("2026-06-01 10:00", periods=8, freq="min")
    frame = pd.DataFrame(dict(timestamp=times, expiry=pd.Timestamp("2026-06-30"),
                             split="validation", open=50000., high=50005., low=49995., close=50000.,
                             signal=-9., return_3m=-.1, return_5m=-.2,
                             atm_ce_symbol="CE", atm_pe_symbol="PE",
                             ce_atm_log_oi=0., ce_atm_log_oi__available=0.))
    candles = pd.DataFrame([dict(timestamp=t, groww_symbol=s, option_type=s, strike=50000.,
                                 strike_step=100., open=100., high=105., low=95., close=100.,
                                 volume=np.nan, oi=np.nan, iv=.2, delta=.5 if s == "CE" else -.5)
                            for s in ("CE", "PE") for t in times])
    frame.to_parquet(root / "obs.parquet", index=False)
    candles.to_parquet(root / "banknifty_options_2026-06-30.parquet", index=False)
    (root / "manifest.json").write_text(json.dumps(dict(observation_columns=[
        "signal", "return_3m", "return_5m", "ce_atm_log_oi", "ce_atm_log_oi__available"])))
    return dict(observations_file=root / "obs.parquet", option_market_dir=root,
                manifest_file=root / "manifest.json"), frame, candles


class AutonomousTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths, self.frame, self.candles = fixture(self.root)

    def env(self, **kwargs):
        settings = dict(reference_features_enabled=False, slippage_pct=0., transaction_costs_enabled=False)
        settings.update(kwargs)
        env = BankNiftyEnv(**self.paths, split="validation", random_day=False, **settings)
        env.reset()
        self.addCleanup(env.close)
        return env

    def test_both_directions_allowed_despite_negative_momentum_and_missing_oi(self):
        env = self.env()
        self.assertEqual(env.action_masks().tolist(), [True, True, True, False, False])
        env.day_df["reference_side"] = "PE"
        env.day_df["reference_bull"] = 0.
        env.day_df["reference_bear"] = 1.
        self.assertTrue(env.action_masks()[BUY_CE])
        self.assertTrue(env.action_masks()[BUY_PE])

    def test_exit_is_learned_and_contract_does_not_follow_atm(self):
        env = self.env()
        rewards = [env.step(BUY_CE)[1]]
        self.assertEqual(env.action_masks().tolist(), [False, False, False, True, True])
        env.day_df["atm_ce_symbol"] = "DIFFERENT_CONTRACT"
        key = ("CE", env.day_df.timestamp.iloc[2])
        env.option_lookup[key] = env.option_lookup[key]._replace(open=110., high=115., low=105., close=110.)
        rewards.append(env.step(EXIT)[1])
        self.assertIsNone(env.position)
        self.assertEqual(env.trade_log[0]["symbol"], "CE")
        self.assertEqual(env.trade_log[0]["exit_reason"], "AGENT_EXIT")
        self.assertAlmostEqual(env.trade_log[0]["option_pnl_points"], 10.)
        self.assertAlmostEqual(sum(rewards), env.trade_log[0]["reward"])

    def test_pe_profit_and_loss_use_option_premium(self):
        for exit_price in (110., 90.):
            env = self.env()
            env.step(BUY_PE)
            key = ("PE", env.day_df.timestamp.iloc[2])
            env.option_lookup[key] = env.option_lookup[key]._replace(open=exit_price)
            env.step(EXIT)
            self.assertAlmostEqual(env.trade_log[0]["option_pnl_points"], exit_price - 100.)

    def test_optional_gate_and_missing_policy_are_explicit(self):
        env = self.env(momentum_gate_enabled=True)
        self.assertFalse(env.action_masks()[BUY_CE])
        self.assertTrue(env.action_masks()[BUY_PE])
        env = self.env(option_liquidity_gate_enabled=True, missing_filter_values_pass=True)
        self.assertTrue(env.action_masks()[BUY_CE])
        env = self.env(option_liquidity_gate_enabled=True, missing_filter_values_pass=False)
        self.assertFalse(env.action_masks()[BUY_CE])

    def test_feature_disable_does_not_disable_trading(self):
        env = self.env(features_momentum_enabled=False, disabled_features="signal")
        obs = env._get_observation()
        self.assertTrue(np.isfinite(obs).all())
        for feature in ("signal", "return_3m", "return_5m"):
            self.assertEqual(obs[env.feature_columns.index(feature)], 0.)
        self.assertTrue(env.action_masks()[BUY_CE])

    def test_open_slippage_does_not_use_future_candle_volume_or_range(self):
        first = self.env(slippage_pct=.001)
        second = self.env(slippage_pct=.001)
        key = ("CE", second.day_df.timestamp.iloc[1])
        second.option_lookup[key] = second.option_lookup[key]._replace(
            volume=1e9, oi=1e9, atr=900., high=900., close=500.)
        np.testing.assert_array_equal(first.action_masks(), second.action_masks())
        first.step(BUY_CE)
        second.step(BUY_CE)
        self.assertAlmostEqual(first.position["entry_price"], second.position["entry_price"])

    def test_missing_quote_cannot_be_bought(self):
        env = self.env()
        env.option_lookup.pop(("CE", env.day_df.timestamp.iloc[0]))
        self.assertFalse(env.action_masks()[BUY_CE])
        self.assertTrue(env.action_masks()[BUY_PE])
        obs, reward, _, _, info = env.step(BUY_CE)
        self.assertTrue(info["invalid_action"])
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(reward, 0.)

    def test_emergency_stop_is_optional(self):
        for enabled in (True, False):
            env = self.env(emergency_stop_enabled=enabled)
            key = ("CE", env.day_df.timestamp.iloc[1])
            env.option_lookup[key] = env.option_lookup[key]._replace(low=50.)
            env.step(BUY_CE)
            self.assertEqual(env.position is None, enabled)

    def test_daily_profit_can_accumulate_across_trades_and_resets_next_day(self):
        env = self.env(daily_profit_limit_enabled=True, daily_profit_target_points=40.)
        for index in (2, 4):
            key = ("CE", env.day_df.timestamp.iloc[index])
            env.option_lookup[key] = env.option_lookup[key]._replace(open=120.)
        env.step(BUY_CE)
        env.step(EXIT)
        self.assertTrue(env.action_masks()[BUY_CE])
        env.step(BUY_CE)
        _, _, _, _, info = env.step(EXIT)
        self.assertEqual(info["daily_net_points"], 40.)
        self.assertEqual(info["daily_limit_reason"], "DAILY_PROFIT_TARGET")
        self.assertEqual(env.action_masks().tolist(), [True, False, False, False, False])
        env.reset()
        self.assertEqual(env.daily_limit_reason, "")
        self.assertTrue(env.action_masks()[BUY_CE])

    def test_daily_loss_budget_accumulates_smaller_losses(self):
        env = self.env(daily_loss_limit_enabled=True, daily_loss_limit_points=20.)
        for index in (2, 4):
            key = ("PE", env.day_df.timestamp.iloc[index])
            env.option_lookup[key] = env.option_lookup[key]._replace(open=90.)
        env.step(BUY_PE)
        env.step(EXIT)
        self.assertTrue(env.action_masks()[BUY_PE])
        env.step(BUY_PE)
        env.step(EXIT)
        self.assertEqual(env._daily_net_points(), -20.)
        self.assertEqual(env.daily_limit_reason, "DAILY_LOSS_LIMIT")
        self.assertFalse(env.action_masks()[BUY_PE])

    def test_daily_mark_trigger_fills_next_open_not_trigger_price(self):
        env = self.env(daily_profit_limit_enabled=True, emergency_stop_enabled=False)
        key = ("CE", env.day_df.timestamp.iloc[1])
        env.option_lookup[key] = env.option_lookup[key]._replace(high=150., close=145.)
        key = ("CE", env.day_df.timestamp.iloc[2])
        env.option_lookup[key] = env.option_lookup[key]._replace(open=135.)
        env.step(BUY_CE)
        self.assertTrue(env.position["pending_exit"])
        env.step(HOLD)
        self.assertEqual(env.trade_log[0]["exit_reason"], "DAILY_PROFIT_TARGET")
        self.assertEqual(env.trade_log[0]["option_pnl_points"], 35.)
        self.assertFalse(env.action_masks()[BUY_CE])  # Sticky even after an adverse fill.

    def test_realized_only_daily_budget_does_not_override_open_position(self):
        env = self.env(daily_profit_limit_enabled=True, close_on_daily_limit_enabled=False)
        key = ("CE", env.day_df.timestamp.iloc[1])
        env.option_lookup[key] = env.option_lookup[key]._replace(high=150., close=145.)
        env.step(BUY_CE)
        self.assertFalse(env.position["pending_exit"])
        self.assertEqual(env.daily_limit_reason, "")

    def test_daily_target_counts_costs_not_gross_points(self):
        env = self.env(daily_profit_limit_enabled=True, transaction_costs_enabled=True)
        key = ("CE", env.day_df.timestamp.iloc[2])
        env.option_lookup[key] = env.option_lookup[key]._replace(open=140.)
        env.step(BUY_CE)
        env.step(EXIT)
        self.assertLess(env._daily_net_points(), 40.)
        self.assertEqual(env.daily_limit_reason, "")


class DataTests(unittest.TestCase):
    def test_missing_oi_not_forward_filled_and_returns_do_not_cross_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, candles = fixture(Path(directory))
        candles.loc[candles.index[0], "oi"] = 1000.
        # Remove one intermediate CE candle: its 1-minute return must be unavailable.
        candles = candles.drop(index=1)
        result = option_features(candles)
        ce = result.loc[result.groww_symbol == "CE"].reset_index(drop=True)
        self.assertTrue(pd.isna(ce.loc[1, "oi"]))
        self.assertTrue(pd.isna(ce.loc[1, "oi_change"]))
        self.assertTrue(pd.isna(ce.loc[1, "option_ret_1m"]))

    def test_split_leakage_rejected(self):
        frame = pd.DataFrame(dict(timestamp=pd.to_datetime(["2026-01-01", "2026-01-01 10:00", "2026-02-01"]),
                                  split=["train", "validation", "test"]))
        with self.assertRaises(ValueError):
            validate_splits(frame)

    def test_preparation_preserves_splits_flags_missing_oi_and_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, base, candles = fixture(root)
            frames, options = [], []
            for days, split in enumerate(("train", "validation", "test")):
                frame = base.copy()
                frame["timestamp"] += pd.Timedelta(days=days)
                frame["split"] = split
                option = candles.copy()
                option["timestamp"] += pd.Timedelta(days=days)
                frames.append(frame)
                options.append(option)
            source = pd.concat(frames, ignore_index=True)
            source.to_parquet(root / "obs.parquet", index=False)
            pd.concat(options, ignore_index=True).to_parquet(root / "banknifty_options_2026-06-30.parquet", index=False)
            underlying = source[["timestamp", "open", "high", "low", "close", "return_3m"]]
            underlying.to_parquet(root / "underlying.parquet", index=False)
            with patch("prepare_rl_data.OPTIONS", root):
                meta = build(root / "obs.parquet", root / "underlying.parquet", root / "prepared")
            df, loaded = load_dataset(root / "prepared/observations.parquet", root / "prepared/manifest.json", root)
            self.assertEqual(meta["dataset_id"], loaded["dataset_id"])
            self.assertEqual(df.split.tolist(), source.split.tolist())
            self.assertTrue(np.isfinite(df[meta["observation_columns"]].to_numpy()).all())
            self.assertTrue(df.ce_atm_log_oi.eq(0).all())
            self.assertTrue(df.ce_atm_log_oi__available.eq(0).all())
            self.assertTrue(df.ce_atm_available.eq(1).all())


class WebConfigurationTests(unittest.TestCase):
    def test_web_defaults_are_autonomous_and_settings_round_trip(self):
        import banknifty_rl_web as web
        config = web.parse_training_config({})
        self.assertFalse(config["environment"]["reference_strategy_enabled"])
        self.assertTrue(config["environment"]["learned_exit_enabled"])
        payload = dict(format="banknifty_training_settings", version=1,
                       environment_version=web.ENV_VERSION,
                       training={k: v for k, v in config.items() if k != "environment"},
                       environment=config["environment"])
        response = web.app.test_client().post("/api/settings/validate", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["settings"]["environment"], config["environment"])
        html = web.app.test_client().get("/")
        self.assertEqual(html.status_code, 200)
        self.assertIn(b"AUTONOMOUS DEFAULTS", html.data)

    def test_cuda_requested_has_no_silent_cpu_fallback(self):
        import banknifty_rl_web as web
        with patch.object(web.th.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA requested"):
                web.training_device(dict(web.TRAIN_DEFAULTS, cuda_enabled=True))

    def test_fractional_integer_and_unknown_config_are_rejected(self):
        import banknifty_rl_web as web
        for payload in ({"seed": 1.5}, {"n_steps": True}, {"not_a_setting": 1}):
            with self.assertRaises(ValueError):
                web.parse_training_config(payload)

    def test_daily_preset_is_complete_and_web_loadable(self):
        import banknifty_rl_web as web
        path = Path(__file__).parent / "settings/banknifty_daily40_loss20_autonomous.json"
        payload = json.loads(path.read_text())
        response = web.app.test_client().post("/api/settings/validate", json=payload)
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(set(payload["environment"]), set(web.ENV_DEFAULTS))
        # Older exports acquire new algorithm/DQN defaults without losing values.
        self.assertTrue(set(payload["training"]).issubset(web.TRAIN_DEFAULTS))
        self.assertTrue(response.json["settings"]["training"]["daily_targets_enabled"])

    def test_daily_validation_includes_no_trade_days_and_loss_breaches(self):
        import banknifty_rl_web as web
        cfg = dict(web.TRAIN_DEFAULTS, daily_targets_enabled=True)
        metrics = dict(total_trades=50, total_trading_days=2, trades_per_day=25,
                       win_rate=80., profit_factor=2., average_R=1., daily_net_points=[40., 0.])
        result = web.validation_target_results(metrics, cfg)
        self.assertEqual(result["daily_target_hit_rate_pct"], 50.)
        self.assertFalse(result["checks"]["daily_profit_target"])
        self.assertFalse(result["targets_met"])
        metrics["daily_net_points"] = [40., -21.]
        self.assertFalse(web.validation_target_results(metrics, cfg)["checks"]["daily_loss_budget"])
        metrics["daily_net_points"] = [40., 45.]
        self.assertTrue(web.validation_target_results(metrics, cfg)["targets_met"])


if __name__ == "__main__":
    unittest.main()
