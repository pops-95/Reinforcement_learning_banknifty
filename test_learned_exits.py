"""Focused synthetic regressions for selective momentum trading; no historical files."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from banknifty_env import BankNiftyEnv, POSITION_FEATURES, validate_config


def synthetic_files(directory, start="2026-06-01 10:00", count=60, days=2):
    root = Path(directory)
    observations, candles = [], []
    for day in range(days):
        times = pd.date_range(pd.Timestamp(start)+pd.Timedelta(days=day), periods=count, freq="min")
        for ts in times:
            observations.append(dict(timestamp=ts, expiry=pd.Timestamp("2026-06-30"),
                split="validation", signal=0.0, return_3m=0.001, acceleration=0.0, momentum_3=0.001,
                open=50000.+(ts-times[0]).total_seconds()/60*5,
                close=50004.+(ts-times[0]).total_seconds()/60*5,
                high=50006.+(ts-times[0]).total_seconds()/60*5,
                low=49998.+(ts-times[0]).total_seconds()/60*5,
                atm_ce_symbol="CE", atm_pe_symbol="PE"))
            for symbol in ("CE", "PE", "OTHER"):
                candles.append(dict(timestamp=ts, groww_symbol=symbol, open=100., high=105., low=95., close=100., delta=0.5, iv=0.2,
                                    strike=50000., option_type="PE" if symbol == "PE" else "CE", oi=1000., volume=100.))
    pd.DataFrame(observations).to_parquet(root/"obs.parquet", index=False)
    pd.DataFrame(candles).to_parquet(root/"banknifty_options_2026-06-30.parquet", index=False)
    (root/"manifest.json").write_text(json.dumps({"observation_columns":["signal","return_3m","acceleration","momentum_3"]}))
    return dict(observations_file=root/"obs.parquet", option_market_dir=root, manifest_file=root/"manifest.json")


class MomentumTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = synthetic_files(self.temp.name)

    def env(self, **kwargs):
        kwargs.setdefault("reference_strategy_enabled", False)
        env = BankNiftyEnv(**self.paths, split="validation", hold_bonus_r=0, giveback_penalty_r=0, **kwargs)
        env.reset(seed=7)
        return env

    def quote(self, env, index, symbol="CE", **changes):
        key = (symbol, env.day_df.timestamp.iloc[index])
        env.option_lookup[key] = env.option_lookup[key]._replace(**changes)

    def feature(self, env, obs, name):
        return float(obs[len(env.feature_columns)+POSITION_FEATURES.index(name)])

    def test_minimum_hold_and_state_semantics(self):
        env = self.env()
        self.assertEqual(env.action_masks().tolist(), [True, True, True])
        env.step(1)
        self.assertEqual(env.position["bars_held"], 1)
        self.assertEqual(env.action_masks().tolist(), [True, False, False])
        env.step(1)  # exit ignored before three completed bars
        self.assertIsNotNone(env.position)
        env.step(2)  # cannot buy PE/reverse while holding CE
        self.assertEqual(env.position["symbol"], "CE")
        self.assertEqual(env.position["bars_held"], 3)
        self.assertEqual(env.action_masks().tolist(), [True, True, False])
        env.step(1)
        self.assertEqual(env.trade_log[0]["exit_reason"], "AGENT_EXIT")
        self.assertEqual(env.trade_log[0]["holding_minutes"], 3)

    def test_cooldown_and_opposite_entry(self):
        env = self.env()
        for a in [1,0,0,1]: env.step(a)
        self.assertEqual(env.cooldown_remaining(), 2)
        env.step(2)  # next open only one minute after exit
        self.assertIsNone(env.position)
        env.step(2)  # next open only two minutes after exit
        self.assertIsNone(env.position)
        self.assertEqual(env.action_masks().tolist(), [True, True, True])
        env.step(2)  # next open is three minutes after exit
        self.assertEqual(env.position["type"], "PE")
        self.assertEqual(env.position["entry_time"]-env.trade_log[0]["exit_time"], pd.Timedelta(minutes=3))

    def test_no_lookahead_and_fixed_symbol(self):
        env = self.env()
        masks = env.action_masks().copy()
        self.quote(env,1,open=110.,high=900.,close=500.)
        np.testing.assert_array_equal(masks,env.action_masks())
        env.step(1)
        self.assertAlmostEqual(env.position["entry_price"], 110*1.001)
        env.day_df["atm_ce_symbol"] = "OTHER"
        env.step(0)
        self.assertEqual(env.position["symbol"], "CE")
        # Next exit open wins over the subsequent candle's low or high.
        env.step(0)
        self.quote(env,4,open=120.,high=1000.,low=1.)
        env.step(1)
        self.assertEqual(env.trade_log[0]["exit_reason"], "AGENT_EXIT")
        self.assertAlmostEqual(env.trade_log[0]["exit_price"],120*.999)

    def test_r_accounting_both_long_premiums(self):
        for side in [1,2]:
            env=self.env()
            symbol="CE" if side==1 else "PE"
            rewards=[env.step(side)[1]]
            for i,close in [(2,110.),(3,120.)]:
                self.quote(env,i,symbol,close=close,high=close+2)
                rewards.append(env.step(0)[1])
            self.quote(env,4,symbol,open=115.)
            rewards.append(env.step(1)[1])
            trade=env.trade_log[0]
            expected=(115*.999-100*1.001-40/30)/30
            self.assertAlmostEqual(trade["R_return"],expected)
            self.assertAlmostEqual(sum(rewards),expected-0.03)
            self.assertAlmostEqual(sum(rewards),trade["reward"])
            self.assertEqual(env.step(0)[1],0)

    def test_risk_optional_atr_and_no_required_warmup(self):
        env=self.env()
        self.assertTrue(env.action_masks()[1])  # ATR isn't ready yet
        self.assertEqual(env._risk_distance(500,None),50)
        env.use_atr_stop=True
        self.assertEqual(env._risk_distance(500,None),50)
        self.assertEqual(env._risk_distance(500,60),72)

    def test_emergency_stop_during_minimum_hold(self):
        env=self.env()
        self.quote(env,1,low=60.)
        env.step(1)
        self.assertIsNone(env.position)
        self.assertEqual(env.trade_log[0]["exit_reason"],"EMERGENCY_STOP")
        self.assertEqual(env.trade_log[0]["bars_held"],0)
        self.assertAlmostEqual(env.episode_reward,env.trade_log[0]["R_return"]-0.03)

    def test_emergency_gap(self):
        env=self.env()
        env.step(1)
        self.quote(env,2,open=60.)
        env.step(0)
        self.assertEqual(env.trade_log[0]["exit_reason"],"EMERGENCY_GAP")
        self.assertAlmostEqual(env.trade_log[0]["exit_price"],60*.999)

    def test_mfe_mae_and_no_profit_target(self):
        env=self.env()
        env.step(1)
        self.quote(env,2,high=150.,low=85.,close=140.)
        obs,*_=env.step(0)
        self.assertIsNotNone(env.position)
        self.assertAlmostEqual(self.feature(env,obs,"MFE_option_points"),49.9,places=4)
        self.assertAlmostEqual(self.feature(env,obs,"MAE_option_points"),-15.1,places=4)
        self.assertEqual(self.feature(env,obs,"has_reached_40_points"),1)
        self.assertAlmostEqual(self.feature(env,obs,"MFE_R"),49.9/30,places=4)

    def test_wait_is_free_and_gate_off_by_default(self):
        env=self.env()
        self.assertFalse(env.tradeability_gate_enabled)
        self.assertEqual(env.step(0)[1],0)
        env.tradeability_gate_enabled=True
        env.tradeability_threshold=1.0
        self.assertEqual(env.action_masks().tolist(),[True,False,False])

    def test_entry_penalty_and_optional_frequency_penalty(self):
        env=self.env(overtrading_penalty_enabled=True,free_trades_per_day=1)
        for a in [1,0,0,1,0,0,2,0,0,1]:env.step(a)
        self.assertAlmostEqual(env.trade_log[0]["entry_penalty_r"],0.03)
        self.assertAlmostEqual(env.trade_log[1]["entry_penalty_r"],0.04)

    def test_min_hold_counts_complete_quotes_and_pending_exit(self):
        env=self.env()
        env.step(1)
        del env.option_lookup["CE",env.day_df.timestamp.iloc[2]]
        env.step(0)
        self.assertEqual(env.position["bars_held"],1)
        env.step(0);env.step(0)
        del env.option_lookup["CE",env.day_df.timestamp.iloc[5]]
        env.step(1)
        self.assertTrue(env.position["pending_exit"])
        self.assertEqual(env.action_masks().tolist(),[True,False,False])
        env.step(0)
        self.assertEqual(env.trade_log[0]["exit_reason"],"AGENT_EXIT")

    def test_time_limit_and_stale_end(self):
        env=self.env(max_hold_minutes=3)
        for a in [1,0,0,0]:env.step(a)
        self.assertEqual(env.trade_log[0]["exit_reason"],"TIME_LIMIT")
        env=self.env()
        env.day_df=env.day_df.iloc[:3]
        env.step(1)
        del env.option_lookup["CE",env.day_df.timestamp.iloc[2]]
        env.step(0)
        self.assertEqual(env.trade_log[0]["exit_reason"],"STALE_OPTION_DATA")
        self.assertIsNone(env.position)

    def test_chronological_reset_seed_and_finite_random_episode(self):
        env=self.env()
        first=env.current_day
        rng=np.random.default_rng(123)
        total=0
        while not env._terminated:
            action=int(rng.choice(np.flatnonzero(env.action_masks())))
            obs,reward,*_=env.step(action)
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(reward))
            total+=reward
        self.assertAlmostEqual(total,sum(t["reward"] for t in env.trade_log))
        env.reset()
        self.assertGreater(env.current_day,first)
        env.reset(seed=7)
        self.assertEqual(env.current_day,first)

    def test_shaping_is_small_bounded_and_no_negative_hold_bonus(self):
        env=self.env()
        env.hold_bonus_r=.005
        env.max_shaping_r_per_trade=.01
        env.step(1)
        for i in range(2,8):
            self.quote(env,i,close=110.,high=112.)
            env.step(0)
        self.assertLessEqual(env.position["shaping_used"],.01)
        self.assertAlmostEqual(env.position["shaping_r"],.01)
        env.position["last_price"]=90
        self.assertLessEqual(env._shape_hold(),0)

    def test_square_off_and_cutoff(self):
        paths=synthetic_files(self.temp.name,start="2026-06-01 14:58",count=29)
        env=BankNiftyEnv(**paths,split="validation",reference_strategy_enabled=False)
        env.reset();env.step(1)
        while not env._terminated:env.step(0)
        self.assertEqual(env.trade_log[0]["exit_reason"],"EOD")
        self.assertEqual(env.trade_log[0]["exit_detail"],"SQUARE_OFF")
        self.assertEqual(str(env.trade_log[0]["exit_time"]),"2026-06-01 15:25:00")

    def test_invalid_configuration(self):
        for values in ({"stop_premium_pct":1},{"hold_bonus_r":.5},{"trade_quantity":0},
                       {"min_hold_minutes":46},{"overtrading_penalty_enabled":"false"}):
            with self.assertRaises(ValueError):validate_config(values)


if __name__=="__main__":unittest.main()
