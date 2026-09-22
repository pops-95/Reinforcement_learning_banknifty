#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
import threading
import time
import traceback
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
import torch as th
import numpy as np
import pandas as pd


from flask import Flask, jsonify, request, render_template_string
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from banknifty_env import (
    BankNiftyEnv, ENV_VERSION, ENV_DEFAULTS, ACTION_NAMES,
    POSITION_FEATURES, validate_config,
    WAIT, BUY_CE, BUY_PE, HOLD, EXIT,
)
from build_candidate_dataset import (
    build_candidate_dataset, CandidateCancelled, CANDIDATE_ENV_DEFAULTS,
)
from train_candidate_model import (
    CANDIDATE_TRAIN_DEFAULTS, parse_candidate_training, train_candidate_model,
    evaluate_candidate_model, trade_frequency_result, MODEL_DIR as CANDIDATE_MODEL_DIR,
)
from reference_strategy import CANDIDATE_VERSION
from rl_data import OBSERVATIONS, MANIFEST as RL_MANIFEST, OPTIONS, load_dataset

BASE = Path(__file__).resolve().parent
OBS, OPT_DIR, MANIFEST = OBSERVATIONS, OPTIONS, RL_MANIFEST
MODEL_DIR = BASE/"models"
CKPT_DIR = MODEL_DIR/"checkpoints"
MODEL_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# The existing Seed field remains the base seed.  Three deterministic seeds are
# derived internally; the user's total-timestep budget is divided among them.
MULTI_SEED_OFFSETS = (0, 17, 41)

TRAIN_DEFAULTS = dict(
    algorithm="ppo",
    dqn_buffer_size=50000, dqn_learning_starts=10000,
    dqn_train_freq=4, dqn_gradient_steps=1, dqn_target_update_interval=2000,
    dqn_tau=1.0, dqn_exploration_fraction=0.30,
    dqn_initial_epsilon=1.0, dqn_final_epsilon=0.05, dqn_double_enabled=True,
    total_timesteps=1000000, learning_rate=0.0003, n_steps=2048, batch_size=512, n_epochs=5,
    gamma=0.995, gae_lambda=0.95, clip_range=0.20, ent_coef=0.01, vf_coef=0.5,
    max_grad_norm=0.5, seed=42,
    multi_seed_enabled=False, num_seeds=3, seed_stride=17,
    normalize_obs_enabled=True, normalize_reward_enabled=True, clip_obs=10.0, clip_reward=10.0,
    cuda_enabled=True, checkpoint_enabled=True, checkpoint_freq=100000,
    cuda_device=0, cuda_tf32_enabled=False, cpu_threads=4,
    target_kl_enabled=True, target_kl=0.03,
    tensorboard_enabled=True, benchmark_reference_enabled=False,
    target_win_rate_pct=70.0, target_profit_factor=1.5, target_average_r=0.0,
    daily_targets_enabled=False, target_daily_points=40.0,
    target_daily_hit_rate_pct=100.0, max_daily_loss_points=20.0,
    min_validation_trades=50, min_trades_per_day=0.0,
    pi_layer1=512, pi_layer2=256, pi_layer3=128,
    vf_layer1=512, vf_layer2=256, vf_layer3=128,
)


def summarize_trades(trades, days=0):
    """Trading-only metrics; shaping and entry penalties never count as profit."""
    def avg(values):
        return float(np.mean(values)) if values else 0.0

    points = [float(t["option_pnl_points"]) for t in trades]
    rs = [float(t["R_return"]) for t in trades]
    winners = [x for x in points if x > 0]
    losers = [x for x in points if x < 0]
    gains, losses = sum(winners), -sum(losers)
    curve = np.r_[0.0, np.cumsum(rs)]
    holdings = [float(t["holding_minutes"]) for t in trades]
    mfe = [float(t["MFE_option_points"]) for t in trades]
    mae = [float(t["MAE_option_points"]) for t in trades]

    reasons = {}
    for trade in trades:
        reasons[trade["exit_reason"]] = reasons.get(trade["exit_reason"], 0) + 1

    return dict(
        total_trades=len(trades),
        total_trading_days=days,
        net_option_points=sum(points),
        average_daily_points=sum(points)/days if days else 0.0,
        trades_per_day=len(trades)/days if days else 0,
        ce_trades=sum(t["side"] == "CE" for t in trades),
        pe_trades=sum(t["side"] == "PE" for t in trades),
        wins=len(winners),
        losses=len(losers),
        breakeven_trades=len(trades)-len(winners)-len(losers),
        win_rate=100*len(winners)/len(trades) if trades else 0,
        average_winner_points=avg(winners),
        average_loser_points=avg(losers),
        payoff_ratio=(avg(winners)/abs(avg(losers))) if losers and avg(losers) != 0 else None,
        trades_ge_1_5R_pct=100*sum(float(t.get("R_return", 0)) >= 1.5 for t in trades)/len(trades) if trades else 0,
        average_R=avg(rs),
        median_R=float(np.median(rs)) if rs else 0,
        cumulative_R=sum(rs),
        max_drawdown_R=float(np.max(np.maximum.accumulate(curve)-curve)),
        profit_factor=gains/losses if losses else None,
        profit_factor_note="no losing trades" if not losses else "net premium points; excludes reward shaping",
        average_holding_minutes=avg(holdings),
        median_holding_minutes=float(np.median(holdings)) if holdings else 0,
        average_MFE_points=avg(mfe),
        average_MAE_points=avg(mae),
        MFE_MAE_ratio=avg(mfe)/abs(avg(mae)) if avg(mae) != 0 else None,
        reached_40_points_pct=100*sum(t["reached_40_points"] for t in trades)/len(trades) if trades else 0,
        exit_counts=reasons,
        stale_exits=sum(t["stale_exit"] for t in trades),
    )


def regime_breakdown(trades):
    """Per-regime out-of-sample trading metrics."""
    buckets = {}
    for trade in trades:
        buckets.setdefault(trade.get("regime", "UNKNOWN"), []).append(trade)
    return {
        regime: summarize_trades(items, 0)
        for regime, items in sorted(buckets.items())
    }


def aggregate_candidate_diagnostics(days):
    if not days:
        return {}
    count_keys = (
        "bullish_opportunities", "bearish_opportunities",
        "bullish_captured", "bearish_captured",
        "bullish_candidates", "bearish_candidates",
        "bullish_candidates_successful", "bearish_candidates_successful",
    )
    out = {k: int(sum(int(d.get(k, 0) or 0) for d in days)) for k in count_keys}
    out["horizon_minutes"] = int(days[0].get("horizon_minutes", 15))
    opportunities = out["bullish_opportunities"] + out["bearish_opportunities"]
    captured = out["bullish_captured"] + out["bearish_captured"]
    candidates = out["bullish_candidates"] + out["bearish_candidates"]
    success = out["bullish_candidates_successful"] + out["bearish_candidates_successful"]
    out["candidate_recall_pct"] = 100*captured/opportunities if opportunities else None
    out["candidate_precision_pct"] = 100*success/candidates if candidates else None
    return out


def validation_target_results(metrics, training_cfg=None, complete=True):
    cfg = training_cfg or TRAIN_DEFAULTS
    pf = metrics.get("profit_factor")
    frequency = trade_frequency_result(metrics, float(cfg.get("min_trades_per_day", 0.0)))
    checks = dict(
        evaluation_complete=bool(complete),
        enough_trades=int(metrics.get("total_trades", 0)) >= int(cfg.get("min_validation_trades", 50)),
        trades_per_day=frequency["meets_minimum"],
        win_rate=float(metrics.get("win_rate", 0)) >= float(cfg.get("target_win_rate_pct", 70)),
        # No losses means PF is undefined: report it, never invent a passing PF.
        profit_factor=pf is not None and np.isfinite(pf) and pf >= float(cfg.get("target_profit_factor", 1.5)),
        average_r=float(metrics.get("average_R", 0)) >= float(cfg.get("target_average_r", 0)),
    )
    daily = metrics.get("daily_net_points", [])
    target_points = float(cfg.get("target_daily_points", 40.0))
    hit_rate = 100 * sum(p >= target_points for p in daily) / len(daily) if daily else 0.0
    if cfg.get("daily_targets_enabled", False):
        checks["daily_profit_target"] = bool(daily) and hit_rate >= float(cfg.get("target_daily_hit_rate_pct", 100.0))
        checks["daily_loss_budget"] = bool(daily) and min(daily) >= -float(cfg.get("max_daily_loss_points", 20.0))
    return dict(checks=checks, targets_met=bool(all(checks.values())),
                minimum_trades=int(cfg.get("min_validation_trades", 50)),
                daily_target_hit_rate_pct=hit_rate,
                daily_target_points=target_points,
                trade_frequency=frequency)


def validation_selection_score(metrics, training_cfg=None):
    """Select on validation expectancy/PF/DD with explicit aspirational targets."""
    training_cfg = training_cfg or TRAIN_DEFAULTS
    n = int(metrics.get("total_trades", 0))
    if n == 0:
        return -1e9
    avg_r = float(metrics.get("average_R", 0.0))
    wr = float(metrics.get("win_rate", 0.0))
    pf = metrics.get("profit_factor")
    pf = float(pf) if pf is not None and np.isfinite(pf) else 4.0
    pf = float(np.clip(pf, 0.10, 4.0))
    dd = float(metrics.get("max_drawdown_R", 0.0))

    target_wr = float(training_cfg.get("target_win_rate_pct", 70.0))
    target_pf = float(training_cfg.get("target_profit_factor", 1.5))
    target_r = float(training_cfg.get("target_average_r", 0.0))

    score = avg_r + 0.10*np.log(pf) - 0.05*(dd/max(np.sqrt(n), 1.0))
    score -= 0.004*max(0.0, target_wr-wr)
    score -= 0.12*max(0.0, target_pf-pf)
    score -= 0.50*max(0.0, target_r-avg_r)
    if training_cfg.get("daily_targets_enabled", False):
        daily = metrics.get("daily_net_points", [])
        target_points = float(training_cfg.get("target_daily_points", 40.0))
        hit_rate = 100 * sum(p >= target_points for p in daily) / len(daily) if daily else 0.0
        score -= 0.01 * max(0.0, float(training_cfg.get("target_daily_hit_rate_pct", 100.0)) - hit_rate)
        if daily:
            score -= 0.01 * max(0.0, -min(daily) - float(training_cfg.get("max_daily_loss_points", 20.0)))
    minimum = max(1, int(training_cfg.get("min_validation_trades", 50)))
    if n < minimum:
        score -= (minimum-n)/minimum
    frequency = trade_frequency_result(metrics, float(training_cfg.get("min_trades_per_day", 0.0)))
    if frequency["minimum"] > 0:
        score -= 0.25 * max(0.0, frequency["minimum"]-frequency["actual"]) / frequency["minimum"]
    return float(score)


def validation_candidate_key(candidate, training_cfg):
    # Preserve the old fallback order unless the frequency minimum is enabled.
    checks = candidate["targets"]["checks"]
    eligible = (checks["enough_trades"] and checks["trades_per_day"]
                if float(training_cfg.get("min_trades_per_day", 0.0)) > 0 else True)
    return candidate["targets"]["targets_met"], eligible, candidate["score"]


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.thread = None
        self._trade_file = None
        self.reset(0)
        self.gpu = th.cuda.get_device_name(0) if th.cuda.is_available() else "CPU"
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        self.torch, self.cuda = th.__version__, str(th.version.cuda)

    def reset(self, target):
        with self.lock:
            self.close_log()
            self.status = "idle" if target == 0 else "starting"
            self.error = ""
            self.target, self.steps, self.progress = int(target), 0, 0.0
            self.started = time.time() if target else None
            self.fps, self.stop = 0.0, False
            self.current_day = self.expiry = self.model_path = self.vec_path = ""
            self.episodes = self.day_episodes = 0
            self._day_pending = False
            self.last_ep_reward = 0.0
            self.ep_rewards = deque(maxlen=200)
            self.all_trades = []
            self.action_counts = dict.fromkeys(ACTION_NAMES, 0)
            self.invalid_actions = self.rejected_entries = 0
            self.entry_diagnostics = dict(flat_steps=0, buy_available_steps=0,
                                          blocked_steps=0, voluntary_wait_steps=0,
                                          entries_opened=0, blocked_by={}, last_blocks={})
            self.evaluation = None
            self.optimizer = {}
            self.history = deque(maxlen=500)
            self.gpu_memory_mb = 0.0
            self.policy_device = "not initialized"
            self.log_path = ""
            self._csv_writer = None

    def begin_log(self, run_id):
        with self.lock:
            directory = MODEL_DIR / "trade_logs"
            directory.mkdir(exist_ok=True)
            self.log_path = str(directory / f"{run_id}.csv")
            self._trade_file = open(self.log_path, "w", newline="", encoding="utf-8")
            self._csv_writer = None

    def close_log(self):
        if self._trade_file is not None:
            self._trade_file.close()
            self._trade_file = None

    def start_day(self, day, expiry):
        with self.lock:
            self.current_day, self.expiry = day, expiry
            self._day_pending = True

    def record_action(self, name, invalid=False, rejected=False):
        with self.lock:
            if self._day_pending:
                self.day_episodes += 1
                self._day_pending = False
            self.action_counts[name] += 1
            self.invalid_actions += int(invalid)
            self.rejected_entries += int(rejected)

    def add_trade(self, trade):
        with self.lock:
            record = {
                k: str(v) if isinstance(v, (pd.Timestamp, datetime)) else v
                for k, v in trade.items()
            }
            self.all_trades.append(record)
            if self._trade_file is not None:
                if self._csv_writer is None:
                    self._csv_writer = csv.DictWriter(
                        self._trade_file, fieldnames=list(record)
                    )
                    self._csv_writer.writeheader()
                self._csv_writer.writerow(record)
                self._trade_file.flush()

    def record_entry_decision(self, available, action, opened, blocks):
        # Count once per actual environment step, not each action-mask query.
        with self.lock:
            d = self.entry_diagnostics
            d["flat_steps"] += 1
            d["buy_available_steps"] += int(available)
            d["blocked_steps"] += int(not available)
            d["voluntary_wait_steps"] += int(available and action == "WAIT")
            d["entries_opened"] += int(opened)
            d["last_blocks"] = {side: list(reasons) for side, reasons in blocks.items()}
            for side, reasons in blocks.items():
                for reason in reasons:
                    key = f"{side}: {reason}"
                    d["blocked_by"][key] = d["blocked_by"].get(key, 0) + 1

    def episode(self, reward):
        with self.lock:
            self.episodes += 1
            self.last_ep_reward = float(reward)
            self.ep_rewards.append(float(reward))
            self.history.append(dict(episode=self.episodes, steps=self.steps,
                                     reward=float(reward), mean20=float(np.mean(list(self.ep_rewards)[-20:]))))

    def snapshot(self):
        with self.lock:
            metrics = summarize_trades(self.all_trades, self.day_episodes)
            if self.evaluation and self.status in ("completed", "stopped"):
                metrics.update(self.evaluation.get("trading_metrics", {}))
            total_actions = sum(self.action_counts.values())
            rates = {
                k: 100*v/total_actions if total_actions else 0
                for k, v in self.action_counts.items()
            }
            return dict(
                status=self.status, error=self.error, steps=self.steps,
                target=self.target, progress=self.progress, fps=round(self.fps, 1),
                episodes=self.episodes,
                elapsed=time.time()-self.started if self.started else 0,
                gpu=self.gpu, device=self.device, torch=self.torch, cuda=self.cuda,
                day=self.current_day, expiry=self.expiry,
                last_ep_reward=self.last_ep_reward,
                mean20=float(np.mean(list(self.ep_rewards)[-20:])) if self.ep_rewards else 0,
                metrics=metrics, action_counts=dict(self.action_counts),
                action_percentages=rates, invalid_actions=self.invalid_actions,
                rejected_entries=self.rejected_entries,
                entry_diagnostics={**self.entry_diagnostics,
                                   "blocked_by": dict(self.entry_diagnostics["blocked_by"]),
                                   "last_blocks": {k: list(v) for k, v in self.entry_diagnostics["last_blocks"].items()}},
                trades=list(reversed(self.all_trades[-300:])),
                model_path=self.model_path, vec_path=self.vec_path,
                optimizer=dict(self.optimizer), history=list(self.history),
                gpu_memory_mb=self.gpu_memory_mb, policy_device=self.policy_device,
                log_path=self.log_path, stop=self.stop, evaluation=self.evaluation,
                **{k: metrics[k] for k in ("win_rate", "profit_factor", "stale_exits", "exit_counts")}
            )


STATE = State()
EVAL_STATE = State()
JOB_LOCK = threading.RLock()
BUSY = {"starting", "training", "validating", "saving", "stopping"}
CANDIDATE_JOB = dict(status="idle", stop=False, error="", kind="", progress={},
                     result=None, model_path="", thread=None)


def jobs_active():
    return (any(s.status in BUSY or (s.thread is not None and s.thread.is_alive())
                for s in (STATE, EVAL_STATE))
            or CANDIDATE_JOB["status"] in BUSY
            or (CANDIDATE_JOB["thread"] is not None and CANDIDATE_JOB["thread"].is_alive()))


def candidate_job_worker(kind, cfg, candidate_cfg, model=None, split="validation"):
    def cancel():
        with JOB_LOCK:
            if CANDIDATE_JOB["stop"]:
                raise CandidateCancelled()

    def progress(value):
        with JOB_LOCK:
            CANDIDATE_JOB["progress"] = value
            if not CANDIDATE_JOB["stop"]:
                CANDIDATE_JOB["status"] = "validating" if value.get("stage", "").startswith("evaluating") else "training"

    def saved(path):
        with JOB_LOCK:
            CANDIDATE_JOB["model_path"] = str(path)

    try:
        cancel()
        if kind == "build":
            result = build_candidate_dataset(cfg["environment"], check_cancel=cancel, progress=progress)
        elif kind == "evaluate":
            result = evaluate_candidate_model(model, split, check_cancel=cancel, progress=progress)
        else:
            _, result = train_candidate_model(
                kind, candidate_cfg=candidate_cfg, training_cfg={k: v for k, v in cfg.items() if k != "environment"},
                expected_config=cfg["environment"], check_cancel=cancel, progress=progress, saved=saved)
        with JOB_LOCK:
            CANDIDATE_JOB["result"] = result
            CANDIDATE_JOB["status"] = "stopped" if CANDIDATE_JOB["stop"] else "completed"
    except CandidateCancelled:
        with JOB_LOCK:
            CANDIDATE_JOB["status"] = "stopped"
    except Exception:
        with JOB_LOCK:
            CANDIDATE_JOB["status"] = "error"
            CANDIDATE_JOB["error"] = traceback.format_exc()


class TrainingStopped(Exception):
    """Cooperative cancellation after preserving the current policy."""


def check_training_stop():
    with STATE.lock:
        if STATE.stop:
            raise TrainingStopped()


def save_training_model(model, env, cfg, run_id, seed):
    """Publish the archive last so the picker only sees complete bundles."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    model_path = MODEL_DIR / f"banknifty_{cfg['algorithm']}_{stamp}.zip"
    vec_path = MODEL_DIR / f"vecnormalize_{stamp}.pkl"
    metadata = dict(
        algorithm=cfg["algorithm"], environment_version=ENV_VERSION, environment=cfg["environment"],
        training={k: v for k, v in cfg.items() if k != "environment"},
        selected_seed=seed, model_id=run_id, timesteps=int(model.num_timesteps),
        normalization_file=vec_path.name, validation_status="not_validated",
        trade_log=STATE.log_path,
        dataset_id=getattr(model, "market_dataset_id", None),
    )
    with tempfile.TemporaryDirectory(dir=MODEL_DIR, prefix=".saving_") as temp:
        staging = Path(temp)
        model.save(staging / model_path.name)
        env.save(staging / vec_path.name)
        staged_meta = staging / model_path.with_suffix(".json").name
        staged_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (staging / vec_path.name).replace(vec_path)
        staged_meta.replace(model_path.with_suffix(".json"))
        (staging / model_path.name).replace(model_path)
    with STATE.lock:
        STATE.model_path, STATE.vec_path = str(model_path), str(vec_path)
    return model_path, vec_path


def maskable_ppo_class():
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError(
            "Learned exits require sb3-contrib. Install in the Python environment "
            "running this app: python -m pip install sb3-contrib"
        ) from exc
    return MaskablePPO


def load_trading_model(path, device="cpu"):
    """Use the saved algorithm, never the current form, to load model weights."""
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("data"))
    algorithm = metadata.get("market_algorithm", "ppo")
    if algorithm == "dqn":
        from banknifty_dqn import BankNiftyDQN
        return BankNiftyDQN.load(path, device=device)
    if algorithm == "ppo":
        return maskable_ppo_class().load(path, device=device)
    raise ValueError(f"Unsupported saved algorithm: {algorithm}")


parse_env_config = validate_config


def training_device(cfg):
    th.set_num_threads(cfg["cpu_threads"])
    if not cfg["cuda_enabled"]:
        return "cpu"
    if not th.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Install a CUDA PyTorch build and compatible NVIDIA driver, or disable CUDA explicitly.")
    index = cfg["cuda_device"]
    if index >= th.cuda.device_count():
        raise ValueError(f"CUDA device {index} does not exist")
    th.cuda.set_device(index)
    th.backends.cuda.matmul.allow_tf32 = cfg["cuda_tf32_enabled"]
    th.backends.cudnn.allow_tf32 = cfg["cuda_tf32_enabled"]
    # Real allocation/computation catches unsupported GPU binaries/driver failures.
    probe = th.ones((16, 16), device=f"cuda:{index}")
    (probe @ probe).sum().item()
    th.cuda.synchronize(index)
    return f"cuda:{index}"


def make_market_env(env_cfg, split="train", state=None, random_day=True, model_id="unassigned"):
    return BankNiftyEnv(
        observations_file=OBS, option_market_dir=OPT_DIR,
        manifest_file=MANIFEST, split=split, state=state,
        random_day=random_day, model_id=model_id, **env_cfg
    )


class LiveCallback(BaseCallback):
    def __init__(self, state, base_steps=0, total_target=None):
        super().__init__()
        self.state = state
        self.t0 = None
        self.base_steps = int(base_steps)
        self.total_target = total_target

    def _on_training_start(self):
        self.t0 = time.time()
        with self.state.lock:
            self.state.status = "stopping" if self.state.stop else "training"

    def _on_step(self):
        elapsed = max(time.time()-self.t0, 1e-6)
        absolute_steps = self.base_steps + int(self.num_timesteps)
        with self.state.lock:
            self.state.steps = absolute_steps
            target = int(self.total_target or self.state.target or 1)
            self.state.progress = min(100.0, 100*absolute_steps/max(target, 1))
            self.state.fps = self.num_timesteps/elapsed
            stop = self.state.stop

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                self.state.episode(float(info.get("episode_reward", 0.0)))
        return not stop

    def _on_rollout_start(self):
        # DQN rollouts may be only four steps. Avoid logging/locking every rollout.
        if self.num_timesteps - getattr(self, "_last_capture", -256) >= 256:
            self._capture_optimizer()
            self._last_capture = self.num_timesteps

    def _on_training_end(self):
        self._capture_optimizer()

    def _capture_optimizer(self):
        with self.state.lock:
            self.state.optimizer = {
                key: float(value) for key, value in self.model.logger.name_to_value.items()
                if key.startswith("train/") and np.isscalar(value) and np.isfinite(value)
            }
            self.state.policy_device = str(next(self.model.policy.parameters()).device)
            if self.model.device.type == "cuda":
                self.state.gpu_memory_mb = th.cuda.memory_allocated(self.model.device) / (1024**2)


def _policy_action(model, vec, raw_env, obs):
    normalized = vec.normalize_obs(obs)
    masks = raw_env.action_masks()
    action, _ = model.predict(normalized, deterministic=True, action_masks=masks)
    if getattr(model, "market_algorithm", "ppo") == "dqn":
        # Q-values are expected returns, not calibrated action probabilities.
        raw_env.set_action_probability(None)
        return int(action)
    with th.no_grad():
        tensor, _ = model.policy.obs_to_tensor(normalized)
        distribution = model.policy.get_distribution(tensor, action_masks=masks)
        probability = distribution.distribution.probs[0, int(action)].item()
    raw_env.set_action_probability(probability)
    return int(action)


def evaluate_saved_policy(model_path, vec_path, env_cfg, split="validation",
                          state=None, reference_only=False, progress=False,
                          check_cancel=None):
    """Chronological evaluation for PPO or reference-only benchmark."""
    model = None
    vec = None
    if check_cancel is not None:
        check_cancel()
    raw_env = make_market_env(
        env_cfg, split=split, state=state, random_day=False,
        model_id=("reference_only" if reference_only else Path(model_path).name)
    )

    if not reference_only:
        model = load_trading_model(model_path, device="cpu")
        if (getattr(model, "market_dataset_id", None) != raw_env.dataset_id
                or getattr(model, "market_feature_columns", None) != raw_env.feature_columns):
            raw_env.close()
            raise ValueError("Model dataset/features differ from current prepared data. Restore that dataset or train a fresh model.")
        vec = VecNormalize.load(vec_path, DummyVecEnv([lambda: raw_env]))
        vec.training = False
        vec.norm_reward = False

    all_trades = []
    daily_results = []
    candidate_days = []
    finished_days = 0
    if state is not None and progress:
        with state.lock:
            state.target = len(raw_env.days)
            state.device = "cpu"
            state.policy_device = "cpu"

    try:
        for _ in raw_env.days:
            if check_cancel is not None:
                check_cancel()
            if state is not None:
                with state.lock:
                    if state.stop:
                        break

            obs, _ = raw_env.reset()
            while True:
                if check_cancel is not None:
                    check_cancel()
                if reference_only:
                    masks = raw_env.action_masks()
                    if raw_env.position is None:
                        if masks[BUY_CE]:
                            action = BUY_CE
                        elif masks[BUY_PE]:
                            action = BUY_PE
                        else:
                            action = WAIT
                    else:
                        action = HOLD
                else:
                    action = _policy_action(model, vec, raw_env, obs)

                obs, _, terminated, truncated, _ = raw_env.step(action)
                if terminated or truncated:
                    break

            if raw_env.position is not None:
                raise RuntimeError("Evaluation ended with an unrecorded position")

            all_trades.extend(raw_env.trade_log)
            diag = raw_env.candidate_diagnostics()
            if diag:
                candidate_days.append(diag)
            finished_days += 1
            daily_results.append(dict(date=str(raw_env.current_day),
                                      net_points=raw_env._daily_net_points(),
                                      net_rupees=raw_env._daily_net_points() * raw_env.trade_quantity,
                                      trades=len(raw_env.trade_log),
                                      limit_reason=raw_env.daily_limit_reason))

            if state is not None:
                state.episode(raw_env.episode_reward)
                with state.lock:
                    if progress:
                        state.steps = finished_days
                        state.progress = round(100*finished_days/len(raw_env.days), 2)

        metrics = summarize_trades(all_trades, finished_days)
        daily_points = [d["net_points"] for d in daily_results]
        metrics.update(daily_net_points=daily_points,
                       worst_day_points=min(daily_points) if daily_points else None,
                       best_day_points=max(daily_points) if daily_points else None,
                       profitable_days_pct=100 * sum(p > 0 for p in daily_points) / len(daily_points) if daily_points else 0.0,
                       average_daily_rupees=metrics["average_daily_points"] * raw_env.trade_quantity,
                       daily_results=daily_results)
        return dict(
            trades=all_trades,
            metrics=metrics,
            regimes=regime_breakdown(all_trades),
            candidate_diagnostics=aggregate_candidate_diagnostics(candidate_days),
            finished_days=finished_days,
            total_days=len(raw_env.days),
            partial=finished_days < len(raw_env.days),
            dataset_id=raw_env.dataset_id,
        )
    finally:
        if vec is not None:
            vec.close()
        else:
            raw_env.close()


def reference_benchmark(env_cfg, split, enabled, check_cancel=None):
    if not enabled:
        return dict(enabled=False, metrics=summarize_trades([], 0), regimes={})
    result = evaluate_saved_policy(None, None, {**env_cfg, "reference_strategy_enabled": True},
                                   split=split, reference_only=True, check_cancel=check_cancel)
    result["enabled"] = True
    return result


def training_worker(cfg):
    """Save each seed before validation, then publish the validation winner."""
    current_env = None
    try:
        algorithm = cfg["algorithm"]
        _, data_meta = load_dataset(OBS, MANIFEST, OPT_DIR)
        device = training_device(cfg)
        root_id = algorithm + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        STATE.begin_log(root_id)

        base_seed = int(cfg["seed"])
        if cfg["multi_seed_enabled"]:
            seeds = [base_seed + i*int(cfg["seed_stride"]) for i in range(int(cfg["num_seeds"]))]
        else:
            seeds = [base_seed]

        # Split the requested budget among seeds, aligned to complete rollouts.
        rollout = cfg["dqn_train_freq"] if algorithm == "dqn" else cfg["n_steps"]
        raw_each = max(rollout, cfg["total_timesteps"] // len(seeds))
        per_seed = max(rollout, (raw_each // rollout) * rollout)
        if algorithm == "dqn" and per_seed <= cfg["dqn_learning_starts"]:
            raise ValueError("DQN steps per seed must exceed dqn_learning_starts")
        total_actual = per_seed * len(seeds)

        with STATE.lock:
            STATE.target = total_actual
            STATE.steps = 0
            STATE.evaluation = {
                "algorithm": algorithm,
                "multi_seed": cfg["multi_seed_enabled"],
                "seeds": seeds,
                "steps_per_seed": per_seed,
                "selection_metric": "average_R + 0.10*log(PF) - drawdown penalty",
                "seed_results": [],
            }

        candidates = []
        base_steps = 0

        for seed in seeds:
            check_training_stop()

            run_id = f"{root_id}_seed{seed}"

            def make_env():
                env = make_market_env(
                    cfg["environment"], state=STATE, model_id=run_id
                )
                if algorithm == "dqn":
                    from banknifty_dqn import NextActionMask
                    env = NextActionMask(env)
                return Monitor(env)

            current_env = DummyVecEnv([make_env])
            current_env = VecNormalize(
                current_env, norm_obs=cfg["normalize_obs_enabled"], norm_reward=cfg["normalize_reward_enabled"],
                clip_obs=cfg["clip_obs"], clip_reward=cfg["clip_reward"], gamma=cfg["gamma"]
            )

            with STATE.lock:
                STATE.device = device
                STATE.gpu = th.cuda.get_device_name(cfg["cuda_device"]) if cfg["cuda_enabled"] else "CPU"

            if algorithm == "dqn":
                from banknifty_dqn import build_dqn
                model = build_dqn(current_env, cfg, device, seed,
                                  str(MODEL_DIR / "tensorboard") if cfg["tensorboard_enabled"] else None)
            else:
                model = maskable_ppo_class()(
                    "MlpPolicy", current_env,
                    learning_rate=cfg["learning_rate"],
                    n_steps=cfg["n_steps"],
                    batch_size=cfg["batch_size"],
                    n_epochs=cfg["n_epochs"],
                    gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"],
                    ent_coef=cfg["ent_coef"], vf_coef=cfg["vf_coef"], max_grad_norm=cfg["max_grad_norm"],
                    target_kl=cfg["target_kl"] if cfg["target_kl_enabled"] else None,
                    policy_kwargs=dict(
                        activation_fn=th.nn.ReLU,
                        net_arch=dict(
                            pi=[cfg["pi_layer1"], cfg["pi_layer2"], cfg["pi_layer3"]],
                            vf=[cfg["vf_layer1"], cfg["vf_layer2"], cfg["vf_layer3"]]
                        )
                    ),
                    device=device, verbose=1, seed=seed,
                    tensorboard_log=str(MODEL_DIR / "tensorboard") if cfg["tensorboard_enabled"] else None,
                )

            model.market_algorithm = algorithm
            model.market_env_config = dict(cfg["environment"])
            model.market_dataset_id = data_meta["dataset_id"]
            with STATE.lock:
                STATE.policy_device = str(next(model.policy.parameters()).device)
            if cfg["cuda_enabled"] and next(model.policy.parameters()).device.type != "cuda":
                raise RuntimeError("Policy parameters are not on the requested CUDA device")
            model.market_env_version = ENV_VERSION
            model.market_feature_columns = list(current_env.get_attr("feature_columns")[0])
            model.market_position_features = list(POSITION_FEATURES)
            model.market_model_id = run_id
            model.market_training_seed = seed
            model.market_training_config = {k: v for k, v in cfg.items() if k != "environment"}

            callback_items = [LiveCallback(STATE, base_steps=base_steps, total_target=total_actual)]
            if cfg["checkpoint_enabled"]:
                callback_items.append(CheckpointCallback(
                    save_freq=max(1000, min(int(cfg["checkpoint_freq"]), per_seed)),
                    save_path=str(CKPT_DIR), name_prefix=run_id,
                    save_vecnormalize=True, verbose=1))
            callback = CallbackList(callback_items)

            with STATE.lock:
                stop_before_learning = STATE.stop
            if not stop_before_learning:
                model.learn(
                    total_timesteps=per_seed,
                    callback=callback,
                    progress_bar=False
                )

            with STATE.lock:
                STATE.status = "stopping" if STATE.stop else "saving"
            candidate_model, candidate_vec = save_training_model(
                model, current_env, cfg, run_id, seed
            )
            current_env.close()
            current_env = None

            check_training_stop()
            with STATE.lock:
                STATE.status = "stopping" if STATE.stop else "validating"

            EVAL_STATE.reset(1)
            EVAL_STATE.status = "validating"
            EVAL_STATE.begin_log(f"validation_{run_id}")
            EVAL_STATE.evaluation = dict(split="validation", seed=seed, automatic=True)
            validation = evaluate_saved_policy(
                candidate_model.with_suffix(".zip"), candidate_vec,
                cfg["environment"], split="validation", state=EVAL_STATE, progress=True,
                reference_only=False, check_cancel=check_training_stop
            )
            EVAL_STATE.close_log()
            with EVAL_STATE.lock:
                EVAL_STATE.status = "stopped" if validation["partial"] else "completed"
                EVAL_STATE.evaluation.update(
                    dataset_id=data_meta["dataset_id"], partial=validation["partial"],
                    trading_metrics=validation["metrics"],
                    targets=validation_target_results(validation["metrics"], cfg, complete=not validation["partial"]),
                    completed_days=validation["finished_days"], total_days=validation["total_days"],
                )
                Path(EVAL_STATE.log_path).with_suffix(".json").write_text(
                    json.dumps(EVAL_STATE.snapshot(), indent=2), encoding="utf-8")
            if validation["partial"]:
                raise TrainingStopped()
            score = validation_selection_score(validation["metrics"], cfg)
            candidates.append(dict(
                seed=seed,
                model=str(candidate_model.with_suffix(".zip")),
                vec=str(candidate_vec),
                validation=validation,
                score=score,
                targets=validation_target_results(validation["metrics"], cfg),
            ))

            with STATE.lock:
                STATE.evaluation["seed_results"].append(dict(
                    seed=seed,
                    score=score,
                    validation_metrics=validation["metrics"],
                    candidate_diagnostics=validation["candidate_diagnostics"],
                    regime_breakdown=validation["regimes"],
                    targets=validation_target_results(validation["metrics"], cfg),
                ))

            base_steps += int(model.num_timesteps)

        if not candidates:
            raise RuntimeError("Training stopped before any seed completed.")

        # Prefer all targets, then the enabled frequency/sample requirements,
        # then score. A fallback still reports every unmet target explicitly.
        best = max(candidates, key=lambda x: validation_candidate_key(x, cfg))

        # Reference-only benchmark on the same validation split/config.
        reference = reference_benchmark(
            cfg["environment"], "validation", cfg["benchmark_reference_enabled"],
            check_cancel=check_training_stop
        )

        check_training_stop()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        final_model = MODEL_DIR / f"banknifty_{algorithm}_{stamp}.zip"
        final_vec = MODEL_DIR / f"vecnormalize_{stamp}.pkl"
        with STATE.lock:
            STATE.status = "stopping" if STATE.stop else "saving"

        meta_path = final_model.with_suffix(".json")
        staged_meta = meta_path.with_suffix(".json.tmp")
        staged_meta.write_text(json.dumps({
            "algorithm": algorithm,
            "validation_status": "completed",
            "environment_version": ENV_VERSION,
            "dataset_id": data_meta["dataset_id"],
            "environment": cfg["environment"],
            "training": {k: v for k, v in cfg.items() if k != "environment"},
            "multi_seed_offsets": list(MULTI_SEED_OFFSETS),
            "steps_per_seed": per_seed,
            "selected_seed": best["seed"],
            "selected_score": best["score"],
            "selected_targets": best["targets"],
            "seed_results": [
                {
                    "seed": c["seed"],
                    "score": c["score"],
                    "validation_metrics": c["validation"]["metrics"],
                    "targets": c["targets"],
                }
                for c in candidates
            ],
            "reference_only_validation": reference["metrics"] if reference["enabled"] else None,
            "normalization_file": final_vec.name,
            "model_id": root_id,
            "trade_log": STATE.log_path,
        }, indent=2), encoding="utf-8")
        with tempfile.TemporaryDirectory(dir=MODEL_DIR, prefix=".saving_") as temp:
            staging = Path(temp)
            shutil.copy2(best["model"], staging / final_model.name)
            shutil.copy2(best["vec"], staging / final_vec.name)
            (staging / final_vec.name).replace(final_vec)
            staged_meta.replace(meta_path)
            (staging / final_model.name).replace(final_model)

        with STATE.lock:
            STATE.model_path = str(final_model)
            STATE.vec_path = str(final_vec)
            STATE.status = "stopped" if STATE.stop else "completed"
            STATE.progress = 100.0 if not STATE.stop else STATE.progress
            STATE.evaluation.update(
                selected_seed=best["seed"],
                selected_validation_score=best["score"],
                selected_targets=best["targets"],
                selected_validation_metrics=best["validation"]["metrics"],
                selected_candidate_diagnostics=best["validation"]["candidate_diagnostics"],
                selected_regime_breakdown=best["validation"]["regimes"],
                reference_only_validation=reference["metrics"] if reference["enabled"] else None,
                reference_only_regime_breakdown=reference["regimes"],
                hybrid_vs_reference=dict(
                    average_R_delta=best["validation"]["metrics"]["average_R"] - reference["metrics"]["average_R"],
                    cumulative_R_delta=best["validation"]["metrics"]["cumulative_R"] - reference["metrics"]["cumulative_R"],
                    win_rate_delta=best["validation"]["metrics"]["win_rate"] - reference["metrics"]["win_rate"],
                    profit_factor_hybrid=best["validation"]["metrics"]["profit_factor"],
                    profit_factor_reference=reference["metrics"]["profit_factor"],
                ) if reference["enabled"] else None,
            )

    except TrainingStopped:
        with STATE.lock:
            STATE.status = "stopped"
    except Exception:
        with STATE.lock:
            STATE.status = "error"
            STATE.error = traceback.format_exc()
    finally:
        STATE.close_log()
        EVAL_STATE.close_log()
        if EVAL_STATE.status == "validating":
            EVAL_STATE.status = "stopped"
        if current_env is not None:
            current_env.close()


def model_files(name):
    if not name or Path(name).name != name or not name.endswith(".zip"):
        raise ValueError("Select a saved model")
    model_path = MODEL_DIR / name
    if not model_path.is_file():
        raise ValueError("Saved model not found")
    stem = model_path.stem
    prefix = next((p for p in ("banknifty_ppo_", "banknifty_dqn_") if stem.startswith(p)), None)
    if prefix is None:
        raise ValueError("Unsupported model filename")
    vec_path = MODEL_DIR / (stem.replace(prefix, "vecnormalize_", 1) + ".pkl")
    if not vec_path.is_file():
        raise ValueError("Matching normalization statistics are missing")

    try:
        with zipfile.ZipFile(model_path) as archive:
            metadata = json.loads(archive.read("data"))
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ValueError("Model archive is incomplete or unreadable") from exc

    if metadata.get("market_env_version") != ENV_VERSION:
        raise ValueError(
            "This model uses an older action/reward format. "
            "Train a fresh autonomous five-action model."
        )
    return model_path, vec_path


def validation_worker(model_path, vec_path, split, override):
    try:
        model = load_trading_model(model_path, device="cpu")
        saved = getattr(model, "market_env_config", None)
        if saved is None or getattr(model, "market_env_version", None) != ENV_VERSION:
            raise ValueError(
                "A fresh model using the current environment version is required; old models are incompatible."
            )

        env_cfg = parse_env_config(override if override is not None else saved)
        EVAL_STATE.begin_log("eval_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

        with EVAL_STATE.lock:
            EVAL_STATE.status = "validating"
            EVAL_STATE.target = 1
            EVAL_STATE.device = "cpu"
            EVAL_STATE.evaluation = dict(
                algorithm=getattr(model, "market_algorithm", "ppo"),
                split=split, model=model_path.name,
                environment=env_cfg,
                settings_source="current form" if override is not None else "saved training settings",
                environment_version=ENV_VERSION,
                partial=True,
            )

        hybrid = evaluate_saved_policy(
            model_path, vec_path, env_cfg, split=split,
            state=EVAL_STATE, reference_only=False, progress=True
        )

        fresh = [t for t in hybrid["trades"] if not t["stale_exit"]]
        saved_training = getattr(model, "market_training_config", TRAIN_DEFAULTS)
        reference = reference_benchmark(env_cfg, split,
                                        saved_training.get("benchmark_reference_enabled", False)
                                        and not hybrid["partial"] and not EVAL_STATE.stop)
        comparison = dict(
            average_R_delta=hybrid["metrics"]["average_R"] - reference["metrics"]["average_R"],
            cumulative_R_delta=hybrid["metrics"]["cumulative_R"] - reference["metrics"]["cumulative_R"],
            max_drawdown_R_delta=hybrid["metrics"]["max_drawdown_R"] - reference["metrics"]["max_drawdown_R"],
            win_rate_delta=hybrid["metrics"]["win_rate"] - reference["metrics"]["win_rate"],
            hybrid_profit_factor=hybrid["metrics"]["profit_factor"],
            reference_profit_factor=reference["metrics"]["profit_factor"],
            hybrid_selection_score=validation_selection_score(hybrid["metrics"], saved_training),
            reference_selection_score=validation_selection_score(reference["metrics"], saved_training),
        )

        with EVAL_STATE.lock:
            EVAL_STATE.evaluation.update(
                partial=hybrid["partial"],
                completed_days=hybrid["finished_days"],
                total_days=hybrid["total_days"],
                fresh_quote_trades=len(fresh),
                fresh_quote_win_rate=(
                    round(
                        100*sum(t["option_pnl_points"] > 0 for t in fresh)/len(fresh), 2
                    ) if fresh else None
                ),
                trading_metrics=hybrid["metrics"],
                targets=validation_target_results(hybrid["metrics"], saved_training,
                                                  complete=not hybrid["partial"]),
                regime_breakdown=hybrid["regimes"],
                candidate_diagnostics=hybrid["candidate_diagnostics"],
                reference_only_metrics=reference["metrics"] if reference["enabled"] else None,
                reference_only_regime_breakdown=reference["regimes"],
                hybrid_vs_reference=comparison if reference["enabled"] else None,
                dataset_id=hybrid["dataset_id"],
                accounting=(
                    "Net option premium points after dynamic slippage, brokerage, "
                    "STT, exchange/SEBI charges, GST and stamp duty; divided by "
                    "fixed initial risk = R. Reward shaping/entry penalties are "
                    "excluded from trading metrics."
                ),
            )
            report_path = Path(EVAL_STATE.log_path).with_suffix(".json")
            EVAL_STATE.evaluation["report_file"] = str(report_path)
            EVAL_STATE.status = (
                "stopped" if hybrid["partial"] else "completed"
            )
            report_path.write_text(
                json.dumps(EVAL_STATE.snapshot(), indent=2), encoding="utf-8"
            )

    except Exception:
        with EVAL_STATE.lock:
            EVAL_STATE.status = "error"
            EVAL_STATE.error = traceback.format_exc()
    finally:
        EVAL_STATE.close_log()


app = Flask(__name__)

HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BANKNIFTY Autonomous PPO / DQN</title><style>
body{font-family:Segoe UI,Arial;background:#0d1117;color:#e6edf3;margin:0}.w{width:min(1500px,96vw);margin:22px auto}
.p{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:15px;margin-bottom:14px}
.g,.c{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}.m{background:#1d2430;padding:12px;border-radius:9px;border:1px solid #30363d}
.l,label{font-size:12px;color:#8b949e}.v{font-size:21px;font-weight:700;margin-top:7px;overflow-wrap:anywhere}
input,select{width:100%;box-sizing:border-box;background:#0d1117;color:white;border:1px solid #30363d;padding:9px;border-radius:6px}input[type=checkbox]{width:auto}
button{padding:10px 14px;border:0;border-radius:6px;font-weight:700;cursor:pointer}.start{background:#3fb950}.stop{background:#f85149;color:white}
.bar{height:16px;background:#0d1117;border:1px solid #30363d;border-radius:10px;overflow:hidden}.fill{height:100%;background:#58a6ff;width:0%}
.muted{color:#8b949e}table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:8px;border-bottom:1px solid #30363d;text-align:left}.sc{max-height:430px;overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:1000px){.g,.c{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:600px){.g,.c{grid-template-columns:1fr}}
</style></head><body><div class="w">
<h2>BANKNIFTY Autonomous PPO / DQN</h2><p class="muted">Select PPO or DQN below. The policy chooses WAIT / BUY CE / BUY PE when flat and HOLD / EXIT while invested. Indicators advise the policy. Both algorithms respect action masks. Entry gates and forced exits are optional controls.</p>
<div class="p"><button onclick="loadDataStatus()">CHECK PREPARED DATA</button><pre id="dataStatus">Checking prepared dataset...</pre></div>
<details class="p"><summary>Optional supervised / candidate research tools (entry selection only)</summary>
<p><button onclick="applyCandidateDefaults()">CANDIDATE DEFAULTS</button> <button onclick="candidateJob('build')">1. BUILD CANDIDATE DATASET</button> <button class="start" onclick="candidateJob('supervised')">2. TRAIN SUPERVISED</button> <button class="start" onclick="candidateJob('ppo')">3. TRAIN CANDIDATE PPO</button> <button class="stop" onclick="stopCandidateJob()">STOP CANDIDATE JOB</button></p>
<div id="candidatefields" class="c"></div>
<p class="muted">Candidate PPO steps count decisions, not minutes. It uses the shared learning rate, network, seed and optimizer settings below, with its own rollout/batch/epochs/gamma here. Candidate PPO runs on CPU; reward is unnormalized net R. Supervised fitting uses CPU. Candidate mode always disables extra entry gates, learned EXIT and reward shaping: those fields belong to the legacy minute trainer. Reference, deterministic exits, costs and cooldown settings define the dataset; rebuild after changing them. Build cancellation keeps the previous published dataset; training cancellation saves the latest fitted model when available. Stop waits for the current fit iteration or PPO update.</p>
<div class="c"><div><label>Candidate model</label><select id="candidatemodel"></select></div><div><label>Evaluation split</label><select id="candidatesplit"><option value="validation">Validation</option><option value="test">Test (final assessment)</option></select></div><div><button onclick="candidateJob('evaluate')">EVALUATE CANDIDATE MODEL</button></div></div>
<p id="candidateStatus" role="status">idle</p><pre id="candidateResults"></pre></details>
<div class="p"><h3>Training</h3><div id="trainfields" class="c"></div>
<h3>Trading, features and reward settings</h3>
<p class="muted">Feature switches control what PPO observes. Gate switches constrain its actions. Forced-exit switches override HOLD. All changes apply to the next fresh run; an evaluation override is explicitly recorded.</p>
<input type="search" placeholder="Find a setting: momentum, CUDA, exit, cost..." oninput="filterSettings(this.value)">
<div id="envfields" class="c"></div>
<details><summary>Disable individual observation features</summary><p>Select feature names to hide from PPO. Availability flags follow their parent feature. Market and position state still exists inside the simulator.</p><select id="featurePicker" multiple size="12" style="width:100%" onchange="setDisabledFeatures()"></select></details>
<pre id="modeSummary"></pre>
<p><button id="savesettings" onclick="saveSettingsFile()">SAVE SETTINGS FILE</button> <button id="loadsettings" onclick="document.getElementById('settingsfile').click()">LOAD SETTINGS FILE</button></p>
<input id="settingsfile" type="file" accept=".json,application/json" hidden onchange="loadSettingsFile(this)">
<p id="settingsmessage" class="muted" role="status">Save training and trading settings as JSON, or choose a saved file to restore the form. Loading does not change a running job.</p>
<p class="muted">Use AUTONOMOUS DEFAULTS for independent entries and learned exits. A missing quote prevents a fill; missing indicators are represented by zero plus availability flags. Validation targets rank models and do not guarantee results. Test is never used for automatic selection.</p>
<p class="muted">Stop premium is a fraction (0.10 = 10%). ATR protection, overtrade penalties and the entry-quality gate are optional. Quantity is a simulation assumption: configure it for the contracts being evaluated. Dynamic slippage and full simulated charges can be enabled/disabled. Verify statutory/broker-specific charge rates before live use.</p>
<button onclick="entryExplorationPreset()">AUTONOMOUS DEFAULTS</button> <button onclick="start()">START TRAINING</button> <button class="stop" onclick="stop()">STOP AND SAVE</button><p id="presetnote" class="muted"></p></div>
<div class="p"><b id="status">idle</b> <span id="steps"></span><br><br><div class="bar"><div id="fill" class="fill"></div></div><p id="runinfo" class="muted"></p></div>
<div class="g" id="trainmetrics"></div><br>
<div class="p"><h3>Learning progress</h3><p class="muted">Last 500 training episodes: grey = net reward including enabled shaping, green = rolling mean of 20. Training results include repeated historical days; use validation/test to assess performance. Optimizer statistics update after PPO updates. CUDA accelerates the neural network; market replay remains on CPU.</p><svg id="learningCurve" viewBox="0 0 700 160" style="width:100%;height:160px" role="img" aria-label="Episode reward and rolling mean"></svg><pre id="optimizerStats"></pre></div>
<div class="p"><h3>Action distribution and exits</h3><p class="muted">Action percentages include forced WAIT/HOLD decisions. BUY percentages count entry decisions, including rejected fills. Trading R excludes entry penalties and shaping. Drawdown is based on cumulative closed-trade R, not account equity.</p><pre id="traindiagnostics"></pre></div>
<div class="p"><h3>Validate Saved Model</h3><div class="c">
<div><label>Compatible model</label><select id="evalmodel"></select></div>
<div><label>Dataset</label><select id="evalsplit"><option value="validation">Validation (tuning)</option><option value="test">Test (final assessment)</option></select></div>
<div><label><input type="checkbox" id="evaloverride"> Use current settings instead of saved settings</label></div>
<div><button class="start" onclick="validateModel()">VALIDATE</button> <button onclick="stopValidation()">STOP AFTER DAY</button></div>
</div><p class="muted">Old models are incompatible. Evaluation is chronological and uses the same masks. Full trade logs and reports are saved locally.</p><p id="evalsummary">idle</p><div class="g" id="evalmetrics"></div><pre id="evaldetails"></pre></div>
<div class="p"><h3>Recent Training Trades</h3><div class="sc"><table><thead><tr><th>Side</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Minutes</th><th>Net points</th><th>R</th><th>Risk points</th><th>MFE / MAE points</th><th>Desired move reached</th><th>Reason</th></tr></thead><tbody id="rows"></tbody></table></div></div>
<div class="p"><pre id="err" style="color:#ff7b72"></pre></div></div>
<script>
const defaults={{ defaults|tojson }};
const trainDefaults={{ train_defaults|tojson }};
const candidateDefaults={{ candidate_defaults|tojson }};
const candidateEnvDefaults={{ candidate_env_defaults|tojson }};
const labels={min_hold_minutes:'Minimum hold (completed bars)',reentry_cooldown_minutes:'Re-entry cooldown (minutes)',min_stop_points:'Minimum emergency stop points',stop_premium_pct:'Stop premium fraction',use_atr_stop:'Include ATR in emergency stop',stop_atr_multiplier:'Emergency stop ATR multiplier',atr_period:'ATR period (1-minute bars)',min_desired_move_points:'Desired move points (not an exit)',max_hold_minutes:'Maximum holding minutes',last_entry_time:'No entries from (exchange time)',square_off_time:'Square-off time',entry_penalty_r:'Entry penalty R',overtrading_penalty_enabled:'Enable overtrade penalty',free_trades_per_day:'Free entries per day',extra_trade_penalty_r:'Extra-entry penalty R',hold_bonus_r:'Favourable momentum HOLD bonus R',giveback_penalty_r:'Giveback penalty R',giveback_fraction:'Peak giveback fraction',max_shaping_r_per_trade:'Maximum absolute shaping per trade R',tradeability_gate_enabled:'Enable entry-quality gate',tradeability_threshold:'Entry-quality threshold (0–1)',trade_quantity:'Simulated option quantity (units)',brokerage_per_order:'Brokerage per order (rupees)',slippage_pct:'Slippage fraction',reward_scale:'Reward scale'};
Object.assign(labels,{reference_strategy_enabled:'Enable reference strategy + PPO',reference_score_gate_enabled:'Enable reference-score gate',reference_window:'Ignition lookback (bars)',reference_ignition_min:'Minimum ignition score (0–1)',reference_direction_gap:'Minimum directional score gap',reference_signal_ttl_enabled:'Enable signal TTL',reference_signal_ttl_minutes:'Signal validity (minutes)',reference_exit_threshold:'Momentum decay score threshold',reference_target_underlying_points:'Base BANKNIFTY target points',reference_strike_window:'OI window (strikes each side)',reference_strike_spacing:'Strike spacing (index points)',reference_structural_lookback:'Structural stop lookback (bars)',reference_pivot_left:'Pivot confirmation bars left',reference_pivot_right:'Pivot confirmation bars right',reference_structural_buffer_fraction:'Structural stop candle-range buffer fraction',learned_exit_enabled:'Enable PPO learned EXIT',risk_reward_target_enabled:'Enable R-multiple take profit',risk_reward_target_r:'Take-profit R multiple',momentum_gate_enabled:'Enable directional momentum gate',path_efficiency_gate_enabled:'Enable path-efficiency gate',atr_expansion_gate_enabled:'Enable ATR-expansion gate',extension_gate_enabled:'Enable extension/exhaustion gate',trend_alignment_gate_enabled:'Enable EMA trend alignment',price_oi_gate_enabled:'Enable price×OI directional gate',option_liquidity_gate_enabled:'Enable option liquidity gate',option_delta_gate_enabled:'Enable option delta gate',regime_filter_enabled:'Enable regime filter',dynamic_slippage_enabled:'Enable dynamic slippage',transaction_costs_enabled:'Enable transaction costs',structural_stop_enabled:'Enable structural stop',momentum_decay_exit_enabled:'Enable momentum-decay exit',underlying_target_enabled:'Enable underlying target',volatility_target_enabled:'Enable ATR-normalized target',emergency_stop_enabled:'Enable emergency option stop',min_stop_enabled:'Enable minimum-point stop',premium_stop_enabled:'Enable premium-% stop',use_atr_stop:'Enable ATR stop',min_hold_enabled:'Enable minimum hold',cooldown_enabled:'Enable re-entry cooldown',max_hold_enabled:'Enable maximum hold',square_off_enabled:'Enable square-off',entry_penalty_enabled:'Enable entry penalty',overtrading_penalty_enabled:'Enable overtrade penalty',hold_shaping_enabled:'Enable HOLD shaping',giveback_penalty_enabled:'Enable giveback penalty',terminal_win_shaping_enabled:'Enable terminal win bonus',terminal_loss_shaping_enabled:'Enable terminal loss penalty',first_entry_time_enabled:'Enable first-entry time',last_entry_time_enabled:'Enable last-entry cutoff'});
const ints=new Set(['atr_period','min_hold_minutes','reentry_cooldown_minutes','max_hold_minutes','free_trades_per_day','trade_quantity','reference_window','reference_signal_ttl_minutes','reference_strike_window','reference_structural_lookback','reference_pivot_left','reference_pivot_right','num_seeds','seed_stride','n_steps','batch_size','n_epochs','seed','checkpoint_freq','pi_layer1','pi_layer2','pi_layer3','vf_layer1','vf_layer2','vf_layer3','total_timesteps']);
function humanize(k){return k.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase())}
function renderFields(parentId,prefix,obj,labelMap={}){const parent=document.getElementById(parentId);for(const [key,value] of Object.entries(obj)){const box=document.createElement('div'),label=document.createElement('label'),input=document.createElement(key==='algorithm'?'select':'input');box.dataset.setting=key;label.textContent=labelMap[key]||humanize(key);label.htmlFor=prefix+key;input.id=prefix+key;if(key==='algorithm'){input.add(new Option('Maskable PPO','ppo'));input.add(new Option('Masked DQN / Double DQN','dqn'));input.value=value}else{input.type=typeof value==='boolean'?'checkbox':typeof value==='string'?(key.endsWith('_time')?'time':'text'):'number';if(input.type==='checkbox')input.checked=value;else input.value=value;if(input.type==='number'){input.min='0';input.step=ints.has(key)?'1':'any'}}box.append(label,input);parent.append(box)}}
Object.assign(labels,{
path_efficiency_window:'Path-efficiency window (completed bars)',
terminal_win_min_points:'Minimum net option points for win bonus (after costs)',
terminal_win_bonus_r:'Win bonus R (within total shaping cap)',
terminal_loss_penalty_r:'Loss penalty R (within total shaping cap)',
giveback_min_mfe_r:'Minimum peak profit R before giveback penalty',
giveback_penalty_enabled:'Enable profit-giveback penalty (independent of HOLD bonus)',
learned_exit_enabled:'Let PPO choose EXIT after minimum hold',
risk_reward_target_enabled:'Force take profit at R target',
underlying_target_enabled:'Force exit at underlying points target',
momentum_decay_exit_enabled:'Force exit on reference momentum decay',
fixed_option_target_enabled:'Enable fixed option-premium target',
fixed_option_target_points:'Fixed option profit target (premium points)',
directional_thresholds_enabled:'Use separate CE / PE thresholds',
ce_reference_ignition_min:'CE minimum ignition score',
pe_reference_ignition_min:'PE minimum ignition score',
ce_reference_direction_gap:'CE minimum score gap',
pe_reference_direction_gap:'PE minimum score gap',
ce_min_directional_return_3m:'CE minimum 3m return',
ce_min_directional_return_5m:'CE minimum 5m return',
pe_min_directional_return_3m:'PE minimum 3m return magnitude',
pe_min_directional_return_5m:'PE minimum 5m return magnitude',
momentum_acceleration_gate_enabled:'Require same-direction momentum acceleration',
min_directional_acceleration:'Minimum directional acceleration',
soft_path_efficiency_enabled:'Use path efficiency in soft quality score',
soft_atr_quality_enabled:'Use ATR regime in soft quality score',
soft_price_oi_quality_enabled:'Use price x OI in soft quality score',
soft_structural_distance_enabled:'Use structural-stop distance in soft quality score',
soft_time_quality_enabled:'Use time-of-day in soft quality score',
structural_distance_gate_enabled:'Enable structural-stop distance gate',
min_structural_stop_atr:'Minimum structural-stop distance (ATR)',
max_structural_stop_atr:'Maximum structural-stop distance (ATR)',
adaptive_cooldown_enabled:'Use exit-reason adaptive cooldown',
target_exit_cooldown_minutes:'Cooldown after +40 target (minutes)',
structural_stop_cooldown_minutes:'Cooldown after structural stop (minutes)',
losing_exit_cooldown_minutes:'Cooldown after other losing exit (minutes)',
exit_reason_shaping_enabled:'Enable exit-reason reward shaping',
option_target_bonus_r:'+40 target reward bonus R',
structural_stop_penalty_r:'Structural-stop extra penalty R',
emergency_stop_penalty_r:'Emergency-stop extra penalty R',
time_limit_loss_penalty_r:'Losing timeout extra penalty R',
momentum_decay_loss_penalty_r:'Losing momentum-decay extra penalty R'
});
ints.add('path_efficiency_window');ints.add('min_validation_trades');['target_exit_cooldown_minutes','structural_stop_cooldown_minutes','losing_exit_cooldown_minutes'].forEach(k=>ints.add(k));
Object.assign(labels,{reference_strategy_enabled:'Restrict entries to reference signals (OFF = PPO chooses both sides)',reference_features_enabled:'Include advisory reference calculations',missing_filter_values_pass:'Optional liquidity/indicator gates permit missing values',missingness_features_enabled:'Show indicator availability flags',disabled_features:'Disabled feature names (comma separated)',nse_option_txn_rate:'Exchange fee rate (turnover fraction)',option_stt_sell_rate:'STT sell rate (turnover fraction)'});
Object.assign(labels,{daily_profit_limit_enabled:'Stop entries after daily profit target',daily_profit_target_points:'Daily NET profit target (option points)',daily_loss_limit_enabled:'Stop entries after daily loss budget',daily_loss_limit_points:'Daily NET loss budget (positive option points)',close_on_daily_limit_enabled:'Include open P&L and request exit on daily limit (next available open)'});
renderFields('envfields','cfg_',defaults,labels);
const trainLabels={target_win_rate_pct:'Validation target win rate %',target_profit_factor:'Validation target profit factor',target_average_r:'Validation target average R',multi_seed_enabled:'Enable multi-seed training',num_seeds:'Number of seeds',seed_stride:'Seed stride',normalize_obs_enabled:'Normalize observations',normalize_reward_enabled:'Normalize rewards',cuda_enabled:'Enable CUDA',checkpoint_enabled:'Enable checkpoints'};
trainLabels.min_validation_trades='Minimum validation trades for target assessment';
trainLabels.min_trades_per_day='Minimum average validation trades/day (0 = disabled)';
Object.assign(trainLabels,{cuda_enabled:'Require CUDA for training (OFF = CPU)',cuda_device:'CUDA GPU index',cuda_tf32_enabled:'Allow TF32 CUDA matrix multiplication',cpu_threads:'PyTorch CPU threads',benchmark_reference_enabled:'Also evaluate deterministic reference benchmark',target_kl_enabled:'Enable KL early stopping per PPO update',tensorboard_enabled:'Write TensorBoard learning logs'});
['cuda_device','cpu_threads'].forEach(k=>ints.add(k));
Object.assign(trainLabels,{daily_targets_enabled:'Require daily objectives in validation assessment',target_daily_points:'Validation daily NET points target',target_daily_hit_rate_pct:'Required % of evaluated days meeting daily target',max_daily_loss_points:'Validation maximum daily loss budget (positive points)'});
['dqn_buffer_size','dqn_learning_starts','dqn_train_freq','dqn_gradient_steps','dqn_target_update_interval'].forEach(k=>ints.add(k));
Object.assign(trainLabels,{algorithm:'Training algorithm',dqn_double_enabled:'DQN: use Double DQN targets',dqn_buffer_size:'DQN: replay capacity (transitions, uses RAM)',dqn_learning_starts:'DQN: valid-action warmup steps',dqn_train_freq:'DQN: collect steps per update',dqn_gradient_steps:'DQN: gradient updates per collection',dqn_target_update_interval:'DQN: target network update interval (steps)',dqn_tau:'DQN: target update fraction',dqn_exploration_fraction:'DQN: fraction of run for epsilon decay',dqn_initial_epsilon:'DQN: initial random-action probability',dqn_final_epsilon:'DQN: final random-action probability'});
renderFields('trainfields','tr_',trainDefaults,trainLabels);
document.getElementById('tr_target_average_r').removeAttribute('min');
const tuningHelp=document.createElement('p');tuningHelp.className='muted';
tuningHelp.textContent='CUDA checks actual GPU execution. Steps are shared across seeds. DQN uses replay/epsilon controls and pi layer widths for its Q-network; PPO-only rollout, GAE, entropy, clipping, epochs, KL and vf layer controls are ignored by DQN. DQN-only fields are ignored by PPO. Each seed is saved before validation. Neither algorithm guarantees profitability.';
document.getElementById('trainfields').after(tuningHelp);
for(const key of ['supervised_iterations','supervised_max_leaf_nodes','supervised_min_samples_leaf','ppo_timesteps','ppo_n_steps','ppo_batch_size','ppo_n_epochs'])ints.add(key);
renderFields('candidatefields','cand_',candidateDefaults,{
good_trade_r:'Good trade target R (0 means net profit > 0; otherwise >= R)',
probability_threshold:'TAKE probability threshold',tune_threshold:'Tune threshold on validation only',
exclude_stale_labels:'Exclude stale exit estimates from supervised labels',
ppo_gamma:'Discount per candidate decision (not per minute)'});
function applyCandidateDefaults(){for(const [key,value] of Object.entries(candidateEnvDefaults)){const el=document.getElementById('cfg_'+key);if(typeof value==='boolean')el.checked=value;else el.value=value}document.getElementById('presetnote').textContent='Candidate defaults applied. Build/rebuild the dataset before training. Current jobs are unchanged.'}
function candidateSettings(){return readSettings('cand_',candidateDefaults)}
async function candidateJob(kind){try{const d=await post('/api/candidates/start',{kind,training:trainingSettings(),environment:envSettings(),candidate_training:candidateSettings(),model:document.getElementById('candidatemodel').value,split:document.getElementById('candidatesplit').value});document.getElementById('candidateStatus').textContent=d.status}catch(e){document.getElementById('candidateStatus').textContent=e.message}}
async function stopCandidateJob(){try{const d=await post('/api/candidates/stop');document.getElementById('candidateStatus').textContent=d.status}catch(e){document.getElementById('candidateStatus').textContent=e.message}}
async function loadCandidateModels(){const r=await fetch('/api/candidates/models');if(!r.ok)throw Error('Cannot list candidate models');const d=await r.json(),s=document.getElementById('candidatemodel'),old=s.value;s.replaceChildren();for(const name of d.models)s.add(new Option(name,name));if(d.models.includes(old))s.value=old}
let candidateRefreshing=false,lastCandidateModel='';
async function refreshCandidate(){if(candidateRefreshing)return;candidateRefreshing=true;try{const r=await fetch('/api/candidates/status');if(!r.ok)throw Error('Cannot read candidate job status');const d=await r.json();document.getElementById('candidateStatus').textContent=`${d.kind||'candidate pipeline'}: ${d.status}${d.model_path?' | Saved: '+d.model_path:''}`;document.getElementById('candidateResults').textContent=d.error||JSON.stringify({progress:d.progress,result:d.result},null,2);if(d.model_path&&d.model_path!==lastCandidateModel){await loadCandidateModels();lastCandidateModel=d.model_path}}catch(e){document.getElementById('candidateStatus').textContent=e.message}finally{candidateRefreshing=false}}
setInterval(refreshCandidate,1500);refreshCandidate();loadCandidateModels().catch(console.error);
function entryExplorationPreset(){
const settings=defaults;
for(const [key,value] of Object.entries(settings)){const el=document.getElementById('cfg_'+key);if(typeof value==='boolean')el.checked=value;else el.value=value}
document.getElementById('presetnote').textContent='Autonomous environment defaults restored for the next run. PPO chooses both entry directions and EXIT. Emergency stop, session entry window, costs and EOD remain enabled. Training settings are unchanged.';
updateModeSummary();syncFeaturePicker();
}
function readSettings(prefix,obj){return Object.fromEntries(Object.entries(obj).map(([key,value])=>{const el=document.getElementById(prefix+key);return [key,typeof value==='boolean'?el.checked:typeof value==='string'?el.value:Number(el.value)]}))}
function envSettings(){return readSettings('cfg_',defaults)}
function trainingSettings(){return readSettings('tr_',trainDefaults)}
function filterSettings(query){for(const box of document.querySelectorAll('[data-setting]'))box.hidden=!box.textContent.toLowerCase().includes(query.toLowerCase())}
function syncFeaturePicker(){const disabled=new Set(document.getElementById('cfg_disabled_features').value.split(',').map(s=>s.trim()));for(const option of document.getElementById('featurePicker').options)option.selected=disabled.has(option.value)}
function setDisabledFeatures(){document.getElementById('cfg_disabled_features').value=Array.from(document.getElementById('featurePicker').selectedOptions).map(o=>o.value).join(',');updateModeSummary()}
function updateModeSummary(){const c=envSettings();const gates=Object.entries(c).filter(([k,v])=>v===true&&(k.endsWith('_gate_enabled')||k==='regime_filter_enabled')).map(([k])=>k);document.getElementById('modeSummary').textContent=JSON.stringify({algorithm:trainingSettings().algorithm,entry_mode:c.reference_strategy_enabled?'Reference signals restrict entries':'Policy chooses CE / PE / WAIT',exit_mode:c.learned_exit_enabled?'Policy chooses HOLD / EXIT':'Agent EXIT disabled',active_entry_gates:gates,forced_exits:['fixed_option_target_enabled','risk_reward_target_enabled','underlying_target_enabled','momentum_decay_exit_enabled','structural_stop_enabled','emergency_stop_enabled','max_hold_enabled','square_off_enabled'].filter(k=>c[k]),daily_budget:{profit_points:c.daily_profit_limit_enabled?c.daily_profit_target_points:null,loss_points:c.daily_loss_limit_enabled?c.daily_loss_limit_points:null,close_on_limit:c.close_on_daily_limit_enabled},disabled_feature_groups:Object.keys(c).filter(k=>k.startsWith('features_')&&!c[k])},null,2)}
document.getElementById('tr_algorithm').addEventListener('change',updateModeSummary);
document.getElementById('envfields').addEventListener('change',()=>{updateModeSummary();syncFeaturePicker()});updateModeSummary();
async function loadDataStatus(){const target=document.getElementById('dataStatus');target.textContent='Checking dataset identity and execution partitions...';try{const response=await fetch('/api/data'),d=await response.json();if(!response.ok)throw Error(d.error+'\n'+(d.command||''));target.textContent=JSON.stringify({ready:d.ready,dataset_id:d.dataset_id,splits:d.splits,feature_count:d.features.length,cuda_available:d.cuda_available,cuda_devices:d.cuda_devices,quality:d.quality},null,2);const picker=document.getElementById('featurePicker');picker.replaceChildren();for(const name of [...d.features,...d.position_features])picker.add(new Option(name,name));syncFeaturePicker()}catch(e){target.textContent=e.message}}loadDataStatus();
function drawLearning(history){const svg=document.getElementById('learningCurve');svg.replaceChildren();if(history.length<2)return;const values=history.flatMap(p=>[p.reward,p.mean20]),lo=Math.min(...values),hi=Math.max(...values),span=Math.max(hi-lo,1e-6);for(const [field,color] of [['reward','#768390'],['mean20','#3fb950']]){const line=document.createElementNS('http://www.w3.org/2000/svg','polyline');line.setAttribute('points',history.map((p,i)=>`${10+i*680/(history.length-1)},${150-(p[field]-lo)*140/span}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',color);line.setAttribute('stroke-width','2');svg.append(line)}}
function settingsBusy(busy){document.getElementById('savesettings').disabled=busy;document.getElementById('loadsettings').disabled=busy}
async function saveSettingsFile(){
settingsBusy(true);const message=document.getElementById('settingsmessage');
try{
const settings={format:'banknifty_training_settings',version:1,environment_version:{{ environment_version|tojson }},training:trainingSettings(),environment:envSettings(),candidate_training:candidateSettings()};
const result=await post('/api/settings/validate',settings);
const file={...result.settings,saved_at:new Date().toISOString()};
const blob=new Blob([JSON.stringify(file,null,2)],{type:'application/json'});
const url=URL.createObjectURL(blob),link=document.createElement('a');
link.href=url;link.download='banknifty_settings_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';
document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
message.textContent='Settings file download requested. Choose its location using your browser download settings.';
}catch(e){message.textContent='Could not save settings: '+e.message}finally{settingsBusy(false)}
}
async function loadSettingsFile(input){
const file=input.files[0];if(!file)return;settingsBusy(true);const message=document.getElementById('settingsmessage');
try{
if(file.size>262144)throw Error('Settings file is too large (maximum 256 KB).');
const payload=JSON.parse((await file.text()).replace(/^\uFEFF/,''));
const result=await post('/api/settings/validate',payload);
// Validate the entire file before changing any form field.
for(const [prefix,values] of [['tr_',result.settings.training],['cfg_',result.settings.environment],['cand_',result.settings.candidate_training]]){
for(const [key,value] of Object.entries(values)){const el=document.getElementById(prefix+key);if(typeof value==='boolean')el.checked=value;else el.value=value}
}
document.getElementById('presetnote').textContent='';
updateModeSummary();syncFeaturePicker();
message.textContent='Loaded '+file.name+'. '+(result.defaulted_fields.length?'Missing fields use current defaults: '+result.defaulted_fields.join(', ')+'. ':'')+'Applies to the next fresh training run. For saved-model validation, select Use current settings.';
}catch(e){message.textContent='Could not load settings; form unchanged: '+e.message}finally{input.value='';settingsBusy(false)}
}
async function post(url,body={}){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d}
async function start(){try{const t=trainingSettings();t.environment=envSettings();await post('/api/start',t)}catch(e){alert(e.message)}}
async function stop(){try{const d=await post('/api/stop');document.getElementById('status').textContent=d.status;await refresh()}catch(e){alert(e.message)}}
async function stopValidation(){try{await post('/api/validate/stop')}catch(e){alert(e.message)}}
async function validateModel(){try{await post('/api/validate',{model:document.getElementById('evalmodel').value,split:document.getElementById('evalsplit').value,use_current_settings:document.getElementById('evaloverride').checked,environment:envSettings()})}catch(e){alert(e.message)}}
async function loadModels(){const d=await(await fetch('/api/models')).json(),s=document.getElementById('evalmodel'),old=s.value;s.replaceChildren();for(const name of d.models)s.add(new Option(name,name));if(d.models.includes(old))s.value=old}
const metrics={total_trading_days:'Trading day episodes',total_trades:'Trades',trades_per_day:'Trades/day',ce_trades:'CE trades',pe_trades:'PE trades',wins:'Wins',losses:'Losses',win_rate:'Win rate %',profit_factor:'Profit factor (net points)',average_winner_points:'Average winner points',average_loser_points:'Average loser points',payoff_ratio:'Winner/Loser payoff ratio',trades_ge_1_5R_pct:'Trades >= +1.5R %',average_R:'Average R',median_R:'Median R',cumulative_R:'Cumulative R',max_drawdown_R:'Max drawdown R',average_holding_minutes:'Average hold minutes',median_holding_minutes:'Median hold minutes',average_MFE_points:'Average MFE points',average_MAE_points:'Average MAE points',MFE_MAE_ratio:'MFE/MAE ratio',reached_40_points_pct:'Desired move reached %',stale_exits:'Estimated stale exits'};
Object.assign(metrics,{net_option_points:'Total NET points',average_daily_points:'Average NET points/day (all day episodes)',average_daily_rupees:'Evaluated average NET rupees/day',worst_day_points:'Evaluated worst day NET points',best_day_points:'Evaluated best day NET points',profitable_days_pct:'Evaluated profitable days %'});
function format(v){return v===null||v===undefined?'N/A':typeof v==='number'?Number(v.toFixed(3)).toLocaleString():String(v)}
function cards(id,m){const parent=document.getElementById(id);parent.replaceChildren();for(const [key,label] of Object.entries(metrics)){const card=document.createElement('div');card.className='m';const title=document.createElement('div'),value=document.createElement('div');title.className='l';title.textContent=label;value.className='v';value.textContent=format(m[key]);card.append(title,value);parent.append(card)}}
function entryExplanation(d){const x=d.entry_diagnostics;if(!x||!x.flat_steps)return 'Waiting for entry decisions';if(!x.buy_available_steps)return 'Both BUY actions were blocked on every flat step. See blocked_by; changing PPO rewards cannot unlock masked actions.';if(!x.entries_opened&&d.rejected_entries)return 'BUY actions were attempted but fills were rejected. Check next-bar quotes, time gaps, premium/risk and structural-stop conditions.';if(!x.entries_opened)return 'BUY actions were available, but PPO has not opened a position. See voluntary_wait_steps.';return `${x.entries_opened} entries opened; ${d.metrics.total_trades} trades closed. Trade cards count closed positions.`}
function diagnostics(d){return {entry_explanation:entryExplanation(d),entry_diagnostics:d.entry_diagnostics,action_percentages:d.action_percentages,exit_counts:d.exit_counts,invalid_actions:d.invalid_actions,rejected_entries:d.rejected_entries,trade_log:d.log_path,model_selection:d.evaluation}}
let lastSaved='',refreshing=false;
async function refresh(){
if(refreshing)return;refreshing=true;
try{
const [d,e]=await Promise.all(['/api/status','/api/validate/status'].map(async url=>(await fetch(url)).json()));
document.getElementById('status').textContent=d.status;
document.getElementById('steps').textContent=`${d.steps.toLocaleString()} / ${d.target.toLocaleString()} | ${format(d.progress)}%`;
document.getElementById('fill').style.width=d.progress+'%';
document.getElementById('runinfo').textContent=`${d.gpu} | Requested ${d.device} | Policy ${d.policy_device} | GPU memory ${format(d.gpu_memory_mb)} MB | ${d.fps} steps/s | Episodes ${d.episodes} | Model ${d.model_path||'not saved yet'}`;
cards('trainmetrics',d.metrics);drawLearning(d.history||[]);
document.getElementById('optimizerStats').textContent=JSON.stringify(d.optimizer||{},null,2);
document.getElementById('traindiagnostics').textContent=JSON.stringify(diagnostics(d),null,2);
cards('evalmetrics',e.metrics);
document.getElementById('evalsummary').textContent=`${e.status} | Days ${e.steps}/${e.target}`;
document.getElementById('evaldetails').textContent=e.error||JSON.stringify({...diagnostics(e),evaluation:e.evaluation},null,2);
document.getElementById('err').textContent=d.error||'';
const rows=document.getElementById('rows');rows.replaceChildren();
for(const t of d.trades){const tr=document.createElement('tr');for(const value of [t.side,t.symbol,t.entry_time,t.exit_time,t.holding_minutes,t.option_pnl_points,t.R_return,t.initial_risk_points,`${format(t.MFE_option_points)} / ${format(t.MAE_option_points)}`,t.reached_40_points,t.exit_reason]){const td=document.createElement('td');td.textContent=format(value);tr.append(td)}rows.append(tr)}
if(d.model_path&&d.model_path!==lastSaved){lastSaved=d.model_path;await loadModels()}
}catch(err){document.getElementById('err').textContent=err.message}finally{refreshing=false}}
setInterval(refresh,1500);loadModels().catch(console.error);refresh();
</script></body></html>
"""
@app.route("/")
def index():
    return render_template_string(HTML, defaults=ENV_DEFAULTS, train_defaults=TRAIN_DEFAULTS,
                                  environment_version=ENV_VERSION,
                                  candidate_defaults=CANDIDATE_TRAIN_DEFAULTS,
                                  candidate_env_defaults=CANDIDATE_ENV_DEFAULTS)

@app.route("/api/status")
def status():
    return jsonify(STATE.snapshot())

def parse_training_config(p):
    if not isinstance(p, dict):
        raise ValueError("Training settings must be an object")
    unknown = set(p) - set(TRAIN_DEFAULTS) - {"environment"}
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    cfg = {}
    for key, default in TRAIN_DEFAULTS.items():
        value = p.get(key, default)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true/false")
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
        elif isinstance(default, int):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
                raise ValueError(f"{key} must be a finite integer")
            value = int(value)
        else:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        cfg[key] = value
    cfg["environment"] = parse_env_config(p.get("environment", {}))

    if cfg["total_timesteps"] <= 0 or cfg["n_steps"] < 2 or cfg["batch_size"] < 2 or cfg["n_epochs"] < 1:
        raise ValueError("Positive timesteps/epochs and rollout/batch sizes >= 2 are required")
    if cfg["algorithm"] not in ("ppo", "dqn"):
        raise ValueError("algorithm must be ppo or dqn")
    if cfg["algorithm"] == "ppo" and (cfg["batch_size"] > cfg["n_steps"] or cfg["n_steps"] % cfg["batch_size"]):
        raise ValueError("Batch size must divide rollout steps")
    if min(cfg["dqn_buffer_size"], cfg["dqn_train_freq"], cfg["dqn_gradient_steps"], cfg["dqn_target_update_interval"]) < 1:
        raise ValueError("DQN replay size, train frequency, gradient steps and target interval must be positive")
    if cfg["dqn_learning_starts"] < 0 or not 0 < cfg["dqn_tau"] <= 1:
        raise ValueError("DQN learning starts must be nonnegative and tau must be in (0,1]")
    if not 0 < cfg["dqn_exploration_fraction"] <= 1 or not 0 <= cfg["dqn_final_epsilon"] <= cfg["dqn_initial_epsilon"] <= 1:
        raise ValueError("Invalid DQN exploration schedule")
    if cfg["algorithm"] == "dqn" and cfg["batch_size"] > cfg["dqn_buffer_size"]:
        raise ValueError("DQN replay buffer must hold at least one batch")
    if cfg["learning_rate"] <= 0 or not 0 < cfg["gamma"] <= 1 or not 0 < cfg["gae_lambda"] <= 1:
        raise ValueError("Invalid learning rate/gamma/GAE lambda")
    if cfg["num_seeds"] < 1 or cfg["num_seeds"] > 20:
        raise ValueError("num_seeds must be 1..20")
    if not 1 <= cfg["cpu_threads"] <= 128 or cfg["cuda_device"] < 0:
        raise ValueError("CPU threads must be 1..128 and CUDA device index must be nonnegative")
    if cfg["target_kl"] <= 0 or not 0 < cfg["clip_range"] < 1 or cfg["max_grad_norm"] <= 0:
        raise ValueError("Positive target KL/gradient norm and clip range in (0,1) required")
    if min(cfg["clip_obs"], cfg["clip_reward"], cfg["checkpoint_freq"]) <= 0:
        raise ValueError("Normalization clips and checkpoint frequency must be positive")
    if min(cfg["ent_coef"], cfg["vf_coef"], cfg["seed"], cfg["seed_stride"]) < 0:
        raise ValueError("Entropy/value coefficients and seeds must be nonnegative")
    if cfg["target_win_rate_pct"] > 100:
        raise ValueError("target_win_rate_pct must be <= 100")
    if cfg["target_win_rate_pct"] < 0 or cfg["target_profit_factor"] <= 0:
        raise ValueError("Target win rate must be >= 0 and target profit factor must be positive")
    if not 0 <= cfg["target_daily_hit_rate_pct"] <= 100 or min(cfg["target_daily_points"], cfg["max_daily_loss_points"]) <= 0:
        raise ValueError("Daily target hit rate must be 0..100; daily point targets/budgets must be positive")
    if cfg["min_validation_trades"] < 1:
        raise ValueError("min_validation_trades must be >= 1")
    if isinstance(p.get("min_trades_per_day"), bool) or cfg["min_trades_per_day"] < 0:
        raise ValueError("min_trades_per_day must be a nonnegative number")
    for k in ("pi_layer1","pi_layer2","pi_layer3","vf_layer1","vf_layer2","vf_layer3"):
        if cfg[k] < 8:
            raise ValueError(f"{k} must be >= 8")
    return cfg


def validate_settings_file(payload):
    if not isinstance(payload, dict):
        raise ValueError("Settings file must contain a JSON object")
    if payload.get("format") != "banknifty_training_settings" or type(payload.get("version")) is not int or payload["version"] != 1:
        raise ValueError("Unsupported settings file format or version")
    if payload.get("environment_version") != ENV_VERSION:
        raise ValueError("Settings file belongs to a different environment version")
    defaulted = []
    for section, defaults in (("training", TRAIN_DEFAULTS), ("environment", ENV_DEFAULTS)):
        values = payload.get(section)
        if not isinstance(values, dict):
            raise ValueError(f"Settings file needs a {section} object")
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
        for key, default in defaults.items():
            if key not in values:
                defaulted.append(f"{section}.{key}")
                continue
            value = values[key]
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, str):
                valid = isinstance(value, str)
            else:
                valid = type(value) in (int, float) and math.isfinite(value)
                if valid and isinstance(default, int):
                    valid = float(value).is_integer()
                if valid and key != "target_average_r":
                    valid = value >= 0
            if not valid:
                raise ValueError(f"Invalid value for {section}.{key}")
    cfg = parse_training_config({**payload["training"], "environment": payload["environment"]})
    candidate_payload = payload.get("candidate_training", {})
    if not isinstance(candidate_payload, dict):
        raise ValueError("candidate_training must be an object")
    candidate_cfg = parse_candidate_training(candidate_payload)
    if "candidate_training" not in payload:
        defaulted.append("candidate_training (candidate pipeline defaults)")
    else:
        defaulted.extend(f"candidate_training.{key}" for key in CANDIDATE_TRAIN_DEFAULTS if key not in candidate_payload)
    return dict(settings=dict(format="banknifty_training_settings", version=1,
                             environment_version=ENV_VERSION,
                             training={k: v for k, v in cfg.items() if k != "environment"},
                             environment=cfg["environment"], candidate_training=candidate_cfg), defaulted_fields=defaulted)


@app.route("/api/settings/validate", methods=["POST"])
def settings_validate_api():
    # Pure validation: no worker, model load, file write, or job-state mutation.
    try:
        return jsonify(validate_settings_file(request.get_json(silent=True)))
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/start", methods=["POST"])
def start_api():
    p = request.get_json(silent=True) or {}
    try:
        cfg = parse_training_config(p)
    except Exception as e:
        return jsonify(error=str(e)),400
    with JOB_LOCK:
        if jobs_active():
            return jsonify(error="A training or evaluation run is already active"),409
        STATE.reset(cfg["total_timesteps"])
        t = threading.Thread(target=training_worker,args=(cfg,),daemon=True)
        STATE.thread = t
        t.start()
    return jsonify(ok=True)


@app.route("/api/data")
def data_status_api():
    try:
        _, meta = load_dataset(OBS, MANIFEST, OPT_DIR)
        return jsonify(ready=True, dataset_id=meta["dataset_id"], splits=meta["splits"],
                       features=meta["observation_columns"], position_features=list(POSITION_FEATURES),
                       quality=meta["quality"], cuda_available=th.cuda.is_available(),
                       cuda_devices=[th.cuda.get_device_name(i) for i in range(th.cuda.device_count())])
    except (OSError, ValueError, KeyError) as exc:
        return jsonify(ready=False, error=str(exc), command="python prepare_rl_data.py"), 400


def candidate_model_path(name):
    if not isinstance(name, str) or Path(name).name != name or not name.startswith("candidate_"):
        raise ValueError("Select a locally trained candidate model")
    path = CANDIDATE_MODEL_DIR / name
    if path.suffix not in (".zip", ".joblib") or not path.is_file():
        raise ValueError("Candidate model not found")
    try:
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Candidate model metadata missing or unreadable") from exc
    if meta.get("version") != CANDIDATE_VERSION:
        raise ValueError("Incompatible candidate model version")
    if meta.get("kind") == "ppo" and not path.with_suffix(".vec.pkl").is_file():
        raise ValueError("Candidate normalization file is missing")
    return path


@app.route("/api/candidates/start", methods=["POST"])
def candidate_start_api():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        kind = payload.get("kind")
        if kind not in ("build", "supervised", "ppo", "evaluate"):
            raise ValueError("Choose build, supervised, ppo or evaluate")
        cfg = parse_training_config({**payload.get("training", {}), "environment": payload.get("environment", {})})
        candidate_cfg = parse_candidate_training(payload.get("candidate_training", {}))
        model = candidate_model_path(payload.get("model", "")) if kind == "evaluate" else None
        split = payload.get("split", "validation")
        if split not in ("validation", "test"):
            raise ValueError("Choose validation or test")
    except (ValueError, TypeError, OverflowError) as exc:
        return jsonify(error=str(exc)), 400
    with JOB_LOCK:
        if jobs_active():
            return jsonify(error="A training, dataset build or evaluation is already active"), 409
        CANDIDATE_JOB.update(status="starting", stop=False, error="", kind=kind,
                             progress={}, result=None, model_path="")
        worker = threading.Thread(target=candidate_job_worker,
                                  args=(kind, cfg, candidate_cfg, model, split), daemon=True)
        CANDIDATE_JOB["thread"] = worker
        worker.start()
    return jsonify(ok=True, status="starting")


@app.route("/api/candidates/status")
def candidate_status_api():
    with JOB_LOCK:
        return jsonify({k: v for k, v in CANDIDATE_JOB.items() if k != "thread"})


@app.route("/api/candidates/stop", methods=["POST"])
def candidate_stop_api():
    with JOB_LOCK:
        if CANDIDATE_JOB["status"] in BUSY:
            CANDIDATE_JOB.update(stop=True, status="stopping")
        return jsonify(ok=True, status=CANDIDATE_JOB["status"])


@app.route("/api/candidates/models")
def candidate_models_api():
    names = []
    for path in sorted(CANDIDATE_MODEL_DIR.glob("candidate_*"), key=lambda p: p.name, reverse=True):
        if path.suffix in (".zip", ".joblib"):
            try:
                candidate_model_path(path.name)
                names.append(path.name)
            except ValueError:
                pass
    return jsonify(models=names)


@app.route("/api/models")
def models_api():
    names = []
    for path in sorted(MODEL_DIR.glob("banknifty_*.zip"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            model_files(path.name)
            names.append(path.name)
        except ValueError:
            continue
    return jsonify(models=names)


@app.route("/api/validate", methods=["POST"])
def validate_api():
    p = request.get_json(silent=True) or {}
    try:
        split = p.get("split", "validation")
        if split not in {"validation", "test"}:
            raise ValueError("Choose validation or test data")
        model_path, vec_path = model_files(p.get("model", ""))
        override = parse_env_config(p.get("environment", {})) if p.get("use_current_settings") is True else None
    except Exception as exc:
        return jsonify(error=str(exc)),400
    with JOB_LOCK:
        if jobs_active():
            return jsonify(error="A training or evaluation run is already active"),409
        EVAL_STATE.reset(1)
        with EVAL_STATE.lock:
            EVAL_STATE.status = "validating"
            EVAL_STATE.model_path = str(model_path)
            EVAL_STATE.vec_path = str(vec_path)
        worker = threading.Thread(target=validation_worker,
            args=(model_path, vec_path, split, override), daemon=True)
        EVAL_STATE.thread = worker
        worker.start()
    return jsonify(ok=True)


@app.route("/api/validate/status")
def validate_status():
    return jsonify(EVAL_STATE.snapshot())


@app.route("/api/validate/stop", methods=["POST"])
def validate_stop():
    with EVAL_STATE.lock:
        EVAL_STATE.stop = True
    return jsonify(ok=True)

@app.route("/api/stop", methods=["POST"])
def stop_api():
    with JOB_LOCK, STATE.lock:
        if STATE.status not in BUSY:
            return jsonify(ok=True, status=STATE.status)
        STATE.stop = True
        STATE.status = "stopping"
    return jsonify(ok=True, status="stopping")

if __name__ == "__main__":
    print("BANKNIFTY PPO Web Trainer")
    print("CUDA:", th.cuda.is_available())
    if th.cuda.is_available():
        print("GPU:", th.cuda.get_device_name(0))
        print("Torch CUDA:", th.version.cuda)
    print("Open: http://127.0.0.1:5000")
    app.run(host="127.0.0.1",port=5000,debug=False,threaded=True,use_reloader=False)
