"""Supervised candidate quality, candidate-only PPO, and chronological evaluation."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from banknifty_env import CandidateBankNiftyEnv
from build_candidate_dataset import (
    BASE, DEFAULT_DATASET, CandidateCancelled, load_candidate_dataset,
)
from reference_strategy import CANDIDATE_VERSION, CANDIDATE_FEATURES, candidate_observations

MODEL_DIR = BASE / "models/candidates"
CANDIDATE_TRAIN_DEFAULTS = dict(
    good_trade_r=0.0, probability_threshold=0.60, tune_threshold=True,
    threshold_min=0.50, threshold_max=0.85, threshold_step=0.05,
    supervised_iterations=200, supervised_learning_rate=0.05,
    supervised_max_leaf_nodes=15, supervised_min_samples_leaf=30,
    supervised_l2=1.0, exclude_stale_labels=True,
    ppo_timesteps=30000, ppo_n_steps=256, ppo_batch_size=64,
    ppo_n_epochs=10, ppo_gamma=1.0,
)


def parse_candidate_training(payload=None):
    payload = {} if payload is None else payload
    if not isinstance(payload, dict) or set(payload)-set(CANDIDATE_TRAIN_DEFAULTS):
        raise ValueError("Unknown candidate-training settings")
    cfg = dict(CANDIDATE_TRAIN_DEFAULTS)
    for key, default in cfg.items():
        value = payload.get(key, default)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true/false")
        else:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            if isinstance(default, int):
                if not float(value).is_integer() or value < 1:
                    raise ValueError(f"{key} must be a positive integer")
                value = int(value)
        cfg[key] = value
    if not 0 <= cfg["probability_threshold"] <= 1:
        raise ValueError("Probability threshold must be in [0, 1]")
    if not 0 <= cfg["threshold_min"] <= cfg["threshold_max"] <= 1 or cfg["threshold_step"] <= 0:
        raise ValueError("Invalid threshold search range")
    if (cfg["threshold_max"]-cfg["threshold_min"])/cfg["threshold_step"] > 100:
        raise ValueError("Use at most 101 validation thresholds")
    if cfg["supervised_learning_rate"] <= 0 or cfg["supervised_max_leaf_nodes"] < 2:
        raise ValueError("Supervised learning rate must be positive and leaf count >= 2")
    if cfg["ppo_n_steps"] < 2 or cfg["ppo_batch_size"] < 2 or cfg["ppo_n_steps"] % cfg["ppo_batch_size"]:
        raise ValueError("Candidate PPO batch size must divide rollout steps; both must be >= 2")
    if not 0 < cfg["ppo_gamma"] <= 1:
        raise ValueError("Candidate PPO gamma must be in (0, 1]")
    return cfg


def good_labels(frame, cfg):
    outcome = frame.realized_R > 0 if cfg["good_trade_r"] == 0 else frame.realized_R >= cfg["good_trade_r"]
    return outcome & frame.filled & (~frame.stale_exit if cfg["exclude_stale_labels"] else True)


def predict_candidate_quality(model, candidates):
    """Public inference output; accepts decision-time rows without outcome labels."""
    result = candidates[["candidate_id", "timestamp", "side", "symbol"]].copy()
    result["probability_good_trade"] = model.predict_proba(candidate_observations(candidates))[:, 1]
    return result


def trade_metrics(trades, days):
    points = np.asarray([float(t["option_pnl_points"]) for t in trades])
    rs = np.asarray([float(t["R_return"]) for t in trades])
    gains = float(points[points > 0].sum())
    losses = float(-points[points < 0].sum())
    curve = np.r_[0.0, np.cumsum(rs)]
    return dict(total_trades=len(trades), total_trading_days=int(days),
                trades_per_day=len(trades)/days if days else 0.0,
                ce_trades=sum(t["side"] == "CE" for t in trades),
                pe_trades=sum(t["side"] == "PE" for t in trades),
                win_rate=float(100*np.mean(points > 0)) if len(points) else 0.0,
                profit_factor=gains/losses if losses else None,
                average_R=float(rs.mean()) if len(rs) else 0.0,
                cumulative_R=float(rs.sum()),
                max_drawdown_R=float(np.max(np.maximum.accumulate(curve)-curve)),
                stale_exits=sum(bool(t["stale_exit"]) for t in trades))


def trade_frequency_result(metrics, minimum=0.0):
    """Average closed trades across all evaluated days, including zero-trade days.

    Zero disables the criterion for older presets/models. An enabled criterion
    cannot pass without an evaluated-day count. Do not trust a separately rounded
    trades_per_day display value when checking the boundary.
    """
    days = int(metrics.get("total_trading_days", 0))
    trades = int(metrics.get("total_trades", 0))
    average = trades/days if days > 0 else 0.0
    return dict(minimum=float(minimum), actual=average, evaluated_days=days,
                meets_minimum=bool(minimum <= 0 or (days > 0 and average >= minimum)))


def selection_key(report, targets):
    m = report["metrics"]
    enough = m["total_trades"] >= int(targets.get("min_validation_trades", 50))
    pf = m["profit_factor"]
    frequency = trade_frequency_result(m, float(targets.get("min_trades_per_day", 0.0)))
    checks = dict(enough_trades=enough,
                  trades_per_day=frequency["meets_minimum"],
                  win_rate=m["win_rate"] >= float(targets.get("target_win_rate_pct", 70)),
                  profit_factor=pf is not None and pf >= float(targets.get("target_profit_factor", 1.5)),
                  average_r=m["average_R"] >= float(targets.get("target_average_r", 0)))
    report["targets"] = dict(checks=checks, targets_met=all(checks.values()), trade_frequency=frequency)
    score = m["average_R"] - 0.05*m["max_drawdown_R"]/max(np.sqrt(m["total_trades"]), 1)
    return (all(checks.values()), enough and frequency["meets_minimum"], score if m["total_trades"] else -1e9)


def replay_candidates(frame, split, decide, cfg, metadata, check_cancel=None, progress=None):
    """Evaluate an executable single-position portfolio, never shuffled rows."""
    env = CandidateBankNiftyEnv(frame, split, random_day=False)
    trades, taken_ids = [], []
    totals = dict(reference_candidates=0, taken_candidates=0, skipped_candidates=0,
                  unavailable_candidates=0, rejected_candidates=0)
    try:
        for day_index in range(len(env.days)):
            obs, _ = env.reset(options={"day_index": day_index})
            done = False
            while not done:
                if check_cancel:
                    check_cancel()
                action = int(decide(obs))
                obs, _, done, _, info = env.step(action)
                if action == 1:
                    taken_ids.append(info["candidate_id"])
                if info["trade"]:
                    trades.append(info["trade"])
            for key in totals:
                totals[key] += env.counts[key]
            if progress:
                progress(dict(stage=f"evaluating_{split}", completed_days=day_index+1,
                              total_days=len(env.days), **totals))
    finally:
        env.close()
    subset = frame.loc[frame.split == split]
    labels = good_labels(subset, cfg)
    positives = set(subset.loc[labels, "candidate_id"])
    true_positive = len(set(taken_ids) & positives)
    totals.update(candidate_precision=100*true_positive/len(taken_ids) if taken_ids else None,
                  candidate_recall=100*true_positive/len(positives) if positives else None,
                  good_reference_candidates=len(positives))
    groups = sorted({t["regime"] for t in trades})
    days = metadata["split_days"][split]
    return dict(split=split, metrics=trade_metrics(trades, days), candidates=totals,
                regime_breakdown={g: trade_metrics([t for t in trades if t["regime"] == g], days) for g in groups},
                side_breakdown={s: trade_metrics([t for t in trades if t["side"] == s], days) for s in ("CE", "PE")},
                precision_recall_note="Positive TAKEs / all TAKEs; captured positive candidates / all positive counterfactual candidates, including overlapping events.")


def save_artifact(model, metadata, kind, vec=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ".joblib" if kind == "supervised" else ".zip"
    path = MODEL_DIR / (f"candidate_{kind}_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+suffix)
    with tempfile.TemporaryDirectory(dir=MODEL_DIR, prefix=".saving_") as temp:
        staged = Path(temp) / path.name
        if kind == "supervised":
            import joblib
            joblib.dump(model, staged)
        else:
            model.save(staged)
            vec.save(staged.with_suffix(".vec.pkl"))
            staged.with_suffix(".vec.pkl").replace(path.with_suffix(".vec.pkl"))
        staged.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staged.with_suffix(".json").replace(path.with_suffix(".json"))
        staged.replace(path)
    return path


def train_candidate_model(kind="supervised", dataset=DEFAULT_DATASET, candidate_cfg=None,
                          training_cfg=None, expected_config=None, check_cancel=None,
                          progress=None, saved=None):
    cfg = parse_candidate_training(candidate_cfg)
    training_cfg = training_cfg or {}
    frame, data_meta = load_candidate_dataset(dataset, expected_config)
    metadata = dict(version=CANDIDATE_VERSION, kind=kind, dataset_id=data_meta["dataset_id"],
                    environment=data_meta["environment"], config_fingerprint=data_meta["config_fingerprint"],
                    feature_columns=list(CANDIDATE_FEATURES), candidate_training=cfg,
                    training=training_cfg, probability_threshold=cfg["probability_threshold"],
                    validation=None, training_status="completed")
    if check_cancel:
        check_cancel()
    vec = None
    stopped = False
    if kind == "supervised":
        from sklearn.ensemble import HistGradientBoostingClassifier
        fit = frame.loc[(frame.split == "train") & frame.filled]
        if cfg["exclude_stale_labels"]:
            fit = fit.loc[~fit.stale_exit]
        x = candidate_observations(fit)
        y = good_labels(fit, cfg).to_numpy(dtype=int)
        if len(np.unique(y)) != 2:
            raise ValueError("Supervised training requires both positive and negative train labels")
        model = HistGradientBoostingClassifier(
            learning_rate=cfg["supervised_learning_rate"], max_iter=1, warm_start=True,
            max_leaf_nodes=cfg["supervised_max_leaf_nodes"], min_samples_leaf=cfg["supervised_min_samples_leaf"],
            l2_regularization=cfg["supervised_l2"], early_stopping=False,
            random_state=int(training_cfg.get("seed", 42)))
        # No random validation split: all threshold tuning uses the held-out dates.
        try:
            for iteration in range(1, cfg["supervised_iterations"]+1):
                if check_cancel:
                    check_cancel()
                model.set_params(max_iter=iteration)
                model.fit(x, y)
                if progress:
                    progress(dict(stage="training_supervised", steps=iteration,
                                  target=cfg["supervised_iterations"]))
        except CandidateCancelled:
            if not hasattr(model, "classes_"):
                raise
            stopped = True
        metadata["completed_iterations"] = int(model.n_iter_)
    elif kind == "ppo":
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        class ProgressCallback(BaseCallback):
            def _on_training_start(self):
                self.trades = []
                self.taken = self.skipped = self.rejected = 0

            def _on_step(self):
                latest = self.locals["infos"][0]
                self.taken += int(latest["action_name"] == "TAKE")
                self.skipped += int(latest["action_name"] == "SKIP")
                self.rejected += int(latest["action_name"] == "TAKE" and latest["trade"] is None)
                if latest["trade"]:
                    self.trades.append(latest["trade"])
                if progress and (self.n_calls % 25 == 0 or self.locals["dones"][0]):
                    progress(dict(stage="training_ppo", steps=self.num_timesteps,
                                  target=cfg["ppo_timesteps"],
                                  taken_candidates=self.taken, skipped_candidates=self.skipped,
                                  rejected_candidates=self.rejected,
                                  last_candidate=latest,
                                  metrics=trade_metrics(self.trades, 0)))
                if check_cancel:
                    try:
                        check_cancel()
                    except CandidateCancelled:
                        return False
                return True

        raw = CandidateBankNiftyEnv(frame, "train")
        vec = VecNormalize(DummyVecEnv([lambda: raw]), norm_obs=training_cfg.get("normalize_obs_enabled", True),
                           norm_reward=False, clip_obs=float(training_cfg.get("clip_obs", 10)))
        try:
            model = PPO("MlpPolicy", vec, n_steps=cfg["ppo_n_steps"], batch_size=cfg["ppo_batch_size"],
                        n_epochs=cfg["ppo_n_epochs"], gamma=cfg["ppo_gamma"],
                        learning_rate=float(training_cfg.get("learning_rate", 0.0003)),
                        ent_coef=float(training_cfg.get("ent_coef", 0.01)),
                        gae_lambda=float(training_cfg.get("gae_lambda", 0.95)),
                        clip_range=float(training_cfg.get("clip_range", 0.2)),
                        vf_coef=float(training_cfg.get("vf_coef", 0.5)),
                        max_grad_norm=float(training_cfg.get("max_grad_norm", 0.5)),
                        policy_kwargs=dict(net_arch=dict(
                            pi=[int(training_cfg.get(f"pi_layer{i}", d)) for i, d in enumerate((128, 64, 32), 1)],
                            vf=[int(training_cfg.get(f"vf_layer{i}", d)) for i, d in enumerate((128, 64, 32), 1)])),
                        seed=int(training_cfg.get("seed", 42)), device="cpu", verbose=0)
            model.learn(cfg["ppo_timesteps"], callback=ProgressCallback())
            if check_cancel:
                check_cancel()
        except CandidateCancelled:
            stopped = True
        except Exception:
            vec.close()
            raise
        metadata["completed_timesteps"] = int(model.num_timesteps)
    else:
        raise ValueError("Choose supervised or ppo")
    metadata["training_status"] = "stopped" if stopped else "completed"
    try:
        path = save_artifact(model, metadata, kind, vec)
        if saved:
            saved(path)
        if stopped:
            return path, metadata
        if kind == "supervised":
            thresholds = [cfg["probability_threshold"]]
            if cfg["tune_threshold"]:
                thresholds = sorted(set(thresholds + np.arange(cfg["threshold_min"], cfg["threshold_max"]+1e-9, cfg["threshold_step"]).round(8).tolist()))
            best_key, best_report, best_threshold = None, None, None
            for threshold in thresholds:
                report = replay_candidates(frame, "validation", lambda obs: int(model.predict_proba(obs[None, :])[0, 1] >= threshold),
                                           cfg, data_meta, check_cancel, progress)
                key = selection_key(report, training_cfg)
                if best_key is None or key > best_key:
                    best_key, best_report, best_threshold = key, report, threshold
            metadata.update(validation=best_report, probability_threshold=best_threshold)
        else:
            vec.training = False
            report = replay_candidates(frame, "validation", lambda obs: int(model.predict(vec.normalize_obs(obs), deterministic=True)[0]),
                                       cfg, data_meta, check_cancel, progress)
            selection_key(report, training_cfg)
            metadata["validation"] = report
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        temporary.replace(path.with_suffix(".json"))
        return path, metadata
    finally:
        if vec is not None:
            vec.close()


def evaluate_candidate_model(path, split="validation", dataset=DEFAULT_DATASET,
                             check_cancel=None, progress=None):
    if split not in ("validation", "test"):
        raise ValueError("Evaluate validation or test only")
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    frame, data_meta = load_candidate_dataset(dataset)
    if (metadata.get("version") != CANDIDATE_VERSION
            or metadata.get("feature_columns") != list(CANDIDATE_FEATURES)
            or metadata.get("dataset_id") != data_meta["dataset_id"]):
        raise ValueError("Model belongs to a different candidate dataset/schema")
    vec = None
    predictions = None
    try:
        if metadata["kind"] == "supervised":
            import joblib
            model = joblib.load(path)
            threshold = metadata["probability_threshold"]
            predictions = predict_candidate_quality(model, frame.loc[frame.split == split])
            decide = lambda obs: int(model.predict_proba(obs[None, :])[0, 1] >= threshold)
        else:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            model = PPO.load(path, device="cpu")
            vec = VecNormalize.load(path.with_suffix(".vec.pkl"), DummyVecEnv([lambda: CandidateBankNiftyEnv(frame, split, False)]))
            vec.training = False
            vec.norm_reward = False
            decide = lambda obs: int(model.predict(vec.normalize_obs(obs), deterministic=True)[0])
        report = replay_candidates(frame, split, decide, metadata["candidate_training"], data_meta, check_cancel, progress)
        selection_key(report, metadata["training"])
        report_file = path.with_name(path.stem+f"_{split}_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".report.json")
        if predictions is not None:
            prediction_file = report_file.with_suffix(".predictions.parquet")
            predictions.to_parquet(prediction_file, index=False)
            report["predictions_file"] = str(prediction_file)
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_file"] = str(report_file)
        return report
    finally:
        if vec is not None:
            vec.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("supervised", "ppo", "evaluate"))
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, help="Exported website settings")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    if args.command == "evaluate":
        if not args.model:
            parser.error("evaluate requires --model")
        result = evaluate_candidate_model(args.model, args.split, args.dataset)
    else:
        path, result = train_candidate_model(args.command, args.dataset, settings.get("candidate_training"),
                                              settings.get("training"), settings.get("environment"),
                                              progress=lambda p: print(p))
        result = {**result, "model_path": str(path)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
