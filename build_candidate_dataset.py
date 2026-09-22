"""Build counterfactual reference-candidate labels without training any model."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from banknifty_env import BankNiftyEnv, ENV_DEFAULTS, validate_config
from reference_strategy import CANDIDATE_VERSION, CANDIDATE_VALUES, CANDIDATE_FEATURES, reference_candidate_table
from rl_data import OBSERVATIONS, MANIFEST, OPTIONS

BASE = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE / "data/candidates/banknifty_candidate_dataset.parquet"

# Entry selection belongs to the learner, not a stack of minute-level gates.
# Reward is net realized R, independent of entry count or shaping history.
CANDIDATE_OVERRIDES = {
    "reference_strategy_enabled": True, "learned_exit_enabled": False,
    **{key: False for key in (
        "tradeability_gate_enabled", "momentum_gate_enabled", "path_efficiency_gate_enabled",
        "atr_expansion_gate_enabled", "extension_gate_enabled", "trend_alignment_gate_enabled",
        "price_oi_gate_enabled", "option_liquidity_gate_enabled", "option_delta_gate_enabled",
        "regime_filter_enabled", "entry_penalty_enabled", "overtrading_penalty_enabled",
        "hold_shaping_enabled", "giveback_penalty_enabled", "terminal_win_shaping_enabled",
        "terminal_loss_shaping_enabled", "agent_exit_penalty_enabled",
        "exit_reason_shaping_enabled", "momentum_acceleration_gate_enabled", "structural_distance_gate_enabled",
    )},
    "reward_scale": 1.0,
}
CANDIDATE_ENV_DEFAULTS = {
    **ENV_DEFAULTS, **CANDIDATE_OVERRIDES,
    "reference_ignition_min": 0.55, "reference_direction_gap": 0.03,
    "reference_signal_ttl_minutes": 4,
    "risk_reward_target_enabled": True, "risk_reward_target_r": 1.5,
    "momentum_decay_exit_enabled": True, "underlying_target_enabled": True,
}


class CandidateCancelled(Exception):
    pass


def candidate_environment_config(config=None):
    return validate_config({**CANDIDATE_ENV_DEFAULTS, **(config or {}), **CANDIDATE_OVERRIDES})


def config_fingerprint(config):
    payload = json.dumps(dict(version=CANDIDATE_VERSION, environment=config), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LabelSimulator(BankNiftyEnv):
    def _compute_candidate_recall_diagnostics(self, horizon=15):
        # The builder labels actual option trades instead of future spot proxies.
        return {}


def build_candidate_dataset(config=None, output=DEFAULT_DATASET, check_cancel=None,
                            progress=None, observations=None, options=None, manifest=None):
    cfg = candidate_environment_config(config)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = dict(observations_file=observations or OBSERVATIONS,
                 option_market_dir=options or OPTIONS,
                 manifest_file=manifest or MANIFEST)
    rows, split_days = [], {}
    for split in ("train", "validation", "test"):
        if check_cancel:
            check_cancel()
        env = LabelSimulator(**paths, split=split, random_day=False, **cfg)
        try:
            split_days[split] = len(env.days)
            for day_index in range(len(env.days)):
                if check_cancel:
                    check_cancel()
                env.reset()
                events = reference_candidate_table(env.day_df, env._get_option_row, cfg, env._market_regime)
                for event in events.to_dict("records"):
                    if check_cancel:
                        check_cancel()
                    trade = env.simulate_reference_candidate(event["bar_index"], check_cancel)
                    realized = float(trade["R_return"]) if trade else 0.0
                    rows.append({**event, "split": split, "filled": trade is not None,
                                 "realized_R": realized, "profitable": bool(trade and realized > 0),
                                 "hit_1R": bool(trade and trade["MFE_R"] >= 1.0),
                                 "hit_1_5R": bool(trade and trade["MFE_R"] >= 1.5),
                                 "MFE_R": trade["MFE_R"] if trade else None,
                                 "MAE_R": trade["MAE_R"] if trade else None,
                                 "stale_exit": bool(trade and trade["stale_exit"]),
                                 "exit_reason": trade["exit_reason"] if trade else "ENTRY_REJECTED",
                                 "holding_minutes": trade["holding_minutes"] if trade else 0.0,
                                 "next_decision_time": trade["next_decision_time"] if trade else event["timestamp"],
                                 "trade_json": json.dumps(trade, default=str) if trade else "null"})
                if progress:
                    progress(dict(split=split, day=str(env.current_day),
                                  completed_days=day_index+1, total_days=len(env.days),
                                  reference_candidates=len(rows)))
        finally:
            env.close()
    if not rows:
        raise ValueError("No reference candidates. Lower reference score/gap thresholds and rebuild.")
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    if frame.candidate_id.duplicated().any():
        raise ValueError("Duplicate candidate IDs across chronological splits")
    metadata = dict(version=CANDIDATE_VERSION, created_at=datetime.now(timezone.utc).isoformat(),
                    environment=cfg, config_fingerprint=config_fingerprint(cfg),
                    feature_columns=list(CANDIDATE_FEATURES), split_days=split_days,
                    rows=len(frame), counts=frame.groupby("split").size().to_dict(),
                    outcome_note="Independent counterfactual trades may overlap; replay enforces occupancy/cooldown.",
                    hit_note="hit_1R/hit_1_5R refer to observed gross MFE, not guaranteed executable net profit.")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".building_") as temp:
        staged = Path(temp) / output.name
        frame.to_parquet(staged, index=False)
        metadata["dataset_id"] = file_digest(staged)
        staged_meta = staged.with_suffix(".json")
        staged_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if check_cancel:
            check_cancel()
        staged.replace(output)
        staged_meta.replace(output.with_suffix(".json"))
    return metadata


def load_candidate_dataset(path=DEFAULT_DATASET, expected_config=None):
    path = Path(path)
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError("Build the candidate dataset first")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("version") != CANDIDATE_VERSION or metadata.get("feature_columns") != list(CANDIDATE_FEATURES):
        raise ValueError("Candidate schema changed; rebuild the dataset")
    if metadata.get("dataset_id") != file_digest(path):
        raise ValueError("Candidate dataset/manifest mismatch; rebuild before training")
    if metadata.get("config_fingerprint") != config_fingerprint(candidate_environment_config(metadata.get("environment", {}))):
        raise ValueError("Candidate settings metadata is inconsistent; rebuild the dataset")
    if expected_config is not None and metadata.get("config_fingerprint") != config_fingerprint(candidate_environment_config(expected_config)):
        raise ValueError("Settings differ from the candidate dataset. Rebuild it using current settings first.")
    frame = pd.read_parquet(path)
    required = {"timestamp", "split", "candidate_id", "side", "filled", "realized_R",
                "next_decision_time", "trade_json", "stale_exit", *CANDIDATE_VALUES}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing candidate columns: {sorted(required-set(frame.columns))}")
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    if not np.isfinite(frame.realized_R.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite candidate rewards; rebuild after correcting market data")
    if not frame.split.isin(("train", "validation", "test")).all():
        raise ValueError("Unknown chronological split in candidate dataset")
    if frame.candidate_id.duplicated().any():
        raise ValueError("Duplicate candidate IDs")
    previous_end = None
    for split in ("train", "validation", "test"):
        subset = frame.loc[frame.split == split]
        if subset.empty:
            raise ValueError(f"No candidates in {split}; adjust reference settings and rebuild")
        start, end = subset.timestamp.min(), subset.timestamp.max()
        if previous_end is not None and start.normalize() <= previous_end.normalize():
            raise ValueError("Candidate splits must use disjoint chronological days")
        previous_end = end
    return frame, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Environment JSON or exported website settings")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()
    payload = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    cfg = payload.get("environment", payload)
    print(json.dumps(build_candidate_dataset(cfg, args.output, progress=lambda p: print(p)), indent=2))


if __name__ == "__main__":
    main()
