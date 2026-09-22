"""Shared data contract for the autonomous PPO trainer (no API or CUDA calls)."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_VERSION = "autonomous_observations_v1"
BASE = Path(__file__).resolve().parent
RL_DIR = BASE / "data/rl"
OBSERVATIONS = RL_DIR / "observations.parquet"
MANIFEST = RL_DIR / "manifest.json"
OPTIONS = BASE / "data/sb3/option_market"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_group(name):
    """Stable grouping; indicators and their availability flags share a switch."""
    n = name.lower().removesuffix("__available")
    if "oi" in n and any(t in n for t in ("_oi", "oi_", "log_oi")):
        return "oi"
    if "volume" in n:
        return "volume"
    if any(t in n for t in ("delta", "gamma", "theta", "vega", "rho", "_iv")):
        return "greeks"
    if any(t in n for t in ("ema", "trend", "rsi")):
        return "trend"
    if any(t in n for t in ("atr", "volatility")):
        return "volatility"
    if any(t in n for t in ("return", "momentum", "velocity", "acceleration", "efficiency")):
        return "momentum"
    if n in ("hour", "minute", "minutes_from_open", "time_sin", "time_cos", "dte_days", "tte_years"):
        return "time"
    return "price"


def validate_splits(frame):
    if frame.empty or frame.timestamp.isna().any() or frame.timestamp.duplicated().any():
        raise ValueError("Observations must have unique, nonmissing timestamps")
    if not frame.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Every row must belong to train, validation or test")
    dates = frame.timestamp.dt.date
    if frame.assign(_date=dates).groupby("_date").split.nunique().max() != 1:
        raise ValueError("A trading day cannot cross dataset splits")
    end = None
    result = {}
    for split in ("train", "validation", "test"):
        ts = frame.loc[frame.split == split, "timestamp"]
        if ts.empty or (end is not None and ts.min() <= end):
            raise ValueError("Need nonempty, disjoint chronological train/validation/test periods")
        end = ts.max()
        result[split] = dict(rows=len(ts), days=int(ts.dt.date.nunique()),
                             start=str(ts.min()), end=str(ts.max()))
    return result


@lru_cache(maxsize=1)
def _load_checked(obs_path, manifest_path, obs_stamp, manifest_stamp):
    meta = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if meta.get("dataset_version") != DATA_VERSION:
        raise ValueError("Run prepare_rl_data.py to prepare autonomous PPO observations")
    if sha256_file(obs_path) != meta.get("dataset_id"):
        raise ValueError("Observation/manifest mismatch. Rebuild with prepare_rl_data.py")
    df = pd.read_parquet(obs_path)
    df["timestamp"] = pd.to_datetime(df.timestamp)
    df["expiry"] = pd.to_datetime(df.expiry)
    validate_splits(df)
    columns = meta["observation_columns"]
    if not set(columns).issubset(df.columns):
        raise ValueError("Manifest features are missing from observations")
    if not np.isfinite(df[columns].to_numpy(dtype=np.float32)).all():
        raise ValueError("Prepared observations contain NaN/Inf; rebuild them")
    return df, meta


def load_dataset(observations=OBSERVATIONS, manifest=MANIFEST, options=OPTIONS):
    obs, meta_path = Path(observations), Path(manifest)
    if not obs.is_file() or not meta_path.is_file():
        raise ValueError("Prepared data missing. Run: python prepare_rl_data.py")
    df, meta = _load_checked(str(obs.resolve()), str(meta_path.resolve()),
                            obs.stat().st_mtime_ns, meta_path.stat().st_mtime_ns)
    for name, signature in meta.get("option_files", {}).items():
        path = Path(options) / name
        if not path.is_file():
            raise ValueError(f"Missing execution partition: {path}")
        stat = path.stat()
        if [stat.st_size, stat.st_mtime_ns] != signature:
            raise ValueError(f"Execution data changed ({name}); rebuild prepared data and retrain")
    return df, meta
