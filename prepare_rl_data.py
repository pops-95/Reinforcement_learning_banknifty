"""Offline preparation from existing files. No downloads or model training.

Preserves existing chronological split labels unless explicit boundaries are
provided. Writes a separate data/rl dataset; source data is never overwritten.
Missing indicators become zero WITH per-field availability flags. Execution
quotes are never forward/back-filled. Incomplete days remain in evaluation.
"""
import argparse
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from rl_data import BASE, DATA_VERSION, OPTIONS, RL_DIR, sha256_file, validate_splits

OPTION_METRICS = (
    "open", "high", "low", "close", "option_ret_1m", "option_ret_3m",
    "log_volume", "log_oi", "volume_change", "oi_change", "oi_change_pct",
    "iv", "delta", "gamma", "theta", "vega", "rho", "moneyness",
    "intrinsic", "time_value",
)


def timestamp(values):
    ts = pd.to_datetime(values)
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return ts


def option_features(options):
    """Only consecutive same-contract, same-session data contributes changes."""
    o = options.copy()
    o["timestamp"] = timestamp(o.timestamp)
    o["groww_symbol"] = o.groww_symbol.astype(str)
    if o.duplicated(["groww_symbol", "timestamp"]).any():
        raise ValueError("Duplicate option contract/timestamp records")
    o = o.sort_values(["groww_symbol", "timestamp"])
    for name in (*OPTION_METRICS, "volume", "oi", "strike", "underlying_close"):
        if name not in o:
            o[name] = np.nan
        o[name] = pd.to_numeric(o[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    prices = o[["open", "high", "low", "close"]]
    valid = (prices.notna().all(axis=1) & prices.gt(0).all(axis=1)
             & o.high.ge(prices.max(axis=1)) & o.low.le(prices.min(axis=1)))
    o.loc[~valid, ["open", "high", "low", "close"]] = np.nan
    o.loc[o.volume < 0, "volume"] = np.nan
    o.loc[o.oi < 0, "oi"] = np.nan
    bad_iv = o.iv.le(0.005001) | o.iv.ge(4.99999) | o.iv.isna()
    o.loc[bad_iv, ["iv", "delta", "gamma", "theta", "vega", "rho"]] = np.nan
    groups = o.groupby([o.groww_symbol, o.timestamp.dt.date], observed=True, sort=False)
    for periods in (1, 3):
        previous = groups.close.shift(periods)
        consecutive = (o.timestamp - groups.timestamp.shift(periods)).eq(pd.Timedelta(minutes=periods))
        o[f"option_ret_{periods}m"] = (o.close / previous.replace(0, np.nan) - 1).where(consecutive)
    consecutive = (o.timestamp - groups.timestamp.shift()).eq(pd.Timedelta(minutes=1))
    for field in ("volume", "oi"):
        prev = groups[field].shift()
        change = (o[field] / prev.replace(0, np.nan) - 1).where(consecutive)
        o["volume_change" if field == "volume" else "oi_change_pct"] = change.clip(-5, 5)
        o[f"log_{field}"] = np.log1p(o[field])
    o["oi_change"] = (o.oi - groups.oi.shift()).where(consecutive)
    return o


def build(source, underlying, output, train_end=None, validation_end=None, overwrite=False):
    source, underlying, output = Path(source), Path(underlying), Path(output)
    target, manifest = output / "observations.parquet", output / "manifest.json"
    if (target.exists() or manifest.exists()) and not overwrite:
        raise ValueError("Prepared files exist. Stop web/CLI jobs and pass --overwrite to rebuild")
    original = pd.read_parquet(source, columns=["timestamp", "expiry", "split"])
    original["timestamp"] = timestamp(original.timestamp)
    original["expiry"] = timestamp(original.expiry).dt.normalize()
    identity = original[["timestamp", "expiry", "split"]].copy()
    if identity.expiry.isna().any():
        raise ValueError("Missing expiry assignments; cannot choose an execution partition")
    if bool(train_end) != bool(validation_end):
        raise ValueError("Supply both --train-end and --validation-end, or neither")
    if train_end:
        a = pd.Timestamp(train_end).normalize() + pd.Timedelta(days=1)
        b = pd.Timestamp(validation_end).normalize() + pd.Timedelta(days=1)
        identity["split"] = np.where(identity.timestamp < a, "train",
                                     np.where(identity.timestamp < b, "validation", "test"))
    validate_splits(identity)
    u = pd.read_parquet(underlying)
    u["timestamp"] = timestamp(u.timestamp)
    if u.timestamp.duplicated().any():
        raise ValueError("Underlying timestamps must be unique")
    # No target/outcome columns, symbol IDs or index volume in observations.
    excluded = {"timestamp", "date", "expiry", "split", "trading_date", "volume"}
    underlying_features = [c for c in u if c not in excluded
                           and pd.api.types.is_numeric_dtype(u[c])
                           and not re.search(r"future|forward_|label|outcome|target_hit|next_return|exec_", c)]
    base = identity.merge(u[["timestamp", *underlying_features]], on="timestamp", how="left", validate="one_to_one")
    prices = base[["open", "high", "low", "close"]]
    if (not np.isfinite(prices.to_numpy(dtype=float)).all() or not prices.gt(0).all().all()
            or base.high.lt(prices.max(axis=1)).any() or base.low.gt(prices.min(axis=1)).any()):
        raise ValueError("Underlying OHLC missing/invalid at observation timestamps; repair source prices")
    parts, signatures, partition_quality = [], {}, []
    for expiry, b in base.groupby("expiry", sort=True):
        path = OPTIONS / f"banknifty_options_{expiry:%Y-%m-%d}.parquet"
        if not path.exists():
            raise ValueError(f"Missing partition {path}; no synthetic prices will be created")
        o = option_features(pd.read_parquet(path))
        stat = path.stat()
        signatures[path.name] = [stat.st_size, stat.st_mtime_ns]
        step_values = pd.to_numeric(o.get("strike_step", pd.Series([100.])), errors="coerce").dropna()
        spacing = float(step_values.median()) if len(step_values) else 100.
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError(f"Invalid strike spacing in {path.name}")
        b = b.copy()
        # Exact arithmetic ATM, independent of which strikes happen to be downloaded.
        b["atm_strike"] = np.floor(b.close / spacing + 0.5) * spacing
        q = o.merge(b[["timestamp", "atm_strike"]], on="timestamp", how="inner", validate="many_to_one")
        offset = (q.strike - q.atm_strike) / spacing
        q = q.loc[np.isclose(offset, offset.round()) & offset.abs().le(5)].copy()
        q["offset"] = offset.loc[q.index].round().astype(int)
        q["slot"] = q.option_type.astype(str).str.lower() + "_" + q.offset.map(
            lambda n: "atm" if n == 0 else f"{'m' if n < 0 else 'p'}{abs(n)}")
        if q.duplicated(["timestamp", "slot"]).any():
            raise ValueError(f"Duplicate strike/type snapshot in {path.name}")
        fields = {}
        for side in ("ce", "pe"):
            for n in range(-5, 6):
                slot = f"{side}_" + ("atm" if n == 0 else f"{'m' if n < 0 else 'p'}{abs(n)}")
                selected = q.loc[q.slot == slot].set_index("timestamp").reindex(b.timestamp)
                for metric in OPTION_METRICS:
                    fields[f"{slot}_{metric}"] = selected[metric].to_numpy()
                fields[f"{slot}_available"] = selected.close.notna().to_numpy(dtype=np.float32)
                if n == 0:
                    b[f"atm_{side}_symbol"] = selected.groww_symbol.where(selected.close.notna()).to_numpy()
        remaining = (expiry + pd.Timedelta(hours=15, minutes=30) - b.timestamp).dt.total_seconds().clip(lower=0)
        b["dte_days"], b["tte_years"] = remaining / 86400, remaining / (365 * 86400)
        part = pd.concat([b.reset_index(drop=True), pd.DataFrame(fields)], axis=1)
        parts.append(part)
        partition_quality.append(dict(file=path.name, rows=len(o), oi_missing=int(o.oi.isna().sum()),
                                      volume_missing=int(o.volume.isna().sum()), iv_missing=int(o.iv.isna().sum())))
        print(f"Prepared {path.name}: {len(part):,} minutes", flush=True)
    frame = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    splits = validate_splits(frame)
    numeric = [*underlying_features, "dte_days", "tte_years", *fields.keys()]
    numeric = list(dict.fromkeys(numeric))
    values = frame[numeric].replace([np.inf, -np.inf], np.nan)
    flags = {f"{c}__available": values[c].notna().astype(np.float32)
             for c in numeric if not c.endswith("_available")}
    frame[numeric] = values.fillna(0.0).clip(-1e10, 1e10).astype(np.float32)
    frame = pd.concat([frame, pd.DataFrame(flags)], axis=1)
    features = numeric + list(flags)
    daily = frame.assign(day=frame.timestamp.dt.date).groupby(["split", "day"]).agg(
        rows=("timestamp", "size"), ce=("ce_atm_available", "mean"), pe=("pe_atm_available", "mean"))
    quality = dict(option_partitions=partition_quality,
                   days_with_low_atm_coverage=[dict(split=s, date=str(d), rows=int(r.rows),
                                                   ce_fraction=float(r.ce), pe_fraction=float(r.pe))
                                             for (s, d), r in daily.iterrows() if min(r.ce, r.pe) < .9],
                   note="All dates retained. Missing quotes block entries, missing indicators get explicit flags. No future-based day filtering.")
    output.mkdir(parents=True, exist_ok=True)
    meta = dict(dataset_version=DATA_VERSION, observation_columns=features, splits=splits,
                row_counts=frame.split.value_counts().to_dict(), option_files=signatures, quality=quality,
                source_observations=str(source.resolve()), underlying=str(underlying.resolve()))
    with tempfile.TemporaryDirectory(dir=output, prefix=".preparing_") as temp:
        staged = Path(temp) / target.name
        frame.to_parquet(staged, index=False)
        meta["dataset_id"] = sha256_file(staged)
        staged_meta = Path(temp) / manifest.name
        staged_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        staged.replace(target)
        staged_meta.replace(manifest)
    print(json.dumps(dict(observations=str(target), manifest=str(manifest), splits=splits,
                          features=len(features), low_coverage_days=len(quality["days_with_low_atm_coverage"])), indent=2))
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE / "data/sb3/banknifty_sb3_observations.parquet")
    parser.add_argument("--underlying", type=Path, default=BASE / "data/processed/banknifty_1min_features.parquet")
    parser.add_argument("--output", type=Path, default=RL_DIR)
    parser.add_argument("--train-end")
    parser.add_argument("--validation-end")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.source, args.underlying, args.output, args.train_end, args.validation_end, args.overwrite)
