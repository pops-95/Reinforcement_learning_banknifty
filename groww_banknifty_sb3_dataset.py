#!/usr/bin/env python3
"""
Groww BANKNIFTY -> Stable-Baselines3 Dataset Builder (v2)

Prerequisite input (one required):
    data/processed/banknifty_1min_features.parquet
or
    data/raw/banknifty_1min_raw.parquet

Outputs:
    data/sb3/banknifty_sb3_observations.parquet
    data/sb3/dataset_manifest.json
    data/sb3/option_market/banknifty_options_YYYY-MM-DD.parquet
    data/sb3/observation_parts/observations_YYYY-MM-DD.parquet

Key fixes in this version:
- Never concatenates all historical option rows into one huge DataFrame.
- Stores exact option market data expiry-by-expiry.
- Uses observed=True in pandas groupby/pivot_table.
- Excludes BANKNIFTY index volume from RL observations.
- Missing option values become 0 with explicit *_available masks.
- Missing OI-change/Greek values no longer delete entire timestamps.
- Historical IV/Greeks are derived from historical option prices.
- Uses CuPy/CUDA for IV/Greek calculations when available.
- Produces chronological train/validation/test splits.

Intended V1 actions:
    0 = WAIT
    1 = BUY ATM CE
    2 = BUY ATM PE

Important:
A decision made after candle t closes must be executed no earlier than t+1.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from growwapi import GrowwAPI

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "data"

START_DATE = pd.Timestamp("2025-01-01")
END_DATE = pd.Timestamp("2026-08-31")
UNDERLYING_SYMBOL = "BANKNIFTY"

ATM_WINGS = 5
FETCH_EXTRA_WINGS = 2

RISK_FREE_RATE = 0.065
DIVIDEND_YIELD = 0.0

REQUEST_CHUNK_DAYS = 25
REQUEST_SLEEP_SEC = 1.5
API_MIN_INTERVAL_SEC = 2.0
API_RETRY_BASE_SEC = 5.0
MAX_RETRIES = 7

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

TRAIN_END = pd.Timestamp("2026-01-31 23:59:59")
VALIDATION_END = pd.Timestamp("2026-07-31 23:59:59")

MAX_MISSING_FEATURE_RATIO = 0.95

RAW_UNDERLYING = DATA_ROOT / "raw" / "banknifty_1min_raw.parquet"
FEATURE_UNDERLYING = DATA_ROOT / "processed" / "banknifty_1min_features.parquet"

CACHE_ROOT = DATA_ROOT / "option_cache"
CONTRACT_CACHE = CACHE_ROOT / "contracts"
CANDLE_CACHE = CACHE_ROOT / "candles"

OUT_ROOT = DATA_ROOT / "sb3"
OPTION_DIR = OUT_ROOT / "option_market"
OBS_PARTS_DIR = OUT_ROOT / "observation_parts"

OBS_FILE = OUT_ROOT / "banknifty_sb3_observations.parquet"
MANIFEST_FILE = OUT_ROOT / "dataset_manifest.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("BANKNIFTY_SB3")

# =============================================================================
# CUDA
# =============================================================================

CUDA_AVAILABLE = False
cp = None
gpu_erf = None

try:
    import cupy as _cp
    from cupyx.scipy.special import erf as _gpu_erf

    if _cp.cuda.runtime.getDeviceCount() > 0:
        cp = _cp
        gpu_erf = _gpu_erf
        CUDA_AVAILABLE = True
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        log.info("CUDA enabled for IV/Greeks: %s", gpu_name)
except Exception as exc:
    log.warning("CUDA/CuPy unavailable; IV/Greeks use CPU: %s", exc)

# Global Groww API throttle state.
# Every API request passes through api_call(), which enforces a minimum delay
# between requests and adds a short pause after successful responses.
_LAST_API_CALL_MONOTONIC = 0.0


def _wait_for_api_slot():
    """Enforce a minimum spacing between Groww API requests."""
    global _LAST_API_CALL_MONOTONIC
    now = time.monotonic()
    elapsed = now - _LAST_API_CALL_MONOTONIC
    wait = max(0.0, API_MIN_INTERVAL_SEC - elapsed)
    if wait > 0:
        log.info("Groww API throttle: sleeping %.2f sec", wait)
        time.sleep(wait)
    _LAST_API_CALL_MONOTONIC = time.monotonic()


# =============================================================================
# BASIC HELPERS
# =============================================================================

def create_directories():
    for p in [CONTRACT_CACHE, CANDLE_CACHE, OUT_ROOT, OPTION_DIR, OBS_PARTS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def create_groww_client():
    token = os.getenv("GROWW_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "\nGROWW_ACCESS_TOKEN not found.\n\n"
            "PowerShell:\n"
            '  $env:GROWW_ACCESS_TOKEN="YOUR_TOKEN"\n\n'
            "Linux/macOS:\n"
            '  export GROWW_ACCESS_TOKEN="YOUR_TOKEN"\n'
        )
    return GrowwAPI(token)


def api_call(func, *args, **kwargs):
    """Call Groww with global throttling and exponential retry backoff."""
    global _LAST_API_CALL_MONOTONIC
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        _wait_for_api_slot()

        try:
            result = func(*args, **kwargs)

            # Pause even after a successful request so consecutive calls are not
            # fired back-to-back.
            time.sleep(REQUEST_SLEEP_SEC)
            _LAST_API_CALL_MONOTONIC = time.monotonic()
            return result

        except Exception as exc:
            last_error = exc
            retry_wait = min(API_RETRY_BASE_SEC * attempt, 30.0)
            log.warning(
                "API attempt %d/%d failed: %s | retrying after %.1f sec",
                attempt, MAX_RETRIES, exc, retry_wait
            )
            time.sleep(retry_wait)
            _LAST_API_CALL_MONOTONIC = time.monotonic()

    raise last_error


def parse_market_timestamp(values):
    s = pd.Series(values)
    if pd.api.types.is_numeric_dtype(s):
        dt = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
        return dt.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)

    dt = pd.to_datetime(s, errors="coerce")
    try:
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    except Exception:
        pass
    return dt


def month_iterator(start, end):
    current = pd.Timestamp(start.year, start.month, 1)
    final = pd.Timestamp(end.year, end.month, 1)
    while current <= final:
        yield current.year, current.month
        current = current + pd.offsets.MonthBegin(1)


def chunk_dates(start_date, end_date):
    current = pd.Timestamp(start_date).normalize()
    final = pd.Timestamp(end_date).normalize()
    while current <= final:
        chunk_end = min(current + pd.Timedelta(days=REQUEST_CHUNK_DAYS - 1), final)
        yield current, chunk_end
        current = chunk_end + pd.Timedelta(days=1)


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)

# =============================================================================
# UNDERLYING
# =============================================================================

def load_underlying():
    path = FEATURE_UNDERLYING if FEATURE_UNDERLYING.exists() else RAW_UNDERLYING
    if not path.exists():
        raise FileNotFoundError(
            "Underlying file not found.\n"
            "Expected either:\n"
            f"  {FEATURE_UNDERLYING}\n"
            f"  {RAW_UNDERLYING}\n"
        )

    log.info("Loading underlying: %s", path)
    df = pd.read_parquet(path)

    required = {"timestamp", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Underlying missing columns: " + ", ".join(sorted(missing)))

    df["timestamp"] = parse_market_timestamp(df["timestamp"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
    df = df[
        (df["timestamp"] >= START_DATE)
        & (df["timestamp"] < END_DATE + pd.Timedelta(days=1))
    ].copy()

    clock = df["timestamp"].dt.strftime("%H:%M")
    df = df[(clock >= MARKET_OPEN) & (clock <= MARKET_CLOSE)].copy()
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates("timestamp", keep="last", inplace=True)
    df.index = range(len(df))

    suspicious = [
        c for c in df.columns
        if any(
            token in c.lower()
            for token in [
                "future", "forward_return", "target_hit", "outcome",
                "label", "next_return"
            ]
        )
    ]
    if suspicious:
        raise ValueError(
            "Potential future-leakage columns found: " + ", ".join(suspicious)
        )

    log.info(
        "Underlying rows=%d | %s -> %s",
        len(df), df["timestamp"].min(), df["timestamp"].max()
    )
    return df

# =============================================================================
# EXPIRIES / CONTRACTS
# =============================================================================

def fetch_expiries(groww):
    expiries = set()
    query_end = END_DATE + pd.offsets.MonthBegin(1)

    for year, month in month_iterator(START_DATE, query_end):
        response = api_call(
            groww.get_expiries,
            exchange=groww.EXCHANGE_NSE,
            underlying_symbol=UNDERLYING_SYMBOL,
            year=year,
            month=month,
        )
        for value in response.get("expiries", []):
            expiries.add(pd.Timestamp(value).normalize())

        # Extra spacing between month-by-month expiry requests.
        time.sleep(1.0)

    result = sorted(expiries)
    if not result:
        raise RuntimeError("Groww returned no BANKNIFTY expiries.")

    log.info("Fetched %d BANKNIFTY expiries.", len(result))
    return result


def assign_front_expiry(underlying, expiries):
    expiry_array = np.array(expiries, dtype="datetime64[ns]")
    dates = underlying["timestamp"].dt.normalize().values.astype("datetime64[ns]")
    idx = np.searchsorted(expiry_array, dates, side="left")
    valid = idx < len(expiry_array)

    selected = np.full(len(underlying), np.datetime64("NaT"), dtype="datetime64[ns]")
    selected[valid] = expiry_array[idx[valid]]

    out = underlying.copy()
    out["expiry"] = pd.to_datetime(selected)
    out.dropna(subset=["expiry"], inplace=True)

    expiry_clock = out["expiry"] + pd.Timedelta(hours=15, minutes=30)
    seconds = (expiry_clock - out["timestamp"]).dt.total_seconds().clip(lower=60)
    out["tte_years"] = (seconds / (365.0 * 24.0 * 3600.0)).astype(np.float32)
    out["dte_days"] = (seconds / 86400.0).astype(np.float32)
    return out


CONTRACT_RE = re.compile(
    r"^NSE-BANKNIFTY-(?P<exp>\d{2}[A-Za-z]{3}\d{2})-"
    r"(?P<strike>\d+(?:\.\d+)?)-(?P<type>CE|PE)$",
    re.IGNORECASE,
)


def parse_contract(symbol):
    match = CONTRACT_RE.match(symbol)
    if not match:
        return None
    return {
        "groww_symbol": symbol,
        "strike": float(match.group("strike")),
        "option_type": match.group("type").upper(),
    }


def contracts_for_expiry(groww, expiry):
    expiry_string = expiry.strftime("%Y-%m-%d")
    cache_file = CONTRACT_CACHE / f"{expiry_string}.json"

    if cache_file.exists():
        symbols = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        response = api_call(
            groww.get_contracts,
            exchange=groww.EXCHANGE_NSE,
            underlying_symbol=UNDERLYING_SYMBOL,
            expiry_date=expiry_string,
        )
        symbols = response.get("contracts", [])
        cache_file.write_text(json.dumps(symbols, indent=2), encoding="utf-8")

    rows = [parse_contract(x) for x in symbols]
    rows = [x for x in rows if x is not None]

    if not rows:
        return pd.DataFrame(
            columns=["groww_symbol", "strike", "option_type", "expiry"]
        )

    df = pd.DataFrame(rows)
    df["expiry"] = expiry
    return df.sort_values(["strike", "option_type"]).reset_index(drop=True)


def infer_strike_step(contracts):
    strikes = np.sort(contracts["strike"].dropna().unique())
    diffs = np.diff(strikes)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        raise RuntimeError("Unable to infer strike interval.")
    return float(pd.Series(diffs).round(4).mode().iloc[0])


def select_needed_contracts(contracts, underlying_expiry):
    if contracts.empty or underlying_expiry.empty:
        return contracts.iloc[0:0].copy(), np.nan

    step = infer_strike_step(contracts)
    margin = step * (ATM_WINGS + FETCH_EXTRA_WINGS)
    lo = float(underlying_expiry["close"].min()) - margin
    hi = float(underlying_expiry["close"].max()) + margin

    selected = contracts[
        (contracts["strike"] >= lo) & (contracts["strike"] <= hi)
    ].copy()
    return selected, step

# =============================================================================
# HISTORICAL OPTION CANDLES
# =============================================================================

def fetch_contract_candles(groww, symbol, start_date, end_date):
    cache_file = (
        CANDLE_CACHE
        / f"{safe_filename(symbol)}_{start_date.date()}_{end_date.date()}.parquet"
    )

    if cache_file.exists():
        return pd.read_parquet(cache_file)

    pieces = []

    for chunk_start, chunk_end in chunk_dates(start_date, end_date):
        response = api_call(
            groww.get_historical_candles,
            exchange=groww.EXCHANGE_NSE,
            segment=groww.SEGMENT_FNO,
            groww_symbol=symbol,
            start_time=f"{chunk_start:%Y-%m-%d} 09:15:00",
            end_time=f"{chunk_end:%Y-%m-%d} 15:30:00",
            candle_interval=groww.CANDLE_INTERVAL_MIN_1,
        )

        candles = response.get("candles", [])
        if not candles:
            continue

        rows = []
        for x in candles:
            rows.append(
                {
                    "timestamp": x[0] if len(x) > 0 else None,
                    "open": x[1] if len(x) > 1 else np.nan,
                    "high": x[2] if len(x) > 2 else np.nan,
                    "low": x[3] if len(x) > 3 else np.nan,
                    "close": x[4] if len(x) > 4 else np.nan,
                    "volume": x[5] if len(x) > 5 else np.nan,
                    "oi": x[6] if len(x) > 6 else np.nan,
                }
            )
        pieces.append(pd.DataFrame(rows))

    if not pieces:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "oi"]
        )

    df = pd.concat(pieces, ignore_index=True)
    df["timestamp"] = parse_market_timestamp(df["timestamp"])

    for c in ["open", "high", "low", "close", "volume", "oi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.dropna(subset=["timestamp", "open", "high", "low", "close"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates("timestamp", keep="last", inplace=True)
    df.index = range(len(df))

    clock = df["timestamp"].dt.strftime("%H:%M")
    df = df[(clock >= MARKET_OPEN) & (clock <= MARKET_CLOSE)].copy()

    df.to_parquet(cache_file, index=False, compression="zstd")
    return df

# =============================================================================
# BLACK-SCHOLES CPU
# =============================================================================

def normal_cdf_np(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def normal_pdf_np(x):
    x = np.asarray(x, dtype=np.float64)
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price_cpu(S, K, T, sigma, is_call):
    sqrt_t = np.sqrt(np.maximum(T, 1e-12))
    sigma = np.maximum(sigma, 1e-6)

    d1 = (
        np.log(np.maximum(S, 1e-12) / np.maximum(K, 1e-12))
        + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * sigma * sigma) * T
    ) / (sigma * sqrt_t)

    d2 = d1 - sigma * sqrt_t
    dq = np.exp(-DIVIDEND_YIELD * T)
    dr = np.exp(-RISK_FREE_RATE * T)

    call = S * dq * normal_cdf_np(d1) - K * dr * normal_cdf_np(d2)
    put = K * dr * normal_cdf_np(-d2) - S * dq * normal_cdf_np(-d1)
    return np.where(is_call, call, put)


def implied_vol_cpu(S, K, T, price, is_call):
    dq = np.exp(-DIVIDEND_YIELD * T)
    dr = np.exp(-RISK_FREE_RATE * T)

    intrinsic = np.where(
        is_call,
        np.maximum(S * dq - K * dr, 0.0),
        np.maximum(K * dr - S * dq, 0.0),
    )
    upper = np.where(is_call, S * dq, K * dr)

    valid = (
        np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(price)
        & (S > 0) & (K > 0) & (T > 0) & (price > 0)
        & (price >= intrinsic - 0.5) & (price <= upper + 0.5)
    )

    lo = np.full_like(S, 0.005, dtype=np.float64)
    hi = np.full_like(S, 5.0, dtype=np.float64)

    for _ in range(45):
        mid = (lo + hi) * 0.5
        theoretical = bs_price_cpu(S, K, T, mid, is_call)
        low_model = theoretical < price
        lo = np.where(low_model, mid, lo)
        hi = np.where(low_model, hi, mid)

    iv = (lo + hi) * 0.5
    iv[~valid] = np.nan
    return iv


def greeks_cpu(S, K, T, iv, is_call):
    sqrt_t = np.sqrt(np.maximum(T, 1e-12))
    sigma = np.maximum(iv, 1e-6)

    d1 = (
        np.log(np.maximum(S, 1e-12) / np.maximum(K, 1e-12))
        + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * sigma * sigma) * T
    ) / (sigma * sqrt_t)

    d2 = d1 - sigma * sqrt_t
    dq = np.exp(-DIVIDEND_YIELD * T)
    dr = np.exp(-RISK_FREE_RATE * T)
    cdf1 = normal_cdf_np(d1)
    cdf2 = normal_cdf_np(d2)
    pdf1 = normal_pdf_np(d1)

    delta_c = dq * cdf1
    delta_p = dq * (cdf1 - 1.0)
    gamma = dq * pdf1 / (S * sigma * sqrt_t)
    vega = S * dq * pdf1 * sqrt_t / 100.0

    common = -(S * dq * pdf1 * sigma) / (2.0 * sqrt_t)

    theta_c = (
        common - RISK_FREE_RATE * K * dr * cdf2
        + DIVIDEND_YIELD * S * dq * cdf1
    ) / 365.0

    theta_p = (
        common + RISK_FREE_RATE * K * dr * normal_cdf_np(-d2)
        - DIVIDEND_YIELD * S * dq * normal_cdf_np(-d1)
    ) / 365.0

    rho_c = K * T * dr * cdf2 / 100.0
    rho_p = -K * T * dr * normal_cdf_np(-d2) / 100.0
    valid = np.isfinite(iv)

    return {
        "delta": np.where(valid, np.where(is_call, delta_c, delta_p), np.nan),
        "gamma": np.where(valid, gamma, np.nan),
        "vega": np.where(valid, vega, np.nan),
        "theta": np.where(valid, np.where(is_call, theta_c, theta_p), np.nan),
        "rho": np.where(valid, np.where(is_call, rho_c, rho_p), np.nan),
    }

# =============================================================================
# BLACK-SCHOLES GPU
# =============================================================================

def implied_vol_gpu(S, K, T, price, is_call):
    Sg = cp.asarray(S, dtype=cp.float64)
    Kg = cp.asarray(K, dtype=cp.float64)
    Tg = cp.asarray(T, dtype=cp.float64)
    Pg = cp.asarray(price, dtype=cp.float64)
    Cg = cp.asarray(is_call, dtype=cp.bool_)

    def cdf(x):
        return 0.5 * (1.0 + gpu_erf(x / cp.sqrt(2.0)))

    def price_fn(sigma):
        sqrt_t = cp.sqrt(cp.maximum(Tg, 1e-12))
        sigma = cp.maximum(sigma, 1e-6)
        d1 = (
            cp.log(cp.maximum(Sg, 1e-12) / cp.maximum(Kg, 1e-12))
            + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * sigma * sigma) * Tg
        ) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        dq = cp.exp(-DIVIDEND_YIELD * Tg)
        dr = cp.exp(-RISK_FREE_RATE * Tg)
        call = Sg * dq * cdf(d1) - Kg * dr * cdf(d2)
        put = Kg * dr * cdf(-d2) - Sg * dq * cdf(-d1)
        return cp.where(Cg, call, put)

    dq = cp.exp(-DIVIDEND_YIELD * Tg)
    dr = cp.exp(-RISK_FREE_RATE * Tg)

    intrinsic = cp.where(
        Cg,
        cp.maximum(Sg * dq - Kg * dr, 0.0),
        cp.maximum(Kg * dr - Sg * dq, 0.0),
    )
    upper = cp.where(Cg, Sg * dq, Kg * dr)

    valid = (
        cp.isfinite(Sg) & cp.isfinite(Kg) & cp.isfinite(Tg) & cp.isfinite(Pg)
        & (Sg > 0) & (Kg > 0) & (Tg > 0) & (Pg > 0)
        & (Pg >= intrinsic - 0.5) & (Pg <= upper + 0.5)
    )

    lo = cp.full_like(Sg, 0.005)
    hi = cp.full_like(Sg, 5.0)

    for _ in range(45):
        mid = (lo + hi) * 0.5
        theoretical = price_fn(mid)
        low_model = theoretical < Pg
        lo = cp.where(low_model, mid, lo)
        hi = cp.where(low_model, hi, mid)

    iv = (lo + hi) * 0.5
    iv = cp.where(valid, iv, cp.nan)
    return cp.asnumpy(iv)


def greeks_gpu(S, K, T, iv, is_call):
    Sg = cp.asarray(S, dtype=cp.float64)
    Kg = cp.asarray(K, dtype=cp.float64)
    Tg = cp.asarray(T, dtype=cp.float64)
    Vg = cp.asarray(iv, dtype=cp.float64)
    Cg = cp.asarray(is_call, dtype=cp.bool_)

    def cdf(x):
        return 0.5 * (1.0 + gpu_erf(x / cp.sqrt(2.0)))

    sqrt_t = cp.sqrt(cp.maximum(Tg, 1e-12))
    sigma = cp.maximum(Vg, 1e-6)

    d1 = (
        cp.log(cp.maximum(Sg, 1e-12) / cp.maximum(Kg, 1e-12))
        + (RISK_FREE_RATE - DIVIDEND_YIELD + 0.5 * sigma * sigma) * Tg
    ) / (sigma * sqrt_t)

    d2 = d1 - sigma * sqrt_t
    pdf1 = cp.exp(-0.5 * d1 * d1) / cp.sqrt(2.0 * cp.pi)
    dq = cp.exp(-DIVIDEND_YIELD * Tg)
    dr = cp.exp(-RISK_FREE_RATE * Tg)

    delta_c = dq * cdf(d1)
    delta_p = dq * (cdf(d1) - 1.0)
    gamma = dq * pdf1 / (Sg * sigma * sqrt_t)
    vega = Sg * dq * pdf1 * sqrt_t / 100.0

    common = -(Sg * dq * pdf1 * sigma) / (2.0 * sqrt_t)

    theta_c = (
        common - RISK_FREE_RATE * Kg * dr * cdf(d2)
        + DIVIDEND_YIELD * Sg * dq * cdf(d1)
    ) / 365.0

    theta_p = (
        common + RISK_FREE_RATE * Kg * dr * cdf(-d2)
        - DIVIDEND_YIELD * Sg * dq * cdf(-d1)
    ) / 365.0

    rho_c = Kg * Tg * dr * cdf(d2) / 100.0
    rho_p = -Kg * Tg * dr * cdf(-d2) / 100.0
    valid = cp.isfinite(Vg)

    result = {
        "delta": cp.where(valid, cp.where(Cg, delta_c, delta_p), cp.nan),
        "gamma": cp.where(valid, gamma, cp.nan),
        "vega": cp.where(valid, vega, cp.nan),
        "theta": cp.where(valid, cp.where(Cg, theta_c, theta_p), cp.nan),
        "rho": cp.where(valid, cp.where(Cg, rho_c, rho_p), cp.nan),
    }

    return {k: cp.asnumpy(v) for k, v in result.items()}


def add_iv_and_greeks(df):
    if df.empty:
        return df

    S = df["underlying_close"].to_numpy(np.float64)
    K = df["strike"].to_numpy(np.float64)
    T = df["tte_years"].to_numpy(np.float64)
    price = df["close"].to_numpy(np.float64)
    is_call = df["option_type"].astype(str).to_numpy() == "CE"

    if CUDA_AVAILABLE:
        iv = implied_vol_gpu(S, K, T, price, is_call)
        greek_values = greeks_gpu(S, K, T, iv, is_call)
    else:
        iv = implied_vol_cpu(S, K, T, price, is_call)
        greek_values = greeks_cpu(S, K, T, iv, is_call)

    df["iv"] = iv.astype(np.float32)
    for name, values in greek_values.items():
        df[name] = values.astype(np.float32)
    return df

# =============================================================================
# OPTION MARKET PARTITIONS
# =============================================================================

def build_option_market_partitions(groww, underlying):
    output_files = []

    for expiry, u in underlying.groupby("expiry", sort=True):
        expiry = pd.Timestamp(expiry).normalize()
        expiry_string = expiry.strftime("%Y-%m-%d")
        output_file = OPTION_DIR / f"banknifty_options_{expiry_string}.parquet"

        if output_file.exists():
            log.info("Reusing option partition: %s", output_file.name)
            output_files.append(output_file)
            continue

        contracts = contracts_for_expiry(groww, expiry)
        if contracts.empty:
            log.warning("No contracts for %s", expiry_string)
            continue

        selected, strike_step = select_needed_contracts(contracts, u)
        if selected.empty:
            log.warning("No selected strikes for %s", expiry_string)
            continue

        active_start = u["timestamp"].min().normalize()
        active_end = min(u["timestamp"].max().normalize(), expiry)

        log.info(
            "Expiry %s | active %s -> %s | step %.2f | contracts=%d",
            expiry_string,
            active_start.date(),
            active_end.date(),
            strike_step,
            len(selected),
        )

        ujoin = (
            u[["timestamp", "close", "tte_years", "dte_days"]]
            .rename(columns={"close": "underlying_close"})
        )

        parts = []

        for _, row in selected.iterrows():
            symbol = row["groww_symbol"]
            c = fetch_contract_candles(groww, symbol, active_start, active_end)
            if c.empty:
                continue

            c = c.merge(ujoin, on="timestamp", how="inner")
            if c.empty:
                continue

            c["groww_symbol"] = symbol
            c["expiry"] = expiry
            c["strike"] = np.float32(row["strike"])
            c["option_type"] = row["option_type"]
            c["strike_step"] = np.float32(strike_step)
            parts.append(c)

        if not parts:
            log.warning("No option candles assembled for %s", expiry_string)
            continue

        expiry_df = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()

        expiry_df["groww_symbol"] = expiry_df["groww_symbol"].astype("category")
        expiry_df["option_type"] = expiry_df["option_type"].astype("category")

        expiry_df.sort_values(["groww_symbol", "timestamp"], inplace=True)
        expiry_df.index = range(len(expiry_df))

        grp = expiry_df.groupby("groww_symbol", observed=True, sort=False)

        expiry_df["option_ret_1m"] = grp["close"].pct_change()
        expiry_df["option_ret_3m"] = grp["close"].pct_change(3)
        expiry_df["volume_change"] = grp["volume"].pct_change()
        expiry_df["oi_change"] = grp["oi"].diff()
        expiry_df["oi_change_pct"] = grp["oi"].pct_change()

        expiry_df["log_volume"] = np.log1p(expiry_df["volume"].clip(lower=0))
        expiry_df["log_oi"] = np.log1p(expiry_df["oi"].clip(lower=0))

        spot = expiry_df["underlying_close"].to_numpy()
        strike = expiry_df["strike"].to_numpy()
        is_ce = expiry_df["option_type"].astype(str).to_numpy() == "CE"

        intrinsic = np.where(
            is_ce,
            np.maximum(spot - strike, 0.0),
            np.maximum(strike - spot, 0.0),
        )

        expiry_df["intrinsic"] = intrinsic.astype(np.float32)
        expiry_df["time_value"] = (
            expiry_df["close"] - expiry_df["intrinsic"]
        ).clip(lower=0).astype(np.float32)

        expiry_df["moneyness"] = (
            (expiry_df["underlying_close"] - expiry_df["strike"])
            / expiry_df["underlying_close"]
        ).astype(np.float32)

        expiry_df = add_iv_and_greeks(expiry_df)

        expiry_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        expiry_df["oi_change_pct"] = expiry_df["oi_change_pct"].clip(-5.0, 5.0)

        for c in expiry_df.select_dtypes(include=["float64"]).columns:
            expiry_df[c] = expiry_df[c].astype(np.float32)

        expiry_df.to_parquet(output_file, index=False, compression="zstd")

        log.info(
            "Saved option partition: %s | rows=%d",
            output_file.name,
            len(expiry_df),
        )

        output_files.append(output_file)
        del expiry_df
        gc.collect()

    if not output_files:
        raise RuntimeError("No option-market partitions produced.")

    return output_files

# =============================================================================
# WIDE OBSERVATIONS
# =============================================================================

def nearest_strike_indices(strikes, spot):
    pos = np.searchsorted(strikes, spot)
    pos = np.clip(pos, 0, len(strikes) - 1)
    left = np.clip(pos - 1, 0, len(strikes) - 1)
    choose_left = np.abs(strikes[left] - spot) <= np.abs(strikes[pos] - spot)
    return np.where(choose_left, left, pos)


def offset_name(offset):
    if offset == 0:
        return "atm"
    return f"m{abs(offset)}" if offset < 0 else f"p{offset}"


def build_wide_observations_one_expiry(u, option_df):
    if u.empty or option_df.empty:
        return pd.DataFrame()

    expiry = pd.Timestamp(u["expiry"].iloc[0]).normalize()
    strikes = np.sort(option_df["strike"].dropna().unique().astype(np.float64))
    if len(strikes) == 0:
        return pd.DataFrame()

    minute_map = u[["timestamp", "close"]].copy()
    idx = nearest_strike_indices(
        strikes,
        minute_map["close"].to_numpy(np.float64),
    )
    minute_map["atm_strike"] = strikes[idx]
    minute_map["expiry"] = expiry

    em = option_df.merge(
        minute_map[["timestamp", "atm_strike"]],
        on="timestamp",
        how="inner",
    )

    strike_index = {float(value): i for i, value in enumerate(strikes)}
    em["strike_idx"] = em["strike"].map(strike_index).astype("Int64")
    em["atm_idx"] = em["atm_strike"].map(strike_index).astype("Int64")
    em["offset"] = (em["strike_idx"] - em["atm_idx"]).astype("Int64")
    em = em[em["offset"].between(-ATM_WINGS, ATM_WINGS)].copy()
    em["available"] = np.float32(1.0)

    metrics = [
        "open", "high", "low", "close",
        "option_ret_1m", "option_ret_3m",
        "log_volume", "log_oi",
        "volume_change", "oi_change", "oi_change_pct",
        "iv", "delta", "gamma", "theta", "vega", "rho",
        "moneyness", "intrinsic", "time_value", "available",
    ]

    pivots = []

    for metric in metrics:
        if metric not in em.columns:
            continue

        p = em.pivot_table(
            index="timestamp",
            columns=["option_type", "offset"],
            values=metric,
            aggfunc="last",
            observed=True,
        )

        if p.empty:
            continue

        p.columns = [
            f"{str(option_type).lower()}_{offset_name(int(offset))}_{metric}"
            for option_type, offset in p.columns
        ]
        pivots.append(p)

    if not pivots:
        return pd.DataFrame()

    options_wide = pd.concat(pivots, axis=1).reset_index()
    options_wide = minute_map.merge(options_wide, on="timestamp", how="left")
    options_wide.drop(columns=["close"], errors="ignore", inplace=True)

    out = u.merge(
        options_wide,
        on=["timestamp", "expiry"],
        how="left",
        suffixes=("", "_opt"),
    )

    if "atm_strike_opt" in out.columns:
        if "atm_strike" not in out.columns:
            out.rename(columns={"atm_strike_opt": "atm_strike"}, inplace=True)
        else:
            out["atm_strike"] = out["atm_strike"].fillna(out["atm_strike_opt"])
            out.drop(columns=["atm_strike_opt"], inplace=True)

    # Exact ATM symbols for execution later.
    identity = option_df[
        ["timestamp", "expiry", "strike", "option_type", "groww_symbol"]
    ].copy()
    identity["groww_symbol"] = identity["groww_symbol"].astype(str)
    identity["option_type"] = identity["option_type"].astype(str)

    identity = identity.merge(
        out[["timestamp", "expiry", "atm_strike"]],
        on=["timestamp", "expiry"],
        how="inner",
    )
    identity = identity[
        np.isclose(identity["strike"], identity["atm_strike"])
    ].copy()

    ids = identity.pivot_table(
        index=["timestamp", "expiry"],
        columns="option_type",
        values="groww_symbol",
        aggfunc="last",
        observed=True,
    ).reset_index()

    ids.rename(
        columns={"CE": "atm_ce_symbol", "PE": "atm_pe_symbol"},
        inplace=True,
    )

    out = out.merge(ids, on=["timestamp", "expiry"], how="left")

    expiry_clock = out["expiry"] + pd.Timedelta(hours=15, minutes=30)
    seconds = (expiry_clock - out["timestamp"]).dt.total_seconds().clip(lower=60)
    out["dte_days"] = (seconds / 86400.0).astype(np.float32)
    out["tte_years"] = (
        seconds / (365.0 * 24.0 * 3600.0)
    ).astype(np.float32)

    # Missing option values are safe because availability masks are retained.
    avail_cols = [
        c for c in out.columns
        if (c.startswith("ce_") or c.startswith("pe_"))
        and c.endswith("_available")
    ]
    if avail_cols:
        out[avail_cols] = out[avail_cols].fillna(0.0).astype(np.float32)

    option_numeric_cols = [
        c for c in out.columns
        if (c.startswith("ce_") or c.startswith("pe_"))
        and pd.api.types.is_numeric_dtype(out[c])
    ]

    if option_numeric_cols:
        out[option_numeric_cols] = (
            out[option_numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    for c in option_numeric_cols:
        if c.endswith("_oi_change_pct"):
            out[c] = out[c].clip(-5.0, 5.0)

    return out


def build_observation_partitions(underlying, option_files):
    output_files = []

    for option_file in option_files:
        option_df = pd.read_parquet(option_file)
        if option_df.empty:
            continue

        expiry = pd.Timestamp(option_df["expiry"].iloc[0]).normalize()
        expiry_string = expiry.strftime("%Y-%m-%d")
        output_file = OBS_PARTS_DIR / f"observations_{expiry_string}.parquet"

        if output_file.exists():
            log.info("Reusing observation partition: %s", output_file.name)
            output_files.append(output_file)
            del option_df
            gc.collect()
            continue

        u = underlying[underlying["expiry"] == expiry].copy()
        if u.empty:
            del option_df
            gc.collect()
            continue

        log.info("Building observation partition for %s", expiry_string)
        wide = build_wide_observations_one_expiry(u, option_df)

        if wide.empty:
            log.warning("No wide observations for %s", expiry_string)
            del option_df
            gc.collect()
            continue

        for c in wide.select_dtypes(include=["float64"]).columns:
            wide[c] = wide[c].astype(np.float32)

        wide.to_parquet(output_file, index=False, compression="zstd")

        log.info(
            "Saved observation partition: %s | rows=%d",
            output_file.name,
            len(wide),
        )

        output_files.append(output_file)
        del option_df, u, wide
        gc.collect()

    if not output_files:
        raise RuntimeError("No observation partitions produced.")

    return output_files


def combine_observation_partitions(files):
    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates("timestamp", keep="last", inplace=True)
    df.index = range(len(df))
    return df

# =============================================================================
# MODEL FEATURE CLEANING
# =============================================================================

def add_split_column(df):
    df["split"] = np.where(
        df["timestamp"] <= TRAIN_END,
        "train",
        np.where(
            df["timestamp"] <= VALIDATION_END,
            "validation",
            "test",
        ),
    )
    return df


def choose_observation_columns(df):
    excluded_exact = {
        "timestamp",
        "date",
        "expiry",
        "split",
        "trading_date",
        "atm_strike",
        "atm_ce_symbol",
        "atm_pe_symbol",

        # BANKNIFTY is an index; this was all-NaN in your dataset.
        "volume",
    }

    columns = []

    for c in df.columns:
        if c in excluded_exact:
            continue

        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        lower = c.lower()

        if any(
            token in lower
            for token in [
                "future", "label", "outcome", "target_hit",
                "forward_", "next_return", "exec_"
            ]
        ):
            continue

        columns.append(c)

    return columns


def clean_and_validate_observations(df):
    if df.empty:
        raise RuntimeError("Final observation dataset is empty.")

    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    protected = {"timestamp", "expiry", "atm_ce_symbol", "atm_pe_symbol"}

    all_nan = [
        c for c in df.columns
        if c not in protected and df[c].isna().all()
    ]

    if all_nan:
        log.warning("Dropping all-NaN columns: %s", all_nan)
        df.drop(columns=all_nan, inplace=True)

    observation_columns = choose_observation_columns(df)

    obs = df[observation_columns]
    nan_counts = obs.isna().sum().sort_values(ascending=False)
    problem = nan_counts[nan_counts > 0]

    print("\n================ NON-FINITE REPORT ================\n")

    if len(problem):
        print(problem.head(100))
    else:
        print("No NaN/Inf values in selected observation features.")

    print("\nTotal rows:", len(df))
    print(
        "Rows containing at least one NaN/Inf:",
        int(obs.isna().any(axis=1).sum()),
    )

    # Remove bad FEATURES, not entire rows.
    missing_ratio = df[observation_columns].isna().mean()
    unusable = (
        missing_ratio[
            missing_ratio > MAX_MISSING_FEATURE_RATIO
        ].index.tolist()
    )

    if unusable:
        log.warning(
            "Removing model features with >%.0f%% missing data: %s",
            MAX_MISSING_FEATURE_RATIO * 100.0,
            unusable,
        )
        df.drop(columns=unusable, inplace=True)
        observation_columns = [
            c for c in observation_columns if c not in unusable
        ]

    # Remaining missing values become neutral 0.
    # Option availability masks tell the model when option data was absent.
    df[observation_columns] = df[observation_columns].fillna(0.0)

    for c in observation_columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype(np.float32)

    values = df[observation_columns].to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite observations remain after cleaning.")

    if df["timestamp"].duplicated().any():
        raise RuntimeError("Duplicate timestamps in final dataset.")

    if (df["timestamp"].diff().dropna() < pd.Timedelta(0)).any():
        raise RuntimeError("Final dataset is not chronological.")

    log.info(
        "Validated observations: rows=%d | features=%d",
        len(df),
        len(observation_columns),
    )

    return df, observation_columns

# =============================================================================
# MANIFEST / REPORT
# =============================================================================

def write_manifest(df, observation_columns):
    manifest = {
        "dataset_version": "banknifty_sb3_v2",

        "period": {
            "requested_start": str(START_DATE.date()),
            "requested_end": str(END_DATE.date()),
            "actual_start": str(df["timestamp"].min()),
            "actual_end": str(df["timestamp"].max()),
        },

        "files": {
            "observations": str(OBS_FILE),
            "option_market_dir": str(OPTION_DIR),
            "observation_parts_dir": str(OBS_PARTS_DIR),
        },

        "actions": {
            "0": "WAIT",
            "1": "BUY_ATM_CE",
            "2": "BUY_ATM_PE",
        },

        "execution_rule": (
            "Observe completed candle t. Execute a new entry no earlier than "
            "historical option candle t+1; prefer t+1 OPEN plus modeled slippage."
        ),

        "position_rule": (
            "Hold the exact groww_symbol chosen at entry until exit. Do not switch "
            "the open position merely because the ATM strike changes."
        ),

        "historical_greeks": {
            "source": "Derived from historical option close using Black-Scholes",
            "risk_free_rate": RISK_FREE_RATE,
            "dividend_yield": DIVIDEND_YIELD,
            "cuda_used": bool(CUDA_AVAILABLE),
        },

        "option_surface": {
            "atm_wings": ATM_WINGS,
            "fetch_extra_wings": FETCH_EXTRA_WINGS,
        },

        "splits": {
            "train": f"{START_DATE.date()} through {TRAIN_END.date()}",
            "validation": (
                f"{(TRAIN_END + pd.Timedelta(seconds=1)).date()} "
                f"through {VALIDATION_END.date()}"
            ),
            "test": (
                f"{(VALIDATION_END + pd.Timedelta(seconds=1)).date()} "
                f"through {END_DATE.date()}"
            ),
        },

        "observation_columns": observation_columns,

        "non_observation_metadata": [
            "timestamp",
            "expiry",
            "split",
            "atm_strike",
            "atm_ce_symbol",
            "atm_pe_symbol",
            "BANKNIFTY index volume",
        ],

        "row_counts": {
            str(k): int(v)
            for k, v in df["split"].value_counts().to_dict().items()
        },
    }

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def print_quality_report(df, observation_columns):
    log.info("=" * 72)
    log.info("FINAL DATASET QUALITY REPORT")
    log.info("=" * 72)
    log.info("Rows: %d", len(df))
    log.info("Observation features: %d", len(observation_columns))
    log.info("Start: %s", df["timestamp"].min())
    log.info("End: %s", df["timestamp"].max())
    log.info("Trading days: %d", df["timestamp"].dt.date.nunique())
    log.info("Duplicate timestamps: %d", int(df["timestamp"].duplicated().sum()))
    log.info("Split counts:\n%s", df["split"].value_counts().to_string())

    if "atm_ce_symbol" in df.columns:
        log.info("Missing ATM CE symbols: %d", int(df["atm_ce_symbol"].isna().sum()))

    if "atm_pe_symbol" in df.columns:
        log.info("Missing ATM PE symbols: %d", int(df["atm_pe_symbol"].isna().sum()))

# =============================================================================
# MAIN
# =============================================================================

def main():
    create_directories()

    log.info("=" * 72)
    log.info("BANKNIFTY GROWW -> STABLE-BASELINES3 DATASET V2")
    log.info("=" * 72)
    log.info("Requested period: %s -> %s", START_DATE.date(), END_DATE.date())
    log.info("Project directory: %s", BASE_DIR)

    groww = create_groww_client()

    # 1) BANKNIFTY underlying.
    underlying = load_underlying()

    # 2) Historical BANKNIFTY expiries.
    expiries = fetch_expiries(groww)
    underlying = assign_front_expiry(underlying, expiries)

    # 3) Long option market, partitioned by expiry.
    log.info("Building/reusing option-market partitions...")
    option_files = build_option_market_partitions(groww, underlying)

    # 4) Wide observations, partitioned by expiry.
    log.info("Building/reusing observation partitions...")
    observation_files = build_observation_partitions(underlying, option_files)

    # 5) Combine only one-row-per-minute observation data.
    wide = combine_observation_partitions(observation_files)
    wide = add_split_column(wide)

    # 6) Clean without deleting all timestamps because of NaNs.
    wide, observation_columns = clean_and_validate_observations(wide)

    # 7) Final RL observation file.
    wide.to_parquet(OBS_FILE, index=False, compression="zstd")

    write_manifest(wide, observation_columns)
    print_quality_report(wide, observation_columns)

    log.info("=" * 72)
    log.info("DATASET COMPLETE")
    log.info("Observations : %s", OBS_FILE)
    log.info("Option market: %s", OPTION_DIR)
    log.info("Manifest     : %s", MANIFEST_FILE)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
