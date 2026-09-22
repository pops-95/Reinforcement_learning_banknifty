#!/usr/bin/env python3

"""
BANKNIFTY 1-minute historical dataset builder
==============================================

Period:
    2024-01-01 to 2026-08-31

Source:
    Groww Backtesting API

Outputs:
    data/raw/banknifty_1min_raw.parquet
    data/processed/banknifty_1min_features.parquet
    data/processed/banknifty_1min_features.csv
    data/processed/timeframes/banknifty_1m.parquet
    data/processed/timeframes/banknifty_5m.parquet
    data/processed/timeframes/banknifty_15m.parquet
    data/processed/timeframes/banknifty_30m.parquet

Multi-timeframe features:
    - completed candle OHLC for 1m, 5m, 15m and 30m
    - EMA 20 / EMA 50 / EMA 100 for every timeframe
    - EMA-distance features for every timeframe

CAUSALITY:
    Higher-timeframe values are aligned only after the higher-timeframe candle
    has completed. For example, the 09:15-09:19 5-minute candle becomes
    available at 09:20. No future high/low is leaked into earlier 1-minute rows.

GPU:
    RAPIDS cuDF/CuPy used where available.
    Falls back to pandas automatically.

IMPORTANT:
    Set your Groww token:

        export GROWW_ACCESS_TOKEN="YOUR_TOKEN"

Then:

        python banknifty_dataset.py
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from growwapi import GrowwAPI


# ============================================================
# CONFIGURATION
# ============================================================

START_DATE = "2024-01-01"
END_DATE = "2026-08-31"

# Keep below Groww's 30-day max for 1-min candles.
CHUNK_DAYS = 25

SYMBOL = "NSE-BANKNIFTY"

MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNK_DIR = DATA_DIR / "chunks"

RAW_FILE = RAW_DIR / "banknifty_1min_raw.parquet"
FEATURE_FILE = PROCESSED_DIR / "banknifty_1min_features.parquet"
CSV_FILE = PROCESSED_DIR / "banknifty_1min_features.csv"

# Multi-timeframe files. The final 1-minute RL feature table also contains the
# last COMPLETED candle/EMA context from each of these timeframes.
MTF_DIR = PROCESSED_DIR / "timeframes"
MTF_TIMEFRAMES = (1, 5, 15, 30)
MTF_EMA_PERIODS = (20, 50, 100)
MTF_FILES = {
    minutes: MTF_DIR / f"banknifty_{minutes}m.parquet"
    for minutes in MTF_TIMEFRAMES
}

MAX_RETRIES = 5
REQUEST_SLEEP = 0.6


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("BANKNIFTY")


# ============================================================
# CUDA / RAPIDS CHECK
# ============================================================

CUDA_AVAILABLE = False

try:
    import cupy as cp
    import cudf

    device_count = cp.cuda.runtime.getDeviceCount()

    if device_count > 0:
        CUDA_AVAILABLE = True

        props = cp.cuda.runtime.getDeviceProperties(0)

        gpu_name = props["name"]

        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()

        logger.info("CUDA available")
        logger.info("GPU: %s", gpu_name)

except Exception as exc:
    logger.warning("CUDA/RAPIDS unavailable: %s", exc)
    logger.warning("Feature calculations will use CPU.")


# ============================================================
# DIRECTORIES
# ============================================================

def create_directories():

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    MTF_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GROWW
# ============================================================

def create_groww_client():

    token = os.getenv("GROWW_ACCESS_TOKEN")

    if not token:
        raise RuntimeError(
            "\nGROWW_ACCESS_TOKEN not found.\n\n"
            "Run:\n"
            'export GROWW_ACCESS_TOKEN="YOUR_GROWW_TOKEN"\n'
        )

    return GrowwAPI(token)


# ============================================================
# DATE CHUNKS
# ============================================================

def generate_chunks(start_date, end_date, chunk_days=25):

    current = pd.Timestamp(start_date)
    final = pd.Timestamp(end_date)

    while current <= final:

        chunk_end = min(
            current + pd.Timedelta(days=chunk_days - 1),
            final
        )

        yield current, chunk_end

        current = chunk_end + pd.Timedelta(days=1)


# ============================================================
# FETCH ONE CHUNK
# ============================================================

def fetch_chunk(groww, start_date, end_date):

    start_time = (
        start_date.strftime("%Y-%m-%d")
        + " 09:15:00"
    )

    end_time = (
        end_date.strftime("%Y-%m-%d")
        + " 15:30:00"
    )

    logger.info(
        "Downloading BANKNIFTY %s -> %s",
        start_time,
        end_time
    )

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = groww.get_historical_candles(
                exchange=groww.EXCHANGE_NSE,
                segment=groww.SEGMENT_CASH,

                groww_symbol=SYMBOL,

                start_time=start_time,
                end_time=end_time,

                candle_interval=groww.CANDLE_INTERVAL_MIN_1
            )

            candles = response.get("candles", [])

            if not candles:
                logger.warning(
                    "No candles returned for %s -> %s",
                    start_time,
                    end_time
                )

                return pd.DataFrame()

            rows = []

            for candle in candles:

                # API normally:
                #
                # timestamp
                # open
                # high
                # low
                # close
                # volume
                # optionally OI

                row = {
                    "timestamp": candle[0],
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": (
                        candle[5]
                        if len(candle) > 5
                        else np.nan
                    ),
                }

                rows.append(row)

            df = pd.DataFrame(rows)

            return df

        except Exception as exc:

            logger.error(
                "Attempt %d/%d failed: %s",
                attempt,
                MAX_RETRIES,
                exc
            )

            if attempt == MAX_RETRIES:
                raise

            time.sleep(2 ** attempt)

    return pd.DataFrame()


# ============================================================
# NORMALIZE RAW DATA
# ============================================================

def normalize_raw(df):

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close"
        ]
    )

    df = (
        df
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    # Restrict to NSE session
    times = df["timestamp"].dt.strftime("%H:%M")

    df = df[
        (times >= MARKET_OPEN)
        &
        (times <= MARKET_CLOSE)
    ].copy()

    return df


# ============================================================
# DOWNLOAD
# ============================================================

def download_banknifty():

    groww = create_groww_client()

    all_chunks = []

    chunks = list(
        generate_chunks(
            START_DATE,
            END_DATE,
            CHUNK_DAYS
        )
    )

    logger.info(
        "Total API chunks: %d",
        len(chunks)
    )

    for i, (start, end) in enumerate(chunks, start=1):

        chunk_file = (
            CHUNK_DIR /
            f"banknifty_{start.date()}_{end.date()}.parquet"
        )

        # ------------------------------
        # Resume capability
        # ------------------------------

        if chunk_file.exists():

            logger.info(
                "[%d/%d] Loading cached %s",
                i,
                len(chunks),
                chunk_file
            )

            df = pd.read_parquet(chunk_file)

        else:

            logger.info(
                "[%d/%d] Requesting Groww",
                i,
                len(chunks)
            )

            df = fetch_chunk(
                groww,
                start,
                end
            )

            df = normalize_raw(df)

            if not df.empty:
                df.to_parquet(
                    chunk_file,
                    index=False
                )

            time.sleep(REQUEST_SLEEP)

        if not df.empty:
            all_chunks.append(df)

    if not all_chunks:
        raise RuntimeError(
            "No BANKNIFTY data downloaded."
        )

    data = pd.concat(
        all_chunks,
        ignore_index=True
    )

    data = normalize_raw(data)

    data.to_parquet(
        RAW_FILE,
        index=False
    )

    logger.info(
        "Raw candles saved: %s",
        RAW_FILE
    )

    logger.info(
        "Raw candle count: %d",
        len(data)
    )

    return data


# ============================================================
# CPU INDICATORS
# ============================================================

def ema(series, span):

    return series.ewm(
        span=span,
        adjust=False
    ).mean()


def rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / (
        avg_loss + 1e-10
    )

    return 100 - (
        100 / (1 + rs)
    )


def atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()



# ============================================================
# MULTI-TIMEFRAME CANDLES / EMAS
# ============================================================

def _ema_mature(series, span):
    """EMA that stays unavailable until `span` completed bars exist."""
    return series.ewm(
        span=span,
        adjust=False,
        min_periods=span,
    ).mean()


def build_completed_timeframe_bars(raw_df, minutes):
    """Build causal completed candles for one timeframe.

    For minutes > 1, each candle is labelled by its END time and contains only
    1-minute candles strictly before that end. A 5-minute bar labelled 09:20 is
    therefore built from 09:15..09:19 and is safe to use at/after 09:20.

    Incomplete higher-timeframe bars caused by missing 1-minute candles are
    excluded. A partial final 30-minute session bar is also excluded because it
    is not a full 30-minute candle.
    """
    minutes = int(minutes)
    if minutes not in MTF_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {minutes}m")

    base = raw_df[["timestamp", "open", "high", "low", "close"]].copy()
    base = base.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    if minutes == 1:
        bars = base.copy()
        bars["source_1m_count"] = np.int16(1)
    else:
        pieces = []
        for trading_date, day in base.groupby(base["timestamp"].dt.date, sort=True):
            day = day.set_index("timestamp").sort_index()
            anchor = pd.Timestamp(trading_date) + pd.Timedelta(hours=9, minutes=15)
            session_end = pd.Timestamp(trading_date) + pd.Timedelta(hours=15, minutes=30)

            ohlc = day[["open", "high", "low", "close"]].resample(
                f"{minutes}min",
                origin=anchor,
                label="right",
                closed="left",
            ).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            })
            count = day["close"].resample(
                f"{minutes}min",
                origin=anchor,
                label="right",
                closed="left",
            ).count()
            ohlc["source_1m_count"] = count.astype("int16")

            # Full-duration bars only. This avoids treating data gaps (or the
            # partial 15:15-15:30 block on a 30m chart) as complete candles.
            ohlc = ohlc[
                (ohlc.index <= session_end)
                & (ohlc["source_1m_count"] == minutes)
            ].dropna(subset=["open", "high", "low", "close"])

            if not ohlc.empty:
                pieces.append(ohlc.reset_index())

        if pieces:
            bars = pd.concat(pieces, ignore_index=True)
        else:
            bars = pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "source_1m_count"]
            )

    bars = bars.sort_values("timestamp").reset_index(drop=True)
    bars["timeframe_minutes"] = np.int16(minutes)

    # Requested EMA context on each timeframe. These are based only on completed
    # closes and therefore remain causal when merged back to 1-minute rows.
    for period in MTF_EMA_PERIODS:
        ema_col = f"ema_{period}"
        bars[ema_col] = _ema_mature(bars["close"], period)
        bars[f"ema_{period}_dist"] = bars["close"] / bars[ema_col] - 1.0
        bars[f"ema_{period}_slope_3"] = bars[ema_col] / bars[ema_col].shift(3) - 1.0

    bars["candle_range"] = bars["high"] - bars["low"]
    bars["body"] = bars["close"] - bars["open"]

    for col in bars.select_dtypes(include=["float64"]).columns:
        bars[col] = bars[col].astype(np.float32)
    return bars


def save_timeframe_datasets(raw_df):
    """Write standalone 1m/5m/15m/30m candle files for inspection/research."""
    result = {}
    for minutes in MTF_TIMEFRAMES:
        bars = build_completed_timeframe_bars(raw_df, minutes)
        bars.to_parquet(MTF_FILES[minutes], index=False, compression="zstd")
        result[minutes] = bars
        logger.info(
            "Saved %dm timeframe: %s | bars=%d",
            minutes, MTF_FILES[minutes], len(bars),
        )
    return result


def add_multi_timeframe_context(features, timeframe_tables):
    """Attach last completed 1m/5m/15m/30m candle context to each 1m row.

    The merge is backward-looking only. A higher-timeframe candle is never used
    before its completion timestamp, preventing look-ahead leakage.
    """
    out = features.sort_values("timestamp").copy()

    # Explicit 1-minute aliases make feature names consistent across timeframes
    # while preserving the original columns used by the trading environment.
    for field in ("open", "high", "low", "close"):
        out[f"{field}_1m"] = out[field]
    for period in MTF_EMA_PERIODS:
        out[f"ema_{period}_1m"] = out[f"ema_{period}"]
        out[f"ema_{period}_1m_dist"] = out[f"ema_{period}_dist"]
        out[f"ema_{period}_1m_slope_3"] = out[f"ema_{period}_slope_3"]

    for minutes in (5, 15, 30):
        bars = timeframe_tables[minutes].copy()
        keep = ["timestamp", "open", "high", "low", "close"]
        for period in MTF_EMA_PERIODS:
            keep.extend([
                f"ema_{period}",
                f"ema_{period}_dist",
                f"ema_{period}_slope_3",
            ])
        bars = bars[keep]

        rename = {
            col: f"{col}_{minutes}m"
            for col in bars.columns
            if col != "timestamp"
        }
        bars.rename(columns=rename, inplace=True)
        bars[f"mtf_{minutes}m_available"] = np.float32(1.0)

        # Global backward merge intentionally allows the previous session's final
        # completed HTF bar to remain the latest known context near today's open.
        out = pd.merge_asof(
            out.sort_values("timestamp"),
            bars.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )
        out[f"mtf_{minutes}m_available"] = (
            out[f"mtf_{minutes}m_available"].fillna(0.0).astype(np.float32)
        )

        # Relative position of the current 1m close versus the most recently
        # completed HTF EMA is often more useful to PPO than raw price alone.
        for period in MTF_EMA_PERIODS:
            ema_col = f"ema_{period}_{minutes}m"
            out[f"price_to_ema_{period}_{minutes}m"] = (
                out["close"] / out[ema_col] - 1.0
            )

        # Location inside the most recent completed HTF candle.
        h = out[f"high_{minutes}m"]
        l = out[f"low_{minutes}m"]
        out[f"range_{minutes}m"] = h - l
        out[f"close_location_{minutes}m"] = (
            (out["close"] - l) / (h - l + 1e-10)
        )

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features_cpu(df):

    logger.info(
        "Building RL features on CPU..."
    )

    df = df.copy()

    # ========================================
    # Time information
    # ========================================

    df["date"] = df["timestamp"].dt.date

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    minutes_from_midnight = (
        df["hour"] * 60 +
        df["minute"]
    )

    market_start = 9 * 60 + 15

    df["minutes_from_open"] = (
        minutes_from_midnight -
        market_start
    )

    # Cyclical time encoding
    session_length = 375

    df["time_sin"] = np.sin(
        2 * np.pi *
        df["minutes_from_open"] /
        session_length
    )

    df["time_cos"] = np.cos(
        2 * np.pi *
        df["minutes_from_open"] /
        session_length
    )

    # ========================================
    # Returns
    # ========================================

    df["return_1m"] = (
        df["close"].pct_change()
    )

    df["return_2m"] = (
        df["close"].pct_change(2)
    )

    df["return_3m"] = (
        df["close"].pct_change(3)
    )

    df["return_5m"] = (
        df["close"].pct_change(5)
    )

    df["return_10m"] = (
        df["close"].pct_change(10)
    )

    df["return_15m"] = (
        df["close"].pct_change(15)
    )

    df["return_30m"] = (
        df["close"].pct_change(30)
    )

    # Log returns
    df["log_return"] = np.log(
        df["close"] /
        df["close"].shift(1)
    )

    # ========================================
    # Candle morphology
    # ========================================

    previous_close = df["close"].shift(1)

    df["gap"] = (
        df["open"] -
        previous_close
    ) / previous_close

    df["candle_range"] = (
        df["high"] -
        df["low"]
    )

    df["body"] = (
        df["close"] -
        df["open"]
    )

    df["body_pct"] = (
        df["body"] /
        df["open"]
    )

    max_oc = df[
        ["open", "close"]
    ].max(axis=1)

    min_oc = df[
        ["open", "close"]
    ].min(axis=1)

    df["upper_wick"] = (
        df["high"] -
        max_oc
    )

    df["lower_wick"] = (
        min_oc -
        df["low"]
    )

    df["body_to_range"] = (
        df["body"].abs() /
        (df["candle_range"] + 1e-10)
    )

    df["upper_wick_ratio"] = (
        df["upper_wick"] /
        (df["candle_range"] + 1e-10)
    )

    df["lower_wick_ratio"] = (
        df["lower_wick"] /
        (df["candle_range"] + 1e-10)
    )

    # Position of close inside candle
    df["close_location"] = (
        (df["close"] - df["low"]) /
        (df["candle_range"] + 1e-10)
    )

    # ========================================
    # EMA
    # ========================================

    for period in [
        5,
        9,
        20,
        50,
        100,
        200,
    ]:

        col = f"ema_{period}"

        df[col] = ema(
            df["close"],
            period
        )

        # Relative distance is better for ML
        df[f"ema_{period}_dist"] = (
            df["close"] /
            df[col]
            - 1
        )

        df[f"ema_{period}_slope_3"] = (
            df[col] /
            df[col].shift(3)
            - 1
        )

    # ========================================
    # EMA relationships
    # ========================================

    df["ema_9_20_spread"] = (
        df["ema_9"] /
        df["ema_20"]
        - 1
    )

    df["ema_20_50_spread"] = (
        df["ema_20"] /
        df["ema_50"]
        - 1
    )

    # ========================================
    # ATR
    # ========================================

    df["atr_14"] = atr(
        df,
        14
    )

    df["atr_pct"] = (
        df["atr_14"] /
        df["close"]
    )

    df["atr_ratio_20"] = (
        df["atr_14"] /
        df["atr_14"]
        .rolling(20)
        .mean()
    )

    # ========================================
    # RSI
    # ========================================

    df["rsi_14"] = rsi(
        df["close"],
        14
    )

    # Better scaled for neural network
    df["rsi_scaled"] = (
        df["rsi_14"] - 50
    ) / 50

    # ========================================
    # Rolling volatility
    # ========================================

    for period in [
        5,
        10,
        20,
        30,
        60,
    ]:

        df[
            f"volatility_{period}"
        ] = (
            df["return_1m"]
            .rolling(period)
            .std()
        )

    # ========================================
    # Rolling highs/lows
    # ========================================

    for period in [
        5,
        10,
        20,
        30,
        60,
    ]:

        rolling_high = (
            df["high"]
            .rolling(period)
            .max()
        )

        rolling_low = (
            df["low"]
            .rolling(period)
            .min()
        )

        df[
            f"dist_high_{period}"
        ] = (
            df["close"] /
            rolling_high
            - 1
        )

        df[
            f"dist_low_{period}"
        ] = (
            df["close"] /
            rolling_low
            - 1
        )

    # ========================================
    # Session / intraday calculations
    # ========================================

    grouped = df.groupby(
        "date",
        sort=False
    )

    df["day_open"] = (
        grouped["open"]
        .transform("first")
    )

    df["day_high"] = (
        grouped["high"]
        .cummax()
    )

    df["day_low"] = (
        grouped["low"]
        .cummin()
    )

    df["return_from_open"] = (
        df["close"] /
        df["day_open"]
        - 1
    )

    df["distance_day_high"] = (
        df["close"] /
        df["day_high"]
        - 1
    )

    df["distance_day_low"] = (
        df["close"] /
        df["day_low"]
        - 1
    )

    # ========================================
    # Running VWAP
    #
    # NOTE:
    # Index volume may be unavailable / zero.
    # Do not rely on this for BANKNIFTY index.
    # ========================================

    if (
        "volume" in df.columns
        and
        df["volume"].fillna(0).sum() > 0
    ):

        typical_price = (
            df["high"] +
            df["low"] +
            df["close"]
        ) / 3

        pv = (
            typical_price *
            df["volume"].fillna(0)
        )

        df["_pv"] = pv

        df["_cum_pv"] = (
            df.groupby("date")["_pv"]
            .cumsum()
        )

        df["_cum_volume"] = (
            df.groupby("date")["volume"]
            .cumsum()
        )

        df["vwap"] = (
            df["_cum_pv"] /
            (
                df["_cum_volume"] +
                1e-10
            )
        )

        df["vwap_dist"] = (
            df["close"] /
            df["vwap"]
            - 1
        )

        df.drop(
            columns=[
                "_pv",
                "_cum_pv",
                "_cum_volume"
            ],
            inplace=True
        )

    # ========================================
    # Acceleration / momentum
    # ========================================

    df["momentum_3"] = (
        df["close"] -
        df["close"].shift(3)
    ) / df["close"].shift(3)

    df["momentum_5"] = (
        df["close"] -
        df["close"].shift(5)
    ) / df["close"].shift(5)

    df["momentum_15"] = (
        df["close"] -
        df["close"].shift(15)
    ) / df["close"].shift(15)

    df["acceleration"] = (
        df["return_3m"] -
        df["return_3m"].shift(3)
    )

    # ========================================
    # Previous candle comparisons
    # ========================================

    df["higher_high"] = (
        df["high"] >
        df["high"].shift(1)
    ).astype(np.int8)

    df["higher_low"] = (
        df["low"] >
        df["low"].shift(1)
    ).astype(np.int8)

    df["lower_high"] = (
        df["high"] <
        df["high"].shift(1)
    ).astype(np.int8)

    df["lower_low"] = (
        df["low"] <
        df["low"].shift(1)
    ).astype(np.int8)

    # ========================================
    # Clean infinities
    # ========================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# GPU CALCULATIONS
# ============================================================

def gpu_postprocess(df):

    """
    Performs the large matrix-friendly calculations with CUDA.

    This function intentionally doesn't put API/Pandas group-by
    session logic on GPU unnecessarily.

    During SB3 sequence creation/training, tensors will remain
    on GPU.
    """

    if not CUDA_AVAILABLE:
        return df

    logger.info(
        "Running CUDA post-processing..."
    )

    feature_columns = [
        c
        for c in df.columns
        if c not in [
            "timestamp",
            "date"
        ]
    ]

    numeric = df[
        feature_columns
    ].select_dtypes(
        include=[np.number]
    )

    # Transfer to GPU
    gpu_df = cudf.from_pandas(
        numeric.astype(np.float32)
    )

    # Example clipping of extreme numerical noise
    gpu_df = gpu_df.clip(
        lower=-1e10,
        upper=1e10
    )

    gpu_result = gpu_df.to_pandas()

    for col in gpu_result.columns:
        df[col] = gpu_result[col]

    return df


# ============================================================
# DATA QUALITY CHECK
# ============================================================

def quality_report(df):

    logger.info("=" * 60)
    logger.info("DATA QUALITY REPORT")
    logger.info("=" * 60)

    logger.info(
        "Rows: %d",
        len(df)
    )

    logger.info(
        "Start: %s",
        df["timestamp"].min()
    )

    logger.info(
        "End: %s",
        df["timestamp"].max()
    )

    logger.info(
        "Trading days: %d",
        df["timestamp"]
        .dt.date
        .nunique()
    )

    duplicates = (
        df["timestamp"]
        .duplicated()
        .sum()
    )

    logger.info(
        "Duplicate timestamps: %d",
        duplicates
    )

    daily_counts = (
        df.groupby(
            df["timestamp"].dt.date
        )
        .size()
    )

    logger.info(
        "Median candles/day: %.1f",
        daily_counts.median()
    )

    logger.info(
        "Minimum candles/day: %d",
        daily_counts.min()
    )

    logger.info(
        "Maximum candles/day: %d",
        daily_counts.max()
    )

    suspicious_days = daily_counts[
        daily_counts < 300
    ]

    logger.info(
        "Days with <300 candles: %d",
        len(suspicious_days)
    )

    if len(suspicious_days):

        logger.warning(
            "Potential incomplete sessions:"
        )

        logger.warning(
            "\n%s",
            suspicious_days.to_string()
        )


# ============================================================
# RL DATASET PREPARATION
# ============================================================

def create_rl_dataset(raw_df):

    # Build/save standalone completed-candle datasets first so the same bars are
    # used both for inspection and for the merged RL feature table.
    timeframe_tables = save_timeframe_datasets(raw_df)

    features = build_features_cpu(
        raw_df
    )

    features = add_multi_timeframe_context(
        features,
        timeframe_tables,
    )

    features = gpu_postprocess(
        features
    )

    # ------------------------------------------------
    # DO NOT fill early rolling/EMA NaNs with future data.
    # Remove only the initial warm-up required for mature 30m EMA100 context.
    # Once this is available, all shorter-timeframe EMA20/50/100 values are also
    # mature. This costs only the first several trading sessions of the dataset.
    # ------------------------------------------------

    important_columns = [
        "ema_200",
        "atr_14",
        "rsi_14",
        "volatility_60",
        "ema_20_5m",
        "ema_50_5m",
        "ema_100_5m",
        "ema_20_15m",
        "ema_50_15m",
        "ema_100_15m",
        "ema_20_30m",
        "ema_50_30m",
        "ema_100_30m",
    ]

    features = features.dropna(
        subset=important_columns
    )

    features = features.reset_index(
        drop=True
    )

    # Float32 is ideal for PyTorch/SB3.
    for col in features.select_dtypes(
        include=["float64"]
    ).columns:

        features[col] = (
            features[col]
            .astype(np.float32)
        )

    features.to_parquet(
        FEATURE_FILE,
        index=False,
        compression="zstd",
    )

    # CSV is useful for inspection.
    features.to_csv(
        CSV_FILE,
        index=False
    )

    mtf_cols = [
        c for c in features.columns
        if c.endswith(("_1m", "_5m", "_15m", "_30m"))
        or c.startswith("price_to_ema_")
        or c.startswith("mtf_")
    ]

    logger.info(
        "RL dataset saved: %s",
        FEATURE_FILE
    )

    logger.info(
        "CSV saved: %s",
        CSV_FILE
    )

    logger.info(
        "RL rows: %d",
        len(features)
    )

    logger.info(
        "Columns: %d | multi-timeframe columns: %d",
        len(features.columns),
        len(mtf_cols),
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    logger.info("=" * 70)
    logger.info("BANKNIFTY RL DATASET BUILDER")
    logger.info("=" * 70)

    logger.info(
        "Requested period: %s -> %s",
        START_DATE,
        END_DATE
    )

    # --------------------------------------------
    # Resume from full raw file if already present
    # --------------------------------------------

    if RAW_FILE.exists():

        logger.info(
            "Existing raw dataset found."
        )

        raw = pd.read_parquet(
            RAW_FILE
        )

        raw["timestamp"] = pd.to_datetime(
            raw["timestamp"]
        )

    else:

        raw = download_banknifty()

    quality_report(raw)

    features = create_rl_dataset(
        raw
    )

    logger.info("=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)

    logger.info(
        "Raw:       %s",
        RAW_FILE
    )

    logger.info(
        "Processed: %s",
        FEATURE_FILE
    )

    logger.info(
        "Timeframes: %s",
        MTF_DIR
    )


if __name__ == "__main__":
    main()