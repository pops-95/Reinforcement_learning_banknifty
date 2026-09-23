"""Autonomous five-action MaskablePPO with advisory indicators and optional gates.

PPO chooses both entry directions and voluntary exits by default. Emergency and
session exits are configurable overlays. Fixed option targets and fixed option
stops can be configured independently. Missing indicators never invent prices.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from reference_strategy import ignition_features, structural_stop, select_contract
from rl_data import DATA_VERSION, load_dataset, feature_group


ENV_VERSION = "autonomous_ppo_v9_htf_veto"

# Keep the same web-visible configuration keys.  No HTML/UI changes are needed.
ENV_DEFAULTS = dict(
    # Causal minute history; one frame retains the legacy observation shape.
    observation_history_bars=1,
    random_start_enabled=False,
    random_start_probability=0.5,
    random_start_max_minutes=180,
    end_inactive_episode_enabled=False,
    # Reference candidate engine
    reference_strategy_enabled=False,
    reference_features_enabled=True,
    features_price_enabled=True,
    features_momentum_enabled=True,
    features_trend_enabled=True,
    features_volatility_enabled=True,
    features_volume_enabled=True,
    features_oi_enabled=True,
    features_greeks_enabled=True,
    features_time_enabled=True,
    disabled_features="",
    missing_filter_values_pass=True,
    missingness_features_enabled=True,
    reference_score_gate_enabled=True,
    reference_window=20,
    reference_ignition_min=0.60,
    reference_direction_gap=0.05,
    # Direction-specific overrides.  When enabled these replace the shared values.
    directional_thresholds_enabled=True,
    ce_reference_ignition_min=0.60,
    pe_reference_ignition_min=0.63,
    ce_reference_direction_gap=0.05,
    pe_reference_direction_gap=0.06,
    reference_signal_ttl_enabled=True,
    reference_signal_ttl_minutes=3,
    reference_strike_window=5,
    reference_strike_spacing=100.0,

    # High-precision entry gates
    tradeability_gate_enabled=False,
    tradeability_threshold=0.55,
    momentum_gate_enabled=False,
    min_directional_return_3m=0.00015,
    min_directional_return_5m=0.00025,
    ce_min_directional_return_3m=0.00015,
    ce_min_directional_return_5m=0.00025,
    pe_min_directional_return_3m=0.00020,
    pe_min_directional_return_5m=0.00035,
    momentum_acceleration_gate_enabled=False,
    min_directional_acceleration=0.0,

    # Soft quality terms improve ranking without turning every signal into a hard AND gate.
    soft_path_efficiency_enabled=True,
    soft_atr_quality_enabled=True,
    soft_price_oi_quality_enabled=True,
    soft_structural_distance_enabled=True,
    soft_time_quality_enabled=True,

    path_efficiency_gate_enabled=False,
    path_efficiency_window=20,
    min_path_efficiency=0.35,
    atr_expansion_gate_enabled=False,
    min_atr_ratio_20=0.95,
    extension_gate_enabled=False,
    max_extension_atr=1.40,
    trend_alignment_gate_enabled=False,
    price_oi_gate_enabled=False,
    min_price_oi_pressure=0.02,

    # Optional option-quality gates (off for autonomous exploration).
    option_liquidity_gate_enabled=False,
    min_option_volume=50.0,
    min_option_oi=500.0,
    option_delta_gate_enabled=False,
    min_abs_delta=0.35,
    max_abs_delta=0.70,

    # Structural-stop geometry gate (BANKNIFTY underlying, ATR-normalised).
    structural_distance_gate_enabled=False,
    min_structural_stop_atr=0.35,
    max_structural_stop_atr=1.50,

    regime_filter_enabled=False,
    allow_range_regime=False,
    allow_low_vol_regime=False,
    allow_normal_vol_regime=True,
    allow_high_vol_regime=True,
    allow_expiry_regime=False,
    first_entry_time_enabled=True,
    first_entry_time="09:15",
    last_entry_time_enabled=True,
    last_entry_time="15:00",

    # Causal higher-timeframe direction.
    # The current 1-minute decision may use only HTF bars that are fully closed
    # by the end of that 1-minute candle (timestamp + 1 minute).
    htf_direction_enabled=False,
    htf_primary_minutes=5,
    htf_secondary_enabled=True,
    htf_secondary_minutes=15,
    htf_fast_ema_span=2,
    htf_slow_ema_span=4,
    htf_return_bars=1,
    htf_primary_min_return=0.00020,
    htf_secondary_min_return=0.00030,
    htf_entry_gate_enabled=False,
    htf_entry_require_secondary=True,
    htf_exit_on_reversal_enabled=False,
    htf_exit_require_secondary_not_supporting=True,
    htf_exit_on_neutral_enabled=False,

    # Structural/reference exits
    structural_stop_enabled=False,
    reference_structural_lookback=45,
    reference_pivot_left=2,
    reference_pivot_right=2,
    reference_structural_buffer_fraction=0.20,
    momentum_decay_exit_enabled=False,
    reference_exit_threshold=0.48,
    underlying_target_enabled=False,
    reference_target_underlying_points=160.0,
    volatility_target_enabled=False,
    underlying_target_atr_multiplier=4.0,
    underlying_target_min_fraction=0.60,

    # Fixed option-premium take-profit.  Both CE and PE are long-premium trades,
    # therefore profit means the selected option premium rises by this many points.
    fixed_option_target_enabled=False,
    fixed_option_target_points=40.0,

    # Independent fixed option-premium stop-loss.
    # This is deliberately separate from fixed_option_target_points so that,
    # for example, a +40 point target can be paired with a -20 point stop.
    fixed_option_stop_enabled=False,
    fixed_option_stop_points=20.0,

    # Legacy R-multiple target retained for A/B tests, disabled in this experiment.
    risk_reward_target_enabled=False,
    risk_reward_target_r=1.50,

    # Learned profitable and losing exits share the EXIT action.
    learned_exit_enabled=True,
    agent_exit_penalty_enabled=False,
    agent_exit_penalty_r=0.005,

    # Emergency premium stop and the initial R scale; both configurable.
    emergency_stop_enabled=True,
    atr_period=14,
    min_stop_enabled=True,
    min_stop_points=30.0,
    premium_stop_enabled=True,
    stop_premium_pct=0.10,
    use_atr_stop=True,
    stop_atr_multiplier=1.2,

    # Position/time controls
    min_hold_enabled=False,
    min_hold_minutes=3,
    cooldown_enabled=False,
    reentry_cooldown_minutes=10,
    adaptive_cooldown_enabled=False,
    target_exit_cooldown_minutes=5,
    structural_stop_cooldown_minutes=20,
    losing_exit_cooldown_minutes=15,
    max_hold_enabled=False,
    max_hold_minutes=45,
    square_off_enabled=True,
    square_off_time="15:25",
    min_desired_move_points=40.0,

    # Session budgets use net premium points after modeled execution costs.
    # Reaching a budget latches entry blocking until the next episode/day.
    daily_profit_limit_enabled=False,
    daily_profit_target_points=40.0,
    daily_loss_limit_enabled=False,
    daily_loss_limit_points=20.0,
    close_on_daily_limit_enabled=True,

    # Optional reward shaping starts off; real net P&L is the primary reward.
    entry_penalty_enabled=False,
    entry_penalty_r=0.03,
    overtrading_penalty_enabled=False,
    free_trades_per_day=4,
    extra_trade_penalty_r=0.05,

    # Tiny opportunity-cost shaping for a deliberate WAIT.
    # This is charged only when flat and BUY_CE or BUY_PE is actually available.
    voluntary_wait_penalty_enabled=False,
    voluntary_wait_penalty_r=0.001,
    max_voluntary_wait_penalty_r_per_day=0.05,

    hold_shaping_enabled=False,
    hold_bonus_r=0.002,
    giveback_penalty_enabled=False,
    giveback_penalty_r=0.002,
    giveback_fraction=0.5,
    giveback_min_mfe_r=1.0,
    max_shaping_r_per_trade=0.10,
    terminal_win_shaping_enabled=False,
    terminal_win_bonus_r=0.03,
    terminal_win_min_points=10.0,
    terminal_loss_shaping_enabled=False,
    terminal_loss_penalty_r=0.05,
    # Optional proportional downside aversion. A 1.15 multiplier adds a
    # 0.15*abs(net R) penalty at close, bounded by the shared shaping cap.
    loss_aversion_enabled=False,
    loss_aversion_multiplier=1.15,
    exit_reason_shaping_enabled=False,
    option_target_bonus_r=0.03,
    structural_stop_penalty_r=0.08,
    emergency_stop_penalty_r=0.10,
    time_limit_loss_penalty_r=0.03,
    momentum_decay_loss_penalty_r=0.02,

    # Execution/cost model
    dynamic_slippage_enabled=True,
    transaction_costs_enabled=True,
    trade_quantity=30,
    brokerage_per_order=20.0,
    slippage_pct=0.001,
    reward_scale=1.0,
    option_stt_sell_rate=0.001,
    nse_option_txn_rate=35.03 / 1e7,
    sebi_rate=10.0 / 1e7,
    stamp_duty_buy_rate=0.00003,
    gst_rate=0.18,
)

ACTION_NAMES = ("WAIT", "BUY_CE", "BUY_PE", "HOLD", "EXIT")
WAIT, BUY_CE, BUY_PE, HOLD, EXIT = range(5)

POSITION_FEATURES = (
    "position_type", "bars_held_normalized", "minutes_held_normalized",
    "cooldown_remaining_normalized", "bars_since_exit_normalized",
    "current_option_return_pct", "current_option_pnl_points", "initial_risk_points",
    "current_R", "MFE_option_points", "MAE_option_points", "MFE_R", "MAE_R",
    "distance_from_entry_points", "distance_from_position_high_points",
    "distance_from_position_low_points", "has_reached_40_points", "held_option_atr",
    "atr_available", "quote_age_minutes", "pending_exit", "delta_change_since_entry",
    "iv_change_since_entry", "velocity_change_since_entry", "acceleration_change_since_entry",
    "momentum_score_change_since_entry", "delta_change_available", "iv_change_available",
    "velocity_change_available", "acceleration_change_available", "momentum_change_available",
    "tradeability_score", "tradeability_score_available",
    "directional_momentum_score", "path_efficiency_value", "atr_expansion_value",
    "extension_atr_value", "price_oi_pressure_value",
    "reference_bull", "reference_bear", "reference_stop_distance_underlying",
    "reference_scores_available",
    "htf_primary_direction", "htf_secondary_direction", "htf_alignment",
    "htf_primary_return", "htf_secondary_return",
    "htf_primary_available", "htf_secondary_available", "htf_fast_exit_signal",
    "net_liquidation_R", "remaining_session_fraction",
    "held_volume", "held_oi", "held_iv", "held_delta", "held_gamma", "held_theta", "held_vega",
    "held_volume_available", "held_oi_available", "held_iv_available", "held_delta_available",
    "held_gamma_available", "held_theta_available", "held_vega_available",
    "daily_realized_points", "daily_net_points", "daily_limit_reached",
    "daily_profit_remaining_points", "daily_loss_remaining_points",
)

# Simulation charge defaults. Verify against the broker/exchange schedule used in
# production. They are deliberately centralized so updating rates is trivial.
OPTION_STT_SELL_RATE = 0.001          # 0.10% on option sell premium turnover
NSE_OPTION_TXN_RATE = 35.03 / 1e7    # Rs 35.03/crore
SEBI_RATE = 10.0 / 1e7               # Rs 10/crore
STAMP_DUTY_BUY_RATE = 0.00003        # 0.003% buy-side
GST_RATE = 0.18


def parse_clock(value, label):
    try:
        parsed = datetime.strptime(value, "%H:%M").time()
    except (ValueError, TypeError):
        raise ValueError(f"{label} must be HH:MM") from None
    if parsed.strftime("%H:%M") != value:
        raise ValueError(f"{label} must be HH:MM")
    return parsed


def validate_config(payload):
    if not isinstance(payload, dict):
        raise ValueError("Environment settings must be an object")
    unknown = set(payload) - set(ENV_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown/obsolete settings: {sorted(unknown)}")

    cfg = {}
    for key, default in ENV_DEFAULTS.items():
        value = payload.get(key, default)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif isinstance(default, str):
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string")
            if key != "disabled_features":
                parse_clock(value, key)
        else:
            if isinstance(value, bool):
                raise ValueError(f"{key} must be numeric")
            value = float(value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            if isinstance(default, int):
                if not value.is_integer():
                    raise ValueError(f"{key} must be an integer")
                value = int(value)
        cfg[key] = value

    probability_keys = (
        "random_start_probability",
        "reference_ignition_min", "reference_direction_gap",
        "reference_exit_threshold", "tradeability_threshold",
        "min_path_efficiency", "underlying_target_min_fraction",
        "giveback_fraction", "min_abs_delta", "max_abs_delta",
        "ce_reference_ignition_min", "pe_reference_ignition_min",
        "ce_reference_direction_gap", "pe_reference_direction_gap",
    )
    for key in probability_keys:
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")

    positive_keys = (
        "observation_history_bars",
        "reference_window", "reference_strike_spacing", "reference_structural_lookback",
        "reference_pivot_left", "reference_pivot_right", "atr_period",
        "risk_reward_target_r", "stop_atr_multiplier", "max_hold_minutes",
        "trade_quantity", "reward_scale", "underlying_target_atr_multiplier",
        "fixed_option_target_points", "fixed_option_stop_points",
        "max_structural_stop_atr",
        "daily_profit_target_points", "daily_loss_limit_points",
        "loss_aversion_multiplier",
        "htf_primary_minutes", "htf_secondary_minutes",
        "htf_fast_ema_span", "htf_slow_ema_span", "htf_return_bars",
    )
    for key in positive_keys:
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")

    if cfg["reference_window"] < 8:
        raise ValueError("reference_window must be >= 8")
    if cfg["observation_history_bars"] > 32:
        raise ValueError("observation_history_bars must be <= 32")
    if cfg["path_efficiency_window"] < 2:
        raise ValueError("path_efficiency_window must be >= 2")
    if cfg["reference_structural_lookback"] < cfg["reference_pivot_left"] + cfg["reference_pivot_right"] + 1:
        raise ValueError("Structural lookback must contain both pivot confirmation widths")
    if cfg["min_hold_minutes"] > cfg["max_hold_minutes"] and cfg["min_hold_enabled"] and cfg["max_hold_enabled"]:
        raise ValueError("max_hold_minutes must be >= min_hold_minutes")
    if cfg["min_abs_delta"] > cfg["max_abs_delta"]:
        raise ValueError("min_abs_delta must be <= max_abs_delta")
    if cfg["min_structural_stop_atr"] > cfg["max_structural_stop_atr"]:
        raise ValueError("min_structural_stop_atr must be <= max_structural_stop_atr")
    if cfg["stop_premium_pct"] >= 1 or cfg["slippage_pct"] >= 1:
        raise ValueError("stop_premium_pct and slippage_pct must be < 1")
    if cfg["max_shaping_r_per_trade"] > 0.25:
        raise ValueError("Keep shaping modest: max_shaping_r_per_trade <= 0.25R")
    if cfg["loss_aversion_multiplier"] < 1:
        raise ValueError("loss_aversion_multiplier must be >= 1")
    if cfg["htf_fast_ema_span"] >= cfg["htf_slow_ema_span"]:
        raise ValueError("htf_fast_ema_span must be < htf_slow_ema_span")
    if cfg["htf_primary_minutes"] < 2 or cfg["htf_secondary_minutes"] < 2:
        raise ValueError("HTF minute values must be >= 2")
    if cfg["htf_secondary_enabled"] and cfg["htf_secondary_minutes"] <= cfg["htf_primary_minutes"]:
        raise ValueError("htf_secondary_minutes must be greater than htf_primary_minutes")
    if cfg["htf_primary_min_return"] >= 1 or cfg["htf_secondary_min_return"] >= 1:
        raise ValueError("HTF minimum returns must be fractional values below 1")

    first = parse_clock(cfg["first_entry_time"], "first_entry_time")
    last = parse_clock(cfg["last_entry_time"], "last_entry_time")
    square = parse_clock(cfg["square_off_time"], "square_off_time")
    if cfg["first_entry_time_enabled"] and cfg["last_entry_time_enabled"] and first >= last:
        raise ValueError("first_entry_time must be before last_entry_time")
    if cfg["last_entry_time_enabled"] and cfg["square_off_enabled"] and last > square:
        raise ValueError("last_entry_time must be <= square_off_time")
    return cfg


class BankNiftyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self, observations_file, option_market_dir, manifest_file,
        split="train", random_day=True, state=None, model_id="unassigned", **config
    ):
        super().__init__()
        self.config = validate_config(config)
        for key, value in self.config.items():
            setattr(self, key, value)

        self.first_entry_time = parse_clock(self.first_entry_time, "first entry")
        self.last_entry_time = parse_clock(self.last_entry_time, "entry cutoff")
        self.square_off_time = parse_clock(self.square_off_time, "square-off")
        self.random_day = bool(random_day and split == "train")
        self.state, self.model_id, self.split = state, model_id, split
        self.option_market_dir = Path(option_market_dir)

        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)
        self.feature_columns = manifest["observation_columns"]
        self.dataset_id = manifest.get("dataset_id")
        if manifest.get("dataset_version") == DATA_VERSION:
            df, manifest = load_dataset(observations_file, manifest_file, option_market_dir)
        else:
            # Tiny fixtures and legacy candidate tooling may supply their own schema.
            df = pd.read_parquet(observations_file)
        df = df.copy(deep=False)
        df["timestamp"] = pd.to_datetime(df.timestamp)
        df["expiry"] = pd.to_datetime(df.expiry)
        disabled = {s.strip() for s in self.disabled_features.split(",") if s.strip()}
        unknown = disabled - set(self.feature_columns) - set(POSITION_FEATURES)
        if unknown:
            raise ValueError(f"Unknown disabled features: {sorted(unknown)}")
        self._disabled_features = disabled
        self._feature_enabled = np.array([
            c not in disabled and c.removesuffix("__available") not in disabled
            and getattr(self, f"features_{feature_group(c)}_enabled")
            and (self.missingness_features_enabled or not c.endswith("__available"))
            for c in self.feature_columns
        ], dtype=np.float32)
        self.df = df.loc[df.split == split].sort_values("timestamp").reset_index(drop=True)
        self.df["trading_date"] = self.df.timestamp.dt.date
        self.days = sorted(self.df.trading_date.unique())
        if not self.days:
            raise ValueError(f"No days in split {split}")

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            -np.inf, np.inf,
            shape=(len(self.feature_columns) * self.observation_history_bars
                   + len(POSITION_FEATURES)
                   + (self.observation_history_bars if self.observation_history_bars > 1 else 0),),
            dtype=np.float32,
        )

        self.loaded_expiry = None
        self.option_lookup = {}
        self.reference_options = None
        self.position = None
        self.trade_log = []
        self.interrupted_trade_log = []
        self._eval_day_idx = 0
        self._terminated = True
        self._action_probability = None
        self._candidate_stats = {}

    # ------------------------------------------------------------------
    # Market loading
    # ------------------------------------------------------------------

    def _load_option_partition(self, expiry):
        expiry = pd.Timestamp(expiry).normalize()
        if expiry == self.loaded_expiry:
            return

        self.option_lookup.clear()
        self.reference_options = None
        path = self.option_market_dir / f"banknifty_options_{expiry:%Y-%m-%d}.parquet"
        options = pd.read_parquet(path)

        columns = [
            "groww_symbol", "timestamp", "open", "high", "low", "close",
            "delta", "iv", "gamma", "theta", "vega", "strike", "option_type", "volume", "oi",
        ]
        options = options.reindex(columns=columns).copy()
        options["timestamp"] = pd.to_datetime(options.timestamp)
        options["groww_symbol"] = options.groww_symbol.astype(str)
        for column in set(columns) - {"timestamp", "groww_symbol", "option_type"}:
            options[column] = pd.to_numeric(options[column], errors="coerce").astype(float)
        options = options.sort_values(["groww_symbol", "timestamp"]).drop_duplicates(
            ["groww_symbol", "timestamp"], keep="last"
        )

        prices = options[["open", "high", "low", "close"]].to_numpy(dtype=float)
        valid = (
            np.isfinite(prices).all(axis=1)
            & (prices > 0).all(axis=1)
            & (options.high >= options[["open", "close", "low"]].max(axis=1))
            & (options.low <= options[["open", "close", "high"]].min(axis=1))
        )

        # Preserve an independently valid OPEN for next-open execution.  Do not
        # use a bad candle's later H/L/C for ATR or stop logic.
        options.loc[~valid, ["high", "low", "close"]] = np.nan
        options.loc[~np.isfinite(options.open) | (options.open <= 0), "open"] = np.nan
        invalid_iv = options.iv.isna() | options.iv.le(0.005001) | options.iv.ge(4.99999)
        options.loc[invalid_iv, ["iv", "delta", "gamma", "theta", "vega"]] = np.nan
        options.loc[options.volume < 0, "volume"] = np.nan
        options.loc[options.oi < 0, "oi"] = np.nan

        keys = [options.groww_symbol, options.timestamp.dt.date]
        gap = (
            options.groupby(keys, observed=True).timestamp.diff().ne(pd.Timedelta(minutes=1))
            | ~valid
        )
        options["atr_segment"] = gap.cumsum()
        groups = options.groupby("atr_segment", sort=False)
        previous = groups.close.shift(1)
        options["true_range"] = pd.concat(
            [
                options.high - options.low,
                (options.high - previous).abs(),
                (options.low - previous).abs(),
            ],
            axis=1,
        ).max(axis=1)
        options["atr"] = options.groupby("atr_segment", sort=False).true_range.transform(
            lambda x: x.rolling(self.atr_period, min_periods=self.atr_period).mean()
        )

        self.option_lookup = {
            (r.groww_symbol, r.timestamp): r for r in options.itertuples(index=False)
        }

        reference_columns = [
            "timestamp", "groww_symbol", "strike", "option_type",
            "close", "volume", "oi"
        ]
        self.reference_options = (
            options[[c for c in reference_columns if c in options.columns]].copy()
            if (self.reference_strategy_enabled or self.reference_features_enabled
                or self.price_oi_gate_enabled or self.momentum_decay_exit_enabled) else None
        )
        self.loaded_expiry = expiry

    # ------------------------------------------------------------------
    # Episode reset / reference preparation
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.position is not None:
            row = self.day_df.iloc[self.step_idx]
            self._close_position(
                self.position["last_price"], "STALE_OPTION_DATA", row.timestamp,
                stale=True, detail="RESET_ABORTED"
            )
            self.interrupted_trade_log = list(self.trade_log)

        if options and "day_index" in options:
            index = int(options["day_index"])
        elif self.random_day:
            index = int(self.np_random.integers(len(self.days)))
        else:
            if seed is not None:
                self._eval_day_idx = 0
            index = self._eval_day_idx
            self._eval_day_idx = (index + 1) % len(self.days)

        self.current_day = self.days[index]
        self.day_df = self.df.loc[
            self.df.trading_date == self.current_day
        ].reset_index(drop=True)
        if len(self.day_df) < 2:
            raise ValueError(f"Insufficient rows on {self.current_day}")

        # Completed candles only; available even with the reference engine off.
        # A window of N bars contains N-1 price changes. Flat paths have score 0.
        close = self.day_df.close
        periods = self.path_efficiency_window - 1
        distance = close.diff().abs().rolling(periods, min_periods=periods).sum()
        self.day_df["path_efficiency"] = (
            close.diff(periods).abs() / distance.replace(0.0, np.nan)
        ).where(distance != 0.0, 0.0).clip(0.0, 1.0)

        self._prepare_htf_direction_day()
        self._load_option_partition(self.day_df.expiry.iloc[0])
        if (self.reference_strategy_enabled or self.reference_features_enabled
                or self.price_oi_gate_enabled or self.momentum_decay_exit_enabled):
            self._prepare_reference_day()
        else:
            self._prepare_underlying_context_only()

        self.step_idx = 0
        # Randomise time exposure, not future profitability. Evaluation always
        # starts at the beginning of the complete chronological session.
        if (self.random_day and self.random_start_enabled
                and self.np_random.random() < self.random_start_probability):
            times = self.day_df.timestamp
            eligible = (times <= times.iloc[0] + pd.Timedelta(minutes=self.random_start_max_minutes))
            if self.last_entry_time_enabled:
                eligible &= times.dt.time < self.last_entry_time
            if self.square_off_enabled:
                eligible &= times.dt.time < self.square_off_time
            indices = np.flatnonzero(eligible.to_numpy())
            indices = indices[indices < len(self.day_df) - 1]
            if len(indices):
                self.step_idx = int(self.np_random.choice(indices))
        self.episode_start_time = str(self.day_df.timestamp.iloc[self.step_idx])
        self.episode_partial_session = self.step_idx != 0
        self.episode_pnl = 0.0
        self.episode_reward = 0.0
        self._reward_value = 0.0
        self._terminal_shaping_pending = 0.0
        self.daily_entries = 0
        self.daily_limit_reason = ""
        self.daily_voluntary_wait_penalty_r = 0.0
        self.last_exit_time = None
        self.last_exit_step = None
        self.last_exit_cooldown_minutes = self.reentry_cooldown_minutes
        self.trade_log = []
        self.action_counts = dict.fromkeys(ACTION_NAMES, 0)
        self._terminated = False
        self._action_probability = None

        if self.state:
            self.state.start_day(str(self.current_day), str(self.day_df.expiry.iloc[0].date()))
        return self._get_observation(), {
            "date": str(self.current_day), "episode_start_time": self.episode_start_time,
            "partial_session": self.episode_partial_session,
        }

    def _market_history(self):
        """Oldest-to-newest completed minutes, padded across missing timestamps.

        Never forward-fill a gap or use another day's/future rows. Flags describe
        timestamp presence; individual feature availability remains in the data.
        """
        if self.observation_history_bars == 1:
            return self.day_df.iloc[self.step_idx][self.feature_columns].to_numpy(
                dtype=np.float32) * self._feature_enabled
        past = self.day_df.iloc[:self.step_idx + 1]
        clock = pd.date_range(end=past.timestamp.iloc[-1],
                              periods=self.observation_history_bars, freq="min")
        indices = pd.DatetimeIndex(past.timestamp).get_indexer(clock)
        present = indices >= 0
        frames = np.zeros((self.observation_history_bars, len(self.feature_columns)), np.float32)
        frames[present] = past.iloc[indices[present]][self.feature_columns].to_numpy(dtype=np.float32)
        frames *= self._feature_enabled
        return np.concatenate((frames.ravel(), present.astype(np.float32)))

    def _no_future_entries(self):
        """Only known irreversible session restrictions justify early completion."""
        if self.position is not None:
            return False
        clock = self.day_df.timestamp.iloc[self.step_idx].time()
        return bool(self.daily_limit_reason
                    or (self.last_entry_time_enabled and clock >= self.last_entry_time)
                    or (self.square_off_enabled and clock >= self.square_off_time))

    def _prepare_htf_direction_day(self):
        """Build causal 5m/15m-style direction from completed BANKNIFTY bars.

        The source 1-minute timestamp is treated as the start of that minute.
        Therefore a row at 09:19 becomes fully known at 09:20.  HTF values are
        mapped with decision_time = timestamp + 1 minute, so an HTF bar is never
        visible before its own closing boundary.
        """
        for prefix in ("primary", "secondary"):
            self.day_df[f"htf_{prefix}_direction"] = 0.0
            self.day_df[f"htf_{prefix}_return"] = 0.0
            self.day_df[f"htf_{prefix}_available"] = 0.0

        if not self.htf_direction_enabled:
            return

        def build(minutes, min_return, prefix):
            source = self.day_df[["timestamp", "open", "high", "low", "close"]].copy()
            source = source.dropna(subset=["timestamp", "open", "high", "low", "close"])
            if source.empty:
                return

            source = source.sort_values("timestamp").set_index("timestamp")
            rule = f"{int(minutes)}min"
            origin = pd.Timestamp(self.day_df.timestamp.iloc[0])

            bars = source.resample(
                rule, origin=origin, label="right", closed="left"
            ).agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                count=("close", "count"),
            )

            # Require every constituent 1-minute candle.  A partial or gapped HTF
            # candle is never allowed to create direction.
            bars = bars.loc[bars["count"] >= int(minutes)].copy()
            if bars.empty:
                return

            bars["fast"] = bars["close"].ewm(
                span=int(self.htf_fast_ema_span), adjust=False
            ).mean()
            bars["slow"] = bars["close"].ewm(
                span=int(self.htf_slow_ema_span), adjust=False
            ).mean()
            bars["ret"] = bars["close"].pct_change(int(self.htf_return_bars))

            enough_history = np.arange(len(bars)) >= int(self.htf_return_bars)
            bull = (
                enough_history
                & bars["ret"].notna().to_numpy()
                & (bars["fast"].to_numpy() > bars["slow"].to_numpy())
                & (bars["ret"].to_numpy() >= float(min_return))
            )
            bear = (
                enough_history
                & bars["ret"].notna().to_numpy()
                & (bars["fast"].to_numpy() < bars["slow"].to_numpy())
                & (bars["ret"].to_numpy() <= -float(min_return))
            )
            bars["direction"] = np.select([bull, bear], [1.0, -1.0], default=0.0)
            bars["available"] = enough_history.astype(float)
            bars = bars.reset_index().rename(columns={"timestamp": "htf_close_time"})

            decisions = pd.DataFrame({
                "decision_time": self.day_df.timestamp + pd.Timedelta(minutes=1)
            })
            mapped = pd.merge_asof(
                decisions.sort_values("decision_time"),
                bars[["htf_close_time", "direction", "ret", "available"]]
                    .sort_values("htf_close_time"),
                left_on="decision_time",
                right_on="htf_close_time",
                direction="backward",
                allow_exact_matches=True,
            )

            self.day_df[f"htf_{prefix}_direction"] = (
                mapped["direction"].fillna(0.0).to_numpy(dtype=float)
            )
            self.day_df[f"htf_{prefix}_return"] = (
                mapped["ret"].fillna(0.0).to_numpy(dtype=float)
            )
            self.day_df[f"htf_{prefix}_available"] = (
                mapped["available"].fillna(0.0).to_numpy(dtype=float)
            )

        build(
            self.htf_primary_minutes,
            self.htf_primary_min_return,
            "primary",
        )
        if self.htf_secondary_enabled:
            build(
                self.htf_secondary_minutes,
                self.htf_secondary_min_return,
                "secondary",
            )

    def _htf_entry_allows(self, row, side):
        """Return (allowed, diagnostics) for causal HTF directional entry confirmation.

        Entry hierarchy:
        - primary HTF (typically 5m) defines the active direction;
        - secondary HTF (typically 15m) may confirm that direction, or when
          `htf_entry_require_secondary` is false it acts only as a veto.

        Veto mode:
        - CE requires primary bullish and secondary not bearish.
        - PE requires primary bearish and secondary not bullish.
        A neutral secondary timeframe therefore does not block a valid 5m move.
        """
        if not (self.htf_direction_enabled and self.htf_entry_gate_enabled):
            return True, {}

        sign = 1.0 if side == "CE" else -1.0
        p_avail = bool(self._number(row, "htf_primary_available") or 0.0)
        p_dir = self._number(row, "htf_primary_direction")
        s_avail = bool(self._number(row, "htf_secondary_available") or 0.0)
        s_dir = self._number(row, "htf_secondary_direction")

        details = {
            "primary_available": p_avail,
            "primary_direction": p_dir,
            "secondary_available": s_avail,
            "secondary_direction": s_dir,
            "secondary_mode": (
                "confirm" if self.htf_entry_require_secondary else "veto"
            ),
        }

        # Primary HTF must always define the active trade side.
        if not p_avail or p_dir != sign:
            return False, details

        if self.htf_secondary_enabled:
            if self.htf_entry_require_secondary:
                # Strict alignment mode.
                if not s_avail or s_dir != sign:
                    return False, details
            else:
                # Veto mode: neutral/unavailable secondary does not block,
                # but an explicit opposite secondary trend does.
                if s_avail and s_dir == -sign:
                    return False, details

        return True, details

    def _htf_exit_reason(self, row, position):
        """Return a forced causal HTF exit reason when direction materially changes.

        Exit hierarchy:
        1. If the secondary HTF fully reverses against the position, force exit.
        2. If the primary HTF reverses and the secondary no longer supports the
           held side (neutral/opposite/unavailable when confirmation is required),
           force exit.
        3. A primary-only reversal does not force an exit; it is handled by
           `_htf_fast_exit_condition`, which makes PPO EXIT immediately available
           even before the normal minimum-hold period has elapsed.
        """
        if not (self.htf_direction_enabled and self.htf_exit_on_reversal_enabled):
            return None

        sign = 1.0 if position["type"] == "CE" else -1.0
        p_avail = bool(self._number(row, "htf_primary_available") or 0.0)
        p_dir = self._number(row, "htf_primary_direction")
        s_avail = bool(self._number(row, "htf_secondary_available") or 0.0)
        s_dir = self._number(row, "htf_secondary_direction")

        primary_reversed = p_avail and p_dir == -sign
        primary_neutral = p_avail and p_dir == 0
        secondary_reversed = self.htf_secondary_enabled and s_avail and s_dir == -sign
        secondary_supports = self.htf_secondary_enabled and s_avail and s_dir == sign

        # A full 15m reversal is strong enough to force the position out.
        if secondary_reversed:
            return "HTF_SECONDARY_REVERSAL"

        # 5m reversal plus loss of 15m support is a confirmed reversal.
        if primary_reversed:
            if not self.htf_secondary_enabled:
                return "HTF_DIRECTION_REVERSAL"

            if self.htf_exit_require_secondary_not_supporting:
                if not secondary_supports:
                    return "HTF_DIRECTION_REVERSAL"
            else:
                return "HTF_DIRECTION_REVERSAL"

        if self.htf_exit_on_neutral_enabled and primary_neutral:
            if not self.htf_secondary_enabled or not secondary_supports:
                return "HTF_DIRECTION_NEUTRAL"

        return None

    def _htf_fast_exit_condition(self, row, position):
        """Primary-HTF reversal unlocks PPO EXIT immediately.

        This is deliberately softer than a forced exit. It lets the 1-minute
        policy decide whether to exit at once when 5m turns against the trade,
        even if `min_hold_minutes` has not yet elapsed.
        """
        if not (self.htf_direction_enabled and self.htf_exit_on_reversal_enabled):
            return False

        sign = 1.0 if position["type"] == "CE" else -1.0
        p_avail = bool(self._number(row, "htf_primary_available") or 0.0)
        p_dir = self._number(row, "htf_primary_direction")
        return bool(p_avail and p_dir == -sign)

    def _prepare_underlying_context_only(self):
        prev = self.day_df.close.shift(1)
        tr = pd.concat(
            [
                self.day_df.high - self.day_df.low,
                (self.day_df.high - prev).abs(),
                (self.day_df.low - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        self.day_df["reference_underlying_atr"] = tr.rolling(
            self.atr_period, min_periods=self.atr_period
        ).mean()
        self._candidate_stats = {}

    def _prepare_reference_day(self):
        options = self.reference_options
        start = pd.Timestamp(self.current_day)
        options = options.loc[
            (options.timestamp >= start)
            & (options.timestamp < start + pd.Timedelta(days=1))
        ]

        needed = {"strike", "option_type", "oi", "volume"}
        if not needed.issubset(options.columns):
            raise ValueError(f"Reference strategy needs option columns: {sorted(needed)}")

        scores = ignition_features(
            self.day_df, options, self.reference_window,
            self.reference_strike_window, self.reference_strike_spacing
        )
        for column in scores:
            self.day_df[column] = scores[column].to_numpy()

        # Causal underlying ATR used to freeze the volatility-normalized target
        # at entry.  No future range is involved.
        prev = self.day_df.close.shift(1)
        tr = pd.concat(
            [
                self.day_df.high - self.day_df.low,
                (self.day_df.high - prev).abs(),
                (self.day_df.low - prev).abs(),
            ],
            axis=1,
        ).max(axis=1)
        self.day_df["reference_underlying_atr"] = tr.rolling(
            self.atr_period, min_periods=self.atr_period
        ).mean()

        if not self.reference_strategy_enabled:
            # Advisory calculations do not select a side, strike or entry time.
            self.day_df["reference_side"] = None
            self.day_df["reference_symbol"] = None
            self.day_df["reference_stop"] = np.nan
            self._candidate_stats = {}
            return

        snapshots = {
            ts: group for ts, group in options.groupby("timestamp", sort=False)
        }
        sides, symbols, stops = [], [], []
        signal_side, signal_time = None, None

        for i, row in self.day_df.iterrows():
            if (
                self.reference_signal_ttl_enabled
                and signal_time is not None
                and (row.timestamp - signal_time).total_seconds() / 60
                > self.reference_signal_ttl_minutes
            ):
                signal_side, signal_time = None, None

            bull, bear = row.reference_bull, row.reference_bear
            if np.isfinite([bull, bear]).all() and bull != bear:
                side = "CE" if bull > bear else "PE"
                if self.directional_thresholds_enabled:
                    ignition_min = (self.ce_reference_ignition_min if side == "CE"
                                    else self.pe_reference_ignition_min)
                    direction_gap = (self.ce_reference_direction_gap if side == "CE"
                                     else self.pe_reference_direction_gap)
                else:
                    ignition_min = self.reference_ignition_min
                    direction_gap = self.reference_direction_gap
                qualifies = (
                    (not self.reference_score_gate_enabled or max(bull, bear) >= ignition_min)
                    and (not self.reference_score_gate_enabled or abs(bull - bear) >= direction_gap)
                )
                if qualifies and side != signal_side:
                    signal_side, signal_time = side, row.timestamp

            symbol, stop = None, None
            if signal_side and np.isfinite(row.close) and row.timestamp in snapshots:
                symbol = select_contract(
                    snapshots[row.timestamp], signal_side, row.close,
                    self.reference_strike_spacing
                )
                stop = structural_stop(
                    self.day_df.iloc[:i+1], signal_side, row.close,
                    self.reference_structural_lookback,
                    self.reference_pivot_left, self.reference_pivot_right,
                    self.reference_structural_buffer_fraction,
                )

            sides.append(signal_side)
            symbols.append(symbol)
            stops.append(stop)

        self.day_df["reference_side"] = sides
        self.day_df["reference_symbol"] = symbols
        self.day_df["reference_stop"] = stops

        # Future information is used ONLY here for post-hoc candidate diagnostics.
        # These values are never added to observations or action masks.
        self._candidate_stats = self._compute_candidate_recall_diagnostics()

    def _dynamic_underlying_target(self, row):
        base = float(self.reference_target_underlying_points)
        if not self.volatility_target_enabled:
            return base
        atr = self._number(row, "reference_underlying_atr", "atr_14")
        if atr is None or atr <= 0:
            return base
        return float(max(self.underlying_target_min_fraction * base,
                         self.underlying_target_atr_multiplier * atr))

    def _compute_candidate_recall_diagnostics(self, horizon=15):
        result = dict(
            horizon_minutes=horizon,
            bullish_opportunities=0, bearish_opportunities=0,
            bullish_captured=0, bearish_captured=0,
            bullish_candidates=0, bearish_candidates=0,
            bullish_candidates_successful=0, bearish_candidates_successful=0,
        )
        n = len(self.day_df)
        if n <= horizon:
            return result

        for i in range(0, n - horizon):
            row = self.day_df.iloc[i]
            future = self.day_df.iloc[i+1:i+1+horizon]
            if future.empty or not np.isfinite(row.close):
                continue

            # Diagnostic move threshold is slightly less strict than the mandatory
            # target so recall measures whether ignition sees important momentum.
            atr = self._number(row, "reference_underlying_atr", "atr_14")
            threshold = max(
                0.50 * self.reference_target_underlying_points,
                2.5 * atr if atr is not None and atr > 0 else 0.0,
            )

            up_move = float(future.high.max() - row.close)
            down_move = float(row.close - future.low.min())
            bull_opp = up_move >= threshold
            bear_opp = down_move >= threshold
            side = row.get("reference_side")

            if bull_opp:
                result["bullish_opportunities"] += 1
                result["bullish_captured"] += int(side == "CE")
            if bear_opp:
                result["bearish_opportunities"] += 1
                result["bearish_captured"] += int(side == "PE")

            if side == "CE":
                result["bullish_candidates"] += 1
                result["bullish_candidates_successful"] += int(bull_opp)
            elif side == "PE":
                result["bearish_candidates"] += 1
                result["bearish_candidates_successful"] += int(bear_opp)

        bull_n = result["bullish_opportunities"]
        bear_n = result["bearish_opportunities"]
        bc = result["bullish_candidates"]
        pc = result["bearish_candidates"]
        total_opp = bull_n + bear_n
        total_hit = result["bullish_captured"] + result["bearish_captured"]
        total_candidates = bc + pc
        total_success = (
            result["bullish_candidates_successful"]
            + result["bearish_candidates_successful"]
        )
        result["candidate_recall_pct"] = 100.0 * total_hit / total_opp if total_opp else None
        result["candidate_precision_pct"] = (
            100.0 * total_success / total_candidates if total_candidates else None
        )
        return result

    def candidate_diagnostics(self):
        return dict(self._candidate_stats)

    # ------------------------------------------------------------------
    # Utility / observations
    # ------------------------------------------------------------------

    def _get_option_row(self, symbol, timestamp):
        if symbol is None or pd.isna(symbol):
            return None
        return self.option_lookup.get((str(symbol), pd.Timestamp(timestamp)))

    @staticmethod
    def _number(source, *names):
        for name in names:
            if isinstance(source, (dict, pd.Series)):
                available = source.get(name + "__available", 1)
                if pd.isna(available) or available == 0:
                    continue
            value = (
                source.get(name)
                if isinstance(source, (dict, pd.Series))
                else getattr(source, name, None)
            )
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
        return None

    def _context(self, row, quote):
        return dict(
            delta=self._number(quote, "delta"),
            iv=self._number(quote, "iv"),
            velocity=self._number(row, "price_velocity", "velocity", "return_3m"),
            acceleration=self._number(row, "price_acceleration", "acceleration"),
            momentum=self._number(row, "momentum_score", "momentum_3", "return_5m"),
        )

    def _session_quality(self, timestamp):
        """Causal soft time-of-day score; never blocks a trade by itself."""
        minute = timestamp.hour * 60 + timestamp.minute
        if minute < 10 * 60 + 45:       # opening momentum window
            return 1.0
        if minute < 13 * 60 + 30:       # midday tends to be less directional
            return 0.65
        return 0.85                      # afternoon momentum window

    def _structural_distance_atr(self, row, side):
        stop = self._number(row, "reference_stop")
        close = self._number(row, "close")
        atr = self._number(row, "reference_underlying_atr", "atr_14")
        if stop is None or close is None or atr is None or atr <= 0:
            return None
        sign = 1.0 if side == "CE" else -1.0
        return sign * (close - stop) / atr

    def _cooldown_for_reason(self, reason, net_points):
        if not self.adaptive_cooldown_enabled:
            return int(self.reentry_cooldown_minutes)
        if reason in {"OPTION_TARGET", "OPTION_TARGET_GAP"}:
            return int(self.target_exit_cooldown_minutes)
        if reason == "STRUCTURAL_STOP":
            return int(self.structural_stop_cooldown_minutes)
        if net_points < 0:
            return int(self.losing_exit_cooldown_minutes)
        return int(self.reentry_cooldown_minutes)

    def _target_enabled(self):
        return bool(self.fixed_option_target_enabled or self.risk_reward_target_enabled)

    def _quality(self, row, side):
        sign = 1.0 if side == "CE" else -1.0
        components = []

        def add(value, weight=1.0):
            if value is not None and np.isfinite(value):
                components.append((float(np.clip(value, 0.0, 1.0)), float(weight)))

        ref = self._number(
            row,
            "reference_bull" if side == "CE" else "reference_bear",
            "ignition_bull" if side == "CE" else "ignition_bear",
            "p_bull" if side == "CE" else "p_bear",
        )
        if ref is not None:
            add(ref, 2.0)

        if self.momentum_gate_enabled:
            r3 = self._number(row, "return_3m", "price_velocity", "velocity")
            r5 = self._number(row, "return_5m", "momentum_3")
            if r3 is not None:
                add(0.5 + 0.5*np.tanh(sign*r3/0.001), 1.0)
            if r5 is not None:
                add(0.5 + 0.5*np.tanh(sign*r5/0.0015), 1.0)

        if self.soft_path_efficiency_enabled:
            eff = self._number(row, "path_efficiency")
            if eff is not None:
                add(eff, 0.75)

        if self.soft_atr_quality_enabled:
            ratio = self._number(row, "atr_ratio_20")
            if ratio is not None:
                # Best around moderate expansion; both dead volatility and exhaustion score lower.
                atr_score = np.exp(-((ratio - 1.15) / 0.35) ** 2)
                add(atr_score, 0.75)

        if self.soft_price_oi_quality_enabled:
            pressure = self._number(row, "reference_price_oi_pressure")
            if pressure is not None:
                add(0.5 + 0.5*np.tanh(3.0*sign*pressure), 0.75)

        if self.soft_structural_distance_enabled:
            dist = self._structural_distance_atr(row, side)
            if dist is not None:
                structural_score = np.exp(-((dist - 0.90) / 0.60) ** 2)
                add(structural_score, 1.0)

        if self.soft_time_quality_enabled:
            add(self._session_quality(row.timestamp), 0.50)

        # Hard-gate variables may also contribute when explicitly enabled.
        if self.path_efficiency_gate_enabled:
            eff = self._number(row, "path_efficiency")
            if eff is not None:
                add(eff, 0.50)
        if self.atr_expansion_gate_enabled:
            ratio = self._number(row, "atr_ratio_20")
            if ratio is not None:
                add(np.clip(ratio/1.5, 0.0, 1.0), 0.50)
        if self.price_oi_gate_enabled:
            pressure = self._number(row, "reference_price_oi_pressure")
            if pressure is not None:
                add(0.5 + 0.5*np.tanh(sign*pressure), 0.50)

        if not components:
            return 0.0, False
        total_weight = sum(w for _, w in components)
        return float(sum(v*w for v, w in components) / total_weight), True

    def _entry_gate_details(self, row, side, quote):
        """Causal high-precision entry filters. Returns (valid, details)."""
        sign = 1.0 if side == "CE" else -1.0
        details = {}
        valid = True

        def check(name, passed, available=True):
            if not available:
                details.setdefault("missing", []).append(name)
                passed = self.missing_filter_values_pass
            details.setdefault("checks", {})[name] = bool(passed)
            return bool(passed)

        if self.momentum_gate_enabled:
            r3 = self._number(row, "return_3m", "price_velocity", "velocity")
            r5 = self._number(row, "return_5m", "momentum_3")
            if self.directional_thresholds_enabled:
                min_r3 = (self.ce_min_directional_return_3m if side == "CE"
                          else self.pe_min_directional_return_3m)
                min_r5 = (self.ce_min_directional_return_5m if side == "CE"
                          else self.pe_min_directional_return_5m)
            else:
                min_r3, min_r5 = self.min_directional_return_3m, self.min_directional_return_5m
            details["r3"] = r3
            details["r5"] = r5
            valid &= check("momentum_3m", r3 is not None and sign*r3 >= min_r3, r3 is not None)
            valid &= check("momentum_5m", r5 is not None and sign*r5 >= min_r5, r5 is not None)

        if self.momentum_acceleration_gate_enabled:
            acceleration = self._number(row, "price_acceleration", "acceleration")
            details["acceleration"] = acceleration
            valid &= check(
                "momentum_acceleration",
                acceleration is not None and sign*acceleration >= self.min_directional_acceleration,
                acceleration is not None,
            )

        if self.path_efficiency_gate_enabled:
            eff = self._number(row, "path_efficiency")
            details["path_efficiency"] = eff
            valid &= check("path_efficiency", eff is not None and abs(eff) >= self.min_path_efficiency, eff is not None)

        if self.atr_expansion_gate_enabled:
            ratio = self._number(row, "atr_ratio_20")
            details["atr_ratio_20"] = ratio
            valid &= check("atr_expansion", ratio is not None and ratio >= self.min_atr_ratio_20, ratio is not None)

        if self.trend_alignment_gate_enabled:
            close = self._number(row, "close")
            e9 = self._number(row, "ema_9")
            e20 = self._number(row, "ema_20")
            e50 = self._number(row, "ema_50")
            aligned = close is not None and e20 is not None and sign*(close-e20) > 0
            if e9 is not None:
                aligned = aligned and sign*(e9-e20) >= 0
            if e50 is not None:
                aligned = aligned and sign*(e20-e50) >= 0
            details["trend_aligned"] = float(bool(aligned))
            valid &= check("trend_alignment", aligned, close is not None and e20 is not None)

        if self.extension_gate_enabled:
            close = self._number(row, "close")
            e20 = self._number(row, "ema_20")
            atr = self._number(row, "reference_underlying_atr", "atr_14")
            extension = None
            if close is not None and e20 is not None and atr is not None and atr > 0:
                extension = sign*(close-e20)/atr
            details["extension_atr"] = extension
            valid &= check("extension", extension is not None and extension <= self.max_extension_atr, extension is not None)

        if self.price_oi_gate_enabled:
            pressure = self._number(row, "reference_price_oi_pressure")
            details["price_oi_pressure"] = pressure
            valid &= check("price_oi", pressure is not None and sign*pressure >= self.min_price_oi_pressure, pressure is not None)

        if self.option_liquidity_gate_enabled:
            volume = self._number(quote, "volume")
            oi = self._number(quote, "oi")
            details["option_volume"] = volume
            details["option_oi"] = oi
            valid &= check("option_volume", volume is not None and volume >= self.min_option_volume, volume is not None)
            valid &= check("option_oi", oi is not None and oi >= self.min_option_oi, oi is not None)

        if self.option_delta_gate_enabled:
            delta = self._number(quote, "delta")
            details["abs_delta"] = abs(delta) if delta is not None else None
            valid &= check("option_delta", delta is not None and self.min_abs_delta <= abs(delta) <= self.max_abs_delta, delta is not None)

        if self.structural_distance_gate_enabled and self.reference_strategy_enabled:
            dist = self._structural_distance_atr(row, side)
            details["structural_stop_atr"] = dist
            valid &= check(
                "structural_distance",
                dist is not None and self.min_structural_stop_atr <= dist <= self.max_structural_stop_atr
            )

        if self.regime_filter_enabled:
            regime = self._market_regime(row)
            details["regime"] = regime
            trend, vol, expiry = regime.split("|")
            if trend == "RANGE" and not self.allow_range_regime:
                valid = False
            if vol == "LOW_VOL" and not self.allow_low_vol_regime:
                valid = False
            if vol == "NORMAL_VOL" and not self.allow_normal_vol_regime:
                valid = False
            if vol == "HIGH_VOL" and not self.allow_high_vol_regime:
                valid = False
            if expiry == "EXPIRY" and not self.allow_expiry_regime:
                valid = False
            check("regime", ((trend != "RANGE" or self.allow_range_regime)
                  and (vol != "LOW_VOL" or self.allow_low_vol_regime)
                  and (vol != "NORMAL_VOL" or self.allow_normal_vol_regime)
                  and (vol != "HIGH_VOL" or self.allow_high_vol_regime)
                  and (expiry != "EXPIRY" or self.allow_expiry_regime)))

        return bool(valid), details

    def _momentum_favorable(self, row, side):
        value = self._number(
            row, "momentum_score", "price_velocity",
            "velocity", "return_3m", "momentum_3"
        )
        return value is not None and value * (1 if side == "CE" else -1) > 0

    def _risk_distance(self, entry, atr):
        candidates = []
        if self.min_stop_enabled:
            candidates.append(float(self.min_stop_points))
        if self.premium_stop_enabled:
            candidates.append(float(self.stop_premium_pct) * float(entry))
        if self.use_atr_stop and atr is not None and np.isfinite(atr) and atr > 0:
            candidates.append(float(self.stop_atr_multiplier) * float(atr))
        if not candidates:
            # A tiny positive denominator is still required for R-normalisation.
            candidates.append(max(1.0, 0.01*float(entry)))
        return float(max(candidates))

    def cooldown_remaining(self):
        if not self.cooldown_enabled or self.last_exit_time is None:
            return 0.0
        next_open = self.day_df.iloc[self.step_idx].timestamp + pd.Timedelta(minutes=1)
        return max(
            0.0,
            (
                self.last_exit_time
                + pd.Timedelta(minutes=self.last_exit_cooldown_minutes)
                - next_open
            ).total_seconds() / 60,
        )

    @property
    def bars_since_exit(self):
        return 0 if self.last_exit_step is None else self.step_idx - self.last_exit_step

    def _market_regime(self, row):
        """Causal regime label for reporting/trade stratification."""
        ret15 = self._number(row, "return_15m", "return_10m", "return_5m") or 0.0
        ema = self._number(row, "ema_9_20_spread")
        if ema is None:
            e9 = self._number(row, "ema_9")
            e20 = self._number(row, "ema_20")
            ema = (e9 / e20 - 1.0) if e9 and e20 else 0.0

        if ret15 > 0.001 and ema > 0:
            trend = "TREND_UP"
        elif ret15 < -0.001 and ema < 0:
            trend = "TREND_DOWN"
        else:
            trend = "RANGE"

        expansion = self._number(row, "atr_ratio_20")
        if expansion is None:
            expansion = 1.0
        vol = "HIGH_VOL" if expansion >= 1.20 else "LOW_VOL" if expansion <= 0.80 else "NORMAL_VOL"

        dte = self._number(row, "dte_days")
        expiry = "EXPIRY" if dte is not None and dte <= 1.0 else "NON_EXPIRY"
        return f"{trend}|{vol}|{expiry}"

    def _get_observation(self):
        row = self.day_df.iloc[self.step_idx]
        values = dict.fromkeys(POSITION_FEATURES, 0.0)
        values["cooldown_remaining_normalized"] = (
            self.cooldown_remaining() / max(self.reentry_cooldown_minutes, 1)
        )
        values["bars_since_exit_normalized"] = min(self.bars_since_exit, 375) / 375
        values["remaining_session_fraction"] = max(0, 930 - row.timestamp.hour*60 - row.timestamp.minute) / 375
        realized = sum(t["option_pnl_points"] for t in self.trade_log)
        net = self._daily_net_points()
        values.update(
            daily_realized_points=realized,
            daily_net_points=net,
            daily_limit_reached=float(bool(self.daily_limit_reason)),
            daily_profit_remaining_points=(max(0.0, self.daily_profit_target_points - net)
                                           if self.daily_profit_limit_enabled else 0.0),
            daily_loss_remaining_points=(max(0.0, self.daily_loss_limit_points + net)
                                         if self.daily_loss_limit_enabled else 0.0),
        )

        if self.position:
            p = self.position
            risk, entry, price = p["initial_risk_points"], p["entry_price"], p["last_price"]
            pnl = price - entry
            mfe = p["high_since_entry"] - entry
            mae = p["low_since_entry"] - entry

            values.update(
                position_type=1 if p["type"] == "CE" else -1,
                bars_held_normalized=p["bars_held"] / self.max_hold_minutes,
                minutes_held_normalized=p["minutes_held"] / self.max_hold_minutes,
                current_option_return_pct=100 * pnl / entry,
                current_option_pnl_points=pnl,
                initial_risk_points=risk,
                current_R=pnl / risk,
                MFE_option_points=mfe,
                MAE_option_points=mae,
                MFE_R=mfe / risk,
                MAE_R=mae / risk,
                distance_from_entry_points=pnl,
                distance_from_position_high_points=price - p["high_since_entry"],
                distance_from_position_low_points=price - p["low_since_entry"],
                has_reached_40_points=float(mfe >= self.min_desired_move_points),
                held_option_atr=p["last_atr"],
                atr_available=float(
                    p["atr_valid"] and p["last_quote_time"] == row.timestamp
                ),
                quote_age_minutes=max(
                    0.0, (row.timestamp - p["last_quote_time"]).total_seconds() / 60
                ),
                pending_exit=float(p["pending_exit"]),
                net_liquidation_R=self._liquidation_mark(p),
            )
            held = self._get_option_row(p["symbol"], row.timestamp)
            for field in ("volume", "oi", "iv", "delta", "gamma", "theta", "vega"):
                value = self._number(held, field)
                group = field if field in ("volume", "oi") else "greeks"
                if getattr(self, f"features_{group}_enabled"):
                    values[f"held_{field}"] = value if value is not None else 0.0
                    values[f"held_{field}_available"] = float(value is not None)

            current = self._context(
                row, self._get_option_row(p["symbol"], row.timestamp)
            )
            for key, feature, available in (
                ("delta", "delta_change_since_entry", "delta_change_available"),
                ("iv", "iv_change_since_entry", "iv_change_available"),
                ("velocity", "velocity_change_since_entry", "velocity_change_available"),
                ("acceleration", "acceleration_change_since_entry", "acceleration_change_available"),
                ("momentum", "momentum_score_change_since_entry", "momentum_change_available"),
            ):
                before, now = p["entry_context"][key], current[key]
                if before is not None and now is not None:
                    values[feature], values[available] = now - before, 1.0

        sign_hint = 1.0  # Signed raw momentum, independent of heuristic direction.
        r3 = self._number(row, "return_3m", "price_velocity", "velocity")
        r5 = self._number(row, "return_5m", "momentum_3")
        values["directional_momentum_score"] = sign_hint*((r3 or 0.0)+(r5 or 0.0))
        values["path_efficiency_value"] = self._number(row, "path_efficiency") or 0.0
        values["atr_expansion_value"] = self._number(row, "atr_ratio_20") or 0.0
        close = self._number(row, "close")
        e20 = self._number(row, "ema_20")
        uatr = self._number(row, "reference_underlying_atr", "atr_14")
        values["extension_atr_value"] = sign_hint*(close-e20)/uatr if close is not None and e20 is not None and uatr and uatr > 0 else 0.0
        values["price_oi_pressure_value"] = self._number(row, "reference_price_oi_pressure") or 0.0

        values["htf_primary_direction"] = self._number(row, "htf_primary_direction") or 0.0
        values["htf_secondary_direction"] = self._number(row, "htf_secondary_direction") or 0.0
        values["htf_primary_return"] = self._number(row, "htf_primary_return") or 0.0
        values["htf_secondary_return"] = self._number(row, "htf_secondary_return") or 0.0
        values["htf_primary_available"] = self._number(row, "htf_primary_available") or 0.0
        values["htf_secondary_available"] = self._number(row, "htf_secondary_available") or 0.0
        values["htf_alignment"] = (
            values["htf_primary_direction"]
            if values["htf_primary_available"]
            and (
                not self.htf_secondary_enabled
                or (
                    values["htf_secondary_available"]
                    and values["htf_primary_direction"] == values["htf_secondary_direction"]
                )
            )
            else 0.0
        )
        values["htf_fast_exit_signal"] = (
            float(self._htf_fast_exit_condition(row, self.position))
            if self.position else 0.0
        )

        scores = [self._quality(row, side) for side in ("CE", "PE")]
        values["tradeability_score"] = max(s[0] for s in scores)
        values["tradeability_score_available"] = float(any(s[1] for s in scores))

        if self.reference_features_enabled:
            values["reference_bull"] = self._number(row, "reference_bull") or 0.0
            values["reference_bear"] = self._number(row, "reference_bear") or 0.0
            values["reference_scores_available"] = float(self._number(row, "reference_bull") is not None
                                                         and self._number(row, "reference_bear") is not None)
            if self.position and self.position.get("structural_stop") is not None:
                sign = 1 if self.position["type"] == "CE" else -1
                values["reference_stop_distance_underlying"] = sign * (
                    row.close - self.position["structural_stop"]
                )

        # Switches mask inputs without changing tensor dimensions or trading rules.
        advisory_groups = {
            "directional_momentum_score": "momentum", "path_efficiency_value": "momentum",
            "atr_expansion_value": "volatility", "extension_atr_value": "trend",
            "price_oi_pressure_value": "oi", "remaining_session_fraction": "time",
            "htf_primary_direction": "trend", "htf_secondary_direction": "trend",
            "htf_alignment": "trend", "htf_fast_exit_signal": "trend",
            "htf_primary_return": "momentum",
            "htf_secondary_return": "momentum",
            "held_option_atr": "volatility", "atr_available": "volatility",
            "iv_change_since_entry": "greeks", "iv_change_available": "greeks",
            "delta_change_since_entry": "greeks", "delta_change_available": "greeks",
            "velocity_change_since_entry": "momentum", "velocity_change_available": "momentum",
            "acceleration_change_since_entry": "momentum", "acceleration_change_available": "momentum",
            "momentum_score_change_since_entry": "momentum", "momentum_change_available": "momentum",
        }
        for key, group in advisory_groups.items():
            if not getattr(self, f"features_{group}_enabled"):
                values[key] = 0.0
        # Aggregate scores include several groups: hide them when any contributor is off.
        if not all((self.features_momentum_enabled, self.features_trend_enabled,
                    self.features_volatility_enabled, self.features_volume_enabled,
                    self.features_oi_enabled)):
            for key in ("reference_bull", "reference_bear", "reference_scores_available",
                        "tradeability_score", "tradeability_score_available"):
                values[key] = 0.0
        for key in self._disabled_features:
            if key in values:
                values[key] = 0.0
                if key + "_available" in values:
                    values[key + "_available"] = 0.0
        if not self.missingness_features_enabled:
            for key in values:
                if key.endswith("_available"):
                    values[key] = 0.0
        obs = np.concatenate(
            [
                self._market_history(),
                np.array([values[k] for k in POSITION_FEATURES], dtype=np.float32),
            ]
        )
        return np.nan_to_num(obs, nan=0, posinf=0, neginf=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Execution economics
    # ------------------------------------------------------------------

    def _dynamic_slippage(self, quote):
        """Liquidity/volatility-aware execution slippage fraction.

        The UI's slippage_pct remains the base assumption.  The actual simulated
        slippage rises smoothly for low-volume/low-OI/high-ATR option candles.
        """
        base = float(self.slippage_pct)
        if not self.dynamic_slippage_enabled:
            return base
        if quote is None:
            return min(base * 2.0, 0.02)

        price = self._number(quote, "open", "close") or 0.0
        volume = max(self._number(quote, "volume") or 0.0, 0.0)
        oi = max(self._number(quote, "oi") or 0.0, 0.0)
        atr = max(self._number(quote, "atr") or 0.0, 0.0)

        # 1.0 for liquid contracts, approaching ~2 for very poor liquidity.
        liquidity_strength = np.log1p(volume) + 0.35 * np.log1p(oi)
        liquidity_mult = 1.0 + 1.0 / (1.0 + liquidity_strength / 5.0)

        atr_pct = atr / price if price > 0 else 0.0
        volatility_mult = 1.0 + min(1.5, max(0.0, atr_pct - 0.03) * 10.0)

        return float(min(base * liquidity_mult * volatility_mult, max(0.02, base * 6.0)))

    def _fill_price(self, raw, quote, is_buy):
        slip = self._dynamic_slippage(quote)
        return float(raw) * (1.0 + slip if is_buy else 1.0 - slip), slip

    def _transaction_costs(self, entry_turnover_price, exit_turnover_price):
        """Return round-trip transaction costs in rupees and premium points."""
        qty = float(self.trade_quantity)
        if not self.transaction_costs_enabled:
            return dict(brokerage=0.0, exchange=0.0, sebi=0.0, stt=0.0, stamp=0.0, gst=0.0, total_rupees=0.0, total_points=0.0)
        buy_turnover = max(0.0, entry_turnover_price) * qty
        sell_turnover = max(0.0, exit_turnover_price) * qty
        total_turnover = buy_turnover + sell_turnover

        brokerage = 2.0 * float(self.brokerage_per_order)
        exchange = total_turnover * self.nse_option_txn_rate
        sebi = total_turnover * self.sebi_rate
        stt = sell_turnover * self.option_stt_sell_rate
        stamp = buy_turnover * self.stamp_duty_buy_rate
        gst = self.gst_rate * (brokerage + exchange + sebi)

        rupees = brokerage + exchange + sebi + stt + stamp + gst
        return dict(
            brokerage=brokerage, exchange=exchange, sebi=sebi,
            stt=stt, stamp=stamp, gst=gst,
            total_rupees=rupees, total_points=rupees / qty,
        )

    def _liquidation_mark(self, p):
        """Net R if the held option were liquidated at its current known quote."""
        quote = self._get_option_row(p["symbol"], p["last_quote_time"])
        exit_fill, _ = self._fill_price(p["last_price"], quote, is_buy=False)
        costs = self._transaction_costs(p["entry_raw_price"], p["last_price"])
        net_points = exit_fill - p["entry_price"] - costs["total_points"]
        return net_points / p["initial_risk_points"]

    def _marked_r(self):
        value = self.episode_pnl
        if self.position:
            value += self._liquidation_mark(self.position)
        return float(value)

    def _daily_net_points(self):
        points = sum(t["option_pnl_points"] for t in self.trade_log)
        if self.position:
            points += self._liquidation_mark(self.position) * self.position["initial_risk_points"]
        return float(points)

    def _update_daily_limits(self):
        """Observe a completed bar; any forced exit fills at a later open.

        These are trigger levels, not guaranteed fills: gaps, stale quotes and
        slippage can overshoot either budget. No intrabar close-price lookahead.
        """
        net = (self._daily_net_points() if self.close_on_daily_limit_enabled
               else sum(t["option_pnl_points"] for t in self.trade_log))
        if not self.daily_limit_reason:
            if self.daily_loss_limit_enabled and net <= -self.daily_loss_limit_points:
                self.daily_limit_reason = "DAILY_LOSS_LIMIT"
            elif self.daily_profit_limit_enabled and net >= self.daily_profit_target_points:
                self.daily_limit_reason = "DAILY_PROFIT_TARGET"
        if self.daily_limit_reason and self.position and self.close_on_daily_limit_enabled:
            self.position["pending_exit"] = True
            self.position.setdefault("pending_exit_reason", self.daily_limit_reason)
            self.position["exit_requested_time"] = self.day_df.iloc[self.step_idx].timestamp

    # ------------------------------------------------------------------
    # Action masks / entries / exits
    # ------------------------------------------------------------------

    def action_masks(self):
        self._entry_blocks = {}
        mask = np.zeros(5, dtype=bool)
        if self._terminated:
            mask[WAIT] = True
            return mask

        if self.position:
            mask[HOLD] = True
            row = self.day_df.iloc[self.step_idx]
            htf_fast_exit = self._htf_fast_exit_condition(row, self.position)
            hold_ok = (
                (not self.min_hold_enabled)
                or self.position["bars_held"] >= self.min_hold_minutes
                or htf_fast_exit
            )
            mask[EXIT] = bool(
                self.learned_exit_enabled
                and hold_ok
                and not self.position["pending_exit"]
            )
            return mask

        mask[WAIT] = True
        if self.daily_limit_reason:
            self._entry_blocks = {"CE": [self.daily_limit_reason], "PE": [self.daily_limit_reason]}
            return mask
        row = self.day_df.iloc[self.step_idx]
        next_time = (row.timestamp + pd.Timedelta(minutes=1)).time()
        if self.square_off_enabled and next_time >= self.square_off_time:
            self._entry_blocks = {"CE": ["square_off"], "PE": ["square_off"]}
            return mask
        if self.first_entry_time_enabled and next_time < self.first_entry_time:
            self._entry_blocks = {"CE": ["before_entry_time"], "PE": ["before_entry_time"]}
            return mask
        if self.last_entry_time_enabled and next_time >= self.last_entry_time:
            self._entry_blocks = {"CE": ["after_entry_cutoff"], "PE": ["after_entry_cutoff"]}
            return mask
        if self.cooldown_remaining() > 0:
            self._entry_blocks = {"CE": ["cooldown"], "PE": ["cooldown"]}
            return mask

        for action, side, column in ((BUY_CE, "CE", "atm_ce_symbol"), (BUY_PE, "PE", "atm_pe_symbol")):
            reasons = []
            symbol = row.get("reference_symbol") if self.reference_strategy_enabled else row[column]
            quote = self._get_option_row(symbol, row.timestamp)
            price = self._number(quote, "close")
            atr = self._number(quote, "atr")
            valid = price is not None and price > 0
            if price is None or price <= 0:
                reasons.append("missing_option_quote")
            elif not valid:
                reasons.append("premium_below_risk_distance")

            if self.reference_strategy_enabled:
                valid = valid and row.reference_side == side
                if row.reference_side != side:
                    reasons.append("no_reference_signal" if pd.isna(row.reference_side) else "opposite_reference_signal")
                if self.structural_stop_enabled:
                    valid = valid and pd.notna(row.reference_stop)
                    if pd.isna(row.reference_stop):
                        reasons.append("missing_structural_stop")

            if self.htf_direction_enabled and self.htf_entry_gate_enabled:
                htf_ok, htf_details = self._htf_entry_allows(row, side)
                valid = valid and htf_ok
                if not htf_ok:
                    reasons.append("htf_direction")

            if self.tradeability_gate_enabled:
                score, available = self._quality(row, side)
                valid = valid and available and score >= self.tradeability_threshold
                if not available or score < self.tradeability_threshold:
                    reasons.append("entry_quality")

            filters_ok, details = self._entry_gate_details(row, side, quote)
            reasons.extend(name for name, passed in details.get("checks", {}).items() if not passed)
            valid = valid and filters_ok
            mask[action] = bool(valid)
            self._entry_blocks[side] = reasons
        return mask

    def set_action_probability(self, probability):
        self._action_probability = (
            float(probability) if probability is not None else None
        )

    def _enter_position(self, side):
        row = self.day_df.iloc[self.step_idx]
        nxt = self.day_df.iloc[self.step_idx + 1]
        if nxt.timestamp - row.timestamp != pd.Timedelta(minutes=1):
            return False
        if self.first_entry_time_enabled and nxt.timestamp.time() < self.first_entry_time:
            return False
        if self.last_entry_time_enabled and nxt.timestamp.time() >= self.last_entry_time:
            return False

        symbol = (
            row.reference_symbol
            if self.reference_strategy_enabled
            else row["atm_ce_symbol" if side == "CE" else "atm_pe_symbol"]
        )
        decision = self._get_option_row(symbol, row.timestamp)
        execution = self._get_option_row(symbol, nxt.timestamp)

        # Only t+1 OPEN is inspected for entry execution.
        raw = self._number(execution, "open")
        if raw is None or raw <= 0 or decision is None:
            return False

        # Opening fill cannot use this candle's future volume/OI/range.
        entry, entry_slippage = self._fill_price(raw, decision, is_buy=True)
        atr = self._number(decision, "atr")
        legacy_risk = min(self._risk_distance(entry, atr), 0.95 * entry)

        # Stop distance and target distance are intentionally independent.
        #
        # Example:
        #   fixed_option_target_points = 40
        #   fixed_option_stop_points   = 20
        #
        # gives a nominal +2R target before costs.  Reward normalisation uses
        # the actual executable premium risk, not the target distance.
        requested_stop_distance = (
            float(self.fixed_option_stop_points)
            if self.fixed_option_stop_enabled
            else float(legacy_risk)
        )
        if requested_stop_distance <= 0:
            return False

        # A long option premium cannot fall below zero.  For very cheap options,
        # clamp the stop to a small positive premium and use the actual distance
        # to that stop for R normalisation.
        stop_price = max(0.05, float(entry) - requested_stop_distance)
        risk = float(entry) - stop_price
        if risk <= 0:
            return False

        spot = self._number(nxt, "open")
        stop = self._number(row, "reference_stop") if self.reference_strategy_enabled else None
        if self.structural_stop_enabled and not self.reference_strategy_enabled:
            stop = structural_stop(self.day_df.iloc[:self.step_idx+1], side, row.close,
                                   self.reference_structural_lookback, self.reference_pivot_left,
                                   self.reference_pivot_right, self.reference_structural_buffer_fraction)
        if (self.reference_strategy_enabled or self.structural_stop_enabled or self.underlying_target_enabled) and spot is None:
            return False
        if self.structural_stop_enabled:
            if stop is None or (spot <= stop if side == "CE" else spot >= stop):
                return False

        htf_ok, htf_details = self._htf_entry_allows(row, side)
        if not htf_ok:
            return False

        filters_ok, filter_details = self._entry_gate_details(row, side, decision)
        if not filters_ok:
            return False
        filter_details["htf"] = htf_details

        target_points = self._dynamic_underlying_target(row)
        target_price = None
        target_mode = None
        if self.fixed_option_target_enabled:
            # User objective: selected option premium itself must rise by a fixed
            # number of points from the executed entry fill, regardless of strike/premium.
            target_price = entry + float(self.fixed_option_target_points)
            target_mode = "FIXED_OPTION_POINTS"
        elif self.risk_reward_target_enabled:
            provisional_raw = entry + self.risk_reward_target_r * risk
            estimated_cost = self._transaction_costs(raw, provisional_raw)["total_points"]
            expected_sell_slip = self._dynamic_slippage(decision)
            target_price = (entry + self.risk_reward_target_r*risk + estimated_cost) / max(1.0-expected_sell_slip, 1e-6)
            target_mode = "R_MULTIPLE"

        self.daily_entries += 1
        penalty = self.entry_penalty_r if self.entry_penalty_enabled else 0.0
        if self.overtrading_penalty_enabled and self.daily_entries > self.free_trades_per_day:
            penalty += self.extra_trade_penalty_r
        self._entry_penalty += penalty

        self.position = dict(
            type=side, symbol=str(symbol),
            entry_raw_price=float(raw), entry_price=entry,
            entry_slippage_fraction=entry_slippage,
            entry_time=nxt.timestamp, initial_risk_points=risk,
            stop_price=stop_price, last_price=float(raw),
            last_quote_time=nxt.timestamp, last_atr=atr or 0.0,
            atr_valid=atr is not None and atr > 0,
            high_since_entry=float(raw), low_since_entry=float(raw),
            bars_held=0, minutes_held=0.0, pending_exit=False,
            exit_requested_time=None,
            entry_context=self._context(row, decision),
            entry_action_probability=self._action_probability,
            exit_action_probability=None, entry_penalty_r=penalty,
            shaping_r=0.0, shaping_used=0.0,
            agent_exit_penalty_paid=0.0,
            regime=self._market_regime(row),
            underlying_target_points=target_points,
            target_price=target_price, target_mode=target_mode,
            entry_filter_details=filter_details,
        )

        self.position.update(
                spot_entry=spot, structural_stop=stop,
                entry_ignition_bull=self._number(row, "reference_bull"),
                entry_ignition_bear=self._number(row, "reference_bear"),
                entry_price_oi_pressure=self._number(row, "reference_price_oi_pressure"),
                entry_htf_primary_direction=self._number(row, "htf_primary_direction"),
                entry_htf_secondary_direction=self._number(row, "htf_secondary_direction"),
                entry_htf_primary_return=self._number(row, "htf_primary_return"),
                entry_htf_secondary_return=self._number(row, "htf_secondary_return"),
            )
        return True

    def _reference_exit_reason(self):
        row, p = self.day_df.iloc[self.step_idx], self.position
        sign = 1 if p["type"] == "CE" else -1

        if self.structural_stop_enabled and p.get("structural_stop") is not None:
            if (row.low <= p["structural_stop"] if sign == 1 else row.high >= p["structural_stop"]):
                return "STRUCTURAL_STOP"

        htf_reason = self._htf_exit_reason(row, p)
        if htf_reason:
            return htf_reason

        if self.underlying_target_enabled and p.get("spot_entry") is not None:
            if sign * (row.close - p["spot_entry"]) >= p["underlying_target_points"]:
                return "UNDERLYING_TARGET"

        if self.momentum_decay_exit_enabled:
            own, other = ((row.reference_bull, row.reference_bear) if sign == 1
                          else (row.reference_bear, row.reference_bull))
            if np.isfinite([own, other]).all() and own < self.reference_exit_threshold and other > own:
                return "MOMENTUM_DECAY"
        return None

    def _close_position(self, raw, reason, timestamp, stale=False, detail="", phase="OPEN"):
        p = self.position
        # Use the preceding completed quote for OPEN/intrabar execution slippage.
        quote = self._get_option_row(p["symbol"], timestamp if phase == "CLOSE"
                                     else timestamp - pd.Timedelta(minutes=1))
        exit_price, exit_slippage = self._fill_price(raw, quote, is_buy=False)

        costs = self._transaction_costs(p["entry_raw_price"], float(raw))
        gross_points = exit_price - p["entry_price"]
        net_points = gross_points - costs["total_points"]
        risk = p["initial_risk_points"]

        high = max(p["high_since_entry"], raw)
        low = min(p["low_since_entry"], raw)
        mfe, mae = high - p["entry_price"], low - p["entry_price"]
        holding = max(
            0.0,
            (timestamp - p["entry_time"]).total_seconds() / 60
            + (1 if phase == "CLOSE" else 0),
        )

        terminal_shaping = 0.0
        if self.exit_reason_shaping_enabled and not stale:
            if reason in {"OPTION_TARGET", "OPTION_TARGET_GAP"}:
                terminal_shaping += self.option_target_bonus_r
            elif reason == "STRUCTURAL_STOP":
                terminal_shaping -= self.structural_stop_penalty_r
            elif reason in {"EMERGENCY_STOP", "EMERGENCY_GAP"}:
                terminal_shaping -= self.emergency_stop_penalty_r
            elif reason == "TIME_LIMIT" and net_points < 0:
                terminal_shaping -= self.time_limit_loss_penalty_r
            elif reason in {"MOMENTUM_DECAY", "HTF_DIRECTION_REVERSAL", "HTF_SECONDARY_REVERSAL"} and net_points < 0:
                terminal_shaping -= self.momentum_decay_loss_penalty_r
        elif (net_points > 0 and net_points >= self.terminal_win_min_points
                and self.terminal_win_shaping_enabled and not stale):
            terminal_shaping += self.terminal_win_bonus_r
        elif net_points < 0 and self.terminal_loss_shaping_enabled:
            terminal_shaping -= self.terminal_loss_penalty_r
        shaping_capacity = max(0.0, self.max_shaping_r_per_trade - p["shaping_used"])
        loss_aversion_penalty = 0.0
        if net_points < 0 and self.loss_aversion_enabled:
            loss_aversion_penalty = min(
                (self.loss_aversion_multiplier - 1.0) * abs(net_points / risk),
                shaping_capacity,
            )
            terminal_shaping -= loss_aversion_penalty
        terminal_shaping = float(np.sign(terminal_shaping) * min(
            abs(terminal_shaping),
            shaping_capacity,
        ))

        record = dict(
            date=str(self.current_day), side=p["type"], type=p["type"],
            symbol=p["symbol"], entry_time=p["entry_time"], exit_time=timestamp,
            entry_price=p["entry_price"], exit_price=exit_price,
            entry_raw_price=p["entry_raw_price"], exit_raw_price=float(raw),
            entry_slippage_fraction=p["entry_slippage_fraction"],
            exit_slippage_fraction=exit_slippage,
            holding_minutes=holding, exit_reason=reason, exit_detail=detail,
            exit_phase=phase, initial_risk_points=risk, stop_points=risk,
            option_pnl_points=net_points, gross_option_pnl_points=gross_points,
            transaction_cost_points=costs["total_points"],
            transaction_cost_rupees=costs["total_rupees"],
            brokerage_rupees=costs["brokerage"], stt_rupees=costs["stt"],
            exchange_charges_rupees=costs["exchange"], gst_rupees=costs["gst"],
            stamp_duty_rupees=costs["stamp"], sebi_charges_rupees=costs["sebi"],
            option_return_pct=100 * net_points / p["entry_price"],
            R_return=net_points / risk, MFE_option_points=mfe,
            MAE_option_points=mae, MFE_R=mfe / risk, MAE_R=mae / risk,
            bars_held=p["bars_held"],
            reached_40_points=mfe >= self.min_desired_move_points,
            desired_move_points=self.min_desired_move_points,
            entry_action_probability=p["entry_action_probability"],
            exit_action_probability=p["exit_action_probability"],
            model_id=self.model_id, stale_exit=stale,
            quote_time=p["last_quote_time"],
            quote_age_minutes=max(
                0.0, (timestamp - p["last_quote_time"]).total_seconds() / 60
            ),
            entry_penalty_r=p["entry_penalty_r"], shaping_r=p["shaping_r"],
            reward=net_points / risk - p["entry_penalty_r"] + p["shaping_r"] + terminal_shaping - p.get("agent_exit_penalty_paid", 0.0),
            agent_exit_penalty_r=p.get("agent_exit_penalty_paid", 0.0),
            terminal_shaping_r=terminal_shaping,
            loss_aversion_penalty_r=loss_aversion_penalty,
            trade_quantity=self.trade_quantity, regime=p["regime"],
            underlying_target_points=p["underlying_target_points"],
        )
        record.update(
            reference_strategy_enabled=self.reference_strategy_enabled,
            underlying_entry=p.get("spot_entry"),
            structural_stop=p.get("structural_stop"),
            entry_ignition_bull=p.get("entry_ignition_bull"),
            entry_ignition_bear=p.get("entry_ignition_bear"),
            entry_price_oi_pressure=p.get("entry_price_oi_pressure"),
            target_price=p.get("target_price"),
            target_mode=p.get("target_mode"),
            fixed_option_target_points=(
                self.fixed_option_target_points
                if self.fixed_option_target_enabled else None
            ),
            fixed_option_stop_points=(
                self.fixed_option_stop_points
                if self.fixed_option_stop_enabled else None
            ),
            fixed_option_stop_enabled=self.fixed_option_stop_enabled,
            entry_filter_details=p.get("entry_filter_details"),
            entry_htf_primary_direction=p.get("entry_htf_primary_direction"),
            entry_htf_secondary_direction=p.get("entry_htf_secondary_direction"),
            entry_htf_primary_return=p.get("entry_htf_primary_return"),
            entry_htf_secondary_return=p.get("entry_htf_secondary_return"),
            exit_htf_primary_direction=self._number(
                self.day_df.iloc[min(self.step_idx, len(self.day_df)-1)],
                "htf_primary_direction"
            ),
            exit_htf_secondary_direction=self._number(
                self.day_df.iloc[min(self.step_idx, len(self.day_df)-1)],
                "htf_secondary_direction"
            ),
        )

        self.trade_log.append(record)
        self.episode_pnl += record["R_return"]
        self._terminal_shaping_pending += terminal_shaping
        self.last_exit_time = timestamp + (
            pd.Timedelta(minutes=1) if phase == "CLOSE" else pd.Timedelta(0)
        )
        self.last_exit_cooldown_minutes = self._cooldown_for_reason(reason, net_points)
        self.last_exit_step = self.step_idx
        self.position = None
        if self.state:
            self.state.add_trade(record)

    def _manage_position(self, final=False):
        if self.position is None:
            return

        row, p = self.day_df.iloc[self.step_idx], self.position
        deadlines = []
        if self.square_off_enabled:
            deadlines.append(row.timestamp.normalize() + pd.Timedelta(
                hours=self.square_off_time.hour, minutes=self.square_off_time.minute))
        if self.max_hold_enabled:
            deadlines.append(p["entry_time"] + pd.Timedelta(minutes=self.max_hold_minutes))
        deadline = min(deadlines) if deadlines else None

        if deadline is not None and row.timestamp >= deadline:
            quote = self._get_option_row(p["symbol"], deadline)
            raw = self._number(quote, "open")
            stale = raw is None or raw <= 0
            if not stale:
                p["last_price"], p["last_quote_time"] = raw, deadline
            if stale:
                reason = "STALE_OPTION_DATA"
            elif self.square_off_enabled and deadline.time() == self.square_off_time:
                reason = "EOD"
            else:
                reason = "TIME_LIMIT"
            if not stale and self.emergency_stop_enabled and raw <= p["stop_price"]:
                reason = "EMERGENCY_GAP"
            if not stale and self._target_enabled() and p.get("target_price") is not None and raw >= p["target_price"]:
                reason = ("OPTION_TARGET_GAP" if p.get("target_mode") == "FIXED_OPTION_POINTS"
                          else "R_TARGET_GAP")
            self._close_position(p["last_price"], reason, deadline, stale,
                                 "SQUARE_OFF" if reason == "EOD" else "MAX_HOLD")
            return

        candle = self._get_option_row(p["symbol"], row.timestamp)
        raw = self._number(candle, "open")
        if raw is not None and raw > 0:
            p["last_quote_time"] = row.timestamp
            if self.emergency_stop_enabled and raw <= p["stop_price"]:
                self._close_position(raw, "EMERGENCY_GAP", row.timestamp)
                return
            if self._target_enabled() and p.get("target_price") is not None and raw >= p["target_price"]:
                reason = ("OPTION_TARGET_GAP" if p.get("target_mode") == "FIXED_OPTION_POINTS"
                          else "R_TARGET_GAP")
                self._close_position(raw, reason, row.timestamp)
                return
            if p["pending_exit"]:
                self._close_position(raw, p.get("pending_exit_reason", "AGENT_EXIT"), row.timestamp)
                return

        low, high, close = (self._number(candle, k) for k in ("low", "high", "close"))
        complete = all(v is not None and v > 0 for v in (low, high, close))

        # Conservative intrabar ordering: if stop and target are both touched, stop wins.
        if self.emergency_stop_enabled and low is not None and low > 0 and low <= p["stop_price"]:
            self._close_position(p["stop_price"], "EMERGENCY_STOP", row.timestamp, phase="INTRABAR")
            return
        if (self._target_enabled() and p.get("target_price") is not None
                and high is not None and high >= p["target_price"]):
            reason = ("OPTION_TARGET" if p.get("target_mode") == "FIXED_OPTION_POINTS"
                      else "R_TARGET")
            self._close_position(p["target_price"], reason, row.timestamp, phase="INTRABAR")
            return

        p["minutes_held"] = (row.timestamp - p["entry_time"]).total_seconds() / 60 + 1
        if complete:
            p["bars_held"] += 1
            p["last_price"], p["last_quote_time"] = close, row.timestamp
            p["high_since_entry"] = max(p["high_since_entry"], high)
            p["low_since_entry"] = min(p["low_since_entry"], low)
            atr = self._number(candle, "atr")
            p["atr_valid"] = atr is not None and atr > 0
            if p["atr_valid"]:
                p["last_atr"] = atr

        if final and self.position is not None:
            self._close_position(p["last_price"], "EOD" if complete else "STALE_OPTION_DATA",
                                 row.timestamp, stale=not complete, detail="DATA_END", phase="CLOSE")

    # ------------------------------------------------------------------
    # Reward shaping / step
    # ------------------------------------------------------------------

    def _shape_hold(self):
        p = self.position
        if p is None or p["pending_exit"]:
            return 0.0

        row = self.day_df.iloc[self.step_idx]
        if p["last_quote_time"] != row.timestamp:
            return 0.0

        current_r = self._liquidation_mark(p)
        # MFE stays gross because high/low are not executable bid quotes; shaping
        # is capped and therefore remains a weak auxiliary signal only.
        mfe_r = (
            p["high_since_entry"] - p["entry_price"]
        ) / p["initial_risk_points"]

        amount = 0.0
        if self.giveback_penalty_enabled and mfe_r >= self.giveback_min_mfe_r and current_r <= mfe_r * (1 - self.giveback_fraction):
            amount = -self.giveback_penalty_r
        elif self.hold_shaping_enabled and current_r > 0 and self._momentum_favorable(row, p["type"]):
            amount = self.hold_bonus_r

        amount = float(
            np.sign(amount)
            * min(
                abs(amount),
                max(0.0, self.max_shaping_r_per_trade - p["shaping_used"]),
            )
        )
        p["shaping_used"] += abs(amount)
        p["shaping_r"] += amount
        return amount

    def step(self, action):
        if self._terminated:
            raise RuntimeError("Call reset after episode end")

        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Action outside Discrete(5)")

        was_holding = self.position is not None
        masks = self.action_masks()
        invalid = not bool(masks[action])
        if invalid:
            action = HOLD if was_holding else WAIT

        semantic = ACTION_NAMES[action]
        self.action_counts[semantic] += 1
        self._entry_penalty = 0.0

        voluntary_wait = bool(
            not was_holding
            and action == WAIT
            and not invalid
            and (masks[BUY_CE] or masks[BUY_PE])
        )
        voluntary_wait_penalty = 0.0
        if self.voluntary_wait_penalty_enabled and voluntary_wait:
            remaining_cap = max(
                0.0,
                float(self.max_voluntary_wait_penalty_r_per_day)
                - float(self.daily_voluntary_wait_penalty_r),
            )
            voluntary_wait_penalty = min(
                float(self.voluntary_wait_penalty_r),
                remaining_cap,
            )
            self.daily_voluntary_wait_penalty_r += voluntary_wait_penalty

        rejected = False

        if not was_holding and action in (BUY_CE, BUY_PE):
            rejected = not self._enter_position("CE" if action == BUY_CE else "PE")
        elif was_holding and action == EXIT and self.learned_exit_enabled:
            self.position["pending_exit"] = True
            self.position["exit_requested_time"] = self.day_df.iloc[
                self.step_idx
            ].timestamp
            self.position["exit_action_probability"] = self._action_probability

        if not was_holding and self.state and hasattr(self.state, "record_entry_decision"):
            self.state.record_entry_decision(
                bool(masks[BUY_CE] or masks[BUY_PE]), semantic,
                action in (BUY_CE, BUY_PE) and not rejected, self._entry_blocks,
            )

        # Hard/reference exits remain safety/discipline overlays.
        if (
            was_holding and self.position is not None
            and not self.position.get("pending_exit_reason")
        ):
            reason = self._reference_exit_reason()
            if reason:
                self.position["pending_exit"] = True
                self.position["pending_exit_reason"] = reason
                self.position["exit_requested_time"] = self.day_df.iloc[
                    self.step_idx
                ].timestamp

        if was_holding and self.position is not None and self.position["pending_exit"]:
            self.position.setdefault("pending_exit_reason", "AGENT_EXIT")

        agent_exit_penalty = (self.agent_exit_penalty_r if (was_holding and action == EXIT and self.learned_exit_enabled and self.agent_exit_penalty_enabled) else 0.0)
        if self.position is not None:
            self.position["agent_exit_penalty_paid"] = self.position.get("agent_exit_penalty_paid", 0.0) + agent_exit_penalty
        self._action_probability = None
        self.step_idx += 1
        self._terminated = self.step_idx == len(self.day_df) - 1
        self._manage_position(final=self._terminated)
        self._update_daily_limits()
        if self.end_inactive_episode_enabled and self._no_future_entries():
            self._terminated = True

        shaping = (
            self._shape_hold()
            if was_holding and action == HOLD and self.position is not None
            else 0.0
        )

        marked = self._marked_r()
        pnl_delta = marked - self._reward_value
        self._reward_value = marked
        terminal_shaping = self._terminal_shaping_pending
        self._terminal_shaping_pending = 0.0
        unscaled = (
            pnl_delta
            - self._entry_penalty
            - agent_exit_penalty
            - voluntary_wait_penalty
            + shaping
            + terminal_shaping
        )
        self.episode_reward += unscaled

        obs = (
            np.zeros(self.observation_space.shape, np.float32)
            if self._terminated else self._get_observation()
        )
        info = dict(
            date=str(self.current_day), episode_pnl=self.episode_pnl,
            episode_reward=self.episode_reward, trades=len(self.trade_log),
            action_name=semantic, invalid_action=invalid,
            entry_available=bool(not was_holding and (masks[BUY_CE] or masks[BUY_PE])),
            voluntary_wait=voluntary_wait,
            voluntary_wait_penalty_r=voluntary_wait_penalty,
            daily_voluntary_wait_penalty_r=self.daily_voluntary_wait_penalty_r,
            entry_opened=bool(not was_holding and action in (BUY_CE, BUY_PE) and not rejected),
            episode_start_time=self.episode_start_time,
            partial_session=self.episode_partial_session,
            entry_rejected=rejected, pnl_reward_r=pnl_delta,
            entry_penalty_r=self._entry_penalty,
            shaping_reward_r=shaping,
            terminal_shaping_r=terminal_shaping,
            agent_exit_penalty_r=agent_exit_penalty,
            daily_net_points=self._daily_net_points(),
            daily_limit_reason=self.daily_limit_reason,
            cooldown_remaining=self.cooldown_remaining(),
            bars_since_exit=self.bars_since_exit,
            candidate_diagnostics=self._candidate_stats if self._terminated else None,
        )
        if self.state:
            self.state.record_action(semantic, invalid, rejected)
        return obs, float(unscaled * self.reward_scale), self._terminated, False, info

    def simulate_reference_candidate(self, bar_index, check_cancel=None):
        """Independent deterministic label using the same execution and exit code."""
        if bar_index >= len(self.day_df) - 1:
            return None
        saved = self.__dict__.copy()
        try:
            self.state = None
            self.step_idx = int(bar_index)
            self.position = None
            self.trade_log = []
            self.episode_pnl = self.episode_reward = self._reward_value = 0.0
            self._terminal_shaping_pending = self._entry_penalty = 0.0
            self.daily_entries = 0
            self.daily_limit_reason = ""
            self.last_exit_time = self.last_exit_step = None
            self.action_counts = dict.fromkeys(ACTION_NAMES, 0)
            self._terminated = False
            side = self.day_df.iloc[self.step_idx].get("reference_side")
            if side not in ("CE", "PE"):
                return None
            self.step(BUY_CE if side == "CE" else BUY_PE)
            while self.position is not None and not self._terminated:
                if check_cancel:
                    check_cancel()
                self.step(HOLD)
            if not self.trade_log:
                return None
            trade = dict(self.trade_log[-1])
            # Decision at close t can execute at open t+1. Never reuse the exit bar.
            ready = max(self.day_df.iloc[self.step_idx].timestamp,
                        self.last_exit_time + pd.Timedelta(minutes=(
                            self.last_exit_cooldown_minutes if self.cooldown_enabled else 0))
                        - pd.Timedelta(minutes=1))
            trade["next_decision_time"] = ready
            return trade
        finally:
            self.__dict__.clear()
            self.__dict__.update(saved)

# ---------------------------------------------------------------------------
# Compatibility candidate environment
# ---------------------------------------------------------------------------
# The web trainer still imports train_candidate_model/build_candidate_dataset at
# startup.  Those modules import CandidateBankNiftyEnv even when the user runs
# only the legacy/minute MaskablePPO trainer.  Keep this lightweight event-level
# environment available so the web application can start without forcing the
# candidate pipeline to be used.
SKIP, TAKE = 0, 1
CANDIDATE_ACTION_NAMES = ("SKIP", "TAKE")


class CandidateBankNiftyEnv(gym.Env):
    """Compatibility SKIP/TAKE environment for the optional candidate pipeline.

    The V4 fixed-40 legacy PPO trainer does not use this class.  It is retained
    because train_candidate_model.py imports it during banknifty_rl_web.py startup.
    Future outcome columns are never part of the observation tensor.
    """
    metadata = {"render_modes": []}

    def __init__(self, candidates, split="train", random_day=True):
        super().__init__()
        from reference_strategy import CANDIDATE_FEATURES, candidate_observations

        self._candidate_observations = candidate_observations
        self.feature_columns = list(CANDIDATE_FEATURES)
        self.frame = candidates.loc[candidates.split == split].sort_values("timestamp").copy()
        if self.frame.empty:
            raise ValueError(f"No reference candidates in {split}")
        self.frame["timestamp"] = pd.to_datetime(self.frame.timestamp)
        self.frame["date"] = self.frame.timestamp.dt.date
        self.days = sorted(self.frame.date.unique())
        self.random_day = bool(random_day and split == "train")
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(self.feature_columns),), dtype=np.float32
        )
        self._day_index = 0
        self._terminated = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if options and "day_index" in options:
            index = int(options["day_index"])
        elif self.random_day:
            index = int(self.np_random.integers(len(self.days)))
        else:
            if seed is not None:
                self._day_index = 0
            index = self._day_index
            self._day_index = (index + 1) % len(self.days)

        self.day = self.frame.loc[self.frame.date == self.days[index]].reset_index(drop=True)
        self.features = self._candidate_observations(self.day)
        self.index = 0
        self._terminated = False
        self.episode_reward = 0.0
        self.counts = dict(
            reference_candidates=len(self.day),
            taken_candidates=0,
            skipped_candidates=0,
            unavailable_candidates=0,
            rejected_candidates=0,
        )
        self.trades = []
        return self.features[0].copy(), {"date": str(self.days[index]), **self.counts}

    def step(self, action):
        if self._terminated:
            raise RuntimeError("Reset before stepping the candidate environment")
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError("Candidate actions are SKIP=0 and TAKE=1")

        row = self.day.iloc[self.index]
        info = dict(
            candidate_id=row.candidate_id,
            action_name=CANDIDATE_ACTION_NAMES[action],
            timestamp=str(row.timestamp),
            side=row.side,
            trade=None,
        )
        reward = 0.0
        self.index += 1

        if action == SKIP:
            self.counts["skipped_candidates"] += 1
        else:
            self.counts["taken_candidates"] += 1
            if bool(row.filled):
                reward = float(row.realized_R)
                trade = json.loads(row.trade_json)
                self.trades.append(trade)
                info["trade"] = trade
                ready = pd.Timestamp(row.next_decision_time)
                while self.index < len(self.day) and self.day.iloc[self.index].timestamp < ready:
                    self.index += 1
                    self.counts["unavailable_candidates"] += 1
            else:
                self.counts["rejected_candidates"] += 1

        self.episode_reward += reward
        self._terminated = self.index >= len(self.day)
        info.update(self.counts, episode_reward=self.episode_reward)
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if self._terminated
            else self.features[self.index].copy()
        )
        return observation, reward, self._terminated, False, info
