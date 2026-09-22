"""Causal BANKNIFTY momentum reference strategy.

The reference layer is intentionally heuristic: it proposes candidate direction,
contract and structural risk context; MaskablePPO decides whether to participate
and how to manage the position.  All signal inputs are completed candles only.

Important:
- Scores are ignition strengths, not calibrated probabilities.
- Option OI is interpreted jointly with option-price movement.
- Contract selection uses only the completed decision snapshot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_EPS = 1e-9

# Explicit allowlist: future trade outcomes must never enter either learner.
CANDIDATE_VERSION = "reference_candidates_v1"
CANDIDATE_VALUES = (
    "side_ce", "reference_bull", "reference_bear", "score_gap",
    "stop_distance_atr", "return_3m", "return_5m", "path_efficiency",
    "atr_ratio", "extension", "price_oi_pressure", "delta", "iv",
    "log_volume", "log_oi", "option_price", "option_atr", "minutes_from_open",
    "trend_up", "trend_down", "low_vol", "high_vol", "expiry_day",
)
CANDIDATE_FEATURES = CANDIDATE_VALUES + tuple(f"{k}_available" for k in CANDIDATE_VALUES)


def candidate_observations(frame):
    """Decision-time features only, with fixed missingness indicators."""
    values = frame.reindex(columns=CANDIDATE_VALUES).apply(pd.to_numeric, errors="coerce")
    values = values.to_numpy(dtype=np.float64)
    available = np.isfinite(values)
    values = np.where(available, np.clip(values, -1e10, 1e10), 0.0)
    return np.concatenate([values, available.astype(float)], axis=1).astype(np.float32)


def reference_candidate_table(bars, quote_at, config, regime_at):
    """Emit reference events from causal prepared bars, without TAKE/SKIP logic.

    `bars` carries the reference engine's direction, symbol and structural stop.
    No future quote, outcome, extra entry-quality gate or portfolio state is used.
    Each qualifying timestamp is an event, including successive same-side signals.
    """
    records = []

    def number(source, key):
        value = source.get(key) if isinstance(source, (dict, pd.Series)) else getattr(source, key, None)
        return float(value) if value is not None and np.isfinite(value) else np.nan

    for index, row in bars.iterrows():
        side, symbol = row.get("reference_side"), row.get("reference_symbol")
        if side not in ("CE", "PE") or pd.isna(symbol):
            continue
        # Use the clock, never the presence/price of a future candle, to define events.
        next_clock = (row.timestamp + pd.Timedelta(minutes=1)).strftime("%H:%M")
        if config["first_entry_time_enabled"] and next_clock < config["first_entry_time"]:
            continue
        if config["last_entry_time_enabled"] and next_clock >= config["last_entry_time"]:
            continue
        if config["square_off_enabled"] and next_clock >= config["square_off_time"]:
            continue
        quote = quote_at(symbol, row.timestamp)
        price = number(quote, "close")
        if not np.isfinite(price) or price <= 0:
            continue
        stop = number(row, "reference_stop")
        if config["structural_stop_enabled"] and not np.isfinite(stop):
            continue
        sign = 1 if side == "CE" else -1
        atr = number(row, "reference_underlying_atr")
        close, ema = number(row, "close"), number(row, "ema_20")
        volume, oi = number(quote, "volume"), number(quote, "oi")
        regime = regime_at(row)
        records.append(dict(
            candidate_id=f"{row.timestamp.isoformat()}_{side}_{symbol}",
            bar_index=int(index), timestamp=row.timestamp, side=side, symbol=str(symbol),
            reference_bull=number(row, "reference_bull"),
            reference_bear=number(row, "reference_bear"),
            score_gap=abs(number(row, "reference_bull")-number(row, "reference_bear")),
            structural_stop=stop, side_ce=float(side == "CE"),
            stop_distance_atr=sign*(close-stop)/atr if atr > 0 else np.nan,
            return_3m=number(row, "return_3m"), return_5m=number(row, "return_5m"),
            path_efficiency=number(row, "path_efficiency"),
            atr_ratio=number(row, "atr_ratio_20"),
            extension=sign*(close-ema)/atr if atr > 0 else np.nan,
            price_oi_pressure=number(row, "reference_price_oi_pressure"),
            delta=number(quote, "delta"), iv=number(quote, "iv"), volume=volume, oi=oi,
            log_volume=np.log1p(max(volume, 0)) if np.isfinite(volume) else np.nan,
            log_oi=np.log1p(max(oi, 0)) if np.isfinite(oi) else np.nan,
            option_price=price, option_atr=number(quote, "atr"), regime=regime,
            minutes_from_open=float(row.timestamp.hour*60 + row.timestamp.minute - 555),
            trend_up=float("TREND_UP" in regime), trend_down=float("TREND_DOWN" in regime),
            low_vol=float("LOW_VOL" in regime), high_vol=float("HIGH_VOL" in regime),
            expiry_day=float(regime.endswith("|EXPIRY")),
        ))
    return pd.DataFrame(records)


def _safe_tanh(series, scale):
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.tanh(values / max(float(scale), _EPS))


def option_price_oi_pressure(options, bars, strike_window=5, spacing=100.0):
    """Return causal underlying-direction pressure from option price x OI states.

    Positive values favour an underlying rise; negative values favour a fall.

    For each option contract:
      price up + OI up   -> long build-up
      price down + OI up -> short build-up
      price up + OI down -> short covering
      price down + OI down -> long unwinding

    CE and PE contributions are mirrored into underlying direction.  Near-ATM
    strikes receive greater weight.  Nothing from t+1 is used at timestamp t.
    """
    needed = {"timestamp", "groww_symbol", "strike", "option_type", "oi", "close"}
    if options.empty or not needed.issubset(options.columns):
        return pd.Series(0.0, index=pd.Index(bars["timestamp"], name="timestamp"))

    spot = bars[["timestamp", "close"]].rename(columns={"close": "spot"})
    o = options[list(needed)].merge(spot, on="timestamp", how="inner")
    if o.empty:
        return pd.Series(0.0, index=pd.Index(bars["timestamp"], name="timestamp"))

    o = o.sort_values(["groww_symbol", "timestamp"]).copy()
    o["m"] = (o["strike"] - o["spot"]) / float(spacing)
    o = o.loc[o["m"].abs() <= float(strike_window)].copy()
    if o.empty:
        return pd.Series(0.0, index=pd.Index(bars["timestamp"], name="timestamp"))

    grp = o.groupby("groww_symbol", observed=True, sort=False)
    previous_price = grp["close"].shift(1)
    previous_oi = grp["oi"].shift(1)

    o["price_ret"] = (
        (o["close"] - previous_price) / (previous_price.abs() + 1.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    o["oi_ret"] = (
        (o["oi"] - previous_oi) / (previous_oi.abs() + 1.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Saturating transforms stop isolated bad ticks from dominating the score.
    p = _safe_tanh(o["price_ret"], 0.02)
    q = _safe_tanh(o["oi_ret"], 0.10)

    p_up, p_dn = np.maximum(p, 0.0), np.maximum(-p, 0.0)
    q_up, q_dn = np.maximum(q, 0.0), np.maximum(-q, 0.0)

    long_build = p_up * q_up
    short_build = p_dn * q_up
    short_cover = p_up * q_dn
    long_unwind = p_dn * q_dn

    # Option-direction pressure.  Long build and short build carry the strongest
    # evidence; covering/unwinding are useful but less persistent.
    option_pressure = (
        1.00 * long_build
        - 1.00 * short_build
        + 0.60 * short_cover
        - 0.40 * long_unwind
    )

    # Rising CE demand is bullish underlying; rising PE demand is bearish.
    underlying_pressure = np.where(o["option_type"].astype(str).eq("CE"),
                                   option_pressure, -option_pressure)

    o["weight"] = np.exp(-0.35 * o["m"].abs())
    o["weighted_pressure"] = underlying_pressure * o["weight"]

    agg = o.groupby("timestamp", observed=True)[["weighted_pressure", "weight"]].sum()
    pressure = agg["weighted_pressure"] / (agg["weight"] + _EPS)
    return pressure.reindex(pd.Index(bars["timestamp"], name="timestamp")).fillna(0.0)


def ignition_features(bars, options, window=20, strike_window=5, spacing=100.0):
    """Compute bullish/bearish ignition scores from completed data only."""
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(bars.columns):
        raise ValueError(f"Reference strategy needs underlying columns: {sorted(required)}")

    x = bars.sort_values("timestamp").reset_index(drop=True)
    h, l, c = (x[k].to_numpy(float) for k in ("high", "low", "close"))
    v = x.get("volume", pd.Series(0.0, index=x.index)).fillna(0.0).to_numpy(float)
    dc = np.r_[0.0, np.diff(c)]
    bull, bear = np.full(len(x), np.nan), np.full(len(x), np.nan)

    price_oi = option_price_oi_pressure(options, x, strike_window, spacing)
    price_oi_values = price_oi.to_numpy(float)

    for i in range(window, len(x)):
        sl = slice(i - window + 1, i + 1)
        ranges, local_c, local_dc = h[sl] - l[sl], c[sl], dc[sl]

        finite = np.r_[h[i-window:i+1], l[i-window:i+1], local_c]
        if not np.isfinite(finite).all():
            continue

        resistance = h[i-window:i].max()
        support = l[i-window:i].min()

        half = max(4, window // 2)
        old = ranges[:half].mean()
        new = ranges[-half:].mean()
        compression = np.clip((old - new) / (old + _EPS), -1.0, 1.0)

        path_efficiency = (
            abs(local_c[-1] - local_c[0]) /
            (np.abs(np.diff(local_c)).sum() + _EPS)
        )
        scale = np.median(ranges) + _EPS

        up = np.maximum(local_dc, 0.0).sum()
        down = np.maximum(-local_dc, 0.0).sum()
        asym = (up - down) / (up + down + _EPS)

        bullish_pressure = np.clip(
            0.45 * np.mean(resistance - local_c <= scale)
            + 0.35 * max(asym, 0.0)
            + 0.20 * max(compression, 0.0),
            0.0, 1.0
        )
        bearish_pressure = np.clip(
            0.45 * np.mean(local_c - support <= scale)
            + 0.35 * max(-asym, 0.0)
            + 0.20 * max(compression, 0.0),
            0.0, 1.0
        )

        posv = v[sl][local_dc > 0].sum()
        negv = v[sl][local_dc < 0].sum()
        volume_asym = (posv - negv) / (posv + negv + _EPS)

        scores = []
        for direction, pressure, penetration, retention in (
            (1, bullish_pressure, max(h[i] - resistance, 0.0), max(c[i] - resistance, 0.0)),
            (-1, bearish_pressure, max(support - l[i], 0.0), max(support - c[i], 0.0)),
        ):
            acceptance = (
                np.clip(retention / (penetration + _EPS), 0.0, 1.0)
                if penetration > 0 else 0.0
            )
            sweep = np.clip(
                penetration / (scale + _EPS) * (1.0 - acceptance), 0.0, 3.0
            ) / 3.0

            raw = (
                1.50 * pressure
                + 1.20 * compression
                + 0.70 * path_efficiency
                + 0.50 * acceptance
                + 0.45 * direction * volume_asym
                + 1.00 * direction * np.tanh(price_oi_values[i])
                - 0.80 * sweep
            )
            scores.append(1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0))))

        bull[i], bear[i] = scores

    return pd.DataFrame(
        {
            "reference_bull": bull,
            "reference_bear": bear,
            "reference_price_oi_pressure": price_oi_values,
        },
        index=x.index,
    )


def structural_stop(history, side, spot, lookback=45, left=2, right=2,
                    buffer_fraction=0.2):
    """Return a causal confirmed-pivot structural stop."""
    history = history.tail(lookback)
    values = history["low" if side == "CE" else "high"].to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(spot):
        return None

    buffer = float((history.high - history.low).tail(15).median()) * buffer_fraction
    anchors = []
    for i in range(left, len(values) - right):
        neighborhood = values[i-left:i+right+1]
        if side == "CE" and values[i] == neighborhood.min() and values[i] < spot:
            anchors.append(values[i])
        elif side == "PE" and values[i] == neighborhood.max() and values[i] > spot:
            anchors.append(values[i])

    anchor = anchors[-1] if anchors else (values.min() if side == "CE" else values.max())
    stop = anchor - buffer if side == "CE" else anchor + buffer
    return float(stop) if (stop < spot if side == "CE" else stop > spot) else None


def select_contract(snapshot, side, spot, spacing=100.0):
    """Choose ATM/one-ITM using causal normalized liquidity."""
    required = {"strike", "option_type", "groww_symbol", "close"}
    if snapshot.empty or not required.issubset(snapshot.columns):
        return None

    candidates = snapshot.loc[
        (snapshot.option_type.astype(str) == side)
        & pd.to_numeric(snapshot.close, errors="coerce").gt(0)
    ].copy()
    if candidates.empty:
        return None

    atm = round(float(spot) / float(spacing)) * float(spacing)
    itm = atm - spacing if side == "CE" else atm + spacing
    candidates["distance"] = np.minimum(
        abs(candidates.strike - atm), abs(candidates.strike - itm)
    )

    volume = pd.to_numeric(
        candidates.get("volume", pd.Series(0.0, index=candidates.index)),
        errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    oi = pd.to_numeric(
        candidates.get("oi", pd.Series(0.0, index=candidates.index)),
        errors="coerce"
    ).fillna(0.0).clip(lower=0.0)

    candidates["liquidity"] = 0.60 * np.log1p(volume) + 0.40 * np.log1p(oi)

    chosen = candidates.sort_values(
        ["distance", "liquidity", "groww_symbol"],
        ascending=[True, False, True],
    ).iloc[0]
    return str(chosen.groww_symbol)
