# Selective momentum PPO with reference strategy (reference_gated_ppo_v1)

The primary website workflow has moved to **candidate-level SKIP/TAKE learning**.
See [CANDIDATE_LEARNING.md](CANDIDATE_LEARNING.md) for dataset construction,
supervised training, candidate PPO, deterministic exits and evaluation. The
minute-environment behavior described below is retained as a legacy comparison.

## Current exit-learning defaults

**SAVE SETTINGS FILE** downloads all current training and trading/reward form
values as a timestamped JSON file. **LOAD SETTINGS FILE** lets you choose a
previously exported file and restores both groups of controls. The complete file
is validated before applying it; invalid types, unknown settings and incompatible
versions are rejected without changing the form. Missing fields use current
defaults and are listed in the status message. Loading changes only the form,
not an active training job. For saved-model evaluation, enable **Use current
settings** to apply the loaded environment settings. Files contain configuration,
not model weights or validation model selection. The server must be running to
validate a save/load; the downloaded file location follows browser preferences.

For a run with no trades, inspect **Action distribution and exits** on the website.
Entry diagnostics count flat steps, steps with a BUY available, blocked steps,
voluntary WAIT decisions, and opened entries. Per-side `blocked_by` counts show
every failed entry condition, so multiple reasons may count on the same step.
Repeated mask queries do not inflate these counts. Trade cards count closed
positions, whereas `entries_opened` includes positions that have not yet closed.

The **ENTRY EXPLORATION PRESET** button updates the form for the next fresh run.
It keeps reference-directed entries at ignition 0.55 / direction gap 0.03 and
disables additional quality, momentum, ATR-expansion, extension, EMA, path,
OI, liquidity, delta and regime gates. Risk controls and costs are left at their
current settings. Use entry diagnostics to verify entry flow before re-enabling
filters one at a time. The preset does not modify an active run and does not
promise profitable trades. With the default 4096-step rollout, PPO has not yet
performed its first optimizer update at 1124 steps.

This section supersedes the historical defaults described below.

- Path-efficiency filtering defaults OFF. Efficiency is now calculated from
  completed closes within each day, using a configurable 20-bar window. Enabling
  the filter works without rebuilding the dataset; warmup bars remain ineligible.
- PPO learned EXIT defaults ON after the minimum hold. Fixed R take profit,
  underlying-points take profit, and momentum-decay forced exits default OFF.
  Emergency stops, structural stops, maximum holding time and square-off remain
  active. PPO may exit a losing position; profit is never a prerequisite to EXIT.
- Net P&L in risk units remains the principal reward. A small 0.02R win bonus
  defaults ON for exits earning at least 10 net option points after modeled costs.
  Estimated stale exits do not receive that bonus. Hold/giveback and terminal
  shaping share the per-trade absolute shaping budget. The giveback penalty can
  be enabled independently of the HOLD bonus, with a configurable peak-R trigger.
- Validation targets remain 70% win rate and 1.5 profit factor, with at least
  50 validation trades by default. Undefined PF (no losses) does not qualify as
  meeting the PF target. Zero-trade candidates receive the lowest selection score.
  Candidates meeting all targets are preferred; otherwise the best score is saved
  with unmet targets explicitly reported. These are selection criteria, not a
  performance guarantee or proof of statistical reliability.

The optional **Minimum average validation trades/day (0 = disabled)** control
adds a frequency check to legacy PPO and candidate-model selection. It divides
closed trades by all evaluated days, including days without trades. Older
settings default to zero, preserving their selection behavior. A model below
the configured minimum cannot meet all targets. If none meets every target,
selection first prefers models meeting the enabled frequency and sample-size
minimums, then compares scores. A fallback model can still be saved with unmet
checks reported.

For the V4 legacy minute PPO environment, load
`settings/banknifty_settings_v1.6_min2_trades.json` after restarting the updated
website. This untested variant of v1.5 targets an average of at least 2 trades/day,
70% wins and net-point profit factor 1.5, with at least 100 validation trades.
It widens the entry window, moderately relaxes entry filters and shortens
cooldowns while retaining costs and protective exits. This is an average
validation requirement, not a forced quota for every day or a performance
guarantee. Tune on validation data and reserve test data for final assessment.

All these switches, thresholds, rewards, and validation targets appear on the
website and are saved with the model. Reload the website after restarting the
server to see new defaults, then start fresh training. Existing saved model
settings are preserved. Evaluate on validation data for tuning and reserve the
test split for final assessment. No training or checks were executed for this edit.

The reference strategy is now enabled by default. See **Reference strategy with
PPO** below for the added settings, execution rules and migration requirements.

The project retains its parquet datasets, chronological splits, Flask workflow,
CUDA-capable MaskablePPO (SB3 PPO with action masks), VecNormalize and checkpoints.
No dataset rebuild is required. One expiry is loaded at a time; the previous
option lookup is released before the next is built.

## Commands (PowerShell, from this project directory)

Use the same Python installation that already runs the app. The local .venv may
not contain the project's dependencies. sb3-contrib is the existing additional
requirement; install only if missing:

```powershell
python -m pip install -r requirements-learned-exits.txt
python -B -m unittest test_learned_exits -v
python -B check_env.py
python -B check_env.py --date 2025-01-02
python -B check_env.py --synthetic --cuda-smoke
python -B banknifty_rl_web.py
```

Open http://127.0.0.1:5000 for training/validation. Alternative CLI commands use
exactly the same environment configuration and workers:

```powershell
python -B train_ppo.py train --timesteps 100000
python -B train_ppo.py validate --model latest --split validation
python -B train_ppo.py validate --model latest --split test
```

The 100k command starts actual training when YOU run it. PPO rounds the requested
budget up to a whole rollout (default 4096). No long run is started by checks.
The --cuda-smoke check trains only 32 synthetic steps and verifies saving/loading,
frozen evaluation normalization, chronological evaluation, Flask and JSON reports.

Optional environment settings can be supplied with `--config settings.json` to
train or validate. Example JSON:

```json
{
  "min_hold_minutes": 3,
  "reentry_cooldown_minutes": 3,
  "min_stop_points": 30,
  "stop_premium_pct": 0.10,
  "entry_penalty_r": 0.03,
  "overtrading_penalty_enabled": false,
  "tradeability_gate_enabled": false
}
```

Validation otherwise uses saved settings. Keep test data reserved for final
assessment. Do not run CLI training concurrently with Flask training; the Flask
job lock only coordinates jobs within that server process.

The website's **STOP AND SAVE** button shows `stopping` immediately. Training
stops at the next rollout callback (an ongoing optimizer update or data load must
finish first), saves the current policy and matching normalization statistics,
and skips remaining seed validation and benchmarking. Automatic validation and
benchmarking also check the training stop signal between environment steps.
Stopping before a model is initialized ends cleanly without creating a model.

Each seed is saved before automatic validation, so an interrupted validation or
benchmark cannot prevent access to the trained model. Model ZIPs become visible
only after their normalization file and settings JSON are in place. Seed saves
remain available in the model picker; a completed run publishes its validation
winner as the newest model. An interrupted run retains the current seed rather
than claiming it is the best validated model. Starting another job is blocked
until the previous worker has finished saving and cleaning up.

## Behaviour

Three state-dependent actions, masked both during learning and evaluation:

| State | 0 | 1 | 2 |
|---|---|---|---|
| Flat | WAIT | BUY CE | BUY PE |
| Holding | HOLD | EXIT | masked |
| Minimum hold / pending exit | HOLD | masked | masked |
| Flat in cooldown / after cutoff | WAIT | masked | masked |

One position at a time; an exit cannot also open a new position in the same step.
There is no direct reversal or pyramiding. Minimum hold defaults to three complete,
valid option candles including the entry candle. Missing candles do not satisfy
minimum hold. Emergency stop, maximum holding time and square-off always override
minimum hold. Re-entry fills cannot occur before exit time + 3 minutes.
Cooldown observation measures time until the next possible fill is eligible.

Entry: observe completed candle t, select t's exact ATM symbol, buy t+1 OPEN with
slippage. If that open is missing, the entry is rejected without penalty. Masks
never inspect t+1 prices/availability. Once entered, keep the same symbol.
Voluntary EXIT uses t+1 OPEN or the next available open if a quote is missing;
its pending order cannot be cancelled by HOLD. It cannot use later candle high/low.

Emergency stop is frozen at entry:

    risk_points = max(30, 0.10 * entry_price)

Optional ATR term (OFF by default): also include `1.2 * option_atr`. ATR absence
never prevents entry or breaks the stop calculation. Stop price = entry - risk.
Gaps can fill beyond that stop. No compulsory profit target: +40 points is a
configurable diagnostic threshold, never a forced exit.

Defaults retain no entries from 15:00, square-off at 15:25 OPEN, and a 45-minute
maximum holding time. Exit reasons are AGENT_EXIT, EMERGENCY_STOP, EMERGENCY_GAP,
EOD, STALE_OPTION_DATA, and TIME_LIMIT. Details distinguish scheduled square-off,
maximum hold and early data end. Missing deadline quotes use explicitly flagged
last-price estimates. Early manual resets preserve interrupted trades separately;
normal training/evaluation completes episodes before resetting.

## Observations and momentum

All manifest features remain in the same order. The appended position feature
schema is `POSITION_FEATURES` in banknifty_env.py: side, holding/cooldown state,
points, percent return, initial risk, current/MFE/MAE R, distances to entry/high/low,
desired-move flag, held-contract ATR and quote age, pending-exit state, and changes
in delta/IV/velocity/acceleration/momentum since the entry decision. Optional
changes have availability flags and default to zero when unavailable.

The supplied dataset has returns, momentum_3, acceleration, volatility and Greeks.
It does not have all the requested ignition/p_bull/p_bear/path-efficiency features.
Existing returns/momentum serve as fallbacks; there are no fabricated regime
labels and no extra feature pipeline. MFE/MAE accumulate only on already held
candles. On early intrabar exits the order of high/low is unknown: only prior
completed candles and the exit price contribute, avoiding future excursions.

The optional quality gate (OFF) averages available [0,1] components: directional
ignition/probability, absolute path efficiency, a tanh transform of directional
3m/5m returns, and capped ATR expansion. No future quantiles or labels are used.
If none of those inputs exists, the enabled gate blocks entry. Threshold defaults
to 0.55. The policy otherwise chooses entries and exits.

## R reward and costs

Each trade uses its fixed initial risk as denominator, for CE AND PE long options.
Marked open R = (last premium - slippage-adjusted entry - entry fee points) / risk.
Realized net R uses slippage-adjusted exit and both order fees. Each step receives
the difference in realized-plus-marked R, so the exit adds only the remainder.
Undiscounted base rewards sum exactly to final net trading R over a full episode.

Brokerage is converted to premium points with `brokerage_per_order / trade_quantity`.
Default simulated quantity is 30 units; this is NOT a claim about historical lot
sizes. Set a consistent appropriate quantity for the experiment. Exchange taxes,
other fees, and real bid/ask spread are not modeled beyond configured slippage.

Entry penalty defaults to 0.03 R, charged only on a successful entry. Optional
extra-entry penalty is OFF; when enabled, entries after the first five/day cost
another 0.01 R. WAIT has no penalty and there is no hard daily trade cap.

Small HOLD shaping: +0.002 R when profitable with favourable existing momentum;
-0.002 R when MFE >= 1 R and at least 50% of peak is given back. Giveback takes
priority over the bonus. No bonus for negative current R. Shaping is limited to
an absolute budget of 0.05 R/trade. All coefficients are configurable and can be
zeroed. The trading metrics EXCLUDE these shaping terms and entry penalties.
With reward_scale=1: episode learning reward = cumulative trading R - entry
penalties + shaping. VecNormalize rescales this for PPO, not for reports.

## Results and migration

CSV logs are written to models/trade_logs for every training/evaluation run and
flushed per completed trade. They include risk, net/gross points, R, MFE/MAE,
holding time, desired-move flag, costs, model ID, and exit reason/detail.
Evaluation probabilities are logged when available; training probabilities are
null because the standard rollout callback runs after environment execution.
Validation additionally saves a JSON report beside its CSV. Action distributions
count semantic decisions, including forced WAIT/HOLD; failed buy fills are counted
separately. Trading-day denominator counts started episodes with at least one action
(training can repeat dates). Validation visits each date once chronologically.
Profit factor uses net premium points (same configured quantity), while drawdown
uses cumulative closed-trade R; neither is an account-equity return percentage.

Old models and VecNormalize states are incompatible with this three-action/R
schema. They are preserved, rejected on load and hidden from the model picker.
Start fresh training; do not reuse old normalization. Checkpoints use unique run
prefixes. No TensorBoard integration is added.
# Reference strategy with PPO

The default environment is now `reference_gated_ppo_v1`. Start fresh training;
older models and their VecNormalize files are incompatible with the new observation
fields and masks. No dataset rebuild is needed. The original external project is
unchanged.

The website's existing trading settings include an **Enable reference strategy +
PPO** checkbox, enabled by default. When flat, PPO may WAIT or buy only the side
permitted by the reference ignition signal. It still learns whether to enter and
when to exit. Disabling the checkbox restores unrestricted PPO entry selection
subject to the existing protection and optional quality gate.

Adapted rules from `Banknifty_trad-deep-learning/backtest.py` and its price/OI
feature helper:

- Completed-candle compression, directional pressure, path efficiency, breakout
  acceptance, sweep and OI migration produce bullish/bearish ignition scores.
  Default lookback: 20 bars, minimum score: 0.55, directional gap: 0.03, signal
  validity: 4 minutes. These scores are not calibrated probabilities.
- Choose ATM or one ITM contract on the signaled side by completed-candle
  `volume + 0.05 * OI`, falling back to the closest available strike. The exact
  selected symbol is retained throughout the trade. Strike spacing defaults to
  100 and the OI window to five strikes each side.
- Freeze the structural underlying stop from the last confirmed pivot, otherwise
  the lookback extreme: 45-bar history, two left and two right confirmation bars,
  buffer of 0.20 times the median of the last 15 candle ranges.
- Mandatory reference exits: structural stop touched, 160 BANKNIFTY index points
  of favorable closing-price movement from underlying entry OPEN, or own ignition
  below 0.48 while the opposite score is stronger. **160 is not option premium
  profit.** These reference exits override minimum holding time. PPO EXIT still
  obeys minimum hold. If stop and target are both observed, stop takes priority.
- Reference signals use completed candles and execute at the next available held
  contract OPEN. Existing emergency premium stop, 45-minute maximum holding time,
  15:00 entry cutoff, 15:25 square-off, costs and cooldown remain effective.

All parameters above, alongside existing risk/reward settings, are configurable
on the website and saved in model metadata. Validation uses saved settings unless
the existing current-settings override is selected. The reference's XGBoost
confirmation and its ML-only settings are not used: PPO supplies the learned
entry/exit decisions. Its external code checks `label_bull/label_bear` although
the label generator produces `y_bull/y_bear`, causing its ML path to fall back to
raw ignition rules.

Intentional execution differences: each day starts with a fresh feature warmup;
OI uses same-minute snapshots from this project's one-minute dataset, missing OI
contributes zero; unavailable index volume contributes zero. No signal uses the
future last timestamp of a contract. Entry/exit fills use next OPEN rather than
the reference's same-candle CLOSE. Gaps can make an exit worse than its trigger.

Checks (no long training):

```powershell
python -B -m unittest test_reference_strategy test_learned_exits -q
python -B check_env.py --synthetic
```

Start the website with `python banknifty_rl_web.py`. Use its fresh-training and
validation buttons. Reference exit reasons appear as `STRUCTURAL_STOP`,
`UNDERLYING_TARGET`, or `MOMENTUM_DECAY` in the existing diagnostics and trade CSV.
