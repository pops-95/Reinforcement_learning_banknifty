# Autonomous BANKNIFTY PPO

For the opportunity-learning update, see [PPO_OPPORTUNITY.md](PPO_OPPORTUNITY.md)
and load `settings/banknifty_ppo_opportunity_v3.json`. It adds causal minute
history, time-diverse training starts, adaptive PPO exploration and intermediate
validation selection. It requires fresh training; performance is unverified.

PPO and masked DQN now share the website. See [DQN_TRADING.md](DQN_TRADING.md)
for the DQN implementation, user-run checks and CLI commands. The new
`settings/banknifty_dqn_trade_quality.json` and
`settings/banknifty_ppo_trade_quality.json` presets disable all daily trading
limits and daily income targets while keeping a per-trade protective stop.

For the new +40 net daily points / -20 daily loss-trigger experiment, see
[DAILY_TARGET_PRESET.md](DAILY_TARGET_PRESET.md) and load
`settings/banknifty_daily40_loss20_autonomous.json`. It requires fresh training
with the daily-budget environment; targets are not guaranteed performance.

The default policy chooses `WAIT`, `BUY_CE`, or `BUY_PE` when flat, and `HOLD`
or `EXIT` while invested. Both CE and PE are long option-premium positions. The
contract is ATM at entry and stays fixed until exit. Strike/expiry selection
and position sizing are not learned in this version.

Reference scores, momentum, trend, volatility, OI, volume and Greeks are advisory
observations. All discretionary entry gates are off by default. Learned EXIT is
on. Fixed targets, structural/momentum exits, minimum hold, cooldown and maximum
hold are off. Emergency premium stop, an entry session window, transaction costs,
and end-of-day square-off remain on and can each be switched off in the UI.
Episodes end each historical day, so a position is always accounted for at the
end of available daily data even if scheduled square-off is disabled.

No training, preparation, tests, dependency installation, or CUDA checks were run
as part of this edit. The commands below are for you to run.

## Install (Linux / bash)

Run from the repository directory. A separate environment keeps the downloader's
existing `.venv` independent.

```bash
python3 -m venv .venv-rl
source .venv-rl/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-rl.txt
python -m pip check
```

The CUDA 12.6 wheel command requires a compatible NVIDIA driver and GPU. Select
another supported wheel from [PyTorch's official installation instructions](https://pytorch.org/get-started/locally/)
if needed. The [official wheel list](https://pytorch.org/get-started/previous-versions/)
includes the CUDA 12.6 index. No torchvision or RAPIDS dependency is required.

## Prepare the existing data, offline

```bash
python prepare_rl_data.py
```

This uses `data/sb3/banknifty_sb3_observations.parquet` for timestamps, expiry
assignment and existing split labels, joins the current underlying features, and
rebuilds CE/PE option features from `data/sb3/option_market/`. It does not call
Groww, download data, or overwrite the source files. Outputs are:

- `data/rl/observations.parquet`
- `data/rl/manifest.json`, including coverage, feature list and dataset identity

For the dataset inspected in this conversation, existing splits are training
through May 2026, validation June–July 2026, and test August 2026. These existing
labels are authoritative; the downloader's different hardcoded dates do not
silently resplit them. To deliberately change boundaries before a fresh experiment:

```bash
python prepare_rl_data.py --train-end 2026-01-31 --validation-end 2026-07-31 --overwrite
```

`--overwrite` replaces only the prepared outputs. Stop all training/evaluation
jobs before rebuilding. Changed data requires fresh training; do not tune dates
based on test results. No candidate dataset is required for autonomous PPO.

Missing observations get zero values plus separate per-feature availability
flags. Option availability is distinct from OI/Greek availability. Option returns
and OI changes require consecutive same-contract candles within the same session.
Invalid IV solver-bound values and their Greeks are marked unavailable. Missing
option prices are never filled or invented. Exact ATM strikes are defined from
underlying close and strike spacing, independently of missing strikes.

Incomplete dates are retained in all splits and listed in the coverage report.
Missing quotes block entries; pending exits wait for available execution prices.
If an exit must be estimated at a deadline/data end, it is tagged stale in reports.
These measures allow training on incomplete data; they do not recover missing
market information or prove execution accuracy.

## User-run checks

```bash
python check_rl_setup.py --cuda
python -m unittest test_autonomous_rl -v
python check_env.py --synthetic --cuda-smoke
```

The first command validates the prepared data and runs a small CUDA allocation
and matrix multiplication, without training. The unit tests use synthetic data.
The last command intentionally performs a 32-step synthetic training smoke test,
model/normalization save-load, and web evaluation checks. These are separate from
historical training. Older test modules contain pre-existing three-action and
old-default assertions; `test_autonomous_rl` covers the new five-action workflow.

## Website

```bash
python banknifty_rl_web.py
```

Open **http://127.0.0.1:5000**. Use **CHECK PREPARED DATA**, inspect the coverage,
then **AUTONOMOUS DEFAULTS** and **START PPO TRAINING**.

- Every environment/training configuration field is exposed. Search the form
  for a setting. Feature-group switches hide inputs; optional gates restrict
  actions; forced exits override HOLD. Selecting individual feature names also
  disables their availability flags. Aggregate advisory scores are hidden when
  one of their main feature groups is disabled.
- **SAVE SETTINGS FILE** and **LOAD SETTINGS FILE** round-trip the form. Changes
  apply to a new run, never midway through an optimizer update. Evaluation uses
  saved settings unless **Use current settings** is explicitly selected.
- **STOP AND SAVE** saves the policy and matching observation normalization.
  It responds at an environment step after any current PPO update finishes.
  With multiple seeds enabled, total steps are shared across seeds.
- Progress includes steps, episodes, trade metrics, action frequencies, blocked
  entries, exit reasons, a reward curve, PPO losses/KL/entropy when logged, actual
  policy device, and CUDA allocated memory. GPU memory is not a utilization metric.
- Each completed seed is saved and evaluated on validation. The validation panel
  shows its progress. Test is only run when explicitly requested.
- Select a saved compatible model, choose **Validation** or **Test**, and click
  **VALIDATE**. **STOP AFTER DAY** finishes the current day and labels the report
  partial. Training and evaluation jobs are serialized within this server.
- Reports/trades live in `models/trade_logs/`. Model archives, metadata and
  matching `VecNormalize` statistics live in `models/`. Dataset identity and
  feature schema must match at evaluation. Old models/settings are incompatible.
- Optional candidate tools are under a collapsed research panel. They use a
  different entry-only formulation and their existing CPU training path; they
  are not needed for independent entry/exit PPO. Rebuild their labels before use.

Do not launch separate CLI jobs while the web server is training/evaluating;
the job lock is process-local. The server binds to localhost by default.

## CUDA behaviour

**Require CUDA for training** defaults to on. An unavailable GPU, invalid device
index or failed CUDA operation produces an error rather than a CPU fallback.
The policy is explicitly created on the selected device and its placement is
checked. PyTorch controls model forward/backward passes and optimization there.
Pandas processing and market replay run on CPU. Chronological evaluation runs
on CPU with frozen training normalization.

A small MLP with sequential market replay need not saturate a GPU or run faster
than CPU. SB3 documents this [PPO performance limitation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html).
CUDA selection, batch size, rollout length, network widths, CPU thread count,
TF32 and KL early stopping are configurable. GPU execution has not been verified
on your machine during this edit.

Optional TensorBoard:

```bash
tensorboard --logdir models/tensorboard --host 127.0.0.1 --port 6006
```

## CLI alternatives

```bash
python train_ppo.py train --timesteps 1000000
python train_ppo.py train --config /path/to/web-exported-settings.json
python train_ppo.py validate --model latest --split validation
python train_ppo.py validate --model latest --split test
```

Explicit CPU training:

```bash
python train_ppo.py train --cpu --timesteps 1000000
```

The reward is the change in estimated cost-adjusted liquidation R. Optional
shaping is off initially. Winning/losing exits share the EXIT action, and a loss
is counted even if it was voluntary. Closed-trade metrics exclude shaping.
Win rate and PF targets are validation criteria, not guaranteed outcomes.
Costs and quantity are configurable assumptions; this version does not apply
historical fee/lot-size schedules automatically. Bid/ask data remains absent.
