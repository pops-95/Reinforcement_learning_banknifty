# PPO opportunity-learning preset

Load `settings/banknifty_ppo_opportunity_v3.json` in the existing website.
Start a **fresh model**: four history frames change the observation dimensions.
Existing settings are preserved; omitted new controls default to the old behavior.
No data rebuild is required solely for these changes if the prepared dataset is
already valid. No training, tests, CUDA checks, dependency installation or Git
operations were executed for this update.

## What changed

- **Temporal input:** four completed clock minutes of market features, followed
  by timestamp-availability flags and the current position state. Missing minutes
  are zero-padded, not filled with future values or treated as adjacent bars.
  History never crosses a session boundary. The one-frame setting preserves the
  previous observation layout. This is a simple memory aid for MLP PPO, not an LSTM.
- **Time-diverse experience:** half of training episodes can start at a uniformly
  sampled available bar within the first 180 minutes. Sampling never uses future
  returns or successful-trade labels. Earlier causal observations remain available.
  Validation and test still replay complete days from their first bar.
- **Less forced waiting:** a flat episode can finish once a sticky daily lock or
  session cutoff prevents every future entry. This does not skip quote gaps,
  voluntary waiting, the pre-entry window or an open position. Daily limits are
  disabled in this preset, so the main effect is removing post-cutoff flat tails.
- **Adaptive exploration:** PPO entropy decays from 0.02 toward 0.005. If at least
  128 eligible flat decisions occur in a rollout and at least 90% are voluntary
  WAIT, the coefficient temporarily rises to 0.04 (subject to the configured cap).
  Forced waiting is excluded. Actions are still sampled from PPO's masked policy;
  there is no forced entry, WAIT penalty or profit invented by reward shaping.
- **Intermediate validation:** approximately every 250,000 steps per seed, after
  an optimizer update, save and evaluate a policy with its matching frozen
  normalization statistics in a separate CPU environment. Training then resumes
  with its environment and random-number-generator states preserved. The best
  candidate may be an earlier checkpoint instead of final weights.
- **Selection:** prefer candidates meeting all configured targets; otherwise
  prefer candidates meeting the sample, frequency, PF and expectancy requirements,
  then sample/frequency eligibility and the existing net-R/PF/drawdown score.
  Unmet requirements remain visible. If none meets all targets, the saved result
  is explicitly labeled a research fallback in `selection_warning`.

The MLP receives recent history following the practical approach discussed in the
[Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.0.0/modules/ppo.html).
That recommendation is not evidence of profitability on this dataset.

## Preset choices and limitations

The 3,000,000-step budget is divided across three seeds and rounded to complete
2,048-step rollouts. Batch size is 256, learning rate 0.0001, five PPO epochs and
gamma 0.995. Observation normalization remains on; reward normalization is off
so net-R reward retains a stable scale for the entropy coefficient. Existing
transaction-cost and dynamic-slippage assumptions are retained, with quantity 30.

PPO chooses CE/PE entries and voluntary EXIT. Mathematical features are advisory;
discretionary entry gates, minimum hold, cooldown, profit targets and extra
loss-aversion/reward penalties are off. **The per-trade 20-point protective stop
and session square-off remain on.** Gaps, slippage and fees can exceed 20 points;
this is not a guaranteed maximum loss. There is no daily profit/loss lock or
Rs.1,000 daily income restriction. Nothing widens stops just to inflate win rate.

Assessment targets are 70% win rate, net PF >= 1.5, average R >= 0.1, at least
100 validation trades and an average of at least one trade per evaluated day.
**The frequency requirement is a validation criterion, not a trade quota.**
Zero-trade days count in its denominator. It does not force a trade every day.

These changes do not guarantee the targets or eliminate WAIT convergence. An
unprofitable action set can rationally favor WAIT. Entropy encourages training
exploration but cannot manufacture a tradable signal. No-entry evaluation and
insufficient trade frequency fail validation. If they persist, inspect feature
quality, execution costs and held-out performance before increasing exploration.

## Commands for you to run

Use the Python environment containing this project's RL dependencies.

Optional regression checks (not run by the coding agent):

```bash
python -m unittest test_opportunity_learning test_autonomous_rl test_training_targets test_trade_frequency
```

Start the website:

```bash
python banknifty_rl_web.py
```

Open http://127.0.0.1:5000, load `settings/banknifty_ppo_opportunity_v3.json`,
check the prepared data and click the training button. New controls can be
changed or disabled in the same form. CUDA is required by this preset; the
existing actual-execution and policy-device checks remain in place. GPU training
does not mean market replay or validation runs on the GPU.

Watch **Learning progress / optimizer statistics** for
`exploration/voluntary_wait_fraction`, `exploration/eligible_entry_decisions`,
`exploration/entropy_coefficient` and `exploration/wait_collapse_warning`.
The validation panel shows each evaluation. Model-selection diagnostics record
each candidate's seed, steps, metrics, targets and the selected checkpoint.
Periodic validation pauses training; temporarily idle CUDA during CPU evaluation
is expected. It adds evaluation time and saved-model disk usage.

Sampled training episodes may be partial sessions: training cards labeled daily
averages are **not** estimates of full-day income. Use chronological validation
and test metrics, including all zero-trade days, for daily assessment.

CLI alternative (not concurrently with website training):

```bash
python train_ppo.py train --config settings/banknifty_ppo_opportunity_v3.json
```

After final model selection, use its exact filename in place of the placeholder:

```bash
python train_ppo.py validate --model banknifty_ppo_SELECTED_TIMESTAMP.zip --split test
```

Do not override its saved environment with an older settings file: history size
must match. Do not use the test split to tune settings/checkpoints repeatedly.
Compare this preset against the previous preset on the same chronological
validation split; use the untouched test split only after configuration selection.
Repeated checkpoint selection can itself overfit validation. Report net PF,
expectancy, drawdown, trade count, day coverage and per-seed results alongside
win rate; a high win rate alone is not sufficient.
