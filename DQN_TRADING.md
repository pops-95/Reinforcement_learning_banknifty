# DQN and PPO in the same trading website

New code: `banknifty_dqn.py` implements masked DQN, with optional Double DQN
targets. `train_dqn.py` is its command-line entry point. Both use the existing
`banknifty_rl_web.py`, data, five-action trading environment, cost model,
observation normalization, CUDA checks, trade logs, validation and test reports.
No dependency beyond the existing `requirements-rl.txt` is added.

**Implementation is untested. No application, training, tests, CUDA checks,
data preparation or Git operations were run for this change.** Commands below
are for you to run using the Python environment that already runs your website.

## New presets, with no daily limits

- `settings/banknifty_dqn_trade_quality.json`: masked Double DQN, CUDA required.
- `settings/banknifty_ppo_trade_quality.json`: PPO comparison preset.

Both disable `daily_profit_limit_enabled`, `daily_loss_limit_enabled`,
`close_on_daily_limit_enabled` and `daily_targets_enabled`. There is no Rs1,000
daily income objective or compulsory number of trades. Inactive daily numeric
fields remain for compatibility with the settings schema; they have no effect.
The older daily-budget preset is preserved and still enables those limits if
you load it instead.

PPO/DQN can choose either CE or PE, hold, exit a winner, exit a loser, or wait.
Indicators are advisory: entry-signal gates remain disabled. The existing
20-point per-trade protective stop, costs, session entry window and end-of-day
square-off remain enabled. Fixed profit-taking is disabled. Removing daily loss
limits allows multiple full-stop losses in one day; it does not make risk vanish.
Stops remain trigger prices, not guaranteed net loss caps.

The presets target validation win rate >=70%, net-point profit factor >=1.5,
average R >=0.1 and at least 100 validation trades. These are assessment/selection
criteria, not guaranteed outcomes or rewards that force trades to win. A fallback
model can be saved with `targets_met=false`. Training reward remains cost-adjusted
liquidation P&L in risk units; artificial win/entry/hold bonuses are off.

Disabling daily limits removes the long daily-lock WAIT periods in your earlier
run. It does not prove entry quality. WAIT can still be the right decision when
available trades have poor expected returns. Training includes exploration;
evaluate the saved deterministic policy before interpreting win rate as quality.

## Website

First stop/save any active job yourself. Restart the web process to load the
code changes, then reload the browser:

```bash
python banknifty_rl_web.py
```

Open http://127.0.0.1:5000. Use **LOAD SETTINGS FILE** and choose
`settings/banknifty_dqn_trade_quality.json`. The **Training algorithm** selector
should show DQN. Click **START TRAINING**. Do not restore AUTONOMOUS DEFAULTS
after loading a preset, because it replaces the environment fields.

Use the same save/stop, model picker and validation/test controls as PPO. DQN's
optimizer panel exposes loss, mean Q, replay size, update count and epsilon after
updates. Policy CUDA device/memory are reported. Q-values are not calibrated
probabilities: DQN trade probability fields are intentionally empty.

## CLI

```bash
python train_dqn.py train --config settings/banknifty_dqn_trade_quality.json
python train_dqn.py validate --model latest --split validation
# Only after model/settings choices are frozen:
python train_dqn.py validate --model latest --split test
```

`latest` in `train_dqn.py` searches DQN models only. An explicit filename is
safer when comparing multiple runs. Do not start CLI and web jobs concurrently;
the website's job lock only coordinates jobs in that server process.

Existing current-environment settings exports are reusable:

```bash
python train_dqn.py train --config settings/banknifty_ppo_trade_quality.json
```

The DQN CLI selects DQN even when an input preset says PPO. Shared settings
remain unchanged; absent DQN fields receive defaults. For the DQN-specific
learning rate, batch size, network and reward-normalization choices, use its own
preset. Alternatively:

```bash
python train_ppo.py train --algorithm dqn --config settings/banknifty_dqn_trade_quality.json
python train_ppo.py train --algorithm ppo --config settings/banknifty_ppo_trade_quality.json
```

Old exports without `algorithm` still load as PPO in the website. Settings from
older incompatible environment versions remain rejected. PPO weights cannot be
converted to DQN: start fresh training. Switching the form's algorithm does not
affect an already running job or the class used to load saved model weights.

## DQN implementation and settings

SB3's [DQN implementation](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/dqn/dqn.html)
provides replay learning and target-network updates. The new subclass adds masks
to random warmup, epsilon-greedy choices, deterministic evaluation and Bellman
targets. A wrapper captures each next-state mask before vector-environment
auto-reset; replay stores it with that transition. Masking only live actions
would leave learning targets vulnerable to invalid actions.

With `dqn_double_enabled=true`, the online network selects the next valid action
and the target network evaluates it. Turning it off selects ordinary masked DQN
targets. This is one-step DQN with uniform replay, not recurrent, distributional
or prioritized replay. The current implementation deliberately supports one
vectorized market environment. It does not claim DQN outperforms PPO here.

The DQN preset uses 50,000 replay transitions, 10,000 valid-action warmup steps,
one optimizer step per four environment steps, batch size 256, learning rate
0.0001, a 256/256/128 Q-network, target updates every 2,000 steps, and epsilon
decay from 1.0 to 0.05 over the first 30% of each seed's run. Approximately
3 million total steps are shared across three seeds. Evaluation is greedy,
not epsilon-random. Those values are research starting points, not tuned results.

Observation normalization stays on. Reward normalization is off in the DQN
preset so its value targets remain in the environment's reward units. PPO-only
`n_steps`, `n_epochs`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, target-KL
and `vf_layer*` settings are retained for file compatibility but not used by DQN.
`pi_layer*` sets DQN's Q-network widths. `gamma`, batch size, learning rate,
gradient clipping, CUDA and validation controls are shared.

Replay is held in system RAM. Just the two float32 observation arrays need
approximately `buffer_size * observation_dimension * 8` bytes, plus masks and
other arrays; the default can need hundreds of MB with this dataset. The neural
networks and sampled optimization batches use CUDA when enabled. Market replay
and data processing remain CPU-side. Saved models include Q-networks and the
matching normalization file, not replay memory; no exact training-resume command
is provided. All CLI training commands above start new runs.

## Checks you can run

```bash
python -m unittest test_autonomous_rl test_dqn -v
python check_rl_setup.py --cuda
```

`test_dqn` includes short synthetic CPU training, masked Bellman-target checks,
replay alignment, terminal-mask capture, save/load, shared-environment integration,
and settings-import checks. These are **not executed yet**. Use validation to
compare PPO/DQN with identical execution settings; reserve test for final
assessment and paper-trade before risking capital. Profitability remains unproven.
