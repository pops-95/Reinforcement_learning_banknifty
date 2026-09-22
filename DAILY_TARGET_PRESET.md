# Daily +40 / -20 autonomous PPO preset

Load `settings/banknifty_daily40_loss20_autonomous.json` in the website.
This is an **untested research starting point**, not a calibrated strategy or
a promise of 70% wins, PF 1.5, or daily income. No training, tests, CUDA checks,
or data preparation were run for this change.

## What it does

| Control | Preset |
| --- | --- |
| Quantity | 30 units, using your requested assumption |
| Entry | PPO chooses WAIT, BUY_CE or BUY_PE; no indicator entry gates |
| Ordinary exit | PPO chooses HOLD or EXIT, for profits or losses |
| Per-trade protective stop | 20 premium points before sell slippage/fees; cheaper premiums have a smaller executable stop distance |
| Daily profit trigger | +40 net premium points across all trades, including estimated open-position liquidation P&L |
| Daily loss trigger | -20 net premium points on the same basis |
| Once either daily trigger is reached | Request exit at the next available open and block new entries until the next day |
| Trades per day | No forced minimum or fixed maximum; one or several can contribute |
| Scheduled square-off | 15:25; entries stop at 15:00 |
| Training | About 3 million total steps shared across 3 seeds; CUDA required |
| Validation | >=70% winning trades, PF >=1.5, nonnegative average R, >=30 trades |
| Daily validation objectives | Every evaluated day >=40 net points; no day below -20 net points |

“Only one -20” is interpreted as a **total net daily loss budget**, not a
one-losing-trade limit. Several smaller losses can use that budget. A loss after
earlier profits may leave room for another trade. Disable/adjust the daily
controls in the website if this is not your intended policy.

Daily triggers are observed at completed candle closes. They are not guaranteed
fill prices or hard loss caps. Per-trade intrabar stops can trigger first; costs
can make a nominal -20 stop a worse-than-20 net loss. Gaps, slippage and unavailable
quotes can also overshoot budgets. A +40 liquidation mark can become +35 at the
next open; this still stops trading for the day and fails the +40 validation
check. There is no lookahead fill at a favorable previous price. With
`close_on_daily_limit_enabled=false`, only realized P&L triggers the entry lock,
and open positions are not automatically closed by this daily overlay.

One trade of +40 **net** points is Rs1,200 at quantity 30. Rs1,000 requires about
33.34 net points. A +40 **gross** move is not +40 net points: every round trip
incurs costs. These figures exclude personal income tax and operating costs.
Even an idealized strategy winning 70% of its trades at +40 and losing 30% at
-20 averages only 22 gross points (Rs660) per trade. A win-rate target alone
cannot ensure Rs1,000 every day. There is no forced-trading quota to chase losses.

## Learning and assessment

Indicators remain observations, not permissions to trade. Per-trade fixed
take-profit is OFF, so PPO may take smaller profits, cut losses early, or hold
until the remaining daily profit budget is reached. The fixed protective stop
and daily/EOD overlays deliberately limit autonomy for risk management.

Net liquidation R remains the training reward. The 70%/1.5/daily goals are
validation checks and seed-selection criteria, not differentiable guarantees.
Gamma 0.999 gives later session outcomes more weight; the learning rate 0.0002,
KL limit and modest entropy are experimental choices, not optimized values.
Increasing steps alone does not establish an edge.

Daily reports include all evaluated sessions, including zero-trade days, with
date, net points, modeled rupees, trade count and daily-limit reason. Check the
validation details / JSON report for `daily_results` and `targets`. Worst-day,
best-day and average daily metrics are displayed after evaluation completes.
Training's average daily points aggregate sampled episodes, not distinct dates.

The requested every-day objective is intentionally strict: `target_daily_hit_rate_pct`
is 100. If no seed meets every target, the best available fallback can still be
saved, but `targets_met` remains false. “Completed” or “saved” is not approval for
live trading. Thirty trades is only a minimum assessment floor, not statistical
proof. Tune on validation only; reserve test for final assessment. Paper-trade
before risking money, including days with missing/stale quotes.

## Costs and assumptions

The new preset overrides legacy rates without rewriting your older settings.
It uses sell-side STT 0.0015, exchange/IPFT combined turnover fraction 0.0003553,
SEBI 0.000001, buy-side stamp duty 0.00003, GST 0.18, brokerage Rs20/order and
base slippage 0.001 with dynamic slippage enabled. Verify brokerage and realized
slippage for your account. The dataset has no bid/ask execution guarantee.

NSE lists sell-side option STT at 0.15% from April 1, 2026 in its
[STT schedule](https://www.nseindia.com/static/products-services/equity-derivatives-securities-transaction-tax).
Its [February 27, 2026 fee circular](https://nsearchives.nseindia.com/content/circulars/FA73061.pdf)
lists combined equity-option exchange/IPFT charges of Rs3,553 per crore of
premium turnover on each side. Other levies are listed on
[NSE's charges page](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies).
The combined rate is stored in `nse_option_txn_rate` because the simulator has
no separate IPFT field. These fixed assumptions apply across the historical
dataset; this is not a reconstruction of date-specific fee or lot-size schedules.

## Commands for you to run

Use the project's RL Python environment. Installation and initial data preparation
are described in [README.md](README.md); already-prepared data need not be rebuilt
for the new daily-budget observations. The environment version is now
`autonomous_ppo_v6_daily_budget`: **train a fresh model**, not an old checkpoint.

```bash
source .venv-rl/bin/activate
python -m unittest test_autonomous_rl -v
python check_rl_setup.py --cuda
python banknifty_rl_web.py
```

Open http://127.0.0.1:5000, use **LOAD SETTINGS FILE**, select
`settings/banknifty_daily40_loss20_autonomous.json`, inspect the settings, and
start fresh PPO training. Do not click AUTONOMOUS DEFAULTS after loading: that
would replace the preset's environment configuration. The candidate-pipeline
defaults warning is harmless; this preset uses autonomous PPO, not that pipeline.

Alternatively, with no web training/evaluation job running:

```bash
python train_ppo.py train --config settings/banknifty_daily40_loss20_autonomous.json
python train_ppo.py validate --model latest --split validation
# Only after choices are frozen:
python train_ppo.py validate --model latest --split test
```

Use an explicit model filename instead of `latest` if multiple runs exist.
New tests cover cumulative daily budgets, cost-adjusted targets, next-open fills,
reset behavior, preset import, and daily validation including zero-trade days.
They are provided for you to run and have not been executed here.
