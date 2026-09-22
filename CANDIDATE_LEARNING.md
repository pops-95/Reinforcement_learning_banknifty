# Candidate-level learning

The website's primary workflow is now reference candidates → supervised quality
model → optional candidate PPO. The old minute trainer remains available under
the explicitly marked legacy buttons; its models cannot be loaded as candidate
models.

## Website workflow

1. Restart the server and reload the page to load the new code and controls.
2. Review **CANDIDATE DEFAULTS** and the deterministic exit/cost settings. Reference
   defaults are ignition 0.55, directional gap 0.03, and four-minute signal TTL.
   Candidate defaults enable 1.5R, underlying-target, momentum-decay, structural,
   emergency and time exits. Each deterministic exit remains configurable.
3. Click **BUILD CANDIDATE DATASET**. This labels all three chronological splits
   and publishes `data/candidates/banknifty_candidate_dataset.parquet` plus JSON
   metadata. It performs historical simulation, not model training. Empty days
   remain in the metadata's trading-day totals; there are no fake WAIT events.
4. Click **TRAIN SUPERVISED** first. This fits a CPU
   `HistGradientBoostingClassifier` on training dates only. The output is
   `probability_good_trade`, a classifier estimate, not a calibrated guarantee.
   Good-trade R of 0 means strictly positive net R; a positive setting means net
   R at least that value (for example, 1.0). Failed fills are excluded from fitting;
   stale exit labels are excluded by default. Both remain visible during replay.
5. Optional: click **TRAIN CANDIDATE PPO**. Each timestep is one SKIP/TAKE decision,
   not a minute. This uses the same event data and cached deterministic outcomes.
6. Select a candidate model and evaluate validation or test. Automatic model
   training evaluates validation only. Keep test for a final assessment.

Candidate learning rate, model capacity, label target, probability threshold,
threshold search, PPO rollout/batch/epoch/gamma and validation targets are exposed
on the website. The common PPO form supplies seed, policy/value network sizes,
learning rate, entropy, GAE, clipping, value coefficient, gradient clipping, and
observation normalization. Candidate PPO uses CPU, a single seed per run, and
unnormalized net-R rewards. Legacy CUDA, reward-normalization, checkpoint and
multi-seed switches do not control it. Use different shared seeds for separate
candidate experiments.

Save/load settings includes the new `candidate_training` section. Older exported
settings load with reported defaults for that section. Loading a file changes
only the form. Changing reference, exit or execution settings requires rebuilding
the candidate dataset. Models record both the effective environment settings and
the dataset identity; evaluation rejects incompatible datasets.

## Event and reward semantics

`reference_strategy.reference_candidate_table` emits every qualifying reference
timestamp with direction and contract. Same-direction consecutive events may
appear; neither labels nor the learner decide which timestamps exist. Optional
minute-entry quality/momentum/ATR/extension/EMA/OI/liquidity/delta/regime gates are
disabled in candidate mode. Their values become features for the learner instead.
Reference score/direction rules, decision-time quote availability, entry clock
and structural-stop availability define the event universe.

The builder calls `BankNiftyEnv.simulate_reference_candidate` for each independent
TAKE. That method reuses the existing next-open execution, costs, dynamic slippage,
emergency stops, reference exits and time exits. It produces realized R, profitable,
hit_1R, hit_1_5R, MFE_R, MAE_R, exit reason and holding duration, plus execution
metadata. `hit_*` reports observed **gross MFE**, not guaranteed executable net R.
Failed future fills stay in the dataset as rejected candidates with zero reward.

`CandidateBankNiftyEnv` exposes exactly `0=SKIP`, `1=TAKE`. SKIP earns zero. TAKE
returns the cached realized net R immediately, and advances past events while the
position was open or cooldown prevented re-entry. There are no HOLD/EXIT actions,
entry penalties, hold bonuses or terminal shaping in this environment. The cache
is an offline implementation of the shared deterministic simulator, not a second
execution model. One episode is a historical day containing candidates.

PPO gamma is per candidate event (default 1.0), not elapsed minutes. The learner
may rationally skip everything if opportunities are unattractive; the pipeline
does not invent profitable rewards or force trading.

## Leakage controls and evaluation

- `CANDIDATE_FEATURES` is an explicit allowlist of decision-time numeric features
  and availability flags. Timestamp IDs, labels, future fill success, exits,
  realized R, MFE and MAE are never model observations. Missing features become
  zero with an accompanying missingness flag.
- Training, validation and test are inherited from existing chronological splits.
  The loader requires disjoint calendar days in chronological order. Trades close
  within their historical day, so outcome horizons do not cross split boundaries.
- Supervised early stopping's random holdout is disabled. Optional probability
  threshold selection uses executable, non-overlapping validation replay only.
- The training dataset contains independently simulated, potentially overlapping
  candidates. Reported portfolio results enforce one position and cooldown; they
  are not a sum of all independent labels.
- Candidate precision is the percentage of TAKE decisions with good labels.
  Candidate recall is captured good candidates divided by **all** good reference
  candidates, including those unavailable due to overlap/cooldown. Consequently
  100% recall may be impossible in a single-position portfolio.
- Reports include reference/taken/skipped/unavailable/rejected counts, PF, average
  R, win rate, drawdown, CE/PE and regime breakdowns. Stale exits are counted and
  identifiable. Trading-day totals include empty candidate days.
- Threshold selection prefers meeting all configured targets with sufficient
  trades, then sufficient sample size, then average R with a drawdown penalty.
  A model failing targets is still saved and reported as failing; no outcome is
  guaranteed. No-loss PF is reported as undefined rather than a fabricated value.

## Files and cancellation

Candidate models live under `models/candidates/`: `.joblib` for supervised,
`.zip` plus matching `.vec.pkl` for PPO, with a JSON metadata file for each model.
Model files are published after their companion files. A trained model is saved
before validation, so stopping evaluation does not lose it. Supervised stopping
is checked between boosting iterations; PPO stopping is checked at the next
candidate callback after any ongoing optimizer update. A stop before fitting
anything finishes without creating a model. Build cancellation does not publish
a partial dataset. Jobs are serialized with the existing web training jobs.

Explicit supervised evaluation also writes a predictions Parquet file containing
`candidate_id`, timestamp, side, symbol and `probability_good_trade`, alongside its
JSON report. `predict_candidate_quality(model, candidates)` is the same inference
entry point and accepts decision-time rows with no outcome columns.

The web controls create local files on the server. As before, do not run separate
CLI builders/trainers concurrently with a web job: the lock is process-local.

## CLI equivalents (run only when ready)

The added dependency is scikit-learn/joblib in `requirements-learned-exits.txt`.

```powershell
python build_candidate_dataset.py --config my_settings.json
python train_candidate_model.py supervised --config my_settings.json
python train_candidate_model.py ppo --config my_settings.json
python train_candidate_model.py evaluate --model models/candidates/<model>.joblib --split test
```

`test_candidate_pipeline.py` contains regressions for label isolation, skip/take
transitions, rejected fills, overlapping candidates, target labels and parity
with the minute execution engine. The implementation was statically reviewed;
no dataset generation, application, training, dependency installation or tests
were run during this change, following the user's instruction.
