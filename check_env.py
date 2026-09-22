"""Environment and bounded integration checks. Never launches a long training job."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env as gym_check
from stable_baselines3.common.env_checker import check_env as sb3_check

from banknifty_env import BankNiftyEnv, ENV_VERSION, POSITION_FEATURES


def check_episode(env):
    gym_check(env, skip_render_check=True)
    sb3_check(env, warn=True)
    env.reset(seed=42)
    rng = np.random.default_rng(42)
    rewards = 0.0
    while True:
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        obs, reward, done, truncated, _ = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        rewards += reward
        if done or truncated:
            break
    assert env.position is None
    assert np.isclose(rewards, sum(t["reward"] for t in env.trade_log)*env.reward_scale)
    print(f"Gymnasium + SB3 checks passed; random episode: {len(env.trade_log)} trades, reward {rewards:.6f}")


def smoke_check(directory):
    """32 synthetic steps, one-epoch updates; no historical training."""
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from test_learned_exits import synthetic_files
    torch.set_num_threads(1)
    root = Path(directory)
    paths = synthetic_files(root, count=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = BankNiftyEnv(**paths, split="validation")
    vec = VecNormalize(DummyVecEnv([lambda: Monitor(env)]))
    model = MaskablePPO("MlpPolicy", vec, n_steps=16, batch_size=8, n_epochs=1,
        policy_kwargs=dict(net_arch=dict(pi=[512,256,128], vf=[512,256,128])),
        device=device, seed=42, verbose=0)
    model.market_env_version = ENV_VERSION
    model.market_env_config = env.config
    model.market_feature_columns = env.feature_columns
    model.market_position_features = list(POSITION_FEATURES)
    model.market_dataset_id = env.dataset_id
    model.learn(32)
    print(f"32-step MaskablePPO smoke passed on {device}; policy device={model.device}")
    model.save(root/"model.zip")
    vec.save(root/"vec.pkl")
    vec.close()
    raw = BankNiftyEnv(**paths, split="validation")
    restored = VecNormalize.load(root/"vec.pkl", DummyVecEnv([lambda: raw]))
    restored.training = False
    restored.norm_reward = False
    loaded = MaskablePPO.load(root/"model.zip", device="cpu")
    days = []
    before = restored.obs_rms.mean.copy()
    for _ in raw.days:
        obs, info = raw.reset()
        days.append(info["date"])
        while True:
            a, _ = loaded.predict(restored.normalize_obs(obs), deterministic=True, action_masks=raw.action_masks())
            obs, reward, done, _, _ = raw.step(a)
            assert np.isfinite(reward)
            if done: break
    np.testing.assert_array_equal(before, restored.obs_rms.mean)
    assert days == sorted(days) and len(set(days)) == len(days)
    restored.close()
    # Exercise the same worker and report serialization used by Flask/CLI.
    import banknifty_rl_web as web
    from unittest.mock import patch
    (root/"trade_logs").mkdir(exist_ok=True)
    with patch.multiple(web, OBS=paths["observations_file"], OPT_DIR=root,
                        MANIFEST=paths["manifest_file"], MODEL_DIR=root):
        web.EVAL_STATE.reset(1)
        web.EVAL_STATE.status = "validating"
        web.validation_worker(root/"model.zip", root/"vec.pkl", "validation", None)
        assert web.EVAL_STATE.status == "completed", web.EVAL_STATE.error
        result = web.EVAL_STATE.snapshot()
        assert result["steps"] == len(days)
        report = Path(result["evaluation"]["report_file"])
        assert json.loads(report.read_text())["status"] == "completed"
        client = web.app.test_client()
        assert client.get("/").status_code == 200
        assert client.get("/api/status").status_code == 200
        assert client.post("/api/start", json={"environment":{"min_hold_minutes":99}}).status_code == 400
    print("Model/VecNormalize round-trip, chronological validation, report and Flask checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true", help="Use tiny generated candles instead of historical files")
    parser.add_argument("--cuda-smoke", action="store_true", help="Also run a 32-step synthetic CUDA/CPU training check")
    parser.add_argument("--date", help="Check a specific historical trading date (YYYY-MM-DD)")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        if args.synthetic:
            from test_learned_exits import synthetic_files
            paths = synthetic_files(directory)
            env = BankNiftyEnv(**paths, split="validation")
        else:
            from rl_data import OBSERVATIONS, OPTIONS, MANIFEST
            env = BankNiftyEnv(OBSERVATIONS, OPTIONS, MANIFEST, split="train", random_day=False)
        if args.date:
            chosen = [day for day in env.days if str(day) == args.date]
            if not chosen:
                parser.error("Requested date is not in the selected check dataset")
            env.days = chosen
        check_episode(env)
        env.close()
        if args.cuda_smoke:
            smoke_dir = Path(directory)/"smoke"
            smoke_dir.mkdir()
            smoke_check(smoke_dir)


if __name__ == "__main__":
    main()
