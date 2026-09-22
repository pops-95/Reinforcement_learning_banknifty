"""CLI entry points using exactly the same configuration/workers as Flask."""
import argparse
import json
from pathlib import Path


def main(default_algorithm=None):
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    train = subs.add_parser("train")
    train.add_argument("--algorithm", choices=["ppo", "dqn"], default=default_algorithm)
    train.add_argument("--timesteps", type=int)
    train.add_argument("--n-steps", type=int)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--seed", type=int)
    train.add_argument("--cpu", action="store_true", help="Explicitly train on CPU instead of requiring CUDA")
    train.add_argument("--config", type=Path, help="Web-exported settings JSON or environment-only JSON")
    validate = subs.add_parser("validate")
    validate.add_argument("--model", default="latest", help="Compatible model filename in models/, or latest")
    validate.add_argument("--split", choices=["validation", "test"], default="validation")
    validate.add_argument("--config", type=Path, help="Explicit override of saved environment settings")
    args = parser.parse_args()
    import banknifty_rl_web as web
    payload = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else None
    if payload and payload.get("format") == "banknifty_training_settings":
        settings = web.validate_settings_file(payload)["settings"]
        environment, training = settings["environment"], settings["training"]
    else:
        environment, training = payload, {}
    if args.command == "train":
        overrides = dict(algorithm=args.algorithm, total_timesteps=args.timesteps, n_steps=args.n_steps,
                         batch_size=args.batch_size, n_epochs=args.epochs,
                         learning_rate=args.learning_rate, seed=args.seed)
        training.update({k: v for k, v in overrides.items() if v is not None})
        if args.cpu:
            training["cuda_enabled"] = False
        cfg = web.parse_training_config({**training, "environment": environment or {}})
        web.STATE.reset(cfg["total_timesteps"])
        web.training_worker(cfg)
        state = web.STATE
    else:
        name = args.model
        if name == "latest":
            name = None
            pattern = "banknifty_dqn_*.zip" if default_algorithm == "dqn" else "banknifty_*.zip"
            for path in sorted(web.MODEL_DIR.glob(pattern), key=lambda p: p.stat().st_mtime_ns, reverse=True):
                try:
                    web.model_files(path.name)
                    name = path.name
                    break
                except ValueError:
                    continue
            if name is None:
                parser.error("No compatible trading model; train a fresh model first")
        model, vec = web.model_files(name)
        override = web.parse_env_config(environment) if environment is not None else None
        web.EVAL_STATE.reset(1)
        web.EVAL_STATE.status = "validating"
        web.EVAL_STATE.model_path, web.EVAL_STATE.vec_path = str(model), str(vec)
        web.validation_worker(model, vec, args.split, override)
        state = web.EVAL_STATE
    result = state.snapshot()
    # Full trade records are in CSV; avoid dumping hundreds to the terminal.
    result.pop("trades", None)
    print(json.dumps(result, indent=2))
    if state.status == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
