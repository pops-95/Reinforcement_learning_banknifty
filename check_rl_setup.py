"""User-run dependency, prepared-data and optional CUDA check. No training."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Require and exercise CUDA")
    args = parser.parse_args()
    from rl_data import load_dataset
    from banknifty_rl_web import TRAIN_DEFAULTS, training_device
    import torch
    import stable_baselines3
    import sb3_contrib
    _, meta = load_dataset()
    device = training_device({**TRAIN_DEFAULTS, "cuda_enabled": args.cuda})
    print(json.dumps(dict(device=device, torch=torch.__version__, cuda_build=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(0) if args.cuda else None,
                          stable_baselines3=stable_baselines3.__version__, sb3_contrib=sb3_contrib.__version__,
                          dataset_id=meta["dataset_id"], splits=meta["splits"],
                          features=len(meta["observation_columns"]), quality=meta["quality"]), indent=2))


if __name__ == "__main__":
    main()
