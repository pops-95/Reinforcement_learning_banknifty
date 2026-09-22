"""DQN CLI; shares Flask training, saving and validation without starting a server."""
from train_ppo import main


if __name__ == "__main__":
    main(default_algorithm="dqn")
