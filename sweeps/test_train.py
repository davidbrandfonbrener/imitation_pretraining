"""Script to test training."""
import argparse
from configs import test
from imitation_pretraining.experiments.training import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--alg", type=str, default="bc")
    args = parser.parse_args()

    test_config = test.get_config(args.alg, local=args.local)
    test_config["device"] = "cpu"
    train.run(test_config, test=True)
