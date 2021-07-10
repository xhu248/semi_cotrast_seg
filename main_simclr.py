from experiments.simclr_experiment import SimCLR
import yaml
import argparse


def parse_option():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--dataset", type=str, default="hippo")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-f", "--fold", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_option()
    if args.dataset == "mmwhs":
        with open("config_mmwhs.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    elif args.dataset == "hippo":
        with open("config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    config['batch_size'] = args.batch_size
    config['epochs'] = args.epoch
    print(config)

    simclr = SimCLR(config)
    simclr.train()
