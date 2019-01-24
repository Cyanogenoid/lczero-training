import argparse
import os
import yaml

import training


def main(args):
    cfg = yaml.safe_load(args.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    # TODO make dir to store checkpoints in

    session = training.Session(cfg)
    if args.resume_path is not None:
        session.resume(args.resume_path)
    session.train_loop() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch pipeline for training Leela Chess')
    parser.add_argument('--cfg', type=argparse.FileType('r'), required=True, help='yaml configuration with training parameters')
    parser.add_argument('--resume-path', help='Path to .pth checkpoint file to resume from')

    main(parser.parse_args())
