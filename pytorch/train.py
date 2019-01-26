import argparse
import os
import yaml

import training
import checkpoint


def main(args):
    cfg = yaml.safe_load(args.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    session = training.Session(cfg)
    try:
        checkpoint.resume(session, args.resume_from)
    except OSError as e:
        print('Warning: could not resume from latest checkpoint')
    session.train_loop() 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch pipeline for training Leela Chess')
    parser.add_argument('--cfg', type=argparse.FileType('r'), required=True, help='yaml configuration with training parameters')
    parser.add_argument('--resume-from', type=str, help='Override which checkpoint to resume from')

    main(parser.parse_args())
