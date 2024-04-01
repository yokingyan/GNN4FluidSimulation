import os
import sys
import argparse
import numpy as np
import random
import torch
from src.utils.tools import init_seed
from src.config.configs import _C as cfg
from src.utils.tools import get_data_root


def args_parse():
    parser = argparse.ArgumentParser("The Configs of Train Simulator", add_help=False)

    parser.add_argument('--seed', type=int, help='random seed', default=0)
    #

    return parser.parse_args()


def main():

    # ---- setup training environment
    # Set Random Seed
    args = args_parse()
    rng_seed = args.seed

    init_seed(rng_seed)

    # ---- setup config files
    cfg.merge_from_file(args.cfg)
    cfg.DATA_ROOT = get_data_root()
    cfg.freeze()

    # ---- setup model
    # model = gns()
    # model.to(cuda) ...

    # ---- setup optimizer
    # ...

    # ---- if resume experiments, use --init ${model_name}
    if args.init:
        print(f'loading pretrained model from {args.init}')
        cp = torch.load(args.init)
        model.load_state_dict(cp['model'], False)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    # ...

    # ---- setup solver
    # kwargs


    # train


if __name__ == '__main__':

    main()
