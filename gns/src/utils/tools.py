import socket
import re
import numpy as np
import random
import torch
import getpass
import yaml
from src.config.configs import _C as C


# define the decorator
def get_noise(func):
    def set_noise_type(*args, **kwargs):
        sampled_noise = func(*args, **kwargs)
        return sampled_noise
    return set_noise_type


@get_noise
def get_random_walk_noise(pos_seq, idx_timestep, noise_std):
    noise_shape = (pos_seq.shape[0], pos_seq.shape[1]-1, pos_seq.shape[2])
    n_step_vel = noise_shape[1]
    acc_noise = np.random.normal(0, noise_std / n_step_vel ** 0.5, size=noise_shape).astype(np.float32)
    vel_noise = np.cumsum(acc_noise, axis=1)
    pos_noise = np.cumsum(vel_noise, axis=1)
    pos_noise = np.concatenate([np.zeros_like(pos_noise[:, :1]),
                                pos_noise], axis=1)

    return pos_noise


# [check!! More Python Style]
def time_diff(input_seq):
    return input_seq[:, 1:] - input_seq[:, :-1]


# Set Random Seed
def init_seed(rng_seed=0):
    """Initialize Random Seed."""
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic=True
        torch.cuda.manual_seed(0)
    else:
        raise NotImplementedError


def get_data_root():
    hostname = socket.gethostname()
    username = getpass.getuser()
    paths_yaml_fn = 'configs/paths.yaml'
    with open(paths_yaml_fn, 'r') as f:
        paths_config = yaml.load(f, Loader=yaml.Loader)

    for hostname_re in paths_config:
        if re.compile(hostname_re).match(hostname) is not None:
            for username_re in paths_config[hostname_re]:
                if re.compile(username_re).match(username) is not None:
                    return paths_config[hostname_re][username_re]['data_dir']

    raise Exception('No matching hostname or username in config file')
