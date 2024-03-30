import numpy as np


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

