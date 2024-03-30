"""The Basics Dataset."""
import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import pickle
from torch.utils.data import Dataset
from load_data_info import read_metadata
import torch
import numpy as np
from glob import glob

from config.configs import _C as C
from utils.tools import get_random_walk_noise, time_diff


def get_non_kinematic_mask(particle_types):
    """
    Returns a boolean mask, set to true for kinematic (obstacle) particles.
    """

    return particle_types != C.KINEMATIC_PARTICLE_ID


class DatasetBase(Dataset):
    """The Base Class of Datasets."""
    def __init__(self, data_dir, noise_func, time_diff_style, phase='train'):

        self.data_dir = data_dir
        self.phase = phase

        # Get metadata information
        self.metadata = read_metadata(os.path.join(self.data_dir, '..'))

        # Write to Dic
        for key in self.metadata:
            self.metadata[key] = torch.from_numpy(np.array(self.metadata[key]).astype(np.float32))

        if self.phase == 'val':
            self.pred_steps = C.ROLLOUT_STEPS - C.N_HIS
        else:
            self.pred_steps = C.PRED_STEPS

        # Other Changeable params
        # Add Noise
        assert noise_func is not None, "Please Set the Function of Add Noise!!!"
        self.set_noise_func = noise_func

        # Time Difference
        assert time_diff_style is not None, "Please Specified the Mode of Time Difference!!!"
        self.time_diff = time_diff_style

    def __len__(self):
        """The Size of Dataset."""
        num_vids = len(glob(os.path.join(self.data_dir, '*')))  # return the num of file

        if self.phase == 'val':
            num_vids = min(num_vids, C.MAX_VAL)

        return num_vids * num_vids * (C.ROLLOUT_STEPS - C.N_HIS - self.pred_steps + 1)

    def load_data_file(self, idx_rollout):
        file = os.path.join(self.data_dir, f'{idx_rollout}.pkl')
        data = pickle.load(open(file, 'rb'))
        data['position'] = data['position'].transpose([1, 0, 2])
        return data

    def __getitem__(self, idx):
        """Get the data according to the idx."""

        # Compute the pkl file index.
        idx_rollout = idx // (C.ROLLOUT_STEPS - C.N_HIS - self.pred_steps + 1)
        # Compute the predict index.
        idx_timestep = (C.N_HIS - 1) + idx % (C.ROLLOUT_STEPS - C.N_HIS - self.pred_steps + 1)
        self.idx_time_step = idx_timestep

        # Get the data according to idx
        data = self.load_data_file(idx_rollout)

        poss = data['position'][:, idx_timestep-C.N_HIS+1:idx_timestep+1]
        tgt_poss = data['position'][:, idx_timestep+1:idx_timestep+self.pred_steps+1]

        nonk_mask = get_non_kinematic_mask(data['particle_type'])

        # Inject random walk noise
        if self.phase == 'train':
            sampled_noise = self.set_noise_func(poss, idx_timestep, C.NET.NOISE)
            poss = poss + sampled_noise

            tgt_poss = tgt_poss + sampled_noise[:, -1:]

        # Difference
        tgt_vels = self.time_diff(np.concatenate([poss, tgt_poss], axis=1))
        tgt_accs = self.time_diff(tgt_vels)

        tgt_vels = tgt_vels[:, -self.pred_steps:]
        tgt_accs = tgt_accs[:, -self.pred_steps:]

        poss = torch.from_numpy(poss.astype(np.float32))
        tgt_vels = torch.from_numpy(tgt_vels.astype(np.float32))
        tgt_accs = torch.from_numpy(tgt_accs.astype(np.float32))
        particle_type = torch.from_numpy(data['particle_type'])
        nonk_mask = torch.from_numpy(nonk_mask.astype(np.int32))
        tgt_poss = torch.from_numpy(tgt_poss.astype(np.float32))

        return poss, tgt_accs, tgt_vels, particle_type, nonk_mask, tgt_poss


class WaterRamps(DatasetBase):
    """This is a 2D Dataset named WaterRamps."""
    def __init__(self, data_dir, noise_add=get_random_walk_noise, time_diff_style=time_diff, phase='train'):
        super(WaterRamps, self).__init__(
            data_dir=data_dir,
            phase=phase,
            noise_func=noise_add,
            time_diff_style=time_diff_style
        )


if __name__ == '__main__':

    # Test
    dataset = WaterRamps(r"test\test_dataset\3D\test")
    print(len(dataset))
