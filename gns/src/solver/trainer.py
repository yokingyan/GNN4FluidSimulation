import torch
from torch.utils.tensorboard import SummaryWriter
import os


class Trainer:
    """
    The Base Class of Training Models.
        :parameter
    """
    def __init__(self, train_loader, val_loader, model, optim, max_iters, exp_name):

        # Basic Interface
        self.train_loader, self.val_loader = train_loader, val_loader
        self.model = model
        self.optimizer = optim

        # Train Parameters
        self.start_iters = 0
        self.max_iters = max_iters

        # Device Information
        self.exp_name = exp_name
        self.setup_device()

        # Init Dirs
        self.setup_dirs()

        # Init Loss

        # SummaryWriter
        self.tb_writer = SummaryWriter(self.log_dir)

    def setup_dirs(self):
        """Init Dirs."""
        self.log_dir = f'./logs/{self.exp_name}'
        self.ckpt_dir = f'./ckpts/{self.exp_name}'

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def setup_loss(self):
        pass

    def setup_device(self):
        """Set The Equipment of Train and Validation.([Need to Be Revised!])"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        """Base Training Step."""
        pass

    def train_epoch(self):
        pass

    def adjust_learning_rate(self):
        pass
