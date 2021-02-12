import os
import torch
from torch import nn


class SubModuleBase(nn.Module):
    def __init__(self, save_file):
        super().__init__()
        self.save_file = save_file

    def try_load(self):
        if os.path.exists(self.save_file):
            self.load_model()

    def load_model(self):
        state = torch.load(self.save_file)
        self.load_state_dict(state)

    def save_model(self):
        torch.save(self.state_dict(), self.save_file)
