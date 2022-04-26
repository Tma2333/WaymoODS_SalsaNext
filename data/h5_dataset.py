import logging
from pathlib import Path

import torch
import torchvision.transforms as T
import numpy as np


class H5Dataset(torch.utils.data.Dataset):
    def __init__ (self,):
        raise NotImplementedError


    def __len__ (self):
        raise NotImplementedError


    def __getitem__ (self, index):
        raise NotImplementedError
    

    def get_data (self, path, ):
        pass