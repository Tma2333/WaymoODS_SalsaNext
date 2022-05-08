import torch.nn as nn

class BaseEncoder (nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    def _connection (self):
        # this shoule be a list of key for your connection in the forward output
        raise NotImplementedError()


def get_salsa_encoder(num_inchannel=6):
    from .SalsaNext import SalsaNextEncoder
    return SalsaNextEncoder(num_inchannel)


def get_squeeze_encoder(num_inchannel=6, drop_prob=0.25):
    from .SqueezeSegv3 import SqueezeSegEncoder
    return SqueezeSegEncoder(num_inchannel, drop_prob)

