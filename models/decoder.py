import torch.nn as nn

class BaseDecoder (nn.Module):
    def __init__(self):
        super().__init__()
    
    @property
    def _connection (self):
        # this shoule be a list of key for the connection in the forward input
        raise NotImplementedError()


def get_salsa_decoder():
    from .SalsaNext import SalsaNextDecoder
    return SalsaNextDecoder()