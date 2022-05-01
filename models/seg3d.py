import torch.nn as nn

from .encoder import (BaseEncoder,
                      get_salsa_encoder)
from .decoder import (BaseDecoder,
                      get_salsa_decoder)
from .head import get_salsa_head


_AVAILABLE_ENCODER = {'Salsa': get_salsa_encoder}
_AVAILABLE_DECODER = {'Salsa': get_salsa_decoder}
_AVAILABLE_HEAD = {'Salsa': get_salsa_head}


class SphericalSegmentation (nn.Module):
    def __init__(self, parms):
        super().__init__()

        # check availability
        if parms['encoder'] not in _AVAILABLE_ENCODER:
            raise ValueError(f'{parms["encoder"]} is not supported')
        if parms['decoder'] not in _AVAILABLE_DECODER:
            raise ValueError(f'{parms["decoder"]} is not supported')
        if parms['head'] not in _AVAILABLE_HEAD:
            raise ValueError(f'{parms["head"]} is not supported')
        
        self.encoder = _AVAILABLE_ENCODER[parms['encoder']](**parms['encoder_parm'])
        self.decoder = _AVAILABLE_DECODER[parms['decoder']](**parms['decoder_parm'])
        self.head = _AVAILABLE_HEAD[parms['head']](**parms['head_parm'])
        
        # check for base class:
        if not issubclass(type(self.encoder), BaseEncoder):
            raise TypeError(f'{type(self.encoder)} is not supported')
        if not issubclass(type(self.decoder), BaseDecoder):
            raise TypeError(f'{type(self.decoder)} is not supported')
        
        # check compatibility
        if self.encoder._connection != self.decoder._connection:
            raise AttributeError(f'{type(self.encoder)} has {self.encoder._connection} forward connection,\n \
                                   but {type(self.decoder)} receive {self.decoder._connection} connection.')

    
    def forward(self, x):
        conn = self.encoder(x)
        out = self.decoder(conn)
        pred = self.head(out)

        return pred

