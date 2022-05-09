

def get_salsa_head(num_features, num_cls):
    from .SalsaNext import SalsaNextHead
    return SalsaNextHead(num_features, num_cls)


def get_squeeze_head(num_features, num_cls):
    from .SqueezeSegv3 import SqueezeSegHead
    return SqueezeSegHead(num_features, num_cls)