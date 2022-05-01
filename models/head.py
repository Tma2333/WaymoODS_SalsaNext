

def get_salsa_head(num_features, num_cls):
    from .SalsaNext import SalsaNextHead
    return SalsaNextHead(num_features, num_cls)