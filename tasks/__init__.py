from collections import OrderedDict

import torch

from .segmentation3d import Segmentation3DTask


def get_task(args):
    if args.get("task") == "Segmentation3DTask":
        return Segmentation3DTask(args)
    else:
        raise NotImplementedError('Task not implemented.')


def load_task(ckpt_path, **kwargs):
    args = torch.load(ckpt_path, map_location='cpu')['hyper_parameters']
    if args.get("task") == "Segmentation3DTask":
        task = Segmentation3DTask
    else:
        raise NotImplementedError('Task not implemented.')
    return task.load_from_checkpoint(ckpt_path)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = ckpt['hyper_parameters']
    state_dict = ckpt['state_dict']

    model_state_dict = {}
    for key, state in state_dict.items():
        key = key.split('.')
        model = key[0]
        layer = '.'.join(key[1:])
        if model not in model_state_dict:
            model_state_dict[model] = OrderedDict()
        model_state_dict[model][layer] = state.clone()
    
    return model