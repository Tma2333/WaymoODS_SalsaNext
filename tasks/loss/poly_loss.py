# reference paper: https://arxiv.org/pdf/2204.12511.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def compute_poly_n_corss_entropy_loss (inputs, targets, n, eps, weight=None, use_logit=False, ignore_index=None, reduction='mean'):
    """
    Poly N corss entropy loss:
        input: Tensor | [N, C, d1, d2, d3, ..., dk] could be either class probability or raw unnormalized logit, if use probability, 
            use _logit must be true. 
        targets: Tensor |  [N, d1, d2, d3, ..., dk] class label where 0 <= targets[i] <= C-1
        n: int | n'th order poly nomial expansion
        eps: List | list of perturbed coefficient must have same length as n
        weight: Tensor | [C], weight uses for weighted Cross Entropy loss
        use_logit: Boolean | True if input is logit and False if input is probability, If use logit, the inputs will pass through a pytorch 
            softmax first with minimum value calmp to 1e-8 as implemented in torch.nn.CrossEntropyLoss
        ignore_index: int | class index to ignore. This will pass into nll_loss and set expansion probability manually to 1 to kill 
            gradient flow for that class.
        reduction: str | Reduction method. You must choose from ['none', 'mean', 'sum'], if ignore index is also given, 'mean' will average 
            over number of non-ignore instance. 
    """
    if len(eps) != n:
        raise ValueError(f'Length of eps: {len(eps)} does not match with n: {n}')

    n_class = inputs.shape[1]
    
    if ignore_index is None:
        # default pytorch value
        nll_ignore_index = -100
    else:
        if not isinstance(ignore_index, int):
            raise TypeError(f'ignore_index must be a int, but got {type(ignore_index)}')
        nll_ignore_index = ignore_index

    if use_logit:
        probas = F.softmax(inputs, dim=1)
    else:
        probas = inputs
        
    CE = F.nll_loss(torch.log(probas.clamp(min=1e-8)), targets, weight=weight, ignore_index=nll_ignore_index, reduction='none')

    targets_one_hot = F.one_hot(targets, n_class).movedim(-1, 1).type_as(probas)
    
    Pt = targets_one_hot * probas
    # NOTE:
    # Not in original paper, might need more check and consideration
    # Ignore index by setting Pt to 1, killing the gradient flow and set loss to 0
    if ignore_index is not None:
        Pt[targets==ignore_index, ignore_index, ...] = 1.0
    Pt = torch.sum(Pt, dim=1)

    poly_exp_loss = torch.zeros_like(Pt).type_as(Pt)
    for i in range(n):
        poly_exp_loss += eps[i] * (1 - Pt).pow(i+1)
    
    loss = CE + poly_exp_loss
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'mean':
        if ignore_index is None:
            return torch.mean(loss)
        else:
            return torch.sum(loss) / torch.sum(loss!=0)
    else:
        raise ValueError(f'{reduction} is not a valid value for reduction')
    

class Poly1XentropyLoss (nn.Module):
    def __init__(self, eps=1, weight=None, use_logit=False, ignore_index=None, reduction='mean'):
        super().__init__()

        self.eps = eps
        self.register_buffer('weight', weight, persistent=False)
        self.use_logit=use_logit
        self.ignore_index = ignore_index
        self.reduction = reduction


    def forward(self, inputs, targets):
        return compute_poly_n_corss_entropy_loss(inputs, targets, n=1, 
                                                 eps=[self.eps], weight=self.weight, 
                                                 use_logit=self.use_logit,
                                                 ignore_index = self.ignore_index,
                                                 reduction=self.reduction)


class PolyNXentropyLoss (nn.Module):
    def __init__(self, n, eps, weight=None, use_logit=False, ignore_index=None, reduction='mean'):
        super().__init__()

        self.n = n
        self.eps = eps
        self.register_buffer('weight', weight, persistent=False)
        self.use_logit=use_logit
        self.ignore_index = ignore_index
        self.reduction = reduction


    def forward(self, inputs, targets):
        return compute_poly_n_corss_entropy_loss(inputs, targets, n=self.n, 
                                                 eps=self.eps, weight=self.weight, 
                                                 use_logit=self.use_logit, 
                                                 ignore_index = self.ignore_index,
                                                 reduction=self.reduction)