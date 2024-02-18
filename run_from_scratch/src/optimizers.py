import numpy as np
import torch


def get_optimizer(opt_dict, params):
    """
    opt: name or dict
    params: parameters to optimize
    """
    opt_name = opt_dict['name']
    if opt_name == "adam":
        opt = torch.optim.Adam(params, lr=opt_dict.get('lr', 1e-3))
    elif opt_name == 'sgd':
        opt = torch.optim.SGD(params, lr=opt['lr'])

    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt

