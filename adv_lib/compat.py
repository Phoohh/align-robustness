"""Import compatibility for AdverTorch 0.2.3 on modern PyTorch/torchvision.

These aliases are needed by AdverTorch's eager imports, not by its PGD computation.
"""
import importlib


def prepare_advertorch():
    import torch
    gradcheck = importlib.import_module('torch.autograd.gradcheck')
    if not hasattr(gradcheck, 'zero_gradients'):
        def zero_gradients(x):
            if isinstance(x, torch.Tensor):
                if x.grad is not None:
                    x.grad.detach_(); x.grad.zero_()
            else:
                for y in x:
                    zero_gradients(y)
        gradcheck.zero_gradients = zero_gradients
    import torchvision.datasets.mnist as mnist
    import torchvision.datasets.utils as utils
    for name in ('download_url', 'makedir_exist_ok'):
        if not hasattr(mnist, name) and hasattr(utils, name):
            setattr(mnist, name, getattr(utils, name))
