import torch
import numpy as np


def fix_seed(seed, cudnn_deterministic=True):
    """
Fixes the seed for pytorch, numpy, etc. If requested also makes
cuDNN deterministic (warning, might slow down training/inference).

Follows recommendations from https://pytorch.org/docs/stable/notes/randomness.html

Warning: Even all this does not ensure perfect determinism since there
is no way to make atomic operations from CUDA deterministic
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.backends.cudnn.enabled and cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
