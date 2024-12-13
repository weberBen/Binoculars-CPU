from .detector import Binoculars
import torch
import numpy as np
import random

torch.set_grad_enabled(False)

# Force deterministic behavior
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

__all__ = ["Binoculars"]
