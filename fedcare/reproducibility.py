"""
fedcare/reproducibility.py - Global deterministic seeding.

Ensures that all experiments (random splits, model initialization, 
data loading, and DP noise) are fully reproducible.
"""

import os
import random
import numpy as np
import torch

def seed_everything(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.
    
    Parameters
    ----------
    seed : int
        The seed value to use across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Force PyTorch to use deterministic algorithms if possible
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
