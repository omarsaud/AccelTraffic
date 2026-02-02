"""
Miscellaneous utility functions for reproducibility and logging.
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set all random seeds for full reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility
    print(f"🎲 Random seed set to {seed} for reproducibility")
