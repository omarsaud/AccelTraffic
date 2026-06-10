"""
Shared hyperparameter constants for the STGIN model components.

Kept minimal and side-effect-free (no prints, no GPU configuration on import)
so importing any model is safe in every environment. GPU/runtime optimizations
live in ``utils/global_configuration.py`` and are applied by the training
scripts, not at model-import time.
"""

DROPOUT = 0.3
