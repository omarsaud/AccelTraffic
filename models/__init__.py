"""
Traffic Prediction Models
==========================

Implementations of standard traffic prediction models.
All models support 2-channel input (speed + acceleration).
"""

from .dcrnn_model import DCRNN
from .gwnet_model import GraphWaveNet
from .agcrn_model import AGCRN
from .stgin_full import STGIN

__all__ = ['DCRNN', 'GraphWaveNet', 'AGCRN', 'STGIN']
