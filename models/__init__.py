"""
Traffic Prediction Models
==========================

Implementations of the five spatiotemporal backbones evaluated in the paper.
All models support 2-channel input (speed + acceleration).
"""

from .dcrnn_model import DCRNN
from .gwnet_model import GraphWaveNet
from .agcrn_model import AGCRN
from .stgin_model import STGIN
from .staeformer_model import STAEformer

__all__ = ['DCRNN', 'GraphWaveNet', 'AGCRN', 'STGIN', 'STAEformer']
