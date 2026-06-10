"""
Model Factory for Traffic Prediction
=====================================

Creates instances of different traffic prediction models.
All models support 2-channel input (speed + acceleration).
"""

import torch
import torch.nn as nn

def create_model(model_name, num_nodes, input_dim=1, output_dim=1, 
                 hidden_dim=64, historical_window=12, prediction_horizon=3,
                 dropout=0.3, **kwargs):
    """
    Create a traffic prediction model.
    
    Args:
        model_name: str - Model name ('stgin', 'dcrnn', 'gwnet', 'agcrn')
        num_nodes: int - Number of nodes in graph
        input_dim: int - Input dimension (1=speed only, 2=speed+accel)
        output_dim: int - Output dimension (1 for speed prediction)
        hidden_dim: int - Hidden dimension
        historical_window: int - Input sequence length
        prediction_horizon: int - Output sequence length
        dropout: float - Dropout rate
    
    Returns:
        model: nn.Module - Traffic prediction model
    """
    
    model_name = model_name.lower()
    
    if model_name == 'stgin':
        # Note: STGIN has a different interface (needs STE embeddings)
        # Use testing_withenhancement.py for STGIN experiments
        raise NotImplementedError(
            "STGIN uses a different interface with spatiotemporal embeddings. "
            "Please use 'train_stgin.py' for STGIN experiments. The multi-model "
            "script (train_multimodel.py) supports DCRNN, AGCRN, Graph WaveNet, and STAEformer."
        )
        
    elif model_name == 'dcrnn':
        from .dcrnn_model import DCRNN
        model = DCRNN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            K=2,  # Diffusion steps
            dropout=dropout,
            seq_len=historical_window,
            horizon=prediction_horizon
        )
    
    elif model_name == 'gwnet':
        from .gwnet_model import GraphWaveNet
        # ⚡ VECTORIZED GCN: Now processes all time steps in parallel (not sequential)
        # This fixes the ~52 min/epoch issue - should now be ~1-3 min/epoch
        model = GraphWaveNet(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim  ,  # Same as other models (64)
            num_layers=2,           # 2 ST-Conv blocks (reasonable depth)
            kernel_size=2,
            dropout=dropout,
            seq_len=historical_window,
            horizon=prediction_horizon,
            support_len=2,  # Fixed + adaptive adjacency
            embed_dim=10
        )
    
    elif model_name == 'agcrn':
        from .agcrn_model import AGCRN
        model = AGCRN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            embed_dim=10,
            cheb_k=2,
            dropout=dropout,
            seq_len=historical_window,
            horizon=prediction_horizon
        )
    
    elif model_name == 'staeformer':
        from .staeformer_model import STAEformer
        # Vanilla transformer with spatio-temporal adaptive embeddings.
        # Same (batch, nodes, seq_len, input_dim) -> (batch, nodes, horizon, output_dim)
        # interface as the other backbones; the adjacency argument is ignored.
        model = STAEformer(
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=hidden_dim,
            num_heads=4,
            num_layers=3,
            d_ff=256,
            dropout=dropout,
            seq_len=historical_window,
            horizon=prediction_horizon
        )

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Choose from: stgin, dcrnn, gwnet, agcrn, staeformer")

    return model


def get_model_info(model_name):
    """
    Get information about a model.
    
    Args:
        model_name: str - Model name
    
    Returns:
        dict - Model information
    """
    
    info = {
        'stgin': {
            'name': 'STGIN',
            'full_name': 'Spatial-Temporal Graph Informed Network',
            'year': 2022,
            'type': 'GCN + Transformer',
            'description': 'Graph + attention model; trained via train_stgin.py (uses spatiotemporal embeddings)',
            'implemented': True,
            'note': 'Different interface - needs spatiotemporal embeddings',
            'citation_count': '~50'
        },
        'dcrnn': {
            'name': 'DCRNN',
            'full_name': 'Diffusion Convolutional Recurrent Neural Network',
            'year': 2018,
            'type': 'Diffusion + GRU',
            'description': 'Standard baseline, diffusion convolution with encoder-decoder',
            'implemented': True,
            'citation_count': '~2,500'
        },
        'gwnet': {
            'name': 'Graph WaveNet',
            'full_name': 'Graph WaveNet for Spatial-Temporal Modeling',
            'year': 2019,
            'type': 'TCN + Adaptive GCN',
            'description': 'Temporal convolution with adaptive graph learning',
            'implemented': True,
            'citation_count': '~1,500'
        },
        'agcrn': {
            'name': 'AGCRN',
            'full_name': 'Adaptive Graph Convolutional Recurrent Network',
            'year': 2020,
            'type': 'Adaptive GCN + GRU',
            'description': 'Node-adaptive graph convolution, efficient',
            'implemented': True,
            'citation_count': '~400'
        },
        'staeformer': {
            'name': 'STAEformer',
            'full_name': 'Spatio-Temporal Adaptive Embedding Transformer',
            'year': 2023,
            'type': 'Transformer + Adaptive Embeddings',
            'description': 'Vanilla transformer with learnable spatio-temporal embeddings (no graph convolution)',
            'implemented': True,
            'citation_count': 'CIKM 2023'
        }
    }
    
    return info.get(model_name.lower(), None)


def list_available_models():
    """List all available models"""
    models = ['dcrnn', 'gwnet', 'agcrn', 'staeformer', 'stgin']
    
    print("="*80)
    print("AVAILABLE TRAFFIC PREDICTION MODELS")
    print("="*80)
    
    for model_name in models:
        info = get_model_info(model_name)
        status = "✅ Implemented" if info['implemented'] else "⏳ Coming soon"
        print(f"\n{info['name']} ({info['year']}) - {status}")
        print(f"  Type: {info['type']}")
        print(f"  Citations: {info['citation_count']}")
        print(f"  {info['description']}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    # Test model factory
    list_available_models()
