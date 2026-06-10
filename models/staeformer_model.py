"""
STAEformer: Spatio-Temporal Adaptive Embedding Transformer
===========================================================

Reference:
    Liu et al., "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer 
    SOTA for Traffic Forecasting", CIKM 2023
    
Key Features:
    - Vanilla Transformer encoder (no GNN!)
    - Spatio-Temporal Adaptive Embeddings (learnable per-node, per-time)
    - Simple yet effective architecture
    - Supports 2-channel input (speed + acceleration)

Architecture:
    Input: (batch, nodes, seq_len, input_dim)
    Adaptive Embedding: Learnable spatial + temporal embeddings
    Transformer Encoder: Standard multi-head self-attention
    Output Projection: Linear layer to prediction horizon
    Output: (batch, nodes, horizon, output_dim)

Why STAEformer?
    - Shows that proper embeddings matter more than complex GNN architectures
    - Pure attention-based, no graph convolution
    - Competitive with DGCRN, D2STGNN, PDFormer
    - Great baseline for comparing acceleration enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatioTemporalAdaptiveEmbedding(nn.Module):
    """
    Spatio-Temporal Adaptive Embedding.
    
    Creates learnable embeddings for:
    1. Spatial: Per-node embedding (captures node-specific patterns)
    2. Temporal: Per-timestep embedding (captures time-of-day patterns)
    
    These are ADDED to input features, enabling the vanilla transformer
    to implicitly learn spatial and temporal relationships.
    """
    
    def __init__(self, num_nodes, seq_len, d_model):
        """
        Args:
            num_nodes: Number of nodes in graph
            seq_len: Input sequence length (historical window)
            d_model: Model dimension
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.d_model = d_model
        
        # Learnable spatial embedding: (num_nodes, d_model)
        self.spatial_embed = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)
        
        # Learnable temporal embedding: (seq_len, d_model)
        self.temporal_embed = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
    
    def forward(self, x):
        """
        Add adaptive embeddings to input.
        
        Args:
            x: (batch, nodes, seq_len, d_model)
        
        Returns:
            x + spatial_embed + temporal_embed
        """
        batch = x.size(0)
        
        # Expand spatial embedding: (1, nodes, 1, d_model) -> broadcast
        spatial = self.spatial_embed.unsqueeze(0).unsqueeze(2)  # (1, N, 1, D)
        
        # Expand temporal embedding: (1, 1, seq_len, d_model) -> broadcast
        temporal = self.temporal_embed.unsqueeze(0).unsqueeze(1)  # (1, 1, T, D)
        
        # Add embeddings
        return x + spatial + temporal


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer Encoder Layer.
    
    Pre-norm architecture for better training stability.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq, d_model)
        """
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Pre-norm feed-forward
        x_norm = self.norm2(x)
        x = x + self.ff(x_norm)
        
        return x


class STAEformer(nn.Module):
    """
    Spatio-Temporal Adaptive Embedding Transformer.
    
    A vanilla transformer with adaptive spatio-temporal embeddings
    that achieves SOTA performance without explicit graph convolutions.
    """
    
    def __init__(self, 
                 num_nodes,
                 input_dim=1,
                 output_dim=1,
                 d_model=64,
                 num_heads=4,
                 num_layers=3,
                 d_ff=256,
                 dropout=0.3,
                 seq_len=12,
                 horizon=12):
        """
        Args:
            num_nodes: Number of nodes in graph
            input_dim: Input feature dimension (1=speed, 2=speed+accel)
            output_dim: Output feature dimension
            d_model: Model dimension (hidden_dim)
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            d_ff: Feed-forward dimension (default: 4*d_model)
            dropout: Dropout rate
            seq_len: Input sequence length (H)
            horizon: Prediction horizon (Q)
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Input projection: input_dim -> d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Spatio-Temporal Adaptive Embedding
        self.st_embed = SpatioTemporalAdaptiveEmbedding(num_nodes, seq_len, d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection: seq_len * d_model -> horizon * output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(seq_len * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon * output_dim)
        )
        
        # Layer norm before output
        self.final_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, adj=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, nodes, seq_len, input_dim)
            adj: Adjacency matrix (ignored - STAEformer doesn't use graph!)
        
        Returns:
            predictions: (batch, nodes, horizon, output_dim)
        """
        batch, nodes, seq_len, in_dim = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, N, T, D)
        
        # Add spatio-temporal adaptive embeddings
        x = self.st_embed(x)  # (B, N, T, D)
        
        # Reshape for transformer: treat each node independently
        # (B, N, T, D) -> (B*N, T, D)
        x = x.reshape(batch * nodes, seq_len, self.d_model)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Final layer norm
        x = self.final_norm(x)  # (B*N, T, D)
        
        # Flatten temporal dimension for output projection
        x = x.reshape(batch * nodes, seq_len * self.d_model)  # (B*N, T*D)
        
        # Output projection
        x = self.output_proj(x)  # (B*N, Q*out_dim)
        
        # Reshape to output format
        x = x.reshape(batch, nodes, self.horizon, self.output_dim)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class STAEformerWithGraph(STAEformer):
    """
    STAEformer variant that optionally uses graph information.
    
    Adds a simple graph attention layer on top of vanilla STAEformer.
    For ablation study: compare STAEformer vs STAEformerWithGraph.
    """
    
    def __init__(self, *args, use_graph=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_graph = use_graph
        
        if use_graph:
            # Simple graph attention (optional enhancement)
            self.graph_attn = nn.MultiheadAttention(
                self.d_model, 
                num_heads=4, 
                dropout=0.1,
                batch_first=True
            )
            self.graph_norm = nn.LayerNorm(self.d_model)
    
    def forward(self, x, adj=None):
        """
        Forward with optional graph attention.
        """
        batch, nodes, seq_len, in_dim = x.shape
        
        # Input projection
        x = self.input_proj(x)  # (B, N, T, D)
        
        # Add spatio-temporal adaptive embeddings
        x = self.st_embed(x)  # (B, N, T, D)
        
        # Optional: Graph attention across nodes (spatial)
        if self.use_graph and adj is not None:
            # Aggregate temporal: (B, N, T, D) -> (B, N, D)
            x_spatial = x.mean(dim=2)  # (B, N, D)
            
            # Graph attention
            x_graph, _ = self.graph_attn(x_spatial, x_spatial, x_spatial)
            x_graph = self.graph_norm(x_spatial + x_graph)  # (B, N, D)
            
            # Add back to temporal: (B, N, 1, D)
            x = x + x_graph.unsqueeze(2)
        
        # Reshape for transformer
        x = x.reshape(batch * nodes, seq_len, self.d_model)
        
        # Apply transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)
        
        x = self.final_norm(x)
        x = x.reshape(batch * nodes, seq_len * self.d_model)
        x = self.output_proj(x)
        x = x.reshape(batch, nodes, self.horizon, self.output_dim)
        
        return x


# Quick test
if __name__ == '__main__':
    # Test STAEformer
    batch, nodes, seq_len, input_dim = 32, 207, 12, 2
    horizon = 12
    
    model = STAEformer(
        num_nodes=nodes,
        input_dim=input_dim,
        output_dim=1,
        d_model=64,
        num_heads=4,
        num_layers=3,
        seq_len=seq_len,
        horizon=horizon
    )
    
    x = torch.randn(batch, nodes, seq_len, input_dim)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {model.count_parameters():,}")
