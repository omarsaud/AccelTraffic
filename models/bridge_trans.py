import torch
import torch.nn as nn
from .global_configuration import DROPOUT

class BridgeTrans(nn.Module):
    """
    Multi-layer transformer decoder for bridging history to future predictions.
    
    Args:
        in_dim: Input dimension (hidden_dim)
        out_dim: Output dimension (hidden_dim)
        num_layers: Number of transformer decoder layers (1=baseline, 3=enhanced)
        num_heads: Number of attention heads
    
    Key Innovation:
        - Progressive refinement through multiple decoder layers
        - Horizon-aware positional encoding
        - Reduces prediction degradation at long horizons
    """
    def __init__(self, in_dim, out_dim, num_layers=1, num_heads=4):  # Paper uses 4 heads!
        super(BridgeTrans, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        
        if num_layers == 1:
            # BASELINE: Single-layer attention (original implementation)
            self.attention = nn.MultiheadAttention(in_dim, num_heads, dropout=DROPOUT)
            self.fc = nn.Linear(in_dim, out_dim)
        else:
            # ENHANCED: Multi-layer transformer decoder
            self.decoder_layers = nn.ModuleList([
                nn.TransformerDecoderLayer(
                    d_model=in_dim,
                    nhead=num_heads,
                    dim_feedforward=in_dim * 4,
                    dropout=DROPOUT,
                    activation='gelu',
                    batch_first=False,
                    norm_first=True  # Pre-layer normalization for stability
                ) for _ in range(num_layers)
            ])
            
            # Final normalization and projection
            self.norm = nn.LayerNorm(in_dim)
            self.fc_out = nn.Linear(in_dim, out_dim)
            
            # Learnable horizon-aware positional encoding
            # Each future timestep (t+1 to t+12) gets unique encoding
            self.horizon_pe = nn.Parameter(torch.randn(12, 1, in_dim) * 0.02)

    def forward(self, hist, ste):
        """
        Args:
            hist: (batch, nodes, P, in_dim) - Historical features from ST-Blocks
            ste: (batch, nodes, Q, hidden_dim) - Future spatio-temporal embeddings
        
        Returns:
            output: (batch, nodes, Q, out_dim) - Decoded future features
        """
        batch, nodes, P, _ = hist.shape
        _, _, Q, _ = ste.shape

        # Reshape to (sequence_length, batch_size * num_nodes, embedding_dim)
        hist = hist.permute(2, 0, 1, 3).reshape(P, batch * nodes, -1)  # (P, B*N, hidden)
        ste = ste.permute(2, 0, 1, 3).reshape(Q, batch * nodes, -1)    # (Q, B*N, hidden)

        if self.num_layers == 1:
            # BASELINE: Single-layer attention
            attn_output, _ = self.attention(ste, hist, hist)  # query, key, value
            
            # Apply FC and residual
            h = self.fc(attn_output)
            h = h + attn_output  # Residual connection
            h = h.reshape(Q, batch, nodes, -1).permute(1, 2, 0, 3)  # (batch, nodes, Q, out_dim)
        else:
            # ENHANCED: Multi-layer transformer decoder with progressive refinement
            # Add horizon-aware positional encoding to query
            query = ste + self.horizon_pe[:Q, :, :]  # (Q, B*N, hidden)
            
            # Progressive decoding through layers
            for layer in self.decoder_layers:
                # Cross-attention: query=future STE, memory=history
                query = layer(query, hist)
            
            # Final normalization and projection
            output = self.norm(query)
            output = self.fc_out(output)
            
            # Reshape back to (batch, nodes, Q, out_dim)
            h = output.permute(1, 0, 2).reshape(batch, nodes, Q, -1)

        return h