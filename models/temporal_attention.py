from .global_configuration import DROPOUT
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(TemporalAttention, self).__init__()
        self.attention = nn.MultiheadAttention(in_dim, num_heads, dropout=DROPOUT)
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        # x shape: (batch, nodes, P, HIDDEN_DIM)

        batch, nodes, time, in_dim = x.shape

        x = x.permute(2, 0, 1, 3).reshape(time, batch * nodes, in_dim)
        # x shape: (P, batch * nodes, HIDDEN_DIM)

        attn_output, _ = self.attention(x, x, x)
        h = self.fc(attn_output)
        
        # FIXED: Residual connection to preserve temporal information
        h = h + x

        # Reshape back to (batch, nodes, time, in_dim) without BatchNorm inside attention
        return h.view(time, batch, nodes, -1).permute(1, 2, 0, 3)  # (batch, nodes, time, in_dim)