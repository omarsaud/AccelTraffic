from .gat import MultiHeadGAT
from .gcn import MultiHeadGCN
from .temporal_attention import TemporalAttention
import torch.nn as nn
import torch

class STBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(STBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Input projection if dimensions don't match
        self.input_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        self.gcn = MultiHeadGCN(out_dim, out_dim)  # Now use out_dim for both
        # GAT operates on spatially mixed features of dim=out_dim
        # Enable STE in GAT for time-aware spatial attention (matches article)
        self.gat = MultiHeadGAT(out_dim, out_dim, use_ste=True)
        self.feed_spatial = nn.Linear(out_dim * 2, out_dim)

        self.temp_attn = TemporalAttention(out_dim)
        
        # FIXED (optional): Light projection for STE before LSTM concat (numerical stability)
        self.ste_proj = nn.Linear(out_dim, out_dim)
        
        # Single-direction LSTM (no BiLSTM option)
        self.lstm = nn.LSTM(out_dim * 2, out_dim, 1, bidirectional=False, batch_first=True)

        self.fc_fusion = nn.Linear(out_dim * 2, out_dim)
        self.feed_temporal = nn.Linear(out_dim, out_dim)

        # NO BatchNorm (paper doesn't use it)

    def forward(self, x, adj, ste):
        # x : (batch, 207, P, in_dim)
        # adj : (207, 207)
        # ste : (batch, 207, P, HIDDEN_DIM)
        batch, nodes, time, in_dim = x.shape
        
        # Project input to output dimensions
        x_proj = self.input_proj(x)  # (batch, nodes, time, out_dim)

        hps = self.gcn(x_proj, adj)  # (batch, nodes, time, out_dim)
        # Now we can add residual connection
        hps = hps + x_proj
        # True spatial GAT over neighbors using adjacency on spatial features
        # Pass STE to GAT for time-aware spatial attention (matches article)
        hds = self.gat(hps, adj, ste)  # (batch, nodes, time, out_dim)
        hs = torch.cat([hps, hds], dim=-1)  # (batch, nodes, time, out_dim * 2)

        # Apply spatial processing (NO BatchNorm - paper doesn't use it)
        hs = self.feed_spatial(hs)  # (batch, nodes, time, out_dim)

        # Temporal modules consume raw input (PARALLEL processing, matches benchmark)
        hdt = self.temp_attn(x_proj)  # (batch, nodes, time, out_dim)
        
        # FIXED (optional): Project STE for numerical stability before concatenation
        ste_projected = self.ste_proj(ste)
        hit, _ = self.lstm(torch.cat([x_proj, ste_projected], dim=-1).reshape(batch * nodes, time, -1))  # (batch, nodes, time, out_dim)
        hit = hit.reshape(batch, nodes, time, -1)

        ht = hdt + hit # (batch, nodes, time, out_dim)

        # Apply temporal processing (NO BatchNorm - paper doesn't use it)
        ht = self.feed_temporal(ht)  # (batch, nodes, time, out_dim)

        h_concat = torch.cat([hs, ht], dim=-1)  # (batch, nodes, time, out_dim * 2)
        hst = self.fc_fusion(h_concat)  # (batch, nodes, time, out_dim)
        gate = torch.sigmoid(hst)
        return gate * hs + (1 - gate) * ht