"""
Graph Attention over spatial neighbors using the adjacency matrix.

Inputs:
  x:   (batch, nodes, time, in_dim)
  adj: (nodes, nodes)  adjacency (binary or weighted). Non-positives are treated as masked.

Output:
  (batch, nodes, time, out_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .global_configuration import DROPOUT


class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, alpha=0.2, use_ste=False):
        super(MultiHeadGAT, self).__init__()
        self.use_ste = use_ste
        # If using STE, input dimension doubles (features + STE concatenated)
        self.actual_in_dim = in_dim * 2 if use_ste else in_dim
        
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        # Per-head linear projections and attention vectors (use actual_in_dim)
        self.W = nn.ModuleList([nn.Linear(self.actual_in_dim, self.head_dim, bias=False) for _ in range(num_heads)])
        self.a_src = nn.ModuleList([nn.Linear(self.head_dim, 1, bias=False) for _ in range(num_heads)])
        self.a_dst = nn.ModuleList([nn.Linear(self.head_dim, 1, bias=False) for _ in range(num_heads)])

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x, adj, ste=None):
        # x: (B, N, T, Fin), adj: (N, N), ste: (B, N, T, hidden_dim)
        B, N, T, Fin = x.shape
        
        # Concatenate STE if provided (enables time-aware spatial attention)
        if self.use_ste and ste is not None:
            x = torch.cat([x, ste], dim=-1)  # (B, N, T, Fin + hidden_dim)
            assert x.shape[-1] == self.actual_in_dim, f"Expected dim {self.actual_in_dim}, got {x.shape[-1]}"
        else:
            assert Fin == self.actual_in_dim, f"Expected dim {self.actual_in_dim}, got {Fin}"

        # Prepare adjacency mask once
        # Create (1, 1, N, N) broadcastable mask for (B, T, N, N)
        adj_mask = (adj > 0).float().unsqueeze(0).unsqueeze(0)  # 1x1xNxN

        head_outputs = []
        # Compute per-head attention and aggregation per time step
        for k in range(self.num_heads):
            Wh = self.W[k](x)  # (B, N, T, F')

            # Attention logits e_ij using the "trick": a_src(Wh_i) + a_dst(Wh_j)
            e_src = self.a_src[k](Wh)  # (B, N, T, 1)
            e_dst = self.a_dst[k](Wh)  # (B, N, T, 1)

            e_src_btni = e_src.permute(0, 2, 1, 3)  # (B, T, N, 1)
            e_dst_btnj = e_dst.permute(0, 2, 3, 1)  # (B, T, 1, N)
            e = e_src_btni + e_dst_btnj  # (B, T, N, N)
            e = self.leakyrelu(e)

            # Mask with adjacency: set -inf where no edge
            e = e.masked_fill(adj_mask == 0, float('-inf'))

            # Softmax over neighbors j
            alpha = F.softmax(e, dim=-1)  # (B, T, N, N)
            # FIXED: Prevent NaN when no active neighbors (isolated nodes)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            alpha = self.dropout(alpha)

            # Aggregate neighbor features: (B, T, N, N) @ (B, T, N, F') -> (B, T, N, F')
            Wh_btnf = Wh.permute(0, 2, 1, 3)  # (B, T, N, F')
            h_btnf = torch.matmul(alpha, Wh_btnf)  # (B, T, N, F')
            h_bntf = h_btnf.permute(0, 2, 1, 3)  # (B, N, T, F')
            head_outputs.append(h_bntf)

        # Concatenate heads on feature dim
        h = torch.cat(head_outputs, dim=-1)  # (B, N, T, out_dim)

        # Return raw multi-head aggregation (no BatchNorm inside attention block)
        return h