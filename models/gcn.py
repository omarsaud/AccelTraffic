import torch.nn as nn
import torch

class MultiHeadGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1):
        super(MultiHeadGCN, self).__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.conv = nn.ModuleList([nn.Linear(in_dim, self.head_dim) for _ in range(num_heads)])
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x, adj):
        # x : (batch, nodes, P, in_dim)
        # adj : (nodes, nodes)

        batch, nodes, time, in_dim = x.shape

        x = x.permute(0, 2, 1, 3).reshape(batch * time, nodes, in_dim)
        # x : (batch * time, nodes, in_dim)

        # FIXED: Clone adjacency and set self-loop to avoid doubling existing loops
        adj = adj.clone()
        adj.fill_diagonal_(1.0)
        degree = adj.sum(1).pow(-0.5)
        degree = torch.diag(degree)  # Convert to diagonal matrix
        adj_norm = degree @ adj @ degree

        outputs = [head(torch.einsum('ij,bjd->bid', adj_norm, x)) for head in self.conv]
        h = torch.cat(outputs, dim=-1)  # (batch * time, nodes, out_dim)

        h = h.reshape(batch * time * nodes, -1)  # Flatten for BatchNorm1d
        h = self.bn(h).reshape(batch * time, nodes, -1)  # Normalize and reshape back
        return h.reshape(batch, time, nodes, -1).permute(0, 2, 1, 3)  # (batch, nodes, time, out_dim)