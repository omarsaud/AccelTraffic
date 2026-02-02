"""
Graph WaveNet: Adaptive Graph Learning for Spatial-Temporal Modeling
=====================================================================

Reference:
    Wu et al., "Graph WaveNet for Deep Spatial-Temporal Graph Modeling",
    IJCAI 2019
    
Key Features:
    - Adaptive adjacency matrix learning (data-driven graph structure)
    - Stacked temporal convolutional layers with dilations (WaveNet-style)
    - Spatial graph convolution
    - Supports 2-channel input (speed + acceleration)

Architecture:
    Input: (batch, nodes, seq_len, input_dim)
    1. Input projection
    2. Stacked TCN + GCN blocks with residual connections
    3. Skip connections from all blocks
    4. Output projection
    Output: (batch, nodes, horizon, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphLearning(nn.Module):
    """
    Learn adaptive adjacency matrix from node embeddings.
    
    A_adaptive = softmax(ReLU(E1 * E2^T))
    
    Where E1, E2 are learned node embeddings.
    """
    
    def __init__(self, num_nodes, embed_dim=10):
        """
        Args:
            num_nodes: Number of nodes
            embed_dim: Embedding dimension
        """
        super(AdaptiveGraphLearning, self).__init__()
        
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        
        # Learnable node embeddings
        self.embedding1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.embedding2 = nn.Parameter(torch.randn(embed_dim, num_nodes))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize embeddings"""
        nn.init.xavier_uniform_(self.embedding1)
        nn.init.xavier_uniform_(self.embedding2)
    
    def forward(self):
        """
        Compute adaptive adjacency matrix.
        
        Returns:
            adj_adaptive: Adaptive adjacency matrix (num_nodes, num_nodes)
        """
        # Compute similarity: (num_nodes, embed_dim) x (embed_dim, num_nodes)
        adj_adaptive = torch.mm(self.embedding1, self.embedding2)  # (num_nodes, num_nodes)
        
        # Apply activation and normalization
        adj_adaptive = F.relu(adj_adaptive)
        adj_adaptive = F.softmax(adj_adaptive, dim=1)
        
        return adj_adaptive


class GraphConvolution(nn.Module):
    """
    Graph Convolution: aggregate neighbor information.
    
    Supports both fixed and adaptive adjacency matrices.
    ⚡ OPTIMIZED: Supports batched time dimension for vectorized processing.
    """
    
    def __init__(self, in_dim, out_dim, num_nodes, support_len=2, order=2):
        """
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            num_nodes: Number of nodes
            support_len: Number of support matrices (1 fixed + 1 adaptive = 2)
            order: Chebyshev polynomial order
        """
        super(GraphConvolution, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.support_len = support_len
        self.order = order
        
        # Weights for each support and order
        self.weight = nn.Parameter(
            torch.FloatTensor(support_len * order, in_dim, out_dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, x, adj_list, debug=False):
        """
        Forward pass - Supports both 3D and 4D input (vectorized over time).
        
        Args:
            x: Input (batch, nodes, in_dim) OR (batch, time, nodes, in_dim)
            adj_list: List of adjacency matrices [(nodes, nodes), ...]
            debug: Print debug info
        
        Returns:
            out: Output (batch, nodes, out_dim) OR (batch, time, nodes, out_dim)
        """
        if debug:
            print(f"      [GCN] Input shape: {x.shape}, adj_list len: {len(adj_list)}")
        
        # Handle both 3D and 4D input
        if x.dim() == 4:
            # 4D input: (batch, time, nodes, in_dim) -> vectorize over time
            batch_size, time_len, num_nodes, in_dim = x.shape
            # Reshape to (batch * time, nodes, in_dim) for batch processing
            x = x.reshape(batch_size * time_len, num_nodes, in_dim)
            is_4d = True
            if debug:
                print(f"      [GCN] Reshaped 4D->3D: {x.shape}")
        else:
            batch_size, num_nodes, in_dim = x.shape
            is_4d = False
        
        supports = []
        
        # Get dimensions for efficient matmul
        B, N, F = x.shape  # (batch, nodes, features)
        
        for i, adj in enumerate(adj_list):
            if debug:
                print(f"      [GCN] Processing adj[{i}] shape: {adj.shape}")
            
            # Compute Chebyshev polynomials of adjacency matrix
            x0 = x  # Order 0
            supports.append(x0)
            
            if self.order > 1:
                # ⚡ OPTIMIZED: Single large matmul instead of B small ones
                # Reshape: (B, N, F) -> (N, B*F)
                x_flat = x.permute(1, 0, 2).reshape(N, B * F)
                # Single matmul: (N, N) @ (N, B*F) -> (N, B*F)
                x1_flat = torch.mm(adj, x_flat)
                # Reshape back: (N, B*F) -> (B, N, F)
                x1 = x1_flat.reshape(N, B, F).permute(1, 0, 2).contiguous()
                supports.append(x1)
                if debug:
                    print(f"      [GCN] Order 1 done, x1 shape: {x1.shape}")
                
                # Higher orders: 2*A*X_{k-1} - X_{k-2}
                for k in range(2, self.order):
                    x1_flat = x1.permute(1, 0, 2).reshape(N, B * F)
                    x2_flat = 2 * torch.mm(adj, x1_flat)
                    x2 = x2_flat.reshape(N, B, F).permute(1, 0, 2).contiguous() - x0
                    supports.append(x2)
                    x0, x1 = x1, x2
        
        if debug:
            print(f"      [GCN] Stacking {len(supports)} supports...")
        
        # Stack all supports: (K, batch, nodes, in_dim) where K = support_len * order
        supports = torch.stack(supports, dim=0)
        K, B, N, I = supports.shape
        
        if debug:
            print(f"      [GCN] Supports stacked: {supports.shape}, weight: {self.weight.shape}")
        
        # ⚡ OPTIMIZED: Use batch matmul instead of slow einsum
        # Reshape: (K, B, N, I) -> (K, B*N, I)
        supports_flat = supports.reshape(K, B * N, I)
        
        # Batch matmul: (K, B*N, I) @ (K, I, O) -> (K, B*N, O)
        out = torch.bmm(supports_flat, self.weight)
        
        # Sum over K and reshape: (K, B*N, O) -> (B*N, O) -> (B, N, O)
        out = out.sum(dim=0).reshape(B, N, -1)
        
        if debug:
            print(f"      [GCN] After bmm: {out.shape}")
        
        # Reshape back to 4D if input was 4D
        if is_4d:
            out = out.reshape(batch_size, time_len, num_nodes, -1)
            if debug:
                print(f"      [GCN] Reshaped 3D->4D: {out.shape}")
        
        return out


class TemporalConvolution(nn.Module):
    """
    Temporal Convolution with dilation (WaveNet-style).
    
    Uses 1D convolution along temporal dimension.
    Supports causal convolution (no future information leakage).
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            dilation: Dilation rate
        """
        super(TemporalConvolution, self).__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Padding for causal convolution
        self.padding = (kernel_size - 1) * dilation
        
        # 1D convolution
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0  # Manual padding
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input (batch, channels, time)
        
        Returns:
            out: Output (batch, channels, time)
        """
        # Causal padding (left pad only)
        x = F.pad(x, (self.padding, 0), mode='constant', value=0)
        
        # Convolution
        out = self.conv(x)
        
        return out


class STConvBlock(nn.Module):
    """
    Spatial-Temporal Convolutional Block.
    
    Architecture:
        Input -> TCN -> GCN -> TCN -> Output
        
    With residual connection and skip connection.
    """
    
    def __init__(self, num_nodes, in_channels, out_channels, kernel_size=2,
                 dilation=1, support_len=2, dropout=0.3):
        """
        Args:
            num_nodes: Number of nodes
            in_channels: Input channels
            out_channels: Output channels (typically same as in_channels)
            kernel_size: TCN kernel size
            dilation: TCN dilation rate
            support_len: Number of graph supports
            dropout: Dropout rate
        """
        super(STConvBlock, self).__init__()
        
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1. Temporal gated convolution (WaveNet-style, Conv2d over (nodes, time))
        #    Input / output shape for this block: (batch, channels, nodes, time)
        self.filter_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation)
        )
        self.gate_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation)
        )
        
        # 2. Graph convolution (spatial) - uses optimized GraphConvolution
        self.gcn = GraphConvolution(
            in_dim=out_channels,
            out_dim=out_channels,
            num_nodes=num_nodes,
            support_len=support_len,
            order=2
        )
        
        # 3. Skip and residual projections (1x1 Conv2d)
        self.skip_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        self.residual_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        
        # 4. Normalization and dropout
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout_layer = nn.Dropout(dropout)
        
        # 5. Causal padding on temporal dimension (left pad only)
        #    Input layout: (B, C, N, T) → pad on last dimension (time)
        self.pad = nn.ConstantPad2d(
            ((kernel_size - 1) * dilation, 0, 0, 0),  # (left, right, top, bottom)
            0.0
        )
    
    def forward(self, x, adj_list, debug=False):
        """
        Forward pass - Official-style Graph WaveNet block (temporal gating + GCN).
        
        Args:
            x: Input (batch, in_channels, nodes, time)
            adj_list: List of adjacency matrices
            debug: Print debug info
        
        Returns:
            output: Output (batch, out_channels, nodes, time)
            skip: Skip connection (batch, out_channels, nodes, time)
        """
        if debug:
            print(f"    [STConvBlock] Input: {x.shape}")
        
        batch_size, in_channels, num_nodes, time_len = x.shape
        residual = x
        
        # 1. Causal temporal gated convolution (Conv2d over (nodes, time))
        #    x layout: (B, C_in, N, T)
        x_padded = self.pad(x)
        if debug:
            print(f"    [STConvBlock] After pad: {x_padded.shape}")
        
        filter_out = self.filter_conv(x_padded)
        gate_out = self.gate_conv(x_padded)
        x = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        if debug:
            print(f"    [STConvBlock] After gated TCN: {x.shape}")
        
        # 2. Graph convolution (spatial) using optimized GraphConvolution
        #    Reshape to (batch, time, nodes, channels) for GCN
        B, C, N, T_eff = x.shape
        x_g = x.permute(0, 3, 2, 1)  # (B, T, N, C)
        x_g = self.gcn(x_g, adj_list, debug=debug)
        x = x_g.permute(0, 3, 2, 1)  # (B, C, N, T)
        if debug:
            print(f"    [STConvBlock] After GCN: {x.shape}")
        
        # 3. Skip connection for this block (per-layer skip projection)
        skip = self.skip_conv(x)  # (B, C, N, T)
        if debug:
            print(f"    [STConvBlock] Skip shape: {skip.shape}")
        
        # 4. Residual connection (crop residual in time if needed)
        if residual.shape[3] != T_eff:
            residual = residual[..., -T_eff:]
        residual_proj = self.residual_conv(x)
        x = residual_proj + residual
        
        # 5. Normalization + dropout
        x = self.bn(x)
        x = self.dropout_layer(x)
        
        if debug:
            print(f"    [STConvBlock] Output: {x.shape}")
        
        return x, skip


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet Model.
    
    Combines adaptive graph learning with WaveNet-style temporal convolution.
    Supports 2-channel input (speed + acceleration).
    """
    
    def __init__(self, num_nodes, input_dim=1, output_dim=1, hidden_dim=32,
                 num_layers=8, kernel_size=2, dropout=0.3, seq_len=12, horizon=3,
                 support_len=2, embed_dim=10):
        """
        Args:
            num_nodes: Number of nodes
            input_dim: Input dimension (1=speed, 2=speed+accel)
            output_dim: Output dimension (1=speed prediction)
            hidden_dim: Hidden dimension (default: 32)
            num_layers: Number of ST-Conv blocks (default: 8)
            kernel_size: TCN kernel size
            dropout: Dropout rate
            seq_len: Input sequence length
            horizon: Prediction horizon
            support_len: Number of graph supports (1 fixed + 1 adaptive = 2)
            embed_dim: Node embedding dimension for adaptive graph
        """
        super(GraphWaveNet, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.horizon = horizon
        self.support_len = support_len
        
        # Adaptive graph learning
        self.adaptive_graph = AdaptiveGraphLearning(num_nodes, embed_dim)
        
        # Input projection
        self.start_conv = nn.Conv2d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=(1, 1)
        )
        
        # Stacked ST-Conv blocks with increasing dilation
        self.st_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i  # Exponentially increasing dilation
            self.st_blocks.append(
                STConvBlock(
                    num_nodes=num_nodes,
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    support_len=support_len,
                    dropout=dropout
                )
            )
        
        # Skip connection projection
        self.skip_conv = nn.Conv2d(
            in_channels=hidden_dim * num_layers,
            out_channels=hidden_dim * 4,
            kernel_size=(1, 1)
        )
        
        # Output projection
        self.end_conv1 = nn.Conv2d(
            in_channels=hidden_dim * 4,
            out_channels=hidden_dim * 2,
            kernel_size=(1, 1)
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=hidden_dim * 2,
            out_channels=horizon * output_dim,
            kernel_size=(1, 1)
        )
    
    def forward(self, x, adj=None, debug=False):
        """
        Forward pass.
        
        Args:
            x: Input (batch, nodes, seq_len, input_dim)
            adj: Fixed adjacency matrix (nodes, nodes) - optional
            debug: Print debug info
        
        Returns:
            output: Predictions (batch, nodes, horizon, output_dim)
        """
        import sys
        if debug:
            print(f"[GWNET] Input shape: {x.shape}", flush=True)
            sys.stdout.flush()
        
        batch_size = x.shape[0]
        
        # Reshape: (batch, nodes, seq_len, input_dim) -> (batch, input_dim, nodes, seq_len)
        x = x.permute(0, 3, 1, 2)
        
        if debug:
            print(f"[GWNET] After permute: {x.shape}", flush=True)
        
        # Input projection
        x = self.start_conv(x)  # (batch, hidden_dim, nodes, seq_len)
        
        if debug:
            print(f"[GWNET] After start_conv: {x.shape}", flush=True)
        
        # Prepare adjacency matrices
        if debug:
            print(f"[GWNET] Computing adaptive graph...", flush=True)
        
        adj_adaptive = self.adaptive_graph()
        
        if debug:
            print(f"[GWNET] Adaptive graph shape: {adj_adaptive.shape}", flush=True)
        
        if adj is not None and self.support_len > 1:
            # Use both fixed and adaptive
            adj_list = [adj, adj_adaptive]
            if debug:
                print(f"[GWNET] Using fixed ({adj.shape}) + adaptive adj", flush=True)
        elif adj is not None:
            # Use only fixed
            adj_list = [adj]
            if debug:
                print(f"[GWNET] Using only fixed adj", flush=True)
        else:
            # Use only adaptive
            adj_list = [adj_adaptive]
            if debug:
                print(f"[GWNET] Using only adaptive adj", flush=True)
        
        # Pass through ST-Conv blocks with skip connections
        skip_outputs = []
        for i, block in enumerate(self.st_blocks):
            if debug:
                print(f"[GWNET] Processing ST-Conv block {i}...", flush=True)
            x, skip = block(x, adj_list, debug=debug)
            skip_outputs.append(skip)
            if debug:
                print(f"[GWNET] Block {i} done, output: {x.shape}", flush=True)
        
        if debug:
            print(f"[GWNET] All blocks done, aggregating skips...", flush=True)
        
        # Aggregate skip connections
        skip = torch.cat(skip_outputs, dim=1)  # (batch, hidden_dim * num_layers, nodes, seq_len)
        
        if debug:
            print(f"[GWNET] Skip shape: {skip.shape}", flush=True)
        
        # Skip projection
        x = self.skip_conv(skip)  # (batch, hidden_dim * 4, nodes, seq_len)
        x = F.relu(x)
        
        # Output projection
        x = self.end_conv1(x)  # (batch, hidden_dim * 2, nodes, seq_len)
        x = F.relu(x)
        x = self.end_conv2(x)  # (batch, horizon * output_dim, nodes, seq_len)
        
        # Temporal pooling (take last timestep)
        x = x[:, :, :, -1]  # (batch, horizon * output_dim, nodes)
        
        # Reshape to output format
        x = x.permute(0, 2, 1)  # (batch, nodes, horizon * output_dim)
        x = x.reshape(batch_size, self.num_nodes, self.horizon, self.output_dim)
        
        if debug:
            print(f"[GWNET] Output shape: {x.shape}", flush=True)
        
        return x


# Test function
if __name__ == '__main__':
    # Test Graph WaveNet
    batch_size = 32
    num_nodes = 207
    seq_len = 12
    horizon = 3
    input_dim = 2  # Speed + acceleration
    
    # Create model
    model = GraphWaveNet(
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=32,
        num_layers=8,
        kernel_size=2,
        dropout=0.3,
        seq_len=seq_len,
        horizon=horizon,
        support_len=2,
        embed_dim=10
    )
    
    # Create dummy data
    x = torch.randn(batch_size, num_nodes, seq_len, input_dim)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.t()) / 2  # Make symmetric
    
    # Forward pass
    output = model(x, adj)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Graph WaveNet test passed!")
