"""
AGCRN: Adaptive Graph Convolutional Recurrent Network
======================================================

Reference:
    Bai et al., "Adaptive Graph Convolutional Recurrent Network for Traffic 
    Forecasting", NeurIPS 2020
    
Key Features:
    - Node-adaptive parameter learning (different parameters per node)
    - Data-adaptive graph structure learning
    - Efficient: fewer parameters, faster training
    - Supports 2-channel input (speed + acceleration)

Architecture:
    Input: (batch, nodes, seq_len, input_dim)
    1. Node embeddings for adaptive parameters
    2. Stacked AGCN-GRU layers
    3. Output projection
    Output: (batch, nodes, horizon, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGraphConv(nn.Module):
    """
    Adaptive Graph Convolution.
    
    Instead of shared weights for all nodes, learns node-specific parameters
    through node embeddings.
    
    For each node i:
        W_i = tanh(E_i * W_e + b_e)
        Z_i = Σ_j A_ij * X_j * W_i
    
    Where E_i is node i's embedding, W_e is shared weight matrix.
    """
    
    def __init__(self, num_nodes, in_dim, out_dim, embed_dim=10, cheb_k=2):
        """
        Args:
            num_nodes: Number of nodes
            in_dim: Input dimension
            out_dim: Output dimension
            embed_dim: Node embedding dimension
            cheb_k: Chebyshev polynomial order
        """
        super(AdaptiveGraphConv, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        
        # Node embeddings
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        
        # Shared weight generators (for each Chebyshev order)
        self.weight_generators = nn.ModuleList([
            nn.Linear(embed_dim, in_dim * out_dim)
            for _ in range(cheb_k)
        ])
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(num_nodes, out_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.node_embeddings)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass.
        
        Args:
            x: Input (batch, nodes, in_dim)
            adj: Adjacency matrix (nodes, nodes)
        
        Returns:
            out: Output (batch, nodes, out_dim)
        """
        batch_size, num_nodes, in_dim = x.shape
        
        # Generate node-adaptive weights
        # For each node, generate weights from embeddings
        node_weights = []  # List of (nodes, in_dim, out_dim) for each order
        
        for k, weight_gen in enumerate(self.weight_generators):
            # Generate weights: (nodes, embed_dim) -> (nodes, in_dim * out_dim)
            weights = weight_gen(self.node_embeddings)  # (nodes, in_dim * out_dim)
            weights = weights.view(num_nodes, in_dim, self.out_dim)  # (nodes, in_dim, out_dim)
            node_weights.append(weights)
        
        # Compute Chebyshev polynomials
        supports = []
        x_k = x  # Order 0
        supports.append(x_k)
        
        if self.cheb_k > 1:
            # Normalize adjacency matrix
            d = adj.sum(dim=1)
            d_inv_sqrt = torch.pow(d, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
            adj_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
            
            # Order 1
            x_k = torch.einsum('nm,bmi->bni', adj_norm, x)
            supports.append(x_k)
            
            # Higher orders: 2*A*X_{k-1} - X_{k-2}
            x_k_minus_1 = x
            for k in range(2, self.cheb_k):
                x_k_new = 2 * torch.einsum('nm,bmi->bni', adj_norm, x_k) - x_k_minus_1
                supports.append(x_k_new)
                x_k_minus_1, x_k = x_k, x_k_new
        
        # Apply node-adaptive weights
        # For each Chebyshev order and each node, apply its specific weights
        outputs = []
        for k, (support, weights) in enumerate(zip(supports, node_weights)):
            # support: (batch, nodes, in_dim)
            # weights: (nodes, in_dim, out_dim)
            # For each node i: output_i = support_i * weights_i
            out_k = torch.einsum('bni,nio->bno', support, weights)  # (batch, nodes, out_dim)
            outputs.append(out_k)
        
        # Sum over Chebyshev orders
        out = sum(outputs) + self.bias
        
        return out


class AGCRNCell(nn.Module):
    """
    Adaptive GCN-GRU Cell.
    
    Combines node-adaptive graph convolution with GRU.
    """
    
    def __init__(self, num_nodes, input_dim, hidden_dim, embed_dim=10, cheb_k=2):
        """
        Args:
            num_nodes: Number of nodes
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            embed_dim: Node embedding dimension
            cheb_k: Chebyshev order
        """
        super(AGCRNCell, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Update and reset gates
        self.gate_conv = AdaptiveGraphConv(
            num_nodes=num_nodes,
            in_dim=input_dim + hidden_dim,
            out_dim=2 * hidden_dim,
            embed_dim=embed_dim,
            cheb_k=cheb_k
        )
        
        # Candidate state
        self.candidate_conv = AdaptiveGraphConv(
            num_nodes=num_nodes,
            in_dim=input_dim + hidden_dim,
            out_dim=hidden_dim,
            embed_dim=embed_dim,
            cheb_k=cheb_k
        )
    
    def forward(self, x, h, adj):
        """
        Forward pass.
        
        Args:
            x: Input (batch, nodes, input_dim)
            h: Hidden state (batch, nodes, hidden_dim)
            adj: Adjacency matrix (nodes, nodes)
        
        Returns:
            h_new: New hidden state (batch, nodes, hidden_dim)
        """
        # Concatenate input and hidden
        x_h = torch.cat([x, h], dim=-1)  # (batch, nodes, input_dim + hidden_dim)
        
        # Gates
        gates = self.gate_conv(x_h, adj)  # (batch, nodes, 2 * hidden_dim)
        gates = torch.sigmoid(gates)
        
        # Split into reset and update gates
        r, u = torch.split(gates, self.hidden_dim, dim=-1)
        
        # Candidate state
        x_rh = torch.cat([x, r * h], dim=-1)  # (batch, nodes, input_dim + hidden_dim)
        c = self.candidate_conv(x_rh, adj)  # (batch, nodes, hidden_dim)
        c = torch.tanh(c)
        
        # New hidden state
        h_new = u * h + (1 - u) * c
        
        return h_new


class DataAdaptiveGraph(nn.Module):
    """
    Learn data-adaptive graph structure.
    
    Constructs adjacency matrix from learned node embeddings:
    A = softmax(ReLU(E * E^T))
    """
    
    def __init__(self, num_nodes, embed_dim=10):
        """
        Args:
            num_nodes: Number of nodes
            embed_dim: Embedding dimension
        """
        super(DataAdaptiveGraph, self).__init__()
        
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        
        # Node embeddings
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize embeddings"""
        nn.init.xavier_uniform_(self.node_embeddings)
    
    def forward(self):
        """
        Compute adaptive adjacency matrix.
        
        Returns:
            adj: Adaptive adjacency (num_nodes, num_nodes)
        """
        # Compute similarity
        adj = torch.mm(self.node_embeddings, self.node_embeddings.t())  # (nodes, nodes)
        
        # Apply activation and normalization
        adj = F.relu(adj)
        adj = F.softmax(adj, dim=1)
        
        return adj


class AGCRNEncoder(nn.Module):
    """AGCRN Encoder with stacked AGCN-GRU layers"""
    
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers=2, 
                 embed_dim=10, cheb_k=2):
        """
        Args:
            num_nodes: Number of nodes
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            embed_dim: Node embedding dimension
            cheb_k: Chebyshev order
        """
        super(AGCRNEncoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stacked AGCRN cells
        self.agcrn_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.agcrn_cells.append(
                AGCRNCell(num_nodes, in_dim, hidden_dim, embed_dim, cheb_k)
            )
    
    def forward(self, x, adj):
        """
        Encode input sequence.
        
        Args:
            x: Input (batch, nodes, seq_len, input_dim)
            adj: Adjacency matrix (nodes, nodes)
        
        Returns:
            h_list: List of final hidden states for each layer
        """
        batch_size, num_nodes, seq_len, input_dim = x.shape
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
             for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, :, t, :]  # (batch, nodes, input_dim)
            
            # Pass through stacked layers
            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer] = self.agcrn_cells[layer](x_t, h[layer], adj)
                else:
                    h[layer] = self.agcrn_cells[layer](h[layer-1], h[layer], adj)
        
        return h


class AGCRNDecoder(nn.Module):
    """AGCRN Decoder"""
    
    def __init__(self, num_nodes, output_dim, hidden_dim, num_layers=2,
                 embed_dim=10, cheb_k=2):
        """
        Args:
            num_nodes: Number of nodes
            output_dim: Output dimension
            hidden_dim: Hidden dimension
            num_layers: Number of layers
            embed_dim: Node embedding dimension
            cheb_k: Chebyshev order
        """
        super(AGCRNDecoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stacked AGCRN cells
        self.agcrn_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = output_dim if i == 0 else hidden_dim
            self.agcrn_cells.append(
                AGCRNCell(num_nodes, in_dim, hidden_dim, embed_dim, cheb_k)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, h_enc, adj, horizon):
        """
        Decode hidden states to predictions.
        
        Args:
            h_enc: Encoder hidden states (list of tensors)
            adj: Adjacency matrix (nodes, nodes)
            horizon: Prediction horizon
        
        Returns:
            outputs: Predictions (batch, nodes, horizon, output_dim)
        """
        batch_size = h_enc[0].shape[0]
        device = h_enc[0].device
        
        # Initialize decoder hidden states
        h = h_enc
        
        # Initialize decoder input
        decoder_input = torch.zeros(batch_size, self.num_nodes, self.output_dim, device=device)
        
        outputs = []
        
        for t in range(horizon):
            # Pass through stacked layers
            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer] = self.agcrn_cells[layer](decoder_input, h[layer], adj)
                else:
                    h[layer] = self.agcrn_cells[layer](h[layer-1], h[layer], adj)
            
            # Project to output
            output = self.output_proj(h[-1])  # (batch, nodes, output_dim)
            outputs.append(output)
            
            # Use prediction as next input (autoregressive)
            decoder_input = output
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=2)  # (batch, nodes, horizon, output_dim)
        
        return outputs


class AGCRN(nn.Module):
    """
    Adaptive Graph Convolutional Recurrent Network (AGCRN)
    
    Efficient model with node-adaptive parameters and data-adaptive graph.
    Supports 2-channel input (speed + acceleration).
    """
    
    def __init__(self, num_nodes, input_dim=1, output_dim=1, hidden_dim=64,
                 num_layers=2, embed_dim=10, cheb_k=2, dropout=0.3, 
                 seq_len=12, horizon=3):
        """
        Args:
            num_nodes: Number of nodes
            input_dim: Input dimension (1=speed, 2=speed+accel)
            output_dim: Output dimension (1=speed prediction)
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            embed_dim: Node embedding dimension
            cheb_k: Chebyshev polynomial order
            dropout: Dropout rate
            seq_len: Input sequence length
            horizon: Prediction horizon
        """
        super(AGCRN, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Data-adaptive graph learning
        self.adaptive_graph = DataAdaptiveGraph(num_nodes, embed_dim)
        
        # Encoder
        self.encoder = AGCRNEncoder(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embed_dim=embed_dim,
            cheb_k=cheb_k
        )
        
        # Decoder
        self.decoder = AGCRNDecoder(
            num_nodes=num_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            embed_dim=embed_dim,
            cheb_k=cheb_k
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj=None):
        """
        Forward pass.
        
        Args:
            x: Input (batch, nodes, seq_len, input_dim)
            adj: Fixed adjacency matrix (nodes, nodes) - optional
        
        Returns:
            output: Predictions (batch, nodes, horizon, output_dim)
        """
        # Learn adaptive adjacency
        adj_adaptive = self.adaptive_graph()
        
        # Combine fixed and adaptive adjacency (if fixed is provided)
        if adj is not None:
            # Weighted combination
            adj_combined = 0.5 * adj + 0.5 * adj_adaptive
        else:
            adj_combined = adj_adaptive
        
        # Encode
        h_enc = self.encoder(x, adj_combined)
        
        # Apply dropout
        h_enc = [self.dropout(h) for h in h_enc]
        
        # Decode
        output = self.decoder(h_enc, adj_combined, self.horizon)
        
        return output


# Test function
if __name__ == '__main__':
    # Test AGCRN
    batch_size = 32
    num_nodes = 207
    seq_len = 12
    horizon = 3
    input_dim = 2  # Speed + acceleration
    
    # Create model
    model = AGCRN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        embed_dim=10,
        cheb_k=2,
        dropout=0.3,
        seq_len=seq_len,
        horizon=horizon
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
    print("✅ AGCRN test passed!")
