"""
DCRNN: Diffusion Convolutional Recurrent Neural Network
========================================================

Reference:
    Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven 
    Traffic Forecasting", ICLR 2018
    
Key Features:
    - Bidirectional random walk on graph (diffusion process)
    - GRU with diffusion convolution instead of matrix multiplication
    - Encoder-decoder architecture with scheduled sampling
    - Supports 2-channel input (speed + acceleration)

Architecture:
    Input: (batch, nodes, seq_len, input_dim)
    Encoder: Diffusion GRU layers
    Decoder: Diffusion GRU layers with scheduled sampling
    Output: (batch, nodes, horizon, output_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiffusionConv(nn.Module):
    """
    Diffusion Convolution Layer.
    
    Implements bidirectional random walk diffusion process:
    Z = Σ(θ_k * (D_O^-1 * W)^k * X + θ_k' * (D_I^-1 * W^T)^k * X)
    
    Where:
        - D_O^-1 * W: Forward random walk transition matrix
        - D_I^-1 * W^T: Backward random walk transition matrix
        - k: Diffusion steps (typically K=2)
    """
    
    def __init__(self, num_nodes, in_dim, out_dim, K=2, bias=True):
        """
        Args:
            num_nodes: Number of nodes in graph
            in_dim: Input feature dimension
            out_dim: Output feature dimension
            K: Diffusion steps (default: 2)
            bias: Whether to use bias
        """
        super(DiffusionConv, self).__init__()
        
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K
        
        # Weights for each diffusion step (forward and backward)
        # Total: 2 * K supports (forward K steps + backward K steps)
        self.weight = nn.Parameter(torch.FloatTensor(2 * K, in_dim, out_dim))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass.
        
        Args:
            x: Input features (batch, nodes, in_dim)
            adj: Adjacency matrix (nodes, nodes)
        
        Returns:
            out: Output features (batch, nodes, out_dim)
        """
        batch_size, num_nodes, in_dim = x.shape
        
        # Compute transition matrices
        # Forward: D_O^-1 * W (row-normalized)
        d_out = adj.sum(dim=1)  # Out-degree
        d_out_inv = torch.pow(d_out, -1)
        d_out_inv[torch.isinf(d_out_inv)] = 0.
        d_mat_inv = torch.diag(d_out_inv)
        adj_forward = torch.mm(d_mat_inv, adj)  # (nodes, nodes)
        
        # Backward: D_I^-1 * W^T (column-normalized)
        d_in = adj.sum(dim=0)  # In-degree
        d_in_inv = torch.pow(d_in, -1)
        d_in_inv[torch.isinf(d_in_inv)] = 0.
        d_mat_inv = torch.diag(d_in_inv)
        adj_backward = torch.mm(d_mat_inv, adj.t())  # (nodes, nodes)
        
        # Compute diffusion supports
        supports = []
        
        # Forward diffusion
        x_forward = x  # (batch, nodes, in_dim)
        for k in range(self.K):
            # X * (D_O^-1 * W)^k
            if k == 0:
                supports.append(x_forward)
            else:
                x_forward = torch.einsum('bnf,nm->bmf', x_forward, adj_forward)
                supports.append(x_forward)
        
        # Backward diffusion
        x_backward = x
        for k in range(self.K):
            # X * (D_I^-1 * W^T)^k
            if k == 0:
                supports.append(x_backward)
            else:
                x_backward = torch.einsum('bnf,nm->bmf', x_backward, adj_backward)
                supports.append(x_backward)
        
        # Stack supports: (2K, batch, nodes, in_dim)
        supports = torch.stack(supports, dim=0)
        
        # Apply learnable weights: (2K, batch, nodes, in_dim) x (2K, in_dim, out_dim)
        # Result: (batch, nodes, out_dim)
        out = torch.einsum('kbni,kio->bno', supports, self.weight)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU Cell.
    
    Replaces matrix multiplications in standard GRU with diffusion convolution.
    
    Standard GRU:
        r = σ(W_r * [h, x] + b_r)      # Reset gate
        u = σ(W_u * [h, x] + b_u)      # Update gate
        c = tanh(W_c * [r⊙h, x] + b_c) # Candidate state
        h' = u⊙h + (1-u)⊙c             # New hidden state
    
    DCGRU:
        Replace matrix multiplication with diffusion convolution.
    """
    
    def __init__(self, num_nodes, input_dim, hidden_dim, K=2):
        """
        Args:
            num_nodes: Number of nodes
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
            K: Diffusion steps
        """
        super(DCGRUCell, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.K = K
        
        # Gates
        self.gate = DiffusionConv(
            num_nodes=num_nodes,
            in_dim=input_dim + hidden_dim,
            out_dim=2 * hidden_dim,  # For reset and update gates
            K=K,
            bias=True
        )
        
        # Candidate
        self.candidate = DiffusionConv(
            num_nodes=num_nodes,
            in_dim=input_dim + hidden_dim,
            out_dim=hidden_dim,
            K=K,
            bias=True
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
        # Concatenate input and hidden state
        x_h = torch.cat([x, h], dim=-1)  # (batch, nodes, input_dim + hidden_dim)
        
        # Compute gates
        gates = self.gate(x_h, adj)  # (batch, nodes, 2 * hidden_dim)
        gates = torch.sigmoid(gates)
        
        # Split into reset and update gates
        r, u = torch.split(gates, self.hidden_dim, dim=-1)
        
        # Candidate state
        x_rh = torch.cat([x, r * h], dim=-1)  # (batch, nodes, input_dim + hidden_dim)
        c = self.candidate(x_rh, adj)  # (batch, nodes, hidden_dim)
        c = torch.tanh(c)
        
        # New hidden state
        h_new = u * h + (1 - u) * c
        
        return h_new


class DCRNNEncoder(nn.Module):
    """DCRNN Encoder with stacked DCGRU layers"""
    
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers=2, K=2):
        """
        Args:
            num_nodes: Number of nodes
            input_dim: Input dimension (1 for speed, 2 for speed+accel)
            hidden_dim: Hidden dimension
            num_layers: Number of DCGRU layers
            K: Diffusion steps
        """
        super(DCRNNEncoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stacked DCGRU cells
        self.dcgru_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.dcgru_cells.append(
                DCGRUCell(num_nodes, in_dim, hidden_dim, K)
            )
    
    def forward(self, x, adj):
        """
        Encode input sequence.
        
        Args:
            x: Input (batch, nodes, seq_len, input_dim)
            adj: Adjacency matrix (nodes, nodes)
        
        Returns:
            h_list: List of hidden states for each layer
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
                    h[layer] = self.dcgru_cells[layer](x_t, h[layer], adj)
                else:
                    h[layer] = self.dcgru_cells[layer](h[layer-1], h[layer], adj)
        
        return h


class DCRNNDecoder(nn.Module):
    """DCRNN Decoder with scheduled sampling"""
    
    def __init__(self, num_nodes, output_dim, hidden_dim, num_layers=2, K=2):
        """
        Args:
            num_nodes: Number of nodes
            output_dim: Output dimension (typically 1 for speed)
            hidden_dim: Hidden dimension
            num_layers: Number of DCGRU layers
            K: Diffusion steps
        """
        super(DCRNNDecoder, self).__init__()
        
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stacked DCGRU cells
        self.dcgru_cells = nn.ModuleList()
        for i in range(num_layers):
            in_dim = output_dim if i == 0 else hidden_dim
            self.dcgru_cells.append(
                DCGRUCell(num_nodes, in_dim, hidden_dim, K)
            )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, h_enc, adj, horizon, target=None, teacher_forcing_ratio=0.5):
        """
        Decode hidden states to predictions.
        
        Args:
            h_enc: Encoder hidden states (list of tensors)
            adj: Adjacency matrix (nodes, nodes)
            horizon: Prediction horizon
            target: Ground truth (batch, nodes, horizon, output_dim) - for training
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Predictions (batch, nodes, horizon, output_dim)
        """
        batch_size = h_enc[0].shape[0]
        device = h_enc[0].device
        
        # Initialize decoder hidden states with encoder final states
        h = h_enc
        
        # Initialize decoder input (zeros for first step)
        decoder_input = torch.zeros(batch_size, self.num_nodes, self.output_dim, device=device)
        
        outputs = []
        
        for t in range(horizon):
            # Pass through stacked layers
            for layer in range(self.num_layers):
                if layer == 0:
                    h[layer] = self.dcgru_cells[layer](decoder_input, h[layer], adj)
                else:
                    h[layer] = self.dcgru_cells[layer](h[layer-1], h[layer], adj)
            
            # Project to output
            output = self.output_proj(h[-1])  # (batch, nodes, output_dim)
            outputs.append(output)
            
            # Scheduled sampling: use ground truth or prediction as next input
            if target is not None and np.random.random() < teacher_forcing_ratio:
                decoder_input = target[:, :, t, :]  # Teacher forcing
            else:
                decoder_input = output  # Use own prediction
        
        # Stack outputs: (batch, nodes, horizon, output_dim)
        outputs = torch.stack(outputs, dim=2)
        
        return outputs


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network (DCRNN)
    
    Full model with encoder-decoder architecture.
    Supports 2-channel input (speed + acceleration).
    """
    
    def __init__(self, num_nodes, input_dim=1, output_dim=1, hidden_dim=64,
                 num_layers=2, K=2, dropout=0.3, seq_len=12, horizon=3):
        """
        Args:
            num_nodes: Number of nodes in graph
            input_dim: Input dimension (1=speed, 2=speed+accel)
            output_dim: Output dimension (1=speed prediction)
            hidden_dim: Hidden dimension
            num_layers: Number of RNN layers
            K: Diffusion steps
            dropout: Dropout rate
            seq_len: Input sequence length
            horizon: Prediction horizon
        """
        super(DCRNN, self).__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = K
        self.seq_len = seq_len
        self.horizon = horizon
        
        # Encoder
        self.encoder = DCRNNEncoder(
            num_nodes=num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K
        )
        
        # Decoder
        self.decoder = DCRNNDecoder(
            num_nodes=num_nodes,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            K=K
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj, target=None, teacher_forcing_ratio=0.5):
        """
        Forward pass.
        
        Args:
            x: Input (batch, nodes, seq_len, input_dim)
            adj: Adjacency matrix (nodes, nodes)
            target: Ground truth for training (batch, nodes, horizon, output_dim)
            teacher_forcing_ratio: Teacher forcing probability
        
        Returns:
            output: Predictions (batch, nodes, horizon, output_dim)
        """
        # Encode
        h_enc = self.encoder(x, adj)
        
        # Apply dropout to hidden states
        h_enc = [self.dropout(h) for h in h_enc]
        
        # Decode
        output = self.decoder(h_enc, adj, self.horizon, target, teacher_forcing_ratio)
        
        return output


# Test function
if __name__ == '__main__':
    # Test DCRNN
    batch_size = 32
    num_nodes = 207
    seq_len = 12
    horizon = 3
    input_dim = 2  # Speed + acceleration
    
    # Create model
    model = DCRNN(
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=64,
        num_layers=2,
        K=2,
        dropout=0.3,
        seq_len=seq_len,
        horizon=horizon
    )
    
    # Create dummy data
    x = torch.randn(batch_size, num_nodes, seq_len, input_dim)
    adj = torch.rand(num_nodes, num_nodes)
    adj = (adj + adj.t()) / 2  # Make symmetric
    target = torch.randn(batch_size, num_nodes, horizon, 1)
    
    # Forward pass
    output = model(x, adj, target, teacher_forcing_ratio=0.5)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ DCRNN test passed!")
