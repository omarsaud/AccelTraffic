import torch.nn as nn
import torch


class SemanticEnhancement(nn.Module):
    """
    PAPER-ALIGNED Semantic Enhancement (3 Parallel Convolutions).
    
    Matches STGIN paper exactly:
    - THREE PARALLEL 1D convolutions (not sequential)
    - Kernel sizes: [1, 3, 3] for point-wise, backward, forward patterns
    - Reverse INPUT before convolution (not output after)
    - Selective gating on conv2 & conv3 only
    - NO BatchNorm (paper doesn't use it)
    - Simple addition for fusion
    
    Architecture:
        Conv1 (k=1): Point-wise transformation
        Conv2 (k=3): Backward pattern (reverse input → conv → reverse back → gate)
        Conv3 (k=3): Forward pattern (conv → gate)
        Output: Conv1 + Conv2 + Conv3
    
    Args:
        input_dim: Number of input channels (1=speed, 2=speed+acceleration)
        hidden_dim: Output dimension (typically 64)
        kernel_size: Temporal convolution kernel size for conv2/conv3 (default=3)
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(SemanticEnhancement, self).__init__()
        
        # Three PARALLEL convolutions (paper architecture)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
        
        # NO BatchNorm, NO Residual (paper doesn't use them)

    def forward(self, x):
        """
        Forward pass - PAPER-ALIGNED.
        
        Args:
            x: (batch, nodes, P, input_dim)
               - input_dim=1: [speed]
               - input_dim=2: [speed, acceleration] stacked
        
        Returns:
            out: (batch, nodes, P, hidden_dim)
        
        Three parallel convolution paths matching paper:
        1. Point-wise (k=1): Direct feature transformation
        2. Backward (k=3): Reverse input → conv → reverse back → gate
        3. Forward (k=3): Conv → gate
        """
        batch, nodes, time, input_dim = x.shape
        
        # Reshape for Conv1d: (batch*nodes, channels, time)
        x_reshape = x.permute(0, 1, 3, 2).reshape(batch * nodes, input_dim, time)
        
        # Path 1: Point-wise transformation (k=1)
        h1 = self.conv1(x_reshape)  # (B*N, hidden, T)
        
        # Path 2: Backward pattern (reverse input first, then conv)
        x_reverse = torch.flip(x_reshape, dims=[-1])  # Reverse time dimension
        h2 = self.conv2(x_reverse)  # Conv on reversed input
        h2 = torch.flip(h2, dims=[-1])  # Reverse back to normal order
        h2 = h2 * torch.sigmoid(h2)  # GATED (paper methodology)
        
        # Path 3: Forward pattern (standard conv)
        h3 = self.conv3(x_reshape)  # Conv forward
        h3 = h3 * torch.sigmoid(h3)  # GATED (paper methodology)
        
        # Combine three paths (simple addition, paper style)
        out = h1 + h2 + h3  # (B*N, hidden, T)
        
        # Reshape back to (batch, nodes, time, hidden_dim)
        out = out.reshape(batch, nodes, -1, time).permute(0, 1, 3, 2)
        
        return out
