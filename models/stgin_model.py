from .bridge_trans import BridgeTrans
from .semantic_enhancement import SemanticEnhancement
from .st_block import STBlock
from .global_configuration import DROPOUT
import torch.nn as nn


class STGIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=DROPOUT):
        """Baseline STGIN for traffic speed prediction.

        This implementation matches the official STGIN backbone, using:
        - SemanticEnhancement front-end
        - Two STBlocks with single-direction LSTM
        - One bridge transformer layer
        - Deterministic regression head (no uncertainty)
        """
        super(STGIN, self).__init__()

        # Semantic enhancement with multi-channel Conv1d
        self.semantic = SemanticEnhancement(input_dim, hidden_dim)

        # Two standard ST-Blocks
        self.st_blocks = nn.ModuleList([
            STBlock(hidden_dim, hidden_dim),
            STBlock(hidden_dim, hidden_dim),
        ])

        # Single bridge transformer layer
        self.bridge = BridgeTrans(hidden_dim, hidden_dim, num_layers=1)

        # Deterministic prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),  # 64 → 64
            nn.Dropout(dropout),  # Use global DROPOUT (0.3) for consistency
            nn.Linear(hidden_dim, 1),  # 64 → 1
        )
        self.fc = self.predictor

    def forward(self, x, adj, ste):
        """
        Forward pass through the STGIN model.
        
        Args:
            x: (B, N, P, in_dim) input features
            adj: (N, N) adjacency matrix
            ste: (B, N, P+Q, hidden_dim) spatiotemporal embeddings
        
        Returns:
            predictions: (B, N, Q, 1) predicted speeds
        """
        # x : (batch, nodes, P, in_dim)
        # adj : (nodes, nodes)
        # ste : (batch, nodes, P+Q, hidden_dim)
        _, _, P, _ = x.shape

        h = self.semantic(x)
        # h : (batch, nodes, P, in_dim)

        for st_block in self.st_blocks:
            h = st_block(h, adj, ste[:, :, :P, :])

        # h : (batch, nodes, P, hidden_dim)
        
        # Decode future with bridge transformer
        h = self.bridge(h, ste[:, :, P:, :])  # (batch, nodes, Q, hidden_dim)

        # Final prediction
        return self.predictor(h)