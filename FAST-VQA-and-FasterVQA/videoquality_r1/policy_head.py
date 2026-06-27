# policy_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyHead(nn.Module):
    """
    Trainable head that outputs:
      - MOS score (scalar 1–5)
      - reasoning chain (hidden representation for reward shaping)
    """

    def __init__(self, feature_dim: int = 768, hidden_dim: int = 256):
        super().__init__()

        self.reasoning = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Score head: outputs raw logit → scaled to [1, 5]
        self.score_head = nn.Linear(hidden_dim, 1)

        # Uncertainty head: log-variance for reward weighting
        self.uncertainty_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor):
        """
        features: (B, feature_dim) from frozen backbone
        Returns:
            scores:      (B,) MOS predictions in [1, 5]
            log_var:     (B,) log variance (uncertainty)
            hidden:      (B, hidden_dim) reasoning representation
        """
        hidden = self.reasoning(features)                      # (B, H)
        raw_score = self.score_head(hidden).squeeze(-1)        # (B,)
        scores = 1.0 + 4.0 * torch.sigmoid(raw_score)         # scale to [1,5]
        log_var = self.uncertainty_head(hidden).squeeze(-1)    # (B,)
        return scores, log_var, hidden