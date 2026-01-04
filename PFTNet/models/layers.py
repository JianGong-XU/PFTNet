import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# DropPath (Stochastic Depth)
# -------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor = random_tensor.floor()
        return x.div(keep_prob) * random_tensor


# -------------------------------------------------------------
# Re-Scaled Layer Normalization (RSLN)
# -------------------------------------------------------------
class RSLN(nn.Module):
    """
    Re-Scaled Layer Normalization
    Used instead of standard LayerNorm for improved stability
    in image restoration transformers.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x + self.bias


# -------------------------------------------------------------
# Feed-Forward Network (MLP)
# -------------------------------------------------------------
class MLP(nn.Module):
    """
    Standard Transformer MLP with GELU activation.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# -------------------------------------------------------------
# Utility: window / token reshape helpers
# -------------------------------------------------------------
def token_to_feature(
    x: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Convert token sequence to 2D feature map.

    x: [B, N, C]
    return: [B, C, H, W]
    """
    B, N, C = x.shape
    assert N == height * width
    return x.transpose(1, 2).reshape(B, C, height, width)


def feature_to_token(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 2D feature map to token sequence.

    x: [B, C, H, W]
    return: [B, N, C]
    """
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2)


# -------------------------------------------------------------
# Weight initialization utilities
# -------------------------------------------------------------
def init_weights(module: nn.Module):
    """
    Xavier / Truncated Normal initialization
    suitable for Transformer-based restoration models.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(
            module.weight, mode="fan_out", nonlinearity="relu"
        )
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.LayerNorm, RSLN)):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)
