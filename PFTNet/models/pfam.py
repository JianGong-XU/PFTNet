import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import RSLN, MLP
from models.daam import DAAM
from models.dsaam import DSAAM


class ProgressiveFocusedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        use_daam: bool = True,
        use_dsaam: bool = True,
    ):
        super().__init__()
        self.use_daam = use_daam
        self.use_dsaam = use_dsaam

        if self.use_daam:
            self.daam = DAAM(dim)

        if self.use_dsaam:
            self.dsaam = DSAAM(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        ref_points: torch.Tensor,
        prev_focus: Optional[torch.Tensor] = None,
    ):
        """
        x: [B, N, C]
        ref_points: [B, N, 2]
        prev_focus: [B, N, 1] or None
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        feat_2d = x.transpose(1, 2).reshape(B, C, H, W)

        focus_gate = None
        if self.use_daam:
            feat_2d, focus_gate = self.daam(feat_2d)

        feat = feat_2d.flatten(2).transpose(1, 2)

        if prev_focus is not None:
            feat = feat * prev_focus

        if self.use_dsaam:
            attn_out = self.dsaam(feat, ref_points)
        else:
            attn_out = feat

        out = self.proj(attn_out)

        if focus_gate is not None:
            focus_gate = focus_gate.mean(dim=1, keepdim=True)
            focus_gate = focus_gate.flatten(2).transpose(1, 2)

        return out, focus_gate


class PFAM(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        use_daam: bool = True,
        use_dsaam: bool = True,
    ):
        super().__init__()

        self.norm1 = RSLN(dim)
        self.norm2 = RSLN(dim)

        self.pfa = ProgressiveFocusedAttention(
            dim=dim,
            use_daam=use_daam,
            use_dsaam=use_dsaam,
        )

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        ref_points: torch.Tensor,
        prev_focus: Optional[torch.Tensor] = None,
    ):
        """
        x: [B, N, C]
        ref_points: [B, N, 2]
        prev_focus: [B, N, 1] or None
        """
        identity = x

        x_norm = self.norm1(x)
        attn_out, focus_gate = self.pfa(
            x_norm,
            ref_points=ref_points,
            prev_focus=prev_focus,
        )

        x = identity + attn_out

        x = x + self.mlp(self.norm2(x))

        return x
