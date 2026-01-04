import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def deformable_sampling(
    feat: torch.Tensor,
    sampling_locations: torch.Tensor,
):
    """
    feat: [B, C, H, W]
    sampling_locations: [B, N, Hh, K, 2] normalized to [-1, 1]
    """
    B, C, H, W = feat.shape
    _, N, Hh, K, _ = sampling_locations.shape

    feat = feat.view(B, C, H, W)

    grid = sampling_locations.view(B, N * Hh * K, 1, 2)

    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    sampled = sampled.view(B, C, N, Hh, K)
    return sampled


class DSAAM(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        num_points: int = 8,
    ):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.value_proj = nn.Linear(dim, dim)
        self.offset_proj = nn.Linear(dim, num_heads * num_points * 2)
        self.attn_weight_proj = nn.Linear(dim, num_heads * num_points)
        self.output_proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        ref_points: torch.Tensor,
    ):
        """
        x: [B, N, C]
        ref_points: [B, N, 2] normalized
        """
        B, N, C = x.shape
        H = W = int(math.sqrt(N))

        value = self.value_proj(x)
        value = value.view(B, N, self.num_heads, self.head_dim)
        value = value.permute(0, 2, 1, 3).contiguous()

        feat_2d = (
            value.permute(0, 1, 3, 2)
            .reshape(B * self.num_heads, self.head_dim, H, W)
        )

        offset = self.offset_proj(x)
        offset = offset.view(
            B,
            N,
            self.num_heads,
            self.num_points,
            2,
        )

        attn_weight = self.attn_weight_proj(x)
        attn_weight = attn_weight.view(
            B,
            N,
            self.num_heads,
            self.num_points,
        )
        attn_weight = F.softmax(attn_weight, dim=-1)

        ref = ref_points.unsqueeze(2).unsqueeze(3)
        sampling_locations = ref + offset
        sampling_locations = sampling_locations.clamp(-1.0, 1.0)

        sampling_locations = sampling_locations.permute(
            0, 2, 1, 3, 4
        ).contiguous()

        sampled = deformable_sampling(
            feat_2d,
            sampling_locations,
        )

        sampled = sampled.view(
            B,
            self.num_heads,
            self.head_dim,
            N,
            self.num_points,
        )

        attn_weight = attn_weight.permute(0, 2, 1, 3)
        attn_weight = attn_weight.unsqueeze(2)

        out = (sampled * attn_weight).sum(-1)
        out = out.permute(0, 3, 1, 2).contiguous()
        out = out.view(B, N, C)

        out = self.output_proj(out)
        return out
