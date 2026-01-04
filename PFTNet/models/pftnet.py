import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.pfam import PFAM
from models.layers import RSLN


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim // 2, kernel_size=1)

    def forward(self, x: torch.Tensor, size: List[int]) -> torch.Tensor:
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = self.conv(x)
        return x


class EncoderStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([PFAM(dim) for _ in range(depth)])
        self.downsample = Downsample(dim)

    def forward(
        self,
        x: torch.Tensor,
        ref_points: torch.Tensor,
    ):
        for blk in self.blocks:
            x = blk(x, ref_points=ref_points)

        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        feat = x.transpose(1, 2).reshape(B, C, H, W)

        down = self.downsample(feat)
        down = down.flatten(2).transpose(1, 2)

        return x, down


class DecoderStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([PFAM(dim) for _ in range(depth)])
        self.upsample = Upsample(dim)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        ref_points: torch.Tensor,
        out_size: List[int],
    ):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        feat = x.transpose(1, 2).reshape(B, C, H, W)

        feat = self.upsample(feat, out_size)
        feat = feat.flatten(2).transpose(1, 2)

        feat = feat + skip

        for blk in self.blocks:
            feat = blk(feat, ref_points=ref_points)

        return feat


class SoftReconstructionHead(nn.Module):
    def __init__(self, dim: int, out_channels: int = 3):
        super().__init__()
        self.image_head = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        self.weight_head = nn.Sequential(
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, hazy: torch.Tensor) -> torch.Tensor:
        image = self.image_head(feat)
        weight = self.weight_head(feat)
        output = weight * image + (1.0 - weight) * hazy
        return output


class PFTNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        embed_dim: int = 64,
        encoder_depths: List[int] = [2, 2, 4, 4],
        decoder_depths: List[int] = [2, 2, 2],
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.encoder_stages = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()

        dims = [embed_dim * (2**i) for i in range(len(encoder_depths))]

        for i, depth in enumerate(encoder_depths):
            self.encoder_stages.append(
                EncoderStage(
                    dim=dims[i],
                    depth=depth,
                )
            )

        for i, depth in enumerate(decoder_depths):
            self.decoder_stages.append(
                DecoderStage(
                    dim=dims[-(i + 1)],
                    depth=depth,
                )
            )

        self.norm = RSLN(embed_dim)
        self.reconstruction = SoftReconstructionHead(embed_dim, out_channels)

    def _build_ref_points(
        self,
        B: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H, device=device),
            torch.linspace(-1.0, 1.0, W, device=device),
            indexing="ij",
        )
        ref = torch.stack([xx, yy], dim=-1)
        ref = ref.reshape(1, H * W, 2).repeat(B, 1, 1)
        return ref

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape

        feat = self.patch_embed(x)
        feat = feat.flatten(2).transpose(1, 2)

        skips = []
        sizes = []

        cur = feat
        cur_H, cur_W = H, W

        for stage in self.encoder_stages:
            ref = self._build_ref_points(B, cur_H, cur_W, x.device)
            skip, cur = stage(cur, ref)
            skips.append(skip)
            sizes.append((cur_H, cur_W))
            cur_H //= 2
            cur_W //= 2

        for idx, stage in enumerate(self.decoder_stages):
            skip = skips[-(idx + 2)]
            out_H, out_W = sizes[-(idx + 1)]
            ref = self._build_ref_points(B, out_H, out_W, x.device)
            cur = stage(cur, skip, ref, [out_H, out_W])

        cur = self.norm(cur)
        cur = cur.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        output = self.reconstruction(cur, x)
        return output


def build_pftnet_s() -> PFTNet:
    return PFTNet(
        embed_dim=48,
        encoder_depths=[1, 2, 2, 2],
        decoder_depths=[1, 1, 1],
    )


def build_pftnet_l() -> PFTNet:
    return PFTNet(
        embed_dim=64,
        encoder_depths=[2, 2, 4, 4],
        decoder_depths=[2, 2, 2],
    )
