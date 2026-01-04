import torch
import torch.nn.functional as F


def deformable_attention_sampling(
    value: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
):
    """
    Deformable attention sampling core.

    Args:
        value: Tensor of shape
            [B, Hh, Hd, H, W]
            B  : batch size
            Hh : number of heads
            Hd : head dimension
            H,W: spatial resolution

        sampling_locations: Tensor of shape
            [B, Hh, N, K, 2]
            normalized to [-1, 1]

        attention_weights: Tensor of shape
            [B, Hh, N, K]

    Returns:
        output: Tensor of shape
            [B, N, Hh * Hd]
    """

    B, Hh, Hd, H, W = value.shape
    _, _, N, K, _ = sampling_locations.shape

    # reshape value for grid_sample
    value = value.view(B * Hh, Hd, H, W)

    # reshape grid
    grid = sampling_locations.permute(0, 1, 2, 3, 4)
    grid = grid.reshape(B * Hh, N * K, 1, 2)

    # sample
    sampled = F.grid_sample(
        value,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    # reshape sampled features
    sampled = sampled.view(
        B,
        Hh,
        Hd,
        N,
        K,
    )

    # apply attention weights
    attn = attention_weights.unsqueeze(2)
    output = (sampled * attn).sum(-1)

    # merge heads
    output = output.permute(0, 3, 1, 2).contiguous()
    output = output.view(B, N, Hh * Hd)

    return output


def normalize_sampling_locations(
    ref_points: torch.Tensor,
    offsets: torch.Tensor,
):
    """
    Normalize sampling locations.

    Args:
        ref_points: [B, N, 2], normalized
        offsets:    [B, N, Hh, K, 2]

    Returns:
        sampling_locations: [B, Hh, N, K, 2]
    """

    ref = ref_points.unsqueeze(2).unsqueeze(3)
    sampling_locations = ref + offsets
    sampling_locations = sampling_locations.clamp(-1.0, 1.0)
    sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4)
    return sampling_locations


def split_heads(
    x: torch.Tensor,
    num_heads: int,
):
    """
    Split last dimension into multi-head format.

    x: [B, N, C]
    return: [B, Hh, Hd, N]
    """
    B, N, C = x.shape
    Hd = C // num_heads
    x = x.view(B, N, num_heads, Hd)
    x = x.permute(0, 2, 3, 1).contiguous()
    return x


def merge_heads(
    x: torch.Tensor,
):
    """
    Merge multi-head tensor.

    x: [B, Hh, Hd, N]
    return: [B, N, Hh * Hd]
    """
    B, Hh, Hd, N = x.shape
    x = x.permute(0, 3, 1, 2).contiguous()
    return x.view(B, N, Hh * Hd)
