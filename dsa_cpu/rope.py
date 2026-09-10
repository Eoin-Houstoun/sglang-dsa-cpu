import torch


def rope_cos_sin(positions: torch.Tensor, rotary_dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    return freqs.cos(), freqs.sin()


def apply_rope_interleaved(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    """Rotate the first `rotary_dim` features of x's last axis using interleaved pairs (2i, 2i+1).

    x: [N, ..., D]; positions: [N].
    """
    cos, sin = rope_cos_sin(positions, rotary_dim, theta)
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [rotary_dim // 2]
    cos, sin = cos.view(shape), sin.view(shape)
    rot = x[..., :rotary_dim].float()
    x1, x2 = rot[..., 0::2], rot[..., 1::2]
    out = torch.empty_like(rot)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return torch.cat([out.to(x.dtype), x[..., rotary_dim:]], dim=-1)
