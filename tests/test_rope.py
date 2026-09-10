import torch

from dsa_cpu.rope import apply_rope_interleaved


def test_rope_rotates_first_dims_only_and_preserves_norm():
    x = torch.randn(5, 3, 128)
    pos = torch.arange(5)
    y = apply_rope_interleaved(x, pos, rotary_dim=64, theta=8e6)
    assert y.shape == x.shape
    assert torch.equal(y[..., 64:], x[..., 64:])
    assert torch.allclose(y[..., :64].norm(dim=-1), x[..., :64].norm(dim=-1), atol=1e-5)
    assert torch.allclose(y[0], x[0], atol=1e-6)


def test_rope_relative_position_invariance():
    q = torch.randn(1, 64)
    k = torch.randn(1, 64)

    def dot(pq, pk):
        return (apply_rope_interleaved(q, torch.tensor([pq]), 64, 8e6) * apply_rope_interleaved(k, torch.tensor([pk]), 64, 8e6)).sum()

    assert torch.allclose(dot(10, 7), dot(103, 100), atol=1e-4)
