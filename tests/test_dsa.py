import pytest
import torch

from dsa_cpu.attention import dsa_attention, sparse_available
from dsa_cpu.config import GLM52_SMALL
from dsa_cpu.mla import dense_mla_attention
from dsa_cpu.synthetic import make_batches, make_weights
from tests import goldens

CFG = GLM52_SMALL
W = make_weights(CFG)
SPARSE = sparse_available(CFG)


def _run_point(name):
    outs, topks = [], []
    for b in make_batches(name, CFG, W):
        out, topk = dsa_attention(CFG, W, b)
        outs.append(out)
        topks.append(topk)
    return torch.cat(outs), (None if topks[0] is None else torch.cat(topks))


@pytest.mark.parametrize("name", goldens.POINTS)
def test_dense_matches_golden(name):
    outs = [dense_mla_attention(b.q_nope, b.q_pe, b.c_kv, b.k_pe, W.W_UK, W.W_UV, CFG.softmax_scale, b.causal_offset) for b in make_batches(name, CFG, W)]
    assert goldens.check_dense_point(name, torch.cat(outs)) == []


@pytest.mark.skipif(not SPARSE, reason="sparse path not implemented (upstream state)")
@pytest.mark.parametrize("name", goldens.POINTS)
def test_sparse_matches_golden(name):
    out, topk = _run_point(name)
    assert goldens.check_sparse_point(name, out, topk) == []


@pytest.mark.skipif(not SPARSE, reason="sparse path not implemented (upstream state)")
def test_sparse_indices_causal_and_in_range():
    for b in make_batches("prefill_4k", CFG, W):
        _, topk = dsa_attention(CFG, W, b)
        assert topk.dtype == torch.int32
        valid = topk >= 0
        limit = b.positions[:, None].expand_as(topk)
        assert bool((topk[valid] <= limit[valid]).all())
        assert bool((topk[valid] >= 0).all())
