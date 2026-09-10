import torch

from dsa_cpu.attention import dsa_attention, sparse_available
from dsa_cpu.config import GLM52_SMALL
from dsa_cpu.mla import dense_mla_attention
from dsa_cpu.synthetic import make_batches, make_weights


def test_dense_matches_naive_softmax_on_small_input():
    cfg = GLM52_SMALL
    w = make_weights(cfg)
    b = make_batches("prefill_1k", cfg, w)[0]
    T = 64
    out = dense_mla_attention(b.q_nope[:T], b.q_pe[:T], b.c_kv[:T], b.k_pe[:T], w.W_UK, w.W_UV, cfg.softmax_scale, 0, chunk=7)
    q_abs = torch.einsum("thd,hdr->thr", b.q_nope[:T].float(), w.W_UK.float())
    scores = (torch.einsum("thr,sr->ths", q_abs, b.c_kv[:T].float()) + torch.einsum("thd,sd->ths", b.q_pe[:T].float(), b.k_pe[:T].float())) * cfg.softmax_scale
    mask = torch.arange(T)[None, None, :] > torch.arange(T)[:, None, None]
    p = torch.softmax(scores.masked_fill(mask, float("-inf")), -1)
    ref = torch.einsum("thr,hrv->thv", torch.einsum("ths,sr->thr", p, b.c_kv[:T].float()), w.W_UV.float())
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


def test_baseline_falls_back_to_dense_when_sparse_missing():
    cfg = GLM52_SMALL
    w = make_weights(cfg)
    b = make_batches("decode_8k", cfg, w)[0]
    out, topk = dsa_attention(cfg, w, b)
    assert out.shape == (1, cfg.num_attention_heads, cfg.v_head_dim)
    assert (topk is None) == (not sparse_available(cfg))
