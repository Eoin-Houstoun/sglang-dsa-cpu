import torch


def dense_mla_attention(q_nope, q_pe, c_kv, k_pe, W_UK, W_UV, scale, causal_offset=0, chunk=1024):
    """Absorbed MLA attention over every valid KV position: the CPU path that exists today."""
    T = q_nope.shape[0]
    S = c_kv.shape[0]
    q_abs = torch.einsum("thd,hdr->thr", q_nope, W_UK)
    out = torch.empty(T, W_UV.shape[0], W_UV.shape[-1], dtype=q_nope.dtype)
    s_idx = torch.arange(S)
    for t0 in range(0, T, chunk):
        t1 = min(T, t0 + chunk)
        scores = torch.einsum("thr,sr->ths", q_abs[t0:t1], c_kv) + torch.einsum("thd,sd->ths", q_pe[t0:t1], k_pe)
        scores = scores.float() * scale
        limit = (torch.arange(t0, t1) + causal_offset)[:, None, None]
        scores.masked_fill_(s_idx[None, None, :] > limit, float("-inf"))
        p = torch.softmax(scores, dim=-1).to(c_kv.dtype)
        o_lat = torch.einsum("ths,sr->thr", p, c_kv)
        out[t0:t1] = torch.einsum("thr,hrv->thv", o_lat, W_UV)
    return out
