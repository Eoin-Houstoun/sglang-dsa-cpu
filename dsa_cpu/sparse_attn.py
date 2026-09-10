import torch


_DETERMINISTIC_BLOCK = 1024
_SPARSE_QUERY_CHUNK = 16


def sparse_mla_attention(q_nope, q_pe, c_kv, k_pe, topk_indices, W_UK, W_UV, scale):
    """Absorbed MLA attention where query t attends only to c_kv[topk_indices[t]] (-1 = padding).

    Must equal dense_mla_attention restricted to the selected set. Output [T, H, v_head].
    """
    T = q_nope.shape[0]
    H = W_UV.shape[0]
    out = torch.empty((T, H, W_UV.shape[-1]), dtype=q_nope.dtype, device=q_nope.device)

    # Decode uses one fully-populated selection row.  Keep heads as matrix rows
    # and issue direct matrix products rather than entering the general gather
    # chunk and dispatching four einsums.
    if T == 1 and bool((topk_indices[0] >= 0).all()):
        idx = topk_indices[0].long()
        c_sel = c_kv.index_select(0, idx)
        pe_sel = k_pe.index_select(0, idx)
        q_abs = torch.bmm(q_nope[0].unsqueeze(1), W_UK).squeeze(1)
        scores = torch.mm(q_abs, c_sel.T)
        scores.add_(torch.mm(q_pe[0], pe_sel.T))
        p = torch.softmax(scores.float().mul_(scale), dim=-1).to(c_kv.dtype)
        o_lat = torch.mm(p, c_sel)
        out[0] = torch.bmm(o_lat.unsqueeze(1), W_UV).squeeze(1)
        return out

    q_abs = torch.einsum("thd,hdr->thr", q_nope, W_UK)

    # Leading causal rows select their complete prefix. Compute these in large
    # rectangular blocks so every KV row is shared by many queries instead of
    # being copied by a per-row gather.
    dense_end = 0
    max_dense = min(T, topk_indices.shape[1])
    while dense_end < max_dense:
        row = topk_indices[dense_end]
        n = int((row >= 0).sum())
        if n != dense_end + 1 or not torch.equal(row[:n], torch.arange(n, dtype=row.dtype, device=row.device)):
            break
        dense_end += 1

    block = _DETERMINISTIC_BLOCK
    for t0 in range(0, dense_end, block):
        t1 = min(dense_end, t0 + block)
        S = t1
        scores = torch.einsum("thr,sr->ths", q_abs[t0:t1], c_kv[:S])
        scores.add_(torch.einsum("thd,sd->ths", q_pe[t0:t1], k_pe[:S]))
        scores = scores.float().mul_(scale)
        limit = torch.arange(t0, t1, device=q_nope.device)[:, None, None]
        cols = torch.arange(S, device=q_nope.device)
        scores.masked_fill_(cols[None, None, :] > limit, float("-inf"))
        p = torch.softmax(scores, dim=-1).to(c_kv.dtype)
        o_lat = torch.einsum("ths,sr->thr", p, c_kv[:S])
        out[t0:t1] = torch.einsum("thr,hrv->thv", o_lat, W_UV)

    # A small row chunk bounds the irregular gathered working set while all
    # heads remain vectorised so the contractions still reach bf16 AMX kernels.
    chunk = _SPARSE_QUERY_CHUNK
    for t0 in range(dense_end, T, chunk):
        t1 = min(T, t0 + chunk)
        idx = topk_indices[t0:t1].long()
        valid = idx >= 0
        safe = idx.clamp_min(0)
        c_sel = c_kv[safe]
        pe_sel = k_pe[safe]
        scores = torch.einsum("thr,tkr->thk", q_abs[t0:t1], c_sel)
        scores.add_(torch.einsum("thd,tkd->thk", q_pe[t0:t1], pe_sel))
        scores = scores.float().mul_(scale)
        scores.masked_fill_(~valid[:, None, :], float("-inf"))
        p = torch.softmax(scores, dim=-1).to(c_kv.dtype)
        o_lat = torch.einsum("thk,tkr->thr", p, c_sel)
        out[t0:t1] = torch.einsum("thr,hrv->thv", o_lat, W_UV)
    return out
