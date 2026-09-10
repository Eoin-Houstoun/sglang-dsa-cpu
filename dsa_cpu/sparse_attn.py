import torch


def sparse_mla_attention(q_nope, q_pe, c_kv, k_pe, topk_indices, W_UK, W_UV, scale):
    """Absorbed MLA attention where query t attends only to c_kv[topk_indices[t]] (-1 = padding).

    Must equal dense_mla_attention restricted to the selected set. Output [T, H, v_head].
    """
    raise NotImplementedError("sparse MLA attention over top-k indices has no CPU implementation")
