import torch

from .config import DSAConfig, GLM52_SMALL
from .indexer import Indexer
from .mla import dense_mla_attention
from .sparse_attn import sparse_mla_attention
from .synthetic import DSABatch, DSAWeights, make_weights


def dsa_attention(config: DSAConfig, weights: DSAWeights, batch: DSABatch):
    """One attention layer forward: sparse DSA when implemented, else the dense fallback.

    Returns (out [T, H, v_head], topk_indices [T, index_topk] or None).
    """
    try:
        topk = Indexer(config, weights).forward(batch.x, batch.q_lora, batch.positions, batch.index_k_cache)
        out = sparse_mla_attention(batch.q_nope, batch.q_pe, batch.c_kv, batch.k_pe, topk, weights.W_UK, weights.W_UV, config.softmax_scale)
        return out, topk
    except NotImplementedError:
        out = dense_mla_attention(batch.q_nope, batch.q_pe, batch.c_kv, batch.k_pe, weights.W_UK, weights.W_UV, config.softmax_scale, batch.causal_offset)
        return out, None


def sparse_available(config: DSAConfig = GLM52_SMALL) -> bool:
    """Capability probe: True when the indexer and sparse attention both run on a tiny input.

    Only NotImplementedError means "not available"; any other error propagates.
    """
    weights = make_weights(config)
    g = torch.Generator().manual_seed(1)
    n = 16
    rand = lambda *shape: torch.randn(*shape, generator=g).to(config.dtype)
    x, q_lora = rand(n, config.hidden_size), rand(n, config.q_lora_rank)
    positions = torch.arange(n)
    try:
        topk = Indexer(config, weights).forward(x, q_lora, positions, torch.empty(0, config.index_head_dim, dtype=config.dtype))
        sparse_mla_attention(
            rand(n, config.num_attention_heads, config.qk_nope_head_dim),
            rand(n, config.num_attention_heads, config.qk_rope_head_dim),
            rand(n, config.kv_lora_rank),
            rand(n, config.qk_rope_head_dim),
            topk,
            weights.W_UK,
            weights.W_UV,
            config.softmax_scale,
        )
    except NotImplementedError:
        return False
    return True
