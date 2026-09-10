from dataclasses import dataclass

import torch

from .config import DSAConfig, GLM52_SMALL
from .rope import apply_rope_interleaved


@dataclass
class DSAWeights:
    wq_b: torch.Tensor  # [Hi*Di, q_lora_rank]
    wk: torch.Tensor  # [Di, hidden]
    weights_proj: torch.Tensor  # [Hi, hidden]
    k_norm_weight: torch.Tensor  # [Di]
    k_norm_bias: torch.Tensor  # [Di]
    W_UK: torch.Tensor  # [H, qk_nope, kv_lora]
    W_UV: torch.Tensor  # [H, kv_lora, v_head]


@dataclass
class DSABatch:
    name: str
    x: torch.Tensor  # [T, hidden] layer input of the query tokens
    q_lora: torch.Tensor  # [T, q_lora_rank]
    positions: torch.Tensor  # [T] int64
    q_nope: torch.Tensor  # [T, H, qk_nope]
    q_pe: torch.Tensor  # [T, H, rope], rotary applied
    c_kv: torch.Tensor  # [S, kv_lora] full KV cache, query tokens included
    k_pe: torch.Tensor  # [S, rope], rotary applied
    index_k_cache: torch.Tensor  # [S - T, Di] indexer keys of the context before this step
    causal_offset: int  # query t may attend to s <= t + causal_offset


# name -> (query tokens, kv length, sequences)
OPERATING_POINTS = {
    "prefill_1k": (1024, 1024, 1),
    "prefill_2k": (2048, 2048, 1),
    "prefill_4k": (4096, 4096, 1),
    "prefill_8k": (8192, 8192, 1),
    "decode_8k": (1, 8192, 4),
}
WEIGHT_SEED = 20260910


def make_weights(config: DSAConfig = GLM52_SMALL, seed: int = WEIGHT_SEED) -> DSAWeights:
    g = torch.Generator().manual_seed(seed)
    Hi, Di, H = config.index_n_heads, config.index_head_dim, config.num_attention_heads

    def lin(out_f, in_f):
        return (torch.randn(out_f, in_f, generator=g) * in_f**-0.5).to(config.dtype)

    return DSAWeights(
        wq_b=lin(Hi * Di, config.q_lora_rank),
        wk=lin(Di, config.hidden_size),
        weights_proj=lin(Hi, config.hidden_size),
        k_norm_weight=torch.ones(Di, dtype=config.dtype),
        k_norm_bias=torch.zeros(Di, dtype=config.dtype),
        W_UK=(torch.randn(H, config.qk_nope_head_dim, config.kv_lora_rank, generator=g) * config.qk_nope_head_dim**-0.5).to(config.dtype),
        W_UV=(torch.randn(H, config.kv_lora_rank, config.v_head_dim, generator=g) * config.kv_lora_rank**-0.5).to(config.dtype),
    )


def full_context_tensors(name: str, config: DSAConfig, seed: int, seq: int):
    """Activations for the whole context of one sequence: (x, q_lora, positions, q_nope, q_pe, c_kv, k_pe)."""
    _, S, _ = OPERATING_POINTS[name]
    point_id = list(OPERATING_POINTS).index(name)
    g = torch.Generator().manual_seed(seed * 100_003 + point_id * 101 + seq)
    positions = torch.arange(S)
    x = torch.randn(S, config.hidden_size, generator=g).to(config.dtype)
    q_lora = torch.randn(S, config.q_lora_rank, generator=g).to(config.dtype)
    q_nope = torch.randn(S, config.num_attention_heads, config.qk_nope_head_dim, generator=g).to(config.dtype)
    q_pe = apply_rope_interleaved(torch.randn(S, config.num_attention_heads, config.qk_rope_head_dim, generator=g).to(config.dtype), positions, config.qk_rope_head_dim, config.rope_theta)
    c_kv = torch.randn(S, config.kv_lora_rank, generator=g).to(config.dtype)
    k_pe = apply_rope_interleaved(torch.randn(S, config.qk_rope_head_dim, generator=g).to(config.dtype), positions, config.qk_rope_head_dim, config.rope_theta)
    return x, q_lora, positions, q_nope, q_pe, c_kv, k_pe


def make_batches(name: str, config: DSAConfig = GLM52_SMALL, weights: DSAWeights | None = None, seed: int = 0) -> list[DSABatch]:
    """Deterministic activations for one operating point, one DSABatch per sequence."""
    from .indexer import Indexer

    T, S, n_seq = OPERATING_POINTS[name]
    weights = weights or make_weights(config)
    indexer = Indexer(config, weights)
    batches = []
    for i in range(n_seq):
        x, q_lora, positions, q_nope, q_pe, c_kv, k_pe = full_context_tensors(name, config, seed, i)
        q = slice(S - T, S)
        batches.append(
            DSABatch(
                name=name,
                x=x[q],
                q_lora=q_lora[q],
                positions=positions[q],
                q_nope=q_nope[q],
                q_pe=q_pe[q],
                c_kv=c_kv,
                k_pe=k_pe,
                index_k_cache=indexer.project_keys(x[: S - T], positions[: S - T]),
                causal_offset=S - T,
            )
        )
    return batches
