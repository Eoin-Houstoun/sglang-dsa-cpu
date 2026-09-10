from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DSAConfig:
    """GLM-5.2 attention hyperparameters, named as in the HF config and SGLang."""

    hidden_size: int = 1024
    q_lora_rank: int = 512
    num_attention_heads: int = 8
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_topk: int = 2048
    rope_theta: float = 8_000_000.0
    dtype: torch.dtype = torch.bfloat16

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return self.qk_head_dim**-0.5

    @property
    def index_softmax_scale(self) -> float:
        return self.index_head_dim**-0.5


GLM52_SMALL = DSAConfig()
