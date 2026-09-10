import torch
import torch.nn.functional as F

from .config import DSAConfig
from .rope import apply_rope_interleaved


class Indexer:
    """Lightning indexer of DeepSeek Sparse Attention, laid out like SGLang's Indexer.

    The projections are wired. `forward` (logits and top-k selection) has no CPU path.
    """

    def __init__(self, config: DSAConfig, weights):
        self.config = config
        self.w = weights
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = config.index_softmax_scale

    def project_queries(self, q_lora: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        q = (q_lora @ self.w.wq_b.T).view(-1, self.n_heads, self.head_dim)
        return apply_rope_interleaved(q, positions, self.rope_head_dim, self.config.rope_theta)

    def project_keys(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        k = F.layer_norm((x @ self.w.wk.T).float(), (self.head_dim,), self.w.k_norm_weight.float(), self.w.k_norm_bias.float(), eps=1e-6)
        return apply_rope_interleaved(k.to(x.dtype), positions, self.rope_head_dim, self.config.rope_theta)

    def head_gates(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.w.weights_proj.T).float() * self.softmax_scale

    def forward(self, x: torch.Tensor, q_lora: torch.Tensor, positions: torch.Tensor, index_k_cache: torch.Tensor) -> torch.Tensor:
        """Return the top-k KV indices per query token: [T, index_topk] int32, ascending, -1 padded.

        Keys are index_k_cache followed by project_keys(x, positions). For query t and key
        position s <= positions[t]: I[t, s] = sum_h head_gates(x)[t, h] * relu(q[t, h] . k[s]).
        Select the index_topk largest I[t, :]; when fewer positions are valid, select them all.
        """
        raise NotImplementedError("Indexer has no native (pure-torch) path")
