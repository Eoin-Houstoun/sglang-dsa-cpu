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
        T = x.shape[0]
        topk = self.index_topk
        out = torch.full((T, topk), -1, dtype=torch.int32, device=x.device)

        # A causal row with at most topk valid positions has a predetermined
        # answer. Besides being exact, handling these rows here avoids both the
        # indexer GEMMs and topk for short prompts and the early prefill prefix.
        valid_counts = (positions + 1).clamp(min=0)
        deterministic = valid_counts <= topk
        for t in deterministic.nonzero(as_tuple=False).flatten().tolist():
            n = min(int(valid_counts[t]), topk)
            if n:
                out[t, :n] = torch.arange(n, dtype=torch.int32, device=x.device)

        work = (~deterministic).nonzero(as_tuple=False).flatten()
        if work.numel() == 0:
            return out

        q = self.project_queries(q_lora, positions)
        new_k = self.project_keys(x, positions)
        keys = torch.cat((index_k_cache, new_k), dim=0)
        gates = self.head_gates(x)

        # One fp32 [query chunk, KV] accumulator avoids materialising the much
        # larger [query, index head, KV] logits tensor.
        chunk = 256
        for i0 in range(0, work.numel(), chunk):
            rows = work[i0 : i0 + chunk]
            max_valid = min(int(valid_counts[rows].max()), keys.shape[0])
            logits = torch.zeros((rows.numel(), max_valid), dtype=torch.float32, device=x.device)
            qr = q[rows]
            for h in range(self.n_heads):
                dots = qr[:, h] @ keys[:max_valid].T
                logits.add_(torch.relu(dots).float() * gates[rows, h, None])
            cols = torch.arange(max_valid, device=x.device)
            logits.masked_fill_(cols[None, :] >= valid_counts[rows, None], float("-inf"))
            selected = torch.topk(logits, topk, dim=-1, sorted=False).indices
            selected = selected.sort(dim=-1).values.to(torch.int32)
            out[rows] = selected
        return out
