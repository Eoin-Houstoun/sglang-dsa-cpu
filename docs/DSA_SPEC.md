# DeepSeek Sparse Attention on CPU: what this repo asks for

GLM-5.2 (and DeepSeek-V3.2) use DSA: a cheap **lightning indexer** scores every KV position
for each query token, the **top `index_topk` = 2048** positions are kept, and MLA attention
runs over only those. SGLang implements this for CUDA, HIP, XPU and NPU. On CPU,
`Indexer.forward_native` raises `NotImplementedError("Indexer has no native (pure-torch) path")`
and the dense escape hatch is gated on SM90/SM100/MI355X, so nothing runs. This repo is that
gap at GLM-5.2's algorithmic shape, small enough to iterate on in seconds.

## Shape

| Field | GLM-5.2 | Here |
|---|---|---|
| `index_topk` | 2048 | 2048 |
| `index_head_dim` | 128 | 128 |
| `qk_rope_head_dim` | 64 | 64 |
| `index_n_heads` | 32 | 4 |
| `kv_lora_rank` | 512 | 512 |
| `qk_nope_head_dim` / `v_head_dim` | 192 / 256 | same |
| `num_attention_heads` | 64 | 8 |
| `hidden_size` / `q_lora_rank` | 6144 / 2048 | 1024 / 512 |
| dtype | fp8 weights, bf16 activations | bf16 |

Operating points: prefill at 1k, 2k, 4k, 8k tokens (query tokens = KV length, batch 1) and
one decode step for 4 sequences with an 8k KV cache. At 1k and 2k every valid position is
selected; at 4k sparse attention does half the dense work, at 8k a quarter.

## 1. Lightning indexer (`dsa_cpu/indexer.py`, `Indexer.forward`)

Notation: T query tokens at `positions`, S KV positions (the `index_k_cache` rows followed by
the keys of the T new tokens), Hi = 4 indexer heads, Di = 128, rope dim R = 64.

- `q = project_queries(q_lora, positions)` -> [T, Hi, Di] (wq_b, then interleaved rotary on the first R dims).
- `k = cat(index_k_cache, project_keys(x, positions))` -> [S, Di] (wk, LayerNorm over Di, rotary on the first R dims).
- `g = head_gates(x)` -> [T, Hi], already multiplied by Di^-0.5.
- `I[t, s] = sum_h g[t, h] * relu(q[t, h] . k[s])` for `s <= positions[t]`, else excluded.
- Return the indices of the `index_topk` largest `I[t, :]` as int32 [T, index_topk], ascending,
  padded with -1 when fewer than `index_topk` positions are valid (then all valid ones are selected).

The projections exist and are wired exactly like SGLang's. Only the scoring and selection are missing.

## 2. Sparse MLA attention (`dsa_cpu/sparse_attn.py`, `sparse_mla_attention`)

The KV cache stores `c_kv[s]` [512] and `k_pe[s]` [64]. For query t, head h, over the selected
set `sel(t)` from step 1 (ignore -1 entries):

- `q_abs[t, h] = q_nope[t, h] @ W_UK[h]` -> [512]
- `score[t, h, s] = (q_abs[t, h] . c_kv[s] + q_pe[t, h] . k_pe[s]) * scale`, `scale = (192 + 64)^-0.5`
- `p = softmax over s in sel(t)`; `o_lat[t, h] = sum_s p * c_kv[s]`; `out[t, h] = o_lat[t, h] @ W_UV[h]` -> [256]

`dense_mla_attention` in `dsa_cpu/mla.py` is the same maths over every valid position and is
the CPU path that exists today. The sparse result must equal dense restricted to `sel(t)`.

## 3. Contract

- `dsa_attention` (`dsa_cpu/attention.py`) uses the sparse path when both functions run, and
  falls back to dense only on `NotImplementedError`. No environment flags anywhere.
- `tests/` and `bench/` are the referee and are not to be modified. Goldens in `tests/fixtures`
  come from an fp32 reference; tolerances admit a bf16/AMX implementation (see README).
- The interface names above are what the tests import. Add helpers freely; keep the signatures.

## 4. Where this maps in SGLang (`python/sglang/srt/layers/attention/`)

| This repo | SGLang |
|---|---|
| `Indexer.forward` | `dsa/dsa_indexer.py` `Indexer.forward_native` (raises) / `forward_cuda` (`_get_topk_ragged`, `_get_topk_paged`) |
| `Indexer.project_queries` / `project_keys` / `head_gates` | `dsa_indexer.py` `_get_q_k_bf16`, `_get_k_bf16`, `_get_logits_head_gate` |
| `sparse_mla_attention` | `dsa_backend.py` `DeepseekSparseAttnBackend` sparse prefill/decode kernels (flashmla_sparse, aiter, tilelang, trtllm) |
| `dense_mla_attention` | the MHA/dense branch behind `SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD` |
| `DSAConfig` | `config.json` fields of `zai-org/GLM-5.2-FP8` |
