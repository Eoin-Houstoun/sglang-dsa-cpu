# sglang-dsa-cpu

DeepSeek Sparse Attention (DSA), as used by GLM-5.2, has no CPU implementation in SGLang:
`Indexer.forward_native` raises `NotImplementedError`, and the sparse kernels are CUDA, HIP,
XPU or NPU only. This repo reproduces that gap as a small pure-PyTorch harness at GLM-5.2's
attention shape (`index_topk` 2048, 64 attention heads, 32 indexer heads, real MLA dims; only the hidden width is shrunk) so the CPU path can be
implemented and made fast with a benchmark and a correctness gate in the loop. Target
hardware is Xeon with AMX (bf16 matmuls dispatch to oneDNN AMX).

State of the tree: `main` carries a correct CPU sparse path produced by an Artemis Discovery run
(3.2x faster than dense at 8k prefill). The pre-implementation state, with `Indexer.forward` and
`sparse_mla_attention` raising `NotImplementedError` exactly as upstream, is tagged `v0-gap`; check
it out to reproduce the enablement run. The full definition is in [docs/DSA_SPEC.md](docs/DSA_SPEC.md).

## Run

```bash
uv sync
uv run pytest -q tests            # dense goldens pass; sparse tests skip until implemented
uv run python bench/benchmark.py  # writes artemis_results.json
```

## Metrics (`artemis_results.json`)

| Metric | Better | Meaning |
|---|---|---|
| `dsa_sparse_correct` | 1 | 1 only when the sparse path runs on all five operating points and matches the goldens |
| `prefill_ms_4k` | lower | median time of one attention layer, 4096-token prefill |
| `prefill_ms_8k` | lower | same at 8192 tokens |
| `decode_ms_8k` | lower | one decode step for 4 sequences with an 8k KV cache |
| `indexer_ms_8k` | lower | indexer alone at 8k; 0 until the sparse path is correct |
| `gen_ms_8k_in_1k_out` | lower | Intel's target configuration: prefill at 8k plus 1,000 decode steps, derived from the two above |

Baseline (Xeon Platinum 8581C, 16 cores, torch 2.14 CPU): dense 4k about 1.08 s, 8k about
4.3 s, decode about 12 ms. Sparse attention at 8k does a quarter of the dense arithmetic, but a
per-row gather of 2,048 KV rows moves about 19 GB at 8k, so beating dense needs tiled or
page-level gathers shared across query rows, not a naive select.

## Correctness gate

`tests/fixtures/*.pt` hold, for 48 sampled query rows per operating point, the reference top-k
sets and the sparse and dense outputs computed by an fp32 reference. `tests/goldens.py` checks
index-set overlap (at least 0.97 per row; rows with fewer than 2048 valid positions must select
all of them) and output error. Rows whose selected set is fully determined use a tight output
tolerance; rows with a genuine top-2048 selection use a loose one, because a few bf16 boundary
flips legitimately move those outputs. Tolerances were set from a bf16 run of the reference with
a 3x margin and verified to reject a dropped ReLU, missing head gates, non-causal selection and a
wrong attention scale. `tests/` and `bench/` are the referee: do not modify them.

## Layout

```
dsa_cpu/config.py       DSAConfig with GLM-5.2 field names, GLM52_SMALL preset
dsa_cpu/synthetic.py    seeded weights and activations per operating point
dsa_cpu/rope.py         interleaved rotary
dsa_cpu/indexer.py      Indexer: projections wired, forward unimplemented
dsa_cpu/sparse_attn.py  sparse_mla_attention: unimplemented
dsa_cpu/mla.py          dense_mla_attention: the path that exists today
dsa_cpu/attention.py    dsa_attention entry point and the sparse_available probe
bench/benchmark.py      artemis_results.json
tests/                  goldens and gate
docs/DSA_SPEC.md        the maths, the contract, the SGLang file mapping
```
