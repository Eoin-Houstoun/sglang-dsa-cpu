# Accuracy

DSA is part of how GLM-5.2 was trained, so the reference computation is the sparse path (the GPU
implementation), not dense attention. The dense fallback is therefore both slower and, beyond
`index_topk` = 2048 tokens, a different function from the trained model.

## Layer-level fidelity, every row (this repo)

An fp32 reference of the indexer and sparse MLA (kept outside this repo) was compared against
the dense fallback and against the sparse path on `main` (Discovery attempt 18) for every query
row of every operating point. Xeon Platinum 8581C, bf16, torch 2.14. Raw numbers in
[fidelity-attempt18.json](fidelity-attempt18.json).

| operating point | rows | selection agreement mean / min | sparse rel. error mean / p99 / max | dense fallback rel. error mean / p99 / max |
|---|---|---|---|---|
| prefill 1k | 1,024 | 1.000 / 1.000 | 0.9% / 1.0% / 1.2% | 0.9% / 1.0% / 1.2% |
| prefill 2k | 2,048 | 1.000 / 1.000 | 0.8% / 1.0% / 1.1% | 0.8% / 1.0% / 1.1% |
| prefill 4k | 4,096 | 0.9995 / 0.997 | 1.9% / 7.2% / 44% | 20% / 54% / 69% |
| prefill 8k | 8,192 | 0.9987 / 0.995 | 3.1% / 10% / 29% | 40% / 65% / 83% |
| decode 8k | 4 | 0.9976 / 0.997 | 5.8% / 8.9% / 9.0% | 63% / 64% / 64% |

Selection agreement is the fraction of the reference's top-2048 positions that the path also
selected. Relative error is the per-row L2 error of the attention output against the fp32
reference, divided by the reference norm. Up to 2048 tokens sparse and dense are the same
computation and both sit at the bf16 noise floor. The sparse path's rare high-error rows are
bf16 near-ties at the top-k boundary on random data (see README, correctness gate); no row falls
below the gate's 0.97 agreement threshold.

Inputs are seeded synthetic activations at GLM-5.2's attention shape, not model activations.

## GSM8K on the real model (Intel's step, after the port into SGLang)

Launch GLM-5.2-FP8 on a Xeon 6 node with six sub-NUMA clusters, per SGLang's CPU guide, once the
CPU DSA path is in `dsa_indexer.py` / `dsa_backend.py`:

```bash
sglang serve --model-path zai-org/GLM-5.2-FP8 --trust-remote-code \
  --disable-overlap-schedule --device cpu --host 0.0.0.0 --port 30000 --tp 6
```

Then score GSM8K through the OpenAI-compatible endpoint and compare with the CUDA run of the
same model and settings:

```bash
pip install lm-eval
lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 --batch_size 8 \
  --model_args model=zai-org/GLM-5.2-FP8,base_url=http://localhost:30000/v1/completions,tokenized_requests=False
```

Report exact-match accuracy for the CPU run next to the CUDA run; the two should agree within
the run-to-run noise of 5-shot sampling on 1,319 problems (about plus or minus 1 point).
