# MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts

## Paper Metadata

- **Title**: MoQAE: Mixed-Precision Quantization for Long-Context LLM Inference via Mixture of Quantization-Aware Experts
- **Authors**: Wei Tao, Haocheng Lu, Xiaoyang Qu, Bin Zhang, Kai Lu, Jiguang Wan, Jianzong Wang
- **Venue**: arXiv preprint (no confirmed venue)
- **arXiv ID**: arXiv:2506.07533v1
- **Year**: 2025

---

## Problem

KV-cache quantization reduces memory and decoding latency for long-context LLM inference, but existing methods either apply a fixed low bit-width uniformly (hurting accuracy) or require expensive mixed-precision search procedures. The challenge is that different tokens and contexts have different sensitivity to KV-cache quantization, so a static assignment wastes precision on easy tokens while under-protecting hard ones. MoQAE learns a lightweight, trainable router that assigns per-chunk KV-cache bit-widths dynamically, balancing accuracy and memory with a tunable objective.

---

## Approach

MoQAE reframes KV-cache quantization as a routing problem. Each available bit-width configuration (e.g., FP16, INT4, INT2) is treated as a "quantization expert." A small MLP router, trained on top of frozen LLM weights, assigns each chunk of the input sequence to one of these experts.

**Chunked routing** divides the input into fixed-length chunks (default size 32 tokens). For a chunk matrix `C` of shape `[N x D]`, the router computes:

```
P = f(C W1 · C W2) W3
```

where `W1, W2 in R^{D x M}`, `W3 in R^{D x M}`, `f` is SiLU activation, and `P in R^{N x M}` gives per-token expert probabilities. The chunk-level quantization strategy is the expert selected by the majority of tokens:

```
R = argmax_k sum_{i=1}^{N} I(argmax_j p_i^j = k)
```

**Joint accuracy-memory training** optimizes only the router parameters (all LLM weights frozen) using a combined loss. The model loss weights the NLL by the selected expert's bit-width penalty:

```
L_model = (1/N) sum_i I(argmax_k p_i^k = j) * (p_i^j * L_nll / B_j)
```

The memory loss encourages lower-bit selection:

```
L_mem = (1/N) sum_i I(argmax_k p_i^k = j) * (16 p_i^j / B_j)
```

The final loss `L = lambda * L_model + (1 - lambda) * L_mem` lets `lambda` tune the accuracy-memory trade-off. Training uses only 5% of the training set.

Two efficiency optimizations reduce routing overhead. **Routing Freezing (RF)** forces the first chunk to FP16 without routing, since early tokens receive disproportionately high attention and are especially sensitive. **Routing Sharing (RS)** groups consecutive LLM blocks and shares the quantization strategy of the first block in each group, removing routers from the remaining blocks and cutting routing latency at a small accuracy cost (default group size 3).

The router itself is tiny: about 1.6 KB of parameters per attention layer, negligible compared to the KV cache.

---

## Key Results

- **WikiText2 perplexity (4-16 bit range)**: MoQAE at 4.13 average bits achieves PPL 5.76 on Llama-7B vs. 5.93 for QoQ-4b and 5.93 for AWQ-gs128, with an average increase over FP16 of only 0.08 PPL across five models.
- **LongBench**: MoQAE matches or exceeds FP16 on TriviaQA (87.89 vs. 87.72) and Qasper (9.79 vs. 9.52) while outperforming all tested baselines (KIVI-2b, CQ-4c8b, MiKV) on most datasets.
- **Memory and latency**: MoQAE-lambda0.1 reduces memory by 0.79 GB and decoding latency by 0.44 ms on average vs. state-of-the-art baselines; MoQAE-lambda0.5 reduces FP16 memory by 2.99 GB on average.

---

## Relevance

MoQAE's learned chunk-level KV-cache routing is orthogonal to weight quantization and directly applicable to Blackwell long-context inference, where KV-cache pressure is a primary bottleneck and per-chunk precision assignment can exploit the GPU's native FP8/INT4 memory hierarchy.
