# "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization

## Paper Metadata

- **Title**: "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization
- **Authors**: Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, Dan Alistarh
- **Venue**: arXiv preprint (no conference venue listed)
- **arXiv ID**: arXiv:2411.02355v3 [cs.LG]
- **Year**: 2025

---

## Problem

Practitioners deploying large LLMs face a confusing landscape of quantization formats with no clear empirical guidance on when each is appropriate. Prior work often evaluates formats in isolation, on limited benchmarks, or without measuring real serving costs. This paper asks: across model sizes, workload types, and GPU generations, which quantization format (FP8 W8A8, INT8 W8A8, or INT4 W4A16) actually wins on accuracy, latency, throughput, and cost per query?

---

## Approach

The authors evaluate three deployment-ready quantization formats on vLLM (v0.6.4.post1) using Llama-3.1-Instruct at 8B, 70B, and 405B, plus DeepSeek-R1-Distill models for reasoning tasks. All formats share the same base quantization formulation:

```
Q(x, b) = rnd((x - z(x)) / s(x))
z(x) = min(x),  s(x) = (max(x) - min(x)) / (2^b - 1)
```

**W8A8-FP** uses symmetric per-output-channel weight quantization and dynamic per-token activation quantization in 8-bit floating point, requiring no calibration data. It targets Hopper and Ada Lovelace hardware.

**W8A8-INT** applies GPTQ with symmetric per-output-channel weight quantization and dynamic per-token INT8 activation quantization. For 70B models where naive INT8 hurts accuracy, SmoothQuant migrates activation difficulty into weights before quantization. This format targets Ampere and older GPUs.

**W4A16-INT** keeps activations in 16-bit and quantizes weights to 4-bit integers using GPTQ with MSE-optimal clipping in groups of 128 elements, calibrated on OpenPlatypus data. The authors compare GPTQ vs. AWQ for this format and find GPTQ is better on real-world tasks (coding, Arena-Hard) despite AWQ being marginally better on academic benchmarks.

Accuracy is measured across academic benchmarks (Open LLM Leaderboard V1/V2), real-world tasks (Arena-Hard, HumanEval, RULER), reasoning (AIME 2024, MATH-500, GPQA-Diamond), and output-level text similarity (ROUGE, BERTScore, STS). Efficiency is measured as latency, QPS, and cost per query across synchronous and asynchronous serving workloads on A6000, A100, and H100 GPUs.

---

## Key Results

- **Accuracy**: All three formats recover approximately 99% of BF16 accuracy on average across benchmarks and model sizes. On Llama-3.1-405B Leaderboard V1, W8A8-FP scores 86.89 vs. BF16's 86.79; W4A16-INT scores 86.78. Accuracy differences are within noise for most tasks.
- **Cost efficiency (synchronous)**: W4A16-INT cuts cost per query by 5-7x for 405B, enabling deployment on 4x A100/H100 instead of 16, with code-completion latency dropping from 81.9s to 48.9s on A100.
- **Throughput (asynchronous)**: W8A8 formats win at high throughput; on 405B with 16x H100, FP8 reaches 20.7 QPS and INT4 reaches 24.7 QPS vs. BF16's 8.5 QPS. The crossover point between W4A16 and W8A8 depends on workload latency target.

---

## Relevance

This paper provides the empirical foundation for choosing between FP8, INT8, and INT4 in mixed-precision MoE inference on Blackwell GPUs, showing that format selection should be driven by workload type (latency-sensitive vs. throughput-bound) and GPU generation rather than accuracy concerns alone.
