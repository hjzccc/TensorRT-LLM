# DeepSeek-V3 Technical Report

## Paper Metadata
- **Title**: DeepSeek-V3 Technical Report
- **Authors**: DeepSeek-AI
- **Venue**: arXiv preprint
- **arXiv ID**: arXiv:2412.19437v2 [cs.CL]
- **Year**: 2025

---

## Problem

Training and deploying a frontier-quality open-source LLM at MoE scale requires solving three intertwined problems: (1) KV-cache memory grows quadratically with sequence length and model size, making large-context inference expensive; (2) standard auxiliary-loss-based load balancing for MoE routing degrades model quality by distorting the training objective; (3) cross-node all-to-all communication for expert dispatch becomes a bottleneck that limits training throughput on large GPU clusters.

---

## Approach

DeepSeek-V3 is a 671B-parameter sparse MoE Transformer with 37B parameters activated per token, trained on 14.8T tokens using 2048 H800 GPUs. It introduces four core technical contributions.

**Multi-Head Latent Attention (MLA)** compresses the KV cache by projecting keys and values through a shared low-rank bottleneck. For each token `h_t`, a latent vector `c_t^{KV} = W^{DKV} h_t` is computed, from which keys and values are recovered as `k_t^C = W^{UK} c_t^{KV}` and `v_t^C = W^{UV} c_t^{KV}`. Queries are similarly compressed: `c_t^Q = W^{DQ} h_t`, then `q_t^C = W^{UQ} c_t^Q`. Decoupled RoPE embeddings `k_t^R` and `q_t^R` are appended to handle positional encoding. At inference, only `c_t^{KV}` and `k_t^R` are cached per token, dramatically shrinking KV cache size versus standard MHA.

**DeepSeekMoE with auxiliary-loss-free load balancing** replaces the standard auxiliary loss with a per-expert routing bias `b_i`. During top-K selection, the score used for routing is `s_{i,t} + b_i`, but the actual gate weight applied to the expert output uses only the original affinity `s_{i,t}`. After each training step, `b_i` is decreased by `gamma` for overloaded experts and increased by `gamma` for underloaded ones. This keeps batch-level load balance without corrupting the gradient signal. Each MoE layer has 1 shared expert plus 256 routed experts, with top-8 selected per token and each token sent to at most 4 nodes.

**Multi-Token Prediction (MTP)** adds D sequential prediction heads that each predict one additional future token. At depth k, the module combines the previous depth's hidden state with the embedding of the next token: `h_i'^k = M_k [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]`, then passes through a transformer block. The MTP loss `L_MTP = (lambda/D) sum_k L_MTP^k` is added to the main language modeling loss. MTP modules are discarded at standard inference but can be reused as speculative decoding draft heads.

**DualPipe training pipeline** overlaps forward/backward computation with all-to-all and pipeline communication, achieving near-zero exposed communication overhead. FP8 mixed-precision training is used for most GEMMs, with fine-grained quantization (activations in 1x128 tiles, weights in 128x128 blocks) and periodic FP32 accumulation on CUDA cores to prevent precision loss.

---

## Key Results

- **Training cost**: 2.788M H800 GPU hours total (~$5.576M at $2/hour), with FP8 relative loss error below 0.25% versus BF16 baseline.
- **Benchmark performance**: DeepSeek-V3 achieves 90.2 on MATH-500, 82.6 on HumanEval-Mul, 42.0% on SWE-Bench Verified, and 85.5 on Arena-Hard, matching or exceeding GPT-4o and Claude-3.5-Sonnet on most benchmarks as an open-source model.
- **MTP speculative decoding**: second-token acceptance rate of 85-90%, yielding **1.8x TPS** improvement in production decode.

---

## Relevance

DeepSeek-V3 is the primary open-source MoE model target for mixed-precision inference on Blackwell GPUs; its 671B/37B-active architecture, FP8 training pipeline, node-limited routing (max 4 nodes per token), and MLA KV-cache design directly define the workload characteristics that a Blackwell dual-tile MoE inference kernel must handle.
