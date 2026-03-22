# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

## Paper Metadata

- **Title**: SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
- **Authors**: Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
- **Venue**: arXiv preprint
- **arXiv ID**: arXiv:2512.14080v1
- **Year**: 2025

---

## Problem

Modern MoE language models are trending toward finer granularity (smaller expert intermediate size `n`) and higher sparsity (larger total expert count `E` with constant active experts `K`). Both trends hurt training efficiency: arithmetic intensity drops as `AI = 3 / ((2 + 2G)/d + 3/(T*rho))` where `G = d/n` is granularity and `rho = K/E` is activation ratio, pushing computation into the memory-bound regime. Existing grouped GEMM kernels (ScatterMoE, MoMoE, MegaBlocks, DeepGEMM) also waste FLOPs through tile-quantization padding when expert token counts are small, and their backward passes cache large intermediate tensors that scale poorly with granularity.

---

## Approach

SonicMoE addresses these problems through three coordinated contributions: a memory-efficient backward algorithm, an IO-aware 8-kernel pipeline, and a tile-aware routing strategy called token rounding.

**Memory-efficient backward.** The standard backward pass caches per-expert outputs `Y` and gathered inputs `X_e`, which scale as `O(T*K*d)` and grow with granularity. SonicMoE avoids materializing these tensors by reordering the backward computation. For the down-projection backward, it defines `dA'_e = dO_e * W_{2,e}^T` and observes that the router-score gradient can be computed as `dS_{t,e} = <dA'_{e,t}, A_{e,t}>` without ever forming `Y`. The weight gradient becomes `dW_{2,e} = (Broadcast(s_e) * A_e)^T * dO_e`. This reduces per-layer activation memory to `2*T*d + 4*T*K*n` bytes, which is constant with respect to granularity and matches the theoretical minimum without GEMM recomputation.

**IO-aware kernel pipeline.** The 8-kernel design (3 forward, 5 backward) fuses gather operations directly into GMEM-to-SMEM loads via TMA/`cp.async`, eliminating separate gather-and-pad kernels. The forward up-projection fuses SwiGLU into the GEMM epilogue; the backward `dH` kernel fuses `dSwiGLU`, `dS`, and the `A'` computation into a single epilogue, saving `2*T*K*d` bytes of HBM traffic. For output storage, SonicMoE writes per-expert outputs contiguously with TMA and aggregates them in a separate kernel, explicitly avoiding fused scatter stores in the epilogue because synchronous `st.global` on Hopper blocks the next MMA wave. Ping-pong warpgroup scheduling overlaps asynchronous TMA loads with MMA execution for compute-heavy epilogues.

**Token rounding (TR).** In sparse MoE, expert token counts `f_e` are rarely multiples of the GEMM tile size `M_tile`, causing padding waste. TR adjusts routing so each expert receives a token count that is a multiple of `M_tile`. After standard top-K selection, TR rounds each expert's count to the nearest multiple of `M_tile` by either dropping some top-K tokens or adding extra non-top-K tokens, with the constraint that deviation from original top-K is bounded by one tile per expert. The default policy ("NR-f") rounds to the nearest multiple by expert frequency.

---

## Key Results

- **1.86x compute throughput** over ScatterMoE BF16 on H100 for a fine-grained 7B MoE (`n=256`, `E=128`, `K=8`); forward time 1.237 ms vs. ScatterMoE 2.255 ms, backward 2.173 ms vs. 3.968 ms.
- **45% activation memory reduction** for the fine-grained 7B MoE vs. ScatterMoE; at 120B scale, saves more than 3 GiB per layer vs. MoMoE.
- Token rounding adds up to **1.16x kernel speedup** in highly sparse settings (e.g., 25.7% forward TFLOPS gain at `K/E = 1/128`) while maintaining equivalent downstream model quality (validation perplexity within 0.1 nats).

---

## Relevance

SonicMoE's IO-aware grouped GEMM design, TMA-based gather fusion, and tile-quantization-aware routing are directly applicable to mixed-precision MoE inference on Blackwell GPUs, where fine-grained expert configurations and FP8/BF16 mixed-precision kernels face the same arithmetic-intensity and padding-waste challenges.
