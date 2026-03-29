# NVFP4 Sub-Format Compression

## Direction

We want to compress NVFP4 quantized weights below 4 bits while keeping the decompressed output as valid FP4 codes for Blackwell tensor core compute. Two directions to explore:

1. **Pure compression**: Take the 16 FP4 codes within each NVFP4 block (or across blocks), find a per-block codebook (subset of the 16 FP4 values), and store indices into that codebook. Different blocks may use different codebooks. The codebook itself must be stored as side information. Explore different block sizes and block combinations.

2. **Quantized compression**: Apply a secondary quantization (like lattice VQ, product quantization, or learned codebooks inspired by GLVQ/AQLM/QuIP#) to the FP4 codes, but constrain the output to always reconstruct valid FP4 values. This is harder but could exploit cross-element correlations that pure compression misses.

For both directions: the decompressed result must be one of the 16 FP4 E2M1 values {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}. The block scales (FP8 E4M3) and global scale (FP32) are preserved unchanged from the original NVFP4 quantization.

Key insight from entropy analysis: per-block Shannon entropy at block-16 is **3.095 bits/elem** (mean across 2 billion blocks), much lower than the global marginal of 3.857. Average block uses only 9.72 of 15 codes. 16% of blocks use ≤8 codes (3-bit representable). Block-16 is the sweet spot — block-32 entropy rises to 3.463 because it spans two different block scales.

## Constraints

Things that are **fixed** (do not change these):
- Decompressed values must be valid FP4 E2M1 codes from the set {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}
- Block scales (FP8 E4M3, one per 16 elements) are preserved from original NVFP4, never recomputed
- Global scale (FP32, one per tensor) is preserved from original NVFP4, never recomputed
- Model: Qwen3.5-35B-A3B, MoE expert weights only (gate_up_proj and down_proj)
- Evaluation: WikiText-2 test set perplexity, full 145 chunks, all 40 layers
- Calibration: WikiText-2 train split, 128 samples (if calibration needed)
- No retraining or fine-tuning — strictly post-training compression
- Compute pipeline: compressed codes → decompress to FP4 → feed to NVFP4 tensor cores on Blackwell
- Docker container: trtllm-phase12 with TRT-LLM and exact NVFP4 kernels

Things that are **flexible** (explore freely):
- Block size for codebook: **16 is the sweet spot** (matches NVFP4 block, 3.095 bits entropy). Block-32 is worse (3.463 bits, crosses block scale boundaries).
- Codebook selection algorithm (exhaustive search, k-means, learned)
- Number of codebooks (1 global, per-layer, per-expert, per-block)
- Side information budget (codebook metadata bits per block)
- Whether to use uniform or non-uniform code subsets
- Compression target (2-bit, 3-bit, or fractional)

Things to **avoid**:
- Never recompute block scales or global scale from compressed weights — use originals
- Never produce non-FP4 values that require re-quantization
- Don't use stochastic rounding (proven 25x worse PPL despite 3x lower MSE)
- Don't flip individual rounding directions (sparse corrections destroy coherent error, PPL 113)
- Don't use full SVD per expert (too slow — 47 min per layer)
- Don't use block-32 for codebook (worse entropy than block-16 because it spans two block scales)
- Don't measure joint pattern entropy at block≥8 (every pattern unique — only per-block Shannon entropy is meaningful)

## Success criteria

How you know an approach is working (in rough priority order):
1. Nearly PPL degradation vs NVFP4 baseline for both 3 bits and 2 bits.
2. Effective bits per element (including all side information: codebook metadata, indices, scales)
3. Decompression is simple and fast (table lookup or matrix multiply, not iterative)
4. The approach works on the full model (all 40 layers, all 256 experts), not just a subset
5. Results are reproducible (deterministic, no random seeds affecting output)

**Test cases to validate:**
- Layer 0 (sparse routing, many zero-weight experts)
- Layer 39 (largest activation magnitudes, highest error sensitivity)
- Expert 25 in layer 0 (only 186/1024 non-zero channels — edge case)
- gate_up_proj (shape [1024, 2048]) and down_proj (shape [2048, 512]) — different aspect ratios

## Setup

1. **Create the branch**: `git checkout -b explore/nvfp4-compress` from current main.
2. **Read existing context**: Before writing any code, read all files currently in the repo to understand what exists.
   Key files (all paths relative to repo root, docker prefix = `/code/tensorrt_llm/`):
   - `scripts/nvfp4_compress/program.md` — this document
   - `scripts/nvfp4_compress/exploration.md` — exploration log (create when starting)
   - `scripts/nvfp4_compress/per_block_entropy.py` — per-block Shannon entropy analysis (verified, fast)
   - `scripts/nvfp4_compress/code_entropy_fast.py` — global marginal + pairwise entropy (verified)
   - `scripts/channel_quant_new/profiling/sub_nvfp4_compression.py` — previous experiments (has BF16-roundtrip bug)
   - `scripts/channel_quant_new/profiling/optimize_allocation.py` — BF16+NVFP4 allocation and PPL evaluation
   - `scripts/channel_quant_new/exact_docker_eval.py` — TRT-LLM kernel wrappers (nvfp4_linear, bf16_linear)
   - `scripts/channel_quant/spike1_ground_truth.py` — model utilities (WeightStore, layer_keys, RMSNorm)
   Docker mount: host `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/` → docker `/code/tensorrt_llm/`
3. **Check prerequisites**: Docker container `trtllm-phase12` must be running. Model cached at `/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/`. TRT-LLM installed in `/code/tensorrt_llm/`.
4. **Confirm and go**: Confirm setup looks good, then begin exploring.

## How to explore

You are an autonomous researcher/engineer. Your job is to try things, learn from what works, and converge on the best approach. Here's how:

### Phase 1: Fix the pipeline (CRITICAL — do this first)

The current implementation has a fundamental bug: it works in BF16 space and lets `ee.nvfp4_linear` re-quantize, instead of operating directly on FP4 codes with preserved scales. Fix this:

1. Start from actual NVFP4-quantized codes (use `torch.ops.trtllm.fp4_quantize` to get packed FP4 codes + block scales)
2. Unpack the FP4 codes to get the 16-valued integers per element
3. Apply sub-codebook mapping purely in code space (integer mapping, no floating point)
4. Repack the mapped codes into FP4 format
5. Feed to tensor core with the ORIGINAL block scales (never recompute)

Re-run the 3-bit uniform and 2-bit optimal experiments with the fixed pipeline to get corrected baselines.

### Phase 2: Per-block-16 codebook compression

The idea: for each NVFP4 block of 16 elements, choose the best K-element subset of the 15 FP4 values as the codebook, store indices.

**Key data from entropy analysis (2 billion blocks, all experts):**
- Mean per-block entropy: 3.095 bits/elem (theoretical floor with optimal per-block coding)
- Mean unique codes per block: 9.72
- 16% of blocks use ≤8 unique codes (perfect for 3-bit)
- 0.8% use ≤4 unique codes (perfect for 2-bit)

**Experiments to try (all at block-16):**
- **Per-block optimal 8-code codebook (3-bit)**: For each block, find the best 8 of 15 FP4 values by MSE. C(15,8)=6435 candidates per block — feasible. Store: 3-bit × 16 + codebook ID overhead.
- **Per-block optimal 4-code codebook (2-bit)**: Best 4 of 15 per block. C(15,4)=1365 candidates. Store: 2-bit × 16 + codebook ID overhead.
- **Shared codebook library**: Pre-compute K codebooks (e.g., K=256 codebooks of 8 codes each). Per block, select best from library. Cost: log₂(K)/16 bits/elem overhead. Avoids per-block codebook storage.
- **Hierarchical**: Layer-level library + per-block selector.

**Codebook overhead accounting:**
- Per-block codebook of 8 codes: 8 × 4 bits = 32 bits = 2 bits/elem overhead. Total: 3 + 2 = 5 bits (WORSE than NVFP4!)
- Shared library of 256 codebooks: 8 bits per block selector = 0.5 bits/elem. Total: 3 + 0.5 = 3.5 bits.
- Shared library of 16 codebooks: 4 bits per block = 0.25 bits/elem. Total: 3 + 0.25 = 3.25 bits.
- **Critical: codebook overhead can easily negate the savings. Library approach is essential.**

### Phase 3: Quantized compression

Apply secondary quantization techniques to FP4 codes, constrained to output valid FP4:
- **Lattice VQ on FP4 codes**: Learn a generation matrix G where G×z produces valid FP4 code vectors. Hard constraint but potentially powerful.
- **Product quantization with FP4 constraint**: Split block into sub-vectors, cluster, but constrain centroids to be vectors of valid FP4 values.
- **GPTQ-style compensation in code space**: Quantize codes left-to-right, compensate remaining codes for each error. Since codes are discrete, compensation means shifting neighboring codes to cancel the error.

### Phase 4: Combine and optimize

Take the best approaches from phases 2-3, combine with:
- Adaptive block scaling (Four-Over-Six style) applied at the sub-codebook level
- Per-layer or per-expert codebook libraries tuned on calibration data
- Mixed bit-rate across blocks based on block importance

## The exploration loop

LOOP FOREVER:

1. **Decide what to try next.** Look at `exploration.md` and the current state of the code. What's the most useful next experiment or improvement?

2. **Do the work.** Write code, run experiments in the docker container.

3. **Test it.** Run PPL evaluation:
   ```
   docker exec -d trtllm-phase12 bash -c "cd /code/tensorrt_llm && python3 -u scripts/channel_quant_new/profiling/<script>.py <args> > <logfile> 2>&1"
   ```
   Check results: `docker exec trtllm-phase12 tail -20 <logfile>`

4. **Log what happened.** Append to `exploration.md`:
   ```
   ## [N] Short Title
   **Approach**: What you tried
   **Result**: PPL, bits/elem, timing
   **Verdict**: keep / discard / pivot
   **Next**: What this suggests trying next
   ```

5. **Git commit.** Commit working code with descriptive message.

6. **Repeat.** Go back to step 1.

### Exploration log format

Maintain `exploration.md` at the repo root. This is your lab notebook.

### Decision-making guidelines

- **When in doubt, try it.** A quick experiment beats speculation.
- **Simpler is always better.** Our 3-bit uniform ±{0,2,4,6} is embarrassingly simple and nearly lossless. Beat it with something equally clean.
- **Fail fast.** If an approach isn't working after one eval run, log and move on.
- **The pipeline must be correct.** Never work in BF16 space — always operate on actual FP4 codes with preserved scales.
- **MSE is misleading.** Always evaluate with full PPL, not proxy metrics. (Stochastic rounding: 3x lower MSE → 25x worse PPL.)
- **Coherent error > random error.** Deterministic, consistent rounding decisions are better than "optimal" per-element choices that break error coherence.
- **When stuck, read the papers.** 25 related work PDFs are in `scripts/channel_quant_new/papers/`. Key papers for codebook design: GLVQ (per-group learned lattice), AQLM (additive multi-codebook), BOF4 (EM-optimized codebook), QuIP# (lattice codebook), Four Over Six (adaptive block scaling). Read the actual methods sections — the ideas often transfer even when the format differs.

### Autonomy rules

**NEVER STOP** to ask the human if you should continue. The human may be away and expects you to keep exploring until manually stopped.

## Reference material

### Papers (all downloaded in scripts/channel_quant_new/papers/)
- **Four Over Six** (2512.02010): Adaptive block scaling for NVFP4. Our adaptive 3-bit extends this.
- **RaZeR** (2501.04052): Redundant zero remapping, FP8 scale bit stealing. Shows E3M3 scales work.
- **MR-GPTQ** (2509.23202): Proves rotations hurt NVFP4. MSE-optimized scales.
- **BOF4** (2505.06653): EM-optimized codebook + outlier-preserving quantization.
- **GLVQ** (2510.20984): Per-group learned lattice codebooks with companding. SOTA 2-bit on Llama.
- **AQLM** (2401.06118): Additive multi-codebook VQ. Learned per-group codebooks.
- **QuIP#** (2402.04396): E8 lattice + Hadamard incoherence. Codebook-free VQ.
- **QTIP** (2406.11235): Trellis coded quantization. Sub-entropy coding via state memory.
- **Double Compression** (EMNLP 2024): Lossless compression on top of quantized INT8 weights.
- **Float8@2bits** (2601.22787): Entropy coding of Float8 weights to 2 bits effective.
- **INT vs FP Comprehensive** (2510.25602): Crest factor theory for NVFP4 vs NVINT4.

### Key experimental findings
- **Per-block entropy at block-16: 3.095 bits/elem** (19.8% below global marginal of 3.857). Average block uses 9.72 of 15 codes. This is the theoretical floor for per-block codebook compression.
- **Per-block entropy at block-32: 3.463 bits/elem** (worse — crosses block scale boundaries).
- **Pairwise MI between adjacent elements: 0.0075 bits** (negligible — elements are independent within blocks).
- Global marginal entropy: 3.857 bits/elem (near-uniform).
- 3-bit uniform ±{0,2,4,6}: +0.017 PPL at 3.31 bits/elem. Works because max gap (2) = NVFP4's own worst gap.
- Adaptive 3-bit (4/6): -0.009 PPL (beats NVFP4!) at 3.37 bits/elem.
- 2-bit hard floor at +0.6 PPL. PQ and adaptive multi-scale both converge there.
- Stochastic rounding: 3x lower MSE → 25x worse PPL. Coherent error is beneficial for PTQ.
- Sparse corrections: PPL 113. Flipping individual rounding directions destroys coherent error.
- Zero must be in codebook. Asymmetric codebooks beat symmetric at 2-bit.
- 3-bit truncate ±{0,1,2,4}: catastrophic (+1.125 PPL) because can't represent block max (6).
- Previous implementation bug: worked in BF16 space with scale recomputation instead of pure code-space mapping — must fix in Phase 1.
