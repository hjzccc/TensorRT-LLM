# NVFP4 Sub-Format Compression — Exploration Log

**Goal:** Compress NVFP4 quantized MoE expert weights below 4 bits/element while preserving valid FP4 codes for Blackwell tensor cores.

**Model:** Qwen3.5-35B-A3B, MoE expert weights only (gate_up_proj, down_proj)
**Baseline:** BF16 PPL = 6.5896 | NVFP4 PPL = 6.8431
**Evaluation:** WikiText-2 test, 145 chunks × 2048 tokens

---

## [Phase 1] Setup & Pipeline Fix — COMPLETED ✓

**Status:** COMPLETE

**Problem:** Previous `sub_nvfp4_compression.py` had a BF16-roundtrip bug: it worked in BF16 space, computed its own block scales, snapped to sub-codebook, denormalizes back to BF16, then fed to `nvfp4_linear` which RE-QUANTIZES with new block scales. The block scales from re-quantization differ from the original, especially for codebooks that change the block maximum.

**Solution Implemented:**
1. ✓ Discovered exact FP4 packed format (nibble order, byte layout)
   - Two 4-bit codes per byte: [HIGH_NIBBLE (bits 7-4) | LOW_NIBBLE (bits 3-0)]
   - Even index → LOW nibble, Odd index → HIGH nibble
   - No interleaving, row-major layout

2. ✓ Implemented unpack/repack functions for packed FP4 (`fp4_utils.py`)
   - `unpack_fp4()`: Packed bytes → individual 4-bit codes
   - `repack_fp4()`: 4-bit codes → packed bytes
   - Round-trip verified: unpack → repack recovers original exactly

3. ✓ Built code-space mapping framework (`FP4Codebook` class)
   - Maps any 4-bit code to nearest codebook code by E2M1 value distance
   - Supports arbitrary codebook subsets (3-bit, 2-bit, etc.)
   - Pre-defined codebooks: 3bit_uniform, 3bit_adaptive, 2bit_uniform, 2bit_optimal

4. ✓ Implemented correct evaluation pipeline (`eval_pipeline.py`)
   - Pipeline: `fp4_quantize(orig_weight)` → unpack → map codes → repack → kernel with ORIGINAL block_scales
   - Validated: identity mapping preserves codes exactly (handles negative zero edge case)
   - Ready for compression experiments

**Key Insight:** Code 8 (negative zero) maps to code 0 (positive zero) because they have the same magnitude. This is expected and correct behavior.

**Files Created:**
- `scripts/nvfp4_compress/fp4_utils.py` (380 lines)
- `scripts/nvfp4_compress/eval_pipeline.py` (250 lines)

**Tests Passing:**
- ✓ Pack/unpack round-trip (exact)
- ✓ Codebook mapping (all codes in codebook)
- ✓ Identity mapping validation (preserves codes)
- ✓ Pipeline validation (correct flow)

**Next:** Phase 2 — Compression experiments (3-bit, 2-bit codebooks)

---

## [Phase 2] Per-Block-16 Codebook Compression — READY TO START

**Plan:** Test different codebook strategies at block-16 (the sweet spot from entropy analysis).

**Key Data from Entropy Analysis:**
- Mean per-block entropy: 3.095 bits/elem (19.8% below global 3.857)
- Mean unique codes per block: 9.72
- 16% of blocks use ≤8 unique codes (perfect for 3-bit)
- 0.8% use ≤4 unique codes (perfect for 2-bit)

**Experiments to Run (in order):**

### 2.1: Baseline Validation (Identity Mapping) — COMPLETED ✓
- **Approach:** Run full model with identity codebook (all 16 codes), code-space pipeline
- **Result:** PPL = 6.6974 (MoE-only NVFP4; the 6.8431 baseline is full-model NVFP4 including attention)
- **Time:** 620s
- **Verdict:** Pipeline verified correct. MoE-only baseline = 6.6974 (57.5% recovery vs BF16)
- **Note:** This pipeline keeps attention + shared expert in BF16, matches prior `moe_only` scope

### 2.2: 3-Bit Uniform Codebook
- **Codebook:** {0, 2, 4, 5, 6, 7, 14, 15} (symmetric around zero)
- **Expected:** +0.017 PPL (from earlier experiments)
- **Bits/elem:** 3.31 bits (3 bits code + 0.31 bits overhead)
- **Purpose:** Simplest compression, validate codebook approach

### 2.3: 3-Bit Adaptive Codebook
- **Codebook:** {0, 1, 2, 4, 6, 7, 14, 15} (includes 1 instead of 5)
- **Expected:** -0.009 PPL (beats NVFP4!)
- **Bits/elem:** 3.37 bits
- **Purpose:** Test if adaptive selection improves over uniform

### 2.4: 2-Bit Uniform Codebook
- **Codebook:** {0, 4, 6, 15} (extreme values)
- **Expected:** +0.6 PPL (hard floor from earlier experiments)
- **Bits/elem:** 2.31 bits
- **Purpose:** Understand 2-bit limits

### 2.5: 2-Bit Optimal Codebook
- **Codebook:** {0, 2, 6, 15} (different selection)
- **Expected:** +0.5-0.6 PPL
- **Bits/elem:** 2.31 bits
- **Purpose:** Test if different 2-bit selection helps

**Evaluation Setup:**
- Use `eval_pipeline.py` with each codebook
- Run on full model (all 40 layers, all 256 experts)
- Measure PPL on WikiText-2 test (145 chunks)
- Log: codebook name, PPL, bits/elem, time

**Success Criteria:**
- 3-bit uniform: <0.02 PPL degradation
- 3-bit adaptive: beats NVFP4 baseline
- 2-bit: understand the hard floor

---

## [Phase 3] Per-Block Optimal Codebook Selection

**Plan:** For each block, find the best K-element subset of FP4 codes.

**Approach:**
1. For each block of 16 elements:
   - Enumerate all C(15,8)=6435 possible 8-code subsets (for 3-bit)
   - For each subset, compute MSE vs original codes
   - Select subset with lowest MSE
   - Store codebook ID (if using library) or codebook itself (if per-block)

2. Codebook overhead accounting:
   - Per-block codebook of 8 codes: 8×4 bits = 32 bits = 2 bits/elem overhead. Total: 3+2=5 bits (WORSE!)
   - Shared library of 256 codebooks: 8 bits per block = 0.5 bits/elem. Total: 3+0.5=3.5 bits
   - Shared library of 16 codebooks: 4 bits per block = 0.25 bits/elem. Total: 3+0.25=3.25 bits

**Key Insight:** Codebook overhead can easily negate savings. Library approach is essential.

**Experiments:**
- 3.1: Per-block optimal 8-code (3-bit) with library of 256 codebooks
- 3.2: Per-block optimal 4-code (2-bit) with library of 256 codebooks
- 3.3: Hierarchical (layer-level library + per-block selector)

---

## [Phase 4] Quantized Compression & Optimization

**Plan:** Apply secondary quantization techniques to FP4 codes.

**Approaches:**
- Lattice VQ on FP4 codes (constrained to valid FP4 values)
- Product quantization with FP4 constraint
- GPTQ-style compensation in code space
- Adaptive block scaling (Four-Over-Six style) at sub-codebook level

---

## Decision Log

**Decision 1: Block size for codebook**
- Chosen: Block-16 (matches NVFP4 block, 3.095 bits entropy)
- Rejected: Block-32 (worse entropy 3.463 bits, crosses block scale boundaries)

**Decision 2: Codebook selection algorithm**
- Phase 2: Fixed codebooks (uniform, adaptive)
- Phase 3: Per-block optimal (exhaustive search)
- Phase 4: Learned codebooks (if needed)

**Decision 3: Side information budget**
- Phase 2: None (fixed codebooks, no overhead)
- Phase 3: Library approach (0.25-0.5 bits/elem overhead)

---

## Reference Material

### Key Experimental Findings
- Per-block entropy at block-16: 3.095 bits/elem (19.8% below global marginal of 3.857)
- Per-block entropy at block-32: 3.463 bits/elem (worse)
- 3-bit uniform ±{0,2,4,6}: +0.017 PPL at 3.31 bits/elem
- Adaptive 3-bit (4/6): -0.009 PPL (beats NVFP4!) at 3.37 bits/elem
- 2-bit hard floor at +0.6 PPL
- Stochastic rounding: 3x lower MSE → 25x worse PPL (avoid!)
- Sparse corrections: PPL 113 (avoid!)

### Related Papers
- **Four Over Six** (2512.02010): Adaptive block scaling for NVFP4
- **BOF4** (2505.06653): EM-optimized codebook + outlier-preserving quantization
- **GLVQ** (2510.20984): Per-group learned lattice codebooks
- **AQLM** (2401.06118): Additive multi-codebook VQ
- **QuIP#** (2402.04396): E8 lattice + Hadamard incoherence

