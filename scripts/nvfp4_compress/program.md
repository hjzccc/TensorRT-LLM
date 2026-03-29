# NVFP4 Sub-Format Compression

## Direction

We want to compress NVFP4 quantized weights below 4 bits while keeping the decompressed output as valid FP4 codes for Blackwell tensor core compute. Two directions to explore:

1. **Pure compression**: Take the 16 FP4 codes within each NVFP4 block (or across blocks), find a per-block codebook (subset of the 16 FP4 values), and store indices into that codebook. Different blocks may use different codebooks. The codebook itself must be stored as side information. Explore different block sizes and block combinations.

2. **Quantized compression**: Apply a secondary quantization (like lattice VQ, product quantization, or learned codebooks inspired by GLVQ/AQLM/QuIP#) to the FP4 codes, but constrain the output to always reconstruct valid FP4 values. This is harder but could exploit cross-element correlations that pure compression misses.

For both directions: the decompressed result must be one of the 16 FP4 E2M1 values {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}. The block scales (FP8 E4M3) and global scale (FP32) are preserved unchanged from the original NVFP4 quantization.

Key insight from entropy analysis: per-block Shannon entropy at block-16 is **3.095 bits/elem** (mean across 2 billion blocks), much lower than the global marginal of 3.857. Average block uses only 9.72 of 15 codes. 16% of blocks use ≤8 codes (3-bit representable). Block-16 is the sweet spot — block-32 entropy rises to 3.463 because it spans two different block scales.

## Evaluation approach

We measure **end-to-end accuracy** (zero-shot benchmarks), not just perplexity. This shows real-world impact of compression.

**Comparison**: Original NVFP4 model accuracy vs. compressed-then-decompressed model accuracy.

**Setup**:
- **Base model**: `Qwen/Qwen3.5-35B-A3B` (BF16, already downloaded in docker)
- **Quantization**: We quantize BF16 → NVFP4 ourselves using `torch.ops.trtllm.fp4_quantize`, layer by layer (BF16 model is ~70GB, doesn't fit in 32GB GPU at once). The quantized result is repacked into a TRT-LLM-native NVFP4 checkpoint (~22GB) that fits entirely in GPU memory.
- **Inference engine**: TRT-LLM with the NVFP4 checkpoint (auto_deploy or engine build — whichever works first)
- **Evaluation harness**: `lm-evaluation-harness` (EleutherAI), already installed
- **Benchmarks**: MMLU (zero-shot), GSM8K (zero-shot)
- **GPU**: RTX 5090 (32GB, Blackwell) — fits the entire NVFP4 model in memory
- **Docker container**: `trtllm-dual-tile` with TRT-LLM 1.3.0rc3, lm-eval pre-installed

**Workflow**:
1. Quantize: Load BF16 model layer by layer → `fp4_quantize` each weight → save packed FP4 codes + scales → repack as TRT-LLM-native NVFP4 checkpoint
2. Run baseline: Load NVFP4 checkpoint → TRT-LLM → lm-eval → MMLU/GSM8K scores
3. Compress: Load NVFP4 checkpoint → compress to sub-4-bit format (per-element indices + per-block codebook identity)
4. Decompress: Expand compressed format back to standard NVFP4 checkpoint (valid FP4 codes + original scales)
5. Run experiment: Load decompressed checkpoint → TRT-LLM → lm-eval → MMLU/GSM8K scores
6. Compare: Accuracy delta from compression

## Constraints

Things that are **fixed** (do not change these):
- Decompressed values must be valid FP4 E2M1 codes from the set {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}
- Block scales (FP8 E4M3, one per 16 elements) are preserved from original NVFP4, never recomputed
- Global scale (FP32, one per tensor) is preserved from original NVFP4, never recomputed
- Starting model: `Qwen/Qwen3.5-35B-A3B` (BF16), quantized to NVFP4 by us using `fp4_quantize`
- Evaluation: MMLU and GSM8K, zero-shot, via lm-evaluation-harness
- Inference: TRT-LLM with our NVFP4 checkpoint (auto_deploy or engine build — whichever works first)
- No retraining or fine-tuning — strictly post-training compression
- Compute pipeline: compressed codes → decompress to FP4 → feed to NVFP4 tensor cores on Blackwell

Things that are **flexible** (explore freely):
- Block size for codebook: **16 is the sweet spot** (matches NVFP4 block, 3.095 bits entropy). Block-32 is worse (3.463 bits, crosses block scale boundaries).
- Codebook selection algorithm (exhaustive search, k-means, learned)
- Number of codebooks (1 global, per-layer, per-expert, per-block)
- Side information budget (codebook metadata bits per block)
- Whether to use uniform or non-uniform code subsets
- Compression target (2-bit, 3-bit, or fractional)
- Which weights to compress (all quantized weights, or MoE experts only)

Things to **avoid**:
- Never recompute block scales or global scale from compressed weights — use originals
- Never produce non-FP4 values that require re-quantization
- Don't use stochastic rounding (proven 25x worse PPL despite 3x lower MSE)
- Don't flip individual rounding directions (sparse corrections destroy coherent error)
- Don't use block-32 for codebook (worse entropy than block-16 because it spans two block scales)

## Success criteria

How you know an approach is working (in rough priority order):
1. Minimal accuracy degradation on MMLU/GSM8K vs the original NVFP4 model
2. Effective bits per element (including all side information: codebook metadata, indices, scales)
3. Decompression is simple and fast (table lookup or matrix multiply, not iterative)
4. The approach works on the full model, not just a subset
5. Results are reproducible (deterministic, no random seeds affecting output)

## Setup

1. **Create the branch**: `git checkout -b explore/nvfp4-compress` from current main.
2. **Read existing context**: Before writing any code, read all files currently in the repo to understand what exists.
   Key files (all paths relative to repo root, docker prefix = `/code/tensorrt_llm/`):
   - `scripts/nvfp4_compress/program.md` — this document
   - `scripts/nvfp4_compress/exploration.md` — exploration log (create when starting)
   - `scripts/nvfp4_compress/per_block_entropy.py` — per-block Shannon entropy analysis (verified, fast)
   - `scripts/nvfp4_compress/code_entropy_fast.py` — global marginal + pairwise entropy (verified)
   - `scripts/nvfp4_compress/sub_fp4_compress.py` — FP4 pack/unpack utilities (round-trip verified)
   Docker mount: host `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/` → docker `/code/tensorrt_llm/`
3. **Quantize and repack**: Write a script that loads the BF16 model layer by layer, quantizes each weight tensor with `torch.ops.trtllm.fp4_quantize`, and saves the packed FP4 codes + block scales + global scales as a TRT-LLM-native NVFP4 checkpoint. The output checkpoint (~22GB) should be loadable by TRT-LLM auto_deploy as a whole model.
   - **Reference**: Study [`nvidia/Qwen3-30B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3-30B-A3B-NVFP4) — NVIDIA's official NVFP4 quantization of a similar Qwen3 MoE model. Replicate its safetensors layout (weight keys, scale keys, dtypes, config.json) so TRT-LLM loads it natively via `LLM(model=...)`. Key details: quantized with `nvidia-modelopt v0.31.0`, only linear weights in transformer blocks are quantized, deploys with `tensorrt_llm.LLM(model="nvidia/Qwen3-30B-A3B-FP4")`. Note: that model uses `model_type: qwen3_moe` (Qwen3, in transformers 4.57.1); our model `Qwen/Qwen3.5-35B-A3B` uses `model_type: qwen3_5_moe` (Qwen3.5, NOT in transformers 4.57.1) — use the reference for checkpoint format only, the model_type will need to be resolved separately.
   - Strip the `model.language_model.` prefix from weight keys (the BF16 model is a multimodal wrapper; we only want the text model).
4. **Run baseline**: Load the NVFP4 checkpoint with TRT-LLM, run lm-eval MMLU/GSM8K, record baseline scores.
5. **Confirm and go**: Confirm baseline scores, then begin exploring compression.

## How to explore

You are an autonomous researcher/engineer. Your job is to try things, learn from what works, and converge on the best approach. Here's how:

### Phase 1: Build the compression / decompression pipeline

The goal is to produce a **sub-4-bit compressed format** on disk, and a decompressor that reconstructs valid 4-bit FP4 codes at inference time.

1. **Compress** (`compress_checkpoint.py`):
   - Loads the NVFP4 checkpoint's packed FP4 codes (4 bits/element)
   - For each block of 16 elements, finds the best sub-codebook (subset of the 16 FP4 values) and encodes each element as an index into that sub-codebook
   - Saves the compressed representation: per-element indices (e.g., 2 bits each) + per-block codebook identity (e.g., 1 bit selecting from a small library of codebooks)
   - Example: 3-bit format = 2-bit index per element + 1-bit codebook selector per 16-element block → 2 + 1/16 = 2.0625 bits/element
   - All original scales (block scales, global scale) are preserved unchanged

2. **Decompress** (`decompress_checkpoint.py`):
   - Reads the compressed checkpoint
   - For each block: look up the codebook, expand indices back to 4-bit FP4 codes
   - Repacks as standard NVFP4 (4 bits/element) that tensor cores can consume
   - Output is a normal NVFP4 checkpoint loadable by TRT-LLM

3. **Verify**: compress → decompress → load with TRT-LLM → scores match the original NVFP4 baseline (for identity/lossless codebook).

## The exploration loop

LOOP FOREVER:

1. **Decide what to try next.** Look at `exploration.md` and current results. What's the most impactful next experiment?

2. **Do the work.** Write code, modify checkpoints, run experiments.

3. **Test it.** Load the compressed checkpoint with TRT-LLM and run lm-eval MMLU/GSM8K.

4. **Log what happened.** Append to `exploration.md`:
   ```
   ## [N] Short Title
   **Approach**: What you tried
   **Result**: MMLU=XX.X%, GSM8K=XX.X%, bits/elem=X.XX
   **vs Baseline**: MMLU Δ=±X.X%, GSM8K Δ=±X.X%
   **Verdict**: keep / discard / pivot
   **Next**: What this suggests trying next
   ```

5. **Git commit.** Commit working code with descriptive message.

6. **Repeat.** Go back to step 1.

### Decision-making guidelines

- **When in doubt, try it.** A quick experiment beats speculation.
- **The pipeline must be correct.** Always operate on actual FP4 codes with preserved scales. Never re-quantize.
- **Accuracy > proxy metrics.** Always evaluate with MMLU/GSM8K, not just MSE or PPL.
- **Coherent error > random error.** Deterministic, consistent rounding decisions are better than "optimal" per-element choices that break error coherence.
- **When stuck, read the papers.** 25 related work PDFs are in `scripts/channel_quant_new/papers/`. Key papers for codebook design: GLVQ (per-group learned lattice), AQLM (additive multi-codebook), BOF4 (EM-optimized codebook), QuIP# (lattice codebook), Four Over Six (adaptive block scaling). Read the actual methods sections — the ideas often transfer even when the format differs.

### Autonomy rules

**NEVER STOP** to ask the human if you should continue. The human may be away and expects you to keep exploring until manually stopped.

## Reference material

### Key findings from entropy analysis (verified, from `per_block_entropy.py` and `code_entropy_fast.py`)
- Per-block entropy at block-16: 3.095 bits/elem (19.8% below global marginal of 3.857)
- Per-block entropy at block-32: 3.463 bits/elem (worse — crosses block scale boundaries)
- Pairwise MI between adjacent elements: 0.0075 bits (negligible)
- Average block uses only 9.72 of 15 codes; 16% of blocks use ≤8 codes (3-bit representable)

### Earlier PPL experiments (unreliable — used buggy BF16-roundtrip pipeline, treat as directional only)
- 3-bit uniform ±{0,2,4,6}: +0.017 PPL (BF16-roundtrip with adapted scales — NOT frozen scales)
- Code-space pipeline with frozen scales: 3-bit uniform causes ~20% relative error per linear, compounds to garbage across 40 layers
- Stochastic rounding: 3x lower MSE → 25x worse PPL
- Lesson: frozen block scales + aggressive sub-codebook = unusable. The compress/decompress approach avoids this by preserving exact codes within each sub-codebook.

### Papers (in `scripts/channel_quant_new/papers/`)
- **Four Over Six** (2512.02010): Adaptive block scaling for NVFP4
- **BOF4** (2505.06653): EM-optimized codebook + outlier-preserving quantization
- **GLVQ** (2510.20984): Per-group learned lattice codebooks
- **AQLM** (2401.06118): Additive multi-codebook VQ
- **QuIP#** (2402.04396): E8 lattice + Hadamard incoherence
- **Float8@2bits** (2601.22787): Entropy coding of Float8 weights to 2 bits effective