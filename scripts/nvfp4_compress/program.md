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
- **Starting model**: `Sehyo/Qwen3.5-35B-A3B-NVFP4` — pre-quantized NVFP4 model in `compressed-tensors` format, calibrated with 512 samples (ultrachat + Nemotron), all experts calibrated. 75k+ downloads/month.
- **Inference engine**: vLLM (nightly) — the model is designed for vLLM's compressed-tensors support.
- **Evaluation harness**: `lm-evaluation-harness` (EleutherAI)
- **Benchmarks**: MMLU (zero-shot), GSM8K (zero-shot)
- **GPU**: RTX 5090 (32GB, Blackwell) — fits the entire NVFP4 model in memory, no layer-by-layer needed

**Workflow**:
1. Run baseline: `Sehyo/Qwen3.5-35B-A3B-NVFP4` → vLLM → lm-eval → MMLU/GSM8K scores
2. Compress: Load NVFP4 safetensors → unpack FP4 codes → apply sub-codebook mapping → repack → save modified checkpoint
3. Run experiment: Modified checkpoint → vLLM → lm-eval → MMLU/GSM8K scores
4. Compare: Accuracy delta from compression

## Constraints

Things that are **fixed** (do not change these):
- Decompressed values must be valid FP4 E2M1 codes from the set {-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6}
- Block scales (FP8 E4M3, one per 16 elements) are preserved from original NVFP4, never recomputed
- Global scale (FP32, one per tensor) is preserved from original NVFP4, never recomputed
- Starting model: `Sehyo/Qwen3.5-35B-A3B-NVFP4` (compressed-tensors format)
- Evaluation: MMLU and GSM8K, zero-shot, via lm-evaluation-harness
- Inference: vLLM with compressed-tensors support
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
   - `scripts/nvfp4_compress/sub_fp4_compress.py` — code-space FP4 compression pipeline (pack/unpack verified)
   Docker mount: host `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/` → docker `/code/tensorrt_llm/`
3. **Install prerequisites**:
   - vLLM nightly: `pip install vllm --pre` (or whatever version supports Qwen3.5 MoE + compressed-tensors)
   - lm-evaluation-harness: `pip install lm-eval`
   - Download model: `huggingface-cli download Sehyo/Qwen3.5-35B-A3B-NVFP4`
4. **Run baseline**: Get MMLU/GSM8K scores for the unmodified NVFP4 model.
5. **Confirm and go**: Confirm baseline scores, then begin exploring compression.

## How to explore

You are an autonomous researcher/engineer. Your job is to try things, learn from what works, and converge on the best approach. Here's how:

### Phase 1: Establish baselines

1. Install vLLM + lm-eval-harness in the docker container
2. Download `Sehyo/Qwen3.5-35B-A3B-NVFP4`
3. Run baseline evaluation:
   ```bash
   lm_eval --model vllm \
     --model_args pretrained=Sehyo/Qwen3.5-35B-A3B-NVFP4,trust_remote_code=True,gpu_memory_utilization=0.9 \
     --tasks mmlu,gsm8k \
     --num_fewshot 0 \
     --batch_size auto
   ```
4. Record MMLU and GSM8K scores as the baseline to beat.

### Phase 2: Build the compression pipeline

1. Understand the `compressed-tensors` format: figure out how FP4 weights are stored in the safetensors files, what keys hold quantized weights vs scales, what metadata is needed.
2. Write a script that:
   - Loads the NVFP4 checkpoint
   - Unpacks all FP4 codes (reuse `unpack_fp4_codes` / `repack_fp4_codes` from `sub_fp4_compress.py`)
   - Applies a code-space LUT (sub-codebook mapping)
   - Repacks and saves as a new checkpoint
3. Verify: identity mapping (full codebook) → modified model scores match baseline exactly.

### Phase 3: Run compression experiments

Start with the simplest codebooks and measure accuracy impact:

- **3-bit uniform ±{0,2,4,6}**: 7 unique values → 3 bits/code. Simplest, proven nearly lossless on PPL.
- **3-bit dense ±{0,1,2,6}**: Keeps fine-grained codes near zero.
- **2-bit experiments**: ±{0,2,4,6}, ±{0,3,6}, etc.
- **Per-block optimal 8-code** (3-bit): Best 8 of 15 per block (C(15,8)=6435 search).
- **Shared codebook library**: Pre-compute K codebooks, per-block selector.

For each experiment:
```bash
python3 scripts/nvfp4_compress/compress_checkpoint.py \
  --input Sehyo/Qwen3.5-35B-A3B-NVFP4 \
  --output /path/to/modified_checkpoint \
  --codebook 3bit_uniform

lm_eval --model vllm \
  --model_args pretrained=/path/to/modified_checkpoint,trust_remote_code=True \
  --tasks mmlu,gsm8k \
  --num_fewshot 0
```

### Phase 4: Quantized compression & optimization

Apply secondary quantization techniques constrained to valid FP4 output:
- Lattice VQ on FP4 codes
- Product quantization with FP4 constraint
- GPTQ-style compensation in code space
- Adaptive block scaling (Four-Over-Six style) at sub-codebook level

## The exploration loop

LOOP FOREVER:

1. **Decide what to try next.** Look at `exploration.md` and current results. What's the most impactful next experiment?

2. **Do the work.** Write code, modify checkpoints, run experiments.

3. **Test it.** Run lm-eval:
   ```bash
   lm_eval --model vllm \
     --model_args pretrained=/path/to/checkpoint,trust_remote_code=True \
     --tasks mmlu,gsm8k --num_fewshot 0
   ```

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
- **Simpler is always better.** Our 3-bit uniform ±{0,2,4,6} is embarrassingly simple and nearly lossless. Beat it with something equally clean.
- **Fail fast.** If an approach isn't working after one eval run, log and move on.
- **The pipeline must be correct.** Always operate on actual FP4 codes with preserved scales. Never re-quantize.
- **Accuracy > proxy metrics.** Always evaluate with MMLU/GSM8K, not just MSE or PPL.
- **Coherent error > random error.** Deterministic, consistent rounding decisions are better than "optimal" per-element choices that break error coherence.

### Autonomy rules

**NEVER STOP** to ask the human if you should continue. The human may be away and expects you to keep exploring until manually stopped.

## Reference material

### FP4 packed format (verified)
- Two 4-bit codes per byte: `[HIGH_NIBBLE (bits 7-4) | LOW_NIBBLE (bits 3-0)]`
- Even index → LOW nibble, Odd index → HIGH nibble
- E2M1 lookup: `{0:0, 1:0.5, 2:1, 3:1.5, 4:2, 5:3, 6:4, 7:6}` + sign bit (bit 3)
- No interleaving, row-major layout
- Pack/unpack round-trip verified in `sub_fp4_compress.py`

### Key experimental findings (from PPL evaluation)
- Per-block entropy at block-16: 3.095 bits/elem (19.8% below global marginal of 3.857)
- Per-block entropy at block-32: 3.463 bits/elem (worse — crosses block scale boundaries)
- Pairwise MI between adjacent elements: 0.0075 bits (negligible)
- 3-bit uniform ±{0,2,4,6}: +0.017 PPL at 3.31 bits/elem
- Adaptive 3-bit (4/6): -0.009 PPL (beats NVFP4!) at 3.37 bits/elem
- 2-bit hard floor at +0.6 PPL
- Stochastic rounding: 3x lower MSE → 25x worse PPL
- MoE-only NVFP4 baseline (code-space pipeline, identity mapping): PPL = 6.6974

### Papers (in scripts/channel_quant_new/papers/)
- **Four Over Six** (2512.02010): Adaptive block scaling for NVFP4
- **BOF4** (2505.06653): EM-optimized codebook + outlier-preserving quantization
- **GLVQ** (2510.20984): Per-group learned lattice codebooks
- **AQLM** (2401.06118): Additive multi-codebook VQ
- **QuIP#** (2402.04396): E8 lattice + Hadamard incoherence
- **Float8@2bits** (2601.22787): Entropy coding of Float8 weights to 2 bits effective
