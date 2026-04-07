# NVFP4 2-Bit Codebook Compression Pipeline

> **Pipeline**: BF16 → Hadamard+GPTQ → NVFP4 → 2-bit codebook (MoE only) → decompress → dequant BF16 → eval
> **Model**: Qwen/Qwen3-30B-A3B (48L, 128 experts, top-8, 61.1GB BF16)
> **Eval**: MMLU (5-shot) + GSM8K (4-shot CoT) — 4-way comparison
> **Hardware**: 8×A100-80GB (SM80 — no FP4 TC, eval via BF16 dequant)

---

## 🔥 Lessons Learned (MUST READ before any future work)

**These cost us ~11 hours of GPU time. Do NOT repeat.**

### 1. TRT-LLM dequant checkpoint MUST match original HF shard structure
TRT-LLM's concurrent weight loader fails silently if shards have different key distribution than original.
**FIX**: Always rebuild dequanted checkpoints using original shards as template:
```python
for orig_shard in sorted(orig_dir.glob('model-*.safetensors')):
    orig = load_file(orig_shard); out = {k: deq.get(k, v) for k, v in orig.items()}; save_file(out, ...)
```
Copy original `config.json`, `tokenizer*`, `model.safetensors.index.json` from the original HF checkpoint. Never generate your own index.

### 2. TRT-LLM on A100 (SM80) needs TRTLLM_ENABLE_PDL=0
TRT-LLM 1.2.0 uses `griddepcontrol` (SM90+ only) by default. Set `TRTLLM_ENABLE_PDL=0` for A100.

### 3. fast_hadamard_transform must match torch's CUDA version exactly
When `pip install tensorrt-llm` downgrades torch (e.g. 2.11→2.9), fast_hadamard_transform's CUDA extension breaks.
**FIX**: After any torch version change, recompile: patch torch's CUDA version check first, then `pip install --force-reinstall --no-deps git+https://github.com/Dao-AILab/fast-hadamard-transform.git --no-build-isolation`.

### 4. Hadamard+GPTQ requires inverse rotation at dequant time (STILL BUGGED)
Hadamard rotation changes the weight basis. Dequanted weights are in rotated space → model outputs garbage (0.36% MMLU).
Applying inverse rotation after dequant still gave 0.94% → the `hadamard_rotate_matrix(H)` Hessian rotation or the integration point in GPTQ is wrong.
**STATUS**: Fell back to plain GPTQ (no Hadamard). Needs deeper debugging — likely the Hessian rotation or the GPTQ column-update interaction.

### 5. Never run multiple GPU jobs competing for memory
lm-eval with device_map=auto spreads across ALL GPUs. Two model loads = OOM.
**FIX**: One eval at a time, TP8 with TRT-LLM. Kill previous jobs before starting new ones.

### 6. CBINT2 `--output-format nvfp4` produces packed uint8 (half columns)
When dequanting, `unpack_uint8_to_fp4` doubles columns. If you dequant then rebuild in original shard structure using `deq.get(k, v)`, the half-column tensors overwrite original full-column tensors → shape mismatch.
**FIX**: Use `--output-format bf16` for CBINT2 codebook compression when evaluating on A100 (dequant path). This writes full-size BF16 with codebook-constrained values directly.

### 7. CBINT2 `--num-gpus N>1` has a race condition on shared shards
Multiple blocks share the same safetensor shard file (e.g. shard-00001 has blocks 0,1,2). With multi-GPU, different workers read/modify/write the same shard concurrently → data corruption.
**FIX**: Use `--num-gpus 1` until the code adds per-shard locking or processes shards atomically. The single-GPU run takes ~40 min for codebook compression on 48 blocks — acceptable.

### 8. All caches (pip, HF, torch) must point to /data
Root filesystem has only 64GB. Set `PIP_CACHE_DIR`, `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME` to `/data/junzhou/.cache/` via `source /data/junzhou/env.sh`.

---

## ⚠️ Disk Constraint

`/` has **64GB free**. `/data` has **18TB**. Everything goes to `/data/junzhou/`.

Create `/data/junzhou/env.sh` and **source it before every command**:
```bash
export PIP_CACHE_DIR=/data/junzhou/.cache/pip
export HF_HOME=/data/junzhou/.cache/huggingface
export TRANSFORMERS_CACHE=/data/junzhou/.cache/huggingface
export TORCH_HOME=/data/junzhou/.cache/torch
export XDG_CACHE_HOME=/data/junzhou/.cache
source /data/junzhou/venv/bin/activate
```

| Path | Content | Size |
|------|---------|------|
| `/data/junzhou/venv/` | Fresh Python 3.12 venv | ~5GB |
| `/data/junzhou/models/` | All checkpoints (BF16, NVFP4, dequanted) | ~300GB |
| `/data/junzhou/hessians/` | GPTQ Hessian data (48 blocks) | ~50GB |
| `/data/junzhou/results/` | lm-eval outputs + comparison.json | ~1GB |
| `/data/junzhou/repos/FP-Quant/` | MR-GPTQ reference code (read-only) | ~100MB |
| `/data/junzhou/.cache/` | pip + HF + torch caches | ~60GB |
| `/home/junzhou/CBINT2/` | Source code (small edits only) | ~1.5MB |

---

## Guardrails

- **Never** run NVFP4 inference on A100 — always dequant to BF16 first
- **Never** write large files to `/` or `/home/` — all to `/data/junzhou/`
- **Never** use system Python — always the venv
- **Never** install TensorRT-LLM — not needed
- **Never** modify CBINT2's codebook design (364 entries) without approval
- **Never** do partial eval — always full MMLU/GSM8K
- Output is fake-quantized NVFP4 (constrained to 4 values/block). Actual 2-bit packing is future work.

---

## Key References

| What | Where |
|------|-------|
| CBINT2 codebase | `/home/junzhou/CBINT2/` — 4,200 lines: CodebookQuantizer, GPTQ, dequant, eval |
| NVFP4 quant math | `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py` — `_quantize_nvfp4`, `_dequantize_nvfp4`, E2M1 table |
| MR-GPTQ paper | arXiv:2509.23202 (ICLR 2026) — block-local Hadamard + GPTQ for NVFP4 |
| FP-Quant repo | https://github.com/IST-DASLab/FP-Quant — `src/transforms/transforms.py` (HadamardTransform), `src/quantization/gptq.py` |
| CBINT2 GPTQ CLI | `fakequant_model_gptq.py` — flags: `--input-path`, `--output-path`, `--hessian-dir`, `--output-format nvfp4`, `--mlp-only`, `--hadamard` (new) |
| CBINT2 dequant | `dequant_nvfp4.py` — NVFP4→BF16, multi-GPU |
| CBINT2 codebook compress | `fakequant_model.py --mlp-only --output-format nvfp4` — per-block codebook selection |
| Calibration | `gptq/calibrate.py` — WikiText-2 train, 128×2048 chunks, MoE-aware hooks |
| nvidia NVFP4 checkpoint | `nvidia/Qwen3-30B-A3B-NVFP4` on HuggingFace |

---

## Execution Waves

### Wave 1 — Environment + Downloads (all parallel)

- [x] **1. Create venv + install deps** `[quick]`
  ```bash
  python3 -m venv /data/junzhou/venv
  # Create env.sh with cache redirects (see above)
  source /data/junzhou/env.sh
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -r /home/junzhou/CBINT2/requirements.txt
  pip install fast_hadamard_transform
  # Verify: python -c "import torch; assert torch.cuda.device_count()==8; from fast_hadamard_transform import hadamard_transform"
  ```

- [x] **2. Download BF16 model** `[quick]`
  ```bash
  huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /data/junzhou/models/Qwen3-30B-A3B
  # Verify: 16 shards, ~61GB, config.json has num_experts=128
  ```

- [x] **3. Download nvidia NVFP4 checkpoint** `[quick]`
  ```bash
  huggingface-cli download nvidia/Qwen3-30B-A3B-NVFP4 --local-dir /data/junzhou/models/Qwen3-30B-A3B-NVFP4
  ```

- [x] **4. Clone FP-Quant** `[quick]`
  ```bash
  git clone https://github.com/IST-DASLab/FP-Quant /data/junzhou/repos/FP-Quant
  # Key files: src/transforms/transforms.py, src/quantization/gptq.py, src/quantization/quant_ops.py
  ```

- [x] **5. Verify CBINT2 tests** `[quick]`
  ```bash
  cd /home/junzhou/CBINT2
  python test_fakequant.py        # Script-style (uses argparse, NOT pytest)
  python test_codebook_analysis.py
  python test_codebook_mse.py
  python gptq/test_gptq.py
  ```

### Wave 2 — Baselines (parallel after Wave 1)

- [ ] **6. BF16 baseline eval** `[unspecified-high]` → depends: 1, 2
  ```bash
  lm_eval --model hf --model_args pretrained=/data/junzhou/models/Qwen3-30B-A3B,dtype=bfloat16,trust_remote_code=True \
    --tasks mmlu --num_fewshot 5 --batch_size auto --output_path /data/junzhou/results/bf16_mmlu
  lm_eval ... --tasks gsm8k_cot --num_fewshot 4 ... --output_path /data/junzhou/results/bf16_gsm8k
  # Expected: MMLU ~81%, GSM8K ~92%
  ```

- [x] **7. Dequantize nvidia NVFP4 → BF16** `[quick]` → depends: 1, 3
  ```bash
  python /home/junzhou/CBINT2/dequant_nvfp4.py \
    --input-path /data/junzhou/models/Qwen3-30B-A3B-NVFP4 \
    --output-path /data/junzhou/models/Qwen3-30B-A3B-NVFP4-dequant
  # Verify output tensors are bfloat16
  ```

- [ ] **8. Vanilla NVFP4 eval** `[unspecified-high]` → depends: 7
  ```bash
  lm_eval --model hf --model_args pretrained=/data/junzhou/models/Qwen3-30B-A3B-NVFP4-dequant,dtype=bfloat16,trust_remote_code=True \
    --tasks mmlu --num_fewshot 5 --batch_size auto --output_path /data/junzhou/results/nvfp4_vanilla_mmlu
  lm_eval ... --tasks gsm8k_cot --num_fewshot 4 ... --output_path /data/junzhou/results/nvfp4_vanilla_gsm8k
  ```

### Wave 3 — Hadamard + GPTQ Pipeline

- [x] **9. Hessian calibration** `[deep]` → depends: 1, 2
  ```bash
  python /home/junzhou/CBINT2/gptq/calibrate.py \
    --model-path /data/junzhou/models/Qwen3-30B-A3B \
    --output-dir /data/junzhou/hessians \
    --num-samples 128 --seq-len 2048
  # Output: block_00.safetensors .. block_47.safetensors
  # WikiText-2 train split, seed=0
  ```

- [x] **10. Add Hadamard rotation to CBINT2** `[deep]` → depends: 4, 5

  **What to do**: Study FP-Quant's `src/transforms/transforms.py` (HadamardTransform class). Create `/home/junzhou/CBINT2/hadamard.py`:
  ```python
  from fast_hadamard_transform import hadamard_transform
  def hadamard_rotate(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
      """Block-local Hadamard rotation on weight columns. Scale = 1/sqrt(group_size)."""
      # Reshape [N, K] → [N, K//gs, gs], apply hadamard_transform, reshape back
  ```
  Modify `fakequant_model_gptq.py`:
  - Add `--hadamard` CLI flag
  - Before GPTQ per block: rotate weights with `hadamard_rotate(weight, group_size=128)`
  - Hessian also needs rotation: H_rot = R^T @ H @ R (block-diagonal)

  **Key insight**: Hadamard spreads outliers uniformly across 16-element FP4 blocks → better rounding.

  **Verify**: `R^T(R(x)) == x` (orthogonality), rotated GPTQ produces different output than non-rotated.

- [ ] **11. Run Hadamard+GPTQ → NVFP4** `[unspecified-high]` → depends: 9, 10
  ```bash
  python /home/junzhou/CBINT2/fakequant_model_gptq.py \
    --input-path /data/junzhou/models/Qwen3-30B-A3B \
    --output-path /data/junzhou/models/Qwen3-30B-A3B-HadGPTQ-NVFP4 \
    --hessian-dir /data/junzhou/hessians \
    --output-format nvfp4 \
    --hadamard
  # NOTE: --output-format nvfp4 required! A100 auto-detect defaults to bf16.
  # No --mlp-only → all nn.Linear layers quantized.
  # ~24h for 48 blocks.
  ```

- [ ] **12. Dequant our NVFP4 → BF16** `[quick]` → depends: 11
  ```bash
  python /home/junzhou/CBINT2/dequant_nvfp4.py \
    --input-path /data/junzhou/models/Qwen3-30B-A3B-HadGPTQ-NVFP4 \
    --output-path /data/junzhou/models/Qwen3-30B-A3B-HadGPTQ-dequant
  ```

- [ ] **13. Our NVFP4 eval** `[unspecified-high]` → depends: 12
  ```bash
  lm_eval --model hf --model_args pretrained=/data/junzhou/models/Qwen3-30B-A3B-HadGPTQ-dequant,dtype=bfloat16,trust_remote_code=True \
    --tasks mmlu --num_fewshot 5 --batch_size auto --output_path /data/junzhou/results/nvfp4_hadgptq_mmlu
  lm_eval ... --tasks gsm8k_cot --num_fewshot 4 ... --output_path /data/junzhou/results/nvfp4_hadgptq_gsm8k
  # Expected: better than vanilla NVFP4 (GPTQ compensation)
  ```

### Wave 4 — Codebook Compression + Final

- [ ] **14. Codebook compress MoE experts** `[unspecified-high]` → depends: 11
  ```bash
  python /home/junzhou/CBINT2/fakequant_model.py \
    --input-path /data/junzhou/models/Qwen3-30B-A3B-HadGPTQ-NVFP4 \
    --output-path /data/junzhou/models/Qwen3-30B-A3B-CBINT2 \
    --output-format nvfp4 \
    --mlp-only
  # --mlp-only: For Qwen3-30B-A3B (all layers are MoE, no dense MLP), this effectively = experts-only.
  # Per-block: exhaustive search over 364 codebooks (zero + C(14,3) non-zero), pick MSE-minimizing.
  # Output: valid NVFP4 checkpoint with expert weights constrained to 4 values/block (~3 bits/elem effective).
  # Verify: each 16-element block has ≤4 unique FP4 codes.
  ```

- [ ] **15. Dequant compressed → BF16** `[quick]` → depends: 14
  ```bash
  python /home/junzhou/CBINT2/dequant_nvfp4.py \
    --input-path /data/junzhou/models/Qwen3-30B-A3B-CBINT2 \
    --output-path /data/junzhou/models/Qwen3-30B-A3B-CBINT2-dequant
  ```

- [ ] **16. Compressed eval** `[unspecified-high]` → depends: 15
  ```bash
  lm_eval --model hf --model_args pretrained=/data/junzhou/models/Qwen3-30B-A3B-CBINT2-dequant,dtype=bfloat16,trust_remote_code=True \
    --tasks mmlu --num_fewshot 5 --batch_size auto --output_path /data/junzhou/results/cbint2_compressed_mmlu
  lm_eval ... --tasks gsm8k_cot --num_fewshot 4 ... --output_path /data/junzhou/results/cbint2_compressed_gsm8k
  ```

- [ ] **17. Compile 4-way comparison** `[quick]` → depends: 6, 8, 13, 16

  Collect all 8 result dirs (`{bf16,nvfp4_vanilla,nvfp4_hadgptq,cbint2_compressed}_{mmlu,gsm8k}`), parse scores, produce:

  | Variant | MMLU (5-shot) | GSM8K (4-shot CoT) | Effective bits/elem |
  |---------|---------------|---------------------|---------------------|
  | BF16 (original) | XX.X% | XX.X% | 16 |
  | Vanilla RTN NVFP4 | XX.X% | XX.X% | 4.5 |
  | Hadamard+GPTQ NVFP4 | XX.X% | XX.X% | 4.5 |
  | CBINT2 Compressed→NVFP4 | XX.X% | XX.X% | ~3.0 |

  Save to `/data/junzhou/results/comparison.json`.

---

## Final Verification (after all tasks)

4 review agents in parallel. All must approve. Present results to user for explicit okay.

- [ ] **F1. Plan Compliance** `oracle` — verify each Must Have is implemented, each guardrail is respected, evidence files exist
- [ ] **F2. Code Quality** `unspecified-high` — run all CBINT2 tests, check for bare excepts, verify hadamard.py has docstrings
- [ ] **F3. Results QA** `unspecified-high` — verify all 8 result dirs exist, parse MMLU/GSM8K scores, assert all > 70%/80%, verify compressed checkpoint has valid E2M1 codes
- [ ] **F4. Scope Fidelity** `deep` — no TRT-LLM installed, no large files on `/`, no extra benchmarks, only expert tensors modified by codebook compression

---

## Success Criteria

```bash
# All results exist
for v in bf16 nvfp4_vanilla nvfp4_hadgptq cbint2_compressed; do
  for b in mmlu gsm8k; do
    ls /data/junzhou/results/${v}_${b}/*/results.json
  done
done

# Comparison file valid
python -c "import json; c=json.load(open('/data/junzhou/results/comparison.json')); assert len(c)==4"
```

- [ ] All MMLU scores > 70%, all GSM8K scores > 80%
- [ ] Compressed checkpoint: all blocks ≤4 unique FP4 codes, all nibbles in [0,15]
- [ ] All heavy files on `/data/junzhou/`, nothing large on `/`
