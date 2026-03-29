# Session ses_2eaa Summary — Verified from Transcript

**Session:** ses_2eaae979dffenAASCMYKIBLH0q
**Period:** 3/22/2026 – 3/24/2026
**Model:** Qwen3.5-35B-A3B
**Dataset:** WikiText-2 (128-sample train calibration, 145-chunk test evaluation)

---

## The Method

A framework for per-channel mixed-precision (BF16+NVFP4) quantization of MoE expert weights:

1. **Calibrate** on WikiText-2 train (128 × 2048 tokens) — collect per-channel `E[a²] × ||w||²` and routing weights for every expert in every layer
2. **Rank** all 31M channels globally by `routing_weight × E[a²] × ||w||²`
3. **Allocate** top-B% to BF16, rest to NVFP4 (global greedy knapsack)
4. **Evaluate** on WikiText-2 test (145 × 2048 tokens) with exact TRT-LLM NVFP4 kernels

---

## Results — Verified from GPU Output

All numbers below were directly observed in tool outputs within this transcript. Numbers run multiple times were identical across runs unless noted.

### No-boost baseline (DEPTH_BOOST=0)

| Budget | PPL | Gap Recovery | Runs in transcript | Notes |
|--------|-----|-------------|-------------------|-------|
| 0% (NVFP4) | 6.8431 | 0% | — | Baseline, cited from prior work |
| 5% | 6.6548 | 74.3% | 1 (Run 6) | |
| 10% | 6.6398 | 80.2% | 1 (Run 6) | |
| 15% | 6.6363 / 6.6391 | 81.6% / 80.5% | 1 each | ⚠️ Two different values — see note below |
| 20% | 6.6338 | 82.6% | 1 (Run 6) | |
| 25% | 6.6227 | 86.9% | 3 (Runs 2, 3, 6) | Identical across all 3 |
| 30% | 6.6261 | 85.6% | 4 (Runs 2, 3, 3, 6) | Dip confirmed every time |
| 35% | 6.6169 | 89.2% | 2 (Runs 1, 6) | |
| 40% | 6.6152 | 89.9% | 2 (Runs 1, 6) | |
| 45% | 6.6100 | 91.9% | 1 (Run 3) | |
| 50% | 6.6119 | 91.2% | 1 (Run 1) | Dip vs 45% |
| 55% | 6.6061 | 93.5% | 1 (Run 3) | |
| 60% | 6.5998 | 96.0% | 1 (Run 2) | |
| 65% | 6.5946 | 98.0% | 4 (Run 3 + Run 4 ×2) | Identical all 4 runs |
| 70% | 6.5978 | 96.7% | 3 (Run 2 + Run 4 ×2) | Dip vs 65%, identical all 3 |
| 75% | 6.5938 | 98.3% | 1 (Run 3) | |
| 80% | 6.5953 | 97.8% | 1 (Run 2) | Dip vs 75% |
| 90% | 6.5918 | 99.1% | 1 (Run 3) | |
| 100% (BF16) | 6.5896 | 100% | — | Baseline, cited from prior work |

#### ⚠️ 15% discrepancy

The 15% budget produced 81.6% (PPL=6.6363) in the pre-transcript runs but 80.5% (PPL=6.6391) in Run 6 (the final re-run). The agent attributes this to different code state — the DEPTH_BOOST code path was added between runs (set to 0 so it should be skipped, but the code structure changed). This is the one number that needs re-verification with a fully clean codebase.

### Depth-boost experiment (DEPTH_BOOST=2)

| Budget | No boost | Depth boost | Change |
|--------|----------|-------------|--------|
| 25% | 86.9% | 86.8% | -0.1pp |
| 30% | 85.6% (dip) | 87.9% | +2.3pp — dip fixed |
| 35% | 89.2% | 90.5% | +1.3pp |
| 50% | 91.2% | 89.6% | -1.6pp |
| 65% | 98.0% | 92.8% | -5.2pp — severe degradation |

**Conclusion:** Depth boost fixes local dips at low budgets but destroys high-budget performance. Reverted to DEPTH_BOOST=0.

---

## Findings Verified from This Transcript

These are things directly demonstrated by tool outputs in the transcript:

1. **Non-monotonic dips are deterministic** — 65% vs 70% tested 4 times (twice each, back-to-back, twice). PPL was identical to 4+ decimal places every run. The dips are not noise.

2. **Dip channels are early-layer W2** — the channels ranked 25-30% (causing the 30% dip) are W1:W2 ratio 4:1 concentrated in layers 1-5, vs the good top-25% channels which are W1:W2 ratio 27:1 concentrated in layers 35-39.

3. **Late layers hold disproportionate score mass** — layers 30-39 contain 24.9% of channels but 50.1% of total sensitivity score.

4. **Numpy memory optimization works** — ranking memory dropped from ~4.5GB (Python tuples) to 628MB (numpy arrays), confirmed by "126MB scores + 502MB meta" in first run output.

5. **Model downloads blocked** — both docker container and host failed to query HuggingFace model info. No second model was tested.

## Findings Cited from Prior Work (NOT verified in this transcript)

The agent references these but they were established in earlier sessions:

1. act_weighted is Hessian-diagonal optimal for NVFP4 (correlation=1.0)
2. NVFP4 blocks are per-channel (16-element blocks along input dimension)
3. NVFP4 weight global scale must use the FULL matrix (the fix itself predates this transcript)
4. Within-expert error is uniform (Gini ~0.06), across-expert concentrated by routing (Gini ~0.56)
5. Error grows with depth — W1 6×, W2 17×
6. Naive GPTQ doesn't work with NVFP4
7. Weight-magnitude selection gives 16× more PPL improvement than random

---

## Bugs Fixed (referenced but all predate this transcript)

1. **NaN from sparse experts** — zero-weight NVFP4 channels cause division by zero in `fp4_global_scale`. Fixed by promoting to BF16.
2. **Zero-padding corrupts block scales** — `F.pad` adds zero rows that dilute per-block scales. Fixed by demoting unaligned rows to BF16.
3. **NVFP4 global scale from sub-matrix** — removing channels changes amax → changes quantization grid. Fixed by computing scale from full weight matrix.
4. **OOM from Python tuple list** — 31M tuples at ~150 bytes each = 4.5GB. Fixed with numpy arrays (628MB).

The only fix directly visible in this transcript is #4 (numpy), confirmed by the "126MB scores + 502MB meta" output. The duplicate function definition bug (two `assign_tiers_from_ranking` definitions) was also fixed at the start of this transcript.

---

## What Exists on Disk

**Code** (in `scripts/channel_quant_new/profiling/`):
- `calibrate.py` — calibration on WikiText-2 train
- `optimize_allocation.py` — global greedy knapsack allocation + evaluation
- `collect_error_profile.py` — per-expert NVFP4 error profiling
- `collect_activation_stats.py` — activation distribution analysis
- `make_figures.py` — profiling figures
- `make_pareto_figure.py` — Pareto curve figure (created in this session, hardcoded PPL values)
- `hadamard_utils.py` — block-wise Hadamard transform
- `gptq_nvfp4.py` — GPTQ attempt (documented failure)

**Data** (in `profiling/`):
- `calibration.json` + `calibration_layers/` — per-channel sensitivity scores
- `error_profile.json` — per-expert error analysis
- `activation_distributions.json` — 145-chunk activation stats (verified: 40 layers, 296960 tokens)
- `optimal_allocation.json` — latest budget sweep results (⚠️ last written by depth-boost run, then overwritten by incomplete final run)

**Figures** (in `figures/`):
- `pareto_curve.png/pdf` — uses hardcoded values including pre-transcript 15%=81.6%
- `within_expert_view.png/pdf`
- `across_expert_view.png/pdf`

**Documentation**:
- `explore.md` — updated with full 18-point table (563 lines)
- `program.md` — updated with "ACHIEVED: 99.1%" results

---

## What Remains

- Zero-shot benchmarks (ARC, HellaSwag, PIQA, WinoGrande, MMLU) — never started
- Test on another MoE model (DeepSeek-V2-Lite, OLMoE, Qwen3-30B-A3B) — blocked by network
- Re-verify 15% with clean codebase (81.6% vs 80.5% discrepancy)
- Regenerate Pareto figure with consistent numbers from a single clean run
- Paper writing

---

## Time Spent (estimated from transcript)

Across 8 cycles with auto re-fires:

| Activity | Cycles | Approx wall time |
|----------|--------|-------------------|
| Running budget sweeps + polling | Cycles 0-2 | ~5 hours |
| Verifying 65/70% dip (4 runs) | Cycle 3 | ~1.5 hours |
| Updating markdown files + figure | Cycles 3-4 | ~30 min |
| Dip channel analysis | Cycle 5 | ~15 min |
| Failed model downloads + file checks | Cycle 6 | ~15 min |
| Depth-boost experiment | Cycle 7 | ~2 hours |
| Final re-run (incomplete, killed) | Cycle 8 | ~2 hours |