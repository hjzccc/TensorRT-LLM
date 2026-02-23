# Proposal: Bucket-Aware GroupGEMM Tactic Selection for MoE

**Author**: huanchen  
**Date**: 2026-02-22  
**Status**: Draft  
**Scope**: `fused_moe_heter.py`, `torch_custom_ops.py`, AutoTuner, C++ MoE kernels

---

## 1. Problem

### Current Behavior

Each `torch.ops.trtllm.fused_moe()` call runs a single CUTLASS GroupGEMM
over **all experts in a precision group** using **one tactic** (tile config,
cluster shape, mainloop schedule). The AutoTuner picks this tactic during
warmup by profiling with a synthetic workload, and reuses it for every
inference call regardless of per-expert token distribution.

```
run_moe()  →  for each precision group:
                torch.ops.trtllm.fused_moe(ALL experts in group, ONE tactic)
```

### Why This Hurts

In a GroupGEMM with E experts, each expert i has M_i tokens. The tactic's
tile shape (e.g., 128×128×128B on SM90) determines how work is decomposed
across SMs. When a single tactic is used for all experts:

- **Large tile on small M**: Expert with M=4 tokens using a 128×128 tile
  wastes 96% of the tile's M-dimension capacity. The CTA launches with
  mostly padding — SMs do useless work.
- **Small tile on large M**: Expert with M=512 tokens using a 64×64 tile
  needs 8× more CTAs than a 128×128 tile, increasing launch overhead and
  reducing per-CTA arithmetic intensity.

This is **especially bad for single-GPU inference** with large MoE group
sizes (e.g., DeepSeek-V3 with 256 experts). With top-K routing, the token
distribution across experts follows a long tail — a few experts get many
tokens, most get very few. One tactic cannot serve both regimes well.

### Quantifying the Waste

Example: DeepSeek-V3, 256 experts, top-8 routing, batch=128 tokens:
- ~8 experts get 20+ tokens each (top bucket)
- ~40 experts get 4–19 tokens each (mid bucket)
- ~80 experts get 1–3 tokens (small bucket)
- ~128 experts get 0 tokens (skipped by kernel)

A 128×128 tile config wastes most of its capacity on the ~120 experts
with <20 tokens. A 64×16 tile would be far more efficient there, but
would under-utilize SMs on the 8 high-traffic experts.

---

## 2. Proposed Solution: Bucket-Aware Sub-GroupGEMMs

### Core Idea

After routing, **bucket experts by their token count** into 2–4 ranges,
then issue **separate GroupGEMM calls per bucket**, each with its own
autotuned tactic. This adds minimal kernel launch overhead (~2–5 µs per
extra launch) but allows each sub-group to use the tile config best suited
to its M dimension.

```
run_moe()  →  for each precision group:
                bucket experts by token count
                for each bucket:
                    torch.ops.trtllm.fused_moe(bucket_experts, BUCKET_TACTIC)
                sum outputs
```

### Bucket Strategy

Proposed bucket boundaries based on available CUTLASS tile shapes:

| Bucket | M Range       | Ideal Tile (SM90)     | Ideal Tile (SM100)    |
|--------|---------------|-----------------------|-----------------------|
| tiny   | 1–15 tokens   | 64×16×128B            | 64×32×128B            |
| small  | 16–63 tokens  | 64×64×128B            | 64×128×128B           |
| medium | 64–127 tokens | 128×128×128B          | 128×128×128B          |
| large  | 128+ tokens   | 256×128×128B          | 128×256×256B          |

These thresholds are **not hardcoded** — the AutoTuner profiles each
bucket range independently, so the optimal tactic is discovered empirically.
The boundaries themselves can be tuning parameters.

### Why Buckets (Not Per-Expert Tactics)

- Per-expert tactics would require E separate GEMM launches — too much
  launch overhead for 256 experts.
- Bucketing into 2–4 groups amortizes launch cost while capturing 80%+
  of the tile-size benefit.
- Experts within a bucket have similar M → the shared tactic is near-optimal
  for all of them.

---

## 3. Design

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│ fused_moe_heter.py :: run_moe()                            │
│                                                             │
│  1. Dispatch policy: expert_to_group (precision groups)     │
│  2. For each precision group:                               │
│     a. Count tokens per expert from routing result          │
│     b. Bucket experts by token count                        │
│     c. For each bucket:                                     │
│        - Build sub-group expert mask                        │
│        - Call torch.ops.trtllm.fused_moe() with bucket      │
│          experts + bucket-specific tactic                   │
│     d. Accumulate sub-group outputs                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Token Counting and Bucketing (Python Layer)

**Location**: `fused_moe_heter.py` or new helper in `fused_moe/utils.py`

```python
def bucket_experts_by_token_count(
    token_selected_experts: torch.Tensor,  # [N, K]
    num_experts: int,
    bucket_boundaries: List[int] = [16, 64, 128],
) -> List[torch.Tensor]:
    """Returns list of expert_masks, one per bucket.
    
    Each mask is a bool tensor [num_experts] indicating which experts
    fall into this bucket's token-count range.
    """
    # Count tokens per expert
    expert_counts = torch.zeros(num_experts, device=token_selected_experts.device)
    # flatten [N,K] -> [N*K], count occurrences
    flat = token_selected_experts.view(-1)
    expert_counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=expert_counts.dtype))
    
    # Bucket by count
    boundaries = [0] + bucket_boundaries + [float('inf')]
    buckets = []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (expert_counts >= lo) & (expert_counts < hi)
        if mask.any():
            buckets.append(mask)
    return buckets
```

**CUDA graph concern**: `expert_counts` changes every step. The bucketing
itself is cheap (scatter_add + comparisons), but the **number of buckets**
must be fixed for CUDA graph compatibility. Solution: always issue all
bucket calls, but empty buckets get zero-expert masks → kernel exits early.

### 3.3 Per-Bucket Tactic Selection (AutoTuner Integration)

**Location**: `torch_custom_ops.py :: fused_moe()`

Currently the custom op calls `tuner.choose_one()` twice (GEMM1, GEMM2)
with a single `MoERunner`. For bucket-aware tactics, we need:

```python
# Option A: Separate custom_op names per bucket
for bucket_idx, bucket_mask in enumerate(buckets):
    _, gemm_tactic_1 = tuner.choose_one(
        f"trtllm::fused_moe::gemm1::bucket_{bucket_idx}",
        [moe_runner],
        bucket_tuning_config,  # tune_max_num_tokens = bucket upper bound
        [...],
        gemm_idx=1,
    )
```

This gives each bucket its **own cache key** in the AutoTuner, so
profiling discovers the best tactic for that bucket's typical M dimension.

**Cache key change**: Currently the cache key is:
```
(custom_op, runner_class, unique_id, optimization_profile)
```

With buckets, the `custom_op` name encodes the bucket:
```
("trtllm::fused_moe::gemm1::bucket_0", runner_class, unique_id, profile)
("trtllm::fused_moe::gemm1::bucket_1", runner_class, unique_id, profile)
```

No changes needed to AutoTuner internals — just different op names.

### 3.4 Sub-Group Dispatch (Modified run_moe)

**Location**: `fused_moe_heter.py :: run_moe()`

```python
for group_idx, (grp_experts, grp_scales) in enumerate(dispatches):
    # ... resolve weights ...
    
    # NEW: bucket experts by token count within this precision group
    buckets = bucket_experts_by_token_count(
        grp_experts, self.num_experts, self._bucket_boundaries)
    
    for bucket_idx, bucket_mask in enumerate(buckets):
        # Sentinel-mask non-bucket experts (same mechanism as group dispatch)
        bucket_experts = torch.where(
            bucket_mask[grp_experts],  # broadcast mask to [N,K]
            grp_experts,
            torch.full_like(grp_experts, self.num_experts),  # sentinel
        )
        bucket_scales = torch.where(
            bucket_mask[grp_experts],
            grp_scales,
            torch.zeros_like(grp_scales),
        )
        
        group_result += torch.ops.trtllm.fused_moe(
            qi.x, bucket_experts, bucket_scales,
            src_w3_w1, ...,
            # bucket-specific tactic selected inside the custom op
        )[0]
    
    accumulated += group_result
```

### 3.5 C++ Changes (Minimal or None)

The C++ `FusedMoeRunner::run_moe()` already handles sentinel-masked expert
slots (zero scale → skip). By sending it a subset of experts via masking,
the existing kernel naturally processes only the bucket's experts.
The tactic is already passed per-call via `[gemm_tactic_1, gemm_tactic_2]`.

**No C++ changes required for the basic version.**

Optional future optimization: teach the C++ kernel to accept multiple
tactics in a single launch (one per expert sub-group), avoiding the
Python-level loop. This is a Phase 2 optimization.

---

## 4. Implementation Plan

### Phase 1: Python-Level Bucketing (No C++ Changes)

| Step | Description | Files | Effort |
|------|-------------|-------|--------|
| 1.1  | Add `bucket_experts_by_token_count()` utility | `fused_moe/utils.py` (new) | Small |
| 1.2  | Modify `fused_moe_heter.py::run_moe()` to loop over buckets within each precision group | `fused_moe_heter.py` | Medium |
| 1.3  | Parameterize bucket boundaries via `HeterCutlassFusedMoE.__init__()` | `fused_moe_heter.py` | Small |
| 1.4  | Encode bucket index in `custom_op` name for AutoTuner cache separation | `torch_custom_ops.py` | Small |
| 1.5  | Ensure CUDA graph compatibility (fixed number of bucket calls) | `fused_moe_heter.py` | Medium |
| 1.6  | Unit tests with mock expert distributions | `tests/` | Medium |
| 1.7  | Benchmark: single-GPU DeepSeek-V3 inference, measure latency vs baseline | benchmarks | Medium |

### Phase 2: Autotuner-Aware Bucket Boundaries

| Step | Description | Files | Effort |
|------|-------------|-------|--------|
| 2.1  | Profile tactic performance across M-dimension sweep (1–512) | `examples/layer_wise_benchmarks/` | Medium |
| 2.2  | Use profiling data to auto-derive optimal bucket boundaries | `autotuner.py` or new utility | Medium |
| 2.3  | Integrate auto-derived boundaries into warmup phase | `model_engine.py` | Medium |

### Phase 3: C++ Multi-Tactic GroupGEMM (Optional)

| Step | Description | Files | Effort |
|------|-------------|-------|--------|
| 3.1  | Extend `FusedMoeRunner::run_moe()` to accept per-expert-group tactics | `thop/moeOp.cpp`, `moe_kernels.cu` | Large |
| 3.2  | Launch multiple tile configs within a single kernel | `moe_kernels.cu` | Very Large |

---

## 5. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Extra kernel launch overhead (2–4 launches per group) | +8–20 µs per MoE layer | Profile; fall back to 2 buckets if overhead > tactic gain |
| CUDA graph incompatibility from variable bucket counts | Graph capture fails | Always launch fixed N_BUCKETS calls; empty buckets exit early |
| Autotuner warmup time increases (N_BUCKETS × 2 × original) | Longer startup | Cache persistence via `TLLM_AUTOTUNER_CACHE_PATH`; marginal increase |
| Token count computation overhead per step | Added per-step latency | scatter_add is O(N×K), negligible vs GEMM time |
| Interaction with alltoall / EP | May change token distribution | Bucket after alltoall scatter, before GroupGEMM |

---

## 6. Expected Impact

### Latency Improvement Estimate

For DeepSeek-V3 single-GPU (256 experts, top-8, batch=128):
- Baseline: 1 GroupGEMM with tactic tuned for ~M=4 (median expert)
- Bucketed: 3 GroupGEMMs with tactics tuned for M∈{2, 16, 64}
- Expected: **15–30% latency reduction** per MoE layer (net of launch overhead)

The gain is larger when:
- Group size is large (more experts → wider M distribution)
- Batch size is small (more skewed token distribution)
- Single GPU (no EP to spread load)

### Who Benefits Most

| Scenario | Benefit |
|----------|---------|
| Single GPU, large MoE (DeepSeek-V3) | ★★★★★ High |
| 2–4 GPU EP, large MoE | ★★★☆☆ Medium |
| 8+ GPU EP (few experts per rank) | ★☆☆☆☆ Low (few experts, less skew) |
| Dense models (no MoE) | N/A |

---

## 7. Key File References

| File | Role |
|------|------|
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_heter.py` | run_moe() — main modification point |
| `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` | fused_moe custom op — tactic selection |
| `tensorrt_llm/_torch/autotuner.py` | AutoTuner — cache keyed by custom_op name |
| `tensorrt_llm/_torch/modules/fused_moe/policy/heter_dispatch.py` | Dispatch policy — expert-to-group assignment |
| `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/gemm_configs.h` | Tile shapes: SM80 16×128 to 256×128; SM90 64×16 to 256×256; SM100 64×32 to 128×256 |
| `cpp/tensorrt_llm/thop/moeOp.cpp` | C++ FusedMoeRunner — receives tactic per call |
| `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` | GroupGEMM kernel — already handles sentinel-masked experts |

---

## 8. Open Questions

1. **Number of buckets**: 2, 3, or 4? Need profiling data to find the sweet spot
   between tactic granularity and launch overhead.
2. **Bucket boundaries**: Static (config-time) vs dynamic (per-step based on
   actual token counts)? Dynamic is more adaptive but CUDA-graph-hostile.
3. **Interaction with CuTE DSL MoE backend**: `fused_moe_cute_dsl.py` has its
   own tactic selection — should bucketing apply there too?
4. **Should this be heter-only or also benefit the homogeneous CUTLASS path?**
   The same M-dimension skew problem exists in `fused_moe_cutlass.py`.
5. **Phase 3 feasibility**: Is a multi-tactic single-kernel launch worth the
   C++ complexity, or is Python-level bucketing sufficient?