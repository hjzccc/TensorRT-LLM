# File Change Log — Heterogeneous Precision MoE Backend

## New Files

| File | Description |
|------|-------------|
| `tensorrt_llm/_torch/modules/fused_moe/policy/__init__.py` | Policy module exports (`BaseDispatchPolicy`, `DispatchPlan`, `RandomDispatchPolicy`, `ConfidenceThresholdPolicy`, `ExpertLoadPolicy`) |
| `tensorrt_llm/_torch/modules/fused_moe/policy/dispatch_plan.py` | `DispatchPlan` dataclass — maps experts to precision groups, with `validate()` |
| `tensorrt_llm/_torch/modules/fused_moe/policy/strategies.py` | `BaseDispatchPolicy` ABC, `RandomDispatchPolicy`, `ConfidenceThresholdPolicy` (mean routing weight), `ExpertLoadPolicy` (activation frequency), `_assign_by_score()` shared helper |
| `tests/unittest/_torch/modules/moe/test_heter_moe.py` | Unit tests: forward equivalence (2), config validation (8), dispatch policy (10 — random, confidence, expert-load, signal fallback, custom), **Phase 2 runtime benchmarks (2)** |
| `.agent/TODOs.md` | Future work tracking — Phase 2 (dual weights, per-group quant), Phase 3 (CUDA ops), policy improvements, testing, docs |

## Modified Files

| File | What Changed |
|------|-------------|
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_heter.py` | Core `HeterCutlassFusedMoE` class. **Phase 1**: Removed BF16-centric `update_hot_experts()`, `_find_bf16_group_index()`, `_make_default_assignment()`. Added policy-based dispatch: `set_dispatch_policy()`, `_recompute_dispatch()` called per `run_moe()`. **Signal passing**: Added `forward_chunk()` override to stash `router_logits` in `_pending_router_logits` before calling `super().forward_chunk()`, consumed by `run_moe()` override within the same call stack. `_recompute_dispatch()` now accepts `token_selected_experts`, `token_final_scales`, and `router_logits` and forwards to `policy.assign()`. Documented torch.compile/CUDA graph safety. **This session**: Added temporary debug `print("### [group]", group_idx, tok_idx.shape, tok_idx)` in the `run_moe()` loop (~line 903). |
| `tensorrt_llm/_torch/modules/fused_moe/policy/heter_dispatch.py` | **This session**: Major refactor of `_dispatch_by_assignment()` — changed from mask-based multi-group token assignment to exclusive single-group assignment via argmax of per-group aggregate routing weight scores. Builds `expert_to_group` lookup tensor; computes `group_scores[token, group]` as sum of routing weights for experts in that group; assigns each token to exactly one group (highest total score). Per-group dispatch tuples now contain full (unmasked) scales. |
| `tests/unittest/_torch/modules/moe/test_heter_moe.py` | **This session**: Major overhaul — see sections below |
| `tensorrt_llm/_torch/modules/fused_moe/create_moe.py` | Added `HeterCutlassFusedMoE` import, `"HETER"` branch in `get_moe_cls()` and `create_moe_backend()` |
| `tensorrt_llm/_torch/modules/fused_moe/__init__.py` | Added imports/exports for `HeterCutlassFusedMoE`, `BaseDispatchPolicy`, `DispatchPlan`, `RandomDispatchPolicy`, `ConfidenceThresholdPolicy`, `ExpertLoadPolicy` |
| `tensorrt_llm/llmapi/llm_args.py` | Added `"HETER"` to `MoeConfig.backend` Literal, added `heter_config: Optional[Dict[str, Any]]` field |
| `tensorrt_llm/_torch/pyexecutor/model_loader.py` | Added 3 lines to pass `heter_config` from `moe_config` into `config.extra_attrs['heter_moe_config']` |

## Dispatch Refactor (policy/heter_dispatch.py)

### `_dispatch_by_assignment()` — Semantic change: mask-based → exclusive assignment

**Old behavior**: Each token could appear in multiple groups; routing scales were zeroed out for non-group experts, effectively masking tokens across groups.

**New behavior**: Each token is assigned to exactly ONE group via argmax of per-group aggregate routing weight scores.

**Implementation**:
1. Builds `expert_to_group` lookup tensor (shape: `[num_experts]`)
2. Computes `group_scores[token, group]` = sum of routing weights for experts belonging to that group
3. Assigns each token to the group with the highest total score via `torch.argmax`
4. Per-group dispatch tuples now contain full (unmasked) scales — no zeroing needed

## Test Overhaul (test_heter_moe.py — This Session)

### Imports
- Removed `NVFP4QuantizeUtil` import
- Added manual quantization via `torch.ops.trtllm.fp4_quantize`

### Weight creation helpers
- `_create_unquantized_weights()` — now accepts `kaiming_fan_out=True` parameter; applies Kaiming fan-out scaling to keep activations in sane range
- `_quantize_bf16_to_nvfp4()` — new function replacing `_create_nvfp4_weights()`; quantizes existing BF16 weights directly using `torch.ops.trtllm.fp4_quantize` for fair BF16-vs-NVFP4 comparison

### Benchmark infrastructure overhaul (TestRuntimeBenchmark)

| Helper | Description |
|--------|-------------|
| `_get_l2_cache_size_bytes()` | Uses cuda-python driver bindings to query device L2 cache size |
| `_prepare_single_runner()` | Replaces `_make_benchmark_fn()` — handles torch.compile wrapping, warmup, optional CUDA graph capture |
| `_create_rotating_runner()` | Builds a rotating runner from N independent workspaces (CUTLASS example 79e pattern) to ensure L2-cache-cold measurements |
| `_benchmark_timed()` | Replaces `_benchmark_forward()` — CUDA event timing only (median ms) |
| `_create_one_workspace()` | Creates a single workspace tuple with fresh GPU allocations |
| `_estimate_workspace_bytes()` | Estimates GPU bytes per workspace per scheme |
| `_create_benchmark_workspaces()` | Builds rotated workspaces until cumulative footprint > 3× L2 cache |
| `_MAX_WORKSPACE_COUNT = 16` | Class constant added to TestRuntimeBenchmark |

### Scaled-up test parameters (TestRuntimeBenchmark)

| Parameter | Old | New |
|-----------|-----|-----|
| `NUM_EXPERTS` | 8 | 128 |
| `HIDDEN_SIZE` | 4096 | 2048 |
| `INTERMEDIATE_SIZE` | 4096 | 768 |
| `SEQ_LEN` | 64 | 128 |
| `TOP_K` | 2 | 8 |

### Enhanced policy tests
- `test_confidence_threshold_dispatch` — now verifies ordering invariant (later groups have higher mean routing weights); uses `seq_len=256`, `top_k=4`
- `test_expert_load_dispatch` — now verifies ordering invariant (later groups have higher activation counts); uses `seq_len=256`, `top_k=4`
- `_create_backend_with_weights()` — now accepts `seq_len` and `top_k` keyword overrides

### Scaled correctness test
- `test_heter_two_groups_matches_cutlass` — updated to `seq=128`, `top_k=8`, `num_experts=128` (was 8/2/8)

### Temporary debug prints added
- `test_heter_single_group_matches_cutlass`
- `test_all_nvfp4_faster_than_all_bf16`

### What was NOT changed
Config validation tests untouched. Phase 2 config helpers (`_all_bf16_heter_config`, `_all_nvfp4_heter_config`, `_mixed_heter_config`) retained.
