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
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_heter.py` | Core `HeterCutlassFusedMoE` class. **Phase 1**: Removed BF16-centric `update_hot_experts()`, `_find_bf16_group_index()`, `_make_default_assignment()`. Added policy-based dispatch: `set_dispatch_policy()`, `_recompute_dispatch()` called per `run_moe()`. **Signal passing**: Added `forward_chunk()` override to stash `router_logits` in `_pending_router_logits` before calling `super().forward_chunk()`, consumed by `run_moe()` override within the same call stack. `_recompute_dispatch()` now accepts `token_selected_experts`, `token_final_scales`, and `router_logits` and forwards to `policy.assign()`. Documented torch.compile/CUDA graph safety. *(no changes in this session)* |
| `tests/unittest/_torch/modules/moe/test_heter_moe.py` | **Phase 2 test updates** — see section below |
| `tensorrt_llm/_torch/modules/fused_moe/create_moe.py` | Added `HeterCutlassFusedMoE` import, `"HETER"` branch in `get_moe_cls()` and `create_moe_backend()` |
| `tensorrt_llm/_torch/modules/fused_moe/__init__.py` | Added imports/exports for `HeterCutlassFusedMoE`, `BaseDispatchPolicy`, `DispatchPlan`, `RandomDispatchPolicy`, `ConfidenceThresholdPolicy`, `ExpertLoadPolicy` |
| `tensorrt_llm/llmapi/llm_args.py` | Added `"HETER"` to `MoeConfig.backend` Literal, added `heter_config: Optional[Dict[str, Any]]` field |
| `tensorrt_llm/_torch/pyexecutor/model_loader.py` | Added 3 lines to pass `heter_config` from `moe_config` into `config.extra_attrs['heter_moe_config']` |

## Phase 2 Test Changes (test_heter_moe.py)

### Imports
- Added `NVFP4QuantizeUtil` from `quantize_utils` (creates NVFP4-quantized weights)
- Added `QuantConfig` from `tensorrt_llm.models.modeling_utils`

### Global flags (lines 47-59)
- `ENABLE_TORCH_COMPILE = False` — wraps forward with `torch.compile(mode="reduce-overhead")` when True
- `ENABLE_CUDA_GRAPHS = False` — captures + replays CUDA graphs for benchmark timing when True
- `_WARMUP_ITERS = 10`, `_BENCH_ITERS = 50` — benchmark iteration counts
- Existing correctness tests run in eager mode; flags only affect benchmark tests

### New helpers
- `_create_nvfp4_weights(num_experts, hidden_size, intermediate_size, dtype, x)` — NVFP4-quantized weights via `NVFP4QuantizeUtil`, derives activation scale from input
- `_create_cutlass_backend(..., weights, quant_config=None)` — generic CutlassFusedMoE factory, loads weights, moves to CUDA
- `_make_benchmark_fn(backend, x, router_logits, all_rank_num_tokens)` — creates forward callable, optionally wraps with torch.compile
- `_benchmark_forward(fn, warmup_iters, bench_iters)` — CUDA event timing (median ms), warmup, optional manual CUDA graph capture/replay

### Phase 2 config helpers
- `_all_bf16_heter_config()` — single-group all-BF16
- `_all_nvfp4_heter_config()` — single-group all-NVFP4
- `_mixed_heter_config(bf16_ratio=0.5)` — two-group BF16+NVFP4

### TestPhase2RuntimeBenchmark class (8 experts, h=4096, inter=4096, seq=64, top_k=2)

| Test | Status | What |
|------|--------|------|
| `test_all_nvfp4_faster_than_all_bf16` | **Active** (skipif no NVFP4 hw) | Creates BF16 + NVFP4 CutlassFusedMoE, logs output deviation, benchmarks via CUDA events, asserts `nvfp4_ms < bf16_ms` |
| `test_heter_mixed_runtime_between_extremes` | **Skipped** (needs phase 2) | Creates all-BF16, all-NVFP4, mixed HeterCutlassFusedMoE; asserts `nvfp4_ms <= mixed_ms <= bf16_ms`; has TODO for dual weight set loading |

### What was NOT changed
All existing tests untouched: 3 forward equivalence, 9 config validation, 10 dispatch policy.
