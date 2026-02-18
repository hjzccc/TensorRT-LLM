# File Change Log — Heterogeneous Precision MoE Backend

## New Files

| File | Description |
|------|-------------|
| `tensorrt_llm/_torch/modules/fused_moe/policy/__init__.py` | Policy module exports (`BaseDispatchPolicy`, `DispatchPlan`, `RandomDispatchPolicy`, `ConfidenceThresholdPolicy`, `ExpertLoadPolicy`) |
| `tensorrt_llm/_torch/modules/fused_moe/policy/dispatch_plan.py` | `DispatchPlan` dataclass — maps experts to precision groups, with `validate()` |
| `tensorrt_llm/_torch/modules/fused_moe/policy/strategies.py` | `BaseDispatchPolicy` ABC, `RandomDispatchPolicy`, `ConfidenceThresholdPolicy` (mean routing weight), `ExpertLoadPolicy` (activation frequency), `_assign_by_score()` shared helper |
| `tests/unittest/_torch/modules/moe/test_heter_moe.py` | Unit tests: forward equivalence (2), config validation (8), dispatch policy (10 — random, confidence, expert-load, signal fallback, custom) |
| `.agent/TODOs.md` | Future work tracking — Phase 2 (dual weights, per-group quant), Phase 3 (CUDA ops), policy improvements, testing, docs |

## Modified Files

| File | What Changed |
|------|-------------|
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_heter.py` | Core `HeterCutlassFusedMoE` class. **Phase 1**: Removed BF16-centric `update_hot_experts()`, `_find_bf16_group_index()`, `_make_default_assignment()`. Added policy-based dispatch: `set_dispatch_policy()`, `_recompute_dispatch()` called per `run_moe()`. **Signal passing**: Added `forward_chunk()` override to stash `router_logits` in `_pending_router_logits` before calling `super().forward_chunk()`, consumed by `run_moe()` override within the same call stack. `_recompute_dispatch()` now accepts `token_selected_experts`, `token_final_scales`, and `router_logits` and forwards to `policy.assign()`. Documented torch.compile/CUDA graph safety. |
| `tensorrt_llm/_torch/modules/fused_moe/create_moe.py` | Added `HeterCutlassFusedMoE` import, `"HETER"` branch in `get_moe_cls()` and `create_moe_backend()` |
| `tensorrt_llm/_torch/modules/fused_moe/__init__.py` | Added imports/exports for `HeterCutlassFusedMoE`, `BaseDispatchPolicy`, `DispatchPlan`, `RandomDispatchPolicy`, `ConfidenceThresholdPolicy`, `ExpertLoadPolicy` |
| `tensorrt_llm/llmapi/llm_args.py` | Added `"HETER"` to `MoeConfig.backend` Literal, added `heter_config: Optional[Dict[str, Any]]` field |
| `tensorrt_llm/_torch/pyexecutor/model_loader.py` | Added 3 lines to pass `heter_config` from `moe_config` into `config.extra_attrs['heter_moe_config']` |
