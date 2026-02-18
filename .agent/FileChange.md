# File Change Log — Heterogeneous Precision MoE Backend

## New Files

| File | Description |
|------|-------------|
| `tensorrt_llm/_torch/modules/fused_moe/policy/__init__.py` | Policy module exports (`BaseDispatchPolicy`, `DispatchPlan`, `RandomDispatchPolicy`, etc.) |
| `tensorrt_llm/_torch/modules/fused_moe/policy/dispatch_plan.py` | `DispatchPlan` dataclass — maps experts to precision groups, with `validate()` |
| `tensorrt_llm/_torch/modules/fused_moe/policy/strategies.py` | `BaseDispatchPolicy` ABC, `RandomDispatchPolicy`, stubs for `ConfidenceThresholdPolicy` and `ExpertLoadPolicy` |
| `tests/unittest/_torch/modules/moe/test_heter_moe.py` | Unit tests for `HeterCutlassFusedMoE` — forward equivalence, config validation, dispatch policy |

## Modified Files

| File | What Changed |
|------|-------------|
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_heter.py` | Core `HeterCutlassFusedMoE` class. Removed BF16-centric `update_hot_experts()`, `_find_bf16_group_index()`, `_make_default_assignment()`. Added policy-based dispatch: `set_dispatch_policy()`, `_recompute_dispatch()` called per `run_moe()`. |
| `tensorrt_llm/_torch/modules/fused_moe/create_moe.py` | Added `HeterCutlassFusedMoE` import, `"HETER"` branch in `get_moe_cls()` and `create_moe_backend()` |
| `tensorrt_llm/_torch/modules/fused_moe/__init__.py` | Added imports/exports for `HeterCutlassFusedMoE`, `BaseDispatchPolicy`, `DispatchPlan`, `RandomDispatchPolicy` |
| `tensorrt_llm/llmapi/llm_args.py` | Added `"HETER"` to `MoeConfig.backend` Literal, added `heter_config: Optional[Dict[str, Any]]` field |
| `tensorrt_llm/_torch/pyexecutor/model_loader.py` | Added 3 lines to pass `heter_config` from `moe_config` into `config.extra_attrs['heter_moe_config']` |
