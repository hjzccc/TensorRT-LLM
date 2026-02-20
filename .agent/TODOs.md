# TODOs — Heterogeneous Precision MoE Backend

## Completed

- [x] **Dual weight set loading** — `register_group_weights()`, `_load_group_from_checkpoint()`, `_load_all_group_checkpoints()` load per-group checkpoints via `_GroupWeightSet`.
- [x] **Per-group quant flags in `run_moe()`** — Each group resolves its own weight source, dtype, and quant scales.
- [x] **NVFP4 input quantization** — `_quantize_input_for_group()` handles per-group input quantization (`fp4_quantize` for NVFP4, passthrough for BF16).
- [x] **Per-group quant scales** — Each `_GroupWeightSet` carries its own `quant_scales`, `fc31_input_scale`, etc.
- [x] **Sentinel-mask dispatch** — `_dispatch_from_expert_to_group()` sends all tokens to every group with non-group slots sentinel-masked (zero scale). No weight subsetting or token filtering needed.
- [x] **CUDA graph-friendly dispatch policies** — All policies use pre-allocated buffers (`_expert_to_group_buf`, `_group_labels`) initialized in the base class `__init__`. `_assign_by_score_gpu` writes in-place. No per-call allocations in the hot path.
- [x] **Unified `_assign_by_score_gpu` path** — All three policies (`RandomHeterDispatch`, `ConfidenceThresholdHeterDispatch`, `ExpertLoadHeterDispatch`) use the same GPU scoring function. `RandomHeterDispatch` uses `torch.rand` scores at construction.
- [x] **L2-cache-cold benchmark infrastructure** — Workspace rotation via cuda-python driver bindings for cold-cache microbenchmarks.
- [x] **Runtime benchmark tests** — `TestPhase2RuntimeBenchmark` with NVFP4 vs BF16 comparison, L2 cache rotation, fair weight quantization.
- [x] **Policy ordering invariant tests** — Strengthened ordering assertions for confidence threshold and expert load policies.

## Performance Optimizations

- [ ] **Fused multi-precision kernel** — Currently dispatches N separate `fused_moe()` calls (one per group) and sums outputs. A single fused kernel handling mixed-precision experts would eliminate redundant memory traffic on input/output tensors.
  - Files: CUDA kernels in `cpp/tensorrt_llm/kernels/`

## Policy Improvements

- [ ] **EMA smoothing for `ExpertLoadHeterDispatch`** — Current implementation computes activation frequency from a single batch. Add exponential moving average to smooth estimates across calls for more stable assignment.
  - Files: `policy/heter_dispatch.py` (`ExpertLoadHeterDispatch`)
  - Design: track `_ema_counts` state, update each `_assign()` call with configurable decay factor

- [ ] **Router logits-based policy** — Policy using `router_logits` (pre-softmax) rather than `token_final_scales` (post-softmax) for expert importance scoring.
  - Files: `policy/heter_dispatch.py`

- [ ] **Adaptive threshold policy** — Adjusts high/low precision split based on running quality estimates (e.g., perplexity feedback from an evaluation hook).

- [ ] **CUDA graph re-dispatch** — Dispatch decisions under CUDA graphs are baked on capture. Implement a mechanism to invalidate and re-capture graphs when assignment changes significantly.
  - Needs coordination with TRT-LLM's graph capture infrastructure

## Testing

- [ ] **Enable mixed runtime test** — Remove `pytest.mark.skip` on `test_heter_mixed_runtime_between_extremes` and verify with dual weight loading.

- [ ] **Integration test with real model** — End-to-end test loading a small MoE model (e.g., Mixtral-8x7B) with HETER backend configured with 2 groups.

- [ ] **Benchmark dispatch overhead** — Measure Python overhead of per-call dispatch (policy + `_assign_by_score_gpu`) to confirm it's acceptable relative to kernel time.

## Documentation

- [ ] **User guide** — Document how to configure `MoeConfig(backend="HETER", heter_config={...})`, available policies, and expected behavior.

- [ ] **Developer guide** — Document how to add new dispatch policies (inherit `HeterDispatchPolicy`, implement `_assign()`).
