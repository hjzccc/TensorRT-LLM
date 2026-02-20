# TODOs — Heterogeneous Precision MoE Backend

## Phase 2: Dual Weight Loading & Per-Group Quantization

### P0 — Must have for functional multi-precision dispatch

- [ ] **Dual weight set loading** — Load weights from per-group checkpoints (one checkpoint per precision group).  Currently all groups subset from a single parent weight set.  Requires hooking into `post_load_weights()` or the model loader to read N sets of `w3_w1_weight`, `w2_weight`, and corresponding quant scales.
  - Files: `fused_moe_heter.py` (`_build_group_caches`, `post_load_weights`), `model_loader.py`
  - Depends on: understanding how TRT-LLM weight loading pipeline works for quantized checkpoints

- [ ] **Per-group quant flags in `run_moe()`** — Branch on `desc.quant_algo` to set different weight dtypes and quant flags for each group's `fused_moe()` call.  Currently all groups use the parent's single-precision flags (marked as TODO(phase2) in code).
  - Files: `fused_moe_heter.py` (`run_moe`, around line 726-733)

- [ ] **NVFP4 input quantization** — For groups with `quant_algo=NVFP4`, call `torch.ops.trtllm.fp4_quantize` on the input before passing to `fused_moe()`.  The parent class does this in `forward_chunk()` but we need per-group quantization.
  - Files: `fused_moe_heter.py` (`run_moe`)
  - Reference: `fused_moe_cutlass.py` line ~720-750 (FP4 quantization path)

### P1 — Important for correctness and usability

- [ ] **Weight subset cache from correct weight set** — `_build_group_caches()` currently subsets from `self.w3_w1_weight` (parent's single set).  When dual weights are loaded, each group must select from its own weight set.
  - Files: `fused_moe_heter.py` (`_build_group_caches`)

- [ ] **Per-group quant scales** — Each precision group needs its own quant scales (e.g., NVFP4 has block scales, BF16 has none).  Currently `_subset_quant_scales` takes from the parent's single scales.
  - Files: `fused_moe_heter.py` (`_build_group_caches`, `_subset_quant_scales`)

## Phase 3: CUDA Operation Modifications

- [ ] **Fused multi-precision kernel** — Currently dispatches N separate `fused_moe()` calls (one per group) and sums outputs.  A single fused kernel that handles mixed-precision experts would eliminate redundant memory traffic on the input and output tensors.
  - Files: CUDA kernels in `cpp/tensorrt_llm/kernels/`
  - This is a significant performance optimization — the current Python multi-call approach is correct but suboptimal.

- [ ] **Avoid weight subset copies** — `self.w3_w1_weight[ids_tensor].contiguous()` does an expensive copy.  A CUDA kernel that indexes directly into the full weight tensor with a remap table would avoid this.
  - Currently mitigated by caching (only rebuilds when assignment changes)

## Policy Improvements

- [x] **Exclusive token-to-group dispatch** — `_dispatch_by_assignment()` in `policy/heter_dispatch.py` was refactored so each token appears in exactly ONE group (argmax of per-group aggregate routing weight), replacing the old mask-based approach where each token appeared in multiple groups with zeroed scales for non-group experts.  This is an architectural correctness fix.
  - Files: `policy/heter_dispatch.py` (`_dispatch_by_assignment`)

- [ ] **EMA smoothing for ExpertLoadPolicy** — Current `ExpertLoadPolicy` computes frequency per-call from a single batch.  Add exponential moving average (EMA) to smooth frequency estimates across multiple `run_moe()` calls for more stable assignment.
  - Files: `policy/strategies.py` (`ExpertLoadPolicy`)
  - Design: track `_ema_counts` state in policy, update each `assign()` call with configurable decay factor

- [ ] **Router logits-based policy** — Implement a policy that uses `router_logits` (pre-softmax) rather than `token_final_scales` (post-softmax) for expert importance scoring.  Pre-softmax logits may better distinguish expert confidence.
  - Files: `policy/strategies.py`

- [ ] **Adaptive threshold policy** — Policy that adjusts the high/low precision split based on a running estimate of model quality (e.g., perplexity feedback from an evaluation hook).

- [ ] **CUDA graph-aware re-dispatch** — Currently, dispatch decisions under CUDA graphs are baked on first capture.  Implement a mechanism to invalidate and re-capture graphs when assignment changes significantly.
  - Likely needs coordination with TRT-LLM's graph capture infrastructure

## Signal Extraction (Deferred)

- [ ] **Attention score extraction** — Extract per-token attention scores from attention modules for use as additional dispatch signals.  Very invasive — attention scores are computed in C++ kernels.  Only `softmax_stats` (max_logit + sum_exp) exists for MLA+ContextParallelism.
  - Status: Deferred indefinitely.  Agreed to use router signals only for now.
  - If revisited: would need C++ kernel modifications or a custom attention wrapper

## Testing

- [ ] **Run existing test suite** — `test_heter_moe.py` is written but not yet executed.  Requires either Docker with C++ bindings or mocking `torch.ops.trtllm.fused_moe`.
  - Container: `df06e32fa3c0` (image `nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc2`)
  - Working install: `/home/huanchen/TensorRT-LLM/` → `/code/tensorrt_llm`
  - Note: API divergence (`swiglu_gptoss_style` vs `gptoss_style`) between repos

- [x] **Phase 2 runtime benchmark tests** — Added `TestPhase2RuntimeBenchmark` class with:
  - `test_all_nvfp4_faster_than_all_bf16` (active) — CutlassFusedMoE(BF16) vs CutlassFusedMoE(NVFP4) runtime comparison using CUDA events
  - `test_heter_mixed_runtime_between_extremes` (skipped until phase 2) — asserts mixed heter runtime is between all-BF16 and all-NVFP4
  - Global flags `ENABLE_TORCH_COMPILE` / `ENABLE_CUDA_GRAPHS` to toggle those code paths
  - Mixed test has TODO for dual weight set loading once phase 2 weight API lands
  - [x] **L2-cache-cold benchmark infrastructure** — Workspace rotation implemented (CUTLASS example 79e pattern):
    - `_get_l2_cache_size_bytes()` via cuda-python driver bindings
    - `_prepare_single_runner()`, `_create_rotating_runner()`, `_benchmark_timed()`
    - `_create_one_workspace()`, `_estimate_workspace_bytes()`, `_create_benchmark_workspaces()`
    - Workspaces rotate to exceed 3× L2 cache for cold measurements
  - [x] **Fairer NVFP4 weight comparison** — `_create_nvfp4_weights()` replaced by `_quantize_bf16_to_nvfp4()` which quantizes directly from BF16 weights for fair comparison; `_create_unquantized_weights()` now uses `kaiming_fan_out=True` for variance control
  - [x] **Scaled-up benchmark parameters** — 128 experts, top_k=8, seq=128, hidden=2048, intermediate=768

- [x] **Enhanced policy ordering invariant tests** — `test_confidence_threshold_dispatch` and `test_expert_load_dispatch` have strengthened ordering assertions to verify policy correctness.

- [ ] **Enable mixed runtime test** — Remove `pytest.mark.skip` on `test_heter_mixed_runtime_between_extremes` and update dual weight loading once phase 2 is implemented.

- [ ] **Integration test with real model** — End-to-end test loading a small MoE model (e.g., Mixtral-8x7B) with HETER backend configured with 2 groups.

- [ ] **Benchmark dispatch overhead** — Measure the Python overhead of per-call dispatch (policy + cache check + potential cache rebuild) to confirm it's acceptable.

## Documentation

- [ ] **User guide** — Document how to configure `MoeConfig(backend="HETER", heter_config={...})`, available policies, and expected behavior.

- [ ] **Developer guide** — Document how to add new dispatch policies (inherit `BaseDispatchPolicy`, implement `assign()`).

## Infrastructure

- [x] **L2-cache-cold benchmark infrastructure** — Added cuda-python driver bindings (`cuda.cuda`) to query L2 cache size (`CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE`) for realistic cold-cache microbenchmarks.  Workspace rotation buffers exceed 3× L2 to ensure cache eviction between timed runs.
  - Files: `tests/unittest/_torch/modules/moe/test_heter_moe.py`

- [ ] **Reconcile repo divergence** — Our project repo (`/home/huanchen/heter-moe-proj/TensorRT-LLM/`) vs working TRT-LLM install (`/home/huanchen/TensorRT-LLM/`) have minor API differences (`gptoss_style` naming).  Need to either sync or establish a consistent workflow.
