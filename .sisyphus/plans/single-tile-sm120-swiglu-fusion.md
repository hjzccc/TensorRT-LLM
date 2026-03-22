# Single-Tile SM120 NVFP4 SwiGLU Fusion for `runMoe()`

## TL;DR
> **Summary**: Enable true SwiGLU epilogue fusion for the SM120 NVFP4 single-tile `runMoe()` path only, using a fail-closed CTA_M=128 gate and an alternate interleaved FC1 layout that matches the existing fused visitor contract.
> **Deliverables**:
> - Single-tile `runMoe()` fused SwiGLU enablement behind an explicit supported-path gate
> - Caller-side interleaved FC1 weight/scale preparation for single-tile validation surfaces only
> - Fixed SM120 packed-FP4 `end_loop()` store path with no hang on supported tiles
> - Deterministic correctness, fallback, `ncu`, and `nsys` evidence under `.sisyphus/evidence/`
> **Effort**: Large
> **Parallel**: YES - 2 waves
> **Critical Path**: 1 -> 2 -> 3 -> 4 -> 5 -> 7

## Context
### Original Request
- Make the normal single-tile `run_moe` kernel work with SwiGLU fusion.
- Stay on SM120 with NVFP4.
- Keep it high performance.
- Build with `-j2`.
- Validate with both `ncu` and `nsys`.

### Interview Summary
- Scope is the single-tile CUTLASS `runMoe()` path, not `runMoeDualTile()`.
- Test strategy is `tests-after`.
- Interleaving strategy is a single-tile-only alternate layout, not a global `moe.py` format change and not a per-dispatch runtime reorder.
- Defaults applied from repo evidence: support only CTA_M=128 in v1; keep dual-tile unchanged; keep unsupported tactics/conditions on the unfused path.
- Verification will use direct `FusedMoeRunner.run_moe(...)` and `added_benchmark` scripts rather than backend-wide enablement work.

### Metis Review (gaps addressed)
- Treat `sm90_visitor_swiglu_store.hpp:586` as a blocking correctness/stability fix; do not enable the feature if the packed-FP4 store still hangs.
- Gate fusion to CTA_M=128 only; the current visitor supports M128/M32, and current SM120 evidence already forces M128 in `added_benchmark/test_swiglu_fusion.py:146`.
- Permute FC1 NVFP4 block scales together with FC1 weights; never permute only weights.
- Carry dual-tile's FC2 activation-scale zeroing rule into single-tile fused execution so padding rows do not poison GEMM2 dequantization.
- Keep global loaders and backend capability tables out of scope for this plan.

## Work Objectives
### Core Objective
- Land a fail-closed SM120 NVFP4 single-tile SwiGLU fusion path in `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3865` that reuses the existing fused visitor, skips the standalone activation kernel on the supported path, and falls back cleanly everywhere else.

### Deliverables
- Single-tile fusion gate and TMA epilogue parameter plumbing in `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`.
- Deterministic single-tile correctness and fallback benchmark scripts under `added_benchmark/`.
- Stable packed-FP4 epilogue store behavior in `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp`.
- Reproducible profiling commands and saved evidence for fused vs unfused single-tile runs.

### Definition of Done (verifiable conditions with commands)
- `python3 scripts/build_wheel.py -a "120-real" -j 2 -l -s --build_dir cpp/build_sm120_swiglu` exits `0`.
- `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0` exits `0` and prints `PASS: single-tile fused matches unfused within tolerance`.
- `python3 added_benchmark/test_single_tile_swiglu_guard.py` exits `0` and prints `PASS: unsupported single-tile cases stayed unfused`.
- `ncu --set full --kernel-name-base demangled --kernel-name regex:'GemmUniversal' --launch-skip 10 --launch-count 5 -o .sisyphus/evidence/task-7-single-tile-fused python3 added_benchmark/ncu_tile_roofline.py 0` exits `0`.
- `nsys profile --trace=cuda,nvtx --output=.sisyphus/evidence/task-7-single-tile-fused python3 added_benchmark/ncu_tile_roofline.py 0` exits `0`, and `nsys stats --report gpukernsum` shows no standalone gated-activation kernel in the fused run.

### Must Have
- Fusion enablement only when all of these are true: explicit `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1` opt-in, SM120, NVFP4, single-tile `runMoe()`, `ActivationType::Swiglu`, TMA-warp-specialized path, CTA_M=128, no prequant-scale kernel, and `FORCE_UNFUSED_SWIGLU` unset.
- FC1 interleave order is exactly `[up_0, gate_0, up_1, gate_1, ...]`, matching the fused visitor math documented in `added_benchmark/test_swiglu_fusion.py:29`.
- FC1 NVFP4 block scales follow the exact same permutation as FC1 weights.
- Dual-tile logic and all unsupported single-tile cases keep current unfused behavior.
- Agent-generated evidence files for correctness, fallback, `ncu`, and `nsys` runs.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No edits to `tensorrt_llm/layers/moe.py:538` or any global MoE weight-layout contract.
- No backend-wide SM120 capability-table expansion in `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py:84`.
- No dual-tile redesign, no M64/M256 fusion bring-up, and no M32 expansion in this change set.
- No per-dispatch GPU reorder of FC1 weights in the hot path.
- No manual-only validation; every acceptance step must be command-driven.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: `tests-after` using direct runner benchmark scripts and existing single-tile profiler harnesses.
- QA policy: every task includes a happy-path and failure-path command with a saved evidence artifact.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`.
- Build policy: use `python3 scripts/build_wheel.py ... -j 2` for native rebuilds.
- Profiling policy: compare fused vs `FORCE_UNFUSED_SWIGLU=1` on the same single-tile tactic and workload.

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. Foundation first, then dependent kernel enablement and profiling.

Wave 1: Task 1 benchmark harnesses; Task 2 single-tile fusion gate/workspace rules; Task 3 TMA epilogue plumbing; Task 6 unsupported-case guardrails.

Wave 2: Task 4 activation/prequant bypass; Task 5 packed-FP4 store fix; Task 7 profiling automation; Task 8 final verification sweep and regression cleanup.

### Dependency Matrix (full, all tasks)
- `1 -> 2, 6, 7, 8`
- `2 -> 3, 4, 6, 8`
- `3 -> 4, 5, 7, 8`
- `4 -> 7, 8`
- `5 -> 7, 8`
- `6 -> 8`
- `7 -> 8`
- `8 -> Final Verification Wave`

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 4 tasks -> `writing`, `unspecified-high`, `quick`
- Wave 2 -> 4 tasks -> `unspecified-high`, `deep`, `quick`
- Final Verification Wave -> 4 tasks -> `oracle`, `unspecified-high`, `deep`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Build deterministic single-tile SwiGLU fusion harnesses

  **What to do**: Add `added_benchmark/test_single_tile_swiglu_fusion.py` and `added_benchmark/test_single_tile_swiglu_guard.py`. The fusion harness must use the same deterministic dimensions and gate/up semantics as `added_benchmark/test_swiglu_fusion.py:65`, but it must call `torch.classes.trtllm.FusedMoeRunner.run_moe(...)` instead of `run_moe_dual_tile(...)`. Build FC1 test tensors from separate gate/up sources and materialize the fused layout as `[up_0, gate_0, up_1, gate_1, ...]`; permute FC1 NVFP4 block scales with the same order. The guard harness must exercise tactics `1`, `2`, and `3` and prove they stay numerically aligned with the explicit `FORCE_UNFUSED_SWIGLU=1` path.
  **Must NOT do**: Do not reuse `run_moe_dual_tile(...)`; do not mutate global weight-loading code; do not rely on manual tensor inspection.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: new deterministic benchmark scripts drive all downstream verification.
  - Skills: `[]` - repo-local Python and runner usage only.
  - Omitted: `[]` - no extra skill load is required.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 2, 6, 7, 8 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `added_benchmark/test_swiglu_fusion.py:29` - exact `[up, gate]` interleave order required to make visitor math match the unfused path.
  - Pattern: `added_benchmark/test_swiglu_fusion.py:75` - existing NVFP4 quantization helper for weight/scale generation.
  - Pattern: `added_benchmark/test_swiglu_fusion.py:168` - how the current fused-vs-unfused subprocess A/B flow is structured.
  - Pattern: `added_benchmark/ncu_tile_roofline.py:20` - existing single-tile `run_moe(...)` tactic selection and invocation pattern.
  - API/Type: `tensorrt_llm/_torch/modules/fused_moe/quantization.py:162` - existing `interleave_linear_and_gate(...)` helper semantics to mirror in the harness.
  - API/Type: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:91` - `FORCE_UNFUSED_SWIGLU` escape hatch for A/B comparison.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 | tee .sisyphus/evidence/task-1-single-tile-swiglu.log` exits `0` and prints `PASS: single-tile fused matches unfused within tolerance`.
  - [ ] `python3 added_benchmark/test_single_tile_swiglu_guard.py | tee .sisyphus/evidence/task-1-single-tile-guard.log` exits `0` and prints `PASS: unsupported single-tile cases stayed unfused`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Single-tile fused A/B correctness
    Tool: Bash
    Steps: Run `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 | tee .sisyphus/evidence/task-1-single-tile-swiglu.log`
    Expected: Exit code 0; log contains `PASS: single-tile fused matches unfused within tolerance`
    Evidence: .sisyphus/evidence/task-1-single-tile-swiglu.log

  Scenario: Unsupported tactics stay unfused
    Tool: Bash
    Steps: Run `python3 added_benchmark/test_single_tile_swiglu_guard.py | tee .sisyphus/evidence/task-1-single-tile-guard.log`
    Expected: Exit code 0; log contains `PASS: unsupported single-tile cases stayed unfused`
    Evidence: .sisyphus/evidence/task-1-single-tile-guard.log
  ```

  **Commit**: YES | Message: `test(moe): add single-tile swiglu fusion harnesses` | Files: `added_benchmark/test_single_tile_swiglu_fusion.py`, `added_benchmark/test_single_tile_swiglu_guard.py`

- [ ] 2. Add a fail-closed single-tile fusion gate and fused workspace rules in `runMoe()`

  **What to do**: In `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3865`, introduce a local `enable_swiglu_fusion` for the single-tile path. The predicate is exact: `getEnableSingleTileSwigluFusion()` returns true from a new env-backed helper beside `getForceUnfusedSwiglu()`, `is_gated_activation`, NVFP4 path active, `fc1_activation_type.activation_type == ActivationType::Swiglu`, `!usePrequantScaleKernel(quant_params)`, `!getForceUnfusedSwiglu()`, TMA-warp-specialized GEMM1 selected, and `gemm1_config_->tile_config_sm120 == tkc::CutlassTileConfigSM120::CtaShape128x128x128B` (the only supported grouped-GEMM M128 config from `cutlass_heuristic.cpp:519`). When the gate is true, keep `fc1_result_` as the packed-FP4 destination and zero `fc2_fp4_act_scale_` padding rows exactly like the dual-tile path does.
  **Must NOT do**: Do not touch the dual-tile gate at `moe_kernels.cu:4295`; do not enable fusion for `CtaShape64x128x64B`, `CtaShape32x128x64B`, or `CtaShape256x128x64B`; do not add a per-dispatch reorder kernel.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: this is the main safety gate for the kernel feature.
  - Skills: `[]` - repo-local CUDA/C++ changes only.
  - Omitted: `[]` - no extra skill load is required.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 3, 4, 6, 8 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4117` - single-tile TMA input setup entry point.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4148` - current single-tile FC2 prequant-scale path and output-buffer choice.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:91` - existing env-backed unfused escape hatch; add the symmetric single-tile opt-in helper here.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4295` - dual-tile fusion gate to mirror structurally, not behaviorally.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:519` - grouped-GEMM SM120 tactic list; only `CtaShape128x128x128B` is allowed in v1.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h:494` - SM120 grouped-GEMM tile dispatch cases that must remain fail-closed outside M128.
  - API/Type: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:3154` - existing `fused_swiglu_output` workspace alias for `fc1_result_`.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python3 scripts/build_wheel.py -a "120-real" -j 2 -l -s --build_dir cpp/build_sm120_swiglu | tee .sisyphus/evidence/task-2-build.log` exits `0`.
  - [ ] `python3 added_benchmark/test_single_tile_swiglu_guard.py | tee .sisyphus/evidence/task-2-gate.log` exits `0` and reports fusion enabled only when `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1` and tactic `0` are both true.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Supported tactic enables the gate
    Tool: Bash
    Steps: Run `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --print-gate-state | tee .sisyphus/evidence/task-2-gate-supported.log`
    Expected: Exit code 0; log contains `FUSION_GATE=1`
    Evidence: .sisyphus/evidence/task-2-gate-supported.log

  Scenario: Unsupported tactics stay fail-closed
    Tool: Bash
    Steps: Run `python3 added_benchmark/test_single_tile_swiglu_guard.py | tee .sisyphus/evidence/task-2-gate.log`
    Expected: Exit code 0; log contains `PASS: unsupported single-tile cases stayed unfused`
    Evidence: .sisyphus/evidence/task-2-gate.log
  ```

  **Commit**: YES | Message: `feat(moe): gate single-tile sm120 swiglu fusion` | Files: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`

- [ ] 3. Thread single-tile TMA SWIGLU epilogue parameters and packed-output pointers

  **What to do**: After `setupTmaWarpSpecializedInputs(...)` in `runMoe()`, set `gemm1_tma_ws_input.fusion = EpilogueFusion::SWIGLU` when Task 2's gate is true and fill `fused_swiglu_epilogue.ptr_tma_d_base`, `fc2_act_sf_flat`, `expert_first_token_offset`, `global_sf_scale_ptr`, `inter_size`, and `num_experts` with the same semantics as the dual-tile path. Ensure the single-tile TMA input computation still passes `output = fc1_result_` so `computeTmaWarpSpecializedInputPointers(...)` writes `ptr_swiglu_output[...]` into packed FC1 storage while `ptr_d` remains the raw TMA destination (`glu_inter_result_`).
  **Must NOT do**: Do not change `gemm2_tma_ws_input` finalize fusion; do not change `ptr_swiglu_output` byte-offset math; do not repoint `ptr_tma_d_base` away from `glu_inter_result_`.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: TMA param wiring controls both destination buffers and per-expert addressing.
  - Skills: `[]` - repo-local CUDA/C++ changes only.
  - Omitted: `[]` - no extra skill load is required.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 5, 8 | Blocked By: 2

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4153` - single-tile `Self::gemm1(...)` call site.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4303` - dual-tile SWIGLU param population to mirror.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:1265` - how `ptr_swiglu_output` and `ptr_d` are derived during TMA input pointer setup.
  - API/Type: `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h:179` - `FusedSwiGLUEpilogue` fields and meaning.
  - API/Type: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu:119` - backing storage for `ptr_swiglu_output` and `stride_swiglu_output` arrays.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --dump-epilogue-metadata | tee .sisyphus/evidence/task-3-epilogue.log` exits `0` and prints `EPILOGUE_FUSION=SWIGLU`.
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 | tee .sisyphus/evidence/task-3-correctness.log` exits `0` and prints `PASS: single-tile fused matches unfused within tolerance`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: SWIGLU epilogue metadata is populated
    Tool: Bash
    Steps: Run `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --dump-epilogue-metadata | tee .sisyphus/evidence/task-3-epilogue.log`
    Expected: Exit code 0; log contains `EPILOGUE_FUSION=SWIGLU`
    Evidence: .sisyphus/evidence/task-3-epilogue.log

  Scenario: Forced unfused path bypasses SWIGLU epilogue metadata
    Tool: Bash
    Steps: Run `FORCE_UNFUSED_SWIGLU=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --dump-epilogue-metadata | tee .sisyphus/evidence/task-3-epilogue-unfused.log`
    Expected: Exit code 0; log contains `EPILOGUE_FUSION=NONE`
    Evidence: .sisyphus/evidence/task-3-epilogue-unfused.log
  ```

  **Commit**: YES | Message: `feat(moe): wire single-tile swiglu epilogue params` | Files: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`, `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h`, `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_tma_warp_specialized_input.cu`

- [ ] 4. Bypass standalone activation and BF16-only prequant logic on the fused path

  **What to do**: In the single-tile `runMoe()` path, skip the standalone gated-activation kernel and any BF16 activation-to-prequant transform only when Task 2's fusion gate is true. The fused path must pass packed FP4 activations in `fc1_result_` plus `fc2_fp4_act_scale_` directly into GEMM2; the unfused path must keep the current `doActivation(...)` and `applyPrequantScale(...)` behavior. Preserve existing AWQ/non-NVFP4 behavior exactly.
  **Must NOT do**: Do not bypass `applyPrequantScale(...)` for unfused or AWQ cases; do not change `skip_activation` semantics outside the supported fused path.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: this is where fused and unfused dataflow diverge.
  - Skills: `[]` - repo-local CUDA/C++ changes only.
  - Omitted: `[]` - no extra skill load is required.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7, 8 | Blocked By: 2, 3

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4148` - current single-tile FC2 prequant handling.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4169` - current single-tile GEMM2 input selection logic.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4396` - dual-tile activation-skip pattern to mirror.
  - API/Type: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:2286` - standalone `doGatedActivation(...)` entry point that must disappear from the fused supported path.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --assert-packed-gemm2-input | tee .sisyphus/evidence/task-4-packed-input.log` exits `0` and prints `PASS: GEMM2 consumed packed FP4 fused output`.
  - [ ] `FORCE_UNFUSED_SWIGLU=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 | tee .sisyphus/evidence/task-4-unfused.log` exits `0` and still prints the unfused success line.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Fused path bypasses standalone activation
    Tool: Bash
    Steps: Run `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --assert-packed-gemm2-input | tee .sisyphus/evidence/task-4-packed-input.log`
    Expected: Exit code 0; log contains `PASS: GEMM2 consumed packed FP4 fused output`
    Evidence: .sisyphus/evidence/task-4-packed-input.log

  Scenario: Forced unfused path still uses legacy flow
    Tool: Bash
    Steps: Run `FORCE_UNFUSED_SWIGLU=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 | tee .sisyphus/evidence/task-4-unfused.log`
    Expected: Exit code 0; log contains `PASS: single-tile unfused reference completed`
    Evidence: .sisyphus/evidence/task-4-unfused.log
  ```

  **Commit**: YES | Message: `feat(moe): bypass unfused activation on single-tile swiglu path` | Files: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`

- [ ] 5. Stabilize the SM120 packed-FP4 `end_loop()` store path for M128 tiles

  **What to do**: Repair `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp` for the supported M128 single-tile path. Keep the full-warp `quantize_bf16x8_to_fp4_result(...)` participation rules intact, but remove any unsafe read/write behavior after quantization: active-thread guard must cover every shared-memory read used by `end_loop()`, the post-quantization barrier contract must be explicit, and the packed FP4 global write must only execute for in-bounds rows/half-columns. Because v1 is M128-only, fix and verify the M128 path first; leave M32 code compiled but still unreachable via the gate.
  **Must NOT do**: Do not broaden the gate to M32/M64/M256; do not predicate the shuffle-based quantize helper in a way that breaks warp participation; do not treat a comment about "garbage reads" as a valid synchronization strategy for `end_loop()`.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: this is the highest-risk kernel-stability fix in the plan.
  - Skills: `[]` - repo-local CUDA/CUTLASS callback work only.
  - Omitted: `[]` - no extra skill load is required.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 7, 8 | Blocked By: 3

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp:453` - current M128 active-thread region and quantize call.
  - Pattern: `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp:483` - packed FP4 + SF writes into shared storage.
  - Pattern: `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp:586` - `end_loop()` global-store path.
  - Pattern: `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp:613` - current shared-memory source tensor used by the global copy.
  - API/Type: `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp:365` - callback state carried from epilogue launch into reduce/end_loop.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --num-tokens 3 | tee .sisyphus/evidence/task-5-residue-small.log` exits `0`.
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --num-tokens 65 | tee .sisyphus/evidence/task-5-residue-large.log` exits `0`.
  - [ ] Neither log contains `hang`, `cudaError`, or `illegal memory access`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Small residue tile completes without hang
    Tool: Bash
    Steps: Run `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --num-tokens 3 | tee .sisyphus/evidence/task-5-residue-small.log`
    Expected: Exit code 0; no `hang`, `cudaError`, or `illegal memory access` in the log
    Evidence: .sisyphus/evidence/task-5-residue-small.log

  Scenario: Cross-tile residue completes without hang
    Tool: Bash
    Steps: Run `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --num-tokens 65 | tee .sisyphus/evidence/task-5-residue-large.log`
    Expected: Exit code 0; no `hang`, `cudaError`, or `illegal memory access` in the log
    Evidence: .sisyphus/evidence/task-5-residue-large.log
  ```

  **Commit**: YES | Message: `fix(moe): stabilize sm120 swiglu fp4 epilogue store` | Files: `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp`

- [ ] 6. Encode and verify unsupported-case fallback behavior

  **What to do**: Finalize the exact fallback matrix in code and harnesses. Unsupported cases are: `ENABLE_SINGLE_TILE_SWIGLU_FUSION` absent, `FORCE_UNFUSED_SWIGLU=1`, tactic `1`, tactic `2`, tactic `3`, non-SwiGLU activations, non-NVFP4 quantization, and any prequant-scale path. All of them must take the pre-existing unfused single-tile path and keep current outputs stable. The new guard harness must assert this explicitly instead of inferring it from timing.
  **Must NOT do**: Do not silently switch layouts based on heuristics; do not enable fusion without explicit opt-in; do not let unsupported cases write packed FP4 into `fc1_result_`.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: the logic is bounded once Tasks 1-4 define the supported path.
  - Skills: `[]` - repo-local benchmark and guard logic only.
  - Omitted: `[]` - no extra skill load is required.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 8 | Blocked By: 1, 2

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:91` - unfused escape hatch already present.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:519` - exact grouped-GEMM tactic list for SM120 single-tile.
  - Pattern: `added_benchmark/ncu_tile_roofline.py:10` - tactic id mapping used by single-tile profiling.
  - Pattern: `added_benchmark/test_m64_tile.py:212` - existing single-tile tactic-comparison benchmark to keep in mind when asserting fallback behavior.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python3 added_benchmark/test_single_tile_swiglu_guard.py | tee .sisyphus/evidence/task-6-guard.log` exits `0` and prints `PASS: unsupported single-tile cases stayed unfused`.
  - [ ] `python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --print-gate-state | tee .sisyphus/evidence/task-6-no-opt-in.log` exits `0` and prints `FUSION_GATE=0` when `ENABLE_SINGLE_TILE_SWIGLU_FUSION` is absent.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: No opt-in means no fusion
    Tool: Bash
    Steps: Run `python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --print-gate-state | tee .sisyphus/evidence/task-6-no-opt-in.log`
    Expected: Exit code 0; log contains `FUSION_GATE=0`
    Evidence: .sisyphus/evidence/task-6-no-opt-in.log

  Scenario: Unsupported tactics stay on unfused path
    Tool: Bash
    Steps: Run `python3 added_benchmark/test_single_tile_swiglu_guard.py | tee .sisyphus/evidence/task-6-guard.log`
    Expected: Exit code 0; log contains `PASS: unsupported single-tile cases stayed unfused`
    Evidence: .sisyphus/evidence/task-6-guard.log
  ```

  **Commit**: YES | Message: `test(moe): lock single-tile swiglu fallback matrix` | Files: `added_benchmark/test_single_tile_swiglu_guard.py`, `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`

- [ ] 7. Capture fused vs unfused single-tile profiling evidence with `ncu` and `nsys`

  **What to do**: Reuse `added_benchmark/ncu_tile_roofline.py:1` with tactic `0` for the supported path and collect both fused and unfused profiles. Add a small parser script `added_benchmark/compare_single_tile_swiglu_profile.py` that consumes `nsys stats --report gpukernsum --format csv` outputs for fused and unfused runs and prints a binary pass/fail summary. The pass condition is exact: fused run has no standalone gated-activation kernel, unfused run still has one, and both runs contain GEMM1/GEMM2 kernels.
  **Must NOT do**: Do not use dual-tile profilers; do not rely on manual Nsight UI inspection; do not compare different tactics or different routing inputs between fused and unfused runs.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: this task produces the performance evidence the user explicitly requested.
  - Skills: `[]` - repo-local profiling harness and parser work only.
  - Omitted: `[]` - no extra skill load is required.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 8 | Blocked By: 1, 4, 5

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `added_benchmark/ncu_tile_roofline.py:1` - existing single-tile profiler workload and tactic mapping.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:2286` - standalone activation entry point whose kernel must disappear from the fused profile.
  - Pattern: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu:4153` - single-tile `gemm1` call site being profiled.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 ncu --set full --kernel-name-base demangled --kernel-name regex:'GemmUniversal' --launch-skip 10 --launch-count 5 -o .sisyphus/evidence/task-7-single-tile-fused python3 added_benchmark/ncu_tile_roofline.py 0` exits `0`.
  - [ ] `FORCE_UNFUSED_SWIGLU=1 ncu --set full --kernel-name-base demangled --kernel-name regex:'GemmUniversal' --launch-skip 10 --launch-count 5 -o .sisyphus/evidence/task-7-single-tile-unfused python3 added_benchmark/ncu_tile_roofline.py 0` exits `0`.
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 nsys profile --trace=cuda,nvtx --output=.sisyphus/evidence/task-7-single-tile-fused python3 added_benchmark/ncu_tile_roofline.py 0` exits `0`.
  - [ ] `FORCE_UNFUSED_SWIGLU=1 nsys profile --trace=cuda,nvtx --output=.sisyphus/evidence/task-7-single-tile-unfused python3 added_benchmark/ncu_tile_roofline.py 0` exits `0`.
  - [ ] `nsys stats --report gpukernsum --format csv .sisyphus/evidence/task-7-single-tile-fused.nsys-rep > .sisyphus/evidence/task-7-single-tile-fused.csv` and the unfused equivalent both exit `0`.
  - [ ] `python3 added_benchmark/compare_single_tile_swiglu_profile.py --fused .sisyphus/evidence/task-7-single-tile-fused.csv --unfused .sisyphus/evidence/task-7-single-tile-unfused.csv | tee .sisyphus/evidence/task-7-profile-compare.log` exits `0` and prints `PASS: fused run removed standalone gated activation kernel`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Fused single-tile profiling completes
    Tool: Bash
    Steps: Run the fused `ncu` and `nsys` commands from Acceptance Criteria and save the resulting `.ncu-rep` and `.nsys-rep` files
    Expected: Every command exits 0 and creates its report under `.sisyphus/evidence/`
    Evidence: .sisyphus/evidence/task-7-single-tile-fused.ncu-rep

  Scenario: Profile comparison proves activation-kernel removal
    Tool: Bash
    Steps: Export `gpukernsum` CSV for fused and unfused runs, then run `python3 added_benchmark/compare_single_tile_swiglu_profile.py --fused .sisyphus/evidence/task-7-single-tile-fused.csv --unfused .sisyphus/evidence/task-7-single-tile-unfused.csv | tee .sisyphus/evidence/task-7-profile-compare.log`
    Expected: Exit code 0; log contains `PASS: fused run removed standalone gated activation kernel`
    Evidence: .sisyphus/evidence/task-7-profile-compare.log
  ```

  **Commit**: YES | Message: `perf(moe): add single-tile swiglu profiling evidence` | Files: `added_benchmark/compare_single_tile_swiglu_profile.py`

- [ ] 8. Run the full single-tile validation matrix and fix regressions without widening scope

  **What to do**: Rebuild with `-j2`, run the deterministic fused harness, the fallback harness, the residue-size checks, and the fused/unfused profiler sweep. Fix only regressions that block the supported single-tile SM120 NVFP4 path. Do not expand scope to dual-tile, M32, M64, M256, backend capability tables, or global weight loaders.
  **Must NOT do**: Do not add new features after the matrix is green; do not rewrite tests to hide instability; do not merge profiling-only noise into kernel commits.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: this is the final integration and regression pass.
  - Skills: `[]` - repo-local build/test/profiling only.
  - Omitted: `[]` - no extra skill load is required.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: Final Verification Wave | Blocked By: 1, 2, 3, 4, 5, 6, 7

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `scripts/build_wheel.py:1023` - validated `-j/--job_count` build flag.
  - Pattern: `added_benchmark/test_single_tile_swiglu_fusion.py` - fused/unfused correctness harness added in Task 1.
  - Pattern: `added_benchmark/test_single_tile_swiglu_guard.py` - unsupported-case fallback harness added in Task 1.
  - Pattern: `added_benchmark/ncu_tile_roofline.py:59` - warmup and profiled iteration structure for single-tile runs.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python3 scripts/build_wheel.py -a "120-real" -j 2 -l -s --build_dir cpp/build_sm120_swiglu | tee .sisyphus/evidence/task-8-build.log` exits `0`.
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 | tee .sisyphus/evidence/task-8-fused.log` exits `0`.
  - [ ] `python3 added_benchmark/test_single_tile_swiglu_guard.py | tee .sisyphus/evidence/task-8-guard.log` exits `0`.
  - [ ] `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1 python3 added_benchmark/test_single_tile_swiglu_fusion.py --tactic 0 --num-tokens 65 | tee .sisyphus/evidence/task-8-residue.log` exits `0`.
  - [ ] `python3 added_benchmark/compare_single_tile_swiglu_profile.py --fused .sisyphus/evidence/task-7-single-tile-fused.csv --unfused .sisyphus/evidence/task-7-single-tile-unfused.csv | tee .sisyphus/evidence/task-8-summary.log` exits `0`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Full supported-path matrix passes
    Tool: Bash
    Steps: Run the Task 8 build, fused harness, guard harness, and residue command in order, saving all logs under `.sisyphus/evidence/`
    Expected: Every command exits 0
    Evidence: .sisyphus/evidence/task-8-summary.log

  Scenario: Regression cleanup stays in scope
    Tool: Bash
    Steps: Re-run `python3 added_benchmark/test_single_tile_swiglu_guard.py | tee .sisyphus/evidence/task-8-scope-guard.log` after the last fix
    Expected: Exit code 0; log still contains `PASS: unsupported single-tile cases stayed unfused`
    Evidence: .sisyphus/evidence/task-8-scope-guard.log
  ```

  **Commit**: YES | Message: `chore(moe): verify single-tile sm120 swiglu fusion` | Files: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`, `cpp/tensorrt_llm/cutlass_extensions/include/cutlass_extensions/epilogue/fusion/sm90_visitor_swiglu_store.hpp`, `added_benchmark/test_single_tile_swiglu_fusion.py`, `added_benchmark/test_single_tile_swiglu_guard.py`, `added_benchmark/compare_single_tile_swiglu_profile.py`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [ ] F1. Plan Compliance Audit - oracle
- [ ] F2. Code Quality Review - unspecified-high
- [ ] F3. Real Runtime QA - unspecified-high
- [ ] F4. Scope Fidelity Check - deep

## Commit Strategy
- Commit 1: benchmark harnesses and unsupported-case guard scripts.
- Commit 2: single-tile fusion gate, TMA epilogue plumbing, and activation bypass.
- Commit 3: packed-FP4 `end_loop()`/store stabilization and profiling automation.
- Never mix dual-tile or backend-capability-table work into these commits.

## Success Criteria
- Single-tile CTA_M=128 SM120 NVFP4 SwiGLU runs through fused `gemm1` without a standalone activation kernel when `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1` is set.
- Fused vs unfused numerical comparison passes on deterministic synthetic data.
- Unsupported tactics and conditions stay on the unfused path without crash or silent miscompute.
- `ncu` and `nsys` evidence both show the intended kernel-shape change on the fused path.
