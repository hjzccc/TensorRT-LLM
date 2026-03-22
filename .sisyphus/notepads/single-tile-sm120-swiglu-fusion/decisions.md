## Task 2 - Single-Tile Fail-Closed Gate

- Added `getForceUnfusedSwiglu()` and `getEnableSingleTileSwigluFusion()` in `moe_kernels.cu` so the single-tile path is opt-in and can still be forced back to unfused.
- `runMoe()` now computes a local `enable_swiglu_fusion` gate that requires all of: `ENABLE_SINGLE_TILE_SWIGLU_FUSION=1`, gated `ActivationType::Swiglu`, NVFP4 block scaling, `!use_deepseek_fp8_block_scale`, `!usePrequantScaleKernel(quant_params)`, `!getForceUnfusedSwiglu()`, TMA warp-specialized GEMM1, and `gemm1_config_->tile_config_sm120 == CutlassTileConfigSM120::CtaShape128x128x128B`.
- Workspace sizing now pre-allocates `fused_swiglu_output`, `fp4_act_scale_fc1`, and `fp4_act_scale_fc2` on the opt-in NVFP4 SwiGLU path because plugin workspace setup runs before `setTactic()` and cannot know the final GEMM1 tactic yet.
- `runMoe()` only rebinds `fc1_result_`, `fc1_fp4_act_scale_`, and `fc2_fp4_act_scale_` to those dedicated buffers when the exact local gate is true; all other single-tile and dual-tile cases keep the existing overlapped aliases.
