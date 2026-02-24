# NVFP4 MoE Pipeline Code Trace — Qwen3-30B-A3B-NVFP4 on SM120

> All line numbers reference the **`dual_tile_strategy`** branch of this worktree.
> Model: Qwen3-30B-A3B-NVFP4 · 128 experts · top_k=8 · hidden=2048 · inter=768 · 48 layers
> GPU: NVIDIA GeForce RTX 5090 (SM 12.0 / Blackwell consumer)

---

## Pipeline Overview

```
Input BF16 [batch, 2048]
    ↓  Step 0: Route tokens to experts
    ↓  Step 1: Permute rows + quantize BF16 → FP4
FP4 [expanded, 2048] + fc1_fp4_act_scale_
    ↓  Step 2a: GEMM1 (FP4 × FP4 weights → BF16)
BF16 [expanded, 768*2]
    ↓  Step 2b: SwiGLU activation + quantize BF16 → FP4
FP4 [expanded, 768] + fc2_fp4_act_scale_
    ↓  Step 3: GEMM2 (FP4 × FP4 weights → BF16) + finalize (unpermute + weighted reduce)
Output BF16 [batch, 2048]
```

---

## Master File

Almost the entire pipeline lives in one file:

```
cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu
```

The orchestrator is `CutlassMoeFCRunner::runMoe()` starting at **line 3829**.
For the non-min-latency NVFP4 path (default), execution enters the `else` branch at **line 4030**.

---

## Step 0 — Route Tokens to Experts

| | |
|---|---|
| **Call site** | `moe_kernels.cu` line **4036** |
| **Function** | `fusedBuildExpertMapsSortFirstToken()` |
| **Fallback** | `threeStepBuildExpertMapsSortFirstToken()` at line **4044** |
| **Input** | `token_selected_experts` (gate routing decisions) |
| **Output** | `permuted_row_to_unpermuted_row_`, `expert_first_token_offset_` |

Builds the mapping of which tokens go to which experts, sorted by expert ID.
`expert_first_token_offset_[i]` gives the index of the first token for expert `i` in the permuted layout.

---

## Step 1 — Permute + Quantize Input Activations (BF16 → FP4)

| | |
|---|---|
| **Call site** | `moe_kernels.cu` line **4072** |
| **Launcher** | `expandInputRowsKernelLauncher()` at line **1803** |
| **Kernel selection** | line **1851–1857** → selects `expandInputRowsKernel<..., NVFP4, false>` |
| **Kernel** | `expandInputRowsKernel()` at line **1595–1724** |
| **Quantization call** | line **1688** → `quantizePackedFPXValue()` at line **994** |
| **Core quant function** | → `cvt_warp_fp16_to_fp4()` in **`quantization.cuh`** line **427** |
| **PTX instruction** | `cvt.rn.satfinite.e2m1x2.f32` in **`quantization.cuh`** line **353** |

**What it does**: Fuses two operations into a single kernel:
1. **Permute**: reorders token rows from original order into per-expert-sorted order
2. **Quantize**: converts each BF16 element to FP4 (E2M1) with per-16-element block scale factors

**I/O**:
- Input: BF16 `input_activations` `[batch, 2048]`
- Output: FP4 `permuted_data_` `[expanded, 2048]` + scale factors `fc1_fp4_act_scale_`
- `expanded` = `batch × top_k` = `batch × 8`

### Quantization Algorithm (shared by all FP4 quantization sites)

Located in **`quantization.cuh`** lines 427–510 (`cvt_warp_fp16_to_fp4`):

```
For every block of 16 BF16 elements:
1. Each thread loads 8 elements (128 bits)
2. Compute local absmax across 8 values                    [line 431-438]
3. Warp shuffle reduction across 2 threads → 16-elem max   [line 442-448]
4. SF = global_scale × (absmax / 6.0)                      [line 468]
     6.0 = max representable value in E2M1 format
5. Narrow SF to FP8 E4M3: sf_fp8 = (__nv_fp8_e4m3) SF     [line 470]
6. Compute inverse scale: 1 / (float(sf_fp8) / global_scale)  [line 475]
7. Scale each element: elem × inv_scale                     [line 498-499]
8. Convert FP32 → E2M1 via PTX: cvt.rn.satfinite.e2m1x2.f32  [line 353]
9. Pack 8 × E2M1 nibbles → one uint32_t                    [line 503]
10. Write sf_fp8 to swizzled scale factor buffer            [line 478-481]
```

### Scale Factor Layout

Scale factors use a **swizzled 128×4 layout** for efficient CUTLASS TMA access.
- Layout function: `get_sf_out_offset_128x4()` in **`quantization.cuh`** line **673–711**
- Offset computation: `cvt_quant_get_sf_out_offset()` at line **713–756**
- SF element type: `uint8_t` (FP8 E4M3)

SF layout dimensions:
```
[numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4 (kTile)]
```
where `mTile = 128`, `kTile = 4 × VecSize`.

---


## Input Scale Factor (ISF) Origin — Where `input_sf` Comes From

The `input_sf` passed to the C++ `runMoe()` kernel is **NOT computed inside the MoE kernel**.
It arrives pre-computed from the Python side. There are two paths that produce it, depending
on whether the upstream RMSNorm fuses quantization.

### Which path does Qwen3-30B-A3B-NVFP4 use?

**Path B (separate `fp4_quantize`)**. The fused RMSNorm+FP4 path (Path A) is only wired up
for Llama-based dense models — `modeling_llama.py` line 813 sets `nvfp4_scale` on the layernorm.
Qwen3 MoE (`modeling_qwen3_moe.py`) does **not** attach `nvfp4_scale` to any layernorm,
so `RMSNorm.forward()` takes the non-fused path and returns plain BF16.

### Path A: Fused RMSNorm + FP4 Quantization (Llama-only, documented for completeness)

| | |
|---|---|
| **Where** | `rms_norm.py` line **119** |
| **Op** | `torch.ops.trtllm.fused_add_rms_norm_quant()` |
| **Input** | BF16 hidden states + BF16 residual + `nvfp4_scale` (= `fc31_input_scale`) |
| **Output** | `Fp4QuantizedTensor(normed_fp4_u8, sf_fused)` — FP4 data + scale factors bundled together |

How `nvfp4_scale` gets attached (Llama example):
```python
# modeling_llama.py line 813
self.post_attention_layernorm.nvfp4_scale = self.mlp.gate_up_proj.input_scale
```
The fused kernel performs RMSNorm + residual add + FP4 quantization in one launch.
The MoE layer would unpack via `x, x_sf = x.fp4_tensor, x.scaling_factor` (line 488).

### Path B: Separate `fp4_quantize` Call (Qwen3 MoE — the active path)

| | |
|---|---|
| **Call site** | `fused_moe_cutlass.py` line **492** (non-comm path) or line **500** (comm path) |
| **Function** | `quantize_input()` at line **440** |
| **Op** | `torch.ops.trtllm.fp4_quantize(x, self.fc31_input_scale, self.scaling_vector_size, False, ...)` |
| **C++ binding** | `fp4Quantize.cpp` line **43** → `invokeFP4Quantization()` |
| **Kernel** | `quantize_with_block_size()` in `quantization.cuh` line **758** |

The `quantize_input()` method at line 440 of `fused_moe_cutlass.py` handles both paths:
```python
# Path A: input arrives as Fp4QuantizedTensor (already quantized by fused RMSNorm)
if isinstance(x, Fp4QuantizedTensor):
    x, x_sf = x.fp4_tensor, x.scaling_factor  # just unpack

# Path B: input arrives as plain BF16 (Qwen3 MoE case)
else:
    x, x_sf = torch.ops.trtllm.fp4_quantize(
        x, self.fc31_input_scale, self.scaling_vector_size, False, True)
```

### `fc31_input_scale` — The Global Scale (a single scalar float)

| | |
|---|---|
| **Type** | Single `float32` scalar — NOT per-token, NOT per-expert |
| **Loaded at** | Model load time, once per layer |
| **Defined in** | `quantization.py` line **2086–2088** |

**How it's computed:**

```python
# quantization.py line 2086-2088
# fc31_input_scale is the reciprocal of the maximum of all w1 input scales and w3 input scales.
module.fc31_input_scale.data.copy_(tmp_fc31_input_scale.max().reciprocal())
```

Where `tmp_fc31_input_scale` is populated per-expert during weight loading (line 2082):
```python
# quantization.py line 1943-1949
def load_expert_fc31_input_scale_nvfp4(self, w1_input_scale, w3_input_scale, dst_fc31_input_scale):
    w1_input_scale = w1_input_scale[...].reshape([])
    w3_input_scale = w3_input_scale[...].reshape([])
    assert torch.allclose(w1_input_scale, w3_input_scale)  # must be equal
    dst_fc31_input_scale.copy_(w1_input_scale)
```

So the full derivation is:
```
For each expert e:
    tmp_fc31_input_scale[e] = w1_input_scale[e]   (from checkpoint, calibration data)
                            = w3_input_scale[e]   (asserted equal)

fc31_input_scale = 1.0 / max(tmp_fc31_input_scale[0..127])
```

This scalar is then used as the `global_scale` parameter in the FP4 quantization algorithm
(see Step 1 above, line 4 of the algorithm: `SF = global_scale × (absmax / 6.0)`).

### `fc2_input_scale` — The GEMM2 Global Scale (analogous)

| | |
|---|---|
| **Defined in** | `quantization.py` line **2090–2091** |
| **Formula** | `fc2_input_scale = 1.0 / max(all_expert_w2_input_scales)` |

Used by `doActivationKernel` (Step 2b) as the `global_scale` when quantizing the SwiGLU output
for GEMM2 input. Same algorithm, different scalar.

### Summary: ISF Flow for Qwen3 MoE

```
Model Load (once per layer):
    checkpoint w1_input_scale[128 experts]
        → max() → reciprocal()
        → fc31_input_scale (single scalar)

Runtime (every forward pass):
    BF16 hidden_states
        → RMSNorm (returns plain BF16, no fused FP4)
        → quantize_input() in fused_moe_cutlass.py
            → torch.ops.trtllm.fp4_quantize(x, fc31_input_scale, ...)
            → invokeFP4Quantization() in fp4Quantize.cpp
            → quantize_with_block_size() kernel in quantization.cuh
            → returns (fp4_data, scale_factors)
        → passed as (input_activations, input_sf) to C++ runMoe()
        → expandInputRowsKernel just COPIES the pre-computed SFs (line 1702-1704)
```

---

## Step 2a — GEMM1: FP4 Activations × FP4 Weights → BF16

| | |
|---|---|
| **Call site** | `moe_kernels.cu` line **4117** |
| **Function** | `CutlassMoeFCRunner::gemm1()` at line **3300** |
| **GEMM launch** | line **3370** → `gemm_runner.moeGemm(universal_input, tma_ws_input)` |
| **Kernel type** | CUTLASS TMA warp-specialized grouped GEMM with `OpClassBlockScaledTensorOp` |

**I/O**:
- Input A: FP4 `permuted_data_` `[expanded, 2048]` + `fc1_fp4_act_scale_`
- Input B: FP4 `fc1_expert_weights` `[128 experts, 768*2, 2048]` + weight scale factors
- Output: BF16 `glu_inter_result_` `[expanded, 768*2]`

The CUTLASS kernel handles FP4 block-scaled inputs natively via SM120 hardware.
The `×2` in the output dimension is because SwiGLU is a gated activation: the GEMM produces both the gate path and the linear path in a single output.

---

## Step 2b — SwiGLU Activation + Quantize (BF16 → FP4)

| | |
|---|---|
| **Call site** | `moe_kernels.cu` line **3399** (inside `gemm1()`) |
| **Launcher** | `doActivation<T, UnfusedGemmOutputType>(...)` → `doActivation()` at line **2540** |
| **Kernel** | `doActivationKernel()` at line **2275–2520** |
| **SwiGLU** | line **2408–2425** — `fn(fc1_value, linear_value)` |
| **Scale** | line **2427** — `post_act_val = gate_act * quant_scale` |
| **Quantize** | line **2439** → `quantizePackedFPXValue()` at line **994** → `cvt_warp_fp16_to_fp4()` |

**What it does**: Fuses three operations into one kernel:
1. **SwiGLU activation**: `output = silu(gated_half) × linear_half`
2. **Per-expert quantization scaling**: multiply by `fc2_fp8_quant[expert]`
3. **FP4 quantization**: block-wise scale + E2M1 conversion (same algorithm as Step 1)

**I/O**:
- Input: BF16 `glu_inter_result_` `[expanded, 768*2]`
- Output: FP4 `fc1_result_` `[expanded, 768]` + scale factors `fc2_fp4_act_scale_`

The kernel also pads scale factors for CUTLASS alignment:
- K-dimension padding: line **2447–2454**
- N-dimension (token) padding: line **2467–2517**

---

## Step 3 — GEMM2: FP4 Activations × FP4 Weights → BF16 + Finalize

| | |
|---|---|
| **Call site** | `moe_kernels.cu` line **4143** |
| **Function** | `CutlassMoeFCRunner::gemm2()` at line **3507** |
| **Output memset** | line **3556–3557** — `cudaMemsetAsync(final_output, 0x0, ...)` |
| **GEMM launch** | line **3586** — `gemm_runner.moeGemmBiasAct(universal_input, tma_ws_input)` |
| **Finalize (fallback)** | line **3609** — `finalizeMoeRoutingKernelLauncher()` at line **2103** |

**I/O**:
- Input A: FP4 `fc1_result_` `[expanded, 768]` + `fc2_fp4_act_scale_`
- Input B: FP4 `fc2_expert_weights` `[128 experts, 2048, 768]` + weight scale factors
- Output: BF16 `final_output` `[batch, 2048]`

### FINALIZE Epilogue (fused unpermute + reduce)

When `gemm2_config.epilogue_fusion_type == FINALIZE`:
- The CUTLASS epilogue **atomically accumulates** results from all top-k experts directly into `final_output`
- Each expert's contribution is weighted by its router score (`token_topk_unpermuted_scales`)
- This fuses unpermute + weighted reduce into the GEMM epilogue — **no separate finalize kernel needed**
- Requires output buffer to be zero-initialized first (line 3557)

### Fallback: Separate Finalize Kernel

When NOT using fused FINALIZE (line 3607–3621):
- `finalizeMoeRoutingKernelLauncher()` at line **2103**
- `finalizeMoeRoutingKernel()` — unpermutes rows, scales by router weights, reduces across top-k experts

---

## FP4 Quantization Sites Summary

There are **4 distinct sites** in the codebase that quantize activations to FP4:

### Site 1: expandInputRowsKernel (MoE GEMM1 input)
- **File**: `moe_kernels.cu` line **1595–1724**
- **When**: Before GEMM1 — fused with token permutation
- **Call chain**: line 1688 → `quantizePackedFPXValue()` (line 994) → `cvt_warp_fp16_to_fp4()` (`quantization.cuh:427`)

### Site 2: doActivationKernel (MoE GEMM2 input)
- **File**: `moe_kernels.cu` line **2275–2520**
- **When**: Between GEMM1 and GEMM2 — fused with SwiGLU activation
- **Call chain**: line 2439 → `quantizePackedFPXValue()` (line 994) → `cvt_warp_fp16_to_fp4()` (`quantization.cuh:427`)

### Site 3: quantize_with_block_size (Standalone kernel)
- **File**: **`quantization.cuh`** line **758–902**
- **When**: Non-MoE quantization, Python-callable `fp4_quantize()` op
- **Entry points**:
  - `invokeFP4Quantization()` in `quantization.cu` line **135**
  - `fp4_quantize()` PyTorch op in `thop/fp4Quantize.cpp` line **43**
  - Fused layernorm path in `fusedLayernormKernels/fp4_converter.cuh`

### Site 4: quantize_nvfp4_sharedmem (Fused MoE AllReduce)
- **File**: **`fusedMoeCommKernels.cu`** line **41–199**
- **When**: Inside fused MoE AllReduce communication kernel (multi-GPU)
- **Different layout**: outputs packed `[e2m1 values | scales | global_scale]` in shared memory

---

## Key Constants

Defined in `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h`:

| Constant | Value | Meaning |
|---|---|---|
| `NVFP4BlockScaleVectorSize` | 16 | Elements per scale factor block |
| `MinKDimAlignmentNVFP4` | 256 | K-dimension padding for scale factors |
| `MinNDimAlignmentNVFP4` | 128 | Token/N-dimension padding for scale factors |
| `ElementSF` | `uint8_t` | Scale factor data type (FP8 E4M3) |

---

## Buffer Aliasing

Important for understanding memory layout:

```
permuted_data_   = fc1_result_       (SAME MEMORY — aliased)
glu_inter_result_ = fc2_result_      (SAME MEMORY — aliased)
fc1_fp4_act_scale_ = fc2_fp4_act_scale_  (SAME MEMORY — aliased, reused between GEMM1/GEMM2)
```

- GEMM1 writes to `glu_inter_result_` (intermediate), NOT to `permuted_data_`
- `doActivation` reads from `glu_inter_result_` and writes FP4 output to `fc1_result_` (= `permuted_data_`), which is safe because GEMM1 is done
- GEMM2 reads from `fc1_result_` and writes to `fc2_result_` (= `glu_inter_result_`), which is safe because activation is done

---

## Scale Factor Functions

All in `moe_kernels.cu`:

| Function | Line | Purpose |
|---|---|---|
| `getOffsetWeightSF()` | 935 | Compute offset into weight scale factor buffer |
| `getOffsetActivationSF()` | 962 | Compute offset into activation scale factor buffer |
| `quantizePackedFPXValue()` | 994 | Quantize 8 elements to FP4, write scale factor |
| `writeSF()` | 1043 | Copy/write scale factors to swizzled layout |
| `setupFP4BlockScalingFactors()` | 1123 | Set up CUTLASS block-scaling strides/pointers per expert |

In `quantization.cuh`:

| Function | Line | Purpose |
|---|---|---|
| `cvt_warp_fp16_to_fp4()` | 427 | Core: BF16/FP16 → FP4 E2M1 with block scale |
| `cvt_warp_fp8_to_fp4()` | 512 | Core: FP8 → FP4 E2M1 with block scale |
| `cvt_warp_fp16_to_mxfp8()` | 603 | Core: BF16/FP16 → MXFP8 with block scale |
| `fp32_vec_to_e2m1()` | 321 | PTX wrapper: 16×FP32 → 8-byte packed E2M1 |
| `get_sf_out_offset_128x4()` | 673 | Swizzled SF layout offset calculation |
| `cvt_quant_get_sf_out_offset()` | 713 | Get SF write pointer for a given row/col |
| `quantize_with_block_size()` | 758 | Standalone quantization kernel (non-MoE) |

---

## File Index

| File | What's in it |
|---|---|
| `cpp/.../moe_gemm/moe_kernels.cu` | Everything: orchestrator, GEMM1, GEMM2, activation, permutation, finalize, scale factor setup |
| `cpp/.../kernels/quantization.cuh` | Core FP4/MXFP8 quantization device functions + standalone kernel |
| `cpp/.../kernels/quantization.cu` | Launcher for standalone quantization kernel |
| `cpp/.../kernels/quantization.h` | Public API: `invokeFP4Quantization()`, `computePerTokenGlobalScaleForFP4Quantization()` |
| `cpp/.../kernels/fusedMoeCommKernels.cu` | Shared-memory FP4 quant/dequant for fused MoE AllReduce |
| `cpp/.../include/moe_kernels.h` | Runner class definition, `gemm1()`/`gemm2()` signatures, buffer members |
| `cpp/.../include/moe_gemm_kernels.h` | `NVFP4BlockScaledConfig`, alignment constants, `MoeGemmRunner` |
| `cpp/.../fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h` | SM120 CUTLASS GEMM kernel template (FP4×FP4) |
| `cpp/.../moe_gemm/moe_gemm_template_dispatch_tma_ws.h` | TMA warp-specialized GEMM dispatch for MoE |
| `cpp/.../moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl` | CUTLASS GEMM launcher with epilogue configuration |
| `cpp/.../thop/fp4Quantize.cpp` | PyTorch binding: `fp4_quantize()` op |
| `cpp/.../thop/moeOp.cpp` | PyTorch binding: `runMoe()` / `runMoeDualTile()` entry points |
| `cpp/.../fusedLayernormKernels/fp4_converter.cuh` | FP4 converter for fused layernorm+quantization |
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py` | `quantize_input()` (line 440): Path A/B dispatch for FP4 quantization, `_compute_moe_result()` |
| `tensorrt_llm/_torch/modules/fused_moe/quantization.py` | Weight loading: `fc31_input_scale` computation (line 2086), `load_expert_fc31_input_scale_nvfp4()` (line 1943) |
| `tensorrt_llm/_torch/modules/rms_norm.py` | Fused RMSNorm+FP4 path (line 119): `fused_add_rms_norm_quant()` → `Fp4QuantizedTensor` |
| `tensorrt_llm/_torch/models/modeling_llama.py` | Llama-only: attaches `nvfp4_scale` to layernorm (line 813) for fused FP4 path |

---

## Autotuner + Piecewise CUDA Graph Interaction

Three subsystems collaborate to select and freeze GEMM tactics for MoE:

1. **Autotuner** — profiles tactics at power-of-2 token buckets during warmup
2. **Piecewise optimizer** — splits torch.compile FX graph at attention ops, each piece becomes a `PiecewiseRunner`
3. **CUDA graph capture** — for each token bucket, captures kernel launches into a replayable graph

### Autotuner: Profiling Phase (Warmup)

Every `fused_moe` call invokes the autotuner twice — once for GEMM1, once for GEMM2:

```python
# torch_custom_ops.py lines 232-252
_, gemm_tactic_1 = tuner.choose_one("trtllm::fused_moe::gemm1", [moe_runner], ...)
_, gemm_tactic_2 = tuner.choose_one("trtllm::fused_moe::gemm2", [moe_runner], ...)
```

On first invocation (cache miss, `is_tuning_mode=True`):
- `_optimization_profiles()` generates power-of-2 buckets: `[1, 2, 4, 8, ..., 8192]`
- For each bucket, iterates all tactics (CUTLASS kernel configurations)
- Times each via `run_gemm_profile()` → selects fastest → stores in `profiling_cache`
- Cache key = `(custom_op, "MoERunner", str(unique_id()), bucketed_shape)`
  - **No layer index** — all 48 MoE layers share the same profiling result
  - `unique_id()` contains: dtypes, top_k, tp_size, ep_size, etc.

On subsequent calls (cache hit):
- `search_cache()` at `autotuner.py` line **400** → dict lookup → returns `(runner, tactic)` immediately
- Runtime token count mapped to bucket via `last_positive_power_of_2` (e.g., 157 → 128, 300 → 256)

| | |
|---|---|
| **Cache key function** | `get_cache_key()` at `autotuner.py` line **432** |
| **Bucket mapping** | `last_positive_power_of_2` via `DynamicTensorSpec` at `torch_custom_ops.py` line **42** |
| **Bucket generator** | `get_last_power_of_2_num_tokens_buckets()` at `utils.py` line **277** |
| **Max bucket** | `tune_max_num_tokens=8192` (line 44) |
| **Shared runner** | `MoERunner.runner_dict` (class-level, line 39) — one C++ `FusedMoeRunner` per dtype combo |

### Piecewise CUDA Graph: Capture Phase

`PiecewiseRunner.__call__` at `piecewise_optimizer.py` line **163** manages per-bucket graph capture:

```
For each token bucket (128, 256, 512, ...):
  1. Executor pads tokens UP to nearest bucket: bisect_left at model_engine.py:1448
  2. PiecewiseRunner.__call__ checks: runtime_num_of_token in self.entries?
  3. First 3 calls: warmup (line 195-197) — runs eagerly, no capture
  4. 4th call: CUDA graph capture (lines 203-214):

     graph = torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph, pool=self.graph_pool_handle):
         output = entry.callable(*args)    # ← records all CUDA kernel launches
```

The FX graph is split at attention ops (which need dynamic seq len metadata):
- GEMM/MoE/norm pieces → captured into CUDA graphs
- Attention pieces → always run eagerly (excluded from graph)

| | |
|---|---|
| **Piecewise interpreter** | `PiecewiseInterpreter` at `piecewise_optimizer.py` line **22** |
| **Graph split** | `split_module()` at `piecewise_optimizer.py` — splits FX graph at `exclude_modules` |
| **Bucket padding** | `get_padded_piecewise_tokens()` with `bisect_left` at `model_engine.py` line **1448** |
| **Graph pool** | Shared across all `PiecewiseRunner` instances for memory reuse |
| **Warmup count** | 3 eager runs before capture (line 195) |

### How Tactic Gets Frozen Into CUDA Graph

During CUDA graph capture, `entry.callable(*args)` executes the compiled subgraph.
When execution hits `fused_moe`, here is the exact call chain:

```
entry.callable(*args)                              [inside torch.cuda.graph() context]
  → torch.ops.trtllm.fused_moe(...)                [torch_custom_ops.py:158]
    → tuner.choose_one("gemm1", ...)               [line 232, HOST-SIDE Python]
      → search_cache() → CACHE HIT                 [autotuner.py:912]
      → returns (runner, tactic=2)                  [already profiled in warmup]
    → tuner.choose_one("gemm2", ...)               [line 243, HOST-SIDE Python]
      → search_cache() → CACHE HIT
      → returns (runner, tactic=5)
    → run_moe(input, ..., [tactic_1=2, tactic_2=5], ...)  [line 256]
      → C++ FusedMoeRunner::runMoe()                [moeOp.cpp]
        → launches CUTLASS grouped GEMM kernels     [GPU-SIDE, RECORDED INTO GRAPH]
```

**Critical distinction**:
- `choose_one()` = Python dict lookup = **host-side code** → NOT recorded into CUDA graph
- `run_moe()` → C++ → CUDA kernel launches = **GPU-side** → RECORDED into CUDA graph
- The specific CUTLASS kernels (grid dims, block dims, shared mem config) for tactic 2/5
  are what get recorded. The tactic number itself is gone — only its concrete kernel launches remain.

### Inference: Graph Replay

After capture, every forward pass with the same token bucket:

```python
entry.cuda_graph.replay()   # piecewise_optimizer.py line 237
```

This replays the exact CUDA kernel launches recorded during capture.
**No Python runs**: no `choose_one`, no `search_cache`, no `run_moe`.
Just raw CUDA driver replaying the graph — maximum throughput.

### End-to-End Timeline

```
Phase 1 — Warmup (autotuner profiling):
  Layer 0 MoE call #1: cache_miss → profile all tactics for all buckets
    → best: gemm1_tactic=2, gemm2_tactic=5 for each bucket
  Layer 1-47 MoE call #2-#480: cache_hit → returns tactic instantly (dict lookup)

Phase 2 — Warmup (piecewise eager, 3 passes):
  PiecewiseRunner.__call__ → entry.warmup_count < 3 → runs eagerly
  fused_moe → choose_one → cache hit → run_moe with cached tactics

Phase 3 — CUDA Graph Capture (4th pass per bucket):
  PiecewiseRunner.__call__ → entry.cuda_graph is None → capture
  torch.cuda.graph() context → entry.callable(*args)
    → fused_moe → choose_one → cache hit → run_moe
    → CUDA kernels from run_moe get recorded into graph
  entry.cuda_graph = graph  (stored per bucket)

Phase 4 — Inference (all subsequent passes):
  PiecewiseRunner.__call__ → entry.cuda_graph.replay()
  No Python, no autotuner, no tactic selection
  Tactic is implicitly frozen in the recorded kernel launches
```

### Key Implications

1. **Tactic is frozen per-bucket**: Each token bucket gets its own CUDA graph with
   potentially the same tactic (since autotuner profiles by bucketed shape, and all
   buckets for a given model config typically pick the same tactic).

2. **All 48 layers share one profiling result**: Since the cache key has no layer index,
   profiling happens once (layer 0), and layers 1-47 all get cache hits.

3. **Changing tactics requires re-capture**: If you want different tactics, you must
   invalidate both the autotuner cache AND the captured CUDA graphs.

4. **Attention is NOT graphed**: Attention ops run eagerly between graph replays
   because they need dynamic sequence length metadata. Only MoE/GEMM/norm pieces
   are inside CUDA graphs.

---

## Two Template Instantiations of expandInputRowsKernel

The C++ runner is template-instantiated based on the dtype of the input tensor:

| `input.dtype` | `mActivationDtype` | `NeedQuant` | Runner `InputType` | Kernel behavior |
|---|---|---|---|---|
| `Long` (FP4 packed) | Long | **false** | `__nv_fp4_e2m1` | Permute only — copies SFs, no quantization |
| `BFloat16` | BFloat16 | **true** | `__nv_bfloat16` | Permute + Quantize BF16→FP4 inside kernel |

Selection in `moeOp.cpp` lines 170–182:

```cpp
switch (mActivationDtype) {
    case Half:
    case BFloat16:
        mKernelRunner = switch_output_type<fp4, fp4, true>(mOutputDtype);  // NeedQuant=true
        break;
    default:  // Long (pre-quantized FP4)
        mKernelRunner = switch_output_type<fp4, fp4, false>(mOutputDtype); // NeedQuant=false
}
```

**For Qwen3 MoE**: Python calls `fp4_quantize()` first → input arrives as `Long` (FP4)
→ kernel does NOT re-quantize, only permutes + copies SFs.

**Single-GPU overhead implication**: The Python-side `fp4_quantize` adds an extra kernel launch
+ global memory round-trip. The fused permute+quantize path (`NeedQuant=true`) exists in the
kernel but is unused for Qwen3. Eliminating the separate Python `fp4_quantize` and sending
BF16 directly to `expandInputRowsKernel` would save one kernel launch + memory traffic.

---

## Updated File Index

| File | What's in it |
|---|---|
| `tensorrt_llm/_torch/autotuner.py` | `choose_one()` (line 834), `search_cache()` (line 400), `get_cache_key()` (line 432), `_find_nearest_profile()` (line 1312), `_optimization_profiles()` (line 1230) |
| `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` | `fused_moe` custom op (line 158), `MoERunner` class (line 37), `tuning_config` (line 40-46), `unique_id()` (line 107), `runner_dict` (line 39/90-101) |
| `tensorrt_llm/_torch/utils.py` | `get_last_power_of_2_num_tokens_buckets()` (line 277), `piecewise_cuda_graph` flag (line 314-334) |
| `tensorrt_llm/_torch/compilation/piecewise_optimizer.py` | `PiecewiseInterpreter` (line 22), `PiecewiseRunner` (line 125), `__call__` with bucket match (line 163-176), CUDA graph capture (line 190-227) |
| `tensorrt_llm/_torch/compilation/backend.py` | `Backend` class with `enable_piecewise_cuda_graph` |
| `tensorrt_llm/_torch/pyexecutor/cuda_graph_runner.py` | `CUDAGraphRunner` (line 83), batch-level graph keyed by `(batch_size, draft_len, ...)` |
| `tensorrt_llm/_torch/pyexecutor/model_engine.py` | `_get_padding_params()` (line 1430), `get_padded_piecewise_tokens` with `bisect_left` (line 1446-1449), piecewise capture setup (line 260-299) |
