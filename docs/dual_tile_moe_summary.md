# Dual-Tile MoE GEMM Strategy — Complete Technical Guide

> **Audience**: Someone who is NOT familiar with the TensorRT-LLM codebase.
>
> **Validated on**: Qwen3-30B-A3B-NVFP4 (128 experts, top-8, H=2048, I=768) on RTX 5090 (SM 120).

---

## Table of Contents

1. [What This Is](#1-what-this-is)
2. [Background: How Standard MoE Works in TRT-LLM](#2-background-how-standard-moe-works-in-trt-llm)
3. [The Problem: Skewed Expert Routing](#3-the-problem-skewed-expert-routing)
4. [Step 1: Profile Expert Routing (Real Model Inference)](#4-step-1-profile-expert-routing)
5. [Step 2: Configure Dual-Tile in Python](#5-step-2-configure-dual-tile-in-python)
6. [Step 3: Runtime — Python Pipeline Call Chain](#6-step-3-runtime--python-pipeline-call-chain)
7. [Step 4: Runtime — C++ Kernel Internals](#7-step-4-runtime--c-kernel-internals)
8. [Step 5: Testing & Verification](#8-step-5-testing--verification)
9. [Performance Results](#9-performance-results)
10. [NVFP4 Weight Conversion — Critical Details](#10-nvfp4-weight-conversion--critical-details)
11. [Limitations & Caveats](#11-limitations--caveats)
12. [All Modified Files (Complete List)](#12-all-modified-files)
13. [Bugs Fixed During Development](#13-bugs-fixed-during-development)
14. [Build & Run Instructions](#14-build--run-instructions)

---

## 1. What This Is

MoE (Mixture of Experts) layers route each token to a subset of "experts" (small FFN blocks). In practice, routing is **skewed**: a few experts get lots of tokens, most get very few or none. The standard TRT-LLM CUTLASS MoE kernel uses a **single CTA tile shape** for ALL experts — this is a compromise: the tile is too big for the many small experts and too small for the few large ones.

**Dual-tile** splits experts into two groups based on how many tokens they receive:
- **Small group**: experts with ≤ `threshold` tokens → run with a smaller/cheaper CUTLASS tile
- **Large group**: experts with > `threshold` tokens → run with a larger/faster CUTLASS tile

The kernel runs the full MoE pipeline **twice** (once per group), accumulating results into a shared output buffer via atomic scatter-adds.

---

## 2. Background: How Standard MoE Works in TRT-LLM

Understanding the existing code path is essential. Here's the full call chain from model forward to kernel execution.

### 2.1 Model Layer (Qwen3 example)

File: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py`

The Qwen3 MoE model defines a `Qwen3MoeSparseMoeBlock` which calls `self.moe(hidden_states, router_logits)`. The `self.moe` object is created by a factory function.

### 2.2 MoE Factory

File: `tensorrt_llm/_torch/modules/fused_moe/create_moe.py`

```python
def create_moe(...) -> MoE:
```

This function:
1. Calls `get_moe_cls(model_config)` which checks `model_config.moe_backend`:
   - `"CUTLASS"` → returns `CutlassFusedMoE` class
   - `"TRTLLM"` → returns `TRTLLMGenFusedMoE` class
   - etc.
2. Instantiates the chosen backend class
3. **If `ENABLE_CONFIGURABLE_MOE=1`** (the default), wraps it in `ConfigurableMoE`

The result is: `ConfigurableMoE(backend=CutlassFusedMoE(...))`.

### 2.3 ConfigurableMoE Wrapper

File: `tensorrt_llm/_torch/modules/fused_moe/configurable_moe.py`

`ConfigurableMoE` is a **composition wrapper** that separates:
- **Backend**: the actual GEMM computation (e.g., `CutlassFusedMoE`)
- **Communication**: AllGather/ReduceScatter, AllToAll, etc.
- **Load balancing**: optional EPLB

When `forward()` is called, `ConfigurableMoE` handles communication and routing, then delegates the actual MoE GEMM to `self.backend.run_moe(...)`.

**Important**: To access the actual CUTLASS backend from a `ConfigurableMoE`, use `moe.backend` (singular), NOT `moe.backends`. This is where dual-tile config lives.

### 2.4 CutlassFusedMoE Backend

File: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`

Class `CutlassFusedMoE` (starts at line 207) has:
- `__init__()`: Reads `model_config.moe_dual_tile`, stores config on `self`
- `run_moe()`: The actual dispatch — calls either `fused_moe` (standard) or `fused_moe_dual_tile` (our new path)

### 2.5 Custom Op → C++ Binding → CUDA Kernel

File: `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`

The `run_moe()` method calls `torch.ops.trtllm.fused_moe(...)` or `torch.ops.trtllm.fused_moe_dual_tile(...)`. These are **custom PyTorch ops** registered via `@torch.library.custom_op`.

The custom op creates a C++ `MoERunner` object and calls its methods, which ultimately call CUTLASS GEMM kernels on the GPU.

---

## 3. The Problem: Skewed Expert Routing

In a 128-expert, top-8 model (like Qwen3-30B), each input token is routed to 8 out of 128 experts. The routing is highly non-uniform:

- ~40% of experts receive **zero tokens** per step (completely idle)
- ~70% of active experts receive very few tokens (single digits)
- A small handful of experts receive the bulk of tokens

A single CUTLASS CTA tile configuration must handle everything from 1-token experts to 64-token experts. This is wasteful: large tiles on 1-token experts waste GPU resources, while the same tile may be suboptimal for 64-token experts.

**The dual-tile idea**: Profile the actual token distribution, pick a threshold (e.g., the P70 value), and run two passes with different tile shapes optimized for each workload size.

---

## 4. Step 1: Profile Expert Routing

### 4.1 What the Profiler Does

File: `profile_expert_routing_real.py` (project root)

This is a **standalone script** that runs **full transformer forward passes** through the real Qwen3-30B-A3B-NVFP4 checkpoint on real text data (wikitext-103). It is NOT a synthetic benchmark — it loads real NVFP4 weights, dequantizes them to BF16, and runs real attention + MoE computation through all 48 layers.

**Why real forward passes matter**: The router's token→expert decisions depend on the actual hidden states at each layer. Random inputs produce unrealistic routing patterns. By running attention + MoE through all layers sequentially, each layer sees hidden states that reflect real model behavior.

### 4.2 How It Works Internally

The profiler:

1. **Loads model config** (hardcoded for Qwen3-30B):
   ```
   HIDDEN=2048, INTER=768, NUM_EXPERTS=128, TOP_K=8, NUM_LAYERS=48
   NUM_HEADS=32, NUM_KV_HEADS=4, HEAD_DIM=128, ROPE_THETA=1000000.0
   ```

2. **Loads embeddings** from the checkpoint (BF16, shape `(151936, 2048)`)

3. **Tokenizes wikitext-103** using the Qwen3 tokenizer

4. **For each batch size** (e.g., 64, 128, 256 tokens):
   - Takes a chunk of tokenized text
   - Looks up embeddings → `hidden` tensor of shape `(1, batch_size, 2048)`
   - Runs through **all 48 layers** sequentially, each layer doing:

     **Attention block** (function `attention_forward`, line 88):
     - RMSNorm the hidden states
     - Dequantize Q/K/V projection weights from NVFP4 → BF16 on-the-fly
     - Compute Q, K, V projections
     - Apply QK norms (per-head RMSNorm)
     - Apply RoPE (rotary position embeddings)
     - Expand KV heads for GQA (4 KV heads → 32 Q heads)
     - Run scaled dot-product attention (causal)
     - Apply output projection, add residual

     **MoE block** (function `moe_forward`, line 130):
     - RMSNorm the hidden states
     - Compute gate logits: `hidden @ gate_weight.T` → softmax → top-8
     - Normalize top-k scores
     - **Count tokens per expert** (this is what we capture)
     - For each active expert: dequantize gate/up/down projections → SwiGLU → weighted scatter-add
     - Add residual

5. **After all chunks**: Aggregate per-expert token counts across all layers and chunks. Compute percentiles (P30, P50, P70, P90), mean, max, and idle%.

### 4.3 Key Design Decision: Layer-by-Layer Weight Loading

NVFP4 weights are dequantized one layer at a time to fit in GPU VRAM. Each expert has 3 weight matrices (gate_proj, up_proj, down_proj), each ~768×2048 in BF16. With 128 experts × 3 matrices × 48 layers, this is far too much to hold simultaneously. The `WeightLoader` class (line 37) uses lazy-loaded `safetensors` file handles.

### 4.4 NVFP4 Dequantization (in the profiler)

The profiler dequantizes NVFP4 weights using a lookup table approach (function `dequant`, line 57):

```python
# Each byte stores two FP4 values (low nibble + high nibble)
low  = (w_u8.to(int32) & 0x0F)       # element 0
high = ((w_u8.to(int32) >> 4) & 0x0F) # element 1

# FP4 E2M1 lookup table: [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
fp4_vals = LUT[unpacked]

# Apply two-level scaling: per-block scale × per-tensor scale
result = fp4_vals * weight_scale * weight_scale_2
```

### 4.5 Running the Profiler

```bash
docker exec -e LD_LIBRARY_PATH="/usr/local/tensorrt/lib" \
    -e PYTHONPATH="/code/tensorrt_llm" trtllm-build \
    python3 /code/tensorrt_llm/profile_expert_routing_real.py \
    --batch-sizes 64 128 256 \
    --num-chunks 3 \
    --num-layers 48 \
    --output dual_tile_config.json
```

**Arguments**:
- `--batch-sizes`: Token counts to profile (simulates different decoding batch sizes)
- `--num-chunks`: Number of different text chunks per batch size (for statistical robustness)
- `--num-layers`: How many of the 48 layers to process (default: all 48)
- `--output`: Path to write JSON config file (used in Step 2)

**Runtime**: ~45 minutes per chunk on RTX 5090 (mostly spent on per-expert dequantization).

### 4.6 Profiling Results

Results from real Qwen3-30B-A3B-NVFP4 inference on wikitext-103:

| Batch | P30 | P50 | P70 | P90 | Mean | Max | Idle% |
|-------|-----|-----|-----|-----|------|-----|-------|
| 64    | 2   | 4   | **7**  | 19  | 7.7  | 64  | 47.9% |
| 128   | 3   | 7   | **12** | 33  | 13.4 | 128 | 40.5% |
| 256   | 5   | 12  | **23** | 61  | 25.0 | 256 | 36.0% |

**How to read this**: For batch=128, 70% of active experts (P70) receive ≤12 tokens. 40.5% of all expert slots are completely idle (receive 0 tokens). The P70 value is the threshold — the point where 70% of active experts are "small" and 30% are "large".

### 4.7 JSON Output Format

The `--output` flag writes a JSON file like:

```json
{
  "model": "Qwen3-30B-A3B-NVFP4",
  "num_experts": 128,
  "top_k": 8,
  "hidden_size": 2048,
  "intermediate_size": 768,
  "profiles": {
    "64": {
      "threshold": 7,
      "p30": 2.0, "p50": 4.0, "p70": 7.0, "p90": 19.0,
      "mean": 7.7, "max": 64.0, "idle_pct": 47.9
    },
    "128": {
      "threshold": 12,
      "p30": 3.0, "p50": 7.0, "p70": 12.0, "p90": 33.0,
      "mean": 13.4, "max": 128.0, "idle_pct": 40.5
    },
    "256": {
      "threshold": 23,
      "p30": 5.0, "p50": 12.0, "p70": 23.0, "p90": 61.0,
      "mean": 25.0, "max": 256.0, "idle_pct": 36.0
    }
  }
}
```

The `threshold` field in each profile is the P70 value — used directly as the dual-tile threshold.

---

## 5. Step 2: Configure Dual-Tile in Python

### 5.1 The MoeDualTileConfig Dataclass

File: `tensorrt_llm/_torch/model_config.py`, line 31

```python
@dataclass
class MoeDualTileConfig:
    threshold: int = 16            # Token count boundary between small and large groups
    gemm1_small_tactic: int = 0    # CUTLASS tile index for small experts, GEMM1 (gate+up projections)
    gemm2_small_tactic: int = 0    # CUTLASS tile index for small experts, GEMM2 (down projection)
    gemm1_large_tactic: int = 0    # CUTLASS tile index for large experts, GEMM1
    gemm2_large_tactic: int = 0    # CUTLASS tile index for large experts, GEMM2
```

**What "tactic index" means**: CUTLASS discovers multiple valid CTA tile configurations (called "tactics") for a given GEMM shape. Each tactic has a different CTA tile shape (e.g., 128×128×64 vs 128×128×128). The index refers to the position in this discovered list. The list depends on the model dimensions (H, I) and the GPU architecture.

### 5.2 Setting It on ModelConfig

File: `tensorrt_llm/_torch/model_config.py`, line 113

```python
@dataclass(kw_only=True)
class ModelConfig(Generic[TConfig]):
    ...
    moe_backend: str = 'CUTLASS'
    moe_dual_tile: Optional[MoeDualTileConfig] = None  # None = disabled (default)
    ...
```

To enable dual-tile, set `moe_dual_tile` to a `MoeDualTileConfig` instance:

```python
from tensorrt_llm._torch.model_config import ModelConfig, MoeDualTileConfig

cfg = MoeDualTileConfig(
    threshold=12,              # from profiling P70
    gemm1_small_tactic=0,
    gemm2_small_tactic=0,
    gemm1_large_tactic=1,
    gemm2_large_tactic=1,
)

model_config = ModelConfig(
    pretrained_config=pretrained_config,
    mapping=mapping,
    moe_backend="CUTLASS",
    moe_dual_tile=cfg,         # None = disabled (the default)
)
```

### 5.3 Loading from Profiling JSON

```python
import json
from tensorrt_llm._torch.model_config import MoeDualTileConfig

with open('dual_tile_config.json') as f:
    data = json.load(f)

# Pick the profile matching your expected batch size
threshold = data['profiles']['128']['threshold']  # e.g., 12

cfg = MoeDualTileConfig(
    threshold=threshold,
    gemm1_small_tactic=0,
    gemm2_small_tactic=0,
    gemm1_large_tactic=1,
    gemm2_large_tactic=1,
)
```

### 5.4 Available Tile Shapes (SM 120, Qwen3 Dimensions)

These are the CUTLASS CTA tile shapes discovered at H=2048, I=768 on RTX 5090:

| GEMM | Index | Tile Shape (M×N×K) | Notes |
|------|-------|--------------------|-------|
| GEMM1 | 0 | 128×128×64 | Smaller tile |
| GEMM1 | 1 | 128×128×128 | Larger K dimension |
| GEMM2 | 0 | 128×128×64 | Smaller tile |
| GEMM2 | 1 | 128×128×128 | Larger K dimension |
| GEMM2 | 2 | 128×256×64 | Wider N dimension |
| GEMM2 | 3 | 256×128×64 | Taller M dimension |

**The shapes are too similar at these dimensions** — the M×N outer dimensions are all 128×128 or close, which is why dual-tile doesn't produce a speedup for Qwen3. Larger models (e.g., DeepSeek-V3 with H=7168) expose much more tile diversity.

---

## 6. Step 3: Runtime — Python Pipeline Call Chain

Here is the complete call chain from model forward to CUDA kernel, with exact file paths and line numbers.

### 6.1 Overview

```
Model.forward()
  └─ Qwen3MoeSparseMoeBlock.forward()           modeling_qwen3_moe.py
       └─ self.moe(hidden, router_logits)
            └─ ConfigurableMoE.forward()          configurable_moe.py
                 └─ ConfigurableMoE._forward_chunk_impl()
                      └─ self.backend.run_moe()   → CutlassFusedMoE.run_moe()
                           └─ torch.ops.trtllm.fused_moe_dual_tile(...)  OR
                              torch.ops.trtllm.fused_moe(...)
```

### 6.2 CutlassFusedMoE.__init__() — Config Storage

File: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`, lines 357–369

When `CutlassFusedMoE` is instantiated, it reads the dual-tile config from `model_config`:

```python
dual_tile_cfg = getattr(model_config, 'moe_dual_tile', None)
if dual_tile_cfg is not None and self.cluster_size > 1:
    logger.warning(
        "Dual-tile MoE is not supported with cluster_size > 1, disabling."
    )
    dual_tile_cfg = None
self.use_dual_tile = dual_tile_cfg is not None
if self.use_dual_tile:
    self.dual_tile_threshold = dual_tile_cfg.threshold
    self.gemm1_small_tactic = dual_tile_cfg.gemm1_small_tactic
    self.gemm2_small_tactic = dual_tile_cfg.gemm2_small_tactic
    self.gemm1_large_tactic = dual_tile_cfg.gemm1_large_tactic
    self.gemm2_large_tactic = dual_tile_cfg.gemm2_large_tactic
```

**Note**: If `cluster_size > 1` (multi-node MoE), dual-tile is automatically disabled with a warning. This is because the masked kernel doesn't support clustered parallelism.

### 6.3 CutlassFusedMoE.run_moe() — The Dispatch Point

File: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py`, lines 611–646

This is the critical fork in the code path:

```python
if self.use_dual_tile:
    result = torch.ops.trtllm.fused_moe_dual_tile(
        x,
        token_selected_experts,     # (num_tokens, top_k) int32 — which experts each token goes to
        token_final_scales,          # (num_tokens, top_k) float32 — routing weights
        self.w3_w1_weight.view(weight_dtype),  # FC1 weights (gate+up, stacked)
        self.w3_w1_bias,
        self.w2_weight.view(weight_dtype),     # FC2 weights (down projection)
        self.w2_bias,
        output_dtype,
        quant_scales=self.quant_scales,        # [fc1_act_g, fc1_sf, fc1_g, fc2_act_g, fc2_sf, fc2_g]
        input_sf=x_sf,                         # per-token FP4 input scale factors
        swizzled_input_sf=is_sf_swizzled,
        # ... (swiglu params, parallelism params, etc.)
        dual_tile_threshold=self.dual_tile_threshold,
        gemm1_small_tactic=self.gemm1_small_tactic,
        gemm2_small_tactic=self.gemm2_small_tactic,
        gemm1_large_tactic=self.gemm1_large_tactic,
        gemm2_large_tactic=self.gemm2_large_tactic,
    )
else:
    result = torch.ops.trtllm.fused_moe(x, ...)  # original path, completely unchanged
```

**Key differences between the two ops**:

| | `fused_moe` (original) | `fused_moe_dual_tile` (new) |
|---|---|---|
| Tactic selection | AutoTuner picks best dynamically | Fixed indices from config |
| cluster_size/rank | Fully supported | Hardcoded to 1/0 |
| min_latency_mode | Supported | Not supported |
| Extra parameters | — | threshold + 4 tactic indices |

### 6.4 fused_moe_dual_tile Custom Op

File: `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py`, lines 342–418

This is a `@torch.library.custom_op` that bridges Python → C++:

```python
@torch.library.custom_op("trtllm::fused_moe_dual_tile", mutates_args=())
def fused_moe_dual_tile(
    input: torch.Tensor,                    # (num_tokens, hidden_size) BF16
    token_selected_experts: torch.Tensor,   # (num_tokens, top_k) int32
    token_final_scales: torch.Tensor,       # (num_tokens, top_k) float32
    fc1_expert_weights: torch.Tensor,       # (num_experts, 2*inter, hidden//16) int64 (FP4 packed)
    fc1_expert_biases: Optional[torch.Tensor],
    fc2_expert_weights: torch.Tensor,       # (num_experts, hidden, inter//16) int64 (FP4 packed)
    fc2_expert_biases: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],       # 6 tensors: [fc1_act_g, fc1_sf, fc1_g, fc2_act_g, fc2_sf, fc2_g]
    input_sf: Optional[torch.Tensor],       # per-token input scale factors
    ...
    dual_tile_threshold: int = 16,
    gemm1_small_tactic: int = 0,
    gemm2_small_tactic: int = 0,
    gemm1_large_tactic: int = 0,
    gemm2_large_tactic: int = 0,
) -> List[torch.Tensor]:
```

Inside, it:
1. Creates a `MoERunner` C++ object with dtype triple `(bf16, int64, bf16)` for NVFP4
2. Calls `moe_runner.fused_moe_runner.set_dual_tile_profiles([g1s, g2s, g1l, g2l], threshold)` to configure the 4 tactics + threshold
3. Calls `moe_runner.fused_moe_runner.run_moe_dual_tile(...)` to execute

### 6.5 C++ Torch Binding

File: `cpp/tensorrt_llm/thop/moeOp.cpp`

The `FusedMoeRunner` C++ class is registered as a PyTorch class:

```cpp
// Line 1319-1329
TORCH_LIBRARY(trtllm, m) {
    m.class_<FusedMoeRunner>("FusedMoeRunner")
        .def(torch::init<ScalarType, ScalarType, ScalarType, bool, bool, bool, bool, bool>())
        .def("run_gemm_profile", ...)
        .def("get_tactic_num", ...)
        .def("run_moe", ...)
        .def("run_moe_min_latency", ...)
        .def("set_dual_tile_profiles", &FusedMoeRunner::setDualTileProfiles)  // NEW
        .def("run_moe_dual_tile", &FusedMoeRunner::runMoeDualTile);           // NEW
}
```

**`setDualTileProfiles`** (line 829): Takes 4 tactic indices + threshold. Looks up the actual `CutlassGemmConfig` structs from `mGemm1Profiles` / `mGemm2Profiles` (which are populated during GEMM profiling). Calls `mKernelRunner->setDualTileTactic(...)` to store them on the kernel runner.

```cpp
void setDualTileProfiles(ArrayRef<int64_t> profile_ids, int64_t threshold) {
    // profile_ids = [gemm1_small, gemm2_small, gemm1_large, gemm2_large]
    auto g1_small = mGemm1Profiles.at(profile_ids[0]);
    auto g2_small = mGemm2Profiles.at(profile_ids[1]);
    auto g1_large = mGemm1Profiles.at(profile_ids[2]);
    auto g2_large = mGemm2Profiles.at(profile_ids[3]);
    mKernelRunner->setDualTileTactic(g1_small, g2_small, g1_large, g2_large, threshold);
}
```

**`runMoeDualTile`** (line 840): Similar to `runMoe` but:
- Does NOT do AutoTuner profiling (tactics are already fixed)
- Calls `mKernelRunner->runMoeDualTile(...)` instead of `mKernelRunner->runMoe(...)`
- Hardcodes `cluster_size=1, cluster_rank=0`

---

## 7. Step 4: Runtime — C++ Kernel Internals

File: `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu`

This is the core CUDA implementation. `runMoeDualTile` executes the full MoE pipeline twice — once for small experts, once for large.

### 7.1 Algorithm (Pseudocode)

```
1. configureWsPtrs()              ← allocate workspace buffers
2. buildExpertMaps()              ← compute token→expert permutation
3. expandInputRowsKernel()        ← permute input rows + write input scale factors
4. memset(final_output, 0)        ← zero output buffer ONCE (both groups add into it)

─── SMALL GROUP (experts where token_count ≤ threshold) ───
5a. memset(glu_inter_result_, 0)  ← zero intermediate buffer
5b. setupDualTileInputs()         ← masked kernel: set M=0 for large experts
5c. gemm1() + doActivation()      ← CUTLASS GEMM with small tile + SwiGLU + FP4 requant
5d. gemm2(FINALIZE, skip_memset)  ← CUTLASS GEMM + atomic scatter-add into final_output

6. reExpandInputRows()            ← RESTORE aliased buffers destroyed by doActivation

─── LARGE GROUP (experts where token_count > threshold) ───
7a–7d. Same pipeline with large tile configs, M=0 for small experts

8. Restore saved GEMM configs     ← put original tactic back for future standard calls
```

### 7.2 Masked Kernel — setupDualTileInputs()

The key mechanism for group selection. Instead of physically separating tokens into two buffers (expensive), it **masks** the per-expert metadata:

- Runs `computeStridesTmaWarpSpecializedMaskedKernel` which iterates over all experts
- For each expert, checks its token count against the threshold
- If the expert belongs to the OTHER group → sets `gemm_m[expert] = 0`
- CUTLASS detects M=0 and **skips that expert entirely** (no wasted compute)

This means both groups share the same permuted input buffer, same weight buffers, same workspace — only the `gemm_m` array differs.

### 7.3 FINALIZE Epilogue — How Two Groups Merge Results

Both groups' `gemm2()` calls use the **FINALIZE fused epilogue**. This epilogue does:
1. Compute the down-projection GEMM: `expert_output = activation @ W2`
2. Multiply by routing weight
3. **Atomically scatter-add** each row into `final_output` using `permuted_row_to_unpermuted_row` mapping

Since `final_output` is zeroed once before either group runs, and both groups atomically add into it, the final result correctly accumulates contributions from both small and large experts.

The `skip_output_memset` parameter (added in our changes) prevents `gemm2()` from re-zeroing `final_output` internally when FINALIZE is active. Without this, the second group would erase the first group's results.

### 7.4 Buffer Aliasing Problem — reExpandInputRows()

The MoE workspace has memory optimizations where buffers overlap:
- `permuted_data_` (permuted input rows) aliases `fc1_result_` (GEMM1 output)
- `fc1_fp4_act_scale_` aliases `fc2_fp4_act_scale_`

After the first group runs `doActivation()` (SwiGLU + FP4 requantization), it **destroys** the contents of `permuted_data_` and `fc1_fp4_act_scale_` because they share memory with the activation outputs.

**Fix**: `reExpandInputRows()` runs between the two groups. It re-calls `expandInputRowsKernelLauncher` to restore the permuted input data and input scale factors from the original (non-aliased) sources. This is the ~16% overhead floor.

### 7.5 Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Input: (num_tokens, hidden_size)                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  expandInputRows()  →  permuted_data_ (tokens sorted by expert) │
│                                                                  │
│  memset(final_output, 0)                                         │
│                                                                  │
│  ┌─── SMALL GROUP ──────────────────────────────────────────┐   │
│  │ setupDualTileInputs(mask=SMALL)                          │   │
│  │   → gemm_m[large_experts] = 0                            │   │
│  │                                                          │   │
│  │ gemm1(small_tile) → glu_inter_ → doActivation(SwiGLU)   │   │
│  │ gemm2(small_tile, FINALIZE) → atomic add to final_output │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  reExpandInputRows()  ← restore destroyed aliased buffers        │
│                                                                  │
│  ┌─── LARGE GROUP ──────────────────────────────────────────┐   │
│  │ setupDualTileInputs(mask=LARGE)                          │   │
│  │   → gemm_m[small_experts] = 0                            │   │
│  │                                                          │   │
│  │ gemm1(large_tile) → glu_inter_ → doActivation(SwiGLU)   │   │
│  │ gemm2(large_tile, FINALIZE) → atomic add to final_output │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Output: final_output (num_tokens, hidden_size)                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 8. Step 5: Testing & Verification

### 8.1 E2E Pipeline Integration Test

File: `tests/test_dual_tile_e2e.py`

Tests 4 things:
1. **Config propagation**: `MoeDualTileConfig` → `ModelConfig` → `create_moe()` → `ConfigurableMoE` → `CutlassFusedMoE.backend` → verify `use_dual_tile`, `dual_tile_threshold`, and all 4 tactic indices are correct
2. **Disabled by default**: When `moe_dual_tile=None`, verify `use_dual_tile is False`
3. **Various config values**: Multiple threshold and tactic values all propagate correctly
4. **NVFP4 execution**: Actually runs `fused_moe_dual_tile` with synthetic FP4 weights through the C++ runner and verifies correctness against `run_moe` (rel_err < 1%)

**How to run**:
```bash
docker exec -e LD_LIBRARY_PATH="/usr/local/tensorrt/lib" \
    -e PYTHONPATH="/code/tensorrt_llm" trtllm-build \
    python3 /code/tensorrt_llm/tests/test_dual_tile_e2e.py
```

### 8.2 Real-Weights Benchmark Test

File: `test_dual_tile_real_weights.py` (project root)

Uses actual NVFP4 expert weights from the Qwen3-30B-A3B-NVFP4 checkpoint (layer 0). This tests with REAL weights, not random data.

**Phase 1 — Correctness**:
- Loads all 128 experts' gate/up/down projections from checkpoint
- Dequantizes NVFP4 → BF16 → re-quantizes to TRT-LLM FP4 format (see Section 10)
- Generates routing using real gate weights
- Runs single-tile (reference) vs dual-tile, compares output
- Tests batch sizes 32, 64, 128, 256 with profiling-derived thresholds
- Tests 4 different tile config combinations at batch=128

**Phase 2 — Performance**:
- Warm-up runs (5 iterations)
- Timed runs (20 iterations each), measures wall-clock latency
- Reports single-tile vs dual-tile latency and overhead ratio

**How to run**:
```bash
docker exec -e LD_LIBRARY_PATH="/usr/local/tensorrt/lib" \
    -e PYTHONPATH="/code/tensorrt_llm" trtllm-build \
    python3 /code/tensorrt_llm/test_dual_tile_real_weights.py
```

### 8.3 Navigating the CutlassFusedMoE Backend from ConfigurableMoE

When `ENABLE_CONFIGURABLE_MOE=1` (the default), `create_moe()` wraps the backend:

```python
moe = create_moe(...)  # Returns ConfigurableMoE

# WRONG:
inner = moe.backends  # This is a list used for multi-backend setups

# CORRECT:
inner = moe.backend   # This is the actual CutlassFusedMoE instance

# To check dual-tile config:
print(inner.use_dual_tile)          # True/False
print(inner.dual_tile_threshold)    # e.g., 12
```

The test file `tests/test_dual_tile_e2e.py` includes a helper function `get_cutlass_backend(moe)` that traverses the wrapper chain:

```python
def get_cutlass_backend(moe):
    inner = moe
    while hasattr(inner, 'backend') and inner.backend is not None:
        inner = inner.backend
    return inner
```

---

## 9. Performance Results

128 experts, H=2048, I=768, top-8, NVFP4+SwiGLU, RTX 5090. Real checkpoint weights, real gate routing.

### 9.1 Correctness

| Batch | Threshold | Active Experts | Max Error | Relative Error |
|-------|-----------|----------------|-----------|----------------|
| 32    | 4         | 108            | 0.0312    | 0.18%          |
| 64    | 7         | 122            | 0.0312    | 0.20%          |
| 128   | 12        | 127            | 0.0625    | 0.25%          |
| 256   | 23        | 128            | 0.0625    | 0.24%          |

All 4 cross-tile config pairs also pass. ~0.2% error is expected FP4 quantization noise (from re-quantization between groups).

### 9.2 Latency

| Batch | Threshold | Single-Tile | Dual-Tile | Overhead |
|-------|-----------|-------------|-----------|----------|
| 64    | 7         | 0.261ms     | 0.311ms   | +19%     |
| 128   | 12        | 0.267ms     | 0.326ms   | +22%     |
| 256   | 23        | 0.275ms     | 0.334ms   | +21%     |

### 9.3 Threshold Sensitivity (batch=128, baseline single-tile=0.270ms)

| Threshold | Dual Latency | Overhead |
|-----------|-------------|----------|
| 1         | 0.327ms     | +21%     |
| 7         | 0.332ms     | +23%     |
| 12        | 0.334ms     | +24%     |
| 40        | 0.314ms     | +16%     |
| 80        | 0.313ms     | +16%     |

Extreme thresholds push most experts into one group → ~16% overhead floor. Mixed splits add ~5–8% on top.

### 9.4 Why No Speedup at Qwen3 Dimensions

At H=2048, I=768, all available CTA shapes (128×128 through 256×128) perform similarly. The overhead of:
1. Running the pipeline twice (+2× GEMM kernel launches)
2. `reExpandInputRows()` buffer restoration (~16% overhead floor)
3. Two `setupDualTileInputs()` masked kernel calls

...outweighs any tile-shape benefit. The strategy is designed for **larger models** (e.g., DeepSeek-V3 with H=7168) where tile diversity is much greater — a 64×64 tile for 2-token experts vs a 256×256 tile for 100-token experts would show real gains.

---

## 10. NVFP4 Weight Conversion — Critical Details

This section explains how weights flow from the HuggingFace checkpoint into the TRT-LLM CUTLASS kernel. Getting this wrong causes silent numerical corruption.

### 10.1 Checkpoint Format (modelopt NVFP4)

The checkpoint stores each linear layer as:
- `weight` — `uint8` tensor, packed (2 FP4 values per byte)
- `weight_scale` — per-block scale (FP16/BF16), one per 16 elements
- `weight_scale_2` — per-tensor scale (scalar)

### 10.2 TRT-LLM FP4 Format

TRT-LLM expects a different format:
- Weights as `int64` (packed differently than checkpoint)
- Block scales interleaved via `block_scale_interleave()`
- Activation global scale factors

### 10.3 Conversion Pipeline

The test script `test_dual_tile_real_weights.py` shows the full conversion (function `quantize_for_trtllm`, line 38):

```python
def quantize_for_trtllm(w_bf16, global_sf):
    """Convert BF16 weight to TRT-LLM FP4 format."""
    rows, cols = w_bf16.shape

    # Step 1: FP4 quantize using TRT-LLM's built-in op
    packed_u8, sf_u8 = torch.ops.trtllm.fp4_quantize(
        w_bf16,        # input BF16 weight
        global_sf,     # global scale factor (use 1.0 — see below)
        16,            # block size
        False,         # not transposed
        False,         # no SF swizzle
    )

    # Step 2: Reinterpret packed bytes as int64 (TRT-LLM's weight dtype)
    w_int64 = packed_u8.view(torch.int64).reshape(rows, cols // 16)

    # Step 3: Interleave block scales for CUTLASS memory layout
    sf_2d = sf_u8.reshape(rows, cols // 16)
    sf_interleaved = torch.ops.trtllm.block_scale_interleave(sf_2d)
    sf_int32 = sf_interleaved.view(torch.int32).reshape(rows, cols // 16 // 4)

    return w_int64, sf_int32
```

### 10.4 Critical: global_sf Must Be 1.0

The conversion is: **dequantize checkpoint NVFP4 → BF16 → re-quantize to TRT-LLM FP4**.

When re-quantizing, use `global_sf = torch.tensor(1.0)`. The checkpoint's `input_scale` values (~0.001) will cause **underflow** if used as the global scale factor, because the re-quantized weights are already scaled differently.

Similarly, `fc1_act_global_scale` and `fc2_act_global_scale` should be `1.0` when using re-quantized weights. The per-token block scales (`input_sf`) handle the actual activation scaling.

### 10.5 FC1 Weight Stacking

For SwiGLU MoE, FC1 is the gate+up projection stacked:

```python
# Per expert:
fc1_bf16 = torch.cat([gate_proj_bf16, up_proj_bf16], dim=0)  # (2*inter, hidden)
# Then quantize fc1_bf16 → fc1_q (int64), fc1_sf (int32)
```

### 10.6 C++ Runner Constructor Dtypes

The C++ `FusedMoeRunner` constructor takes dtype arguments:

```python
runner = torch.classes.trtllm.FusedMoeRunner(
    torch.bfloat16,  # activation dtype (input)
    torch.int64,     # weight dtype (FP4 packed as int64)
    torch.bfloat16,  # output dtype
    False,           # use_deepseek_fp8_block_scale
    False,           # use_w4_group_scaling
    False,           # use_int8_woq_per_channel
    False,           # use_mxfp8_act_scaling
    True,            # use_fused_finalize (MUST be True for dual-tile)
)
```

The `(bf16, int64, bf16)` triple signals NVFP4 mode to the C++ side. Do NOT use `uint8` or `quint4x2` — those are different quant modes.

### 10.7 quant_scales List

The `quant_scales` parameter is a list of 6 tensors, in this exact order:

```python
quant_scales = [
    fc1_act_global_scale,  # scalar float32, use 1.0
    fc1_weight_scales,     # (num_experts, ...) int32 (interleaved block scales)
    fc1_global_scales,     # (num_experts,) float32, use ones
    fc2_act_global_scale,  # scalar float32, use 1.0
    fc2_weight_scales,     # (num_experts, ...) int32 (interleaved block scales)
    fc2_global_scales,     # (num_experts,) float32, use ones
]
```

---

## 11. Limitations & Caveats

1. **Requires TMA warp-specialized kernels** (SM 90+). Only SM 120 (RTX 5090) has been tested. BF16-only (no quantization) templates do NOT have TMA warp-specialized kernels on SM 120, so **NVFP4 is the only validated quantization path**.

2. **Not supported**:
   - LoRA (finalize fusion must be disabled for LoRA)
   - DeepSeek FP8 block scale
   - Unfused finalize (`use_fused_finalize=False`)
   - min-latency mode
   - `cluster_size > 1` (multi-node MoE parallelism)
   - `AllToAll` communication (the dual-tile op hardcodes `enable_alltoall=False` in practice)

3. **~16% overhead floor** from `reExpandInputRows()`. This could be eliminated by allocating separate GEMM1 input and GEMM2 input buffers instead of aliasing them in the workspace. This would increase memory usage slightly but remove the buffer restoration step.

4. **Global threshold only**: Currently one threshold for all layers. The profiling data shows per-layer variation — a per-layer threshold could be more optimal.

5. **No SM version guard**: The code doesn't check SM version before enabling dual-tile. On GPUs older than SM 90, the TMA warp-specialized kernels won't exist and it will likely crash.

6. **NVFP4 re-quantization**: Checkpoint FP4 (modelopt format) ≠ TRT-LLM FP4 (CUTLASS format). You must dequantize to BF16, then re-quantize using `torch.ops.trtllm.fp4_quantize()` with `global_sf=1.0`. This introduces a small quantization noise (~0.2% relative error).

---

## 12. All Modified Files

### Pipeline Integration (Python)

| File | Line(s) | What Changed |
|------|---------|-------------|
| `tensorrt_llm/_torch/model_config.py` | 31–36 | Added `MoeDualTileConfig` dataclass with 5 fields |
| `tensorrt_llm/_torch/model_config.py` | 113 | Added `moe_dual_tile: Optional[MoeDualTileConfig] = None` field to `ModelConfig` |
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py` | 357–369 | In `CutlassFusedMoE.__init__()`: read `model_config.moe_dual_tile`, store 5 config values on `self`, auto-disable if `cluster_size > 1` |
| `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py` | 611–646 | In `CutlassFusedMoE.run_moe()`: added `if self.use_dual_tile:` branch calling `torch.ops.trtllm.fused_moe_dual_tile(...)` with 5 extra params |

### Custom Op (Python → C++ bridge)

| File | Line(s) | What Changed |
|------|---------|-------------|
| `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` | 342–418 | Added `fused_moe_dual_tile` custom op: creates MoERunner, calls `set_dual_tile_profiles()`, calls `run_moe_dual_tile()` |
| `tensorrt_llm/_torch/custom_ops/torch_custom_ops.py` | 421+ | Added `register_fake` for `fused_moe_dual_tile` (needed for torch.compile) |

### C++ Torch Binding

| File | Line(s) | What Changed |
|------|---------|-------------|
| `cpp/tensorrt_llm/thop/moeOp.cpp` | 829–838 | Added `setDualTileProfiles()` method: maps 4 tactic indices → CutlassGemmConfig, calls `setDualTileTactic()` |
| `cpp/tensorrt_llm/thop/moeOp.cpp` | 840–938 | Added `runMoeDualTile()` method: validates inputs, allocates workspace, calls `mKernelRunner->runMoeDualTile()` |
| `cpp/tensorrt_llm/thop/moeOp.cpp` | 1327–1328 | Registered `set_dual_tile_profiles` and `run_moe_dual_tile` on the `FusedMoeRunner` torch class |
| `cpp/tensorrt_llm/thop/moeOp.cpp` | (access) | Changed `setDualTileProfiles` from private to public (was causing compilation error) |

### C++ CUDA Kernel

| File | What Changed |
|------|-------------|
| `cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu` | Added `runMoeDualTile()`: the main dual-tile algorithm (expand → memset → small group → reExpand → large group → restore). Added `setupDualTileInputs()` masked kernel call. Added `reExpandInputRows()` buffer restoration lambda. 5 bug fixes (see Section 13). |
| `cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h` | Added `skip_output_memset` parameter to `gemm2()` declaration. Added virtual `setDualTileTactic()` and `runMoeDualTile()` declarations. |

### Tests & Profiling Scripts

| File | What It Does |
|------|-------------|
| `profile_expert_routing_real.py` | Standalone profiler: full 48-layer Qwen3 forward pass on wikitext-103, captures per-expert token counts, writes JSON config with `--output` flag |
| `test_dual_tile_real_weights.py` | Loads real NVFP4 checkpoint weights from layer 0, tests correctness + performance with real gate routing |
| `tests/test_dual_tile_e2e.py` | Tests config propagation through the full pipeline chain + NVFP4 execution correctness |

### Documentation

| File | What It Does |
|------|-------------|
| `docs/dual_tile_moe_summary.md` | This file |

---

## 13. Bugs Fixed During Development

Five bugs were found and fixed in the C++ kernel during development:

### Bug 1: CRASH — FINALIZE Metadata Not Populated

**Symptom**: Segfault in `gemm2()` during the FINALIZE epilogue.

**Root Cause**: `setupDualTileInputs()` calls the masked kernel to zero out `gemm_m` for excluded experts, but it didn't set the `g2_tma.fusion = FINALIZE` field or call `setFinalizeFusionParams()`. The FINALIZE epilogue tried to read uninitialized metadata (scatter-add permutation pointers, routing weights, etc.).

**Fix**: Before calling the masked kernel, set `fusion = FINALIZE` and call `setFinalizeFusionParams()` to populate all FINALIZE-required fields.

### Bug 2: Double-Zeroing Output Buffer

**Symptom**: Second group's results correct, but first group's results zeroed out.

**Root Cause**: `gemm2()` with FINALIZE internally calls `memset(final_output, 0)` before its scatter-add. This is correct for single-tile (run once), but for dual-tile the second group's `gemm2()` would zero out the first group's accumulated results.

**Fix**: Added `skip_output_memset` parameter to `gemm2()`. The dual-tile code zeros `final_output` once before either group runs, then passes `skip_output_memset=true` to both groups' `gemm2()` calls.

### Bug 3: CRASH — `permuted_token_final_scales_` Not Allocated

**Symptom**: Segfault when accessing `permuted_token_final_scales_` during FINALIZE.

**Root Cause**: `configureWsPtrs()` allocates `permuted_token_final_scales_` only when it detects FINALIZE mode. But when `configureWsPtrs()` runs, the fusion type hasn't been set yet (it's set later during GEMM config selection). So the buffer is never allocated.

**Fix**: Temporarily set the FINALIZE fusion type before calling `configureWsPtrs()`, then restore it afterward. This ensures the workspace allocation includes the FINALIZE-required buffer.

### Bug 4: Interleaved Execution Corrupts Intermediates

**Symptom**: Incorrect results with the original execution order.

**Root Cause**: The original order was `gemm1_small → gemm1_large → gemm2_small → gemm2_large`. But `gemm1_large` overwrites `fc1_result_` (the GEMM1 output buffer), destroying `gemm1_small`'s results before `gemm2_small` can read them.

**Fix**: Changed to per-group sequential: `{gemm1 → doActivation → gemm2}_small → {gemm1 → doActivation → gemm2}_large`. Each group's full pipeline completes before the next starts.

### Bug 5: Buffer Aliasing Between Groups

**Symptom**: Second group produces garbage output.

**Root Cause**: The workspace aliases `permuted_data_` ↔ `fc1_result_` and `fc1_fp4_act_scale_` ↔ `fc2_fp4_act_scale_`. After the first group runs `doActivation()`, the SwiGLU activation + FP4 requantization overwrites the memory backing `permuted_data_` and `fc1_fp4_act_scale_`. The second group then reads corrupted input.

**Fix**: Added `reExpandInputRows()` between the two groups. This re-runs `expandInputRowsKernelLauncher` to rebuild the permuted input data and input scale factors from the original (non-aliased) source buffers. This is correct but adds ~16% overhead.

---

## 14. Build & Run Instructions

### 14.1 Build TRT-LLM (C++ Only, No Clean)

```bash
docker exec trtllm-build python3 /code/tensorrt_llm/scripts/build_wheel.py \
    --build_type Release \
    --cuda_architectures '120-real' \
    -j$(nproc) \
    --cpp_only \
    --skip_building_wheel
```

**Important flags**:
- `--cuda_architectures '120-real'`: Must match your GPU. RTX 5090 = SM 120.
- `--cpp_only --skip_building_wheel`: Only rebuild C++ code, skip Python wheel (faster).
- Do NOT use `--clean` unless absolutely necessary — full rebuild takes much longer.

### 14.2 Copy Built Library

After building, the built `libth_common.so` needs to be accessible:

```bash
docker exec trtllm-build cp \
    /code/tensorrt_llm/cpp/build/tensorrt_llm/thop/libth_common.so \
    /code/tensorrt_llm/tensorrt_llm/libs/libth_common.so
```

### 14.3 Run Profiler

```bash
docker exec -e LD_LIBRARY_PATH="/usr/local/tensorrt/lib" \
    -e PYTHONPATH="/code/tensorrt_llm" trtllm-build \
    python3 /code/tensorrt_llm/profile_expert_routing_real.py \
    --batch-sizes 64 128 256 \
    --num-chunks 3 \
    --output dual_tile_config.json
```

### 14.4 Run E2E Test

```bash
docker exec -e LD_LIBRARY_PATH="/usr/local/tensorrt/lib" \
    -e PYTHONPATH="/code/tensorrt_llm" trtllm-build \
    python3 /code/tensorrt_llm/tests/test_dual_tile_e2e.py
```

### 14.5 Run Real-Weights Benchmark

```bash
docker exec -e LD_LIBRARY_PATH="/usr/local/tensorrt/lib" \
    -e PYTHONPATH="/code/tensorrt_llm" trtllm-build \
    python3 /code/tensorrt_llm/test_dual_tile_real_weights.py
```

### 14.6 Docker Container Name

All commands above assume a Docker container named `trtllm-build`. The model checkpoint is cached at:
```
/root/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3/
```
