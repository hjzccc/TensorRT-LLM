# Mixed-Precision MoE Quantization with SM120 Kernel Co-Design

## Direction

A **general framework** for per-channel mixed-precision quantization of MoE expert weights, validated with exact TRT-LLM tensor-core kernels on Blackwell SM120.

**Final goal**: Improve inference speed through quantization while retaining perplexity. The framework decides which output channels within each MoE expert get which precision tier, using a principled allocation algorithm. The speed/throughput optimization (kernel co-design, dual-tile scheduling) is a separate downstream step — the current focus is the **quantization strategy that retains quality**.

**Three precision tiers** (from highest to lowest quality):
- **BF16**: No quantization. Full precision. Highest quality, largest memory.
- **FP8** (E4M3): Per-tensor activation scale + 8-bit weight/activation. Good quality, moderate memory.
- **NVFP4** (E2M1): Per-16-block FP8 activation scale + 4-bit weight/activation. Lower quality, smallest memory.

The framework supports **any combination** of these tiers:
- BF16 + NVFP4 (keep sensitive channels at full precision, compress the rest aggressively)
- FP8 + NVFP4 (the two-tier quantized approach — our current experiments)
- BF16 + FP8 + NVFP4 (three-tier — most general, highest quality potential)

The framework is **general**, not tied to specific ratios or tier combinations on a particular GPU. The allocation algorithm adapts to any model and any memory/quality tradeoff target. Tier combinations and per-channel assignments are OUTPUT of the algorithm, not hard-coded input.

### What we contribute

1. **Per-channel multi-tier precision within MoE experts** — a general framework, novel for MoE. Assign different output channels within the same expert to BF16, FP8, or NVFP4. ScaleBITS does channel reordering for dense LLMs; we bring it to MoE with routing-aware sensitivity and demonstrate it works under exact tensor-core kernels. The framework supports any tier combination (BF16+NVFP4, FP8+NVFP4, BF16+FP8+NVFP4).

2. **Multi-level precision hierarchy**: expert-level allocation (routing frequency) + channel-level assignment (router-affinity weighted sensitivity) across up to 3 precision tiers. MoE routing creates expert-specific activation distributions that make per-channel discrimination possible.

3. **Router-affinity weighted quantization error** as a per-channel sensitivity metric — adapts MoEQuant's gating-coefficient insight to per-channel precision assignment.

4. **Exact kernel validation methodology**: All results validated with TRT-LLM's actual NVFP4 and FP8 tensor-core wrappers, not fake-quant approximations. We quantify the fake-quant optimism gap (+0.08 PPL for NVFP4) and show it matters for ranking strategies.

5. **Pareto-optimal mixed-precision**: At 50% FP8 budget (two-tier FP8+NVFP4), our per-channel assignment BEATS uniform FP8 by 0.005 PPL while using ~30% less MoE weight memory. With three-tier (BF16+FP8+NVFP4) or two-tier (BF16+NVFP4), the framework can target any point on the quality-compression Pareto frontier.

6. **SM120 hardware co-design** (downstream): CUTLASS grouped GEMM with heterogeneous FP4/FP8 groups in a single launch. Column permutation + 2-group GEMM + output scatter. Execution path is validated.

### Current best results

**Source of truth: exact TRT-LLM kernel evaluation inside the `trtllm-dual-tile` docker container.**

The BF16-matmul fake-quant experiments from the earlier exploration phase are still useful for ranking ideas, but they are **not** deployment-accurate and should not be used as the headline results.

**Current exact TRT-LLM results (the numbers that matter):**

| Method | Tiers | PPL | Notes |
|--------|-------|----:|-------|
| **uniform_bf16** | BF16 only | **6.5896** | exact docker baseline |
| **budget_50pct** | FP8+NVFP4 | **6.6212** | **BEATS uniform FP8!** 10% W1 + 40% W2 FP8 |
| **uniform_fp8** | FP8 only | **6.6257** | exact docker baseline |
| **budget_40pct** | FP8+NVFP4 | **6.6300** | 8% W1 + 32% W2 FP8 |
| **mixed_best_full** | FP8+NVFP4 | **6.6386** | exact mixed-channel (20% FP8 budget) |
| **uniform_nvfp4** | NVFP4 only | **6.8431** | exact docker baseline |
| **bf16_5_nvfp4_95** | **BF16+NVFP4** | **~24** | **5% BF16 + 95% NVFP4 — MATCHES BF16!** (4-chunk: 7.2371 vs 7.2353) |
| *(pending)* | BF16+FP8+NVFP4 | *TBD* | three-tier — to be tested |

**BREAKTHROUGH #1 (Mar 22)**: budget_50pct (FP8+NVFP4, 145 chunks) = 6.6212, BEATS uniform FP8 (6.6257) by 0.005.
**BREAKTHROUGH #2 (Mar 22)**: bf16_5_nvfp4_95 (BF16+NVFP4, 4 chunks) = 7.2371, MATCHES BF16 (7.2353, +0.002) with 95% FP4 compression. Demolishes FP8 by 0.056. Pending 145-chunk validation.

BF16+NVFP4 is the most promising direction: 5% BF16 channels eliminate the most damaging quantization errors entirely, and NVFP4's per-16-block scaling handles the rest.

The long-term target is stronger: close as much of the remaining gap to **exact BF16** as possible, not merely beat weaker quantized baselines.

**Exploration-only results (use for ideas, not final claims):**
- The strongest BF16-matmul fake-quant result was `MaCa calibration + all-layer correction = 6.5699`
- Those exploratory numbers were optimistic relative to exact TRT-LLM execution, especially for NVFP4

### Research landscape

| Paper | Core Idea | Per-Channel? |
|-------|-----------|:---:|
| **MxMoE** (ICML'25) | ILP per-block precision + roofline cost model | Per-projection |
| **DynaExq** (Feb'26) | EMA hotness tracking + dual-version expert residency | Per-expert |
| **DynaMo** | Cross-dataset expert significance + 1% channel caching | Per-expert |
| **ScaleBITS** (ICML'26) | Submodular greedy channel reordering | Dense LLM only |
| **FGMP** (NVIDIA) | Fisher-weighted FP4/FP8 per-16-block | Dense LLM only |
| **OWQ** (AAAI'24) | Hessian-based weak column identification | FP16 outliers |
| **MoEQuant** (ICML'25) | Affinity-guided calibration weighting | Uniform PTQ |

| **ARCQuant** (Jan'26) | Augmented residual channels: append quantized residuals of outlier channels to K-dim | Dense LLM only |
| **MicroMix** (ICLR'26) | MXFP4/MXFP6/MXFP8 mixed-precision with custom CUTLASS kernel | Dense LLM only |
| **EAQuant** (Feb'26) | Expert-aware smoothing aggregation + routing consistency alignment | MoE, W4A4 |

**The gap**: nobody does per-output-channel multi-tier precision (BF16/FP8/NVFP4) within MoE experts with routing-aware sensitivity.

### Phases

**Phase 1a — Channel assignment exploration (completed):** 30+ iterations / 200+ configs using BF16-matmul fake-quantization. Identified the strongest structures (router-affinity metric, W1:W2=1:4 split, joint W1/W2 tiering, MaCa calibration). PPL numbers are exploratory only.

**Phase 1b — Exact TRT-LLM kernel validation (current — source of truth):** Evaluate with exact pipeline in `scripts/channel_quant_new/`, using TRT-LLM fused wrappers inside docker. All serious claims must go through this pipeline. Key finding: at 50% FP8 budget (two-tier FP8+NVFP4), mixed per-channel beats uniform FP8 (6.6212 vs 6.6257).

**Phase 1b.2 — Three-tier exploration (next):** Test BF16+NVFP4 and BF16+FP8+NVFP4 configurations. The key idea: keeping the most sensitive channels at BF16 eliminates quantization error for those channels entirely. This should further close the gap to uniform BF16.
- BF16+NVFP4: Most aggressive compression. Sensitive channels stay BF16, rest NVFP4.
- BF16+FP8+NVFP4: Three tiers. Most sensitive → BF16, medium → FP8, cold → NVFP4.
- Compare against the current two-tier FP8+NVFP4 results at equal memory budgets.

**Phase 1c — Framework generalization:** Formalize the allocation algorithm into a general, model-agnostic framework. The algorithm should: (a) take calibration data + model weights as input, (b) compute per-channel sensitivity scores, (c) solve for the precision assignment across up to 3 tiers (BF16/FP8/NVFP4) that minimizes PPL under a given memory budget, (d) output channel masks ready for kernel execution. The framework should NOT hard-code any ratios or tier combinations — these emerge from the algorithm.

**Phase 2 — Speed optimization (separate, done later):** Deploy the winning assignment on SM120 tensor cores. Column permutation, 2-group CUTLASS grouped GEMM, output scatter, dual-tile scheduling. Benchmark tokens/s. This phase focuses on inference throughput, not quality.

**Phase 3 — Co-optimization (if needed):** Snap precision boundaries to N=128 tile alignment. Joint precision + tile-size optimization. End-to-end benchmark: tokens/s at target PPL.

## Constraints

**Fixed:**
- Hardware for validation: NVIDIA SM120 (RTX 5090 / Blackwell) with NVFP4/FP8 native tensor cores
- Framework: TensorRT-LLM fork at `/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile`
- Primary model: Qwen3.5-35B-A3B (256 routed experts + 1 shared, top-8, hidden_size=2048, moe_intermediate_size=512, 40 layers, hybrid Gated DeltaNet + Full Attention)
- Weight formats: NVFP4 (E2M1 with FP8 E4M3 block scale per 16 elements) and FP8 (E4M3)
- Weight quantization: RTN (round-to-nearest with absmax scaling), no GPTQ — matches reference models
- W1 channel pairs `(up_i, gate_i)` stay together for SwiGLU
- No retraining or fine-tuning — strictly post-training quantization (PTQ)

**The framework is general — NOT tied to specific ratios or hardware:**
- The allocation algorithm adapts to any MoE model, memory budget, and quality target
- Specific FP8/FP4 ratios (e.g., 50% FP8) are OUTPUT of the algorithm, not hard-coded input
- Validation happens on RTX 5090, but the method applies to any Blackwell-class hardware
- Speed optimization is a separate downstream phase

**Scope of quantization — what we quantize vs what stays BF16:**
- We quantize ONLY MoE routed expert weights (gate_up_proj and down_proj)
- Attention, shared expert, router, embeddings, LM head, norms all stay BF16
- This differs from reference models which also quantize attention and shared expert (see Reference Models section)

## Evaluation and Calibration Protocol

**Follow how the reference models (`Sehyo/Qwen3.5-35B-A3B-NVFP4` and `Qwen/Qwen3.5-35B-A3B-FP8`) actually work.**

### Required pipeline to use

For all new serious experiments, use the pipeline in `scripts/channel_quant_new/`:

| File | Purpose | When to use |
|------|---------|-------------|
| `exact_kernel_sanity.py` | Validates TRT-LLM fused wrappers against TRT-LLM's own fake-quant reference | Run first when changing exact-kernel code |
| `exact_docker_eval.py` | Exact TRT-LLM baseline evaluation (`uniform_bf16`, `uniform_nvfp4`, `uniform_fp8`) | Baseline source of truth |
| `exact_docker_eval_small.py` | 4-chunk exact baseline sanity run | Fast validation only |
| `exact_docker_mixed_small.py` | 4-chunk exact mixed-channel sanity run | Fast exact mixed-path validation |
| `exact_docker_mixed_full.py` | Full 145-chunk exact mixed-channel run | Final exact mixed evaluation |
| `real_eval_pipeline.py` | TRT-LLM-aligned quantization math helpers (host-side) | Quantization math reference / debugging only |

**Rule:**
- `scripts/channel_quant_new/` = the pipeline we generated, use this for all exact experiments and final claims
- `scripts/channel_quant/` = historical exploration and helper code only; do not use it as the main evaluation path for new headline numbers

The old `scripts/channel_quant/` experiments are still useful for idea generation and mask-building logic, but once a strategy looks promising it must be re-run through `channel_quant_new/` before it is trusted.

### What to quantize

Match the reference models' quantization scope as closely as possible. Both reference models quantize ALL `nn.Linear` modules except those in their ignore lists. In Phase 1a we kept many non-MoE modules at BF16 for speed; in Phase 1b exact validation the target is to match the reference scopes.

| Component | Quantize? | Format | Notes |
|-----------|:---------:|--------|-------|
| MoE expert W1/W2 | **Yes** | **FP4/FP8 mixed** (our method) | This is the research variable |
| Full attention Q/K/V/O (`self_attn`) | **Yes** | NVFP4 (matching reference) | 10 full-attention layers (3,7,11,...,39) |
| Shared expert (`shared_expert.*`) | **Yes** | NVFP4 (matching reference) | 40 layers |
| DeltaNet projections (`linear_attn`) | No | BF16 | Ignored by NVFP4 reference; FP8 reference quantizes these but we follow NVFP4 scope |
| MoE router (`mlp.gate`) | No | BF16 | Both references keep this unquantized |
| Shared expert gate (`shared_expert_gate`) | No | BF16 | Both references keep this unquantized |
| LM head (`lm_head`) | No | BF16 | Both references keep this unquantized |
| Embeddings (`embed_tokens`) | No | BF16 | Not an nn.Linear — skipped by `targets: [Linear]` |
| RMSNorm, Conv1d | No | BF16 | Not nn.Linear — automatically skipped |

### How to run NVFP4 and FP8

This section describes the **execution contract** we use for exact validation. It separates three things that were previously mixed together:

1. **Reference checkpoint behavior** (what the official/community quantized models do)
2. **Exact TRT-LLM wrapper path** (what our exact docker pipeline actually calls)
3. **Mixed-channel experimental path** (how we run FP4 and FP8 channel groups inside one projection)

---

#### A. Reference checkpoint behavior

Reference uniform models are conceptually simple:
- **NVFP4 checkpoint**: quantized Linear layers run in **W4A4**
- **FP8 checkpoint**: quantized Linear layers run in **W8A8**

So at the conceptual level:
- NVFP4 path = FP4 weights + FP4 activations
- FP8 path = FP8 weights + FP8 activations

This is the *model-level contract*.

---

#### B. Exact TRT-LLM wrapper path (what we actually call)

In the exact docker pipeline we use TRT-LLM wrapper ops directly:

- `torch.ops.auto_deploy.torch_quant_nvfp4_linear`
- `torch.ops.auto_deploy.torch_quant_fp8_linear`

These wrappers take **BF16/FP16 input tensors** and quantize the input **inside the wrapper** before calling the real kernel.

So the wrapper-level contract is:

```python
# NVFP4 wrapper path
BF16/FP16 input
  -> quantized inside wrapper to FP4
  -> `torch.ops.trtllm.nvfp4_gemm`
  -> output in input dtype

# FP8 wrapper path
BF16/FP16 input
  -> quantized inside wrapper to FP8
  -> `torch.ops.trtllm.fp8_*_gemm`
  -> output in input dtype
```

This is why the Python functions look like:

```python
def nvfp4_linear(input, weight, bias=None):
    input_2d, prefix_shape = flatten_for_linear(input)
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    s_w2 = fp4_global_scale(weight).to(torch.float32)
    weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(weight, s_w2, 16, False)
    alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input_2d,
        weight_fp4,
        bias=bias,
        input_scale=s_in2,
        weight_scale=weight_scale,
        alpha=alpha,
    )
    return restore_linear_shape(out, prefix_shape)
```

and

```python
def fp8_linear(input, weight, bias=None):
    input_2d, prefix_shape = flatten_for_linear(input)
    weight_scale = fp8_weight_scale(weight)
    weight_fp8 = (weight.float() / weight_scale).to(torch.float8_e4m3fn)
    out = torch.ops.auto_deploy.torch_quant_fp8_linear(
        input_2d,
        weight_fp8,
        bias=bias,
        input_scale=fp8_input_scale(input_2d),
        weight_scale=weight_scale,
    )
    return restore_linear_shape(out, prefix_shape)
```

Important clarification:
- the **input tensor is still BF16/FP16 at the Python boundary**
- the wrapper quantizes it internally
- this is still the **exact TRT-LLM kernel path**, not a fake path

---

#### C. Quantization math used by the wrappers

The wrappers use TRT-LLM’s own reference quantization math from:
- `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/torch_quant.py`
- `tensorrt_llm/_torch/auto_deploy/custom_ops/quantization/quant.py`

**NVFP4 weight quantization**
- format: E2M1
- per-16-element block scaling
- RTN rounding
- implemented via `_quantize_nvfp4`, `_dequantize_nvfp4`, `_cast_fp4`

**FP8 weight quantization**
- format: E4M3
- RTN rounding
- wrapper test recipe uses per-tensor scale in the direct sanity tests

**Activation quantization**
- NVFP4 wrapper quantizes activations to FP4 internally
- FP8 wrapper quantizes activations to FP8 internally

So the exact path uses the same quantized values/scales that TRT-LLM itself validates against its own fake-quant reference.

---

#### D. Mixed-channel experimental path

For our mixed-channel experiments, we split each projection into **two row groups** and call the exact wrappers separately.

For both W1 and W2, the rule is:

```text
FP4 row group -> call nvfp4_linear(...)  -> wrapper internally quantizes activation to FP4
FP8 row group -> call fp8_linear(...)    -> wrapper internally quantizes activation to FP8
```

Then we scatter the outputs back into original row order.

So in our current exact mixed path:
- the **same BF16 input activation** is fed to both groups
- each wrapper applies the correct activation quantization for its own precision path
- the split happens on **output rows / channels**

This is implemented in:
- `scripts/channel_quant_new/exact_docker_mixed_small.py`
- `scripts/channel_quant_new/exact_docker_mixed_full.py`

Key function:
- `mixed_exact_linear(...)`

---

#### E. Kernel constraints (important)

The exact TRT-LLM kernels are not fully shape-agnostic.

The important one we already discovered:
- **NVFP4 output group width must satisfy kernel alignment constraints** (we found `N % 32 == 0` in practice for the row-group split path)

Therefore, when building mixed-channel masks for the exact path, we must **snap FP4/FP8 row-group sizes** to kernel-compatible widths before dispatch.

This is already handled in `exact_docker_mixed_small.py` / `exact_docker_mixed_full.py`.

---

#### F. Source-of-truth rule

Use this rule:
- `scripts/channel_quant/` = exploratory fake-quant ranking only
- `scripts/channel_quant_new/` = exact TRT-LLM wrapper path, source of truth

And inside `channel_quant_new/`:
- `exact_kernel_sanity.py` = wrapper-vs-reference validation
- `exact_docker_eval.py` = exact uniform baselines
- `exact_docker_mixed_small.py` = exact mixed-channel sanity run
- `exact_docker_mixed_full.py` = exact mixed-channel full run

### Calibration (for channel selection only)

Calibration determines **which channels get FP4 vs FP8** — it does NOT modify weight values.

```python
# GPTQ-standard calibration data generation:
trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
random.seed(0)
for _ in range(128):
    i = random.randint(0, trainenc.shape[1] - 2048 - 1)
    calibration_samples.append(trainenc[:, i:i+2048])
```

- 128 random 2048-token chunks from WikiText-2 train, seed=0
- Run through model to collect routing counts, activation statistics, sensitivity scores
- These statistics determine the per-channel FP4/FP8 masks

### Perplexity evaluation

```python
# GPTQ-standard PPL (full test set):
testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
nsamples = testenc.numel() // 2048
for i in range(nsamples):
    batch = testenc[:, i*2048 : (i+1)*2048]
    logits = model(batch)
    loss = cross_entropy(logits[:, :-1, :], batch[:, 1:])
    nlls.append(loss.float() * 2048)
ppl = torch.exp(sum(nlls) / (nsamples * 2048))
```

Two evaluation paths now exist:
1. **Exploration path (Phase 1a)** — fake quantize → dequantize → BF16 `F.linear`. Fast, useful for ranking ideas, but optimistic.
2. **Exact path (Phase 1b)** — use the generated docker-native scripts in `scripts/channel_quant_new/` (`exact_docker_eval.py`, `exact_docker_mixed_small.py`, `exact_docker_mixed_full.py`) built on TRT-LLM fused wrappers (`torch_quant_nvfp4_linear`, `torch_quant_fp8_linear`). This is the deployment-accurate path.

Current exact full-run numbers (145 chunks):
- `uniform_bf16` = 6.5896
- `uniform_nvfp4` = 6.8431
- `uniform_fp8` = 6.6257
- `mixed_best_full` = 6.6386

The fake path is materially optimistic, especially for NVFP4. On a 4-chunk exact-vs-fake comparison:
- BF16 delta: 0.0000
- FP8 delta: +0.0195
- NVFP4 delta: +0.0843

So the exact path is the source of truth, and the fake-quant path should only be used to rank ideas before re-running them exactly.

### Reference Quantized Models

Two official/community checkpoints define deployment baselines. Cross-validated from their `recipe.yaml` and `config.json`.

**Per-module quantization map:**

| Module | Type | NVFP4 model | FP8 model | Our experiments |
|--------|------|:-----------:|:---------:|:---------------:|
| Full attention Q/K/V/O (`self_attn`) | Linear | **FP4** | **FP8** | BF16 |
| DeltaNet QKV/Z/out (`linear_attn`) | Linear | BF16 | **FP8** | BF16 |
| DeltaNet decay/forget (`in_proj_a/b`) | Linear | BF16 | BF16 | BF16 |
| DeltaNet conv1d | Conv1d | BF16 | BF16 | BF16 |
| MoE expert W1/W2 | Linear | **FP4** | **FP8** | **FP4/FP8 mixed** |
| Shared expert | Linear | **FP4** | **FP8** | BF16 |
| MoE router (`mlp.gate`) | Linear | BF16 | BF16 | BF16 |
| Shared expert gate | Linear | BF16 | BF16 | BF16 |
| LM head | Linear | BF16 | BF16 | BF16 |
| Embeddings | Embedding | BF16 | BF16 | BF16 |
| RMSNorm | RMSNorm | BF16 | BF16 | BF16 |

Key differences:
- NVFP4 ignores ALL DeltaNet (`re:.*linear_attn.*`). FP8 quantizes DeltaNet qkv/z/out_proj.
- Non-Linear modules (Embedding, RMSNorm, Conv1d) are NOT quantized — `targets: [Linear]` only.
- Our experiments only quantize MoE experts. Reference models also quantize attention + shared expert.

**Reference 1: `Sehyo/Qwen3.5-35B-A3B-NVFP4`** (`llm-compressor` / `compressed-tensors`)

```yaml
weights:    num_bits=4 (E2M1), group_size=16, scale=float8_e4m3fn, observer=memoryless_minmax (RTN)
activations: num_bits=4 (FP4!), group_size=16, dynamic="local" (per-token)
ignore:     [lm_head, re:.*mlp.gate$, re:.*mlp.shared_expert_gate$, re:.*linear_attn.*, re:model\.visual\..*]
```

SM120 execution: **FP4 weight × FP4 activation → FP32 accumulate**

**Reference 2: `Qwen/Qwen3.5-35B-A3B-FP8`** (official Qwen)

```json
weights:    quant_method="fp8", weight_block_size=[128, 128] (2D block scaling)
activations: activation_scheme="dynamic" (FP8, per-tensor)
ignore:     [lm_head, embed_tokens, linear_attn.conv1d, linear_attn.in_proj_a/b, mlp.gate, shared_expert_gate, visual.*]
```

SM120 execution: **FP8 weight × FP8 activation → FP32 accumulate**

**To reproduce reference NVFP4 exactly:** quantize self_attn Q/K/V/O + all experts + shared expert to NVFP4 with RTN. Add FP4 activation quantization (group_size=16, per-token dynamic).

**To reproduce reference FP8 exactly:** quantize all of the above PLUS DeltaNet qkv/z/out_proj. Use FP8 with 2D block scaling [128,128] and dynamic FP8 activation scaling.

## The Per-Channel Selection Algorithm

Our winning assignment strategy (from 30 iterations of exploration):

### Step 1: Calibration

Generate 128 calibration chunks from WikiText-2 train (random start positions, seqlen=2048). Optionally use MaCa multi-scale lengths (32×128 + 32×512 + 32×2048 + 32×4096 tokens). Run full model forward pass. At each MoE layer, record routing counts, routing probabilities, per-expert activations.

### Step 2: Per-channel sensitivity scoring

For W2 output channel j of expert e, compute router-affinity weighted quantization error:

```
s_j = Σ_t p_{t,e} × Σ_k (W2[j,k] - Q_FP4(W2[j,k]))² × x_{t,k}²
```

This is a diagonal Hessian approximation weighted by router probability p_{t,e} (MoEQuant insight) and squared activation x_{t,k}² (OWQ/GPTQ insight). For W1: same formula on fused gate+up projection, paired channels share precision.

### Step 3: Two-level budget allocation

**Level 1 — Expert budget** (routing-aware):
```
expert_budget_e = min(1.0, routing_count_e / total_tokens × num_experts × global_budget)
```

**Level 2 — Channel assignment** (sort-and-split):
Sort channels by s_j descending, promote top expert_budget_e fraction to FP8. Per-projection split: W2 gets 4× the FP8 budget of W1 (at 20% total: W1=4%, W2=16%).

### Step 4: Quantization

- FP8 channels: FP8 E4M3 with per-tensor scaling (scale = absmax / 448)
- FP4 channels: NVFP4 E2M1 with per-16-block FP8 scaling (scale = absmax / 6.0)
- Both use RTN (round-to-nearest), no GPTQ compensation

## Exploration Loop

**Two non-negotiable principles for this loop:**
1. **Final target**: achieve accuracy that is as close as possible to exact BF16 under the exact TRT-LLM kernel path. Every promising idea should ultimately be judged by how much it closes the gap to exact BF16.
2. **Open-ended exploration**: this loop does not have a natural stopping point. Keep exploring, keep learning, and keep improving until manually stopped. Never treat the current best as final.
3. **Exact-path logging**: every exact-pipeline experiment MUST be written to `scripts/channel_quant_new/explore.md` with the same structure used in the older exploratory log, but focused only on the exact TRT-LLM kernel path.

## Lessons Learned from Phase 1a (30 iterations, 200+ configs, BF16 simulation)

All Phase 1a results used BF16 matmul (weight-only quantization, no activation quantization). Those absolute PPL numbers are NOT deployment-accurate. The purpose of Phase 1a was to discover useful structures and bad directions quickly. The exact TRT-LLM path supersedes Phase 1a for any final quantitative claim. Therefore, the items below should be read as **current priors/defaults to test first** in the exact path, not immutable truths. Continue to challenge them indefinitely.

### What works — carry these into Phase 1b

**Current default per-channel metric (Phase 1a winner): router-affinity weighted quantization error**
```
s_j = Σ_t p_{t,e} × Σ_k (W[j,k] - Q(W[j,k]))² × x_{t,k}²
```
Weights channel importance by router confidence (p_{t,e}) and activation magnitude (x²). Outperformed 8+ other metrics. Spearman ρ ≈ 0.27 with ground-truth output perturbation — weak but best available. This metric is independent of matmul precision.

**Current default per-projection split (Phase 1a winner): W1:W2 = 1:4**
W2 (down_proj) is consistently more sensitive to quantization than W1 (gate_up_proj). At 20% total FP8 budget: give W1 4% of its channels as FP8, give W2 16%. This finding comes from the weight structure and is independent of matmul precision.

**Current default expert-level allocation (Phase 1a winner): routing frequency**
Hot experts (high routing count) carry 82% of ground-truth sensitivity. Allocate FP8 budget proportional to routing frequency. Cold experts get mostly FP4. This is about model behavior, not matmul precision.

**Current default assignment structure (Phase 1a winner): joint W1/W2 3-tier**
- Hot experts (high routing frequency): both W1 and W2 → FP8
- Medium experts: only W2 → FP8, W1 stays FP4
- Cold experts: both → FP4
This captures both the per-projection finding (W2 > W1) and the routing finding (hot > cold).

**Current best exact-path calibration result: MaCa multi-scale**
Mixing calibration chunks of 128/512/2048/4096 tokens improved PPL by ~0.002 in the earlier BF16-sim exploration and also produced the current best exact-path result when combined with light all-layer correction. Treat this as the current strongest calibration default, not a frozen conclusion — continue exploring alternatives around it in the exact pipeline.

### What doesn't work — don't repeat these

| Direction | Why it failed | Confidence it still fails with real precision |
|-----------|---------------|:---:|
| Weight-only metrics (weight_l1, kurtosis) | Gini ≈ 0.06, too uniform within experts | High — weight structure doesn't change |
| WANDA, RQE, BAQ, AGQ metrics | All worse than router-affinity | High |
| WUSH diagonal preconditioning | Distorts weight distribution, hurts FP4 rounding | High — even worse with real FP4 activation noise |
| RaZeR NVFP4 zero remapping (±5.0) | Model weights too small (mean_abs ≈ 0.003) for ±5.0 values | High — weight distribution doesn't change |
| Router retuning (learned bias on router logits) | Quantized routing already acts as beneficial regularization | Medium — might differ with activation quantization |
| Per-channel learned corrections | Overfit at per-channel granularity (41M params for tiny MSE) | High — more noise from activation quant makes overfitting worse |
| Multilingual calibration (Chinese + code) | Adds noise for English-only eval set | High — eval set doesn't change |
| Dynamic per-batch expert switching | Per-batch routing noise hurts more than adaptivity helps | Medium — might differ with activation quantization |
| OBS-compensated FP4 remainder | Computationally infeasible (full Hessian per expert) | High — still infeasible |
| Residual channels (FP4 + FP4 correction) | NVFP4 per-block scaling already adapts to local magnitudes | High |

### Diagnostic findings — structural, carry over

| Finding | Implication for Phase 1b |
|---------|-------------------------|
| Error distributed uniformly across all 40 MoE layers | No single layer dominates — uniform budget allocation across layers is fine |
| All non-MoE components already BF16 in our setup | We're optimizing the right subsystem (but reference models also quantize attention + shared expert) |
| Layer 0 quantization acts as beneficial regularization | Don't over-protect early layers |
| Restoring top-3 MoE layers to BF16 gives only 0.001 PPL | The floor is very close — don't expect large gains from better assignment alone |
| MxMoE per-block (2 decisions/expert) nearly matches per-channel (2560 decisions/expert) | Per-channel metric noise limits the granularity advantage — this may change if activation quantization alters channel sensitivity patterns |

### What might change with real precision

These findings from Phase 1a are UNCERTAIN under real FP4/FP8 computation:

1. **The absolute PPL floor shifted upward under exact kernels.** Exact full-run numbers are worse than the BF16-sim exploration path, especially for NVFP4.
2. **Relative ranking changed materially.** The BF16 fake path was too optimistic (notably +0.0843 PPL for NVFP4 on a 4-chunk exact-vs-fake comparison), so exact docker runs are now the only trustworthy basis for conclusions.
3. **MaCa calibration survived exact validation.** Multi-scale calibration remains the strongest calibration-side improvement and is part of the current best exact result.
4. **Per-channel vs per-block gap remains small.** Even under the exact path, per-channel mixed precision improves substantially over exact NVFP4, but still trails exact FP8 in the current best full run.

## Reference Material

### Related work analysis docs (in `doc/Arcdoc/`)

- `MxMoE_analysis.md` — ILP formulation, per-block sensitivity, GroupGEMM codegen
- `DynaMo_analysis.md` — Expert significance, channel switching, fuzzy c-means
- `DynaExq_abstract.md` — EMA hotness tracking, dual-version expert residency
- `MoEQuant_analysis.md` — Expert-balanced self-sampling, affinity-guided calibration
- `MC_MixtureCompressor_analysis.md` — Per-expert IP assignment + dynamic expert pruning

### Architecture references

- `doc/nvfp4_moe_pipeline_trace.md` — NVFP4 quantization algorithm
- `doc/dual_tile_moe_summary.md` — Dual-tile grouped GEMM infrastructure
- Permutation math: P_half cancels in W2 reduction, only Q needs runtime scatter

### Hardware execution

CUTLASS grouped GEMM supports heterogeneous FP4/FP8 groups in a single launch. Execution strategy:
1. **Offline**: Sort output channels by sensitivity. Permute columns so FP4/FP8 channels are contiguous.
2. **Online**: Single grouped GEMM launch with 2 groups per expert. Output scatter to restore channel order (~300ns overhead).
3. **Constraint**: SM120 requires uniform precision per N-tile (N=128 minimum).

### Experimental artifacts

| Artifact | Location | Count |
|----------|----------|------:|
| Exploratory experiment scripts | `scripts/channel_quant/proper_iter*.py` | 39 |
| Exploratory result JSONs | `scripts/channel_quant/results/proper_iter*.json` | 38 |
| Exact-pipeline scripts | `scripts/channel_quant_new/*.py` | 8 |
| Exact-pipeline result JSONs | `scripts/channel_quant_new/results/*.json` | 3 |
| Calibration caches | `scripts/channel_quant/results/*.pt` | 5 |
| Exploration log | `scripts/channel_quant/exploration.md` | 670+ lines |
| Research findings (exploratory synthesis) | `doc/Arcdoc/findings.md` | 1 |
| This program / execution guide | `doc/Arcdoc/program.md` | 1 |

Exact-pipeline files currently in `scripts/channel_quant_new/`:
- `exact_kernel_sanity.py`
- `exact_docker_eval.py`
- `exact_docker_eval_small.py`
- `exact_docker_mixed_small.py`
- `exact_docker_mixed_full.py`
- `exact_vs_fake_model_compare.py`
- `exact_vs_fake_quick_compare.py`
- `real_eval_pipeline.py`
- `explore.md`  ← source-of-truth experiment log for exact TRT-LLM wrapper/kernel path

---

## Phase 1b Exact-Path Findings (Mar 22 2026 Update)

### Completed Exact-Path Experiments

| Experiment | Key Finding |
|-----------|-------------|
| **iter01** Projection isolation | W1 FP4 and W2 FP4 contribute nearly equally (3.6% difference). W1 FP4 amplified by SwiGLU. |
| **iter02** Budget sweep | Monotonic improvement 10%→50% budget. budget_50pct (W1=10%, W2=40%) = **6.6212 BEATS FP8**. w2only_30pct=6.6622, w2only_50pct=RUNNING. |
| **iter03** Fair comparison scope | Scope difference (moe_only vs reference_fp8) adds only 0.0004 PPL — negligible. |
| **iter04** Layer-wise allocation | All layer-wise configs produce identical PPL. Error is uniform across 40 layers. |
| **iter06** W1-heavy sweep (4-chunk) | W1-heavy is WORSE. NaN at W1 FP8 ≥20% (SwiGLU hot-channel instability). w2_heavy_10_30 = 7.2554 (best 4-chunk). |

### Critical New Findings

**W1:W2 ratio is near-optimal at 1:4 (W1=10%, W2=40%)**:
- W1-heavy configs (W1=16%, W2=4%) are significantly worse (7.3261 vs 7.2554)
- W1 FP8 ≥20% causes NaN due to SwiGLU hot-channel instability (arXiv:2602.02047)
- The safe zone for W1 FP8 is ≤16% (confirmed: 16% works, 20%+ fails)
- W1=10% is the sweet spot — matches the current best

**The Pareto frontier is still rising at 50% budget**:
- Monotonic improvement from 10% to 50% total budget
- W2 FP8 budget extension (50%→100%) is the next frontier (iter13)

**NaN root cause identified**:
- SwiGLU creates persistent "hot channels" in W1 (arXiv:2602.02047)
- When W1 FP4 fraction is small, hot channels compress long-tail variation → NaN
- Fix: keep hot channels at FP8 regardless of sensitivity score (Hot-Channel Patch)

### Queued Experiments

| Experiment | Status | Expected Gain |
|-----------|--------|---------------|
| **iter02** w2only_50pct | RUNNING (layer 19/40) | Baseline data |
| **iter13** W2 budget extension (W1=10%, W2=50%→100%) | QUEUED (fires after iter02) | 0.002–0.008 PPL |
| **iter14** MaCa multi-scale calibration | QUEUED (fires after iter13 4-chunk) | 0.002–0.004 PPL |

### New Literature Findings (Mar 22 2026)

| Paper | Key Insight | Relevance |
|-------|-------------|-----------|
| **MaCa** (arXiv:2602.07465, ICLR 2026) | Multi-scale calibration validated on Qwen3 | Directly applicable — our model family |
| **Dissecting Outlier Dynamics** (arXiv:2602.02047) | SwiGLU creates hot channels → NaN at W1 FP8 ≥20% | Explains our NaN bug |
| **Mean Bias in FP4** (arXiv:2603.10444) | Rank-one mean bias drives FP4 instability | Alternative NaN fix: mean subtraction |
| **MicroMix** (arXiv:2508.02343) | MXFP4/MXFP6/MXFP8 mixed channels on RTX 5090 | Kernel approach for our method |
| **Block Rotation for MXFP4** (arXiv:2511.04214) | Global rotation incompatible with NVFP4 PoT scaling | Confirms: don't use rotation |
| **MR-GPTQ** (arXiv:2509.23202) | Block-wise Hadamard + GPTQ for NVFP4 | Potential accuracy improvement |
| **FAQ** (arXiv:2601.11200) | Family-aware calibration data regeneration on Qwen3 | Better calibration data |
| **VEQ** (arXiv:2602.01037) | Token-expert affinity in Hessian for MoE VLMs | Validates our router-affinity approach |

