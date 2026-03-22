# MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design

**Technical Analysis Document**
*Prepared for TRT-LLM Dual-Tile MoE Project Reference*

---

## 1. Paper Metadata

| Field | Value |
|---|---|
| **Title** | MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design |
| **Authors** | Haojie Duanmu, Xiuhong Li, Zhihang Yuan, Size Zheng, Jiangfei Duan, Xingcheng Zhang, Dahua Lin |
| **Affiliations** | Shanghai Jiao Tong University; Shanghai AI Laboratory; Peking University; ByteDance Seed; The Chinese University of Hong Kong |
| **Venue** | Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), Vancouver, Canada. PMLR 267, 2025 |
| **arXiv ID** | arXiv:2505.05799v1 [cs.LG], 9 May 2025 |
| **Correspondence** | Haojie Duanmu `<duanmuhaojie@sjtu.edu.cn>` |
| **Code** | Not explicitly linked in the paper text; check the authors' GitHub pages |

---

## 2. Problem Statement

### The Deployment Challenge for MoE Models

Mixture-of-Experts (MoE) models have become the dominant architecture for frontier LLMs, but their deployment is severely constrained by memory. DeepSeek-V3's 671B parameters, for instance, exceed the capacity of eight H100 GPUs in standard FP16 configurations. Quantization is the natural remedy, but naive uniform quantization applied to MoE models leaves significant accuracy and performance on the table.

The core tension is this: MoE models are not homogeneous. Different experts within a single MoE block have wildly different quantization sensitivities, and different experts are activated at wildly different frequencies depending on the input. A uniform bitwidth assignment ignores both of these facts.

### Why Existing Methods Fail

Prior mixed-precision quantization work was designed for dense LLMs and does not transfer cleanly to MoE for two reasons.

**Accuracy side.** Uniform quantization treats all experts and all linear blocks within each expert identically. But as the paper demonstrates empirically (Fig. 1a), Expert 40 in a DeepSeek-V2-Lite block degrades far more under quantization than Expert 37, and within Expert 40, the `Down proj` linear block is far more sensitive than the `Gate proj`. Assigning the same bitwidth to all of them wastes bits on insensitive blocks and starves sensitive ones.

**System side.** Even when mixed-precision schemes are chosen correctly, existing kernels cannot execute them efficiently. HQQ-Aten(W4) performs non-fused dequantization, achieving only `0.11x` the throughput of FP16 CUTLASS. VLLM-Marlin-MoE invokes the Marlin kernel sequentially per expert, destroying GPU utilization and reaching only `0.41x` FP16 throughput. These problems compound when different experts use different precisions, because no existing kernel handles heterogeneous Group-GEMM efficiently.

The paper's thesis is that accuracy and performance must be co-designed: the bitwidth allocator must be aware of hardware execution costs, and the kernel must be aware of the mixed-precision assignment.

---

## 3. Key Insight / Core Idea

The central observation is that MoE blocks exhibit two orthogonal forms of heterogeneity that existing quantization frameworks ignore entirely.

The first is **sensitivity heterogeneity**. Within a single MoE block, different linear blocks (Gate, Up, Down projections) across different experts have different quantization loss profiles. This is not a minor effect. The paper shows that the Down projection of a frequently-activated expert can be orders of magnitude more sensitive to quantization than the Gate projection of a rarely-activated expert. This means the right granularity for bitwidth assignment is the individual linear block within each expert, not the expert as a whole and certainly not the entire MoE layer.

The second is **activation-frequency heterogeneity**. Expert activation frequencies vary by more than 15x within a single MoE block (Fig. 1b shows a 15.3x ratio for DeepSeek-V2-Lite on HumanEval-X). This matters for performance because the arithmetic intensity of a GEMM depends on how many tokens are routed to that expert. A frequently-activated expert processes many tokens and is compute-bound; a rarely-activated expert processes few tokens and is memory-bound. These two regimes favor different quantization schemes: weight-only quantization (e.g., W4A16) reduces memory bandwidth pressure and wins in the memory-bound regime, while weight-activation quantization (e.g., W8A8 or W4A4) accelerates compute via low-precision arithmetic and wins in the compute-bound regime.

The roofline analysis on RTX 4090 makes this concrete. For a GEMM with shape `[m, n, k]` where `n, k >> m`, arithmetic intensity simplifies to approximately `A = m` (the token count). The crossover points are: W4A16 outperforms W8A8 when `A < 83`, and W2A16 outperforms W4A4 when `A < 42`. So the optimal quantization scheme for each expert is a function of how many tokens it receives, which is a runtime-dependent quantity that must be estimated from activation statistics.

MxMoE's core idea is to jointly optimize over these two dimensions using an Integer Linear Program (ILP) that takes sensitivity, activation frequency, and hardware cost as inputs, then automatically generate a fused mixed-precision Group-GEMM kernel that can execute the resulting heterogeneous assignment efficiently.

---

## 4. Technical Approach — Step by Step

### 4.1 System Overview

MxMoE has two tightly coupled components: a **hardware-aware bitwidth allocator** and a **specialized mixed-precision Group-GEMM computation engine**. The workflow proceeds in three phases:

1. **Offline profiling.** Collect quantization loss statistics per linear block, expert activation frequency distributions, and per-tile runtime costs for each candidate quantization scheme on the target hardware.
2. **Allocation.** Solve the ILP to assign a quantization scheme and tile configuration to each linear block in each expert, subject to a memory budget.
3. **Kernel generation.** Auto-generate a fused mixed-precision Group-GEMM kernel from configurable micro-kernels, with a precision-aware tile scheduler.

At runtime, the tile scheduler maps heterogeneous tiles to SMs in a load-balanced way.

### 4.2 Quantization Sensitivity Analysis

**What is measured.** For each linear block `(i, j)` (expert `i`, linear block `j`) and each candidate quantization scheme `k`, MxMoE measures the **perturbation coefficient** `Δ_{i,j,k}`, defined as the L2 norm of the change in the MoE block's output when that single linear block is quantized:

$$\Delta_{i,j,k} = \| \hat{O} - O \|_2 \tag{6}$$

where $O$ is the full-precision MoE block output and $\hat{O}$ is the output when only linear block $(i, j)$ is quantized with scheme $k$, with all other blocks remaining in full precision.

**How it is computed.** A calibration set of 128 sequences, each of length 4096, drawn from the WikiText2 training set is used. For each linear block, each candidate scheme is applied in isolation and the output perturbation is measured across the calibration samples. This is done sequentially per block, so the total calibration cost scales as $O(E \cdot N \cdot |S|)$ forward passes through the MoE block, where $E$ is the number of experts, $N$ is the number of linear blocks per expert (typically 3: Gate, Up, Down), and $|S|$ is the number of candidate schemes.

**Candidate schemes.** The paper uses the following set $S$:
- `w4a16 g128 asym` (4-bit weight, 16-bit activation, group size 128, asymmetric)
- `w8a8 g-1 sym` (8-bit weight, 8-bit activation, per-tensor symmetric)
- `w4a4 g-1 sym` (4-bit weight, 4-bit activation, per-tensor symmetric)
- `w2a16 g128 asym` (2-bit weight, 16-bit activation, group size 128, asymmetric)

**Preprocessing.** Before quantization, a randomized Hadamard transformation is applied to model weights (QuaRot-style incoherence processing) to reduce activation outliers. Online rotations were disabled in practice because they failed on some models (e.g., DeepSeek-V2-Lite) due to shape constraints. Weight quantization uses GPTQ on the same calibration set. Activations are quantized dynamically at runtime.

**Limitation acknowledged.** The sensitivity statistics can be inaccurate because they measure per-layer loss rather than cross-layer loss. Inter-layer dependencies mean that a block that appears insensitive in isolation may compound errors across layers. The authors note this may explain the weaker 3.25-bit result on Qwen2-MoE and suggest cross-layer loss estimation as future work.

### 4.3 Mixed-Precision Assignment: The ILP Formulation

This is the core algorithmic contribution. The goal is to assign a quantization scheme to each linear block in each expert such that the joint objective of accuracy loss and execution time is minimized, subject to a memory budget.

#### Decision Variables

$$x_{i,j,k} \in \{0, 1\}$$

Binary variable: 1 if scheme $k$ is assigned to linear block $j$ of expert $i$, 0 otherwise.

$$y_{i,j,k,t} \in \{0, 1\}$$

Binary variable: 1 if tile configuration $t$ is selected for linear block $(i, j)$ under scheme $k$, 0 otherwise.

#### Quantization Loss Term

The total quantization loss for one MoE block is:

$$L = \sum_{i=1}^{E} \sum_{j=1}^{N} \sum_{k=1}^{|S|} \Delta_{i,j,k} \cdot x_{i,j,k} \tag{5}$$

where $\Delta_{i,j,k}$ is the perturbation coefficient from Eq. (6).

#### Runtime Cost Term

Given token distributions and expert activation frequencies, the GEMM shape for each linear block is derived from per-expert token allocations. Each GEMM is decomposed into tiles mapped onto SMs. For each quantization scheme, candidate tile configurations are profiled ahead of time.

The true execution time is $T = \max_{p=1}^{P} T_p$ where $P$ is the number of SMs. Because the total tile count typically far exceeds the number of SMs, MxMoE approximates this as:

$$T = \frac{1}{P} \sum_{i=1}^{E} \sum_{j=1}^{N} \sum_{k=1}^{|S|} \sum_{t=1}^{|T|} c_{i,j,k,t} \cdot y_{i,j,k,t} \cdot x_{i,j,k} \tag{approx.}$$

where $c_{i,j,k,t}$ is the pre-profiled execution time of linear block $(i, j)$ under scheme $k$ with tile configuration $t$.

#### Full ILP (Equation 7)

The complete optimization problem for a single MoE block is:

$$\min_{x, y} \quad L^r \cdot T^{1-r} \tag{7}$$

subject to:

$$x_{i,j,k} \in \{0, 1\} \quad \forall i, j, k$$

$$\sum_{k=1}^{|S|} x_{i,j,k} = 1 \quad \forall i, j \quad \text{(exactly one scheme per linear block)}$$

$$y_{i,j,k,t} \in \{0, 1\} \quad \forall i, j, k, t$$

$$\sum_{t=1}^{|T|} y_{i,j,k,t} = 1 \quad \forall i, j, k \quad \text{(exactly one tile config per scheme)}$$

$$\sum_{i=1}^{E} \sum_{j=1}^{N} \sum_{k=1}^{|S|} W_{i,j,k} \cdot x_{i,j,k} \leq M \quad \text{(memory budget)}$$

where:
- $r \in [0, 1]$ is a hyperparameter balancing accuracy ($r=1$) and performance ($r=0$)
- $W_{i,j,k}$ is the memory cost of assigning scheme $k$ to linear block $(i, j)$
- $M$ is the total memory budget for the MoE block

**Global objective.** For an $M$-layer MoE model, the global objective is:

$$\min \left( L(W, X) - L(W_q, X_q) \right)^r \cdot \left( \sum_{i}^{M} T_i \right)^{1-r} \tag{3}$$

This is decomposed per-block (Eq. 4) and solved independently for each MoE block, which is a practical approximation that makes the problem tractable.

**Hyperparameter $r$.** The paper uses $r = 0.75$ for all weight-activation quantization experiments (W5A5 setting), which yields significant performance gains with minimal accuracy loss. For extremely low-bit weight-only quantization (W2.25A16, W3.25A16), $r = 1$ is used, prioritizing accuracy because the performance gains from mixed precision are less critical at those bitwidths.

### 4.4 GroupGEMM Kernel Generation

After the ILP assigns a scheme to each linear block, MxMoE must execute a Group-GEMM where different groups (experts) use different quantization schemes. This is the system contribution.

#### The Core Challenge

Two constraints make this hard. First, different quantization schemes have different optimal tile sizes and warp layouts. A W4A16 kernel prefers small tiles (e.g., BM=16, BN=128, BK=64) because it is memory-bound; a W8A8 kernel prefers large tiles (e.g., BM=128, BN=128, BK=64) because it is compute-bound. Fusing them naively into one kernel forces a compromise that hurts both. Second, the number of possible precision combinations grows factorially: with 5 candidate schemes, there are $5! = 120$ possible pairwise combinations, making handcrafted kernels for every combination impractical.

#### Micro-Kernel Specialization

MxMoE implements each quantization scheme as a **configurable CTA-level micro-kernel** expressed as a CUDA device function. Key properties:

- **CTA-index independence.** Each micro-kernel is written to be independent of the CTA index, enabling horizontal fusion: multiple micro-kernels can be composed into a single CUDA kernel where the tile scheduler routes each CTA to the appropriate micro-kernel based on which expert and linear block it is computing.
- **Template parameterization.** Resources (tile sizes, warp counts, shared memory layout) are specified via C++ template parameters, allowing compile-time specialization.
- **Hand-tuned memory access.** Each micro-kernel has memory access patterns optimized for its specific quantization scheme. For example:
  - The `W2A16` micro-kernel fuses dequantization into the MAC loop and uses bit-manipulation for optimized integer-to-float conversion.
  - The `W4A4-g128` micro-kernel uses multistage software pipelining and must respect 128-element group quantization constraints (which constrains the K-dimension tile size to multiples of 128).

#### Resource Configuration

CUDA requires all CTAs in a kernel to use the same resource configuration (warp count, shared memory size). MxMoE enforces this by:

1. **Warp count consistency.** All fused micro-kernels must use the same warp count. This is a hard constraint on which tile configurations can be fused together. The paper gives a concrete example (Fig. 4):
   - Compatible: W4A16 with `BM:16, BN:128, BK:64, WM:1, WN:4, WK:2` (8 warps) + W8A8 with `BM:128, BN:128, BK:64, WM:2, WN:4, WK:1` (8 warps).
   - Incompatible: W4A16 with `BM:64, BN:128, BK:32, WM:2, WN:2, WK:1` (4 warps) + W8A8 with `BM:128, BN:128, BK:64, WM:2, WN:4, WK:1` (8 warps).

2. **Shared memory.** The kernel allocates shared memory equal to the maximum requirement across all fused micro-kernels.

3. **Slice-K to mitigate waste.** When tile sizes differ significantly across precisions (e.g., W4A16 tiles are much smaller than W8A8 tiles), the smaller-tile micro-kernel would waste shared memory. MxMoE introduces additional K-dimension parallelism (slice-K) for the smaller-tile micro-kernel, partitioning its K dimension into more tiles to better utilize the allocated shared memory and improve warp occupancy.

#### Tile Scheduling

The tile scheduler is auto-generated after the ILP solution is known. It maintains a mapping from CTA index to `(expert, linear_block, scheme, tile_position)` and routes each CTA to the appropriate micro-kernel at runtime.

Tiles have heterogeneous execution times. Scheduling them to minimize makespan (the time until the last SM finishes) is NP-hard in general. MxMoE uses a **greedy heuristic** that prioritizes computationally intensive tiles first. Because the total tile count is typically much larger than the SM count, this greedy approach achieves near-optimal performance in practice while keeping scheduling overhead low. The paper notes that dynamic programming could solve it optimally but is not used.

### 4.5 Calibration and Profiling Steps

The full offline pipeline is:

1. **Activation statistics.** Run the model on the calibration set (128 sequences × 4096 tokens from WikiText2) to collect expert activation frequency distributions.
2. **Sensitivity measurement.** For each linear block $(i, j)$ and each scheme $k \in S$, quantize that block in isolation and measure $\Delta_{i,j,k}$ via Eq. (6).
3. **Tile profiling.** For each candidate scheme and tile configuration, profile single-tile runtime on the target GPU. This is done once per hardware target and reused across models.
4. **ILP solve.** Given $\Delta_{i,j,k}$, $c_{i,j,k,t}$, activation frequencies, and memory budget $M$, solve Eq. (7) to get the assignment $\{x_{i,j,k}^*\}$ and tile configurations $\{y_{i,j,k,t}^*\}$.
5. **Weight quantization.** Apply GPTQ with the same calibration set to quantize weights according to the assigned schemes.
6. **Kernel generation.** Auto-generate the fused Group-GEMM kernel with the precision-aware tile scheduler.

Total calibration time ranges from several minutes to a few hours depending on model size.

---

## 5. Granularity of Quantization

The quantization granularity in MxMoE is **per-linear-block within each expert**. Concretely, for a standard SwiGLU MoE expert with three linear projections (Gate, Up, Down), each of the three projections in each of the $E$ experts gets its own independently assigned quantization scheme.

This is finer than expert-level granularity (one scheme per expert) and coarser than per-channel or per-token granularity. The ablation study (Table 3) confirms that linear-block-level allocation consistently outperforms expert-level allocation:

| Model | Granularity | PPL | Avg-Acc |
|---|---|---|---|
| DeepSeek-V2-Lite | Linear-block | 6.11 | 69.01 |
| DeepSeek-V2-Lite | Expert | 6.32 | 67.88 |
| Qwen1.5-MoE | Linear-block | 6.95 | 67.35 |
| Qwen1.5-MoE | Expert | 6.98 | 67.11 |

Within each linear block, the quantization scheme itself may use group-level quantization for weights (e.g., group size 128 for W4A16) or per-tensor quantization (group size -1 for W8A8). Activations are quantized per-tensor dynamically at runtime.

The concrete example from Table 7 (Qwen1.5-MoE, layer 5, W5A5 allocation) illustrates the diversity: Expert 22 gets `W8A8 g-1 sym` for all three projections (it is rarely activated and/or highly sensitive), while Expert 0 gets `W4A4 g128 sym` for all three (frequently activated, less sensitive). Expert 1 gets `W4A4 g128` for Gate and Up but `W8A8 g-1` for Down, reflecting the Down projection's higher sensitivity to activation quantization.

---

## 6. Static vs. Dynamic Precision Assignment

The precision assignment is **static at deployment time**. The ILP is solved offline using calibration-set activation statistics, and the resulting scheme assignment is baked into the auto-generated kernel. At runtime, the tile scheduler routes each CTA to a fixed micro-kernel based on the pre-computed assignment.

Activations are quantized **dynamically at runtime** (per-tensor, per-token), but the *choice* of which quantization scheme to apply to each linear block is fixed. There is no runtime adaptation based on actual token distributions during inference.

This is a deliberate design choice: the ILP uses expected activation frequencies (from calibration) rather than actual runtime frequencies. The paper does not discuss online adaptation or dynamic scheme switching.

---

## 7. Formulas and Algorithms

### MoE Forward Pass

For expert $e$ with SwiGLU activation:

$$\text{Expert}_e(X_e) = W_e^{\text{down}}\left( \sigma(W_e^{\text{gate}}(X_e)) \odot W_e^{\text{up}}(X_e) \right) \tag{1}$$

Full MoE block output:

$$F = \sum_{e=1}^{E} W_e^{\text{down}}\left( \sigma(W_e^{\text{gate}}(X_e)) \odot W_e^{\text{up}}(X_e) \right) \odot w_e \tag{2}$$

where $\sigma$ is the activation function (SiLU/Swish), $\odot$ is elementwise multiplication, and $w_e$ is the routing weight for expert $e$.

### Uniform Min-Max Quantization

$$\hat{x} = \text{round}\left(\frac{x - x_{\min}}{\Delta}\right) \cdot \Delta + x_{\min}$$

where $x_{\min}$ is the minimum of the quantization range and $\Delta$ is the step size. Rounding is the source of quantization error.

### Perturbation Coefficient

$$\Delta_{i,j,k} = \|\hat{O} - O\|_2 \tag{6}$$

where $O$ is the full-precision MoE block output and $\hat{O}$ is the output when only linear block $(i, j)$ is quantized with scheme $k$.

### Quantization Loss

$$L = \sum_{i=1}^{E} \sum_{j=1}^{N} \sum_{k=1}^{|S|} \Delta_{i,j,k} \cdot x_{i,j,k} \tag{5}$$

### Runtime Cost (Approximation)

$$T \approx \frac{1}{P} \sum_{i=1}^{E} \sum_{j=1}^{N} \sum_{k=1}^{|S|} \sum_{t=1}^{|T|} c_{i,j,k,t} \cdot y_{i,j,k,t} \cdot x_{i,j,k}$$

### Full ILP (Equation 7)

$$\min_{x, y} \quad L^r \cdot T^{1-r}$$

$$\text{s.t.} \quad x_{i,j,k} \in \{0,1\}, \quad \sum_{k} x_{i,j,k} = 1 \quad \forall i,j$$

$$y_{i,j,k,t} \in \{0,1\}, \quad \sum_{t} y_{i,j,k,t} = 1 \quad \forall i,j,k$$

$$\sum_{i,j,k} W_{i,j,k} \cdot x_{i,j,k} \leq M$$

### Global Multi-Layer Objective

$$\min \left( L(W, X) - L(W_q, X_q) \right)^r \cdot \left( \sum_{i=1}^{M} T_i \right)^{1-r} \tag{3}$$

### Arithmetic Intensity Crossover Points (Roofline, RTX 4090)

For GEMM shape $[m, n, k]$ with $n, k \gg m$, arithmetic intensity $A \approx m$:

- W4A16 outperforms W8A8 when $A < 83$ (i.e., fewer than 83 tokens routed to that expert)
- W2A16 outperforms W4A4 when $A < 42$

### Algorithm: MxMoE Offline Allocation

```
Input:  MoE model, calibration set C, candidate schemes S,
        memory budget M, hardware profile H, hyperparameter r
Output: Scheme assignment {x*_{i,j,k}}, tile configs {y*_{i,j,k,t}}

1. Apply randomized Hadamard transformation to model weights

2. For each MoE block b:
   a. Run calibration set C through block b in full precision → O
   b. For each expert i = 1..E:
      For each linear block j = 1..N:
         For each scheme k in S:
            Quantize block (i,j) with scheme k, keep others full precision
            Run calibration set → Ô
            Δ_{i,j,k} = ||Ô - O||_2

3. Collect expert activation frequencies from calibration set
   → derive expected GEMM shapes per linear block

4. For each scheme k and tile config t:
   Look up pre-profiled single-tile runtime c_{i,j,k,t} from H

5. Solve ILP (Eq. 7) for each MoE block:
   min L^r * T^(1-r)
   s.t. one scheme per linear block, memory budget M

6. Apply GPTQ weight quantization using assigned schemes

7. Generate mixed-precision Group-GEMM kernel with precision-aware
   tile scheduler based on {x*_{i,j,k}}, {y*_{i,j,k,t}}
```

### Algorithm: Greedy Tile Scheduler

```
Input:  Tile list T = {(expert, block, scheme, tile_pos, cost)}
        SM count P
Output: Assignment of tiles to SMs

1. Sort T by cost descending (most expensive tiles first)
2. Initialize SM load array load[1..P] = 0
3. For each tile t in T:
   p* = argmin_p load[p]
   Assign t to SM p*
   load[p*] += cost(t)
4. Return assignment
```

---

## 8. Experimental Results

### Models and Hardware

All experiments run on a single **Nvidia RTX 4090** GPU.

| Model | Size | Experts | TopK | Notes |
|---|---|---|---|---|
| Mixtral-8x7B-Instruct-v0.1 | 92.9 GB | 8 | 2 | Fewer experts, smaller mixed-precision design space |
| Qwen1.5-MoE | 26.7 GB | 60+4 | 4 | 60 routed + 4 shared experts |
| Qwen2-MoE-Instruct | 106.9 GB | 64+8 | 8 | 64 routed + 8 shared experts |
| DeepSeek-V2-Lite | 29.3 GB | 64+2 | 6 | First layer is dense MLP, quantized with GPTQ 4-bit |

Attention modules are kept in full precision in all experiments. Only expert computation is counted in throughput measurements.

### Baselines

- **FP16 (CUTLASS):** Full-precision Group-GEMM baseline
- **GPTQ\*:** GPTQ with randomized Hadamard transformation applied (same preprocessing as MxMoE, for fair comparison)
- **QuaRot:** Rotation-based quantization for weight-activation quantization
- **HQQ-Aten(W4):** Non-fused dequantization baseline
- **VLLM-Marlin-MoE:** Sequential per-expert Marlin kernel

### Accuracy Results (Table 1)

Evaluation on 7 benchmarks: Arc-Challenge (AC), Arc-Easy (AE), HellaSwag (HS), LAMBADA-openai (LO), LAMBADA-standard (LS), PIQA (PQ), WinoGrande (WG), plus WikiText-2 perplexity (PPL).

#### DeepSeek-V2-Lite

| Method | Bits (W-A) | AC | AE | HS | LO | LS | PQ | WG | Avg | PPL |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 16-16 | 48.98 | 76.22 | 77.91 | 72.33 | 67.90 | 80.20 | 71.19 | 70.68 | 5.92 |
| GPTQ* | 3.25-16 | 47.35 | 75.04 | 76.44 | 70.41 | 65.65 | 79.05 | 71.27 | 69.32 | 6.18 |
| GPTQ* | 2.25-16 | 37.63 | 63.47 | 65.45 | 52.53 | 48.55 | 74.59 | 64.09 | 58.04 | 8.49 |
| QuaRot | 4-4 | 41.81 | 67.51 | 74.12 | 50.01 | 45.86 | 75.52 | 63.38 | 59.74 | 8.44 |
| **MxMoE** | **3.25-16** | **47.87** | **74.58** | **76.85** | **71.10** | **65.85** | **79.27** | **70.09** | **69.37** | **6.08** |
| **MxMoE** | **2.25-16** | **40.36** | **68.86** | **68.63** | **59.56** | **54.01** | **75.08** | **67.80** | **62.04** | **7.01** |
| **MxMoE** | **5-5** | **46.76** | **74.37** | **77.38** | **68.41** | **64.99** | **79.38** | **69.22** | **68.64** | **6.16** |

At 2.25-bit, MxMoE reduces PPL from 8.49 to 7.01 (1.48 lower). The abstract claims up to 2.4 lower PPL than GPTQ at 2.25-bit across models.

#### Qwen1.5-MoE

| Method | Bits (W-A) | AC | AE | HS | LO | LS | PQ | WG | Avg | PPL |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 16-16 | 44.03 | 69.53 | 77.26 | 71.28 | 64.62 | 80.47 | 69.30 | 68.07 | 6.79 |
| GPTQ* | 3.25-16 | 43.34 | 68.60 | 75.35 | 68.68 | 62.80 | 79.22 | 66.54 | 66.36 | 7.15 |
| GPTQ* | 2.25-16 | 30.89 | 47.14 | 60.77 | 43.72 | 34.81 | 69.97 | 56.20 | 49.07 | 11.19 |
| QuaRot | 4-4 | 27.13 | 40.74 | 57.10 | 35.61 | 25.33 | 66.43 | 51.93 | 43.47 | 18.44 |
| **MxMoE** | **3.25-16** | **43.77** | **66.04** | **75.92** | **69.71** | **62.82** | **79.11** | **68.03** | **66.49** | **7.02** |
| **MxMoE** | **2.25-16** | **31.66** | **53.28** | **62.80** | **56.43** | **51.00** | **71.33** | **61.25** | **55.39** | **8.79** |
| **MxMoE** | **5-5** | **42.92** | **66.04** | **76.27** | **70.06** | **63.40** | **80.58** | **67.80** | **66.72** | **7.01** |

#### Qwen2-MoE-Instruct

| Method | Bits (W-A) | AC | AE | HS | LO | LS | PQ | WG | Avg | PPL |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 16-16 | 55.20 | 77.19 | 84.09 | 74.35 | 62.62 | 82.32 | 72.14 | 72.56 | 5.84 |
| GPTQ* | 3.25-16 | 53.67 | 75.88 | 82.90 | 73.36 | 63.24 | 81.01 | 70.96 | 71.57 | 6.11 |
| GPTQ* | 2.25-16 | 38.82 | 57.66 | 71.27 | 58.99 | 49.72 | 73.29 | 60.30 | 58.58 | 7.98 |
| QuaRot | 4-4 | 33.19 | 42.72 | 54.34 | 23.02 | 9.53 | 63.87 | 50.12 | 39.54 | 110.66 |
| **MxMoE** | **3.25-16** | **53.84** | **76.30** | **82.81** | **72.39** | **60.95** | **81.34** | **69.69** | **71.05** | **6.18** |
| **MxMoE** | **2.25-16** | **45.05** | **68.86** | **77.13** | **66.00** | **56.61** | **75.41** | **62.90** | **64.57** | **7.57** |
| **MxMoE** | **5-5** | **54.86** | **75.55** | **82.69** | **72.87** | **62.68** | **79.49** | **70.96** | **71.30** | **6.25** |

Note: QuaRot completely collapses on Qwen2-MoE at 4-4 (PPL 110.66), showing that uniform weight-activation quantization is catastrophic for this model.

#### Mixtral-8x7B-Instruct-v0.1

| Method | Bits (W-A) | AC | AE | HS | LO | LS | PQ | WG | Avg | PPL |
|---|---|---|---|---|---|---|---|---|---|---|
| FP16 | 16-16 | 66.38 | 85.39 | 85.95 | 77.28 | 73.06 | 85.20 | 76.72 | 78.57 | 3.88 |
| GPTQ* | 3.25-16 | 64.42 | 84.01 | 85.12 | 76.77 | 71.76 | 83.79 | 76.16 | 77.43 | 4.17 |
| GPTQ* | 2.25-16 | 48.89 | 72.35 | 76.95 | 68.39 | 61.44 | 77.15 | 67.72 | 67.56 | 5.69 |
| QuaRot | 4-4 | 50.60 | 68.69 | 75.65 | 40.95 | 38.83 | 76.88 | 61.01 | 58.94 | 9.06 |
| **MxMoE** | **3.25-16** | **64.25** | **84.22** | **85.04** | **76.98** | **71.86** | **84.17** | **75.93** | **77.49** | **4.15** |
| **MxMoE** | **2.25-16** | **48.98** | **72.77** | **77.44** | **68.68** | **62.18** | **76.28** | **68.90** | **67.89** | **5.63** |
| **MxMoE** | **5-5** | **64.08** | **83.71** | **85.10** | **76.21** | **71.78** | **83.79** | **73.80** | **76.92** | **4.20** |

### Throughput Results

**Figure 2 (single MoE block, 60 experts, shape [N,K]=[2816,2048], TopK=4, 512 tokens):**

| Kernel | Throughput | Speedup vs FP16 |
|---|---|---|
| FP16 (CUTLASS) | ~27.72 TFLOPS | 1.00x |
| HQQ-Aten(W4) | 3.14 TFLOPS | 0.11x |
| VLLM-Marlin-MoE(W4) | 11.45 TFLOPS | 0.41x |
| MxMoE(W4) | 74.90 TFLOPS | 2.70x |
| MxMoE(W8A8) | 39.10 TFLOPS | 1.41x |
| MxMoE(mix-precision) | 85.44 TFLOPS | 3.08x |

**Figure 5 (throughput across models and token counts):**

- Memory-bound workload (512 tokens): MxMoE mixed precision achieves **1.6-2.7x** throughput over FP16.
- Compute-bound workload (8192 tokens): MxMoE achieves **3.0-3.4x** throughput over FP16.
- At 512 tokens, W8A8 underperforms both W4A16 and MxMoE(W4.25A15.5) because the workload is memory-bound.
- MxMoE(W4.25A15.5) achieves better accuracy than W4A16 and up to **25% higher throughput** on Qwen1.5-MoE.
- At 8192 tokens, MxMoE(W5A5) achieves up to **29.4% performance improvement** over W8A8 while maintaining comparable perplexity to FP16.

**Table 6 (kernel specialization necessity, W4A4 TOPS on RTX 4090):**

| Kernel | W4A4 per-channel TOPS | W4A4 group-128 TOPS |
|---|---|---|
| Specialized per-channel kernel | 1070.5 | N/A |
| Specialized group-128 kernel | N/A | 667.3 |
| Unified kernel (handles both) | 929.2 | 412.0 |

The unified kernel loses 13% on per-channel and 38% on group-128 compared to specialized kernels, justifying the auto-generation approach.

### Activation Sensitivity Case Study (Table 4)

WikiText-2 PPL for DeepSeek-V2-Lite under RTN quantization at various weight/activation bitwidths:

| W bits \ A bits | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| 4 | 68079 | 41.4 | 11.3 | 9.4 | 8.1 |
| 5 | 12306 | 38.7 | 9.7 | 8.2 | 7.3 |
| 6 | 14252 | 26.3 | 9.2 | 8.2 | 7.2 |
| 7 | 18151 | 34.8 | 9.7 | 8.2 | 7.3 |
| 8 | 19092 | 39.0 | 9.5 | 8.3 | 7.3 |

Dropping activation bits from 5 to 4 causes catastrophic degradation regardless of weight precision. This is attributed to massive outliers in the Down projection's input activations. MxMoE identifies these sensitive blocks and assigns them higher activation precision.

---

## 9. Implementation Details

### Hardware

- **Target GPU:** Nvidia RTX 4090 (Ada Lovelace, SM 8.9)
- FP8 is supported on RTX 4090 but not A100; hardware support constrains the candidate scheme set $S$.
- The paper does not evaluate on H100 or Blackwell.

### Software Stack

- **FP16 baseline:** CUTLASS Group-GEMM
- **Weight quantization:** GPTQ (post-ILP allocation)
- **Incoherence preprocessing:** Randomized Hadamard transformation (QuaRot-style)
- **ILP solver:** Not specified; standard ILP solvers (e.g., GLPK, Gurobi) are implied
- **Kernel generation:** Custom CUDA device functions with C++ template parameterization

### GroupGEMM Kernel Architecture

The kernel is structured as follows:

```
Fused Mixed-Precision Group-GEMM Kernel
├── Tile Scheduler (auto-generated, precision-aware)
│   └── Maps CTA index → (expert_id, linear_block_id, scheme, tile_pos)
├── Micro-kernel: W2A16 (bit-manipulation dequant, fused into MAC)
├── Micro-kernel: W4A16 (dequant fused, small tiles, memory-bound optimized)
├── Micro-kernel: W4A4 per-channel (large tiles, compute-bound optimized)
├── Micro-kernel: W4A4 g128 (multistage pipeline, K-dim constrained)
└── Micro-kernel: W8A8 (large tiles, INT8 tensor core optimized)
```

All micro-kernels share the same warp count (enforced at compile time via template parameters). Shared memory is allocated at the maximum requirement across all fused micro-kernels. Slice-K is applied to smaller-tile micro-kernels to improve utilization.

### Tile Configuration Examples (Figure 4)

Compatible pair (both 8 warps):
```
W4A16: BM=16, BN=128, BK=64, WM=1, WN=4, WK=2  → 8 warps
W8A8:  BM=128, BN=128, BK=64, WM=2, WN=4, WK=1  → 8 warps
```

Incompatible pair (different warp counts):
```
W4A16: BM=64, BN=128, BK=32, WM=2, WN=2, WK=1   → 4 warps
W8A8:  BM=128, BN=128, BK=64, WM=2, WN=4, WK=1  → 8 warps
```

### Example Mixed-Precision Assignment (Table 7, Qwen1.5-MoE Layer 5, W5A5)

A representative sample showing the diversity of assignments:

| Expert | Gate | Up | Down |
|---|---|---|---|
| 0 | W4A4 g128 | W4A4 g128 | W4A4 g128 |
| 1 | W4A4 g128 | W4A4 g128 | W8A8 g-1 |
| 4 | W4A4 g-1 | W4A4 g-1 | W4A4 g128 |
| 22 | W8A8 g-1 | W8A8 g-1 | W8A8 g-1 |
| 44 | W4A4 g-1 | W4A4 g-1 | W4A4 g128 |

Expert 22 receives W8A8 across all projections (high sensitivity or low activation frequency). Experts 4, 14, 44, 54-58 receive W4A4 per-channel (g-1) for Gate and Up but W4A4 g128 for Down, suggesting the Down projection benefits from finer-grained weight quantization in those experts. The Down projection frequently gets higher precision than Gate/Up, consistent with the activation outlier analysis.

---

## 10. Limitations and Gaps

### Acknowledged Limitations

**Sensitivity estimation accuracy.** The perturbation coefficient $\Delta_{i,j,k}$ is measured per-layer in isolation. Inter-layer dependencies mean that errors compound across layers in ways the per-layer metric cannot capture. The authors explicitly note this may explain the weaker 3.25-bit result on Qwen2-MoE and suggest cross-layer loss estimation as future work.

**Online rotation disabled.** QuaRot-style online rotations (applied to activations at runtime) were disabled because they failed on some models (DeepSeek-V2-Lite) due to shape constraints. This limits the effectiveness of activation quantization for those models.

**Hardware scope.** All experiments are on RTX 4090. The candidate scheme set $S$ is hardware-dependent (e.g., FP8 is available on RTX 4090 but not A100; INT4 tensor cores are not available on all GPUs). The paper does not evaluate on H100, A100, or Blackwell.

**Mixtral limited gains.** Mixtral-8x7B has only 8 experts, giving a small mixed-precision design space. The gains over GPTQ are correspondingly modest.

**Static assignment.** The precision assignment is fixed at deployment time based on calibration-set statistics. If actual runtime token distributions differ significantly from calibration (e.g., different tasks, different batch sizes), the assignment may be suboptimal.

**Greedy scheduling.** The tile scheduler uses a greedy heuristic rather than an optimal algorithm. For highly heterogeneous tile cost distributions, this may leave performance on the table.

### Things the Paper Does Not Do

- Does not evaluate on Blackwell (SM120) or H100 (SM90) with FP8 tensor cores
- Does not handle dynamic expert routing changes at runtime
- Does not address the attention module (kept in FP16 throughout)
- Does not evaluate on generation workloads (decode phase with batch size 1), only prefill-style batch inference
- Does not provide a public code release (at time of writing)
- Does not address shared experts (the 4 shared experts in Qwen1.5-MoE and 8 in Qwen2-MoE are mentioned in the architecture table but their quantization treatment is not separately discussed)
- Does not evaluate on models larger than ~107 GB (no DeepSeek-V3 scale experiments)
- Does not compare against FP8 quantization (which is available on RTX 4090 per the paper's own note)

---

## 11. Relevance to Our Work (TRT-LLM Dual-Tile MoE)

### Our Project Context

The TRT-LLM dual-tile MoE project targets Blackwell (SM120) GPUs with:
- M32/M64/M128 tile selection for MoE GEMMs
- SwiGLU epilogue fusion within the MoE kernel
- Heterogeneous MoE tiers (different experts may use different tile sizes)
- TensorRT-LLM integration

### What MxMoE Does That Our Work Doesn't

**Algorithmic precision assignment.** MxMoE has a principled, ILP-based framework for deciding which quantization scheme to assign to each linear block. Our work focuses on tile-size selection for performance but does not (as far as described) include a systematic accuracy-aware bitwidth allocator. MxMoE's ILP formulation with the perturbation coefficient $\Delta_{i,j,k}$ is directly applicable to our setting.

**Sensitivity-aware quantization.** MxMoE explicitly measures per-block quantization sensitivity and uses it to protect sensitive blocks (especially Down projections with activation outliers). This is orthogonal to tile-size selection and could be layered on top of our work.

**Mixed weight-activation quantization.** MxMoE handles both weight-only (W4A16, W2A16) and weight-activation (W8A8, W4A4) quantization and selects between them based on arithmetic intensity. Our dual-tile approach selects tile sizes but presumably within a fixed quantization scheme.

**Roofline-driven scheme selection.** The crossover analysis (W4A16 vs W8A8 at A=83 tokens) is directly relevant to our tile selection: the same arithmetic intensity argument that determines the optimal quantization scheme also determines the optimal tile size. MxMoE makes this explicit.

### What Our Work Does That MxMoE Doesn't

**Blackwell/SM120 support.** MxMoE is evaluated only on RTX 4090 (Ada Lovelace, SM 8.9). Blackwell introduces new tensor core configurations, FP4 support, and different memory hierarchies. Our M32/M64/M128 tile selection is specifically designed for SM120's capabilities.

**SwiGLU epilogue fusion.** MxMoE treats Gate, Up, and Down projections as separate GEMMs. Our work fuses the SwiGLU epilogue (the $\sigma(\text{Gate}) \odot \text{Up}$ operation) into the GEMM kernel, reducing memory traffic. This is a complementary optimization that MxMoE does not address.

**TRT-LLM integration.** MxMoE is a standalone research prototype. Our work integrates into TRT-LLM's production inference stack, including plugin APIs, engine serialization, and multi-GPU support.

**Tile-size heterogeneity within a fixed precision.** Our dual-tile approach selects different tile sizes (M32/M64/M128) for different experts based on token counts, even within a single quantization scheme. MxMoE's tile selection is coupled to its quantization scheme selection; it does not independently optimize tile sizes for a fixed precision.

**Decode-phase optimization.** MxMoE focuses on prefill-style batch inference (512-8192 tokens). Our work likely targets both prefill and decode, where batch size 1 decode is a very different regime.

### Potential Combination Points

**Adopt MxMoE's ILP for bitwidth allocation.** The ILP formulation (Eq. 7) could be integrated into our offline profiling pipeline. Given our tile-size profiling infrastructure, we already have the $c_{i,j,k,t}$ cost estimates needed for the runtime term $T$. Adding the sensitivity measurement $\Delta_{i,j,k}$ would give us a complete accuracy-performance co-design framework.

**Extend the ILP to include tile-size as a decision variable.** MxMoE's ILP jointly selects quantization scheme and tile configuration. We could extend this to also select among M32/M64/M128 tiles independently of the quantization scheme, giving a three-way joint optimization over (scheme, tile-size, epilogue-fusion).

**Apply the roofline crossover analysis to Blackwell.** The arithmetic intensity crossover points (A=83 for W4A16 vs W8A8 on RTX 4090) will be different on SM120 due to different memory bandwidth and compute throughput. Rerunning this analysis on Blackwell would give us the correct thresholds for our tile-size and scheme selection.

**Use MxMoE's micro-kernel composition pattern.** The CTA-index-independent micro-kernel design with template parameterization is a clean pattern for implementing heterogeneous Group-GEMM on Blackwell. Our dual-tile kernel could adopt this structure, with M32/M64/M128 variants as micro-kernels composed by a precision-aware tile scheduler.

**Sensitivity-aware protection of Down projections.** MxMoE's finding that Down projections are consistently more sensitive to activation quantization (due to outliers in their input) is directly actionable: in our quantization scheme, we should assign higher precision or larger group sizes to Down projections by default, even before running the full ILP.

---

## 12. Comparison Table

| Dimension | MxMoE | DynaMo (reference) | Our TRT-LLM Dual-Tile |
|---|---|---|---|
| **Target hardware** | RTX 4090 (SM 8.9) | Not specified | Blackwell SM120 |
| **Precision assignment** | ILP-based, per-linear-block | Dynamic, runtime-adaptive | Fixed (tile-size adaptive) |
| **Granularity** | Per linear block per expert | Per expert or per layer | Per expert (tile size) |
| **Quantization schemes** | W2A16, W4A16, W4A4, W8A8 | Varies | TBD (FP8/INT4 on Blackwell) |
| **Static vs dynamic** | Static (offline ILP) | Dynamic (runtime) | Static (offline profiling) |
| **Kernel approach** | Auto-generated fused Group-GEMM | Custom kernels | Dual-tile Group-GEMM |
| **SwiGLU fusion** | No (separate GEMMs) | Unknown | Yes |
| **Accuracy-aware** | Yes (ILP with sensitivity) | Partially | Not explicitly |
| **Tile-size selection** | Coupled to scheme selection | Unknown | Independent (M32/M64/M128) |
| **Speedup over FP16** | 1.6-3.4x | Unknown | TBD |
| **Accuracy vs GPTQ** | Up to 2.4 lower PPL at 2.25-bit | Unknown | N/A |
| **TRT-LLM integration** | No | No | Yes |
| **Code available** | Not at time of writing | Unknown | Internal |

---

## Key Figures Summary

**Figure 1a** (Quantization sensitivity per expert and per linear block): Shows that Expert 40 degrades far more than Expert 37 under quantization, and within Expert 40, the Down projection is far more sensitive than the Gate projection. This motivates linear-block-level granularity.

**Figure 1b** (Expert activation frequency distribution): Shows a 15.3x variation in activation frequencies within a single MoE block of DeepSeek-V2-Lite on HumanEval-X. This motivates scheme selection based on arithmetic intensity.

**Figure 2** (Throughput comparison): Shows that HQQ-Aten and VLLM-Marlin-MoE actually underperform FP16 CUTLASS, while MxMoE achieves 2.7-3.1x speedup. This is the key system result.

**Figure 4** (Tile configuration compatibility): Illustrates the warp-count constraint for fusing micro-kernels and the slice-K solution for handling tile-size mismatches.

**Figure 5** (Throughput vs token count): Shows the regime-dependent behavior of different quantization schemes and MxMoE's ability to adapt across regimes.

**Figure 6** (Hyperparameter $r$ sweep): Shows the accuracy-performance tradeoff as $r$ varies from 0 to 1, with $r=0.75$ as the recommended operating point.

**Table 4** (Activation sensitivity): The dramatic PPL collapse when activation bits drop from 5 to 4 (e.g., W4A4 PPL 68079 vs W4A5 PPL 41.4) is the empirical justification for protecting Down projection activations.

**Table 7** (Concrete allocation for Qwen1.5-MoE layer 5): Shows the actual per-expert, per-linear-block scheme assignments produced by the ILP, demonstrating the diversity of the solution.

---

*Document prepared: 2026-03-15. Based on arXiv:2505.05799v1.*
