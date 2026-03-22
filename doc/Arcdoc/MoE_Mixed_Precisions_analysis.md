# Technical Analysis: Mixture of Experts with Mixture of Precisions for Tuning Quality of Service

---

## 1. Paper Metadata

| Field | Value |
|---|---|
| **Title** | Mixture of Experts with Mixture of Precisions for Tuning Quality of Service |
| **Authors** | HamidReza Imani, Abdolah Amirany, Tarek El-Ghazawi |
| **Affiliation** | Department of Electrical and Computer Engineering, The George Washington University, Washington, DC, USA |
| **arXiv ID** | 2407.14417 |
| **Year** | 2024 |
| **Venue** | arXiv preprint |
| **Contact** | `{hamidreza, a.amirany, tarek}@gwu.edu` |

---

## 2. Problem Statement

Large MoE models like Mixtral 8x7B reach 94 GB at full 16-bit precision, far exceeding the memory of a single consumer or mid-range GPU. Existing deployment strategies handle this by offloading experts to CPU and transferring them on demand, but CPU-GPU transfers over PCIe become the dominant bottleneck: at 27.35 ms per expert transfer on PCIe Gen4, even a single expert miss stalls the pipeline for tens of milliseconds.

The deeper problem the paper targets is *adaptive* deployment under *changing* constraints. In shared multi-tenant systems, available GPU memory, PCIe bandwidth, and GPU utilization all fluctuate over time. A static deployment plan (e.g., "always use 4-bit") cannot respond to these changes. Users also arrive with heterogeneous preferences: some care about throughput, others about output quality. No prior work offered a single system that could navigate this tradeoff continuously and reconfigure without a full model reload.

Specifically, the paper asks: given a GPU memory budget that may change at runtime, and a user preference for either throughput or quality, how do you assign precisions to experts and decide which experts live on GPU vs. CPU, such that you can serve the model efficiently while meeting the user's quality-of-service (QoS) target?

---

## 3. Key Insight / Core Idea

The central observation is that **non-expert layers and expert layers have asymmetric sensitivity to quantization**. Non-expert layers (attention, layer norms, embeddings) collectively occupy only ~3.16 GB in Mixtral 8x7B (about 5% of total model size) but contribute disproportionately to output quality. Expert layers, by contrast, can absorb aggressive quantization with minimal perplexity impact.

From this asymmetry, the paper derives a clean design principle:

> Keep all non-expert layers in 16-bit on GPU at all times. Apply mixed-precision quantization only to expert layers, varying the number of 4-bit experts as a continuous control knob.

This creates a one-dimensional Pareto frontier: as you increase the number of 4-bit experts, model size shrinks (enabling more experts to fit on GPU, reducing transfer stalls, increasing throughput) while perplexity rises only marginally. The system can slide along this frontier in response to changing memory budgets or user preferences, without touching the non-expert layers at all.

The "quality of service" framing means the system exposes this frontier to a job management layer, which can request a specific throughput target or quality target and receive a concrete deployment plan (how many experts are 4-bit, how many live on GPU).

---

## 4. Technical Approach: Step by Step

### 4.1 Precision Assignment to Experts

The system supports two precision levels for experts: **16-bit (FP16/BF16)** and **4-bit**. Non-expert layers are always 16-bit. The key control variable is `Num_E,4`, the number of experts quantized to 4-bit (out of 256 total in Mixtral 8x7B).

**Assignment policy:** Quantization assignment across experts is **random**. The rationale is that MoE training explicitly encourages load balancing, so all experts should be accessed with roughly equal frequency. A random assignment therefore does not systematically degrade any particular expert more than others, and avoids dependence on a specific dataset or task distribution.

This is a deliberate design choice against importance-based or sensitivity-based assignment. The authors argue that since MoE routing aims for uniform expert utilization, there is no principled basis for preferring one expert over another for quantization.

### 4.2 QoS Tuning Mechanism

"Quality of service" in this paper refers to the tradeoff between two measurable quantities:

- **Throughput** (tokens per second): determined primarily by how many expert misses occur (i.e., how many experts must be transferred from CPU to GPU during inference).
- **Output quality** (perplexity on language modeling benchmarks): determined primarily by how many experts are quantized to 4-bit.

Each incoming task carries a **preference field**: either `Quality` or `Throughput`. The adaptive planner uses this preference, combined with the current GPU memory budget from the job management system, to compute a deployment plan.

**Throughput-preference path:**
1. Maximize the number of experts resident on GPU (minimize expert misses).
2. If `Mem_GPU > Size_N.E. + Num_E × Size_E,4`, there is excess memory after fitting all experts at 4-bit. Use that excess to keep some experts at 16-bit for better quality.
3. Otherwise, all experts are 4-bit and offloading is required.

**Quality-preference path:**
1. Expose a selectable range for `Num_E,4` (from 0 to 256).
2. Upper bound: all experts 16-bit, best quality, largest memory footprint.
3. Lower bound: all experts 4-bit, worst quality, smallest footprint.
4. Given the chosen `Num_E,4` and the GPU memory budget, the planner decides how many experts can reside on GPU and how many must be offloaded.
5. If memory allows more experts on GPU than required by the quality setting, the extra capacity is used to increase the GPU-resident count, improving throughput without changing quality.

### 4.3 Core Formula: Number of 16-bit Experts Under Throughput Preference

When the user prefers throughput and there is excess GPU memory after fitting all experts at 4-bit, the number of experts that can be upgraded to 16-bit is:

$$
Num_{E,16} =
\begin{cases}
\left\lfloor \dfrac{Mem_{GPU} - Size_{N.E.} - Num_E \times Size_{E,4}}{3 \times Size_{E,4}} \right\rfloor & \text{if } Mem_{GPU} > Size_{N.E.} + Num_E \times Size_{E,4} \\
0 & \text{otherwise}
\end{cases}
$$

**Variable definitions:**

| Symbol | Meaning |
|---|---|
| `Mem_GPU` | Available GPU memory (GB) |
| `Size_N.E.` | Total size of non-expert layers (3.16 GB for Mixtral 8x7B) |
| `Num_E` | Total number of experts (256 for Mixtral 8x7B) |
| `Size_E,4` | Size of one expert at 4-bit (336 MB / 4 = 84 MB) |
| `Num_E,16` | Number of experts kept at 16-bit |

The denominator `3 × Size_E,4` reflects the memory delta of upgrading one expert from 4-bit to 16-bit: a 16-bit expert costs 4× as much as a 4-bit expert, so the incremental cost is `3 × Size_E,4`. The floor ensures an integer count.

**Worked example:** With `Mem_GPU = 53.03 GB`, `Size_N.E. = 3.16 GB`, `Num_E = 256`, `Size_E,4 = 84 MB ≈ 0.082 GB`:

- All-4-bit footprint: `3.16 + 256 × 0.082 = 3.16 + 20.99 = 24.15 GB`
- Excess: `53.03 - 24.15 = 28.88 GB`
- `Num_E,16 = floor(28.88 / (3 × 0.082)) = floor(28.88 / 0.246) ≈ floor(117.4) = 117`

So roughly 117 experts can be kept at 16-bit while still fitting everything on GPU.

### 4.4 Expert Placement: CPU vs. GPU

Given the precision assignment, the planner must decide which experts live on GPU and which on CPU. The placement rule is:

> **4-bit experts are prioritized for GPU residency.**

The reasoning: an expert miss (needing to transfer an expert from CPU to GPU) stalls the pipeline. Smaller 4-bit experts have a smaller memory footprint, so more of them fit on GPU, increasing the hit rate. Keeping 4-bit experts on GPU first maximizes the number of experts that can be served without a transfer stall.

16-bit experts that don't fit on GPU are kept in CPU memory and transferred on demand.

### 4.5 Expert State Table

Each expert is tracked via a two-field state entry:

1. **Quantization bit**: is this expert 4-bit or 16-bit?
2. **Location bit**: is this expert currently on GPU or CPU?

At runtime, when the router selects an expert, the system checks the state table. If the expert is on CPU (a "miss"), it transfers the expert to a pre-allocated swap buffer on GPU before computing. The swap buffer is sized to hold at least one expert.

### 4.6 Routing Interaction

The paper makes **no modifications to the routing mechanism**. The standard top-k sparse gating network operates identically regardless of expert precision or location. The precision and placement decisions are entirely orthogonal to routing. This is a key simplicity: the system is a deployment-time wrapper around an unmodified MoE model.

### 4.7 Runtime Reconfiguration

When the job management system signals a change in available GPU memory (or a new task arrives with a different preference), the planner recomputes the deployment plan and performs a **partial reconfiguration**:

- Offload experts from GPU to CPU (if memory budget shrinks).
- Download experts from CPU to GPU (if memory budget grows).
- Re-quantize experts between 4-bit and 16-bit (if the quality/throughput target changes).

Critically, this does not require reloading the full model. Only the affected experts are moved or re-quantized, keeping reconfiguration latency proportional to the number of changed experts rather than total model size.

---

## 5. Granularity of Precision Assignment

The precision assignment operates at **per-expert granularity**. Each of the 256 experts in Mixtral 8x7B (32 layers × 8 experts/layer) is independently assigned either 4-bit or 16-bit precision.

There is no sub-expert granularity (e.g., per-layer-within-expert or per-weight-group). The entire expert feed-forward block is quantized uniformly to the assigned precision using Bitsandbytes NF4 quantization.

Non-expert components (attention Q/K/V/O projections, layer norms, embeddings, LM head) are excluded from the quantization decision entirely and remain at 16-bit throughout.

The granularity choice is coarse by design: it keeps the state table small (256 entries), makes reconfiguration fast (move/quantize whole experts), and avoids the complexity of mixed-precision within a single expert.

---

## 6. Static vs. Dynamic

| Dimension | This Paper's Approach |
|---|---|
| **Precision assignment** | Static per inference session, but reconfigurable between sessions/tasks |
| **Routing** | Static (unmodified top-k gating) |
| **Placement (CPU/GPU)** | Dynamic: experts are swapped in/out based on access patterns and memory budget |
| **Reconfiguration trigger** | External: job management system signals memory budget changes or new task preferences |
| **Within-forward-pass adaptation** | None: precision and placement are fixed for the duration of a forward pass |

The system is **statically configured per task** but **dynamically reconfigurable between tasks**. It does not adapt precision or placement within a single inference call. The "dynamic" aspect is the ability to reconfigure the deployment plan when constraints change, without a full model reload.

This distinguishes it from truly dynamic approaches (e.g., token-level routing changes) and from fully static approaches (e.g., one-time PTQ with a fixed quantization scheme).

---

## 7. Formulas and Algorithms

### Formula 1: Number of 16-bit Experts (Throughput Preference)

$$
Num_{E,16} =
\begin{cases}
\left\lfloor \dfrac{Mem_{GPU} - Size_{N.E.} - Num_E \times Size_{E,4}}{3 \times Size_{E,4}} \right\rfloor & \text{if } Mem_{GPU} > Size_{N.E.} + Num_E \times Size_{E,4} \\
0 & \text{otherwise}
\end{cases}
$$

### Formula 2: Memory Feasibility Check (Throughput Preference)

The condition for having any excess memory after fitting all experts at 4-bit:

$$
Mem_{GPU} > Size_{N.E.} + Num_E \times Size_{E,4}
$$

If this holds, some experts can be upgraded to 16-bit. If not, all experts must be 4-bit and offloading is required.

### Formula 3: Model Size Under Mixed Precision

Total model size as a function of the number of 4-bit experts:

$$
Size_{model} = Size_{N.E.} + Num_{E,4} \times Size_{E,4} + (Num_E - Num_{E,4}) \times Size_{E,16}
$$

Where `Size_E,16 = 4 × Size_E,4` (16-bit is 4× larger than 4-bit for the same weight count).

For Mixtral 8x7B with `Size_E,16 = 336 MB` and `Size_E,4 = 84 MB`:

- All 16-bit: `3.16 + 256 × 0.328 = 3.16 + 83.97 ≈ 87.13 GB` (paper reports 94.21 GB, suggesting slightly different expert size accounting)
- All 4-bit: `3.16 + 256 × 0.082 ≈ 24.15 GB` (paper reports 23.55 GB)
- Mixed: `26.62 to 94.21 GB` (paper's reported range)

### Formula 4: GPU Residency Count (Quality Preference)

Given a chosen `Num_E,4` and GPU memory budget, the number of experts that can reside on GPU:

$$
Num_{E,GPU} = \min\left(Num_E, \left\lfloor \dfrac{Mem_{GPU} - Size_{N.E.}}{Num_{E,4} \times Size_{E,4} / Num_E \cdot Num_E + \ldots} \right\rfloor \right)
$$

The paper does not give this formula explicitly, but the logic is: pack as many experts as possible onto GPU given the memory budget, prioritizing 4-bit experts.

### Algorithm: Adaptive Inference Partitioner (Pseudocode Reconstruction)

The paper provides no explicit pseudocode, but the algorithm can be reconstructed from the text:

```
Input: Mem_GPU, task_preference ∈ {Quality, Throughput}, Num_E,4 (if Quality)
Output: expert_state_table[e] = (precision, location) for e in 1..Num_E

Constants:
  Size_N.E. = 3.16 GB
  Num_E = 256
  Size_E,4 = 84 MB
  Size_E,16 = 336 MB

Procedure PLAN(Mem_GPU, task_preference, Num_E,4):

  if task_preference == Throughput:
    if Mem_GPU > Size_N.E. + Num_E * Size_E,4:
      Num_E,16 = floor((Mem_GPU - Size_N.E. - Num_E * Size_E,4) / (3 * Size_E,4))
      Num_E,4 = Num_E - Num_E,16
    else:
      Num_E,4 = Num_E
      Num_E,16 = 0
      # offloading required

  # Assign quantization randomly
  quantized_set = random_sample(all_experts, Num_E,4)
  for e in all_experts:
    expert_state[e].precision = 4-bit if e in quantized_set else 16-bit

  # Assign placement: 4-bit experts get GPU priority
  remaining_mem = Mem_GPU - Size_N.E.
  for e in sorted(all_experts, key=lambda x: x.precision):  # 4-bit first
    if remaining_mem >= size(e):
      expert_state[e].location = GPU
      remaining_mem -= size(e)
    else:
      expert_state[e].location = CPU

  return expert_state

Procedure RECONFIGURE(new_Mem_GPU, new_preference, new_Num_E,4):
  new_state = PLAN(new_Mem_GPU, new_preference, new_Num_E,4)
  for e in all_experts:
    if expert_state[e] != new_state[e]:
      apply_delta(e, old=expert_state[e], new=new_state[e])
  expert_state = new_state
```

---

## 8. Experimental Results

### 8.1 Setup

| Component | Details |
|---|---|
| **GPU** | NVIDIA A100 80 GB |
| **CPU** | AMD 16-Core MILAN |
| **Interconnect** | PCIe Gen4 |
| **Model** | Mixtral 8x7B (32 layers, 8 experts/layer, 256 total experts) |
| **Frameworks** | PyTorch, Hugging Face Transformers, Bitsandbytes |
| **Benchmarks** | WikiText2, PTB, C4 |
| **Eval protocol** | 128 samples × 2048 tokens per dataset |
| **Throughput protocol** | Mixed prompts, input+output both limited to 16 tokens |
| **GPU memory range tested** | 26.28 to 53.03 GB |

### 8.2 Perplexity vs. Number of Quantized Experts

As `Num_E,4` increases from 0 to 256, perplexity changes as follows:

| Dataset | Min Perplexity (0 quantized) | Max Perplexity (256 quantized) | Delta |
|---|---|---|---|
| WikiText2 | 3.81 | 4.00 | +0.19 (+5.0%) |
| PTB | 13.59 | 14.17 | +0.58 (+4.3%) |
| C4 | 7.24 | 7.40 | +0.16 (+2.2%) |

Key observations:
- The degradation is small in absolute terms: even fully quantizing all 256 experts raises perplexity by less than 5%.
- The trend is not strictly monotonic. On PTB, some configurations with more quantized experts yield *lower* perplexity than configurations with fewer. The authors attribute this to two factors: (1) perplexity's limited sensitivity to minor MoE modifications, and (2) possible quantization-induced smoothing of model decision boundaries (citing approximate multiplier research [32]).
- The non-monotonicity suggests that random assignment introduces variance, and some random assignments happen to quantize less-critical experts.

### 8.3 Throughput vs. Memory Budget and Quantization Level

Throughput range achieved: **0.63 to 13.00 tokens/s**.

Key observations from Figure 3:

- **Offloading region** (low memory, few quantized experts): throughput is low (near 0.63 tokens/s) because frequent expert misses trigger expensive CPU-GPU transfers (27.35 ms each).
- **Transition region**: as either memory increases or more experts become 4-bit (reducing footprint), more experts fit on GPU, reducing misses. Throughput rises hyperbolically.
- **Full-residency region** (yellow triangle in heatmap): all experts fit on GPU. Throughput peaks near 13.00 tokens/s.
- **Within full-residency**: increasing the number of 4-bit experts *slightly reduces* throughput. This is because PyTorch's 4-bit matrix multiplication kernel is slower than its 16-bit counterpart on A100. The memory savings are no longer needed once everything fits on GPU, so the quantization overhead becomes a net negative.

Memory curves tested: 27.68, 30.50, 33.32, 36.13, 38.95, 41.76, 44.58, 47.40, 50.21, 53.03 GB.

### 8.4 Comparison with Homogeneous Quantization Baselines

| Non-Expert Precision | Expert Precision | Model Size (GB) | WikiText2 PPL | PTB PPL | C4 PPL |
|---|---|---|---|---|---|
| 4-bit | 4-bit | 23.55 | 4.20 | 14.38 | 7.61 |
| 8-bit | 8-bit | 47.10 | 3.82 | 13.25 | 7.26 |
| 16-bit | 16-bit | 94.21 | 3.81 | 13.59 | 7.24 |
| **16-bit** | **Mix 4+16-bit** | **26.62–94.21** | **3.81–4.00** | **13.59–14.17** | **7.24–7.40** |

Critical finding: the partial quantization approach (16-bit non-experts + mixed expert precision) achieves **better quality than homogeneous 8-bit** at a **smaller model size than homogeneous 8-bit**. Specifically:

- Homogeneous 8-bit: 47.10 GB, WikiText2 PPL = 3.82
- Partial quantization (all experts 4-bit): 26.62 GB, WikiText2 PPL = 4.00

The partial approach is 44% smaller than 8-bit while achieving comparable perplexity. At intermediate quantization levels (e.g., 128 of 256 experts at 4-bit), the model is ~60 GB and achieves PPL ≈ 3.90, which is better than 8-bit at a similar size.

### 8.5 Comparison with Mixtral-Offloading Baseline

The paper compares throughput against Mixtral-Offloading [5] under varying GPU memory budgets. The partial quantization approach achieves higher throughput at the same memory budget because:

1. More experts fit on GPU (4-bit experts are 4× smaller).
2. Fewer expert misses occur.
3. Each miss transfers a smaller expert (84 MB vs. 336 MB), reducing transfer time.

Exact numeric comparison values are not tabulated in the paper; the comparison is shown in Figure 3.

---

## 9. Implementation Details

### 9.1 Software Stack

| Component | Tool/Library |
|---|---|
| Deep learning framework | PyTorch |
| Model loading and tokenization | Hugging Face Transformers |
| 4-bit quantization | Bitsandbytes (NF4 format) |
| Expert state tracking | Custom expert state table |

### 9.2 Quantization Method

Bitsandbytes NF4 (Normal Float 4-bit) quantization is used for 4-bit experts. NF4 is a data-type optimized for normally distributed weights, which is typical of transformer weight matrices. It stores weights in 4-bit with a lookup table and dequantizes on-the-fly during matrix multiplication.

The quantization is applied at the expert granularity: the entire feed-forward block of an expert (two or three linear layers, depending on the architecture) is quantized together.

### 9.3 Expert State Table

A table with 256 entries (one per expert) tracks:
- `precision`: 4-bit or 16-bit
- `location`: GPU or CPU

This table is consulted at every expert dispatch during inference. If `location == CPU`, the expert is transferred to a pre-allocated GPU swap buffer before computation.

### 9.4 Swap Buffer

A fixed-size swap buffer is allocated on GPU to hold experts being transferred from CPU. The buffer is sized for at least one expert (84 MB for 4-bit, 336 MB for 16-bit). Transfers use standard PyTorch tensor operations over PCIe.

### 9.5 Reconfiguration Procedure

When constraints change:
1. Compute new deployment plan (new precision assignments, new placement).
2. For each expert where the plan changed:
   - If precision changed: re-quantize or dequantize the expert weights.
   - If location changed: move the expert tensor between CPU and GPU memory.
3. Update the state table.

This is incremental: only changed experts are touched. The non-expert layers are never moved or re-quantized.

### 9.6 Model Architecture Details (Mixtral 8x7B)

| Parameter | Value |
|---|---|
| Transformer layers | 32 |
| Experts per layer | 8 |
| Total experts | 256 |
| Non-expert layer size | 3.16 GB total |
| Expert size (16-bit) | 336 MB each |
| Expert size (4-bit) | ~84 MB each |
| Expert transfer time (PCIe Gen4) | 27.35 ms |
| Top-k routing | k=2 (standard Mixtral) |

---

## 10. Limitations and Gaps

### 10.1 Random Quantization Assignment

The most significant limitation is that quantization assignment is random rather than importance-based. The paper justifies this by appealing to MoE load balancing, but in practice:

- Expert utilization is not perfectly uniform. Some experts are accessed more frequently for certain input distributions.
- Quantizing a frequently-accessed expert degrades more tokens than quantizing a rarely-accessed one.
- An importance-aware assignment (e.g., quantize least-accessed experts first) could achieve the same memory savings with lower perplexity impact.

The paper does not compare random assignment against any structured assignment strategy.

### 10.2 Only Two Precision Levels

The system supports only 4-bit and 16-bit. There is no exploration of 8-bit experts, 2-bit experts, or mixed granularities within a single expert. A finer precision ladder (e.g., 2/4/8/16-bit) would provide more points on the Pareto frontier and finer-grained QoS control.

### 10.3 No Sub-Expert Granularity

Precision is assigned at the whole-expert level. Within an expert, all weight matrices use the same precision. Finer granularity (e.g., per-layer within an expert, or per-weight-group) could reduce quality loss for the same memory savings.

### 10.4 Single GPU Only

The evaluation is entirely on a single A100 GPU. Multi-GPU settings (tensor parallelism, expert parallelism) are not addressed. In practice, large MoE deployments often span multiple GPUs, and the interaction between precision assignment and inter-GPU communication is unexplored.

### 10.5 No Latency Breakdown

The paper reports throughput (tokens/s) but not detailed latency breakdowns. It is unclear how much time is spent on: expert computation, expert transfers, attention computation, and routing. This makes it hard to identify the next bottleneck after reducing expert transfer overhead.

### 10.6 No Energy or Cost Analysis

Power consumption and cost per token are not reported. For deployment decisions, these metrics often matter as much as throughput.

### 10.7 No Learned or Adaptive Routing

The routing mechanism is completely unchanged. There is no exploration of whether routing should be modified to prefer 16-bit experts (for quality-sensitive tokens) or 4-bit experts (for throughput-sensitive tokens). Such routing-precision co-design could unlock further QoS control.

### 10.8 Perplexity as the Sole Quality Metric

Only perplexity is used to measure output quality. Downstream task performance (e.g., MMLU, HellaSwag, GSM8K) is not evaluated. The paper itself notes that perplexity has limited sensitivity to minor MoE modifications, which raises questions about whether the quality differences are practically meaningful.

### 10.9 No Comparison with QMoE or MoQE

The paper does not compare against QMoE [6] (sub-1-bit compression) or MoQE [8] (mixture of quantized experts), which are the most directly related prior works on MoE-specific quantization. This makes it hard to assess where this approach sits in the broader landscape.

### 10.10 Scope Limited to Memory Constraints

The paper explicitly scopes to GPU memory as the key constraint. PCIe bandwidth and GPU compute utilization are mentioned as other varying resources in shared systems but are not modeled or optimized.

---

## 11. Relevance to Our Work: Dual-Tile TensorRT-LLM

Our dual-tile work partitions experts by token count: hot experts (receiving many tokens) get larger compute tiles, cold experts get smaller tiles. This paper's QoS-based precision assignment is a natural complement along a second dimension.

### 11.1 The Complementary Axes

| Axis | Our Work | This Paper |
|---|---|---|
| **Compute allocation** | Tile size proportional to token count | Not addressed |
| **Memory precision** | Uniform (not addressed) | Mixed 4/16-bit per expert |
| **Routing** | Unmodified | Unmodified |
| **Granularity** | Per-expert, per-forward-pass | Per-expert, per-task |

The two approaches are orthogonal and can be composed: assign precision based on expert "temperature" (access frequency), then size the compute tile based on the actual token count in the current forward pass.

### 11.2 Hot/Cold Expert Mapping

In our dual-tile framework, experts naturally fall into two categories based on token load:

- **Hot experts** (many tokens routed to them): already receive larger tiles for compute efficiency. This paper suggests they should also receive **16-bit precision** to preserve quality on the tokens that matter most. Hot experts are accessed frequently, so quantization errors accumulate more.
- **Cold experts** (few tokens): already receive smaller tiles. This paper suggests they should receive **4-bit precision** to save memory. Cold experts contribute less to overall output quality, so quantization errors matter less.

This creates a consistent policy: hot = large tile + 16-bit, cold = small tile + 4-bit.

### 11.3 Memory Budget Interaction

In our setting, larger tiles consume more SRAM. If hot experts also use 16-bit weights, their weight loading from HBM is larger. The memory pressure from large tiles and high precision is additive. However, cold experts at 4-bit free up HBM capacity that can be reallocated to hot expert weight caching, potentially reducing HBM bandwidth pressure for the hot path.

### 11.4 Routing-Precision Co-Design Opportunity

This paper leaves routing unmodified. Our dual-tile work already tracks per-expert token counts. We could extend this to:

1. Maintain a running estimate of expert access frequency (exponential moving average of token counts).
2. Use this frequency estimate to drive precision assignment: top-N most-accessed experts stay 16-bit, rest go 4-bit.
3. Reconfigure precision assignments periodically (e.g., between batches) as access patterns shift.

This would be a data-driven, importance-aware version of this paper's random assignment, directly addressing its main limitation.

### 11.5 QoS Knob Integration

This paper's QoS framework (expose a Pareto frontier of throughput vs. quality) maps naturally onto our tile-size framework. We could expose a unified QoS knob that jointly controls:

- Tile size distribution (compute allocation)
- Precision distribution (memory allocation)

A single "quality budget" parameter could drive both: high quality = large tiles + 16-bit for hot experts; high throughput = uniform small tiles + aggressive 4-bit.

### 11.6 Practical Adoption Notes

- The random assignment policy is easy to implement but suboptimal. Our token-count tracking gives us a better signal for free.
- The Bitsandbytes NF4 quantization used here is a standard PTQ method compatible with TensorRT-LLM's quantization pipeline.
- The expert state table concept (precision + location per expert) is directly implementable in our expert dispatch layer.
- Partial reconfiguration (only touching changed experts) is important for low-latency adaptation and aligns with our goal of minimizing reconfiguration overhead between batches.

---

## 12. Comparison Table

### 12.1 Comparison with Related MoE Quantization Works

| Paper | Precision Granularity | Assignment Strategy | Routing Modified | Precision Levels | QoS / Adaptive | GPU Setting |
|---|---|---|---|---|---|---|
| **This paper (2407.14417)** | Per-expert | Random | No | 4-bit + 16-bit | Yes (throughput/quality preference) | Single GPU |
| QMoE [Frantar 2023] | Per-expert | Importance-based | No | Sub-1-bit | No | Multi-GPU |
| MoQE [Kim 2023] | Per-expert | Structured | No | Mixed | No | Multi-GPU |
| He et al. [2024] | Per-expert + block | Sensitivity-based | No | Multiple | No | Not specified |
| Mixtral-Offloading [2023] | Uniform (all 4-bit) | N/A | Speculative prefetch | 4-bit only | No | Single GPU |
| **Our dual-tile work** | Per-expert (proposed) | Token-count-based | Tile-size routing | 4-bit + 16-bit (proposed) | Yes (tile size) | Dual-tile GPU |

### 12.2 Quantitative Summary

| Configuration | Model Size (GB) | WikiText2 PPL | PTB PPL | C4 PPL | Throughput (tok/s) |
|---|---|---|---|---|---|
| All 16-bit (baseline) | 94.21 | 3.81 | 13.59 | 7.24 | ~13.00 (all on GPU) |
| All 8-bit | 47.10 | 3.82 | 13.25 | 7.26 | N/A |
| All 4-bit (homogeneous) | 23.55 | 4.20 | 14.38 | 7.61 | N/A |
| 16-bit non-expert + all 4-bit expert | 26.62 | 4.00 | 14.17 | 7.40 | 0.63–13.00 |
| 16-bit non-expert + mixed expert | 26.62–94.21 | 3.81–4.00 | 13.59–14.17 | 7.24–7.40 | 0.63–13.00 |

### 12.3 Design Space Comparison: This Paper vs. Our Work

| Dimension | This Paper | Our Dual-Tile Work | Combined Potential |
|---|---|---|---|
| Expert precision | Mixed 4/16-bit | Uniform (not addressed) | Mixed, token-count-driven |
| Compute tile size | Uniform | Mixed (token-count-driven) | Mixed, precision-aware |
| Assignment basis | Random | Token count (dynamic) | Token count + frequency |
| Adaptation trigger | Memory budget change | Token count per batch | Both |
| Routing modification | None | Tile-size dispatch | Tile-size + precision-aware |
| QoS knob | Throughput/quality preference | Tile size distribution | Unified quality budget |

---

## Summary

This paper makes a focused, practical contribution: it shows that keeping non-expert layers at 16-bit while randomly mixing 4-bit and 16-bit experts provides a continuous Pareto frontier between throughput and quality, enabling adaptive QoS for MoE deployment under changing memory constraints. The key formula governs how many experts can be upgraded to 16-bit given a memory budget. The approach is simple (random assignment, no routing changes, standard PTQ), which is both its strength (easy to implement, no task-specific tuning) and its main weakness (leaves quality on the table compared to importance-aware assignment).

For our dual-tile work, the most actionable takeaway is the **hot/cold expert precision policy**: hot experts (many tokens, large tiles) should also be 16-bit; cold experts (few tokens, small tiles) should be 4-bit. Our existing token-count tracking provides a better assignment signal than random, directly addressing this paper's primary limitation. The expert state table and partial reconfiguration concepts are directly adoptable in our dispatch layer.
