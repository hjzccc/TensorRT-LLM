# DynaMo: Runtime Switchable Quantization for MoE with Cross-Dataset Adaptation

**Technical Analysis Document**
*Prepared for TensorRT-LLM Dual-Tile MoE Project*
*Date: March 2026*

---

## 1. Paper Metadata

| Field | Value |
|---|---|
| **Title** | DynaMo: Runtime Switchable Quantization for MoE with Cross-Dataset Adaptation |
| **Authors** | Zihao Zheng, Xiuping Cui, Size Zheng, Maoliang Li, Jiayu Chen, Yun Liang, Xiang Chen |
| **Affiliation** | School of Computer Science / School of Integrated Circuits, Peking University, Beijing, China |
| **Corresponding** | Xiang Chen (xiang.chen@pku.edu.cn) |
| **arXiv ID** | 2503.21135 |
| **Venue** | arXiv preprint, 2025 |
| **Index Terms** | Mix-of-Experts, Model Quantization, Multi-Level Analysis, Cross-Dataset Adaptation |

---

## 2. Problem Statement

### What Problem Does This Paper Solve?

Large language models built on the Mixture-of-Experts (MoE) architecture scale parameter counts dramatically while keeping per-token compute sparse. Each MoE layer replaces the dense FFN with a bank of expert networks; a router dispatches each token to a small subset (Top-K) of those experts. Despite sparse activation, MoE models still carry enormous weight footprints, making quantization essential for practical deployment.

The core problem DynaMo addresses is this: **existing quantization methods treat MoE models as if they were static, single-dataset systems, but expert utilization patterns shift dramatically across datasets.** A quantization scheme calibrated on WikiText2 may be badly mismatched when the same model is deployed on C4, code corpora, or domain-specific text.

### Why Static Quantization Fails for MoE

Static quantization methods (GPTQ, QuantEase, SmoothQuant, AWQ, RPTQ, Atom) were designed for dense LLMs. They share a fundamental assumption: the data-weight mapping is one-to-one. In a dense FFN, one token embedding activates one set of channel weights. The calibration dataset determines which weights matter, and those weights get quantized carefully.

MoE breaks this assumption in two ways:

**One-to-many data-weight mappings.** In a MoE layer, a single token embedding is dispatched to multiple experts, each with its own channel weights. The same token therefore activates channel weights across several different expert networks simultaneously. Dense-LLM quantization methods assume one-to-one correlations; they cannot correctly model this distributed activation pattern.

**Expert dynamics across datasets.** Which experts get activated, and how heavily, depends on the input distribution. Expert 28 might carry very high significance on WikiText2 (significance > 0.5) but near-zero significance on C4 (significance < 0.1). A quantization scheme that assigns INT8 to expert 28 because it was important during WikiText2 calibration will waste bits when the model runs on C4, and vice versa: an expert that was unimportant during calibration but becomes critical on a new dataset will be under-quantized.

Static quantization cannot adapt after the fact. Once weights are quantized to INT4, you cannot recover INT8 precision without the original FP16 values. Recalibrating from scratch on every new dataset is prohibitively expensive. The result is that static MoE quantization either over-compresses important experts or under-compresses unimportant ones, depending on which dataset it was calibrated for.

---

## 3. Key Insight / Core Idea

DynaMo rests on two observations that together make the problem tractable:

**Observation 1: Expert significance is dataset-dependent and quantifiable.** The importance of each expert can be measured as a function of how much its channel weights contribute to token processing on a given dataset. This significance is not fixed; it shifts when the input distribution changes. By measuring significance across multiple datasets and fitting a joint distribution, you can build a quantization baseline that is robust to the most common datasets while remaining adaptable.

**Observation 2: Only ~1% of channels drive the cross-dataset dynamics.** When you trace expert-level significance shifts down to individual channel weights, the dynamics are not spread uniformly. A small fraction of channels, roughly 1% of the total per expert, account for the most prominent significance changes across datasets. The other 99% of channels behave similarly regardless of which dataset is being processed.

This second observation is the key that makes runtime switching feasible. If all channels were equally dynamic, you would need to re-quantize the entire model for every new dataset, which is impractical. But if only 1% of channels are responsible for the dynamics, you can:

1. Quantize the full model statically using a joint-distribution-aware baseline.
2. Cache just the 1% most dynamic channels in FP16.
3. When the dataset changes, re-quantize only those cached channels to the new appropriate precision.

The overhead of switching is proportional to 1% of the model, not 100%, making it fast enough for practical deployment.

---

## 4. Technical Approach: Step by Step

### 4a. Multi-Level Significance Analysis

#### Dataset-Specific Channel Weight Significance (S_ch)

The first step is to measure how much each channel weight in each expert contributes to processing tokens from a given dataset.

The procedure:
1. Randomly sample tokens from the target dataset.
2. Run MoE inference on those tokens until the full dataset is traversed.
3. **Hook function**: attach a hook to each expert's channel weight outputs to capture the output logits (activations) produced by each channel weight for each token.
4. For each channel weight in each expert, compute a significance score `S_ch` by statistically summarizing the correlation between that channel's activations and the tokens in the dataset.

The paper does not give an explicit closed-form formula for `S_ch` itself. It is described as a "fine-grained significance analysis" that maps channel-weight activations to token correlations. The most natural interpretation, consistent with prior work like AWQ, is that `S_ch` captures the magnitude or variance of a channel's output activations weighted by token frequency. Channels that produce large, consistent activations across many tokens score high; channels that are rarely activated or produce small outputs score low.

**What the hook captures:** The hook intercepts the output tensor of each channel weight (i.e., the post-linear-layer activations for that channel) during a forward pass. By accumulating these across the calibration tokens, you get a per-channel activation profile that reflects how much that channel "matters" for the current dataset.

#### Expert Significance Formula (Equation 1)

Once per-channel significance is established, expert-level significance is derived by aggregating channel significance across all tokens that were dispatched to that expert.

**Equation 1** (as presented in the paper):

```
S_exp^j = [ sum_{i=1}^{T} S_ch( sum_{k=1}^{K} W_ch^k | token = tau^i ) ]
          / [ sum_{j=1}^{N} sum_{i=1}^{T} S_ch( sum_{k=1}^{K} W_ch^k | token = tau^i ) ]
```

**Variable definitions:**

| Symbol | Meaning |
|---|---|
| `S_exp^j` | Significance of the j-th expert |
| `N` | Total number of experts in the MoE layer |
| `T` | Total number of tokens in the dataset |
| `tau^i` | The i-th token |
| `K` | Number of experts selected per token (Top-K routing) |
| `W_ch^k` | Channel weights of the k-th selected expert |
| `S_ch(...)` | Channel-weight significance function (described above) |

**Plain-English explanation:**

The numerator sums up the channel-weight significance for expert j across all T tokens in the dataset. For each token tau^i, you look at which K experts were selected, take the channel weights of expert j (if it was selected), and compute their significance. Summing this over all tokens gives the total "work" that expert j did on this dataset, measured in terms of significance-weighted channel activations.

The denominator normalizes by summing the same quantity across all N experts, so the result is a fraction between 0 and 1. An expert with `S_exp^j = 0.5` is responsible for half the total channel-weight significance in the layer on this dataset. An expert with `S_exp^j = 0.02` is nearly idle.

This normalization makes the significance scores comparable across layers and datasets.

#### Cross-Dataset Expert Dynamics

The paper evaluates the same MoE layer on WikiText2 and C4 separately, computing `S_exp^j` for each dataset. The results reveal dramatic shifts:

- Expert 28 (E28) in OLMoE has significance > 0.5 on WikiText2 (highly active, important).
- The same expert 28 has significance < 0.1 on C4 (nearly idle).

This is not an edge case; it reflects a systematic pattern where different datasets activate different subsets of experts. The expert-significance vector `{S_exp^j}_{j=1}^N` is essentially a fingerprint of the dataset's interaction with the MoE layer.

### 4b. Expert-Level Baseline Quantization

#### Joint Distribution Fitting (Equation 2)

Since expert significance is dataset-dependent, no single dataset's significance vector is the right basis for quantization. DynaMo instead fits a joint distribution over significance vectors from multiple datasets.

**Equation 2:**

```
J  <-- Fit( {S_exp^j}_{j=1}^N | D1,  {S_exp^j}_{j=1}^N | D2,  ...,  {S_exp^j}_{j=1}^N | Dn )
```

**What this means:** Collect the expert-significance vector from each of n calibration datasets (D1 through Dn). Fit a joint distribution J over all of these vectors. The joint distribution captures both the central tendency (which experts are consistently important across datasets) and the variance (which experts shift dramatically).

The diagonal of the joint distribution represents synthesized expert baseline significance: the "average" importance of each expert across all calibration datasets. The off-diagonal structure captures the dynamics, i.e., how much each expert's significance varies across datasets.

This joint distribution becomes the foundation for the baseline quantization: experts that are consistently important across all datasets get high precision (INT8), while experts that are consistently unimportant get low precision (INT2). Experts with high variance are the ones that need dynamic switching.

#### Fuzzy C-Means Clustering (Algorithm 1)

With the synthesized expert baseline significance in hand, the next step is to assign each expert to one of four precision levels: INT2, INT4, INT6, INT8. The paper uses fuzzy c-means clustering rather than hard k-means.

**Why fuzzy c-means?** Hard k-means assigns each expert to exactly one cluster with a binary membership. Fuzzy c-means assigns each expert a membership degree to every cluster, a value between 0 and 1 that reflects how strongly the expert belongs to each cluster. This is important because expert significance distributions are continuous, and the boundaries between precision levels are not sharp. An expert with significance 0.35 might be genuinely ambiguous between INT4 and INT6; fuzzy membership captures this ambiguity rather than forcing an arbitrary hard assignment.

**Algorithm 1: Clustering Based on Synthesized Expert Baseline Significance**

```
Input:
  X = {x1, x2, ..., xn}   -- synthesized expert baseline significance scores
  c = 4                    -- number of clusters (one per precision level)
  m > 1                    -- fuzzifier (controls softness of membership; m=2 is typical)
  epsilon                  -- convergence tolerance

Output:
  V = {v1, v2, ..., vc}   -- cluster centers
  U                        -- membership matrix (n x c), where U[i][k] = u_ik

Step 1: Initialize U
  For each expert i and cluster k, set u_ik in [0, 1]
  Subject to: for all i, sum_{k=1}^{c} u_ik = 1
  (Each expert's memberships across all clusters sum to 1)

Step 2: Repeat until convergence
  
  Update cluster centers V:
    for k = 1 to c:
      v_k = ( sum_{i=1}^{n} u_ik^m * x_i ) / ( sum_{i=1}^{n} u_ik^m )
    end for
  
  Update membership matrix U:
    for all i, k:
      u_ik = 1 / sum_{j=1}^{c} ( ||x_i - v_k|| / ||x_i - v_j|| )^(2/(m-1))
    end for

Step 3: Check convergence
  if ||U_new - U_old|| < epsilon:
    stop
  else:
    go to Step 2

Return V and U
```

**Step-by-step explanation of what each part does:**

*Initialization (Step 1):* Start with random membership values. Each expert i has a membership vector of length c=4, summing to 1. For example, expert 28 might start with memberships [0.4, 0.3, 0.2, 0.1] across clusters 1-4.

*Cluster center update (Step 2, first loop):* Each cluster center v_k is the weighted average of all expert significance scores, where the weights are the fuzzy memberships raised to the power m. The fuzzifier m controls how much influence low-membership experts have: with m=2, an expert with membership 0.1 contributes 0.01 times as much as an expert with membership 1.0. Higher m makes the clustering softer (more uniform memberships); m approaching 1 makes it harder (approaching k-means).

*Membership update (Step 2, second loop):* Each membership u_ik is updated based on how close expert i's significance is to cluster center v_k relative to all other cluster centers v_j. If expert i is very close to cluster k and far from all others, u_ik approaches 1. If expert i is equidistant from all clusters, all memberships approach 1/c = 0.25.

*Convergence (Step 3):* Repeat until the membership matrix changes by less than epsilon between iterations. In practice, fuzzy c-means converges in tens of iterations.

**Cluster-to-precision mapping:**

After convergence, the four cluster centers v_1 through v_4 are ordered by their significance values. The cluster with the highest center (most important experts) maps to INT8. The cluster with the lowest center (least important experts) maps to INT2. The intermediate clusters map to INT6 and INT4 respectively.

| Cluster | Significance | Precision |
|---|---|---|
| Cluster-1 (highest U) | Highest | INT8 |
| Cluster-2 | High | INT6 |
| Cluster-3 | Low | INT4 |
| Cluster-4 (lowest U) | Lowest | INT2 |

**Hard assignment from fuzzy memberships:** The paper does not explicitly state how it converts fuzzy memberships to hard precision assignments. The most natural interpretation is argmax: each expert is assigned to the cluster for which it has the highest membership. This is a known limitation of the approach (discussed in Section 9).

#### Overlap/Boundary Handling: Low-Precision-First Rule

Fuzzy c-means produces soft cluster boundaries, meaning some experts will have significant membership in two adjacent clusters. For example, an expert might have membership 0.48 in Cluster-2 (INT6) and 0.45 in Cluster-3 (INT4), making the assignment genuinely ambiguous.

DynaMo handles this with the **low-precision-first rule**: when an expert falls in an overlapping region between two clusters, assign it to the lower-precision cluster.

The rationale is empirical: experiments show that experts in boundary regions have negligible impact on quantization accuracy loss regardless of which precision they receive. Since the accuracy cost is the same either way, choosing lower precision improves the overall compression ratio without hurting quality. This is a pragmatic engineering choice that prioritizes compression efficiency at the boundary.

All expert-level analysis and baseline quantization happen **offline**, before deployment. They add no overhead to inference.

### 4c. Channel-Level Dynamic Switching

The expert-level baseline quantization gives a good starting point, but it cannot adapt when a new dataset arrives. The channel-level dynamic switching mechanism handles this.

#### Which Channels Are Cached (~1%) and Why FP16

After tracing expert-level significance dynamics down to individual channel weights, the paper finds that only ~1% of channels per expert show the most prominent cross-dataset dynamics. These are the channels whose significance scores change the most between datasets.

These ~1% most dynamic channels are cached in FP16 (full precision) in GPU global memory (GMEM). The rest of the channels are quantized normally and not cached.

**Why FP16?** Because quantization is irreversible. Once a channel weight is quantized to INT4, you cannot recover the original FP16 value from the quantized representation alone. If you later discover that this channel should have been INT8 (because the new dataset makes it more important), you have no way to upgrade it without the original. Caching in FP16 preserves the option to re-quantize to any target precision, whether higher or lower than the current baseline.

**Cache format:** Each cache entry stores three fields:
- `E_id`: expert index (which expert this channel belongs to)
- `C_id`: channel index (which channel within that expert)
- `FP16 Channel`: the full-precision weight values for that channel

The expert and channel indices serve as headers for rapid lookup during replacement.

**Cache sizes (from the paper):**

| Model Size | Model Weight Size | Cache Size | Search Time |
|---|---|---|---|
| 7B | 15 GB | 152.8 MB | ~13 ms |
| 14B | 33.5 GB | 340.7 MB | ~19 ms |
| 16B | 38.1 GB | 402.3 MB | ~21 ms |

For the 7B model, 152.8 MB is 1.02% of the total weight storage, confirming the ~1% claim.

#### What Triggers Switching

Switching is triggered by a **dataset change**, not by individual batches or forward passes. This is a coarse-grained trigger: the system detects (or is notified) that the input distribution has shifted to a new dataset, then performs the switching procedure once. All subsequent inference on that dataset uses the new quantization configuration without further switching overhead.

This is an important distinction from per-batch or per-step adaptation. DynaMo does not re-quantize on every forward pass; it re-quantizes once per dataset transition.

#### The Switching Procedure: Step by Step

When a new dataset D_new is detected:

**Step 1: Re-compute expert significance for D_new.**
Run inference on a sample of tokens from D_new with the hook function active. Compute `{S_exp^j}_{j=1}^N | D_new` using Equation 1. This gives the expert significance vector for the new dataset.

**Step 2: Re-cluster using Algorithm 1.**
Apply the fuzzy c-means algorithm (Algorithm 1) to the new significance vector `{S_exp^j}_{j=1}^N | D_new`. This produces a new cluster assignment for each expert under the new dataset.

**Step 3: Compare to baseline.**
Compare the new cluster assignments to the joint-distribution-based baseline assignments from the offline phase. For each expert, determine whether its cluster (and therefore its target precision) has changed.

**Step 4: Replace cached FP16 channels.**
For each expert whose precision assignment has changed, retrieve the cached FP16 channel weights using the `E_id` and `C_id` indices. Replace the currently quantized versions of those channels with the FP16 cached values. This restores the original precision for those channels.

**Step 5: Re-quantize to new precision.**
Quantize the restored FP16 channels to the new target precision (INT2/4/6/8) determined by the new cluster assignment. Pack the quantized values into INT32 for efficient loading.

**Step 6: Continue inference.**
All subsequent forward passes on D_new use the updated quantization configuration.

#### Concrete Example: Expert 28, INT8 to INT4

The paper gives a specific example to illustrate the switching:

- **Offline baseline (joint distribution):** Expert 28 is assigned to Cluster-1, which maps to INT8. This is because, averaged across calibration datasets, expert 28 has high significance.
- **New dataset arrives:** Compute `S_exp^28 | D_new`. Expert 28 has very low significance on D_new (analogous to the C4 case where significance < 0.1).
- **Re-clustering:** Algorithm 1 assigns expert 28 to Cluster-3 on D_new, which maps to INT4.
- **Switching:** Retrieve the cached FP16 channels of expert 28. Re-quantize them to INT4. Expert 28 now runs at INT4 instead of INT8 on D_new.
- **Effect:** Expert 28 is now more aggressively compressed, which is appropriate since it contributes little to D_new. This frees up precision budget that can be allocated to experts that are more important on D_new.

The switching can go in either direction: an expert that was INT4 in the baseline might become INT8 on a new dataset if it becomes more important, or an INT8 expert might drop to INT2 if it becomes nearly idle.

### 4d. Implementation

#### Channel Cache Format

The channel cache lives in GPU global memory (GMEM). Each record has the structure:

```
[ E_id | C_id | FP16 Channel Weights ]
```

- `E_id` (expert index): identifies which expert this channel belongs to.
- `C_id` (channel index): identifies which channel within that expert.
- `FP16 Channel Weights`: the full-precision weight values for the entire channel.

During replacement, the system searches the cache by `(E_id, C_id)` pairs to retrieve the correct FP16 weights. The search times are 13-21 ms depending on model size, which is negligible relative to inference time for generating 1024 tokens.

#### Weight Packing into INT32

After quantization, multiple low-precision values are packed into a single INT32 register. For example:
- INT8: 4 values per INT32
- INT4: 8 values per INT32
- INT2: 16 values per INT32

This packing allows the GPU to load multiple weights in a single memory transaction using existing hardware load instructions. After loading, the packed INT32 is unpacked on the GPU to recover the individual low-precision values. This is the primary source of inference speedup: it maximizes effective memory bandwidth utilization by reducing the number of load operations needed to fetch a given number of weights.

#### Channel-Grain Dequantization to FP16

Mixed precision creates a compute incompatibility problem: INT8 weights cannot be directly multiplied with FP16 activations using tensor cores. DynaMo resolves this with channel-grain dequantization.

After loading and unpacking the low-precision weights, CUDA cores dequantize them to FP16 at channel granularity. The dequantization formula for uniform quantization is:

```
W_fp16 = scale * W_int + zero_point
```

where `scale` and `zero_point` are per-channel quantization parameters stored alongside the weights. After dequantization, both weights and activations are in FP16, so tensor cores can execute the matrix multiply-accumulate (MMA) operation normally.

#### Fused Dequant + MMA Kernel

The dequantization and MMA operations are fused into a single CUDA kernel. This avoids the overhead of writing dequantized weights back to global memory and re-loading them for the MMA. Instead, dequantized values flow directly from CUDA cores to tensor cores within the same kernel, keeping data in registers or shared memory throughout.

The GPU memory hierarchy used:
- **GMEM (40G/80G):** stores quantized weights and the FP16 channel cache
- **L2 (40M):** intermediate caching
- **L1 (49K):** per-SM cache
- **Register File (256K):** holds packed INT32 during load, unpacked values during dequant, FP16 values during MMA

The dataflow is: Load (GMEM -> registers) -> Dequant (CUDA cores, registers) -> MMA (tensor cores, registers).

---

## 5. Granularity of Quantization

DynaMo operates at two distinct granularities, which work together:

**Expert-level (coarse baseline):** The entire weight matrix of each expert is quantized to a single precision level (INT2, INT4, INT6, or INT8). This is the coarsest granularity and is determined offline by the fuzzy c-means clustering on the joint distribution. All channels within an expert share the same precision in the baseline. This granularity is computationally efficient because it requires only one quantization scale per expert (or per-channel scales within a uniform quantization scheme).

**Channel-level (fine dynamic switching):** Within each expert, individual channels can be switched to different precisions at runtime. Only the ~1% most dynamic channels are eligible for switching; the other 99% remain at the expert-level baseline precision. This fine granularity allows the system to adapt to dataset-specific channel importance without re-quantizing the entire model.

The two levels are complementary: expert-level quantization handles the bulk of the compression with a dataset-robust baseline, while channel-level switching handles the residual adaptation needed for new datasets.

---

## 6. Static vs. Dynamic: What Changes and When

**Static component (expert-level baseline):**
- Computed entirely offline before deployment.
- Based on the joint distribution J fitted over multiple calibration datasets.
- Assigns each expert a fixed precision (INT2/4/6/8) that is robust across the calibration datasets.
- Does not change during inference unless a dataset switch triggers re-quantization.
- The 99% of non-cached channels are always static.

**Dynamic component (channel-level switching):**
- Triggered once per dataset change, not per batch or per forward pass.
- Affects only the ~1% of cached FP16 channels.
- Procedure: re-compute significance -> re-cluster -> compare to baseline -> replace cached channels -> re-quantize.
- After switching, the new configuration is static until the next dataset change.

**Critical distinction:** DynaMo's "dynamic" switching is coarse-grained in time. It adapts to dataset-level distribution shifts, not to individual input variations within a dataset. Within a single dataset, every forward pass uses the same quantization configuration. The switching overhead (13-21 ms search time plus re-quantization) is amortized over all inference on the new dataset, making it negligible in practice.

This is fundamentally different from per-step or per-batch adaptation, where the quantization configuration would change on every forward pass. DynaMo does not do this.

---

## 7. Formulas and Algorithms: Complete Reference

### Equation 1: Expert Significance

```
S_exp^j = [ sum_{i=1}^{T} S_ch( sum_{k=1}^{K} W_ch^k | token = tau^i ) ]
          / [ sum_{j=1}^{N} sum_{i=1}^{T} S_ch( sum_{k=1}^{K} W_ch^k | token = tau^i ) ]
```

Computes the normalized significance of expert j on a given dataset. Numerator: total channel-weight significance contributed by expert j across all T tokens. Denominator: same sum across all N experts (normalization). Result: a value in [0, 1] representing expert j's share of total layer significance.

### Equation 2: Joint Distribution Fitting

```
J  <-- Fit( {S_exp^j}_{j=1}^N | D1,
            {S_exp^j}_{j=1}^N | D2,
            ...,
            {S_exp^j}_{j=1}^N | Dn )
```

Fits a joint distribution J over expert-significance vectors from n calibration datasets. The diagonal of J gives synthesized expert baseline significance; off-diagonal structure captures cross-dataset dynamics.

### Algorithm 1: Fuzzy C-Means Clustering

```
Input:
  X = {x1, ..., xn}   -- expert baseline significance scores (n experts)
  c = 4               -- number of clusters
  m > 1               -- fuzzifier (typically m = 2)
  epsilon             -- convergence tolerance

Output:
  V = {v1, ..., vc}   -- cluster centers
  U                   -- n x c membership matrix

Initialize:
  u_ik in [0, 1] for all i in {1..n}, k in {1..c}
  sum_{k=1}^{c} u_ik = 1 for all i

Repeat:
  // Update cluster centers
  for k = 1 to c:
    v_k = ( sum_{i=1}^{n} u_ik^m * x_i ) / ( sum_{i=1}^{n} u_ik^m )

  // Update memberships
  for all i, k:
    u_ik = 1 / sum_{j=1}^{c} ( ||x_i - v_k|| / ||x_i - v_j|| )^(2/(m-1))

Until ||U_new - U_old|| < epsilon

Return V, U
```

### Precision Assignment Rule

```
Cluster with highest center -> INT8
Cluster with second-highest center -> INT6
Cluster with second-lowest center -> INT4
Cluster with lowest center -> INT2

Boundary rule: if expert i is in overlapping region between cluster k and cluster k+1,
               assign to cluster k+1 (lower precision)
```

### Dequantization (Uniform Quantization)

```
W_fp16 = scale * W_int + zero_point
```

Applied per-channel after loading packed INT32 weights. `scale` and `zero_point` are stored per-channel alongside the quantized weights.

---

## 8. Experimental Results

### Setup

- **Hardware:** NVIDIA A100 GPUs
- **Baselines:** GPTQ (3-bit), MoEPTQ (3.26-3.36 bit)
- **Models:** OLMoE (1B/7B), MoE-Girl (1B/7B), Qwen1.5-MoE (3B/14B), DeepSeek-MoE / DS-MoE (3B/16B)
- **Language modeling:** WikiText2, C4 (perplexity, lower is better)
- **Zero-shot:** ARC-challenge, ARC-easy, RTE, PIQA, COPA, CB (accuracy, higher is better)

### Language Modeling Results (Table I)

| Model | Bits | Wiki PPL | C4 PPL | Avg PPL | vs. Best Baseline |
|---|---|---|---|---|---|
| OLMoE (FP16) | 16 | 7.41 | 11.42 | 9.42 | -- |
| OLMoE (GPTQ) | 3.00 | 11.65 | 18.86 | 15.26 | -- |
| OLMoE (MoEPTQ) | 3.26 | 15.44 | 26.04 | 20.74 | -- |
| **OLMoE (DynaMo)** | **2.95** | **9.64** | **15.31** | **12.48** | **-2.78** |
| MoE-Girl (FP16) | 16 | 8.43 | 13.13 | 10.78 | -- |
| MoE-Girl (GPTQ) | 3.00 | 12.77 | 21.38 | 17.08 | -- |
| MoE-Girl (MoEPTQ) | 3.26 | 16.88 | 29.40 | 23.14 | -- |
| **MoE-Girl (DynaMo)** | **2.89** | **10.47** | **17.87** | **14.17** | **-2.91** |
| Qwen1.5-MoE (FP16) | 16 | 7.02 | 10.03 | 8.53 | -- |
| Qwen1.5-MoE (GPTQ) | 3.00 | 10.99 | 20.84 | 15.92 | -- |
| Qwen1.5-MoE (MoEPTQ) | 3.35 | 9.93 | 18.49 | 14.21 | -- |
| **Qwen1.5-MoE (DynaMo)** | **3.05** | **8.51** | **14.24** | **11.38** | **-4.54** |
| DS-MoE (FP16) | 16 | 7.36 | 9.22 | 8.29 | -- |
| DS-MoE (GPTQ) | 3.00 | 10.47 | 15.19 | 12.83 | -- |
| DS-MoE (MoEPTQ) | 3.31 | 8.49 | 15.61 | 12.05 | -- |
| **DS-MoE (DynaMo)** | **2.98** | **7.94** | **11.49** | **9.72** | **-3.11** |

DynaMo achieves 2.78-4.54 PPL reduction versus the best baseline at comparable or lower bit-width.

### Zero-Shot Accuracy Results (Table II)

| Model | Method | Bits | ARC-c | ARC-e | RTE | PIQA | COPA | CB | Avg |
|---|---|---|---|---|---|---|---|---|---|
| OLMoE | FP16 | 16 | 29.69 | 48.48 | 54.51 | 61.86 | 71.00 | 41.07 | 51.10 |
| OLMoE | GPTQ | 3.00 | 25.34 | 41.12 | 51.99 | 58.81 | 62.00 | 39.29 | 46.43 |
| OLMoE | MoEPTQ | 3.26 | 24.83 | 38.38 | 50.54 | 56.86 | 65.00 | 42.86 | 42.27 |
| **OLMoE** | **DynaMo** | **2.97** | **25.85** | **43.52** | **54.15** | **58.87** | **65.00** | **46.43** | **48.94** |
| MoE-Girl | FP16 | 16 | 31.31 | 50.84 | 55.95 | 62.62 | 66.00 | 41.07 | 51.30 |
| MoE-Girl | GPTQ | 3.00 | 25.17 | 38.80 | 55.59 | 60.33 | 62.00 | 39.28 | 46.86 |
| MoE-Girl | MoEPTQ | 3.26 | 24.23 | 37.04 | 53.06 | 57.88 | 59.00 | 41.07 | 45.38 |
| **MoE-Girl** | **DynaMo** | **2.88** | **26.02** | **44.87** | **53.43** | **61.32** | **62.00** | **44.64** | **48.71** |
| Qwen1.5 | FP16 | 16 | 33.11 | 51.30 | 71.84 | 72.47 | 81.00 | 25.01 | 55.79 |
| Qwen1.5 | GPTQ | 3.00 | 26.54 | 39.44 | 54.51 | 63.87 | 72.00 | 24.76 | 46.85 |
| Qwen1.5 | MoEPTQ | 3.36 | 24.23 | 37.04 | 53.07 | 63.22 | 67.00 | 24.37 | 44.82 |
| **Qwen1.5** | **DynaMo** | **3.04** | **27.13** | **43.69** | **55.23** | **68.72** | **75.00** | **33.93** | **50.62** |
| DS-MoE | FP16 | 16 | 40.61 | 71.55 | 54.51 | 76.22 | 82.00 | 41.07 | 60.99 |
| DS-MoE | GPTQ | 3.00 | 33.62 | 62.04 | 52.34 | 73.94 | 78.00 | 44.64 | 57.43 |
| DS-MoE | MoEPTQ | 3.36 | 31.99 | 60.06 | 53.43 | 69.42 | 77.00 | 41.07 | 55.50 |
| **DS-MoE** | **DynaMo** | **2.99** | **34.71** | **64.19** | **53.92** | **75.34** | **81.00** | **47.87** | **59.51** |

DynaMo improves average zero-shot accuracy by 1.85%-3.77% versus the best baseline.

### Inference Speedup

DynaMo achieves **2.91-3.08x speedup** over FP16 across all four models. It also shows slight speed improvements over GPTQ and MoEPTQ baselines. The primary speedup source is packing multiple low-precision values into INT32 for efficient GMEM loading, maximizing memory bandwidth utilization.

### Ablation Study (Table III)

| Model | Method | Wiki PPL | CB Acc |
|---|---|---|---|
| OLMoE | GPTQ | 11.65 | 39.29% |
| OLMoE | Only BQ | 10.78 (-0.87) | 42.79% (+3.50%) |
| OLMoE | BQ + DQS | 9.64 (-1.23) | 46.43% (+3.64%) |
| MoE-Girl | GPTQ | 12.77 | 39.28% |
| MoE-Girl | Only BQ | 11.23 (-1.54) | 42.66% (+3.38%) |
| MoE-Girl | BQ + DQS | 10.47 (-0.76) | 44.64% (+1.98%) |
| Qwen1.5 | GPTQ | 10.99 | 24.76% |
| Qwen1.5 | Only BQ | 9.82 (-1.17) | 29.69% (+4.93%) |
| Qwen1.5 | BQ + DQS | 8.51 (-1.31) | 33.93% (+4.24%) |
| DS-MoE | GPTQ | 10.47 | 44.64% |
| DS-MoE | Only BQ | 8.52 (-1.95) | 45.91% (+1.27%) |
| DS-MoE | BQ + DQS | 7.94 (-0.58) | 47.87% (+1.96%) |

Both components contribute independently. Expert-level baseline quantization (BQ) alone reduces PPL by 0.87-1.95 and improves accuracy by 1.27-4.93%. Adding channel-level dynamic switching (DQS) provides further gains of 0.58-1.31 PPL and 1.96-4.24% accuracy.

### Overhead

- **Offline phase:** Multi-level analysis accounts for 14.90-17.33% of offline time; baseline quantization takes the rest.
- **Online phase:** Dynamic switching accounts for only 7.73-10.68% of online time; inference takes the rest.
- Switching is triggered only on dataset changes, so its cost is amortized over all inference on the new dataset.
- The paper characterizes total overhead as "negligible."

---

## 9. Limitations and Gaps

The paper is a 7-page conference-style submission. Several aspects are underspecified or absent:

### Underspecified: S_ch Definition

The channel-weight significance function `S_ch(.)` is described qualitatively ("fine-grained significance analysis," "mapping correlation to each token") but never given a precise formula. The paper says a hook captures output logits of each channel weight, but does not specify whether `S_ch` is the mean activation magnitude, variance, L2 norm, or something else. This makes the method difficult to reproduce exactly.

### Underspecified: The 1% Selection Criterion

The paper states that ~1% of channels show "the most prominent dynamics" and are selected for caching. It does not specify:
- How "prominence of dynamics" is measured (e.g., variance of S_ch across datasets? absolute change? relative change?).
- Whether 1% is a fixed threshold or a model-dependent hyperparameter.
- How sensitive results are to this threshold.

The claim that prior work supports this 1% figure is made without citation in the extracted text.

### Underspecified: Switching Trigger Detection

The paper says switching is triggered by a "new dataset" but does not specify:
- How the system detects that the dataset has changed.
- Whether this requires explicit user notification or automatic distribution shift detection.
- What happens if the distribution shifts gradually rather than abruptly.

In practice, dataset change detection is a non-trivial problem (distribution shift detection), and the paper sidesteps it entirely.

### Hard Assignment from Fuzzy Memberships

Algorithm 1 produces soft memberships, but the system ultimately needs a hard precision assignment for each expert. The paper does not explicitly state the conversion rule. The most natural choice (argmax) is implied but not confirmed. This matters for boundary cases where an expert has nearly equal membership in two clusters.

### No Explicit Limitations Section

The paper has no dedicated limitations section. It does not discuss:
- Failure modes (e.g., what happens if the new dataset is very different from all calibration datasets).
- Sensitivity to the number of calibration datasets n.
- Whether the joint distribution fitting method (Eq. 2) is robust to outlier datasets.
- The choice of fuzzifier m and its effect on results.
- Behavior when expert routing is soft or dynamic (non-Top-K).

### Eq. 1 Ambiguity

The exact typeset form of Equation 1 is partially ambiguous in the paper. The denominator appears to normalize over all experts, but the precise summation structure (whether it sums over both i and j, or just j) is not entirely clear from the text. The interpretation given in this document (sum over all experts and all tokens) is the most consistent with the surrounding text.

### No Comparison to MxMoE

The paper does not compare against MxMoE, which is another mixed-precision MoE quantization method. This is a notable gap given that MxMoE is a relevant baseline.

---

## 10. Relevance to Our Work (TensorRT-LLM Dual-Tile MoE)

### Our System: Per-Step Expert Classification via setupDualTileInputs()

Our TensorRT-LLM dual-tile implementation performs runtime expert classification on every forward pass. The `setupDualTileInputs()` function applies masking to classify experts into "active" and "inactive" (or "heavy" and "light") categories based on the current batch's routing decisions. This classification runs every forward pass, making it fundamentally per-step rather than per-dataset.

This is a much finer temporal granularity than DynaMo's per-dataset switching.

### DynaMo's Switching Granularity vs. Ours

| Aspect | DynaMo | Our Dual-Tile System |
|---|---|---|
| Switching trigger | Dataset change | Every forward pass |
| Classification basis | Expert significance across dataset | Routing decisions for current batch |
| Adaptation speed | Slow (per-dataset) | Fast (per-step) |
| Overhead per switch | 13-21 ms (search) + re-quantization | Masking operation in setupDualTileInputs() |
| Precision levels | INT2/4/6/8 (4 levels) | Tile size / compute path (2 levels) |

DynaMo's switching is coarser in time but finer in precision (4 levels vs. our binary tile classification). Our system is finer in time but coarser in precision.

### Potential Combination: Per-Step Precision Co-Adaptation

The most interesting synthesis would combine our per-step expert classification with DynaMo-style precision switching:

**Concept:** Use our `setupDualTileInputs()` masking to identify, on every forward pass, which experts are "hot" (heavily activated) and which are "cold" (lightly activated or inactive). Then apply DynaMo-style precision assignment: hot experts get INT8, cold experts get INT4 or INT2.

This would be **truly dynamic per-step co-adaptation**: the quantization precision changes every forward pass based on the actual routing decisions for that batch, not just when the dataset changes.

**Challenges:**
1. Re-quantization overhead per step would be significant. DynaMo's 13-21 ms switching time is acceptable once per dataset but not once per forward pass (which might be 10-100 ms total).
2. The 1% channel cache would need to be larger or the switching faster to support per-step adaptation.
3. Precision switching at per-step granularity requires either pre-quantized weights at multiple precisions (memory cost) or fast on-the-fly quantization (compute cost).

**Feasible middle ground:** Per-step tile classification (our current approach) combined with per-dataset precision switching (DynaMo's approach). Our tile classification handles the fast, per-step adaptation of compute paths; DynaMo's precision switching handles the slower, per-dataset adaptation of weight precision. The two mechanisms operate at different timescales and are complementary rather than redundant.

**Another angle:** DynaMo's ~1% dynamic channel identification could inform our tile design. If the most dynamic channels are known, they could be placed in a separate tile that gets special treatment (e.g., always computed in FP16 or INT8 regardless of the expert's overall precision assignment). This would be a channel-aware tile design rather than purely expert-aware.

---

## 11. Comparison Table

| Aspect | DynaMo | MxMoE | Our Dual-Tile Approach |
|---|---|---|---|
| **Core idea** | Expert-level mixed precision + channel-level dynamic switching | Mixed precision for MoE experts | Dual-tile compute path selection per expert per step |
| **Quantization granularity** | Expert-level (coarse) + channel-level (fine, 1%) | Expert-level | Not quantization-focused; tile size selection |
| **Precision levels** | INT2, INT4, INT6, INT8 | INT4, INT8 (typical) | N/A (FP16/BF16 compute paths) |
| **Adaptation trigger** | Dataset change (coarse) | Static (offline only) | Every forward pass (fine) |
| **Adaptation basis** | Expert significance via S_ch and joint distribution | Expert importance (static) | Routing decisions for current batch |
| **Dynamic component** | ~1% most dynamic channels cached in FP16 | None (fully static) | setupDualTileInputs() masking every step |
| **Offline cost** | Multi-level analysis + fuzzy c-means clustering | Calibration + clustering | Profiling for tile size selection |
| **Online cost** | 7.73-10.68% of inference time (per dataset change) | None | Masking overhead per forward pass |
| **Speedup vs FP16** | 2.91-3.08x | Comparable to GPTQ | Depends on tile efficiency gains |
| **PPL improvement** | 2.78-4.54 vs GPTQ/MoEPTQ | Varies | Not directly comparable |
| **Accuracy improvement** | 1.85-3.77% vs GPTQ/MoEPTQ | Varies | Not directly comparable |
| **Cross-dataset robustness** | Yes (explicit design goal) | No (single-dataset calibration) | Yes (per-step adaptation) |
| **Hardware target** | NVIDIA A100 (CUDA cores + tensor cores) | GPU | NVIDIA GPU (SM120 / dual-tile) |
| **Implementation** | Fused dequant + MMA kernel, INT32 packing | Standard quantization kernels | Custom dual-tile MoE kernel |
| **Key limitation** | Per-dataset switching only; S_ch underspecified | Static; no cross-dataset adaptation | No precision adaptation; tile selection only |
| **Synergy potential** | High: per-step tile classification + per-dataset precision switching | Low: fully static | -- |

### Summary Assessment

DynaMo is the most relevant prior work for our dual-tile system because it explicitly addresses the cross-dataset dynamics of MoE models, which is the same problem our per-step tile classification implicitly handles. The key difference is temporal granularity: DynaMo adapts per-dataset, we adapt per-step.

DynaMo's channel-level analysis (identifying the ~1% most dynamic channels) is a potentially valuable tool for our tile design: if we know which channels drive the dynamics, we can design tiles that isolate those channels for special treatment. DynaMo's fuzzy c-means clustering approach for precision assignment could also be adapted to our tile size selection problem, replacing or augmenting our current masking heuristics with a principled significance-based assignment.

The most promising direction is a hybrid: use DynaMo's offline analysis to identify dynamic channels and assign baseline precisions, then use our per-step tile classification to handle within-dataset batch-level variation. This would give us both the cross-dataset robustness of DynaMo and the per-step adaptability of our dual-tile system.

---

*End of analysis. Document covers all sections as specified: paper metadata, problem statement, key insight, full technical approach (multi-level analysis, expert-level baseline quantization with Algorithm 1, channel-level dynamic switching with step-by-step procedure and concrete example, implementation details), granularity discussion, static vs. dynamic comparison, all formulas, experimental results, limitations, relevance to our work, and comparison table.*
