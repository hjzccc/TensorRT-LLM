# Mixture Compressor (MC): Technical Analysis

> **Paper:** Mixture Compressor for Mixture-of-Experts LLMs Gains More  
> **Venue:** ICLR 2025  
> **arXiv:** 2410.06270v2 [cs.LG], 22 Feb 2025

---

## 1. Paper Metadata

| Field | Value |
|---|---|
| Title | Mixture Compressor for Mixture-of-Experts LLMs Gains More |
| Authors | Wei Huang\*, Yue Liao\*, Jianhui Liu, Ruifei He, Haoru Tan, Shiming Zhang†, Hongsheng Li, Si Liu†, Xiaojuan Qi† |
| Affiliations | University of Hong Kong, Chinese University of Hong Kong, Beihang University, CPII Hong Kong |
| Venue | ICLR 2025 (conference paper) |
| arXiv ID | 2410.06270v2 |
| Code | Not linked in paper text |

\* Equal contribution. † Corresponding authors.

---

## 2. Problem Statement

MoE-LLMs (Mixture-of-Experts Large Language Models) achieve strong performance by routing each token to a small subset of expert feed-forward networks, keeping active compute low. But deployment is still painful for two distinct reasons:

**Memory pressure from static storage.** All expert weights must reside in memory even though only 2 of 8 experts fire per token. For Mixtral 8x7b, expert parameters are 33x larger than attention parameters and account for over 96% of total weights. The full model occupies 96.8 GB in FP16, requiring at least two A100-80GB GPUs.

**Redundancy in dynamic activation.** Even among the top-2 routed experts, the second expert often contributes very little. Its routing weight is much smaller than the first expert's, meaning the model is effectively running two experts when one would suffice for most tokens.

Prior work addresses these problems in isolation:
- Uniform low-bit quantization (e.g., GPTQ at 2-bit) collapses on hard tasks because it treats all experts identically, ignoring that some experts are far more sensitive than others.
- Routing-score-only expert pruning misses the fact that routing scores and activation frequencies can disagree, and naively pruning experts for all tokens causes "attention decay" in subsequent blocks.

MC's thesis is that **static expert storage compression and dynamic expert activation pruning must be optimized jointly**, because expert imbalance manifests both statically (some experts are rarely used) and dynamically (some tokens need only one expert).

---

## 3. Key Insight / Core Idea

Three empirical observations drive the design:

**Observation 1: Experts are not equally important, and importance is multi-dimensional.** Heatmaps of expert-drop F-norm, average routing score, and activation frequency across 32 layers x 8 experts of Mixtral 8x7b show clear imbalance. Crucially, routing score and activation frequency do not always agree. A high routing score for a rarely-activated expert is different from a high routing score for a frequently-activated one. Using only one signal produces suboptimal bit allocation.

**Observation 2: Expert importance is task-dependent.** On math tasks, expert usage is sparser than on general text (C4). This means a fixed static allocation calibrated on one domain may not generalize perfectly, but it also means that domain-specific compression could be even more aggressive.

**Observation 3: Not all tokens are equally important, and protecting a tiny fraction prevents catastrophic pruning damage.** When experts are pruned for a salient token in one block, the attention map in the next block degrades ("attention decay"). Protecting just the top 2% of tokens by importance score recovers most of the performance loss while preserving nearly all the compression benefit.

The core idea: assign higher bit-widths to more important/sensitive experts (static), then prune the second expert for low-confidence tokens at runtime (dynamic), but never prune experts for the most important tokens.

---

## 4. Technical Approach: Step by Step

MC has two components that operate at different times:

- **PMQ (Pre-Loading Mixed-Precision Quantization):** runs once before deployment, assigns per-expert bit-widths, quantizes weights.
- **ODP (Online Dynamic Pruning):** runs at inference time, decides per-token whether to skip the second expert.

### 4.1 MoE Forward Pass (Baseline)

For a token $t$, the MoE layer output is:

$$y = \sum_{w_i \in \text{Top-}2\{G(t)\}} w_i E_i(t)$$

where $E_i$ is the $i$-th expert feed-forward network and $w_i$ is the routing weight from gating function $G(t)$. Standard top-2 routing means two experts always fire.

### 4.2 Static Mixed-Precision Quantization (PMQ)

**Goal:** Compress expert weights before loading to reduce memory footprint. Assign bit-widths from $\{1, 2, 3\}$ bits per expert, with higher bits for more important/sensitive experts.

**Step 1: Measure expert significance.**

Run calibration inference on 128 random C4 sequences (length 2048) using the full FP16 model. For each expert $i$, compute:

*Activation frequency:*
$$\phi_i = \frac{n_i}{N}$$

where $n_i$ is the total number of times expert $i$ is activated across the calibration set and $N$ is the dataset size.

*Activation-weighted routing importance:*
$$w_i = \frac{\sum_{j=1}^{N} \sigma_j}{N}$$

where $\sigma_j$ is the routing weight assigned to expert $i$ at inference step $j$ (zero when expert $i$ is not selected).

Combined expert significance: $\phi_i^\alpha \cdot w_i^\beta$, where $\alpha$ and $\beta$ are hyperparameters (ablations show results are stable for $\alpha, \beta \in \{1, 1.5, 2\}$).

**Step 2: Measure quantization sensitivity.**

For each expert $e_i$ and each candidate bit-width $j \in \{1, 2, 3\}$, compute the reconstruction error when only that expert is quantized:

$$\epsilon_{i,j} = \|F(\theta) - F(\theta[e_i \to Q(e_i, j)])\|_F$$

where $F(\theta)$ is the model output with full FP16 parameters, $F(\theta[e_i \to Q(e_i, j)])$ is the output when expert $i$ alone is quantized to $j$ bits, and $\|\cdot\|_F$ is the Frobenius norm. This is computed layer by layer using GPTQ's Hessian estimate $H = 2XX^T$.

**Step 3: Solve the integer program.**

Minimize total weighted reconstruction error subject to a target average bit-width $k$:

$$\text{minimize} \sum_{i=1}^{n} \sum_{j=1}^{3} \phi_i^\alpha \cdot w_i^\beta \cdot (\epsilon_{i,j} \cdot x_{ij})^\gamma$$

subject to:

$$\sum_{i=1}^{n} \sum_{j=1}^{3} j \cdot x_{ij} = n \cdot k \quad \text{(average bit-width constraint)}$$

$$\sum_{j=1}^{3} x_{ij} = 1, \quad \forall i \quad \text{(each expert gets exactly one bit-width)}$$

$$\sum_{i=1}^{n} x_{i3} \ge 1 \quad \text{(at least one 3-bit expert per block)}$$

$$\sum_{i=1}^{n} x_{i2} \ge 1 \quad \text{(at least one 2-bit expert per block)}$$

$$x_{ij} \in \{0, 1\}, \quad \forall i, j$$

Here $x_{ij} = 1$ iff expert $i$ is assigned $j$ bits, $n$ is the number of experts per block, and $\gamma$ controls how strongly quantization error is penalized relative to expert significance. The constraints ensure diversity in bit-widths (no block collapses to all-1-bit) and that the average bit-width across all experts in the block equals the target $k$.

**Step 4: Quantize with GPTQ.**

Apply GPTQ at the assigned bit-width for each expert. Non-expert modules (attention, gating) are quantized uniformly to 4-bit. The 4-bit non-expert contribution adds at most ~0.05 bits to the overall average.

**Compatibility:** PMQ is backend-agnostic. The paper demonstrates it works with both GPTQ and OmniQuant, with OmniQuant giving slightly better results (+1.3 points average at 2.54-bit).

### 4.3 1-Bit Weight Storage and Dequantization

For experts assigned 1-bit, the paper uses a binarization scheme:

$$B = \text{sign}(W), \quad \text{sign}(x) = \begin{cases} 1 & x \ge 0 \\ -1 & \text{otherwise} \end{cases}$$

To store as actual 1-bit values:

$$\tilde{B} = \frac{\text{sign}(W) + 1}{2}, \quad \tilde{B} \in \{0, 1\}^{d \times m}$$

Dequantized matrix-vector product:

$$s \cdot xB = s \left( \sum_{j:\tilde{B}_{ij}=1} x_j - \sum_{j:\tilde{B}_{ij}=0} x_j \right), \quad i = 1, \dots, m$$

with scaling factor:

$$s = \frac{\|W\|_{\ell_1}}{d \times m}$$

This reduces multiply-accumulate operations from $O(dm)$ to $O(m)$ for the binary case. HQQ is used for weight storage and dequantization across all bit-widths.

### 4.4 Dynamic Expert Pruning (ODP)

**Goal:** At inference time, skip the second expert for tokens where it contributes little, reducing activated parameters by ~15%.

**Step 1: Weight-guided pruning rule.**

For top-2 routing with experts having weights $w_0 \ge w_1$, prune the second expert when:

$$\frac{w_1}{w_0} < \mu$$

where $\mu$ is a per-layer threshold set to the **median** of $w_1/w_0$ ratios observed on calibration data. When the ratio is below the median, the second expert's contribution is small enough to skip.

This alone reduces activated parameters by ~15% but causes ~10% performance drop due to attention decay.

**Step 2: Token importance scoring.**

Compute importance for each token $j$:

$$I_j = \|t_j\|_1 \cdot \frac{\sum_{j \le i \le L} A_{j,i}}{L - j}$$

where:
- $\|t_j\|_1$ is the L1 norm of the token's feature vector (magnitude)
- $A_{j,i}$ is the attention score from token $j$ to token $i$, from $A = \text{softmax}\!\left(\frac{K^T Q}{\sqrt{d_k}}\right)$
- $L$ is the sequence length
- The fraction captures how much attention token $j$ receives from subsequent tokens (its "influence" on the rest of the sequence)

**Step 3: Protect important tokens.**

Sort tokens by $I_j$. The top 2% by importance are **never** subject to expert pruning, regardless of the routing weight ratio. For all other tokens, apply the weight-guided pruning rule from Step 1.

This protection costs almost nothing in compression (reduces pruned parameters from 15.1% to 14.8%) but recovers most of the performance loss from attention decay.

**Computational overhead of ODP:** The token importance computation requires attention summation, L1 norm, and top-k selection, costing $O(n^2 + n + mn + n \log n)$ FLOPs. Since expert FFN dimensions $m_1 \gg n$, this overhead is negligible compared to the expert compute saved.

### 4.5 Combined Optimization

PMQ and ODP are complementary:

- PMQ reduces **stored model size** (from 96.8 GB to 16.2 GB for Mixtral 8x7b at 2.54-bit).
- ODP reduces **runtime activated parameters** (from 4.53 GB to 3.96 GB at 2.54-bit, an additional 12.6% reduction in active compute).
- Together they achieve both storage efficiency and inference efficiency.

The two components do not share a joint objective function. They are designed and applied sequentially: PMQ first (offline), ODP second (online). Their interaction is that PMQ-quantized experts are the ones being selectively skipped by ODP.

### 4.6 Note on "MC#"

The paper does not define a variant called "MC#". The abstract and introduction mention compatibility with QuIP# and OmniQuant as alternative quantization backends, but these are existing methods, not a new MC variant. The "MC#" label may refer informally to MC with QuIP# as the quantization backend, but this is not formalized in the paper.

---

## 5. Granularity

| Component | Granularity |
|---|---|
| Expert bit-width assignment | Per-expert, per MoE block |
| Bit-width options | {1, 2, 3} bits for experts; 4-bit for attention/gating |
| Expert significance calibration | Per-expert across all calibration tokens |
| Quantization sensitivity | Per-expert, per candidate bit-width |
| IP optimization | Per MoE block (each block solved independently) |
| Dynamic pruning threshold $\mu$ | Per MoE layer (calibrated separately for each layer) |
| Token importance | Per token, per forward pass |
| Important-token protection | Top 2% of tokens by $I_j$ score |

The per-expert granularity within each block is the key design choice. Coarser granularity (e.g., per-layer uniform) would miss the within-block expert imbalance that the paper's heatmaps clearly show.

---

## 6. Static vs. Dynamic: The Hybrid Architecture

MC is explicitly a **hybrid** compression system:

| Dimension | PMQ (Static) | ODP (Dynamic) |
|---|---|---|
| When | Offline, before deployment | Online, per inference step |
| What | Expert weight bit-widths | Which experts to activate |
| Criterion | Significance + sensitivity | Routing weight ratio + token importance |
| Cost | One-time (90 min for 8x7b on 2xA100) | Per-token overhead (negligible) |
| Benefit | Reduces stored model size | Reduces runtime activated parameters |
| Reversible | No (weights are permanently quantized) | Yes (different tokens get different experts) |

The static/dynamic split maps naturally onto the two distinct deployment bottlenecks: storage (static) and compute/memory bandwidth (dynamic). Neither alone achieves the full compression target without unacceptable accuracy loss.

---

## 7. Formulas and Algorithms

### Algorithm 1: PMQ (Pre-Loading Mixed-Precision Quantization)

```
Input: MoE model θ, calibration data D, target bit-width k, hyperparameters α, β, γ
Output: Quantized model θ_q

For each MoE block b:
  1. Run calibration inference on D with full FP16 model
     - Record n_i (activation count) and σ_j (routing weights) for each expert i
     - Compute φ_i = n_i / N  (activation frequency)
     - Compute w_i = mean(σ_j)  (routing importance)

  2. For each expert i, for each bit-width j ∈ {1, 2, 3}:
     - Compute ε_{i,j} = ||F(θ) - F(θ[e_i → Q(e_i, j)])||_F

  3. Solve integer program:
     minimize  Σ_i Σ_j φ_i^α · w_i^β · (ε_{i,j} · x_{ij})^γ
     subject to:
       Σ_i Σ_j j · x_{ij} = n · k
       Σ_j x_{ij} = 1  ∀i
       Σ_i x_{i3} ≥ 1
       Σ_i x_{i2} ≥ 1
       x_{ij} ∈ {0,1}

  4. Quantize expert i to x_{ij}* bits using GPTQ

5. Quantize all attention/gating weights to 4-bit
```

### Algorithm 2: ODP (Online Dynamic Pruning)

```
Input: Token sequence t_1, ..., t_L, quantized model θ_q, threshold μ_b per block, protection ratio p=0.02
Output: MoE outputs with selective expert pruning

1. Compute token importance scores:
   I_j = ||t_j||_1 · (Σ_{j≤i≤L} A_{j,i}) / (L - j)
   where A = softmax(K^T Q / sqrt(d_k))

2. Identify protected tokens: top p·L tokens by I_j score

3. For each MoE block b, for each token t_j:
   a. Compute routing weights w_0 ≥ w_1 from G(t_j)
   b. If t_j is a protected token:
      - Activate both top-2 experts (standard routing)
   c. Else if w_1 / w_0 < μ_b:
      - Activate only top-1 expert (prune second expert)
   d. Else:
      - Activate both top-2 experts
```

### Key Formulas Summary

| Formula | Meaning |
|---|---|
| $y = \sum_{w_i \in \text{Top-}2\{G(t)\}} w_i E_i(t)$ | Standard MoE forward pass |
| $\phi_i = n_i / N$ | Expert activation frequency |
| $w_i = \frac{1}{N}\sum_j \sigma_j$ | Activation-weighted routing importance |
| $\epsilon_{i,j} = \|F(\theta) - F(\theta[e_i \to Q(e_i,j)])\|_F$ | Quantization reconstruction error |
| $\text{min} \sum_{i,j} \phi_i^\alpha w_i^\beta (\epsilon_{i,j} x_{ij})^\gamma$ | IP objective |
| $I_j = \|t_j\|_1 \cdot \frac{\sum_{j \le i \le L} A_{j,i}}{L-j}$ | Token importance score |
| $w_1 / w_0 < \mu$ | Expert pruning condition |
| $s = \|W\|_{\ell_1} / (d \times m)$ | 1-bit dequantization scale |

---

## 8. Experimental Results

### 8.1 Models Tested

| Model | Total Params | Activated Params | Experts/Block | Blocks |
|---|---|---|---|---|
| Mixtral 8x7b | ~49B (96.8 GB FP16) | ~13B (26.3 GB FP16) | 8 | 32 |
| Mixtral 8x22b | ~141B (281.2 GB FP16) | ~39B (76.5 GB FP16) | 8 | 56 |

### 8.2 The 76.6% Compression Claim

The paper claims "at 2.54 bits, MC compresses 76.6% of the model with only 3.8% average accuracy loss." From Table 4, the stored size drops from 96.8 GB to 16.2 GB, which is an 83.2% reduction in bytes. The 76.6% figure likely refers to a different compression metric (possibly parameter count or a weighted measure), but the paper does not derive it explicitly. What is directly verifiable from the tables:

- Stored size: 96.8 GB → 16.2 GB (83.2% reduction)
- Activated parameters: 26.3 GB → 3.96 GB with PMQ+ODP (84.9% reduction)
- Average accuracy loss on 8 benchmarks: 3.8% (71.29 → 66.94)

### 8.3 Mixtral 8x7b: 8-Benchmark Average (Zero-Shot)

| Method | Avg Bits | LM-Eval Avg | Drop |
|---|---|---|---|
| FP16 | 16.00 | 71.29 | — |
| Uniform GPTQ | 3.00 | 69.09 | 2.2% |
| Uniform GPTQ | 2.00 | 42.67 | 28.6% |
| BSP (Li et al.) | 2.54 | 49.07 | 22.2% |
| Hessian mixed | 2.54 | 67.18 | 4.1% |
| Hessian mixed | 2.05 | 58.85 | 12.4% |
| Hessian mixed | 1.57 | 45.91 | 25.4% |
| **PMQ** | **2.54** | **67.50** | **3.8%** |
| PMQ | 2.05 | 63.25 | 8.0% |
| PMQ | 1.57 | 54.49 | 16.8% |
| PMQ+ODP | 2.54 | 66.94 | 4.5% |
| PMQ+ODP | 2.05 | 62.68 | 8.6% |
| PMQ+ODP | 1.57 | 53.77 | 17.5% |

### 8.4 Per-Benchmark Breakdown (PMQ, Mixtral 8x7b)

| Bits | PIQA | ARC-e | ARC-c | BoolQ | HellaSwag | Winogrande | MathQA | MMLU |
|---|---|---|---|---|---|---|---|---|
| 16.00 | — | — | — | — | — | — | — | — |
| 2.54 | 80.52 | 77.10 | 51.28 | 82.54 | 79.03 | 73.95 | 39.18 | 56.37 |
| 2.05 | 79.16 | 73.06 | 48.38 | 80.58 | 74.95 | 71.27 | 31.79 | 46.80 |
| 1.57 | 72.42 | 62.46 | 37.88 | 73.55 | 63.17 | 66.38 | 26.80 | 32.25 |

### 8.5 WikiText2 Perplexity (Mixtral 8x7b)

| Method | Bits | PPL |
|---|---|---|
| FP16 | 16.00 | 3.84 |
| Uniform GPTQ | 2.00 | 16.38 |
| BSP | 2.54 | 13.61 |
| Hessian | 2.54 | 5.41 |
| Hessian | 2.05 | 6.65 |
| Hessian | 1.57 | 14.20 |
| PMQ | 2.54 | 5.09 |
| PMQ | 2.05 | 5.91 |
| PMQ | 1.57 | 8.50 |

PMQ consistently beats Hessian-based mixed precision, especially at sub-2-bit where the gap widens significantly.

### 8.6 Challenging Benchmarks (Mixtral 8x7b)

| Method | Bits | GSM8K | HumanEval | NIAH |
|---|---|---|---|---|
| FP16 | 16.00 | 58.30 | 59.15 | 100.00 |
| Uniform GPTQ | 3.00 | 38.13 | 29.88 | 98.48 |
| Uniform GPTQ | 2.00 | 0.00 | 0.00 | 0.00 |
| BSP | 2.54 | 4.25 | 3.21 | 42.21 |
| Hessian | 2.54 | 33.59 | 25.49 | 100.00 |
| PMQ | 2.54 | 37.67 | 29.34 | 100.00 |
| PMQ | 2.05 | 19.97 | 11.83 | 100.00 |
| PMQ+ODP | 2.54 | 35.25 | 27.58 | 100.00 |
| PMQ+ODP | 2.05 | 18.04 | 10.02 | 99.26 |

PMQ retains needle-in-a-haystack (NIAH) retrieval perfectly at 2.54-bit, while BSP collapses to 42.21%. Reasoning tasks (GSM8K, HumanEval) degrade more than standard benchmarks, especially below 2-bit.

### 8.7 Memory and Speed (Mixtral 8x7b)

| Method | Bits | Stored (GB) | Act Params (GB) | Speedup |
|---|---|---|---|---|
| FP16 | 16.00 | 96.80 | 26.31 | 1.00x |
| Uniform GPTQ | 2.00 | 13.61 | 3.70 | 1.72x |
| PMQ | 2.54 | 16.24 | 4.53 | 1.63x |
| PMQ+ODP | 2.54 | 16.24 | 3.96 | 1.71x |
| PMQ | 2.05 | 13.41 | 3.73 | 1.67x |
| PMQ+ODP | 2.05 | 13.41 | 3.23 | 1.80x |
| PMQ | 1.57 | 10.82 | 2.94 | 1.82x |
| PMQ+ODP | 1.57 | 10.82 | 2.55 | 1.89x |

### 8.8 Comparison to Dense LLMs

| Model | Bits | LM-Eval | Stored (GB) | Act Params (GB) |
|---|---|---|---|---|
| LLaMA2-7b | FP16 | 61.52 | 13.48 | 13.48 |
| LLaMA2-13b | FP16 | 65.19 | 26.03 | 26.03 |
| Mixtral 8x7b PMQ | 2.54 | 67.50 | 16.24 | 4.53 |
| Mixtral 8x7b PMQ+ODP | 2.54 | 66.94 | 16.24 | 3.96 |

Compressed Mixtral 8x7b at 2.54-bit outperforms FP16 LLaMA2-13b on LM-Eval while using fewer activated parameters (3.96 GB vs 26.03 GB).

### 8.9 Hardware Deployment

| Setup | Loading (GB) | Peak GPU (GB) | LM-Eval | Throughput |
|---|---|---|---|---|
| Mixtral 8x7b FP16, 2xA100 | 96.8 | 112.6 | 71.29 | 23 tok/s |
| Mixtral 8x7b FP16, 1x3090 | OOM | OOM | — | — |
| LLaMA2-13b FP16, 1xA100 | 26.0 | 33.4 | 65.19 | 46 tok/s |
| LLaMA2-13b FP16, 1x3090 | OOM | OOM | — | — |
| MC 2.54-bit, 1xA100 | 16.2 | 20.7 | 66.94 | 38 tok/s |
| MC 2.54-bit, 1x3090 | 16.2 | 20.7 | 66.90 | 52 tok/s |

MC enables Mixtral 8x7b to run on a single consumer GPU (RTX 3090) with better accuracy than LLaMA2-13b and higher throughput than FP16 Mixtral on 2xA100.

### 8.10 OmniQuant Backend

| Method | Bits | LM-Eval Avg |
|---|---|---|
| PMQ + GPTQ | 2.54 | 67.50 |
| PMQ + OmniQuant | 2.54 | 68.80 |
| PMQ + GPTQ | 2.05 | 63.25 |
| PMQ + OmniQuant | 2.05 | 64.01 |
| PMQ + GPTQ | 1.57 | 54.49 |
| PMQ + OmniQuant | 1.57 | 55.79 |

OmniQuant consistently adds ~1-1.5 points but takes ~480 minutes vs ~90 minutes for GPTQ.

---

## 9. Implementation Details

### Hardware and Time

| Task | Hardware | Time |
|---|---|---|
| PMQ calibration + GPTQ (8x7b) | 2x NVIDIA A100-80GB | ~90 min |
| PMQ calibration + OmniQuant (8x7b) | 2x NVIDIA A100-80GB | ~480 min |
| PMQ calibration + GPTQ (8x22b) | 4x NVIDIA A100-80GB | Not stated |

### Calibration Data

- 128 random C4 sequences, each 2048 tokens long (main experiments)
- 256 C4 sequences for OmniQuant experiments

### Quantization Backend

- GPTQ for main results (Hessian $H = 2XX^T$, error compensation)
- OmniQuant as alternative backend
- HQQ for weight storage and dequantization across all bit-widths

### Hyperparameters

| Parameter | Value | Role |
|---|---|---|
| $\alpha$ | 1 (default) | Frequency weight in significance |
| $\beta$ | 1 (default) | Routing weight in significance |
| $\gamma$ | 2 | Quantization error weight in IP objective |
| $\mu$ | Per-layer median of $w_1/w_0$ | Expert pruning threshold |
| Protection ratio $p$ | 0.02 (2%) | Fraction of tokens protected from pruning |
| Expert bit-width options | {1, 2, 3} | Candidate bit-widths for experts |
| Non-expert bit-width | 4 | Fixed for attention/gating |

### ODP Computation Cost

Token importance computation FLOPs per layer:

$$n^2 + n + mn + n \log n$$

Expert skipping FLOPs saved (15% of tokens, one expert with layers $m \times m_1$, $m_1 \times m_1$, $m_1 \times m$):

$$0.15n \times (2mm_1 + 2m_1^2 + 2m_1 m)$$

Since $m_1 \gg n$, the importance computation overhead is negligible.

---

## 10. Limitations and Gaps

### Stated or Implied Limitations

**Reasoning task fragility.** GSM8K and HumanEval degrade much more than standard LM-Eval benchmarks under compression. At 2.05-bit, GSM8K drops from 58.30 to 19.97 (66% relative drop) while LM-Eval average drops only 8%. Chain-of-thought reasoning is more sensitive to quantization noise than single-step classification.

**Attention decay is a real failure mode.** Weight-only pruning without token protection causes ~10% performance drop. The fix (protecting 2% of tokens) works, but it reveals that the pruning criterion is fragile for salient tokens. The paper does not analyze what happens when the important-token set is misidentified.

**Exponential degradation from aggressive token skipping.** Dropping all experts for the least important tokens degrades performance exponentially as the skip ratio increases. This limits how aggressively ODP can be pushed.

**Static calibration for dynamic behavior.** The pruning threshold $\mu$ is calibrated on C4 but applied to arbitrary inference inputs. For highly specialized domains (math, code), the calibrated threshold may be suboptimal.

**No dedicated Limitations section.** The paper does not formally enumerate limitations, making it harder to assess scope.

**76.6% compression claim is not derived.** The paper states this number but Table 4 shows an 83.2% byte reduction. The discrepancy is unexplained.

**No training-time adaptation.** MC is entirely training-free. Quantization-aware training or LoRA fine-tuning after compression could recover more accuracy, but this is not explored.

**Single MoE architecture.** All experiments use Mixtral 8x7b and 8x22b. Generalization to other MoE architectures (DeepSeek-MoE, Switch Transformer, Qwen-MoE) is not demonstrated.

**No latency breakdown for ODP.** The paper shows per-token generation latency but does not isolate the ODP overhead from the quantization speedup.

### Gaps Relative to Production Deployment

- No INT4/INT8 kernel optimization for the mixed-precision weights (relies on HQQ dequantization, which may not be optimal on all hardware).
- No analysis of KV cache interaction with ODP (pruning experts changes hidden states, which affects KV cache quality for subsequent tokens).
- No multi-GPU tensor parallelism analysis for the mixed-precision case.

---

## 11. Relevance to Our Work

### Our Context

We profile expert activation patterns in MoE models and observe that approximately 40% of experts are idle (never or rarely activated) during typical inference. We are working on tile-level optimizations in TensorRT-LLM, where small tiles for rarely-used experts waste GPU resources.

### What MC Offers

**Direct alignment: MC's ODP is exactly the "skip idle experts" optimization we need.** The weight-guided pruning rule ($w_1/w_0 < \mu$) provides a principled, per-token criterion for deciding when to skip the second expert. This is more principled than a static threshold on activation frequency alone.

**The 40% idle expert observation maps to MC's findings.** MC shows that ~15% of activated expert compute can be pruned dynamically with minimal accuracy loss. Our 40% idle figure is about experts that are never selected by the router, which is a stronger statement. MC's static PMQ would assign these experts 1-bit or 2-bit, and ODP would further reduce their runtime cost. Together, the two mechanisms could eliminate most of the wasted tile work.

**Token importance protection is critical for correctness.** Our tile optimization must not skip experts for salient tokens. MC's $I_j$ score (combining token magnitude and attention influence) provides a lightweight, per-token signal that can be computed with negligible overhead. We should implement this check before deciding to skip an expert's tile.

**Specific integration points:**

1. **Static expert ranking for tile scheduling.** Use MC's expert significance metric ($\phi_i^\alpha \cdot w_i^\beta$) to rank experts. Low-significance experts can be scheduled with smaller tiles or lower priority, reducing wasted GPU cycles when they are activated.

2. **Dynamic tile skipping.** Implement ODP's pruning condition as a pre-dispatch check in TensorRT-LLM's MoE kernel. If $w_1/w_0 < \mu$ and the token is not in the top-2% importance set, skip the second expert's tile entirely rather than launching a small, inefficient tile.

3. **Mixed-precision tile dispatch.** MC's per-expert bit-width assignment means different experts have different weight sizes. Our tile scheduler can use this to pack tiles more efficiently: 1-bit expert tiles are much smaller and can be batched differently from 3-bit expert tiles.

4. **Calibration reuse.** MC's calibration (128 C4 sequences) is cheap and produces both the bit-width assignment and the per-layer pruning threshold $\mu$. We can run this calibration once and use the outputs to configure both the quantization and the tile scheduler.

### Key Difference from Our Current Approach

Our current tile optimization runs all experts but tries to make small tiles efficient. MC's insight is that **the right answer for ~15% of expert activations is to not run them at all**, not to run them with a small tile. For the 40% of experts that are structurally idle (never selected by the router), MC's PMQ would compress them aggressively (1-bit), and ODP would skip them dynamically. Our tile scheduler should treat these as zero-cost rather than small-cost.

### Risk: Attention Decay

MC's attention decay finding is a warning for our work. If we skip experts for salient tokens (e.g., the first token, punctuation tokens, or tokens with high attention scores), we can corrupt the attention map in subsequent blocks. The 2% protection rule is cheap to implement and should be included in any expert-skipping optimization.

---

## 12. Comparison Table

### MC vs. Related Compression Methods (Mixtral 8x7b, ~2.54-bit)

| Method | Type | Bits | LM-Eval | WikiText2 PPL | GSM8K | NIAH | Notes |
|---|---|---|---|---|---|---|---|
| FP16 baseline | None | 16.00 | 71.29 | 3.84 | 58.30 | 100.00 | 2xA100 required |
| Uniform GPTQ | Static quant | 3.00 | 69.09 | — | 38.13 | 98.48 | All experts same bits |
| Uniform GPTQ | Static quant | 2.00 | 42.67 | 16.38 | 0.00 | 0.00 | Collapses on hard tasks |
| BSP | Static quant | 2.54 | 49.07 | 13.61 | 4.25 | 42.21 | Routing-score-only allocation |
| Hessian mixed | Static quant | 2.54 | 67.18 | 5.41 | 33.59 | 100.00 | Hessian-based bit allocation |
| **PMQ (MC)** | **Static mixed quant** | **2.54** | **67.50** | **5.09** | **37.67** | **100.00** | **Significance + sensitivity IP** |
| PMQ+ODP (MC) | Static quant + dynamic prune | 2.54 | 66.94 | — | 35.25 | 100.00 | Full MC system |
| PMQ+OmniQuant | Static mixed quant | 2.54 | 68.80 | — | — | — | Better backend, slower |

### MC Design Choices vs. Alternatives

| Design Dimension | MC Choice | Alternative | Why MC Wins |
|---|---|---|---|
| Expert importance signal | Frequency + routing weight | Routing weight only | Frequency and routing weight disagree; both needed |
| Quantization sensitivity | F-norm reconstruction error | Hessian-based | F-norm captures output-level impact better at ultra-low bits |
| Bit allocation | Integer programming | Greedy / random | IP finds globally optimal allocation under bit budget |
| Dynamic pruning criterion | Routing weight ratio + token protection | Weight ratio only | Token protection prevents attention decay |
| Token importance metric | Magnitude × attention influence | Kurtosis / variance / mean | Combined metric outperforms all single-signal alternatives |
| Quantization backend | GPTQ (default) / OmniQuant | Fixed to one backend | Backend-agnostic design; OmniQuant adds ~1.3 points |

### ODP Ablation: Token Protection Strategies

| Strategy | Avg Pruned Params | WikiText2 PPL |
|---|---|---|
| Weight-only pruning (no protection) | 15.1% | 6.46 |
| Protect top 2% important tokens | 14.8% | 6.24 |
| Protect top 5% important tokens | ~13% | ~6.20 |
| Drop all experts for bottom 2% tokens | 15.8% | 6.35 |
| ODP (median threshold + 2% protection) | 14.88% | 6.22 |

### Token Importance Metric Comparison (PMQ 2.05-bit + ODP)

| Metric | Avg Pruned | WikiText2 | LM-Eval | GSM8K | HumanEval | NIAH |
|---|---|---|---|---|---|---|
| Token kurtosis | 15.62% | 7.16 | 57.22 | 14.05 | 6.54 | 93.16 |
| Token variance | 15.62% | 6.69 | 60.02 | 17.33 | 7.92 | 95.37 |
| Token mean | 15.62% | 6.82 | 59.27 | 17.76 | 6.02 | 95.65 |
| **ODP ($I_j$)** | **14.88%** | **6.22** | **63.25** | **18.04** | **10.02** | **99.26** |

MC's combined magnitude-attention metric outperforms all single-signal alternatives across every benchmark.

---

## Summary

MC is a clean, training-free compression system for MoE-LLMs that addresses both storage and runtime bottlenecks. Its core contributions are:

1. A multi-signal expert importance metric (frequency + routing weight) that outperforms single-signal alternatives.
2. An integer programming formulation that optimally allocates bit-widths across experts within each block.
3. A dynamic pruning rule that skips the second expert for low-confidence tokens, with a lightweight token importance score that protects salient tokens from attention decay.
4. Empirical demonstration that compressed MoE can outperform equal-size dense FP16 models.

For our tile optimization work, MC's ODP provides a principled, low-overhead mechanism to skip expert tiles entirely rather than running them inefficiently. The 2% token protection rule is a critical safety check that should be included in any expert-skipping implementation.
