# MoEQuant: Technical Analysis

> **Document purpose:** Deep technical analysis of the MoEQuant paper for the TRT-LLM dual-tile MoE project. Covers all methods, formulas, results, and relevance to SM120/Blackwell deployment.

---

## 1. Paper Metadata

| Field | Value |
|---|---|
| **Title** | MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance |
| **Authors** | Xing Hu\*, Zhixuan Chen\* (equal contribution), Dawei Yang, Zukang Xu, Chen Xu, Zhihang Yuan, Sifan Zhou, Jiangyong Yu |
| **Affiliations** | Houmo AI; Southeast University |
| **Correspondence** | Dawei Yang `<dawei.yang@houmo.ai>` |
| **Venue** | ICML 2025 (41st International Conference on Machine Learning), Vancouver, Canada, PMLR 267 |
| **arXiv** | [arXiv:2505.03804v1](https://arxiv.org/abs/2505.03804) \[cs.LG\], 2 May 2025 |

---

## 2. Problem Statement

### The Memory Wall for MoE LLMs

MoE LLMs achieve high performance by routing each token to only a small subset of experts (sparse activation), but all experts must reside in memory simultaneously during inference. This creates a severe storage and memory bandwidth bottleneck:

- MoE layers account for roughly **80% of activated parameters** and up to **97% of total parameters** including dormant experts.
- Models like Mixtral-8x7B require ~90 GB of GPU memory in FP16, far exceeding single-GPU capacity.

Post-training quantization (PTQ) is the natural remedy. But applying standard PTQ methods (AWQ, GPTQ) designed for dense LLMs to MoE models causes **severe accuracy degradation**, especially on reasoning and code tasks.

### Why Standard PTQ Fails on MoE

The paper identifies two root causes, both arising from MoE's sparse routing and gated aggregation:

**Challenge 1: Inter-expert imbalance**

Standard calibration datasets (WikiText2, C4) produce highly skewed expert utilization. Popular experts receive many calibration samples; rarely-activated experts receive almost none. This means:
- Underloaded experts are calibrated with insufficient data.
- Their quantization parameters are poorly estimated.
- Accuracy degrades disproportionately on tasks that happen to route through those experts.

Figure 2 in the paper shows this directly: on the first MoE layer of Qwen-MoE-A2.7B-14B, WikiText2 and C4 calibration sets produce a long-tailed expert distribution, while EBSS-generated samples are nearly uniform.

**Challenge 2: Intra-expert imbalance**

Even among tokens routed to the same expert, their contributions to the expert's output are not equal. The MoE aggregation is a gated weighted sum:

```
y = sum_i g_i(x) * E_i(x)
```

A token with gating coefficient `g = 0.9` has nine times the influence of a token with `g = 0.1`. Standard PTQ treats all routed tokens identically when computing quantization loss and Hessian statistics, which misrepresents the actual error landscape.

---

## 3. Key Insight / Core Idea

Two complementary observations drive the paper:

**Insight 1 (for inter-expert imbalance):** The calibration data distribution problem is fundamentally a *search problem* over token sequences. Rather than trying to curate domain-specific datasets, one can *self-sample* from the model itself, using the model's own probability distribution as a guide. Self-sampled data has the lowest possible perplexity relative to the model and can be steered toward balanced expert utilization by incorporating an expert-balance metric into the search objective.

**Insight 2 (for intra-expert imbalance):** The gating coefficient `c_i` for a token-expert pair is not just a routing decision; it is a measure of *affinity* between that token and that expert. This affinity propagates multiplicatively through the expert's FFN layers. Therefore, it should directly weight each token's contribution to the quantization loss and Hessian statistics. Tokens with high affinity matter more; their quantization error has greater downstream impact.

Together these insights yield **MoEQuant**, a plug-and-play framework that wraps existing PTQ methods (AWQ, GPTQ) with two new components:
- **EBSS** (Expert-Balanced Self-Sampling): constructs a balanced calibration set.
- **AGQ** (Affinity-Guided Quantization): reweights calibration statistics by token-expert affinity.

---

## 4. Technical Approach: Step by Step

### 4.1 MoE Architecture Preliminaries

An MoE layer contains `m` shared experts and `n` routing experts. For each input token `x`, a gating network assigns probabilities to all routing experts. Only the top-`k` routing experts are activated; shared experts always run.

**Equation 1: MoE layer output**

```
y = sum_{i=1}^{m} g_i(x) * E_i^s(x)  +  sum_{j in K} g_j(x) * E_j^r(x)
```

where:
- `x`: input token representation
- `y`: MoE layer output
- `E_i^s(x)`: output of the i-th shared expert
- `E_j^r(x)`: output of the j-th routing expert
- `g_i(x)`: gating weight for expert `i` given token `x`
- `K = topk({ g_i(x) | i in {1,...,n} })`: the selected top-k routing experts

Each expert is a standard FFN. The gating weights `g_i(x)` are the *affinities* that MoEQuant exploits.

### 4.2 The Calibration Problem for MoE

Standard PTQ calibrates each linear layer by minimizing the layer-wise reconstruction error:

**Equation 7: Standard quantization loss**

```
L(W_hat) = || W X - W_hat X ||_F^2
```

where:
- `W in R^{o x c}`: original weight matrix
- `W_hat`: dequantized approximation
- `X in R^{b x c}`: calibration activations (b samples, c channels)
- `||.||_F`: Frobenius norm

The Hessian approximation used by GPTQ-style methods is:

**Equation 8: Standard Hessian**

```
H = X X^T
```

Both equations treat all calibration samples equally. For a dense LLM this is fine. For MoE, it is not: if `X` is dominated by tokens routed through popular experts, the Hessian poorly represents the sensitivity of rarely-used experts.

Quantization itself uses symmetric uniform per-channel quantization:

**Equation 5: Quantization**

```
Q(W) = clamp( round(W / s), q_min, q_max )
```

**Equation 6: Dequantization**

```
W_hat = Q(W) * s
```

where `s in R^o` is the per-channel scale and `q_min, q_max` are the quantization bounds.

### 4.3 Expert-Balanced Self-Sampling (EBSS)

#### Motivation

The paper first establishes that the calibration data quality problem has two dimensions:
1. **Perplexity** relative to the model: lower perplexity means the data is more representative of the model's learned distribution.
2. **Expert balance**: how evenly the data distributes tokens across experts.

Figure 4 shows perplexity of DeepSeek-MoE-16B on various datasets:
- WikiText2: 6.51
- C4: ~9
- HumanEval: 3.05
- GSM8K: 2.88
- MMLU: 7.10
- **EBSS (self-sampled):** 1.20

Self-sampled data achieves the lowest perplexity by construction, since the model generates it according to its own distribution.

#### Expert Balance Metric

**Equation 3: Global expert balance**

```
sigma = (1/L) * sum_{l=1}^{L} sigma_l
```

**Equation 4: Per-layer standard deviation**

```
sigma_l = sqrt( (1/(E-1)) * sum_{e=1}^{E} (u_l^e - u_hat_l)^2 )
```

where:
- `L`: number of MoE layers
- `E`: number of experts per layer
- `u_l^e`: usage frequency of expert `e` in layer `l` on the calibration set
- `u_hat_l`: mean usage frequency across experts in layer `l`
- `sigma_l`: standard deviation of expert usage in layer `l`

Lower `sigma` means better balance. This metric is cheap to compute incrementally as tokens are generated.

#### Joint Optimization Objective

The ideal calibration set minimizes both perplexity and expert imbalance simultaneously:

**Equation 9: Joint objective**

```
D* = argmin_D { PPL(M, D) * exp( sigma(M, D) / tau ) }
```

where:
- `D*`: optimal calibration dataset
- `PPL(M, D)`: perplexity of model `M` on dataset `D`
- `sigma(M, D)`: expert imbalance metric on `D`
- `tau`: temperature hyperparameter controlling the weight of expert balance

Taking the log, this becomes:

**Equation 10: Reformulated objective**

```
D* = argmin_D { -(1/N) * sum_{i=1}^{n} log P(D_i | D_{1:i-1}) + sigma(M, D) / tau }

subject to D in V x V x ... x V  (n times)
```

where `V = {v_1, ..., v_m}` is the vocabulary of size `m`. The search space is `m^n`, which is intractable by brute force.

#### Probability-Guided Path Pruning

EBSS solves this with a beam-search-like procedure. It maintains `w` candidate branches (partial sequences) and prunes at each step.

**Equation 11: Cumulative log-probability of a branch**

```
R_S = sum_{i=1}^{n} log P(S_i | S_{1:i-1})
```

where `S` is the current partial sequence of length `n`.

**Equation 12: Perplexity of extended sequence**

```
PPL(M, S || v) = exp( -(1/(n+1)) * (R_S + log P(v | S)) )
```

where `S || v` denotes appending token `v` to sequence `S`.

**Equation 13: Pruning score for candidate extension**

```
score(S || v) = -(1/(l+1)) * (R_S + log P(v | S)) + sigma(M, S) / tau

subject to v in V
```

where:
- `l`: current sequence length
- `R_S`: cached cumulative log-probability of branch `S`
- `sigma(M, S)`: expert balance computed on the *current* branch `S` (not `S||v`)
- `tau`: expert-balance temperature

Note the deliberate choice to use `sigma(M, S)` rather than `sigma(M, S||v)`. The paper gives three reasons:
1. Computing expert distribution for every vocabulary token at each step is prohibitively expensive.
2. Pruning should rely primarily on language probability to preserve semantic coherence.
3. Branch-level pruning still achieves balanced calibration while maintaining perplexity.

**Equation 14: Top-w branch selection**

```
S_hat = arg topk_{S||v}(w, score(S || v))

subject to v in V, S in {S^1, S^2, ..., S^w}
```

where `{S^1, ..., S^w}` is the current set of `w` branches and `S_hat = {S_hat^1, ..., S_hat^w}` is the new set after pruning.

This reduces search complexity from `O(m^n)` to `O(w^n)`.

#### EBSS Algorithm (Procedural Summary)

```
Initialize: w empty branches
For each position t = 1, 2, ..., n:
    For each branch S in current set:
        Compute log P(v | S) for all v in V
        Compute score(S || v) using Eq. 13
    Select top-w (S || v) pairs across all branches and tokens
    Update branches to selected extensions
    Cache R_S for each surviving branch
Return: resulting sequences as calibration set
```

Hyperparameters chosen by ablation: `tau = 1.2`, `w = 4`.

### 4.4 Affinity-Guided Quantization (AGQ)

#### Token-Expert Affinity

When a token `x_i` is routed to expert `E` with gating coefficient `c_i`, the expert's contribution to the final output is:

**Equation 15: Token output with affinity**

```
y_i = c_i * E(x_i)
```

For a standard FFN expert with up-projection, gate-projection, and down-projection:

**Equation 16: FFN with affinity**

```
y_i = c_i * { (x_i W_up) ⊙ f(x_i W_gate) } W_down
```

where `⊙` is elementwise product and `f` is the activation function (e.g., SiLU).

#### Affinity Propagation

The affinity `c_i` can be viewed as propagating through each sub-layer of the expert:

**Equation 17: Affinity propagation variants**

```
y_i = { (c_i x_i W_up) ⊙ f(x_i W_gate) } W_down        [into W_up]
y_i = { (x_i W_up) ⊙ f(x_i W_gate) } (c_i W_down)      [into W_down]
y_i ≈ { (x_i W_up) ⊙ f(c_i x_i W_gate) } W_down        [into W_gate]
```

All three formulations are mathematically equivalent or approximately equivalent. The key point: `c_i` modulates how much token `x_i` actually influences the expert's output. A token with `c_i = 0.9` contributes nine times more to the final MoE output than one with `c_i = 0.1`, even though both are "routed" to the same expert.

#### Affinity-Aware Quantization Loss

Standard quantization loss (Eq. 7) weights all tokens equally. AGQ replaces it with:

**Equation 18: Affinity-aware quantization loss**

```
L(W_hat) = sum_{i=0}^{n} c_i * || W x_i - W_hat x_i ||_F^2
```

where `c_i` is the gating coefficient for token `x_i` at this expert. Tokens with higher affinity contribute proportionally more to the loss, so the optimizer prioritizes minimizing error for the most influential tokens.

#### Gate-Aware Hessian

For GPTQ-style methods that use second-order statistics, AGQ replaces the standard Hessian (Eq. 8) with:

**Equation 19: Gate-aware Hessian**

```
H = (X * sqrt(c)) (X * sqrt(c))^T = (X * c) X^T
```

where:
- `X`: token activation matrix (rows are tokens)
- `c`: vector of affinity coefficients
- `sqrt(c)`: elementwise square root applied to affinity weights

The `sqrt(c)` weighting arises naturally from the quadratic form of the quantization loss: if the loss is `sum_i c_i ||e_i||^2`, the effective Hessian scales each outer product `x_i x_i^T` by `c_i`, which is equivalent to scaling `x_i` by `sqrt(c_i)` before forming the outer product.

#### Integration with Existing PTQ Methods

AGQ is plug-and-play:
- **With AWQ:** replace the uniform quantization loss with Eq. 18 when searching for optimal per-channel scales.
- **With GPTQ:** replace the standard Hessian `H = X X^T` with the gate-aware Hessian from Eq. 19.

The paper denotes:
- `MoEQuant+` = EBSS + AGQ applied on top of AWQ
- `MoEQuant++` = EBSS + AGQ applied on top of GPTQ

### 4.5 GPTQ-Specific Implementation Detail

For GPTQ-based MoEQuant++, the paper first removes weight outliers using an equivalent Hadamard transform (consistent with the QuaRot implementation). This avoids the need for online transformations during inference, keeping the method purely offline/static.

---

## 5. Granularity of Quantization

MoEQuant uses **per-channel (per-output-channel) weight quantization**. Each output channel of a weight matrix `W in R^{o x c}` gets its own scale `s in R^o`. This is the standard granularity for AWQ and GPTQ.

The paper does not explore per-group or per-token quantization. Activations are not quantized; this is **weight-only quantization**.

The unit that receives differentiated treatment is the **individual expert's weight matrices** within each MoE layer. Each expert is quantized separately, with its own calibration statistics derived from the tokens actually routed to it.

---

## 6. Static vs. Dynamic

MoEQuant is **purely offline/static PTQ**. There is no runtime component:

- Calibration data is generated once (EBSS) before quantization.
- Affinity weights are computed from the calibration forward pass and baked into the Hessian/loss statistics.
- The resulting quantized weights are fixed; no dynamic adjustment occurs at inference time.
- The Hadamard transform used for outlier removal is applied offline to the weights, not online during inference.

This is a deliberate design choice: the method targets deployment scenarios where inference overhead must be minimal.

---

## 7. Formulas and Algorithms: Complete Reference

All key formulas are reproduced below with full variable definitions for quick reference.

### MoE Layer

| Eq. | Formula | Description |
|---|---|---|
| (1) | `y = sum_{i=1}^{m} g_i(x) E_i^s(x) + sum_{j in K} g_j(x) E_j^r(x)` | MoE layer output |
| (2) | `PPL(D|M) = exp(-(1/N) sum_{i=1}^{N} log P_M(d_i | d_{1:i-1}))` | Perplexity |
| (3) | `sigma = (1/L) sum_{l=1}^{L} sigma_l` | Global expert balance metric |
| (4) | `sigma_l = sqrt((1/(E-1)) sum_{e=1}^{E} (u_l^e - u_hat_l)^2)` | Per-layer expert usage std dev |

### Quantization

| Eq. | Formula | Description |
|---|---|---|
| (5) | `Q(W) = clamp(round(W/s), q_min, q_max)` | Symmetric uniform per-channel quantization |
| (6) | `W_hat = Q(W) * s` | Dequantization |
| (7) | `L(W_hat) = ||WX - W_hat X||_F^2` | Standard layer-wise quantization loss |
| (8) | `H = X X^T` | Standard Hessian approximation |

### EBSS

| Eq. | Formula | Description |
|---|---|---|
| (9) | `D* = argmin_D { PPL(M,D) * exp(sigma(M,D)/tau) }` | Joint calibration objective |
| (10) | `D* = argmin_D { -(1/N) sum log P(D_i|D_{1:i-1}) + sigma(M,D)/tau }` | Reformulated (log-space) |
| (11) | `R_S = sum_{i=1}^{n} log P(S_i | S_{1:i-1})` | Cumulative log-probability of branch S |
| (12) | `PPL(M, S||v) = exp(-(1/(n+1))(R_S + log P(v|S)))` | Perplexity of extended sequence |
| (13) | `score(S||v) = -(1/(l+1))(R_S + log P(v|S)) + sigma(M,S)/tau` | Pruning score for candidate extension |
| (14) | `S_hat = arg topk_{S||v}(w, score(S||v))` | Top-w branch selection |

### AGQ

| Eq. | Formula | Description |
|---|---|---|
| (15) | `y_i = c_i * E(x_i)` | Token output scaled by affinity |
| (16) | `y_i = c_i * {(x_i W_up) ⊙ f(x_i W_gate)} W_down` | FFN with affinity |
| (17) | `y_i = {(c_i x_i W_up) ⊙ f(x_i W_gate)} W_down` | Affinity propagated to W_up |
| (18) | `L(W_hat) = sum_i c_i * ||W x_i - W_hat x_i||_F^2` | Affinity-weighted quantization loss |
| (19) | `H = (X * sqrt(c))(X * sqrt(c))^T` | Gate-aware Hessian |

**Variable glossary:**

| Symbol | Meaning |
|---|---|
| `x` | Input token representation |
| `y` | MoE layer output |
| `E_i^s`, `E_j^r` | Shared / routing expert `i` or `j` |
| `g_i(x)`, `c_i` | Gating weight / affinity for expert `i` and token `x` |
| `K` | Set of top-k selected routing experts |
| `L` | Number of MoE layers |
| `E` | Number of experts per layer |
| `u_l^e` | Usage frequency of expert `e` in layer `l` |
| `u_hat_l` | Mean expert usage in layer `l` |
| `sigma_l`, `sigma` | Per-layer / global expert imbalance std dev |
| `W` | Weight matrix `R^{o x c}` |
| `s` | Per-channel quantization scale `R^o` |
| `q_min`, `q_max` | Quantization bounds |
| `X` | Calibration activation matrix `R^{b x c}` |
| `H` | Hessian / second-order statistics |
| `D`, `D*` | Calibration dataset / optimal calibration dataset |
| `PPL` | Perplexity |
| `tau` | Expert-balance temperature hyperparameter |
| `w` | Number of branches in EBSS beam search |
| `R_S` | Cumulative log-probability of branch `S` |
| `V` | Vocabulary of size `m` |
| `n` | Sequence length |
| `l` | Current sequence length during EBSS |
| `W_up`, `W_gate`, `W_down` | FFN sub-matrices |
| `f` | Activation function (e.g., SiLU) |
| `⊙` | Elementwise product |

---

## 8. Experimental Results

### Setup

- **Quantization:** weight-only, symmetric uniform, per-channel, 4-bit (primary) and 3-bit (secondary)
- **Hardware:** NVIDIA A6000 GPUs
- **Calibration baseline:** 128 segments from WikiText2 (for non-MoEQuant methods)
- **Evaluation tools:** `lm-evaluation-harness` v0.4.4 for zero-shot tasks; official repositories for MMLU, GSM8K, HumanEval

### Models Tested

| Model | Architecture |
|---|---|
| Qwen-MoE-14B (Qwen1.5-MoE-A2.7B) | MoE, 14B total / 2.7B active |
| DeepSeek-MoE-16B | MoE, 16B total |
| Mixtral-8x7B | MoE, 8 experts, top-2 routing |
| Qwen-MoE-14B-Chat | Instruction-tuned |
| DeepSeek-MoE-16B-Chat | Instruction-tuned |

### Baselines

- **RTN** (Round-to-Nearest): naive quantization
- **AWQ**: activation-aware weight quantization
- **GPTQ**: second-order weight quantization with Hessian
- **FP**: full-precision reference

### 4-Bit Results: Base Models (Table 1)

**Qwen-MoE-14B**

| Method | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 7.22 | 9.30 | 59.60 | 32.32 | 62.55 | 79.82 | 57.96 | 30.40 | 35.77 | 51.20 |
| RTN | 10.83 | 12.49 | 48.10 | 14.63 | 16.07 | 72.11 | 51.42 | 25.80 | 30.08 | 36.89 |
| AWQ | 8.59 | 10.93 | 51.63 | 20.73 | 36.77 | 71.96 | 54.78 | 30.40 | 31.39 | 42.52 |
| **MoEQuant+** | 8.77 | 10.67 | 52.33 | 22.10 | 42.22 | 74.52 | 54.92 | 30.40 | 33.44 | **44.27** |
| GPTQ | 7.43 | 10.11 | 57.90 | 28.05 | 56.25 | 78.77 | 56.54 | 29.00 | 36.48 | 49.00 |
| **MoEQuant++** | 7.55 | 9.62 | 58.30 | 29.87 | 58.38 | 78.04 | 56.87 | 30.20 | 35.50 | **49.59** |

**DeepSeek-MoE-16B**

| Method | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 6.51 | 9.04 | 44.60 | 26.83 | 20.16 | 72.72 | 58.06 | 32.20 | 31.49 | 40.86 |
| RTN | 7.47 | 10.01 | 36.10 | 18.90 | 10.54 | 70.21 | 55.76 | 30.60 | 28.87 | 35.85 |
| AWQ | 6.80 | 9.50 | 40.57 | 25.00 | 17.06 | 71.65 | 56.42 | 32.20 | 31.76 | 39.23 |
| **MoEQuant+** | 6.94 | 9.32 | 41.20 | 25.00 | 18.90 | 71.98 | 56.79 | 32.12 | 31.82 | **39.68** |
| GPTQ | 6.66 | 9.39 | 40.60 | 22.56 | 19.18 | 72.17 | 57.03 | 30.60 | 30.95 | 39.01 |
| **MoEQuant++** | 6.78 | 9.22 | 42.20 | 25.00 | 19.18 | 73.49 | 57.20 | 31.40 | 31.66 | **40.01** |

**Mixtral-8x7B**

| Method | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 3.84 | 6.87 | 70.50 | 32.93 | 65.88 | 85.23 | 64.88 | 35.80 | 42.41 | 56.80 |
| RTN | 5.41 | 8.13 | 62.20 | 28.05 | 27.90 | 80.85 | 61.73 | 32.20 | 37.35 | 47.18 |
| AWQ | 5.01 | 7.98 | 62.75 | 25.00 | 38.67 | 79.97 | 62.11 | 33.60 | 38.43 | 48.64 |
| **MoEQuant+** | 5.15 | 7.84 | 64.66 | 25.45 | 50.66 | 81.03 | 62.73 | 34.00 | 39.77 | **51.19** |
| GPTQ | 4.03 | 7.67 | 68.50 | 27.60 | 57.92 | 84.22 | 64.08 | 30.60 | 41.07 | 53.42 |
| **MoEQuant++** | 4.12 | 7.34 | 69.60 | 32.15 | 61.79 | 84.98 | 64.05 | 33.60 | 42.95 | **55.58** |

### 4-Bit Results: Instruction-Tuned Models (Table 2 / Table 8)

**Qwen-MoE-14B-Chat**

| Method | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 8.07 | 9.74 | 59.00 | 21.34 | 30.71 | 81.31 | 59.33 | 31.00 | 34.91 | 45.37 |
| RTN | 12.81 | 14.03 | 43.00 | 7.32 | 9.70 | 71.13 | 51.41 | 24.40 | 28.81 | 33.68 |
| AWQ | 9.97 | 11.90 | 52.06 | 12.20 | 17.74 | 74.74 | 55.37 | 30.40 | 31.46 | 39.14 |
| **MoEQuant+** | 10.12 | 11.55 | 55.34 | 13.60 | 20.87 | 76.22 | 56.64 | 30.60 | 32.50 | **40.82** |
| GPTQ | 8.38 | 10.78 | 57.30 | 15.24 | 26.08 | 78.92 | 58.72 | 31.40 | 34.17 | 43.19 |
| **MoEQuant++** | 8.65 | 10.21 | 58.00 | 21.95 | 29.11 | 79.11 | 58.53 | 33.20 | 34.77 | **44.95** |

Notable: GPTQ drops HumanEval from 21.34 to 15.24 (-6.1 pts). MoEQuant++ recovers to 21.95, slightly *above* full precision.

**DeepSeek-MoE-16B-Chat**

| Method | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 7.35 | 9.96 | 48.90 | 24.39 | 54.28 | 79.81 | 60.69 | 33.40 | 34.27 | 47.96 |
| RTN | 8.63 | 11.06 | 41.40 | 10.41 | 28.88 | 75.84 | 57.59 | 31.40 | 29.04 | 39.22 |
| AWQ | 7.72 | 10.49 | 46.33 | 18.90 | 39.88 | 78.20 | 58.97 | 33.80 | 32.86 | 44.13 |
| **MoEQuant+** | 7.85 | 10.23 | 46.40 | 18.90 | 45.41 | 78.20 | 59.03 | 33.60 | 33.14 | **44.95** |
| GPTQ | 7.55 | 10.24 | 46.60 | 13.41 | 47.08 | 78.87 | 59.64 | 33.20 | 32.76 | 44.50 |
| **MoEQuant++** | 7.70 | 10.08 | 47.60 | 21.95 | 48.97 | 79.20 | 59.30 | 33.80 | 32.60 | **46.20** |

### 3-Bit Results (Table 3 / Table 9)

**DeepSeek-MoE-16B (3-bit)**

| Method | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 6.51 | 9.04 | 44.60 | 26.83 | 20.16 | 72.72 | 58.06 | 32.20 | 31.49 | 40.86 |
| RTN | 26352 | 32357 | 24.80 | 0.00 | 1.59 | 51.62 | 26.18 | 15.60 | 21.44 | 20.17 |
| AWQ | 4622 | 5505 | 27.80 | 1.90 | 2.88 | 53.20 | 27.97 | 17.80 | 23.86 | 22.20 |
| **MoEQuant+** | 5100 | 4924 | 33.20 | 8.72 | 10.44 | 59.24 | 29.22 | 20.60 | 25.14 | **26.65** |
| GPTQ | 7.17 | 11.66 | 37.30 | 17.68 | 11.60 | 72.31 | 53.68 | 27.80 | 29.72 | 35.85 |
| **MoEQuant++** | 7.55 | 10.88 | 40.00 | 20.12 | 12.81 | 69.72 | 54.09 | 29.00 | 29.61 | **36.47** |

**Mixtral-8x7B (3-bit)**

| Method | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| FP | 3.84 | 6.87 | 70.50 | 32.93 | 65.88 | 85.23 | 64.88 | 35.80 | 42.41 | 56.80 |
| RTN | 44944 | 51241 | 25.30 | 0.00 | 0.00 | 41.52 | 25.61 | 18.40 | 19.66 | 18.64 |
| AWQ | 7.38 | 13.13 | 45.80 | 10.37 | 10.39 | 75.23 | 53.04 | 28.00 | 29.55 | 36.05 |
| **MoEQuant+** | 8.77 | 11.44 | 49.40 | 14.44 | 17.29 | 77.22 | 54.29 | 30.10 | 32.34 | **39.30** |
| GPTQ | 4.64 | 9.12 | 57.80 | 22.56 | 22.59 | 79.82 | 61.30 | 30.40 | 40.80 | 45.04 |
| **MoEQuant++** | 4.90 | 8.24 | 64.10 | 28.05 | 43.21 | 82.81 | 60.07 | 31.20 | 38.82 | **49.75** |

### Ablation Study (Table 5)

Ablation on GPTQ as the base method, testing EBSS and AGQ independently.

**DeepSeek-MoE-16B**

| EBSS | AGQ | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x | x | 6.66 | 9.39 | 40.60 | 22.56 | 19.18 | 72.17 | 57.03 | 30.60 | 30.95 | 39.01 |
| x | v | 6.66 | 9.38 | 41.60 | 23.17 | 17.89 | 74.52 | 57.30 | 31.20 | 30.88 | 39.50 |
| v | x | 6.77 | 9.22 | 44.00 | 23.78 | 18.19 | 73.24 | 57.21 | 31.80 | 30.92 | 39.87 |
| v | v | 6.78 | 9.25 | 42.20 | 25.00 | 19.18 | 73.49 | 57.20 | 31.40 | 31.66 | **40.01** |

**Mixtral-8x7B**

| EBSS | AGQ | PPL (Wiki) | PPL (C4) | MMLU | HumanEval | GSM8K | BoolQ | HellaSwag | OBQA | MathQA | Avg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x | x | 4.03 | 7.67 | 68.50 | 27.60 | 57.92 | 84.22 | 64.08 | 30.60 | 41.07 | 53.42 |
| x | v | 4.04 | 7.64 | 68.30 | 29.54 | 60.12 | 83.36 | 64.04 | 32.80 | 41.54 | 54.24 |
| v | x | 4.10 | 7.38 | 69.10 | 31.19 | 60.50 | 84.83 | 64.21 | 34.20 | 42.01 | 55.15 |
| v | v | 4.12 | 7.38 | 69.60 | 32.15 | 61.79 | 84.98 | 64.05 | 33.60 | 42.95 | **55.58** |

Key ablation findings:
- On DeepSeek-MoE-16B: EBSS alone +0.86 avg, AGQ alone +0.49 avg, combined +1.00 avg.
- On Mixtral-8x7B: EBSS alone +1.73 avg, AGQ alone +0.82 avg, combined +2.16 avg.
- For Mixtral, AGQ alone does not consistently beat vanilla GPTQ on all tasks, but the combination always wins.

### EBSS Hyperparameter Sensitivity (Tables 6 and 7)

**Temperature tau (DeepSeek-MoE-16B, MoEQuant++)**

| tau | Avg Score |
|---|---|
| 1.0 | 39.82 |
| 1.1 | 39.89 |
| **1.2** | **40.01** |
| 1.3 | 39.98 |
| 1.4 | 39.69 |
| 1.5 | 39.71 |

**Branch count w (DeepSeek-MoE-16B, MoEQuant++)**

| w | Avg Score |
|---|---|
| 2 | 39.77 |
| 3 | 39.80 |
| **4** | **40.01** |
| 5 | 39.98 |
| 6 | 40.01 |
| 10 | 40.00 |
| 20 | 40.10 |
| 50 | 40.11 |

`w=4` is chosen as the default: it achieves near-optimal performance with much lower generation cost than larger beam widths.

### Inference Efficiency (Table 4)

| Model | FP Speed (tok/s) | Quant Speed (tok/s) | Speedup | FP Memory (GB) | Quant Memory (GB) | Memory Saving |
|---|---|---|---|---|---|---|
| Qwen-MoE-14B | 8.35 | 10.60 | 1.27x | 27.88 | 8.51 | 3.28x |
| DeepSeek-MoE-16B | 20.81 | 24.45 | 1.17x | 32.23 | 9.87 | 3.27x |
| Mixtral-8x7B | 10.24 | 21.25 | 2.08x | 89.64 | 23.97 | 3.74x |

Average: >1.2x speedup, >3.2x memory reduction. Mixtral-8x7B drops from 89.6 GB to 24 GB, enabling deployment on a single RTX 4090.

---

## 9. Implementation Details

- **Quantization scheme:** symmetric uniform, per-output-channel, weight-only
- **Bit-widths tested:** 4-bit (primary), 3-bit (secondary)
- **Hardware:** NVIDIA A6000 GPUs
- **AWQ integration:** official AWQ repository, adapted to support MoE models
- **GPTQ integration:** Hadamard transform applied offline to remove weight outliers (QuaRot-style); no online transformation at inference
- **Evaluation:** `lm-evaluation-harness` v0.4.4 for zero-shot tasks; official repositories for MMLU, GSM8K, HumanEval
- **Calibration baseline:** 128 segments from WikiText2 for non-MoEQuant methods
- **EBSS defaults:** `tau = 1.2`, `w = 4`
- **No fine-tuning:** purely PTQ, no gradient updates to model weights

The paper does not specify the number of EBSS-generated calibration sequences or total token count used for MoEQuant calibration, which is a gap in reproducibility.

---

## 10. Limitations and Gaps

**Stated or implied limitations:**

1. **No explicit limitations section.** The paper does not include a dedicated limitations discussion, which is unusual for an ICML paper.

2. **Calibration budget not specified.** The paper does not state how many tokens or sequences EBSS generates, making direct comparison of calibration cost to baselines difficult.

3. **Weight-only quantization only.** Activation quantization (W4A8, W8A8) is not addressed. For hardware like Blackwell that benefits from INT8/FP8 activation quantization, MoEQuant's approach would need extension.

4. **Per-channel granularity only.** Per-group quantization (e.g., group size 128), which is common in practice for better accuracy at low bits, is not explored.

5. **Static calibration.** The expert routing distribution used for EBSS is derived from the model's own generation, not from actual deployment traffic. If real-world routing differs significantly (e.g., domain shift), the calibration balance may not hold.

6. **EBSS computational cost.** Beam search over vocabulary at each token position is expensive. With `w=4` and vocabulary size ~32K, each step evaluates 128K candidate extensions. The paper does not report EBSS generation time.

7. **No shared-expert treatment.** The paper focuses on routing experts. Shared experts (always active) receive all calibration tokens and may not need special treatment, but this is not discussed.

8. **Mixtral AGQ anomaly.** For Mixtral-8x7B, AGQ alone does not consistently improve over vanilla GPTQ. The paper notes this but does not fully explain why Mixtral's routing structure makes AGQ less effective in isolation.

9. **No comparison to MoE-specific baselines.** The paper compares only to general PTQ methods (AWQ, GPTQ, RTN). There is no comparison to other MoE-aware quantization works such as MxMoE or DynaMo.

10. **No latency breakdown.** The speedup numbers are end-to-end decoder throughput. There is no breakdown of how much comes from reduced memory bandwidth vs. compute.

---

## 11. Relevance to Our Work (TRT-LLM Dual-Tile MoE, SM120/Blackwell)

### Context

Our project implements MoE inference on SM120 (Blackwell) using a dual-tile architecture with M32 and M64 expert tiles. We have a profiler that captures real expert routing distributions from actual inference traffic.

### Direct Relevance: EBSS and Calibration

**The most immediately applicable idea is EBSS.** Our profiler already captures real routing distributions, which is actually *better* than EBSS's self-sampled approximation for our use case:

- EBSS approximates balanced routing by generating synthetic data. We have *real* routing histograms from production traffic.
- We could use our profiler data to directly construct a calibration set that matches the actual expert utilization distribution, rather than the uniform distribution EBSS targets.
- Specifically: if our profiler shows expert `e` in layer `l` receives `p_e` fraction of tokens in production, we can sample calibration data to match that distribution rather than forcing uniformity.

This is a stronger version of EBSS: instead of optimizing for balance (uniform), optimize for *fidelity to deployment distribution*.

**Practical implementation:** Use our routing histograms as importance weights when selecting calibration sequences. Sequences that activate underrepresented experts (in production) should be oversampled.

### AGQ and Blackwell Tile Architecture

AGQ's affinity-weighted Hessian (Eq. 19) is relevant to our quantization calibration pipeline:

- Our dual-tile design routes experts to M32 or M64 tiles based on activation frequency and size. High-affinity tokens (large `c_i`) are exactly the tokens that matter most for tile utilization and output quality.
- When calibrating quantization for experts assigned to M32 tiles (smaller, less frequently used), AGQ's weighting ensures that the rare but high-affinity tokens for those experts are not drowned out by low-affinity noise.
- For M64 tiles (larger, frequently used experts), the standard Hessian already captures most of the signal, but AGQ still provides marginal improvement.

### Static vs. Dynamic Quantization

MoEQuant is purely static PTQ. Our TRT-LLM deployment also uses static quantization (INT8/FP8 weights baked at compile time). The methods are compatible.

However, MoEQuant does not address **FP8 activation quantization**, which Blackwell supports natively. Extending AGQ to weight-activation quantization (W4A8 or W8A8) would require affinity-weighted activation statistics as well, not just weight Hessians.

### Expert Routing Distribution Insight

MoEQuant's Figure 2 (expert distribution on WikiText2 vs. EBSS) is directly relevant to our calibration pipeline design. The key takeaway: **calibration data that does not match deployment routing distribution will systematically under-calibrate certain experts.** Our profiler gives us the deployment distribution; we should use it.

### Potential Integration Points

| Component | MoEQuant Idea | Our Adaptation |
|---|---|---|
| Calibration data | EBSS: self-sample for balance | Use profiler routing histograms to sample for deployment fidelity |
| Quantization loss | AGQ: affinity-weighted loss (Eq. 18) | Apply to per-expert calibration in our PTQ pipeline |
| Hessian statistics | Gate-aware Hessian (Eq. 19) | Replace `H = XX^T` with `H = (X*sqrt(c))(X*sqrt(c))^T` in GPTQ-style calibration |
| Tile assignment | Not addressed | Use routing frequency from profiler to decide M32 vs. M64 assignment |
| FP8 activation | Not addressed | Extend AGQ to activation scale calibration |

### Limitations of Direct Adoption

- MoEQuant targets weight-only quantization. Our Blackwell deployment likely uses FP8 (W8A8), requiring extension.
- EBSS generates synthetic data; our profiler data is real but may not cover all expert combinations.
- The Hadamard transform for outlier removal (QuaRot-style) may interact with our tile layout in non-trivial ways.

---

## 12. Comparison Table

| Dimension | MoEQuant (this paper) | MxMoE | DynaMo | Our Work (TRT-LLM Dual-Tile) |
|---|---|---|---|---|
| **Core problem** | Calibration imbalance + affinity weighting for MoE PTQ | Mixed-precision quantization for MoE | Dynamic quantization for MoE routing | MoE inference efficiency on SM120/Blackwell |
| **Quantization type** | Weight-only (W4, W3) | Mixed-precision weights | Dynamic weight/activation | FP8/INT8 (W8A8 target) |
| **Granularity** | Per-channel | Per-group or per-expert | Dynamic per-token | Per-tile (M32/M64) |
| **Calibration** | EBSS: self-sampled, expert-balanced | Standard domain datasets | Online routing-aware | Profiler-captured real routing distributions |
| **Routing awareness** | Yes (EBSS + AGQ) | Partial | Yes | Yes (dual-tile assignment) |
| **Static vs. dynamic** | Static PTQ | Static PTQ | Dynamic | Static compile-time |
| **Hardware target** | A6000 (evaluation) | General GPU | General GPU | SM120 Blackwell |
| **Key innovation** | Expert-balanced self-sampling + affinity-guided Hessian | Precision assignment per expert | Online quantization adaptation | Dual-tile MoE kernel, profiler-guided routing |
| **Reported gain** | >10 pt HumanEval, 3.2x memory, 1.2x speed | Varies | Varies | TBD |
| **Plug-and-play** | Yes (wraps AWQ/GPTQ) | Depends | No | No (kernel-level) |
| **Activation quantization** | No | Partial | Yes | Yes (FP8) |

---

## Summary

MoEQuant makes two clean, well-motivated contributions to MoE PTQ:

1. **EBSS** solves the calibration distribution problem by framing it as a beam search over self-generated sequences, jointly optimizing for model-aligned perplexity and expert balance. The math is straightforward (Eqs. 9-14) and the implementation is practical with `w=4` branches.

2. **AGQ** solves the intra-expert weighting problem by incorporating gating coefficients directly into the quantization loss and Hessian (Eqs. 18-19). This is a minimal change to existing PTQ pipelines with consistent accuracy gains.

The results are strong: near-FP accuracy at 4-bit on three MoE architectures, with 3.2x memory reduction and 1.2x throughput improvement. The instruction-tuned model results are particularly compelling, showing that MoEQuant preserves fine-tuned capabilities that vanilla GPTQ destroys.

For our TRT-LLM dual-tile project, the most actionable takeaway is: **use our profiler's real routing distributions to construct calibration sets (a stronger version of EBSS), and apply affinity-weighted Hessian statistics (AGQ, Eq. 19) when calibrating per-expert quantization parameters.**
