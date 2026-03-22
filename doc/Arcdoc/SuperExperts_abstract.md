# Unveiling Super Experts in Mixture-of-Experts Large Language Models

## Paper Metadata

- **Title**: Unveiling Super Experts in Mixture-of-Experts Large Language Models
- **Authors**: Zunhai Su, Qingyuan Li, Hao Zhang, Weihao Ye, Qibo Xue, Yulei Qian, Yuchen Xie, Ngai Wong, Kehong Yuan
- **Venue**: ICLR 2026
- **arXiv ID**: arXiv:2507.23279v3
- **Year**: 2026

---

## Problem

MoE compression methods routinely prune or quantize experts based on heuristic importance metrics, but they lack a mechanistic understanding of why some experts are catastrophically important. This paper identifies a tiny subset of experts, called Super Experts (SEs), that are disproportionately critical for inference, and explains the causal chain by which they generate the massive activation outliers and attention sinks that define MoE model behavior. Without this understanding, compression pipelines risk accidentally destroying SEs and causing complete model failure.

---

## Approach

The paper traces the origin of massive activations (MAs) in MoE LLMs by analyzing the `down_proj` output of each expert across layers. A very small number of experts produce extreme activation outliers specifically at `down_proj`, which propagate through residual connections to create MAs in inter-layer hidden states. These MAs then cause certain tokens (typically the first token, acting as a "sink token") to attract disproportionate attention in later layers, forming attention sinks (ASs). The causal chain is: SEs -> MAs -> ASs.

Super Experts are formally defined using two thresholds applied to the per-expert maximum output magnitude `a_{l,e}`:

`SE iff a_{l,e} > P99.5(A)  AND  a_{l,e} > (1/10) * a_max  AND  l in L`

where A is the set of all expert output magnitudes, P99.5 is the 99.5th percentile, a_max is the global maximum, and L is the set of layers where MA formation occurs. A calibration-based profiling algorithm first detects layers L by identifying where MA patterns arise, then measures expert `down_proj` maxima, then applies the threshold rule. The resulting SEs are extremely rare (under 0.5% of all experts), model-specific, data-agnostic, and stable across post-training.

The paper further quantifies SE importance by measuring the Attention Sink Decay Rate after pruning:

`D_sink = 1 - (1/H) sum_h [ (sum_{i in S} p'_i^t) / (sum_{i in S} p_i^t) ]`

where S is the sink-token set and p, p' are attention scores before and after pruning. On Qwen3-30B-A3B, D_sink stays at or above 90% after SE pruning, confirming severe destruction of attention sinks. Heavy-tail analysis of the expert output magnitude distribution estimates power-law tail exponents of alpha = 1.50-1.68 across models, confirming that extreme expert activations are structured rather than noise. Weight-level analysis further maps SE importance to specific "super weights" in `down_proj`, and SAE analysis links these neurons to end-of-text token regulation.

---

## Key Results

- **SE rarity**: Qwen3-30B-A3B has 3 SEs out of 6,144 experts (0.05%); DeepSeek-R1 has 10 out of 15,677 (0.06%); Mixtral-8x7B has 1 out of 256 (0.39%).
- **Catastrophic pruning effect**: Pruning 3 SEs in Qwen3-30B-A3B raises WikiText-2 PPL from 8.70 to 59.86 and drops average accuracy from 70.22 to 55.00 (-21.7%), while pruning 1,000 non-SEs raises PPL only to 10.85. For DeepSeek-R1, SE pruning collapses Math-500 from 97.60 to 4.00 and AIME 2024 from 79.33 to 0.00.
- **Routing asymmetry**: Router scores for sink tokens are highly concentrated on SEs, while non-sink tokens show more uniform routing, explaining why SEs generate outliers despite being rarely activated by most tokens.

---

## Relevance

Super Experts define a hard constraint for mixed-precision MoE inference on Blackwell: the fewer than 0.5% of experts identified as SEs must be preserved at full precision (or at minimum handled with special care), while the remaining 99.5% are candidates for aggressive FP4 quantization, making SE detection a prerequisite for any safe mixed-precision expert-level quantization scheme.
