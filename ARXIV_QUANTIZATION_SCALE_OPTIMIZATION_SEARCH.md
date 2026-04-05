# ArXiv Search Results: Quantization Scale Optimization (2020-2025)

## Executive Summary

**Total Papers Found: 9**
- Papers with post-quantization scale optimization: 8
- Papers with joint/alternating optimization: 2
- Papers using Hessian/second-order information: 1

---

## KEY FINDINGS

### 1. BRECQ (2021) - Most Relevant for Hessian-Based Scale Optimization
**arXiv ID:** 2102.05426v2  
**Published:** 2021-02-10  
**URL:** https://arxiv.org/abs/2102.05426v2

**Optimization Characteristics:**
- ✅ Scales optimized AFTER weight quantization: YES
- ✅ Joint/Alternating optimization: YES (block-wise reconstruction)
- ✅ Uses Hessian/Second-order information: YES (comprehensive theoretical study of second-order error)

**Key Details:**
- First PTQ method to achieve INT2 quantization
- Uses block-wise reconstruction with second-order error analysis
- Incorporates mixed precision via inter/intra-layer sensitivity approximation
- 240x faster than QAT while achieving comparable accuracy
- Theoretical foundation: "comprehensive theoretical study of the second-order error"

---

### 2. AdaRound (2020) - Foundational Post-Training Scale Optimization
**arXiv ID:** 2004.10568v2  
**Published:** 2020-04-22  
**URL:** https://arxiv.org/abs/2004.10568v2

**Optimization Characteristics:**
- ✅ Scales optimized AFTER weight quantization: YES
- ❌ Joint/Alternating optimization: NO
- ❌ Uses Hessian/Second-order information: NO

**Key Details:**
- Proposes adaptive weight rounding mechanism for PTQ
- Formulates rounding as quadratic unconstrained binary optimization (QUBO)
- Uses Taylor series expansion of task loss
- Layer-wise local loss optimization with soft relaxation
- Outperforms round-to-nearest by significant margin
- ResNet18/50 to 4-bit with <1% accuracy loss

---

### 3. Solving Oscillation Problem (2023) - Joint Optimization Approach
**arXiv ID:** 2303.11906v2  
**Published:** 2023-03-21  
**URL:** https://arxiv.org/abs/2303.11906v2

**Optimization Characteristics:**
- ✅ Scales optimized AFTER weight quantization: YES
- ✅ Joint/Alternating optimization: YES (jointly optimize top-k modules)
- ❌ Uses Hessian/Second-order information: NO

**Key Details:**
- Addresses oscillation problem in PTQ through theoretical analysis
- Defines module capacity (ModCap) to measure oscillation degree
- Jointly optimizes and quantizes top-k modules with highest differentials
- Data-dependent and data-free scenarios supported
- Improvements: +1.9% on 2/4-bit ResNet-50, +6.61% on MobileNetV2*0.5

---

### 4. Reclaiming Residual Knowledge (2024) - Residual-Based Approach
**arXiv ID:** 2408.00923v1  
**Published:** 2024-08-01  
**URL:** https://arxiv.org/abs/2408.00923v1

**Optimization Characteristics:**
- ✅ Scales optimized AFTER weight quantization: YES
- ❌ Joint/Alternating optimization: NO
- ❌ Uses Hessian/Second-order information: NO

**Key Details:**
- Novel paradigm: frames quantization as architecture search problem
- Reclaims quantization residual knowledge (lost information)
- Uses low-rank adapters to approximate residual weights
- Search space orders of magnitude smaller than weight spaces
- <250 iterations vs BRECQ's 2×10^4 iterations
- Comparable to QAT with 4-bit and 3-bit quantization

---

### 5. QUBO Formulation (2025) - Exact Optimization
**arXiv ID:** 2510.16075v1  
**Published:** 2025-10-17  
**URL:** https://arxiv.org/abs/2510.16075v1

**Optimization Characteristics:**
- ✅ Scales optimized AFTER weight quantization: YES
- ❌ Joint/Alternating optimization: NO
- ❌ Uses Hessian/Second-order information: NO

**Key Details:**
- ADAROUND-based QUBO formulation for PTQ
- Uses Frobenius distance as objective
- Decomposes global problem into n independent subproblems
- Solves using heuristics (simulated annealing)
- Evaluated on int8 to int1 precision

---

### 6. QuantKAN (2025) - Framework for Spline Networks
**arXiv ID:** 2511.18689v2  
**Published:** 2025-11-24  
**URL:** https://arxiv.org/abs/2511.18689v2

**Optimization Characteristics:**
- ✅ Scales optimized AFTER weight quantization: YES
- ❌ Joint/Alternating optimization: NO
- ❌ Uses Hessian/Second-order information: NO

**Key Details:**
- Unified quantization framework for Kolmogorov Arnold Networks
- Extends modern algorithms: LSQ, LSQ+, PACT, DoReFa, QIL, GPTQ, BRECQ, AdaRound, AWQ, HAWQ-V2
- Branch-specific quantizers for base, spline, and activation components
- First systematic benchmarks for low-bit spline networks

---

### 7. Bits for Privacy (2025) - Privacy-Aware PTQ
**arXiv ID:** 2512.15335v1  
**Published:** 2025-12-17  
**URL:** https://arxiv.org/abs/2512.15335v1

**Optimization Characteristics:**
- ✅ Scales optimized AFTER weight quantization: YES
- ❌ Joint/Alternating optimization: NO
- ❌ Uses Hessian/Second-order information: NO

**Key Details:**
- Analyzes privacy-utility relationship in PTQ
- Evaluates AdaRound, BRECQ, OBC across 4-bit, 2-bit, 1.58-bit
- Lower-precision models reduce membership inference vulnerability
- Up to 10x reduction in privacy leakage vs full-precision

---

### 8. Diversifying Sample Generation (2021) - Data-Free Quantization
**arXiv ID:** 2103.01049v3  
**Published:** 2021-03-01  
**URL:** https://arxiv.org/abs/2103.01049v3

**Optimization Characteristics:**
- ✅ Scales optimized AFTER weight quantization: YES
- ❌ Joint/Alternating optimization: NO
- ❌ Uses Hessian/Second-order information: NO

**Key Details:**
- Addresses homogenization in synthetic data for calibration
- Slacks BN statistics alignment at distribution level
- Layerwise enhancement for different data samples
- Compatible with AdaRound
- Up to 22% improvement on W4A4

---

### 9. MixQuant (2023) - Mixed Precision Search
**arXiv ID:** 2309.17341v1  
**Published:** 2023-09-29  
**URL:** https://arxiv.org/abs/2309.17341v1

**Optimization Characteristics:**
- ❌ Scales optimized AFTER weight quantization: NO (bit-width search)
- ❌ Joint/Alternating optimization: NO
- ❌ Uses Hessian/Second-order information: NO

**Key Details:**
- Searches optimal custom bit-width for each layer weight
- Based on roundoff error analysis
- Can be combined with any quantization method
- Improves BRECQ performance when combined

---

## METHODOLOGY CLASSIFICATION

### Post-Quantization Scale Optimization (6 papers)
1. AdaRound (2020) - Adaptive rounding via QUBO
2. Diversifying Sample Generation (2021) - Data-free calibration
3. BRECQ (2021) - Block reconstruction with Hessian
4. Solving Oscillation (2023) - Module capacity analysis
5. Reclaiming Residual Knowledge (2024) - Low-rank residual adaptation
6. QUBO Formulation (2025) - Exact QUBO decomposition

### Joint/Alternating Optimization (2 papers)
1. BRECQ (2021) - Block-wise reconstruction
2. Solving Oscillation (2023) - Top-k module joint optimization

### Hessian/Second-Order Methods (1 paper)
1. **BRECQ (2021)** - Only paper explicitly using second-order error analysis

---

## OPTIMIZATION TECHNIQUES IDENTIFIED

### 1. **Rounding-Based Methods**
- AdaRound: Quadratic unconstrained binary optimization (QUBO)
- QUBO Formulation: Exact QUBO with simulated annealing

### 2. **Block/Layer-Wise Reconstruction**
- BRECQ: Block-by-block reconstruction with second-order error
- Solving Oscillation: Top-k module joint optimization

### 3. **Residual/Correction Methods**
- Reclaiming Residual Knowledge: Low-rank adapter approximation
- Diversifying Sample Generation: Enhanced synthetic data

### 4. **Sensitivity-Based Methods**
- MixQuant: Roundoff error sensitivity per layer
- BRECQ: Inter/intra-layer sensitivity approximation

### 5. **Calibration Methods**
- QuantKAN: Branch-specific quantizers
- Bits for Privacy: Privacy-aware calibration

---

## RESEARCH TRENDS

### Timeline
- **2020**: AdaRound introduces adaptive rounding (foundational)
- **2021**: BRECQ adds Hessian-based block reconstruction; DSG for data-free
- **2023**: Oscillation problem identified; MixQuant for mixed precision
- **2024**: Residual knowledge reclamation approach
- **2025**: QUBO exact formulation; QuantKAN framework; Privacy analysis

### Key Evolution
1. From simple rounding → adaptive rounding (AdaRound)
2. From layer-wise → block-wise with second-order (BRECQ)
3. From weight optimization → residual optimization (CoRa)
4. From single-precision → mixed-precision search (MixQuant)
5. From accuracy-only → privacy-aware (Bits for Privacy)

---

## RECOMMENDATIONS FOR FURTHER RESEARCH

### High Priority (Hessian-Based Methods)
- **BRECQ (2021)** is the only paper using second-order information
- Opportunity: Extend Hessian-based methods to modern architectures (Transformers, LLMs)
- Opportunity: Combine BRECQ with residual knowledge reclamation

### Medium Priority (Joint Optimization)
- Only 2 papers use joint/alternating optimization
- Opportunity: Explore coordinate descent for scale optimization
- Opportunity: Combine with Hessian information

### Emerging Areas
- Privacy-aware quantization (Bits for Privacy, 2025)
- Spline-based networks (QuantKAN, 2025)
- Exact optimization via QUBO (2025)

---

## PAPER QUALITY ASSESSMENT

### Tier 1 (Most Relevant for Scale Optimization)
1. **BRECQ (2021)** - Only Hessian-based method; block reconstruction
2. **AdaRound (2020)** - Foundational; QUBO formulation
3. **Solving Oscillation (2023)** - Joint optimization theory

### Tier 2 (Complementary Methods)
4. **Reclaiming Residual Knowledge (2024)** - Novel residual approach
5. **MixQuant (2023)** - Mixed precision sensitivity
6. **Diversifying Sample Generation (2021)** - Data-free calibration

### Tier 3 (Specialized Applications)
7. **QuantKAN (2025)** - Spline networks
8. **QUBO Formulation (2025)** - Exact optimization
9. **Bits for Privacy (2025)** - Privacy analysis

