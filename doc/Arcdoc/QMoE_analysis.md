# QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models

**Technical Analysis Document**

---

## 1. Paper Metadata

| Field | Value |
|---|---|
| **Title** | QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models |
| **Authors** | Elias Frantar, Dan Alistarh |
| **Affiliations** | Institute of Science and Technology Austria (ISTA); Neural Magic Inc. |
| **Venue** | MLSys 2024 |
| **arXiv ID** | arXiv:2310.16795v1 [cs.LG] |
| **Date** | 25 October 2023 |
| **Code** | https://github.com/IST-DASLab/qmoe |
| **Contact** | elias.frantar@ist.ac.at |

---

## 2. Problem Statement

### The Memory Wall for Trillion-Parameter MoEs

Mixture-of-Experts (MoE) language models achieve state-of-the-art quality by routing each token to a small subset of "expert" sub-networks, keeping per-token FLOPs manageable. The catch is that *all* expert weights must reside in memory simultaneously, even though only a fraction are active per forward pass. This creates a brutal memory-to-compute mismatch.

The flagship example is **SwitchTransformer-c2048**, a 1.6 trillion parameter model. In bfloat16, it occupies **3.2 TB** of storage. Deploying it naively requires a fleet of over 65 A6000 GPUs (48 GB each) or over 130 RTX 3090s (24 GB each). That's not commodity hardware; that's a data center.

### Why Existing Compression Falls Short

Standard post-training quantization (PTQ) methods like GPTQ can compress dense LLMs to 3-4 bits per parameter with acceptable accuracy loss. For a 1.6T model, even 4-bit compression only gets you to ~800 GB, still requiring ~17 A6000s. The practical deployment target for a single node of commodity GPUs demands roughly **10-20x compression**, which translates to **less than 1 bit per parameter on average**.

Three specific challenges block naive approaches from reaching this regime:

1. **Accuracy collapse at extreme bitrates.** Standard PTQ methods degrade catastrophically below 2-3 bits. Round-to-nearest (RTN) ternary quantization of SwitchTransformer-c2048 raises C4 validation loss from 1.18 to 2.15, an 82% relative increase.

2. **Scalability of data-dependent quantization.** Methods like GPTQ require storing calibration activations. For a 1.6T model, both the weights and the activations vastly exceed available GPU and CPU RAM, making naive application impossible.

3. **GPU-efficient decoding.** Even if you achieve sub-1-bit storage, you need to decompress weights at inference time fast enough that the decompression overhead doesn't negate the memory savings. Standard entropy coding (Huffman, arithmetic) requires sequential decoding with variable-length symbols, which maps poorly to GPU SIMD execution.

---

## 3. Key Insight / Core Idea

QMoE's central observation is that **MoE expert weights are unusually compressible** for two compounding reasons.

**Reason 1: Structural concentration.** In a model like SwitchTransformer-c2048, the vast majority of parameters live in the expert feed-forward layers. The non-expert layers (attention, layer norms, embeddings) are comparatively tiny. This means compressing only the experts captures almost all the storage savings while leaving the accuracy-critical non-expert layers untouched.

**Reason 2: Natural sparsity after ternary quantization.** When you quantize expert weights to a ternary grid `{w_min, 0, w_max}` using a row-wise min/max scale, the grid is wide relative to the approximately Gaussian weight distribution. The result is that a large fraction of weights round to zero. In c2048, **88.6% of ternary expert weights are zero**. This is not sparsity you impose; it emerges naturally from the quantization grid.

This high zero rate means the ternary weight distribution has **low entropy**. The theoretical entropy of a distribution with `P(0) = 0.886` and `P(1) = P(2) = 0.057` is well below 1 bit per symbol. In principle, you can encode these weights in less than 1 bit each.

The challenge is doing this in a way that a GPU can decode efficiently. QMoE's solution is a **fixed-codeword dictionary code** co-designed with a fused GPU decompression+matrix-vector kernel. Instead of variable-length codewords (which require sequential decoding), it uses fixed 16-bit codewords that each decode to a variable number of ternary weights. The dictionary is precomputed to cover the 2^16 highest-probability ternary sequences, and the GPU kernel looks up each codeword in the dictionary to recover the weights.

---

## 4. Technical Approach: Step by Step

### 4.1 Overview of the Pipeline

QMoE is a post-training compression framework with three stages:

1. **Scalable data-dependent quantization** of MoE experts to ternary (or 2-bit) precision using a GPTQ-based method adapted for trillion-parameter scale.
2. **Dictionary-based entropy compression** of the ternary weights to achieve sub-1-bit storage.
3. **Fused GPU decompression+matvec kernel** for inference directly from compressed weights.

Non-expert layers remain in bfloat16 throughout.

### 4.2 Quantization: Scalable GPTQ for MoEs

The quantization objective for each expert layer `l` is:

$$\arg\min_{Q_l} \| Q_l X_l - W_l X_l \|_F^2 \tag{1}$$

where `W_l` are the original weights, `Q_l` are the quantized weights, and `X_l` are calibration inputs observed at layer `l`. This is the standard GPTQ layer-wise second-order objective.

The Hessian for expert `E` is computed directly from its calibration inputs:

$$H_E = X_E X_E^T \tag{2}$$

Because `X_E` (the tokens routed to expert `E`) is small, this matmul is cheap.

Calibration propagates sequentially through the partially quantized network:

$$X_{l+1} = Q_l X_l$$

so each layer's calibration inputs reflect the quantization error accumulated by all previous layers.

**Robustness modifications for MoEs:**

- Hessian dampening is set to `δ = 0.1` (10x higher than standard GPTQ's 0.01) to prevent numerical breakdown from near-singular Hessians.
- If a Hessian remains non-invertible after dampening, the layer falls back to vanilla round-to-nearest.
- When routing is highly skewed, the number of tokens used for a given expert is capped at 4x the mean to prevent OOM.

**Expert grouping for throughput.** Running GPTQ independently per expert is slow. QMoE groups `|E| = 16` experts and runs a batched GPTQ variant that stacks their inputs and Hessians as 3D tensors. This yields roughly 6x speedup:

| Group size | Time (switch-base-128, 10K samples) |
|---|---|
| 1 | 174.1 s |
| 4 | 54.4 s |
| 16 | 28.8 s |

**Special token masking.** MLM separator tokens are excluded from Hessian computation because they have atypical activation patterns that distort the calibration signal. For encoders, special tokens are masked out. For decoders, the token immediately before a special token is skipped. This single change dramatically improves ternary accuracy:

| Configuration | BF16 loss | 2-bit loss | Ternary loss |
|---|---|---|---|
| Without masking | 1.73 | 1.86 | 2.16 |
| With masking | 1.73 | 1.76 | 1.99 |

**Ineffective heuristics.** Activation reordering (a technique that helps in dense models) actually *hurts* ternary MoE performance. True sequential execution is roughly neutral. These findings suggest MoE experts have different quantization dynamics than dense layers.

### 4.3 Scalable Activation Management

The 1.6T model's weights alone exceed 3 TB. Storing calibration activations on top of that is impossible with naive approaches. QMoE uses a carefully designed memory management scheme:

**Single large buffer `B`.** All calibration token hidden states are stored in one contiguous CPU buffer, with delimiter indices marking sample boundaries.

**Dense-part execution loop (per sample):**
1. Fetch sample `X` from CPU to GPU.
2. Run dense (non-expert) layers to get `Y`.
3. Compute and store expert routing assignments for tokens in `Y`.
4. Send `Y` back to CPU, overwriting `X` in `B`.

**Sparse-part execution loop (per expert):**
1. Fetch all tokens assigned to expert `E` (denoted `X_E`) from CPU to GPU.
2. Compress expert `E` using GPTQ to get `E'`.
3. Run `X_E` through `E'` to get `Y_{E'}`.
4. Send `Y_{E'}` back to CPU, overwriting `X_E` in `B`.

Each token is read and written exactly twice per Transformer block. The buffer `B` is never duplicated.

**Lazy weight fetching.** Since the 1.6T model's weights don't fit in CPU RAM either, weights are fetched directly from disk on demand and released after use. Simply iterating through the model's weights on disk takes close to 5 hours on their storage hardware.

### 4.4 Sub-1-Bit Representation: The Dictionary Code

#### Why Ternary Weights Have Low Entropy

The ternary quantization grid for a row is `{w_min, 0, w_max}`. Because expert weight distributions are approximately Gaussian and the grid is wide (spanning the full min-to-max range), most weights round to zero. The observed zero rates are:

| Model | 2-bit zero rate | Ternary zero rate |
|---|---|---|
| switch-base-128 | 72.2% | 85.7% |
| switch-large-128 | 73.1% | 86.4% |
| switch-c2048 | 76.5% | 88.6% |

For c2048 with `P(0) = 0.886`, the theoretical entropy per weight is:

$$H = -0.886 \log_2(0.886) - 2 \times 0.057 \log_2(0.057) \approx 0.62 \text{ bits/weight}$$

This is the information-theoretic lower bound. QMoE achieves approximately 0.807 bits/weight for the full c2048 model (including uncompressed non-expert layers), and roughly 0.5 bits/weight for the expert weights alone.

#### Why Not Standard Sparse Formats?

The obvious approach is to store a bitmask of nonzero positions plus the nonzero values. But:
- The bitmask costs 1 bit/weight.
- Column indices for nonzeros cost 10-13 bits depending on layer width.
- At ternary precision (already very low), this metadata overhead dominates.

Standard sparse formats are not competitive here.

#### Why Not Huffman or Arithmetic Coding?

Entropy codes like Huffman assign variable-length codewords to symbols. Decoding requires sequential processing because you don't know where one symbol ends and the next begins until you've decoded the current one. On a GPU with 32 threads per warp all needing to work in parallel, sequential decoding is a disaster. Different threads would decode different numbers of symbols, causing severe divergence.

#### The Fixed-Codeword Dictionary Solution

QMoE uses a **fixed-width codeword to variable-length symbol sequence** mapping. Every codeword is exactly 16 bits (a `UINT16`). Each codeword maps to a dictionary entry containing up to 28 ternary weights (14 pairs of 2 weights each). The key insight: the codeword width is fixed, but the *number of weights it represents* varies. This is the inverse of Huffman coding.

**Dictionary generation.** The dictionary is built to cover the 2^16 highest-probability ternary sequences. The probability model assumes independent ternary values:

$$P(0) = p_0, \quad P(1) = P(2) = \frac{1 - p_0}{2} \tag{3}$$

For a pair `t = (t_1, t_2)`:

$$P(t) = P(t_1) \cdot P(t_2) \tag{4}$$

The dictionary generation algorithm (Algorithm 1 from the paper):

```
Q ← max priority queue containing (1.0, ())
while |D| < 2^16 do
    p, s ← pop(Q)
    append s to dictionary if 0 < |s| < 28
    for t ∈ {(t1, t2) | t1, t2 ∈ {0, 1, 2}} do
        push((p · P(t), cat(s, t)), Q)
    end for
end while
```

This is essentially a beam search over ternary sequences, keeping the 2^16 most probable ones. The dictionary is sorted highest-to-lowest probability so that frequent codewords cluster in GPU L1/L2 cache.

**Dictionary entry layout.** Each dictionary entry is two consecutive `UINT32` values (64 bits total):
- Each `UINT32` holds up to 7 ternary pairs (14 weights).
- Each ternary value uses 2 bits (not base-3 encoding, to avoid slow modulo/division on GPU).
- The final 4 bits of each `UINT32` encode the pair count for that half-entry.
- Unfilled slots are padded.

This layout means each codeword can represent between 1 and 28 ternary weights. The average, weighted by probability, exceeds 16 (the codeword width in bits), achieving sub-1-bit compression.

**Compression rate validation.** On real c2048 ternary weights, the dictionary achieves 20.07x compression. On independently sampled matrices from the probability model (Equation 3), it achieves 21.11x. The gap is only ~5%, validating the independence assumption. The theoretical entropy limit for `p_0 = 0.885` is 25.40x, so the dictionary scheme is about 20% away from the information-theoretic optimum.

### 4.5 Decompression at Inference: The Fused GPU Kernel

The primary inference operation is a **fused decompression + matrix-vector product**. Weights are never fully decompressed into a dense buffer; instead, each weight is decoded, dequantized, and immediately multiplied by the corresponding input element.

**Stored data per compressed expert matrix:**
- `w_comp`: compressed codeword buffer (variable length per row)
- `row_off`: row offsets into `w_comp`
- `ter_minmax`: per-row dequantization values `(w_min, w_max)` as `bfloat16x2`
- `dec`: the shared dictionary (fits in GPU L2 cache)

**Ternary dequantization mapping:**
```
ternary value 0  →  0.0
ternary value 1  →  w_min  (row-specific)
ternary value 2  →  w_max  (row-specific)
```

**Kernel signature (simplified pseudocode):**

```cpp
__global__ void Sub1MatVec<int num_warps, int w_width>(
    int* dec,
    ushort* w_comp, int* row_off, __nv_bfloat162* ter_minmax,
    __nv_bfloat16* x, __nv_bfloat16* y
)
```

**Shared memory layout:**
- `x_shared[w_width]`: full input vector as float (loaded collaboratively by all warps)
- `deq[3][32 * num_warps]`: ternary dequantization lookup table, replicated 32 times to avoid bank conflicts
- `w_comp_block[32][num_warps]`: staging buffer for codewords

**Kernel execution flow:**

```cpp
// Step 1: Load input vector into shared memory
for (int i = thread; i < w_width; i += 32 * num_warps)
    x_shared[i] = __bfloat162float(x[i]);

// Step 2: Set up per-row dequantization table
deq[0][thread] = 0;
deq[1][thread] = __bfloat162float(ter_minmax[row].x);  // w_min
deq[2][thread] = __bfloat162float(ter_minmax[row].y);  // w_max

// Step 3: Process codewords for this row
for (int i = 0; i < row_off[row + 1] - row_off[row]; i += 32) {
    // Coalesced load of 32 codewords into shared memory
    w_comp_block[warp][lane] = w_comp[i + lane];

    // Each of 28 active threads decodes one weight position
    if (lane < 28) {
        for (int j = 0; j < 32; j++) {
            int enc = w_comp_block[warp][j];
            // Look up dictionary entry; lane/14 selects first or second UINT32
            int wx14 = dec[2 * enc + (lane / 14)];
            // Extract ternary value for this thread's position
            int ter = (wx14 >> (4 + 2 * (lane % 14))) & 0x3;
            // Dequantize and accumulate
            float w = deq[ter][thread];
            res += w * x_shared[idx + lane];
            // Advance input index by number of weights in this codeword
            idx += 2 * (wx14 & 0xf);  // pair count stored in low 4 bits
        }
    }
}

// Step 4: Warp reduction and output
for (int i = 16; i > 0; i /= 2)
    res += __shfl_down_sync(0xffffffff, res, i);
if (lane == 0)
    y[row] += __float2bfloat16(res);
```

**Parallelization policy:**
- One warp (32 threads) per matrix row.
- One threadblock per GPU SM.
- Number of warps per block: `min(#rows in block, 32)`.
- If a block has more than 32 rows, some warps process multiple rows sequentially.
- This avoids wave quantization effects and performs well across the matrix shapes found in MoE experts.

**Key design choices explained:**

*Why 28 active threads, not 32?* Each codeword maps to at most 14 pairs = 28 weights. Threads 0-13 decode from the first UINT32 of the dictionary entry; threads 14-27 decode from the second. Threads 28-31 are idle during weight decoding but participate in the warp reduction.

*Why replicate the dequantization table 32 times?* The 28 active threads may decode different ternary values (0, 1, or 2) and thus access different rows of `deq`. Without replication, threads accessing the same bank simultaneously would serialize. Replication ensures each thread accesses a unique bank.

*Why fixed-width codewords?* Variable-length codewords require sequential decoding. Fixed-width codewords let all 32 threads in a warp fetch their codeword simultaneously with a single coalesced memory transaction.

### 4.6 Encoding Implementation

Compression is done offline (once, before deployment):

1. Build a trie from the dictionary sequences to codewords.
2. For each matrix row, traverse the trie with longest-prefix matching to assign codewords.
3. Pack variable-length compressed rows densely into a contiguous buffer.
4. Record `row_off` as the byte offset of each row's start.

Encoding uses a straightforward GPU kernel with one thread per matrix row.

### 4.7 Mixed-Precision Aspects

QMoE is inherently mixed-precision:

| Component | Precision |
|---|---|
| Expert weights (compressed) | ~0.8 bits/param (ternary + dictionary) |
| Expert dequantization scales | bfloat16 per row (w_min, w_max) |
| Non-expert layers (attention, norms, embeddings) | bfloat16 |
| Activations during inference | bfloat16 |
| Internal accumulation in kernel | float32 |

The non-expert layers are small relative to the experts in large MoEs (c2048 is 99.9% expert parameters), so keeping them in bfloat16 has negligible impact on total model size.

---

## 5. Granularity

Compression operates at **row granularity** for the dictionary encoding. Each matrix row is encoded independently into a variable-length sequence of 16-bit codewords. This means:

- Row offsets (`row_off`) must be stored to locate each row's compressed data.
- Dequantization scales (`w_min`, `w_max`) are stored per row in bfloat16.
- The dictionary itself is shared across all rows and all experts (it's a global constant).

The GPTQ quantization step operates at **column group granularity** within each row (standard GPTQ behavior), but the final ternary values are stored at row granularity for the dictionary encoder.

Expert grouping (`|E| = 16`) is used during the quantization phase for throughput, but each expert's weights are compressed independently.

---

## 6. Static vs. Dynamic

QMoE is **fully static compression**:

- The dictionary is precomputed once based on the expected zero rate `p_0` and never changes.
- Quantization (GPTQ) is run once offline on calibration data.
- The compressed weight buffer is fixed at deployment time.
- No dynamic adaptation occurs during inference.

The only "dynamic" element is the routing in the MoE itself (which tokens go to which experts), but that's a property of the base model, not of QMoE's compression.

This static nature is a deliberate design choice: it enables the simple, fast GPU kernel with no runtime overhead for compression decisions.

---

## 7. Formulas and Algorithms

### Core Quantization Objective

$$\arg\min_{Q_l} \| Q_l X_l - W_l X_l \|_F^2 \tag{1}$$

### Expert Hessian

$$H_E = X_E X_E^T \tag{2}$$

### Ternary Probability Model

$$P(0) = p_0, \quad P(1) = P(2) = \frac{1 - p_0}{2} \tag{3}$$

### Pair Probability (Independence Assumption)

$$P(t) = P(t_1) \cdot P(t_2), \quad t = (t_1, t_2) \tag{4}$$

### Ternary Dequantization

$$w = \begin{cases} 0 & \text{if } q = 0 \\ w_{\min} & \text{if } q = 1 \\ w_{\max} & \text{if } q = 2 \end{cases} \tag{5}$$

where `q` is the stored ternary value and `w_min`, `w_max` are per-row bfloat16 scale factors.

### Theoretical Entropy Bound

For `p_0 = 0.885`:

$$H = -p_0 \log_2 p_0 - 2 \cdot \frac{1-p_0}{2} \log_2 \frac{1-p_0}{2} \approx 0.62 \text{ bits/weight}$$

Theoretical compression limit from bfloat16 (16 bits): `16 / 0.62 ≈ 25.8x` (paper states 25.40x for their specific `p_0`).

### Dictionary Generation Algorithm

```
Input: target dictionary size 2^16, pair probability function P(t)
Output: dictionary D of ternary sequences

Q ← max priority queue initialized with (1.0, ())
while |D| < 2^16 do
    p, s ← pop(Q)                          // highest-probability sequence
    if 0 < |s| < 28 then
        append s to D
    end if
    for each pair t = (t1, t2) with t1, t2 ∈ {0, 1, 2} do
        push((p · P(t), concat(s, t)), Q)
    end for
end while
```

### Compression Rate Formula

$$\text{compression rate} = \frac{16 \cdot N_{\text{weights}}}{\sum_{\text{rows}} 16 \cdot |\text{codewords in row}|}$$

where the numerator is the bfloat16 baseline size and the denominator is the compressed size.

---

## 8. Experimental Results

### 8.1 Accuracy: C4 Validation Loss

Lower is better. BF16 is the uncompressed baseline.

| Model | BF16 | RTN 2-bit | RTN Tern | QMoE 2-bit | QMoE Tern |
|---|---|---|---|---|---|
| switch-base-128 (~7B) | 1.73 | 2.27 | 4.54 | 1.76 | 1.99 |
| switch-large-128 (~26B) | 1.55 | 1.96 | 2.79 | 1.56 | 1.69 |
| switch-c2048 (~1.6T) | 1.18 | 1.33 | 2.15 | 1.20 | 1.26 |

QMoE ternary on c2048 achieves only **6.7% relative loss increase** vs. BF16. RTN ternary on the same model degrades by 82%.

**Effect of calibration data size on c2048:**

| Calibration samples | 2-bit loss | Ternary loss |
|---|---|---|
| 80K (0.5x) | 1.22 | 1.27 |
| 160K (1.0x) | 1.20 | 1.26 |
| 320K (2.0x) | 1.21 | 1.26 |

Diminishing returns beyond 160K samples; the default is 160K for c2048.

**Cross-domain generalization of c2048 compressed model:**

| Domain | BF16 | 2-bit | Ternary |
|---|---|---|---|
| Arxiv | 1.31 | 1.34 | 1.42 |
| GitHub | 0.99 | 1.05 | 1.13 |
| StackExchange | 1.15 | 1.17 | 1.22 |
| Wikipedia | 1.20 | 1.24 | 1.32 |

Calibration used only C4 data (< 0.01% from these domains), yet quality transfers well.

### 8.2 Compression Rates and Model Sizes

| Model | Original size | Compressed size | MoE-only ratio | Full model ratio | Bits/param |
|---|---|---|---|---|---|
| switch-base-128 | 14.9 GB | 1.27 GB | 17.06x | 11.76x | ~1.36 |
| switch-large-128 | 52.7 GB | 3.96 GB | 18.34x | 13.32x | ~1.20 |
| switch-c2048 | 3,142 GB | 158.6 GB | 20.07x | 19.81x | **0.807** |

The c2048 model compresses from **3.2 TB to 158.6 GB**, fitting on 4x A6000 GPUs or 8x RTX 3090s. Compression improves with model scale because larger models have higher natural sparsity and more independence-like weight distributions.

### 8.3 Compression Runtime (Single NVIDIA A6000)

| Model | Samples | Time |
|---|---|---|
| switch-base-128 | 5K | 8.4 min |
| switch-base-128 | 10K | 14.0 min |
| switch-base-128 | 20K | 21.6 min |
| switch-large-128 | 5K | 22.0 min |
| switch-large-128 | 10K | 30.2 min |
| switch-large-128 | 20K | 45.2 min |
| switch-c2048 | 80K | 13.3 h |
| switch-c2048 | 160K | 16.0 h |
| switch-c2048 | 320K | 20.8 h |

The 1.6T model can be compressed in under 24 hours on a single GPU. Note that simply loading the original model's weights from disk takes ~5 hours.

### 8.4 Inference Speed: Per-Layer Kernels

Compressed matvec kernels vs. standard bfloat16 cuBLAS on MoE-relevant matrix shapes:

| Matrix shape | Hardware | Speedup |
|---|---|---|
| 768 × 3072 | RTX 3090 | up to 35% faster |
| 3072 × 768 | RTX 3090 | faster |
| 1024 × 4096 | A6000 | faster |
| 4096 × 1024 | A6000 | faster |
| 2080 × 6144 | both | faster |
| 6144 × 2080 | both | faster |

The compressed kernel is faster than bfloat16 in **all tested cases** on both GPUs. Latency ranges from < 0.02 ms (smallest shape) to < 0.05 ms (largest shape).

The speedup comes from reduced memory bandwidth: fetching 16-bit codewords that decode to 28 weights is far more bandwidth-efficient than fetching 28 bfloat16 values directly.

### 8.5 End-to-End Inference Speed

Task: process one C4 prompt and generate 128 tokens with switch-c2048.

- **Compressed deployment:** 4x A6000 or 8x RTX 3090.
- **Uncompressed baseline:** estimated using a trick where all experts share the same weight pointer (removes memory pressure while preserving compute overhead). This is an optimistic baseline; real uncompressed deployment would need 65+ A6000s with added inter-GPU communication.
- **Result:** end-to-end compressed execution is **< 5% slower** than the idealized uncompressed baseline.

The slight slowdown (despite faster per-layer kernels) comes from the encoder processing multiple tokens per expert: the current kernel runs a separate matvec per token, while the uncompressed baseline can batch them into a single matmul.

---

## 9. Implementation Details

### 9.1 GPU Kernel Architecture

The `Sub1MatVec` kernel is templated on `num_warps` (warps per threadblock) and `w_width` (input vector width). Key implementation choices:

**Threadblock organization:**
- One threadblock per SM.
- One warp per row, up to 32 warps per block.
- If a block has more than 32 rows, warps process multiple rows sequentially.
- This avoids wave quantization effects (where the last "wave" of threadblocks is underutilized).

**Memory access patterns:**
- Input vector: loaded once into shared memory collaboratively by all warps, then accessed from shared memory with no bank conflicts (contiguous access pattern).
- Codewords: fetched in coalesced 32-element transactions (32 threads × 1 UINT16 each = 64 bytes, one cache line).
- Dictionary: accessed via `dec[2 * enc + (lane / 14)]`. The dictionary is small enough to fit in L2 cache; frequent codewords cluster at low indices due to probability-sorted ordering, benefiting L1 prefetching.

**Dequantization table replication:**
- `deq[3][32 * num_warps]` stores `{0, w_min, w_max}` replicated 32 times.
- Without replication, threads 0-27 accessing `deq[ter][0]` for different `ter` values would hit the same bank (bank = address mod 32), causing 3-way serialization.
- With replication, thread `t` accesses `deq[ter][t]`, which maps to bank `t mod 32`, guaranteeing conflict-free access.

**Warp reduction:**
```cpp
for (int i = 16; i > 0; i /= 2)
    res += __shfl_down_sync(0xffffffff, res, i);
```
Standard butterfly reduction using warp shuffle intrinsics. No shared memory needed for the reduction.

### 9.2 Encoding Kernel

The encoding kernel uses one thread per matrix row:
1. Traverse the dictionary trie with longest-prefix matching on the row's ternary values.
2. Emit a 16-bit codeword for each matched sequence.
3. Write codewords to a per-row output buffer.
4. Record the row's codeword count in `row_off`.

After all rows are encoded, a prefix sum over `row_off` converts counts to offsets, and rows are packed into a contiguous buffer.

### 9.3 Framework Integration

QMoE is built on top of the PyTorch backend of HuggingFace Transformers. Several fixes were needed to run the trillion-parameter models:

- **Empty expert skip:** HuggingFace's original code called CUDA kernels even for experts with zero routed tokens. Skipping these calls yielded > 10x speedup for large models with many experts.
- **Config and model setup fixes:** Various bugs in the HuggingFace implementation of large SwitchTransformer variants were patched.
- All changes are applied dynamically at runtime via monkey-patching.

### 9.4 Hardware Requirements

| Task | Hardware |
|---|---|
| Compression (base/large) | Single A6000 (48 GB GPU) + ~100 GB CPU RAM |
| Compression (c2048) | Single A6000 + several hundred GB CPU RAM |
| Inference (c2048 compressed) | 4x A6000 or 8x RTX 3090 |
| Inference (c2048 uncompressed) | 65+ A6000 or 130+ RTX 3090 |

---

## 10. Limitations and Gaps

**Model coverage.** QMoE is evaluated only on SwitchTransformer variants. Few massive, accurate, publicly available MoE models existed at the time of writing. The authors acknowledge this and mention future work on other MoE families (Artetxe et al. 2022, SoftMoEs).

**Bespoke framework dependency.** Most production MoE models use custom frameworks (e.g., Megatron-LM, DeepSpeed-MoE). Integrating QMoE into these requires non-trivial engineering work.

**Batched token handling.** When multiple tokens route to the same expert (common in encoder models), the current kernel runs a separate matvec per token. This is less efficient than a batched matmul. The authors propose two fixes: (1) add an inner token loop inside the kernel, or (2) fully decompress to a dense buffer and run standard matmul for large batches. Neither is implemented in the paper.

**No downstream fine-tuning.** QMoE compresses pretrained base models. There are no experiments on fine-tuned models or on recovering accuracy via post-compression fine-tuning (analogous to QLoRA). The authors note this as future work, pointing out that non-expert layers remain in bfloat16 and could serve as trainable adapters.

**Compression rate gap.** The dictionary scheme achieves ~20.07x compression on c2048 expert weights, while the theoretical entropy limit is 25.40x. The ~20% gap comes from the fixed-codeword constraint (which sacrifices some coding efficiency for GPU-friendly decoding) and the independence assumption (which is approximate).

**Encoder-specific calibration tricks.** The special token masking heuristic is specific to masked language models. Applying QMoE to decoder-only MoEs (e.g., Mixtral-style models) may require different calibration strategies.

**No 2-bit dictionary compression.** The dictionary encoding is designed specifically for ternary weights. The 2-bit quantization results use standard GPTQ without the dictionary step, so 2-bit models don't achieve sub-1-bit compression.

---

## 11. Relevance to Our Work: NVFP4 and the Hot/Cold Expert Paradigm

### Our Context

We work with NVFP4 (4-bit floating point) quantization in TensorRT-LLM. QMoE operates at sub-1-bit, roughly 5x more aggressive than FP4. The question is whether QMoE's ideas are applicable or complementary to our work.

### The Hot/Cold Expert Opportunity

MoE routing is not uniform. In practice, a small fraction of experts ("hot" experts) receive the majority of tokens, while most experts ("cold" experts) are rarely activated. This creates a natural tiered compression opportunity:

- **Hot experts** are on the critical path for latency. They need fast, high-quality computation. FP4 or FP8 is appropriate here.
- **Cold experts** are rarely activated. Their contribution to end-to-end latency is minimal, but they still consume memory. Sub-1-bit compression (QMoE-style) could dramatically reduce their memory footprint with acceptable accuracy cost.

A hybrid scheme could look like:

| Expert tier | Compression | Rationale |
|---|---|---|
| Top-K hot experts | NVFP4 (4-bit) | Low latency, high accuracy, hardware-native |
| Mid-tier experts | INT8 or FP8 | Balance of quality and size |
| Cold experts (rarely activated) | QMoE ternary + dictionary | Minimize memory, tolerate higher latency |

For a model like Mixtral-8x7B or a hypothetical trillion-parameter successor, this tiered approach could allow deployment on far fewer GPUs while maintaining near-baseline quality on common inputs.

### Technical Compatibility Considerations

**Dictionary decoding overhead.** QMoE's kernel is faster than bfloat16 for matvec (single-token) operations. But TensorRT-LLM typically operates in batched mode with multiple tokens per expert. For large batches, the per-token matvec approach becomes a bottleneck. A practical integration would need the "decompress-then-matmul" fallback for hot experts under load, or a batched variant of the fused kernel.

**NVFP4 hardware acceleration.** NVFP4 benefits from dedicated hardware on Blackwell (B100/B200) via the FP4 tensor core instructions. QMoE's ternary+dictionary format has no hardware acceleration path; it relies on software decoding. For hot experts where throughput matters, NVFP4 with hardware acceleration will always win on modern NVIDIA GPUs.

**Calibration data requirements.** QMoE requires calibration data and GPTQ-style optimization. This is already standard practice for PTQ in TensorRT-LLM workflows, so the infrastructure exists. The main addition is the dictionary encoding step, which is fast (minutes to hours depending on model size).

**Memory layout.** QMoE stores compressed weights in a custom format (codeword buffer + row offsets + per-row scales). Integrating this into TensorRT-LLM's weight management would require a new weight format and corresponding plugin/kernel. This is non-trivial but feasible.

### What QMoE Offers That FP4 Doesn't

- **Sub-1-bit compression.** FP4 is 4 bits; QMoE achieves ~0.8 bits. For memory-bound cold experts, this is a 5x additional reduction.
- **Compression of trillion-parameter models on a single GPU.** The scalable activation management system is a genuine engineering contribution applicable to any large-scale PTQ workflow.
- **Faster matvec than bfloat16.** For single-token (decode-phase) inference, the compressed kernel outperforms bfloat16 cuBLAS. This could benefit autoregressive generation where batch size is often 1.

### What FP4 Offers That QMoE Doesn't

- **Hardware acceleration.** Blackwell FP4 tensor cores provide massive throughput for batched matmuls.
- **Simpler integration.** FP4 is a standard format with TensorRT-LLM support.
- **Better accuracy at the same bitrate.** FP4 (4 bits) vs. QMoE ternary (~0.8 bits) is not a fair comparison, but at 2-bit, QMoE's GPTQ-based approach significantly outperforms RTN.
- **Broader applicability.** FP4 works for any layer type; QMoE's dictionary encoding is designed specifically for ternary MoE experts.

### Recommended Integration Strategy

For a TensorRT-LLM deployment of a large MoE model:

1. **Profile expert activation frequency** over representative workloads.
2. **Classify experts** into hot (top 10-20% by activation count) and cold (bottom 80-90%).
3. **Apply NVFP4** to hot experts using TensorRT-LLM's existing quantization pipeline.
4. **Apply QMoE ternary + dictionary** to cold experts, accepting the higher latency for rare activations.
5. **Implement a runtime dispatch** that routes tokens to the appropriate kernel (FP4 matmul vs. QMoE matvec) based on expert ID.

This hybrid approach could reduce total model memory by 3-5x compared to uniform FP4, potentially enabling deployment on half as many GPUs.

---

## 12. Comparison Table

### QMoE vs. Related Compression Methods

| Method | Bits/param | Accuracy (c2048 ternary) | GPU kernel | Scalable to 1.6T | Compression time |
|---|---|---|---|---|---|
| BF16 (baseline) | 16 | 1.18 (loss) | cuBLAS | N/A | N/A |
| RTN 2-bit | 2 | 1.33 | Custom | Yes | Minutes |
| RTN Ternary | ~1.6 | 2.15 | Custom | Yes | Minutes |
| GPTQ 2-bit | 2 | 1.20 | Custom | Requires adaptation | Hours |
| **QMoE 2-bit** | **2** | **1.20** | **Fused decomp+matvec** | **Yes (novel)** | **16 h (c2048)** |
| **QMoE Ternary** | **~0.8** | **1.26** | **Fused decomp+matvec** | **Yes (novel)** | **16 h (c2048)** |
| NVFP4 | 4 | N/A (different models) | FP4 tensor cores | Yes | Hours |
| INT8 | 8 | N/A | INT8 tensor cores | Yes | Hours |

### QMoE Compression Rates by Model

| Model | Params | BF16 size | Compressed size | Ratio | Bits/param |
|---|---|---|---|---|---|
| switch-base-128 | ~7B | 14.9 GB | 1.27 GB | 11.76x | ~1.36 |
| switch-large-128 | ~26B | 52.7 GB | 3.96 GB | 13.32x | ~1.20 |
| switch-c2048 | ~1.6T | 3,142 GB | 158.6 GB | 19.81x | **0.807** |

### QMoE vs. Sparse Formats for Ternary Weights

| Format | Bits/weight | GPU-friendly | Notes |
|---|---|---|---|
| Dense ternary (2-bit) | 2.0 | Yes | No compression of zeros |
| Bitmask + values | ~1.0 + overhead | Partially | Bitmask alone costs 1 bit |
| CSR sparse | ~10-13 bits for indices | Poor | Index overhead dominates |
| **QMoE dictionary** | **~0.5 (expert-only)** | **Yes** | **Fixed-width codewords** |
| Huffman coding | ~0.62 (theoretical) | No | Sequential decoding |
| Arithmetic coding | ~0.62 (theoretical) | No | Sequential decoding |

### Hardware Requirements Comparison

| Deployment scenario | GPUs needed | GPU type | Notes |
|---|---|---|---|
| c2048 BF16 (uncompressed) | 65+ | A6000 (48 GB) | Impractical |
| c2048 BF16 (uncompressed) | 130+ | RTX 3090 (24 GB) | Impractical |
| c2048 QMoE ternary | 4 | A6000 (48 GB) | Practical single node |
| c2048 QMoE ternary | 8 | RTX 3090 (24 GB) | Practical single node |
| c2048 QMoE 2-bit | ~8 | A6000 (48 GB) | Estimated |

---

## Summary

QMoE is a technically elegant solution to a genuine engineering problem. Its core contribution is not a new quantization algorithm but a **co-designed compression format and GPU kernel** that exploits the natural low entropy of ternary MoE expert weights. The fixed-codeword dictionary scheme is the key innovation: it achieves near-entropy-optimal compression while remaining GPU-decodable at inference speed.

The scalable activation management system is an equally important (if less glamorous) contribution, enabling data-dependent quantization of models that don't fit in any reasonable amount of RAM.

For our work with NVFP4 in TensorRT-LLM, QMoE's most actionable insight is the **hot/cold expert tiering** opportunity. Cold experts are memory-bound but latency-tolerant, making them ideal candidates for aggressive sub-1-bit compression. A hybrid FP4/QMoE deployment could significantly reduce the GPU count required for trillion-parameter MoE inference while maintaining quality on common workloads.
