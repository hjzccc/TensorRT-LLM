# CUDA Kernel Proposal: Fused Heterogeneous MoE Dispatch

**Date:** 2026-02-20
**Status:** Proposal
**Author:** Analysis of `policy/heter_dispatch.py` and `fused_moe_heter.py`

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [GPU-CPU Synchronization Audit](#2-gpu-cpu-synchronization-audit)
3. [Proposed Solutions](#3-proposed-solutions)
   - [Solution A: GPU-Resident PyTorch Ops (Quick Win)](#solution-a-gpu-resident-pytorch-ops)
   - [Solution B: Fused Two-Pass Dispatch Kernel](#solution-b-fused-two-pass-dispatch-kernel)
   - [Solution C: Single-Pass Atomic Dispatch](#solution-c-single-pass-atomic-dispatch)
   - [Solution D: Counting Sort + Segment Dispatch](#solution-d-counting-sort--segment-dispatch)
4. [Solution Comparison Matrix](#4-solution-comparison-matrix)
5. [ExpertLoadHeterDispatch Deep Dive](#5-expertloaddispatch-deep-dive)
6. [CUDA Graph Compatibility](#6-cuda-graph-compatibility)
7. [Recommended Implementation Plan](#7-recommended-implementation-plan)
8. [Root Cause: Interface Design](#8-root-cause-interface-design-forces-gpu-cpu-round-trip)
9. [Warp-Level Optimization](#9-warp-level-optimization-for-scatter-kernel)
10. [2-Group Fast Path: torch.topk](#10-2-group-fast-path-torchtopk-best-quick-win-for-expertload)
11. [**Verdict: Optimal 2-Group ExpertLoad Dispatch**](#11-verdict-optimal-2-group-expertload-exclusive-dispatch)

---

## 1. Current Architecture Analysis

### Data Flow Overview

```
                    CURRENT PIPELINE (Python + GPU mixed)
    ================================================================

    token_selected_experts [N, K]     token_final_scales [N, K]
              |                                |
              v                                v
    +-------------------+          +-------------------+
    |  _assign()        |          |                   |
    |  (policy-specific)|          |                   |
    |  GPU: bincount,   |          |                   |
    |    scatter_add     |          |                   |
    |  CPU: .tolist(),   | <-------+                   |
    |    sort, partition |                              |
    +--------+----------+                              |
             |                                         |
             v  List[List[int]]  (CPU)                 |
    +-------------------+                              |
    |  _dispatch_by_    |                              |
    |  assignment()     | <----------------------------+
    |                   |
    |  GPU: build       |
    |    expert_to_group|
    |  Python loop x G: |
    |    mask, filter,  |
    |    clone, zero    |
    +--------+----------+
             |
             v  List[(tok_idx, experts, scales)]
    +-------------------+
    |  run_moe()        |
    |  Python loop x G: |
    |    quantize input |
    |    fused_moe()    |
    |    accumulate     |
    +-------------------+
             |
             v
        accumulated [N, H]
```

### Key Dimensions (Typical)

| Parameter | Decoding | Prefill |
|-----------|----------|---------|
| `num_tokens` (N) | 1-64 | 256-8192 |
| `top_k` (K) | 2-8 | 2-8 |
| `num_experts` (E) | 64-256 | 64-256 |
| `num_groups` (G) | 2-3 | 2-3 |
| `hidden_size` (H) | 2048-7168 | 2048-7168 |

---

## 2. GPU-CPU Synchronization Audit

### Identified Sync Points

```
    GPU                          CPU                     Sync Type
    ===                          ===                     =========

    1. bincount/scatter_add
       (expert scores)
              |
              +--- .tolist() -----> argsort result      GPU->CPU
              |                     as Python list       (BLOCKING)
              |
              |                     sort + partition
              |                     into groups
              |
              +<--- tensor() ----- expert_to_group      CPU->GPU
              |                     from Python lists    (ASYNC but
              |                                          adds latency)
              |
    2. Python for-loop over G groups:
       Each iteration launches:
         - comparison ops
         - nonzero()
         - clone() x 2
         - masked fill
              |
              v
       G separate kernel launches from Python
```

### Quantified Impact

| Sync Point | Latency | Frequency |
|------------|---------|-----------|
| `torch.argsort().tolist()` | ~50-200 us (cudaDeviceSynchronize) | Every `dispatch()` call |
| `torch.tensor(list, device=cuda)` | ~10-30 us (H2D copy) | G times per `dispatch()` |
| Python loop overhead | ~5-10 us per iteration | G iterations |
| Extra kernel launches | ~3-5 us each | ~6G kernels (mask, nonzero, clone, fill per group) |
| **Total overhead** | **~100-400 us** | **Per MoE layer forward** |

For a model with 60 MoE layers, this adds **6-24 ms per forward pass** just from dispatch overhead. During decoding (where the GEMM itself may take only ~50 us per layer), dispatch overhead can be **2-8x the compute time**.

---

## 3. Proposed Solutions

### Solution A: GPU-Resident PyTorch Ops

**Complexity: LOW | Impact: HIGH | Zero C++ Required**

Replace `_assign_by_score` and `_dispatch_by_assignment` with pure PyTorch ops that stay on GPU. No custom CUDA kernel needed.

#### Key Insight

The `.tolist()` call exists only because the code converts GPU argsort results to Python lists for group partitioning. But this partitioning can be done entirely with `torch.scatter` on GPU:

```python
def _assign_by_score_gpu(scores, num_experts, group_sizes):
    """GPU-only: scores -> expert_to_group tensor. Zero CPU sync."""
    # Sort stays on GPU (no .tolist()!)
    sorted_ids = torch.argsort(scores, descending=True)

    # Build expert_to_group directly via scatter
    expert_to_group = torch.empty(num_experts, dtype=torch.long,
                                  device=scores.device)

    # Create group labels for each position in sorted order
    # Example: group_sizes=[102, 26], sorted positions 0..127
    #   positions [0..25]  -> group 1 (last group = high precision)
    #   positions [26..127] -> group 0 (first group = low precision)
    group_labels = torch.empty(num_experts, dtype=torch.long,
                               device=scores.device)
    offset = 0
    # This loop runs on CPU but only iterates G times (2-3)
    # and creates NO GPU-CPU sync
    for gidx, size in enumerate(reversed(group_sizes)):
        group_labels[offset:offset + size] = len(group_sizes) - 1 - gidx
        offset += size

    # Scatter: expert_to_group[sorted_ids[i]] = group_labels[i]
    expert_to_group.scatter_(0, sorted_ids, group_labels)
    return expert_to_group
```

#### Vectorized Dispatch (No Python Loop Over Groups)

```python
def _dispatch_vectorized(expert_to_group, token_selected_experts,
                         token_final_scales, num_groups, num_experts):
    """Produce all G dispatch tuples without Python per-group loops."""
    # [N, K] -> group assignment per expert slot
    slot_groups = expert_to_group[token_selected_experts.long()]  # [N, K]

    results = []
    for gidx in range(num_groups):  # Only 2-3 iterations, no sync
        slot_mask = (slot_groups == gidx)          # [N, K]
        token_has = slot_mask.any(dim=1)           # [N]
        indices = token_has.nonzero(as_tuple=False).squeeze(1)

        if indices.numel() == 0:
            results.append((None, None, None))
            continue

        grp_experts = token_selected_experts[indices].clone()
        grp_experts[~slot_mask[indices]] = num_experts  # sentinel
        grp_scales = token_final_scales[indices].clone()
        grp_scales[~slot_mask[indices]] = 0.0

        results.append((indices, grp_experts, grp_scales))
    return results
```

#### Data Flow (Solution A)

```
    GPU-ONLY PIPELINE (Solution A)
    =============================

    token_selected_experts [N, K]     token_final_scales [N, K]
              |                                |
              v                                |
    +-------------------+                      |
    | GPU: bincount or  |                      |
    |   scatter_add     |                      |
    |   -> scores [E]   |                      |
    +--------+----------+                      |
             |                                 |
             v  (stays on GPU)                 |
    +-------------------+                      |
    | GPU: argsort      |                      |
    |   -> sorted_ids   |                      |
    | GPU: scatter_     |                      |
    |   -> expert_to_   |                      |
    |      group [E]    |                      |
    +--------+----------+                      |
             |                                 |
             v                                 |
    +-------------------+                      |
    | GPU: gather       | <--------------------+
    |   slot_groups     |
    |   [N, K]          |
    | GPU: per-group    |
    |   mask+filter     |
    |   (2-3 iters,     |
    |    no CPU sync)   |
    +--------+----------+
             |
             v
      per-group dispatch tuples (all on GPU)
```

**GPU-CPU syncs: ZERO**
**Extra kernel launches vs current: NONE (same count, just no sync between them)**

---

### Solution B: Fused Two-Pass Dispatch Kernel

**Complexity: MEDIUM | Impact: HIGH | Requires C++ CUDA Kernel**

A custom CUDA kernel that fuses the entire dispatch into two passes:

#### Pass 1: Count Phase

```
    PASS 1: Count per-group token membership
    =========================================

    Input:  expert_to_group[E]           (precomputed, stays on GPU)
            token_selected_experts[N, K]

    Thread assignment: 1 thread per token (N threads)

    Each thread:
      for k in 0..K-1:
        group = expert_to_group[token_selected_experts[tid, k]]
        thread_groups |= (1 << group)   // bitmask

      for g in 0..G-1:
        if thread_groups & (1 << g):
          atomicAdd(&group_counts[g], 1)

    Output: group_counts[G]   (e.g., [1800, 2000] for G=2)

    Kernel config: <<<ceil(N/256), 256>>>
    Shared memory: G integers for block-level reduction before global atomic
```

#### Pass 2: Scatter Phase

```
    PASS 2: Scatter tokens into per-group contiguous buffers
    =========================================================

    Input:  expert_to_group[E]
            token_selected_experts[N, K]
            token_final_scales[N, K]
            group_offsets[G+1]   (prefix sum of group_counts)

    Pre-allocated output buffers (sized from Pass 1):
            out_tok_idx[G][max_N]
            out_experts[G][max_N, K]
            out_scales[G][max_N, K]

    Thread assignment: 1 thread per token

    Each thread:
      for g in 0..G-1:
        if token has expert in group g:
          pos = atomicAdd(&write_cursor[g], 1)
          out_tok_idx[g][pos] = tid
          for k in 0..K-1:
            if expert_to_group[experts[tid,k]] == g:
              out_experts[g][pos, k] = experts[tid, k]
              out_scales[g][pos, k] = scales[tid, k]
            else:
              out_experts[g][pos, k] = E   // sentinel
              out_scales[g][pos, k] = 0.0

    Kernel config: <<<ceil(N/256), 256>>>
```

#### Memory Layout

```
    PRE-ALLOCATED OUTPUT BUFFERS
    ============================

    For G=2, N=4096, K=8:

    group_tok_idx:   [G * N]         = [8192] int32
    group_experts:   [G * N * K]     = [65536] int32
    group_scales:    [G * N * K]     = [65536] float32
    group_counts:    [G]             = [2] int32

    Total: ~0.75 MB  (negligible)

    Layout (contiguous per group):

    group 0:  |--- tok_idx_0 ---|--- experts_0 ---|--- scales_0 ---|
    group 1:  |--- tok_idx_1 ---|--- experts_1 ---|--- scales_1 ---|

    group_counts tells run_moe() how many valid entries each group has.
```

---

### Solution C: Single-Pass Atomic Dispatch

**Complexity: MEDIUM | Impact: HIGHEST | Requires C++ CUDA Kernel**

Combines counting and scattering into a single kernel using atomics. Eliminates the need for a prefix-sum pass.

```
    SINGLE-PASS KERNEL
    ==================

    Pre-allocate worst-case buffers: out_tok_idx[G][N], etc.
    Initialize write_cursors[G] = {0, 0, ...}

    Thread assignment: 1 thread per token

    __global__ void heter_dispatch_fused(
        const int* experts,     // [N, K]
        const float* scales,    // [N, K]
        const int* e2g,         // [E] expert_to_group
        int* out_tok_idx,       // [G, N]
        int* out_experts,       // [G, N, K]
        float* out_scales,      // [G, N, K]
        int* group_counts,      // [G]
        int N, int K, int G, int E)
    {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= N) return;

        // Determine which groups this token belongs to
        unsigned group_mask = 0;
        for (int k = 0; k < K; k++) {
            int expert = experts[tid * K + k];
            int group = e2g[expert];
            group_mask |= (1u << group);
        }

        // Atomically claim a slot in each relevant group
        for (int g = 0; g < G; g++) {
            if (!(group_mask & (1u << g))) continue;

            int pos = atomicAdd(&group_counts[g], 1);

            out_tok_idx[g * N + pos] = tid;

            for (int k = 0; k < K; k++) {
                int expert = experts[tid * K + k];
                int group = e2g[expert];
                int out_idx = g * N * K + pos * K + k;
                if (group == g) {
                    out_experts[out_idx] = expert;
                    out_scales[out_idx] = scales[tid * K + k];
                } else {
                    out_experts[out_idx] = E;   // sentinel
                    out_scales[out_idx] = 0.0f;
                }
            }
        }
    }
```

#### Performance Analysis

```
    SINGLE-PASS PERFORMANCE MODEL
    ==============================

    For N=4096, K=8, G=2, E=128:

    Reads:
      experts[N*K] = 4096*8*4 = 128 KB
      scales[N*K]  = 4096*8*4 = 128 KB
      e2g[E]       = 128*4    = 0.5 KB  (fits in L1/const cache)
      Total reads: ~256 KB

    Writes (worst case, all tokens in all groups):
      tok_idx[G*N]    = 2*4096*4 = 32 KB
      experts[G*N*K]  = 2*4096*8*4 = 256 KB
      scales[G*N*K]   = 2*4096*8*4 = 256 KB
      Total writes: ~544 KB

    Atomics:
      group_counts: G atomics per token = 2*4096 = 8192
      (very low contention, only G=2 counters)

    Expected runtime: ~10-20 us on modern GPU
    vs. current Python dispatch: ~100-400 us

    Speedup: 5-40x on dispatch alone
```

---

### Solution D: Counting Sort + Segment Dispatch

**Complexity: HIGH | Impact: HIGHEST | Requires C++ CUDA Kernel**

Uses counting sort (bin sort) for the expert scoring phase, avoiding the O(E log E) argsort entirely. Particularly effective for `ExpertLoadHeterDispatch` where E is small (64-256).

#### Phase 1: GPU Counting Sort for Expert Assignment

```
    COUNTING SORT FOR EXPERT-TO-GROUP ASSIGNMENT
    =============================================

    Goal: Given expert scores[E], partition into G groups by score rank
          WITHOUT argsort. Output: expert_to_group[E].

    Key insight: E is small (64-256), fits entirely in shared memory.
    We don't need a full sort -- just need to find the partition
    boundary (the k-th largest score) for each group.

    APPROACH: Histogram-based k-th element finding

    Step 1: Find score range [min_score, max_score]
            via parallel reduction (1 warp sufficient for E<=256)

    Step 2: Build histogram of scores into B bins
            B = 256 bins (8-bit quantization of score range)

            +--+--+--+--+--+--+--+--+--+
            |  |  |##|  |##|##|  |##|  |   histogram[B]
            +--+--+--+--+--+--+--+--+--+
             0  1  2  3  4  5  6  7  8

    Step 3: Reverse prefix sum to find partition bin

            For group G-1 (high-precision, gets top-scoring experts):
              size_needed = group_sizes[G-1]
              Walk histogram from right to left:
                cumsum += histogram[b]
                if cumsum >= size_needed:
                    threshold_bin = b  --> threshold_score
                    break

            For group G-2:
              Continue walking for next group's size_needed
              ...

    Step 4: Assign experts to groups

            for each expert e (parallel, 1 thread per expert):
              score = scores[e]
              for g = G-1 downto 0:
                if score >= threshold[g]:
                  expert_to_group[e] = g
                  break

    Complexity: O(E + B) vs O(E log E) for argsort
    All on GPU, ZERO sync points.


    VISUAL: COUNTING SORT vs ARGSORT

    Argsort (current):
    scores:    [3.2, 1.1, 5.7, 0.4, 2.8, 4.3, ...]  (E=128 values)
                 |
                 v
    sorted_ids: [2, 5, 0, 4, 1, 3, ...]  GPU->CPU sync!  <-- PROBLEM
                 |
                 v  (CPU)
    groups:     [[3,1,4,...], [2,5,0,...]]  CPU->GPU       <-- PROBLEM

    Counting Sort (proposed):
    scores:    [3.2, 1.1, 5.7, 0.4, 2.8, 4.3, ...]
                 |
                 v  (all on GPU)
    histogram: |0|0|1|1|0|1|...|  (256 bins)
                 |
                 v  (scan from right)
    threshold: score >= 4.0 -> group 1 (high-prec)
               score <  4.0 -> group 0 (low-prec)
                 |
                 v
    expert_to_group: [0, 0, 1, 0, 0, 1, ...]  (direct, no sort!)
```

#### Phase 2: Warp-Cooperative Token Dispatch

```
    WARP-COOPERATIVE DISPATCH KERNEL
    =================================

    Uses warp-level primitives for efficient group counting
    and scatter without global atomics.

    __global__ void warp_dispatch_kernel(...)
    {
        // 1 warp (32 threads) processes 32 tokens at a time

        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        int token_id = blockIdx.x * (blockDim.x / 32) * 32
                     + warp_id * 32 + lane_id;

        // Each thread checks its token's group membership
        unsigned my_groups = 0;
        for (int k = 0; k < K; k++) {
            int expert = experts[token_id * K + k];
            my_groups |= (1u << e2g[expert]);
        }

        // Warp-level ballot: which threads have tokens for group g?
        for (int g = 0; g < G; g++) {
            unsigned ballot = __ballot_sync(0xFFFFFFFF,
                                           my_groups & (1u << g));
            int count = __popc(ballot);

            if (count == 0) continue;

            // Warp-level exclusive prefix (popcount of lower lanes)
            int warp_offset;
            if (lane_id == 0) {
                warp_offset = atomicAdd(&group_counts[g], count);
            }
            warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);

            if (my_groups & (1u << g)) {
                // My position within this warp's contribution
                unsigned lower_mask = (1u << lane_id) - 1;
                int my_pos = warp_offset
                           + __popc(ballot & lower_mask);

                // Write output
                out_tok_idx[g * N + my_pos] = token_id;
                // ... write experts and scales ...
            }
        }
    }
```

#### Advantage Over Radix Sort

```
    COUNTING SORT vs RADIX SORT for Expert Assignment
    ==================================================

    Expert count E: 64-256

    Counting Sort (Proposed):                Radix Sort (CUB):
    +--------------------------+             +--------------------------+
    | 1. Find min/max: O(E)    |             | 1. CUB setup: temp alloc |
    | 2. Histogram: O(E)       |             | 2. Pass 1 (bits 0-7)    |
    | 3. Scan + threshold: O(B)|             | 3. Pass 2 (bits 8-15)   |
    | 4. Assign: O(E)          |             | 4. Pass 3 (bits 16-23)  |
    +--------------------------+             | 5. Pass 4 (bits 24-31)  |
    Total: O(E + B)                          +--------------------------+
    Kernel launches: 1                       Total: O(E * passes)
    Temp memory: O(B) shared mem             Kernel launches: 4-8
    B = 256 << E, fits in smem              Temp memory: O(E) global

    For E=128:
      Counting sort: ~5 us (1 kernel)
      CUB radix sort: ~15-25 us (4+ kernels)

    Winner: Counting sort (3-5x faster for small E)
```

---

## 4. Solution Comparison Matrix

```
    +----------+---------+----------+---------+---------+----------+--------+
    | Solution | GPU-CPU | Kernel   | Impl    | CUDA    | Dispatch | Best   |
    |          | Syncs   | Launches | Effort  | Graph   | Latency  | For    |
    |          |         | (total)  |         | Safe    | (us)     |        |
    +----------+---------+----------+---------+---------+----------+--------+
    | Current  | 1-2     | ~6G+2    | N/A     | Baked   | 100-400  | N/A    |
    |          |         | (~16)    |         |         |          |        |
    +----------+---------+----------+---------+---------+----------+--------+
    | A: PyTorch| 0      | ~6G+2    | LOW     | Yes*    | 30-80    | Quick  |
    |   GPU-ops |        | (~16)    | (Python |         |          | win,   |
    |          |         |          |  only)  |         |          | no C++ |
    +----------+---------+----------+---------+---------+----------+--------+
    | B: Two-  | 0       | 3        | MEDIUM  | Yes     | 15-30    | Bal-   |
    |   Pass   |         | (count + |         |         |          | anced  |
    |          |         |  prefix  |         |         |          |        |
    |          |         |  +scatter)|        |         |          |        |
    +----------+---------+----------+---------+---------+----------+--------+
    | C: Single| 0       | 1        | MEDIUM  | Yes     | 10-20    | Lowest |
    |   Pass   |         |          |         |         |          | launch |
    |   Atomic |         |          |         |         |          | count  |
    +----------+---------+----------+---------+---------+----------+--------+
    | D: Count | 0       | 2        | HIGH    | Yes     | 8-15     | Best   |
    |   Sort + |         | (assign  |         |         |          | for    |
    |   Warp   |         |  +dispatch)       |         |          | Expert |
    |   Coop   |         |          |         |         |          | Load   |
    +----------+---------+----------+---------+---------+----------+--------+

    * Solution A is CUDA-graph safe if expert_to_group is precomputed
      or recomputed with fixed-shape ops (no dynamic nonzero).
```

### Priority Ranking (Avoiding GPU-CPU Communication)

```
    PRIORITY RANKING
    ================

    #1  Solution A (GPU PyTorch Ops)     <-- START HERE
        - Zero GPU-CPU sync
        - Zero implementation risk
        - Deployable in hours, not weeks
        - Gets 60-75% of the theoretical speedup

    #2  Solution C (Single-Pass Atomic)  <-- NEXT
        - Single kernel launch
        - Minimal atomic contention (G=2-3 counters)
        - Best latency for the dispatch phase
        - Pairs well with Solution A as fallback

    #3  Solution D (Counting Sort)       <-- FOR ExpertLoad SPECIFICALLY
        - Eliminates argsort entirely
        - O(E) vs O(E log E) for assignment
        - Most impactful for dynamic policies
        - Higher implementation effort

    #4  Solution B (Two-Pass)            <-- SKIP
        - Superseded by C (single-pass is simpler AND faster)
        - Only advantage: deterministic output order
        - Not worth the extra kernel launch
```

---

## 5. ExpertLoadHeterDispatch Deep Dive

The `ExpertLoadHeterDispatch` policy is the most latency-sensitive because it recomputes expert scores every forward pass.

### Current Flow

```
    ExpertLoadHeterDispatch._assign()
    ==================================

    Input: token_selected_experts [N, K]

    Step 1: flat_experts = token_selected_experts.reshape(-1)
            counts = torch.bincount(flat_experts, minlength=E)

            experts:  [3, 7, 1, 3, 5, 7, 2, 1, ...]
                              |
                              v
            counts:   [0, 2, 1, 2, 0, 1, 0, 2, ...]
                       e0 e1 e2 e3 e4 e5 e6 e7

    Step 2: _assign_by_score(counts, E, ratios)
            sorted_ids = argsort(counts, descending=True)
                              |
                              v  .tolist() !!  GPU->CPU SYNC
            sorted_ids: [3, 7, 1, 2, 5, 0, 4, 6]

    Step 3: Partition (CPU)
            group_sizes = [6, 2]  (for ratio [0.75, 0.25])
            group 1 (high-prec): [3, 7]      (top 2)
            group 0 (low-prec):  [0, 1, 2, 4, 5, 6]  (bottom 6)
```

### Proposed GPU-Only Replacement

```
    GPU-ONLY ExpertLoadHeterDispatch
    ================================

    Step 1: Same bincount (already on GPU)
            counts = torch.bincount(flat_experts, minlength=E)

    Step 2: GPU threshold finding (NO argsort needed!)

            For G=2, ratio=[0.75, 0.25]:
              need top 25% experts -> high-precision group
              k = round(0.25 * E) = 32  (for E=128)

            METHOD: torch.topk(counts, k=32)
              -> top_values, top_indices

            expert_to_group = torch.zeros(E, device='cuda')
            expert_to_group[top_indices] = 1

            Done! No argsort, no .tolist(), no CPU.

    Step 2 (Alternative): torch.kthvalue for O(E) partition

            # Find the k-th largest count
            kth_val = torch.kthvalue(counts, E - k + 1).values

            # All experts with count >= kth_val -> high-precision
            expert_to_group = (counts >= kth_val).long()

            # Handle ties: if too many experts at boundary,
            # break ties by expert index (deterministic)

    Complexity comparison:
      Current:  O(E log E)  argsort + CPU round-trip
      topk:     O(E log k)  GPU-only
      kthvalue: O(E)        GPU-only, optimal!
```

### Full GPU Pipeline for ExpertLoad

```
    FULL GPU-ONLY EXPERT LOAD DISPATCH
    ====================================

    token_selected_experts [N, K]     token_final_scales [N, K]
              |                                |
              v                                |
    +---------------------+                    |
    | bincount(flat, E)   |                    |
    | counts [E]          |                    |
    | (GPU, ~3 us)        |                    |
    +--------+------------+                    |
             |                                 |
             v                                 |
    +---------------------+                    |
    | kthvalue(counts,     |                    |
    |   E-k+1)            |                    |
    | threshold [scalar]  |                    |
    | (GPU, ~2 us)        |                    |
    +--------+------------+                    |
             |                                 |
             v                                 |
    +---------------------+                    |
    | expert_to_group =   |                    |
    |  (counts>=thr).long |                    |
    | [E] tensor          |                    |
    | (GPU, ~1 us)        |                    |
    +--------+------------+                    |
             |                                 |
             v                                 |
    +---------------------+                    |
    | Single-pass dispatch| <------------------+
    | kernel (Solution C) |
    | (GPU, ~10-20 us)    |
    +--------+------------+
             |
             v
      per-group (tok_idx, experts, scales)
      all on GPU, zero CPU sync

    TOTAL: ~16-26 us   vs.  current ~200-400 us
    SPEEDUP: 8-25x
```

---

## 6. CUDA Graph Compatibility

### The Constraint

CUDA graphs record a fixed sequence of GPU operations. During replay, no Python code re-executes. This means:

1. **Static policies (Random)**: `expert_to_group` is constant -- perfectly CUDA-graph safe. Compute once at graph capture time, reuse forever.

2. **Dynamic policies (ExpertLoad, ConfidenceThreshold)**: `expert_to_group` changes every forward pass based on routing decisions. Under CUDA graphs, the assignment from capture time is "baked in."

### Making Dynamic Policies Graph-Safe

```
    CUDA GRAPH STRATEGIES FOR DYNAMIC DISPATCH
    =============================================

    Strategy 1: Bake-and-Accept (Current Behavior)
    -----------------------------------------------
    - Assignment computed at capture time, frozen during replay
    - Simplest, but stale assignments degrade quality

    Strategy 2: Periodic Recapture
    -----------------------------------------------
    - Recapture graph every M steps (e.g., M=100)
    - Quality degrades between recaptures
    - M is a quality-vs-overhead tradeoff

    Strategy 3: Graph-Internal Dynamic Dispatch (Proposed)
    -----------------------------------------------
    - The entire dispatch pipeline (bincount -> threshold ->
      expert_to_group -> dispatch) runs as GPU ops within
      the captured graph
    - Assignment updates on EVERY replay automatically
    - REQUIRES: all ops are fixed-shape and deterministic

    Requirements for Strategy 3:
    +-------------------------------+----------+
    | Operation                     | Shape    |
    +-------------------------------+----------+
    | bincount(flat, minlength=E)   | [E]      |  FIXED
    | kthvalue(counts, k)           | scalar   |  FIXED
    | (counts >= thr).long()        | [E]      |  FIXED
    | expert_to_group[experts]      | [N, K]   |  FIXED*
    | slot_mask == g                | [N, K]   |  FIXED
    | any(dim=1)                    | [N]      |  FIXED
    | nonzero().squeeze(1)          | [?]      |  VARIABLE!
    +-------------------------------+----------+

    * N must be padded to max batch size for graph capture.

    The ONLY problem: nonzero() produces variable-length output.

    FIX: Replace nonzero() with a fixed-size gather:
      - Allocate output buffers at max size N
      - Use atomic counter for actual count
      - Downstream kernel reads counter to know valid length

    This is exactly what Solution C does!
    Solution C is inherently CUDA-graph compatible.
```

---

## 7. Recommended Implementation Plan

### Phase 0: Interface Refactor (_assign returns GPU tensor) -- 0.5 days

**Goal**: Change the `_assign` abstract method to return `expert_to_group: torch.Tensor` on GPU instead of `List[List[int]]`. This unblocks all subsequent phases.

```
    FILES TO MODIFY:
    ================

    policy/heter_dispatch.py:
      - Change _assign() return type: List[List[int]] -> torch.Tensor
      - Add _assign_gpu() method with GPU-native implementations
      - Modify dispatch() to use expert_to_group tensor directly
      - Remove _validate_assignment() (replaced by shape assertion)

    Estimated changes: ~40 lines modified
```

### Phase 1: Solution A (Python GPU Ops) -- 1-2 days

**Goal**: Eliminate all GPU-CPU sync with zero C++ changes.

```
    FILES TO MODIFY:
    ================

    policy/heter_dispatch.py:
      - RandomHeterDispatch: precompute expert_to_group once at __init__
      - ExpertLoadHeterDispatch: use torch.topk for 2-group fast path
      - ConfidenceThresholdHeterDispatch: use torch.argsort + scatter
      - _dispatch_by_assignment: use expert_to_group tensor directly
        (no more torch.tensor() from Python lists)

    Estimated changes: ~60 lines modified, 0 new files
```

**Validation**: Run existing `test_heter_moe.py` tests. Add microbenchmark comparing dispatch latency before/after.

### Phase 2: Solution C Kernel (Single-Pass Atomic) -- 3-5 days

**Goal**: Reduce dispatch to a single kernel launch.

```
    FILES TO CREATE/MODIFY:
    =======================

    NEW: cpp/tensorrt_llm/kernels/heterDispatchKernel.cu
         cpp/tensorrt_llm/kernels/heterDispatchKernel.h
         - Single-pass atomic dispatch kernel
         - Template on K (top_k) and G (num_groups)
         - Specializations for G=2 (most common)

    NEW: tensorrt_llm/thop/heterDispatchOp.cpp
         - torch custom op binding

    MODIFY: policy/heter_dispatch.py
         - Add torch.ops.trtllm.heter_dispatch() call path
         - Fallback to Solution A when custom op unavailable

    Estimated: ~300 lines C++, ~50 lines Python
```

### Phase 3: Solution D (Counting Sort for ExpertLoad) -- 5-7 days

**Goal**: Eliminate argsort for dynamic policies.

```
    FILES TO CREATE/MODIFY:
    =======================

    NEW: cpp/tensorrt_llm/kernels/heterExpertAssign.cu
         - Counting sort kernel for expert-to-group assignment
         - Histogram-based k-th element finding
         - Fused: bincount + threshold + assign in one kernel

    MODIFY: policy/heter_dispatch.py
         - ExpertLoadHeterDispatch uses new kernel
         - ConfidenceThresholdHeterDispatch uses same kernel
           with scatter_add-computed scores

    Estimated: ~200 lines C++, ~30 lines Python
```

### End-State Architecture

```
    FINAL PIPELINE (Solutions A+C+D Combined)
    ==========================================

    token_selected_experts [N, K]     token_final_scales [N, K]
              |                                |
              v                                v
    +--------------------------------------------------+
    | CUDA Kernel 1: heter_expert_assign()              |
    | (Solution D - counting sort, only for dynamic     |
    |  policies; skipped for RandomHeterDispatch)        |
    |                                                    |
    | Input:  experts[N,K], scales[N,K]                 |
    | Output: expert_to_group[E]                        |
    | Latency: ~5 us                                    |
    +------------------------+-------------------------+
                             |
                             v
    +--------------------------------------------------+
    | CUDA Kernel 2: heter_dispatch_fused()             |
    | (Solution C - single-pass atomic scatter)         |
    |                                                    |
    | Input:  experts[N,K], scales[N,K],                |
    |         expert_to_group[E]                        |
    | Output: per_group_{tok_idx, experts, scales}[G]   |
    | Latency: ~10-20 us                                |
    +------------------------+-------------------------+
                             |
                             v
    +--------------------------------------------------+
    | Python: run_moe() loop over G groups              |
    | (unchanged -- this loop launches fused_moe()      |
    |  kernels which dominate the runtime anyway)       |
    +--------------------------------------------------+

    Total dispatch latency: ~15-25 us
    Current dispatch latency: ~100-400 us
    Speedup: 4-16x on dispatch

    GPU-CPU syncs: ZERO
    CUDA graph compatible: YES (all ops fixed-shape)
```

---

## 8. Root Cause: Interface Design Forces GPU-CPU Round-Trip

The fundamental root cause is the `_assign` abstract method signature at `heter_dispatch.py:183-188`:

```python
@abc.abstractmethod
def _assign(self, token_selected_experts, token_final_scales) -> List[List[int]]:
    """Return group_assignments[i] = sorted list of expert IDs for group i."""
```

This forces a **Python list return type**, which means:
1. GPU scores must be transferred to CPU (`.tolist()`)
2. CPU partitions into Python lists
3. `_dispatch_by_assignment` converts lists back to GPU tensors

The downstream consumer (`_dispatch_by_assignment:232-238`) only needs `expert_to_group[e] = g` -- a 1D GPU tensor. The `List[List[int]]` intermediate is architecturally unnecessary.

### Proposed Interface Refactor

```python
# NEW: GPU-native assignment interface
def _assign_gpu(self, token_selected_experts, token_final_scales) -> torch.Tensor:
    """Return expert_to_group [num_experts] tensor on GPU. Zero CPU sync."""
    ...

# dispatch() calls _assign_gpu() directly, bypassing List[List[int]]
def dispatch(self, token_selected_experts, token_final_scales):
    expert_to_group = self._assign_gpu(token_selected_experts, token_final_scales)
    return self._dispatch_by_expert_to_group(expert_to_group, ...)
```

This is the **single most impactful change** -- once `_assign` returns a GPU tensor, all four solutions become straightforward to implement.

---

## 9. Warp-Level Optimization for Scatter Kernel

The research identified a critical optimization for the scatter phase using warp-level ballot primitives. Instead of one global atomic per token per group (O(N*G) atomics), use warp-cooperative compaction (O(N/32*G) atomics):

```
    WARP-LEVEL BALLOT OPTIMIZATION
    ================================

    Standard (naive):
      Each thread: atomicAdd(&group_counts[g], 1)
      -> N * G global atomics  (e.g., 4096 * 2 = 8192)

    Warp-cooperative:
      Step 1: __ballot_sync() to find which lanes need group g
              ballot = 0b...10110101  (lanes 0,2,4,5,7 need group g)

      Step 2: Lane 0 does ONE atomicAdd for the whole warp
              warp_count = __popc(ballot)  // = 5
              warp_base = atomicAdd(&group_counts[g], 5)

      Step 3: Each lane computes its position within the warp
              my_pos = warp_base + __popc(ballot & ((1 << lane_id) - 1))

      -> N/32 * G global atomics  (e.g., 128 * 2 = 256)
      -> 32x reduction in atomic contention!

    Visual (1 warp = 32 lanes):

    Lane:    0  1  2  3  4  5  6  7  ... 31
    Group 0: Y  N  Y  N  Y  Y  N  Y  ...
    Ballot:  1  0  1  0  1  1  0  1  ... = 0b...10110101

    atomicAdd(&count[0], popc(ballot)) -> warp_base = 42
    Lane 0: pos = 42 + popc(0b0)        = 42
    Lane 2: pos = 42 + popc(0b01)       = 43
    Lane 4: pos = 42 + popc(0b0101)     = 44
    Lane 5: pos = 42 + popc(0b10101)    = 45
    Lane 7: pos = 42 + popc(0b0110101)  = 46
```

### G=2 Template Specialization

For the dominant 2-group case, the inner loop over groups can be eliminated entirely via compile-time specialization:

```
    G=2 SPECIALIZATION (no group loop)
    ====================================

    template<int K=8>
    __global__ void heter_dispatch_2group(...)
    {
        // Each expert-slot is group 0 or group 1
        // Use a single pair of ballots -- no loop over G

        bool in_g0 = false, in_g1 = false;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            int g = e2g[experts[tid * K + k]];
            in_g0 |= (g == 0);
            in_g1 |= (g == 1);
        }

        // Two independent warp compactions (fully unrolled)
        if (__any_sync(0xFFFFFFFF, in_g0)) {
            uint32_t ballot0 = __ballot_sync(0xFFFFFFFF, in_g0);
            // ... compact write for group 0 ...
        }
        if (__any_sync(0xFFFFFFFF, in_g1)) {
            uint32_t ballot1 = __ballot_sync(0xFFFFFFFF, in_g1);
            // ... compact write for group 1 ...
        }
    }

    Benefit: No branching on G, fully unrolled, compiler
    can optimize register allocation for exactly 2 groups.
```

---

## 10. 2-Group Fast Path: torch.topk (Best Quick Win for ExpertLoad)

For the overwhelmingly common 2-group case, `torch.topk` is optimal:

```
    2-GROUP topk ASSIGNMENT (OPTIMAL)
    ===================================

    For ratio [0.8, 0.2] with E=128:
      k_high = round(0.2 * 128) = 26 experts in high-precision group

    expert_to_group = torch.zeros(E, device='cuda', dtype=torch.long)
    _, top_indices = torch.topk(counts, k=26)  # O(E), partial sort
    expert_to_group[top_indices] = 1

    Total: 2 kernel launches, ZERO GPU-CPU sync
    vs. current: argsort + .tolist() + CPU loop + torch.tensor()

    Complexity:
      topk:    O(E)        partial sort (selection algorithm)
      argsort: O(E log E)  full sort
      For E=128: topk ~2x faster than argsort
      For E=256: topk ~3x faster than argsort
```

### Why NOT torch.kthvalue?

Although `kthvalue` is theoretically O(E), it requires **tie-breaking** logic that adds 3-4 more kernel launches (comparison, cumsum, masked assignment). The total kernel count (5-6) exceeds `topk`'s (2), negating the algorithmic advantage at E=128-256.

Additionally, `torch.nonzero()` (needed for naive tie-breaking) has **dynamic output shape** and is NOT CUDA-graph capturable. A `cumsum`-based workaround exists but adds complexity.

---

## 11. Verdict: Optimal 2-Group ExpertLoad Exclusive Dispatch

**Scope**: Exactly G=2 precision groups, `ExpertLoadHeterDispatch` policy,
exclusive token-to-group assignment (each token goes to exactly one group).

### Constraints Evaluated

| Constraint | What It Means for Dispatch |
|------------|---------------------------|
| **Runtime efficient** | Minimize total kernel launches, avoid GPU-CPU sync |
| **torch.compile friendly** | No graph breaks — every op must have static shapes; `nonzero()` and `.tolist()` break the graph |
| **CUDA graph friendly** | All tensor shapes fixed at capture time; no dynamic-length outputs; replaying the graph with different routing must produce correct results |

### The Problem with the Current Code

The current `_dispatch_by_assignment` has **three** compile/graph-breaking operations:

```
    CURRENT DISPATCH: THREE BLOCKERS
    =================================

    1. .tolist()             at _assign_by_score:100
       -> cudaDeviceSynchronize (GPU-CPU sync)
       -> breaks torch.compile graph
       -> NOT capturable in CUDA graph

    2. torch.tensor(list)    at _dispatch_by_assignment:236-237
       -> CPU-to-GPU H2D transfer per group
       -> breaks torch.compile graph
       -> captured value is stale on CUDA graph replay

    3. nonzero()             at _dispatch_by_assignment:255
       -> dynamic output shape (varies per input)
       -> breaks torch.compile graph (guard failure)
       -> CUDA graph replay produces WRONG result
          (baked shape from capture != actual shape at replay)
```

### The Verdict

**Use `torch.topk` for assignment + `torch.where` sentinel masking for dispatch. Skip `nonzero()` entirely. Send all N tokens to both groups.**

```
    OPTIMAL 2-GROUP EXPERTLOAD PIPELINE
    ====================================

    token_selected_experts [N, K]     token_final_scales [N, K]
              |                                |
              v                                |
    +-------------------------+                |
    | bincount(flat, E)       |                |
    | counts [E]              |                |
    | 1 kernel, fixed shape   |                |
    +------------+------------+                |
                 |                             |
                 v                             |
    +-------------------------+                |
    | topk(counts, k_high)    |                |
    | expert_to_group [E]     |                |
    | 2 kernels, fixed shape  |                |
    +------------+------------+                |
                 |                             |
                 v                             |
    +-------------------------+                |
    | slot_groups = e2g[exp]  | <--------------+
    | [N, K], fixed shape     |
    | 1 kernel (gather)       |
    +------------+------------+
                 |
                 v
    +----------------------------------------------+
    | FOR gidx IN {0, 1}:              (unrolled)  |
    |                                              |
    |   in_group = (slot_groups == gidx)  [N, K]   |
    |   experts_g = where(in_group, exp, E) [N, K] |
    |   scales_g  = where(in_group, scl, 0) [N, K] |
    |                                              |
    |   result = fused_moe(x, experts_g, scales_g, |
    |                      weights[gidx], ...)     |
    |   accumulated += result                      |
    +----------------------------------------------+
                 |
                 v
           accumulated [N, H]

    GPU-CPU syncs:        ZERO
    Dynamic-shape ops:    ZERO
    Graph breaks:         ZERO
    CUDA graph safe:      YES (all shapes fixed, all values recomputed)
    torch.compile safe:   YES (no guards, no .tolist, no nonzero)
```

### Why This Works

**Key insight: with exclusive assignment, ALL K expert slots for a
given token belong to the same group.** When token *i* is in group 0
and we prepare group 1's inputs:

```
    Token i (belongs to group 0), K=8 experts:

    Group 0 call:                    Group 1 call:
    experts_0[i] = [3, 7, 12, ...]   experts_1[i] = [E, E, E, ...]  (all sentinel)
    scales_0[i]  = [0.3, 0.1, ...]   scales_1[i]  = [0, 0, 0, ...]  (all zero)
           |                                |
           v                                v
    CUTLASS: real GEMMs              CUTLASS: skips token entirely
    for 8 expert slots               (all K slots are sentinel,
                                      no weight loads, no compute)
```

The CUTLASS `fused_moe` kernel already handles sentinel expert IDs
(`expert_id == num_experts`) — this is the existing mechanism used for
non-group expert slots within multi-group tokens (lines 262-268 of
`heter_dispatch.py`). With exclusive assignment, sentinel tokens have
ALL K slots as sentinel, so the kernel skips them at minimal cost
(reads K int32 expert IDs, finds all out-of-range, moves on).

### Overhead of Sending All N Tokens to Both Groups

```
    OVERHEAD ANALYSIS
    =================

    For N=4096, K=8, E=128, ratio=[0.8, 0.2]:
      Group 0: 4096 tokens, ~3277 real + ~819 sentinel (20% waste)
      Group 1: 4096 tokens, ~819 real + ~3277 sentinel (80% waste)

    Per-sentinel-token cost in CUTLASS:
      Read: K * sizeof(int32) = 8 * 4 = 32 bytes (expert IDs)
      Branch: K comparisons (all fail → skip)
      No weight loads, no GEMM, no output write
      Cost: ~0.1 us per wasted token

    Total sentinel overhead:
      Group 0: 819 * 0.1 = ~82 us
      Group 1: 3277 * 0.1 = ~328 us
      Total: ~410 us

    vs. cost SAVED by eliminating nonzero() + tolist():
      nonzero():   ~50-100 us (kernel + shape materialization)
      .tolist():   ~50-200 us (cudaDeviceSynchronize)
      clone() x4:  ~20-40 us
      Total saved:  ~120-340 us

    NET: Roughly break-even on raw latency for eager mode.
```

**But the real win is not latency — it's compilability:**

```
    THE REAL PAYOFF
    ===============

    Without this change:
      torch.compile: graph breaks at .tolist() and nonzero()
                     → falls back to eager mode
                     → ZERO compile benefit

    With this change:
      torch.compile: entire dispatch is a single fused graph
                     → kernel fusion across where/comparison ops
                     → eliminated intermediate tensors
                     → 3-5x speedup on dispatch portion

    Without this change:
      CUDA graph: assignment baked at capture, nonzero shapes frozen
                  → INCORRECT results on replay when routing changes
                  → must disable CUDA graphs for heter MoE

    With this change:
      CUDA graph: all ops recompute correctly on replay
                  → expert_to_group recomputed from live bincount
                  → where() recomputed from live expert_to_group
                  → full CUDA graph support, no staleness
```

### Why `torch.topk` Beats Every Alternative (G=2)

```
    ASSIGNMENT METHOD COMPARISON (G=2, E=128)
    ==========================================

    Method              Kernels  GPU-CPU  Compile  Graph   Notes
                                 Sync     Safe     Safe
    ---------------------------------------------------------------
    argsort + .tolist()  1+sync   YES      NO       NO     current
    argsort + scatter    3        NO       YES      YES    Solution A
    topk + scatter       2        NO       YES      YES    WINNER
    kthvalue + cumsum    5-6      NO       YES      YES    too many kernels
    counting sort (C++)  1        NO       YES      YES    overkill for E=128
    ---------------------------------------------------------------

    topk wins because:
    1. Fewest kernels (2) among all sync-free options
    2. O(E) partial sort — only finds top-k, no full ordering
    3. Fixed output shape: always returns exactly k indices
    4. Natively supported by torch.compile (no custom ops)
    5. Natively captured by CUDA graph (deterministic shapes)

    For E=128, k=26 (20% high-precision):
      topk:    ~3 us (selection algorithm, single pass over E)
      argsort: ~5 us (full radix sort of E elements)
      Δ = 2 us — small absolute, but topk also has fewer kernels
```

### Why `torch.where` Beats `nonzero` + `clone` (Dispatch)

```
    DISPATCH METHOD COMPARISON
    ===========================

    Method                Kernels  Dynamic  Compile  Graph  Clones
                          per grp  Shape    Safe     Safe
    ---------------------------------------------------------------
    nonzero+clone+fill    4-5      YES      NO       NO     2
    where (sentinel)      2-3      NO       YES      YES    0
    custom CUDA kernel    1        NO       YES      YES    0
    ---------------------------------------------------------------

    where wins over nonzero because:
    1. Fixed output shape [N, K] — same as input
    2. Zero clones — where() produces output directly
    3. Element-wise — torch.compile fuses with adjacent ops
    4. No CPU involvement — pure GPU compute

    where vs custom kernel:
    - Custom kernel (Solution C) is 1 launch vs 2-3 for where
    - BUT custom kernel needs C++ code, build system changes,
      torch.compile custom op registration, testing
    - where is pure PyTorch — zero implementation risk
    - torch.compile can fuse the 2-3 where ops into 1 kernel anyway!
      (comparison + where + where → single fused kernel)

    AFTER torch.compile fusion:
      where approach ≈ custom kernel (both ~1 effective kernel launch)
      → custom kernel provides NO incremental benefit
```

### Recommended Implementation (Pseudocode)

```python
class ExpertLoadHeterDispatch_Optimized:
    """2-group ExpertLoad: topk + where. Zero sync, fully compilable."""

    def dispatch(self, token_selected_experts, token_final_scales):
        E = self._num_experts
        device = token_selected_experts.device

        # --- Phase 1: Assignment via topk (2 kernels) ---
        flat = token_selected_experts.reshape(-1).long()
        counts = torch.bincount(flat, minlength=E).float()

        k_high = self._group_sizes[1]  # high-precision group size
        expert_to_group = torch.zeros(E, dtype=torch.long, device=device)
        _, top_idx = torch.topk(counts, k_high)
        expert_to_group[top_idx] = 1

        # --- Phase 2: Dispatch via where (no nonzero, no clone) ---
        slot_groups = expert_to_group[token_selected_experts.long()]  # [N, K]

        dispatches = []
        for gidx in range(2):
            in_group = (slot_groups == gidx)           # [N, K]
            experts_g = torch.where(in_group,
                                    token_selected_experts,
                                    E)                 # [N, K]
            scales_g = torch.where(in_group,
                                   token_final_scales,
                                   0.0)                # [N, K]

            # tok_idx = None signals "all tokens" to run_moe
            # (or pass arange(N) if interface requires explicit indices)
            dispatches.append((None, experts_g, scales_g))

        return dispatches
```

### Kernel Launch Count Comparison

```
    TOTAL KERNEL LAUNCHES PER DISPATCH CALL
    =========================================

    Current (eager, with sync):
      bincount                    1
      argsort                     1
      .tolist()                   --- (sync, not a kernel)
      torch.tensor() x2          2   (H2D copies)
      expert_to_group[exp]        1
      (== gidx) x2               2
      .any(dim=1) x2             2
      nonzero() x2               2   (+ shape sync each!)
      clone() x4                 4
      masked_fill x4             4
      TOTAL:                    ~19 kernels + 3 syncs

    Proposed (topk + where):
      bincount                    1
      topk                        1
      zeros                       1
      scatter (top_idx)           1
      expert_to_group[exp]        1
      (== 0), where, where        3
      (== 1), where, where        3
      TOTAL:                    ~11 kernels + 0 syncs

    After torch.compile fusion:
      bincount                    1   (not fusible)
      topk                        1   (not fusible)
      fused(zeros+scatter+gather) 1   (pointwise fusion)
      fused(cmp+where+where) x2  2   (pointwise fusion)
      TOTAL:                     ~5 kernels + 0 syncs

    Speedup vs current: ~4x fewer kernels, ZERO syncs
```

### run_moe Simplification

With all-N dispatch (no per-token filtering), `run_moe` simplifies:

```
    CURRENT run_moe (per group):             PROPOSED run_moe (per group):
    ============================             =============================

    tok_idx = dispatch[g][0]                 # tok_idx is None (all tokens)
    qi = quantize(x[tok_idx], ...)           qi = quantize(x, ...)
    result = fused_moe(qi.x, ...)            result = fused_moe(qi.x, ...)
    accumulated[tok_idx] += result           accumulated += result

    - Indexed gather x[tok_idx]              - No gather (use x directly)
    - Indexed scatter accumulated[tok_idx]   - No scatter (direct add)
    - 2 extra kernels per group              - 0 extra kernels per group
```

This saves 4 additional kernels (2 gathers + 2 scatters across 2 groups),
further widening the advantage.

### Final Verdict Summary

```
    +------------------+-------------------+-------------------+
    |                  |  CURRENT          |  PROPOSED         |
    +------------------+-------------------+-------------------+
    | Assignment       | argsort+.tolist() | torch.topk        |
    | Dispatch         | nonzero+clone     | torch.where       |
    | GPU-CPU syncs    | 1-3 per call      | ZERO              |
    | torch.compile    | BROKEN (3 breaks) | FULLY SUPPORTED   |
    | CUDA graph       | BROKEN (dynamic)  | FULLY SUPPORTED   |
    | Kernels (eager)  | ~19 + syncs       | ~11               |
    | Kernels (compile)| N/A (can't compile)| ~5 (fused)       |
    | C++ required     | No                | No                |
    | Impl effort      | N/A               | ~40 lines changed |
    +------------------+-------------------+-------------------+

    VERDICT: topk + where is the OPTIMAL method for 2-group
    ExpertLoad exclusive dispatch. It is the ONLY approach that
    simultaneously satisfies all three constraints (runtime,
    torch.compile, CUDA graph) with ZERO custom C++ code.
```

---

## Appendix A: Why Not CUB Radix Sort?

CUB's `DeviceRadixSort` is the standard GPU sorting primitive, but it's **overkill** for this problem:

1. **E is tiny** (64-256). Radix sort's strength is large arrays. For E=128, the kernel launch overhead dominates the actual sort.

2. **We don't need a full sort**. We only need to partition E experts into G=2-3 groups. A counting sort or even `torch.topk` achieves this in O(E) vs O(E * num_passes).

3. **Multiple kernel launches**. CUB radix sort internally launches 4-8 kernels (one per radix pass). Our counting sort does it in 1 kernel.

4. **Temp memory allocation**. CUB requires temporary storage that must be allocated (another potential sync point if using `cudaMalloc` instead of a memory pool).

CUB radix sort would be appropriate if E were 10,000+, but at E=128-256, simpler approaches win decisively.

## Appendix B: Existing TRT-LLM Patterns to Reuse

The codebase already has similar GPU-side expert routing in:

- **`fusedBuildExpertMapsSortFirstToken`** (`moe_kernels.cu:331`): Single-block kernel that builds expert maps with counting sort using shared memory histograms. Uses `cub::BlockRadixSort` for within-expert token ordering. This is the closest existing pattern to our proposed Solution D.

- **`RoutingKernel.cuh`**: Block-scale MoE routing with warp-cooperative softmax and expert selection. Uses CUB block-level primitives and cooperative groups. Good reference for warp-level dispatch patterns.

- **`topkLastDim.cu`**: AIR TopK implementation using radix-based bucket filtering. Demonstrates histogram-based partitioning in CUDA -- exactly the approach we propose for counting sort expert assignment.

## Appendix C: Memory Budget

```
    DISPATCH BUFFER MEMORY (pre-allocated, reused)
    ================================================

    For N_max=8192, K=8, G=3:

    expert_to_group:     E * 4B           =     512 B  (E=128)
    group_counts:        G * 4B           =      12 B
    out_tok_idx:         G * N_max * 4B   =   96 KB
    out_experts:         G * N_max * K * 4B = 768 KB
    out_scales:          G * N_max * K * 4B = 768 KB

    TOTAL: ~1.6 MB

    This is <0.01% of typical GPU memory (80 GB).
    Pre-allocate once, reuse across all MoE layers and forward passes.
```
