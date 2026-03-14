/***************************************************************************************************
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Fused SwiGLU activation + NVFP4 quantization for MoE GEMM1 epilogue.

  Architecture:
    Two-phase reduce using shared memory staging:
    Phase 1: Compute SwiGLU from interleaved up/gate accumulator pairs → bf16 to smem
    Phase 2: Read contiguous bf16 from smem → NVFP4 quantize → write packed FP4 + SF to gmem

  Weight layout: interleaved [up0,gate0,up1,gate1,...] so adjacent columns still represent one SwiGLU pair.
  The visitor preserves the historical even/odd local variable names, so it still computes the unfused
  `silu(gate) * up` math once that physical layout is applied.

  EpilogueTile = 64x32 → After SwiGLU: 64 rows × 16 output columns per epitile.
  16 columns = exactly 1 NVFP4 SF block per row.
  Phase 2: 128 threads quantize (64 rows × 2 threads/row), remaining 128 idle.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"

#include "cute/tensor.hpp"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_gemm_kernels.h"
#include "tensorrt_llm/kernels/quantization.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format off

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

namespace swiglu_detail {

CUTLASS_HOST_DEVICE constexpr int64_t get_activation_sf_offset(
    int64_t expert_id, int64_t token_offset, int64_t gemm_k) {
  using Input = ::tensorrt_llm::_v1::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
  int64_t constexpr min_n_dim_alignment = Input::MinNDimAlignmentNVFP4;
  int64_t constexpr min_k_dim_alignment = Input::MinKDimAlignmentNVFP4;
  int64_t constexpr block_size = Input::NVFP4BlockScaleVectorSize;
  int64_t padded_sf_start_offset = Input::alignToSfDim(
      token_offset + expert_id * (min_n_dim_alignment - 1), min_n_dim_alignment);
  int64_t padded_gemm_k = Input::alignToSfDim(gemm_k, min_k_dim_alignment);
  return padded_sf_start_offset * padded_gemm_k / block_size;
}

CUTLASS_DEVICE uint8_t* get_sf_out_ptr(uint8_t* sf_base, int64_t expert_idx, int64_t expert_row_base,
    int64_t token_in_expert, int half_n_global, int inter_size) {
  auto* act_sf_expert = sf_base + get_activation_sf_offset(expert_idx, expert_row_base, inter_size);
  return ::tensorrt_llm::_v1::kernels::cvt_quant_get_sf_out_offset<uint8_t, 2>(
      std::nullopt, static_cast<int>(token_in_expert), half_n_global / 8, std::nullopt, inter_size / 8,
      act_sf_expert, ::tensorrt_llm::_v1::QuantizationSFLayout::SWIZZLED);
}

} // namespace swiglu_detail

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// FusionOperation tag for SwiGLU activation with scatter store
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ElementOutput_,
  class ElementCompute_ = float,
  FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
>
struct SwiGLUActivationScatter : FusionOperation {
  using ElementOutput  = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementAux     = ElementOutput_;
  using GmemLayoutTagAux = cutlass::layout::RowMajor;
  static constexpr int AlignmentAux = 128 / cutlass::sizeof_bits<ElementOutput_>::value;
  static constexpr bool IsAuxOutSupported = false;
  static constexpr FloatRoundStyle RoundStyle = RoundStyle_;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// FusionCallbacks specialization for SwiGLU epilogue fusion
//
// Supports multiple R2S layouts via compile-time dispatch:
//   - FragmentSize=8, EpiTile=32x32: M32 path (N-consecutive elements)
//   - FragmentSize=4, EpiTile=64x32: M128 path (warp shuffle for gate/up pairing)
//
/////////////////////////////////////////////////////////////////////////////////////////////////

// ─── Inline NVFP4 quantization helpers (from quantization.cuh) ───────────────

namespace swiglu_detail {

struct QuantizedFp4Result {
    uint32_t packed;
    uint8_t sf_byte;
};

// Fast reciprocal
__device__ __forceinline__ float reciprocal_approximate_ftz(float a)
{
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// Convert 4 float2 → 8 packed e2m1 nibbles in uint32_t
__device__ __forceinline__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4])
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
          "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
    return val;
#else
    return 0;
#endif
}

// Quantize 8 bf16 values to packed FP4 uint32_t + write SF byte.
// Two threads cooperate: each holds 8 elements, together they share one 16-element SF block.
// SFout should be non-null only for the first thread of the pair (threadIdx % 2 == 0).
__device__ __forceinline__ uint32_t quantize_bf16x8_to_fp4(
    __nv_bfloat16 const* vals, float SFScaleVal, uint8_t* SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Load as bfloat16x2 pairs
    __nv_bfloat162 elts[4];
    elts[0] = {vals[0], vals[1]};
    elts[1] = {vals[2], vals[3]};
    elts[2] = {vals[4], vals[5]};
    elts[3] = {vals[6], vals[7]};

    // Compute local abs max
    auto localMax = __habs2(elts[0]);
    #pragma unroll
    for (int i = 1; i < 4; i++) {
        localMax = __hmax2(localMax, __habs2(elts[i]));
    }

    // Reduce across 2 threads for 16-element block
    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    float vecMax = __bfloat162float(__hmax(localMax.x, localMax.y));

    // Compute scale factor
    auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    __nv_fp8_e4m3 sfNarrow = __nv_fp8_e4m3(SFValue);
    uint8_t fp8SFVal = sfNarrow.__x;
    SFValue = static_cast<float>(sfNarrow);
    float outputScale = vecMax != 0
        ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
        : 0.0f;

    // Write SF (only first thread of pair)
    if (SFout) {
        *SFout = fp8SFVal;
    }

    // Scale and convert to e2m1
    float2 fp2Vals[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        fp2Vals[i] = __bfloat1622float2(elts[i]);
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
    }

    return fp32_vec_to_e2m1(fp2Vals);
#else
    return 0;
#endif
}

__device__ __forceinline__ QuantizedFp4Result quantize_bf16x8_to_fp4_result(
    __nv_bfloat16 const* vals, float SFScaleVal)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __nv_bfloat162 elts[4];
    elts[0] = {vals[0], vals[1]};
    elts[1] = {vals[2], vals[3]};
    elts[2] = {vals[4], vals[5]};
    elts[3] = {vals[6], vals[7]};

    auto localMax = __habs2(elts[0]);
    #pragma unroll
    for (int i = 1; i < 4; i++) {
        localMax = __hmax2(localMax, __habs2(elts[i]));
    }

    localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    float vecMax = __bfloat162float(__hmax(localMax.x, localMax.y));

    auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    __nv_fp8_e4m3 sfNarrow = __nv_fp8_e4m3(SFValue);
    uint8_t fp8SFVal = sfNarrow.__x;
    SFValue = static_cast<float>(sfNarrow);
    float outputScale = vecMax != 0
        ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal))
        : 0.0f;

    float2 fp2Vals[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        fp2Vals[i] = __bfloat1622float2(elts[i]);
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
    }

    return {fp32_vec_to_e2m1(fp2Vals), fp8SFVal};
#else
    return {0, 0};
#endif
}

} // namespace swiglu_detail

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class... Args
>
struct Sm90SwiGLUNvf4StoreNode {
  using Operation = fusion::SwiGLUActivationScatter<ElementOutput, ElementCompute, RoundStyle>;

  // Compile-time EpilogueTile dimensions
  static constexpr int kEpiTileM = cute::size<0>(EpilogueTile{});
  static constexpr int kEpiTileN = cute::size<1>(EpilogueTile{});
  static constexpr int kCtaTileM = cute::size<0>(CtaTileShapeMNK{});
  static constexpr int kNumThreads = NumEpilogueWarpGroups * 128;

  // Config detection
  static constexpr bool kIsM32Config =
      (FragmentSize == 4) && (kCtaTileM == 32) && (kEpiTileM == 32) && (kEpiTileN == 32);
  static constexpr bool kIsM64Config =
      (FragmentSize == 4) && (kCtaTileM == 64) && (kEpiTileM == 64) && (kEpiTileN == 32);
  static constexpr bool kIsM128Config =
      (FragmentSize == 4) && (kCtaTileM == 128) && (kEpiTileM == 64) && (kEpiTileN == 32);
  static constexpr bool kIsWide64x32Config = kIsM64Config || kIsM128Config;
  static constexpr bool kIsSupportedConfig = kIsM32Config || kIsWide64x32Config;
  static constexpr int kMaxThreadValues = 64;

  // After SwiGLU, N/2 = 16 columns per epitile (kEpiTileN / 2)
  static constexpr int kHalfN = kEpiTileN / 2;  // = 16
  static constexpr int kPackedFp4WordsPerRow = kHalfN / 8;

  struct SharedStorage {
    alignas(16) uint32_t packed_fp4[kEpiTileM * kPackedFp4WordsPerRow];
    alignas(16) uint8_t sf_bytes[kEpiTileM];
  };

  struct Arguments {
    ElementOutput** ptr_output{};          // Per-expert FP4 output pointers (packed uint32_t per 8 elements)
    int64_t const* stride_output{};        // Per-expert row stride for FP4 output (in elements, = inter_size)
    // FP4 quantization params
    uint8_t* fc2_act_sf_flat{};            // Scale factor buffer for NVFP4
    int64_t const* expert_first_token_offset{}; // Per-expert token offsets for SF addressing
    float const* global_sf_scale_ptr{};    // Device pointer to global scale factor (SFScaleVal)
    int64_t inter_size{0};                 // N/2 output dimension
    int64_t num_experts{0};                // Number of experts
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) { return true; }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
      CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  Params params;
  SharedStorage* shared_storage_ptr;

  CUTLASS_HOST_DEVICE
  Sm90SwiGLUNvf4StoreNode() {}

  CUTLASS_HOST_DEVICE
  Sm90SwiGLUNvf4StoreNode(Params const& p, SharedStorage const& shared_storage)
      : params(p), shared_storage_ptr(const_cast<SharedStorage*>(&shared_storage)) {}

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... PArgs>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<PArgs...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  // ─────────────────────────────────────────────────────────────────────────────
  //  ConsumerStoreCallbacks
  // ─────────────────────────────────────────────────────────────────────────────

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    // FP4 output
    ElementOutput* ptr_output_;
    int64_t stride_output_;  // Row stride in elements (= inter_size)

    // FP4 quantization
    uint8_t* fc2_act_sf_flat_;
    int64_t const* expert_first_token_offset_;
    float global_sf_scale_;
    int64_t inter_size_;
    int64_t num_experts_;

    // Tile coordinates
    int tile_m_;
    int tile_n_;
    int CTA_M_;
    int CTA_N_;
    int residue_m_;
    int residue_n_;

    // Thread index
    int thread_idx_;

    // Expert index for this CTA
    int expert_idx_;

    int m_in_epitile_[kMaxThreadValues]{};
    int n_in_epitile_[kMaxThreadValues]{};

    SharedStorage* shared_storage_ptr_;

    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        ElementOutput* ptr_output, int64_t stride_output,
        uint8_t* fc2_act_sf_flat, int64_t const* expert_first_token_offset,
        float global_sf_scale, int64_t inter_size, int64_t num_experts,
        int tile_m, int tile_n, int CTA_M, int CTA_N,
        int residue_m, int residue_n,
        int thread_idx, int expert_idx, SharedStorage* shared_storage_ptr)
        : ptr_output_(ptr_output), stride_output_(stride_output),
          fc2_act_sf_flat_(fc2_act_sf_flat),
          expert_first_token_offset_(expert_first_token_offset),
          global_sf_scale_(global_sf_scale),
          inter_size_(inter_size), num_experts_(num_experts),
          tile_m_(tile_m), tile_n_(tile_n),
          CTA_M_(CTA_M), CTA_N_(CTA_N),
          residue_m_(residue_m), residue_n_(residue_n),
          thread_idx_(thread_idx), expert_idx_(expert_idx), shared_storage_ptr_(shared_storage_ptr) {}

    // ─── visit() ──────────────────────────────────────────────────────────────
    template <typename ElementAccumulator, typename ElementInput, int FragSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragSize> const& frg_acc,
          int epi_v, int epi_m, int epi_n,
          Array<ElementInput, FragSize> const& frg_input) {
      using ConvertInput = NumericArrayConverter<ElementOutput, ElementInput, FragSize, RoundStyle>;
      auto result = ConvertInput{}(frg_input);
      return result;
    }

    template <class VTensor, class SyncFn>
    CUTLASS_DEVICE void
    reduce_partitioned_fp4(__nv_bfloat16* scratch, int epi_m, int epi_n, VTensor visit_results, SyncFn const& sync_fn) {
      float local_amax = 0.0f;
      int num_epi_v = cute::size(visit_results);
      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < num_epi_v; ++v) {
        auto frag = visit_results(v);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          float frag_f = cutlass::NumericConverter<float, ElementOutput>{}(frag[i]);
          local_amax = fmaxf(local_amax, fabsf(frag_f));
        }
      }
      sync_fn();

      // ═══ Phase 1: Compute SwiGLU, write bf16 to smem ═══

      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < num_epi_v; ++v) {
        auto frag = visit_results(v);

        CUTLASS_PRAGMA_UNROLL
        for (int pair = 0; pair < FragmentSize / 2; ++pair) {
          int gate_frag_i = pair * 2;
          int up_frag_i   = pair * 2 + 1;
          int gate_elem_idx = v * FragmentSize + gate_frag_i;

          int m_local = m_in_epitile_[gate_elem_idx];
          int n_local = n_in_epitile_[gate_elem_idx];  // Even column in full N space

          int m_cta = epi_m * kEpiTileM + m_local;
          if (m_cta >= residue_m_ || (n_local + 1) >= kEpiTileN) continue;

          float gate = cutlass::NumericConverter<float, ElementOutput>{}(frag[gate_frag_i]);
          float up   = cutlass::NumericConverter<float, ElementOutput>{}(frag[up_frag_i]);

          float silu_up = up / (1.0f + expf(-up));
          float result = silu_up * gate;

          // Write to smem: row = m_local, col = n_local / 2 (half-N)
          int half_n_local = n_local / 2;
          scratch[m_local * kHalfN + half_n_local] = __float2bfloat16(result);

        }
      }

      // ═══ Barrier: wait for all threads to finish Phase 1 ═══
      sync_fn();
      // ═══ Phase 2: Read contiguous bf16 from smem, quantize to FP4 ═══
      // 64 rows × 16 half-columns = 64 rows × (2 threads/row × 8 elements/thread) = 128 threads needed
      // With 256 epilogue threads, first 128 are active, rest idle.
      // CRITICAL: ALL 128 active threads must execute the __shfl_xor_sync inside
      // quantize_bf16x8_to_fp4, even for out-of-bounds rows. We move the bounds check
      // to guard only gmem writes AFTER the shuffle completes.
      constexpr int kActiveQuantThreads = kEpiTileM * kPackedFp4WordsPerRow;
      if (thread_idx_ < kActiveQuantThreads) {
        int row = thread_idx_ / kPackedFp4WordsPerRow;     // 0..63
        int packed_col = thread_idx_ % kPackedFp4WordsPerRow;   // 0 or 1

        int m_cta = epi_m * kEpiTileM + row;

        // ALL threads read from smem (out-of-bounds reads garbage — harmless)
        __nv_bfloat16 vals[8];
        int col_start = packed_col * 8;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; i++) {
          vals[i] = scratch[row * kHalfN + col_start + i];
        }

        // Compute the global half_n offset for this output
        int half_n_base = (tile_n_ * CTA_N_ + epi_n * kEpiTileN) / 2;  // Convert full N to half-N
        int half_n_global = half_n_base + col_start;

        // Determine if this thread is in-bounds for BOTH M and N dimensions
        bool m_in_bounds = (m_cta < residue_m_);
        bool n_in_bounds = (half_n_global + 8 <= inter_size_);
        bool in_bounds = m_in_bounds && n_in_bounds;

        uint8_t* sf_out = (in_bounds && packed_col == 0) ? fc2_act_sf_flat_ : nullptr;

        // ALL 128 threads call quantize — the __shfl_xor_sync inside requires
        // all 32 threads in each warp to participate. Out-of-bounds threads
        // compute garbage results that are simply not written to gmem.
        // SF write is guarded by SFout==nullptr inside quantize_bf16x8_to_fp4.
        auto quant = swiglu_detail::quantize_bf16x8_to_fp4_result(vals, global_sf_scale_);
        if (in_bounds) {
          shared_storage_ptr_->packed_fp4[row * kPackedFp4WordsPerRow + packed_col] = quant.packed;
        }
        if (sf_out) {
          shared_storage_ptr_->sf_bytes[row] = quant.sf_byte;
        }
      }
      sync_fn();
    }

    template <class VTensor, class SyncFn>
    CUTLASS_DEVICE void
    reduce_m32_fp4(__nv_bfloat16* scratch, int epi_m, int epi_n, VTensor visit_results, SyncFn const& sync_fn) {
      if (thread_idx_ < 128) {
        int group = thread_idx_ / 64;
        int warp_in_group = (thread_idx_ % 64) / 32;
        int lane = thread_idx_ % 32;
        int row0 = warp_in_group * 16 + lane / 4;
        int row1 = row0 + 8;
        int col_base = (lane % 4) * 2 + group * 8;
        int frag_base = group * 2 + warp_in_group;

        auto frag0 = visit_results(frag_base);
        auto frag1 = visit_results(frag_base + 4);

        auto store_pair = [&](int row_local, int n_local, auto const& frag, int gate_idx, int up_idx) {
          int m_cta = epi_m * kEpiTileM + row_local;
          if (m_cta >= residue_m_ || (n_local + 1) >= kEpiTileN) {
            return;
          }

          float gate = cutlass::NumericConverter<float, ElementOutput>{}(frag[gate_idx]);
          float up = cutlass::NumericConverter<float, ElementOutput>{}(frag[up_idx]);
          float silu_up = up / (1.0f + expf(-up));
          float result = silu_up * gate;

          scratch[row_local * kHalfN + n_local / 2] = __float2bfloat16(result);
        };

        store_pair(row0, col_base, frag0, 0, 1);
        store_pair(row1, col_base, frag0, 2, 3);
        store_pair(row0, col_base + 16, frag1, 0, 1);
        store_pair(row1, col_base + 16, frag1, 2, 3);
      }

      sync_fn();

      constexpr int kActiveQuantThreads = kEpiTileM * kPackedFp4WordsPerRow;
      if (thread_idx_ < kActiveQuantThreads) {
        int row = thread_idx_ / kPackedFp4WordsPerRow;
        int packed_col = thread_idx_ % kPackedFp4WordsPerRow;

        int m_cta = epi_m * kEpiTileM + row;

        __nv_bfloat16 vals[8];
        int col_start = packed_col * 8;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 8; i++) {
          vals[i] = scratch[row * kHalfN + col_start + i];
        }

        int half_n_base = (tile_n_ * CTA_N_ + epi_n * kEpiTileN) / 2;
        int half_n_global = half_n_base + col_start;

        bool m_in_bounds = (m_cta < residue_m_);
        bool n_in_bounds = (half_n_global + 8 <= inter_size_);
        bool in_bounds = m_in_bounds && n_in_bounds;

        uint8_t* sf_out = (in_bounds && packed_col == 0) ? fc2_act_sf_flat_ : nullptr;

        auto quant = swiglu_detail::quantize_bf16x8_to_fp4_result(vals, global_sf_scale_);
        if (in_bounds) {
          shared_storage_ptr_->packed_fp4[row * kPackedFp4WordsPerRow + packed_col] = quant.packed;
        }
        if (sf_out) {
          shared_storage_ptr_->sf_bytes[row] = quant.sf_byte;
        }
      }
      sync_fn();
    }

    // ─── reduce() ─────────────────────────────────────────────────────────────
    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(STensor&& smem_buffer, SyncFn const& sync_fn,
           int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      auto* scratch = reinterpret_cast<__nv_bfloat16*>(cute::raw_pointer_cast(smem_buffer.data()));
      if constexpr (kIsWide64x32Config) {
        reduce_partitioned_fp4(scratch, epi_m, epi_n, visit_results, sync_fn);
      } else if constexpr (kIsM32Config) {
        reduce_m32_fp4(scratch, epi_m, epi_n, visit_results, sync_fn);
      } else {
        // Unsupported config — just call sync to avoid deadlock
        sync_fn();
      }
    }
    // ─── postreduce() ─────────────────────────────────────────────────────────
    CUTLASS_DEVICE void
    postreduce(int, int, int, bool) {}

    CUTLASS_DEVICE void end_loop(int epi_m, int epi_n) {
      if (ptr_output_ == nullptr || expert_first_token_offset_ == nullptr) {
        return;
      }

      if constexpr (kIsSupportedConfig) {
        constexpr int kActiveStoreThreads = kEpiTileM * kPackedFp4WordsPerRow;

        if (thread_idx_ >= kActiveStoreThreads) {
          return;
        }

        int row = thread_idx_ / kPackedFp4WordsPerRow;
        int packed_col = thread_idx_ % kPackedFp4WordsPerRow;
        int m_cta = epi_m * kEpiTileM + row;

        int half_n_base = (tile_n_ * CTA_N_ + epi_n * kEpiTileN) / 2;
        int half_n_global = half_n_base + packed_col * 8;
        bool row_in_bounds = m_cta < residue_m_;
        bool col_in_bounds = (half_n_global + 8) <= inter_size_;
        if (!(row_in_bounds && col_in_bounds)) {
          return;
        }

        int64_t expert_row_base = expert_first_token_offset_[expert_idx_];
        int64_t token_in_expert = int64_t(tile_m_) * CTA_M_ + m_cta;
        int64_t packed_row = expert_row_base + token_in_expert;
        int64_t fp4_row_stride_u32 = inter_size_ / 8;
        int64_t fp4_index = packed_row * fp4_row_stride_u32 + half_n_global / 8;
        uint32_t packed_word = shared_storage_ptr_->packed_fp4[row * kPackedFp4WordsPerRow + packed_col];
        reinterpret_cast<uint32_t*>(ptr_output_)[fp4_index] = packed_word;
        if (packed_col == 0 && fc2_act_sf_flat_ != nullptr) {
          if (auto* sf_out = swiglu_detail::get_sf_out_ptr(fc2_act_sf_flat_, expert_idx_, expert_row_base,
                  token_in_expert, half_n_global, inter_size_)) {
            *sf_out = shared_storage_ptr_->sf_bytes[row];
          }
        }
        return;
      }
    }
  };

  // Build callbacks for this CTA tile
  template <bool ReferenceSrc, class... CstArgs>
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<CstArgs...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    int expert_idx = l;
    ElementOutput* ptr_out = params.ptr_output ? params.ptr_output[expert_idx] : nullptr;
    int64_t stride_out     = params.stride_output ? params.stride_output[expert_idx] : 0;

    int CTA_M = get<0>(args.tile_shape_mnk);
    int CTA_N = get<1>(args.tile_shape_mnk);

    int residue_m = int(get<0>(args.residue_cD));
    int residue_n = int(get<1>(args.residue_cD));

    auto callbacks = ConsumerStoreCallbacks(
        ptr_out, stride_out,
        params.fc2_act_sf_flat, params.expert_first_token_offset,
        params.global_sf_scale_ptr ? *params.global_sf_scale_ptr : 1.0f, params.inter_size, params.num_experts,
        int(m), int(n), CTA_M, CTA_N,
        residue_m, residue_n,
        args.thread_idx, expert_idx, shared_storage_ptr);

    if constexpr (kIsSupportedConfig) {
      auto cta_shape_mn = make_shape(get<0>(CtaTileShapeMNK{}), get<1>(CtaTileShapeMNK{}));
      auto cta_identity = make_identity_tensor(cta_shape_mn);

      auto tRS_cD_mn = sm90_partition_for_epilogue<ReferenceSrc>(
          cta_identity, args.epi_tile, args.tiled_copy, args.thread_idx);

      auto tRS_epi0 = tRS_cD_mn(_, _, _, 0, 0);

      int num_thread_values = cute::size(tRS_epi0);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kMaxThreadValues; ++i) {
        if (i >= num_thread_values) {
          break;
        }
        callbacks.m_in_epitile_[i] = int(get<0>(tRS_epi0(i)));
        callbacks.n_in_epitile_[i] = int(get<1>(tRS_epi0(i)));
      }
    }

    return callbacks;
  }
};

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm100PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                              ReuseSmemC, DelayTmaStore>,
    fusion::ScaledAcc<ElementOutput, ElementCompute, ElementCompute, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    Args...> {
  struct Arguments {
    ElementOutput** ptr_output{};
    int64_t const* stride_output{};
    uint8_t* fc2_act_sf_flat{};
    int64_t const* expert_first_token_offset{};
    float const* global_sf_scale_ptr{};
    int64_t inter_size{0};
    int64_t num_experts{0};
  };

  using Params = Arguments;
  struct SharedStorage {};

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const&, Arguments const& args, void*) {
    return args;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const&) { return true; }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const&, Arguments const&, void*, cudaStream_t,
      CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE FusionCallbacks() {}
  CUTLASS_HOST_DEVICE FusionCallbacks(Params const&, SharedStorage const&) {}

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }

  template <class... PArgs>
  CUTLASS_DEVICE auto get_producer_load_callbacks(ProducerLoadArgs<PArgs...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    int thread_idx_;

    CUTLASS_DEVICE explicit ConsumerStoreCallbacks(int thread_idx) : thread_idx_(thread_idx) {}

    template <typename ElementAccumulator_, int FragSize>
    CUTLASS_DEVICE auto visit(Array<ElementAccumulator_, FragSize> const& frg_acc,
        int epi_v, int epi_m, int epi_n) {
      return frg_acc;
    }

    template <class STensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void reduce(STensor&&, SyncFn const& sync_fn,
        int, int, bool, VTensor) {
      sync_fn();
    }

    CUTLASS_DEVICE void postreduce(int, int, int, bool) {}
  };

  template <bool ReferenceSrc, class... CstArgs>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<CstArgs...> const& args) {
    return ConsumerStoreCallbacks(args.thread_idx);
  }
};

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class... Args
>
using Sm90SwiGLUNvf4StorePtrArray =
    Sm90EVT<
        Sm90SwiGLUNvf4StoreNode<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore,
            NumEpilogueWarpGroups, ElementOutput, ElementCompute, RoundStyle, CtaTileShapeMNK,
            EpilogueTile, Args...>,
        Sm90AccFetch>;

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class... Args
>
struct Sm90SwiGLUDirectCallbacks
    : Sm90SwiGLUNvf4StorePtrArray<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore,
          NumEpilogueWarpGroups, ElementOutput, ElementCompute, RoundStyle, CtaTileShapeMNK,
          EpilogueTile, Args...> {
  using Impl = Sm90SwiGLUNvf4StorePtrArray<StagesC, StagesD, FragmentSize, ReuseSmemC,
      DelayTmaStore, NumEpilogueWarpGroups, ElementOutput, ElementCompute, RoundStyle,
      CtaTileShapeMNK, EpilogueTile, Args...>;

  struct Arguments {
    ElementOutput** ptr_output{};
    int64_t const* stride_output{};
    uint8_t* fc2_act_sf_flat{};
    int64_t const* expert_first_token_offset{};
    float const* global_sf_scale_ptr{};
    int64_t inter_size{0};
    int64_t num_experts{0};
  };

  using Params = typename Impl::Params;
  using SharedStorage = typename Impl::SharedStorage;

  static CUTLASS_HOST_DEVICE constexpr typename Impl::Arguments
  to_impl_arguments(Arguments const& args) {
    return {{}, {args.ptr_output, args.stride_output, args.fc2_act_sf_flat,
        args.expert_first_token_offset, args.global_sf_scale_ptr, args.inter_size,
        args.num_experts}};
  }

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args,
      void* workspace) {
    return Impl::to_underlying_arguments(problem_shape, to_impl_arguments(args), workspace);
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return Impl::can_implement(problem_shape, to_impl_arguments(args));
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return Impl::get_workspace_size(problem_shape, to_impl_arguments(args));
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape, Arguments const& args,
      void* workspace, cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return Impl::initialize_workspace(problem_shape, to_impl_arguments(args), workspace, stream,
        cuda_adapter);
  }

  using Impl::Impl;
};

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                             ReuseSmemC, DelayTmaStore,
                                             NumEpilogueWarpGroups>,
    fusion::ScaledAcc<ElementOutput, ElementCompute, ElementCompute, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    Args...>
    : Sm90SwiGLUNvf4StorePtrArray<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore,
          NumEpilogueWarpGroups, ElementOutput, ElementCompute, RoundStyle, CtaTileShapeMNK,
          EpilogueTile, Args...> {
  using Impl = Sm90SwiGLUNvf4StorePtrArray<StagesC, StagesD, FragmentSize, ReuseSmemC,
      DelayTmaStore, NumEpilogueWarpGroups, ElementOutput, ElementCompute, RoundStyle,
      CtaTileShapeMNK, EpilogueTile, Args...>;

  struct Arguments {
    ElementOutput** ptr_output{};
    int64_t const* stride_output{};
    uint8_t* fc2_act_sf_flat{};
    int64_t const* expert_first_token_offset{};
    float const* global_sf_scale_ptr{};
    int64_t inter_size{0};
    int64_t num_experts{0};

    operator typename Impl::Arguments() const {
      return {{}, {ptr_output, stride_output, fc2_act_sf_flat, expert_first_token_offset,
          global_sf_scale_ptr, inter_size, num_experts}};
    }
  };

  using Impl::Impl;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Explicit SM120 specialization — same implementation, different dispatch policy.
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  int StagesC,
  int StagesD,
  int FragmentSize,
  bool ReuseSmemC,
  bool DelayTmaStore,
  int NumEpilogueWarpGroups,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class... Args
>
struct FusionCallbacks<
    epilogue::Sm120PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                             ReuseSmemC, DelayTmaStore,
                                             NumEpilogueWarpGroups>,
    fusion::ScaledAcc<ElementOutput, ElementCompute, ElementCompute, RoundStyle>,
    CtaTileShapeMNK,
    EpilogueTile,
    Args...
> : FusionCallbacks<
      epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                               ReuseSmemC, DelayTmaStore,
                                               NumEpilogueWarpGroups>,
      fusion::ScaledAcc<ElementOutput, ElementCompute, ElementCompute, RoundStyle>,
      CtaTileShapeMNK,
      EpilogueTile,
      Args...
  > {
  using FusionCallbacks<
      epilogue::Sm90PtrArrayTmaWarpSpecialized<StagesC, StagesD, FragmentSize,
                                               ReuseSmemC, DelayTmaStore,
                                               NumEpilogueWarpGroups>,
      fusion::ScaledAcc<ElementOutput, ElementCompute, ElementCompute, RoundStyle>,
      CtaTileShapeMNK,
      EpilogueTile,
      Args...>::FusionCallbacks;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::fusion

// clang-format on
