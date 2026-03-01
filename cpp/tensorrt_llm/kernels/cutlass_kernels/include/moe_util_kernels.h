/*
 * Copyright (c) 2020-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "./moe_gemm_kernels.h"
#include "cutlass/gemm/gemm.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"
#ifdef ENABLE_FP4
#include <cuda_fp4.h>
#endif
#include <NvInferRuntime.h>
#include <array>
#include <cuda_runtime_api.h>
#include <map>
#include <optional>
#include <random>
#include <utility>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace cutlass_kernels
{

// These kernels are used in moeUtilOp.cpp
int64_t computeNumTokensPerBlock(int64_t const num_tokens, int64_t const num_experts_per_node);

bool fusedBuildExpertMapsSortFirstToken(int const* token_selected_experts, int* unpermuted_token_selected_experts,
    int* permuted_source_token_ids, int64_t* expert_first_token_offset, int64_t const num_tokens,
    int const num_experts_per_node, int const experts_per_token, int const start_expert, int const end_expert,
    cudaStream_t stream);

void threeStepBuildExpertMapsSortFirstToken(int const* token_selected_experts, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int* blocked_expert_counts, int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens, int64_t const num_experts_per_node, int64_t const num_experts_per_token,
    int const start_expert_id, cudaStream_t stream);

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(InputActivationsType const* unpermuted_input,
    ExpandedActivationsType* permuted_output, float const* unpermuted_scales, float* permuted_scales,
    int const* permuted_row_to_unpermuted_row, int64_t const num_rows, int64_t const hidden_size, int const k,
    int const num_experts_per_node, QuantParams const& quant_params, bool use_per_expert_act_scale,
    int64_t* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, bool const swizzled_input_sf,
    void const* prequant_scales, cudaStream_t stream);

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* final_scales,
    int const* unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
    int const* token_selected_experts, int64_t const* expert_first_token_offset, int64_t const num_rows,
    int64_t const padded_cols, int64_t const unpadded_cols, int64_t const experts_per_token,
    int64_t const num_experts_per_node, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
    cudaStream_t stream);

// ========== NEW: Mixed-Precision MoE utility kernels ==========

/**
 * \brief NEW: Assign precision (bf16 vs fp4) to each expert based on token load.
 *
 * Computes tokens_per_expert from expert_first_token_offset, sorts by count
 * descending, assigns top num_high_precision_experts to bf16, rest to fp4.
 * Also builds per-group expert_first_token_offset arrays.
 */
void sortExpertsByTokenCount(
    int64_t const* expert_first_token_offset,  // [E+1] from existing sort
    int const num_experts_per_node,
    int const num_high_precision_experts,       // top-N experts get bf16
    int* expert_precision_assignment,           // output: [E] -> {0=fp4, 1=bf16}
    int* bf16_expert_indices,                   // output: sorted list of bf16 expert IDs
    int* fp4_expert_indices,                    // output: sorted list of fp4 expert IDs
    int64_t* bf16_expert_first_token_offset,    // output: [E+1] per-group offset
    int64_t* fp4_expert_first_token_offset,     // output: [E+1] per-group offset
    cudaStream_t stream);

/**
 * \brief NEW: Dual-buffer expand for mixed-precision MoE.
 *
 * Iterates over permuted tokens, writes bf16-expert rows to bf16 output buffer
 * and fp4-expert rows to fp4 output buffer (with quantization + scaling factors).
 */
template <class InputActivationsType>
void expandInputRowsMixedPrecisionKernelLauncher(
    InputActivationsType const* unpermuted_input,
    void* bf16_expanded_output,                  // bf16 group output buffer
    void* fp4_expanded_output,                   // fp4 group output buffer (quantized)
    int* bf16_permuted_row_to_unpermuted_row,
    int* fp4_permuted_row_to_unpermuted_row,
    float const* unpermuted_scales,
    float* bf16_permuted_scales,
    float* fp4_permuted_scales,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fp4_act_sf,  // fp4 act scaling factors
    int const* permuted_row_to_unpermuted_row,
    int const* expert_precision_assignment,       // [E]: 0=fp4, 1=bf16
    int64_t const* expert_first_token_offset,     // [E+1] global offsets
    int64_t const* bf16_expert_first_token_offset, // [E+1] per-group offsets for bf16
    int64_t const* fp4_expert_first_token_offset,  // [E+1] per-group offsets for fp4
    int64_t const num_rows,
    int64_t const hidden_size,
    int const experts_per_token,
    int const num_experts_per_node,
    float const* fc1_act_global_scale,            // fp4 quantization global scale
    bool use_per_expert_act_scale,
    cudaStream_t stream);

// ========== END Mixed-Precision MoE utility kernels ==========

} // namespace cutlass_kernels
} // namespace kernels

TRTLLM_NAMESPACE_END
