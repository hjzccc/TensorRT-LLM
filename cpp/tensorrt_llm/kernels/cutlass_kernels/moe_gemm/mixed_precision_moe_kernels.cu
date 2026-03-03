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

#include "tensorrt_llm/kernels/cutlass_kernels/include/mixed_precision_moe_kernels.h"

#include "tensorrt_llm/common/workspace.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

#if defined(ENABLE_FP4) && defined(ENABLE_BF16)

TRTLLM_NAMESPACE_BEGIN

namespace kernels::cutlass_kernels
{

// ================================================================================================
// Helper: Compute NVFP4 scaling factor offset for workspace sizing.
//
// Mirrors getOffsetActivationSF() from moe_kernels.cu specialized for NVFP4 scaling type.
// We need a local copy because that function is defined in a different translation unit.
// ================================================================================================
namespace
{

constexpr int64_t computeNvfp4SfOffset(int64_t num_experts, int64_t num_tokens, int64_t gemm_k)
{
    constexpr int64_t min_n = TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4;
    constexpr int64_t min_k = TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4;
    constexpr int64_t block_size = TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize;

    int64_t padded_sf_start
        = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(num_tokens + num_experts * (min_n - 1), min_n);
    int64_t padded_k = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k);
    return padded_sf_start * padded_k / block_size;
}

} // anonymous namespace

// ================================================================================================
// MixPrecisionMoeFCRunner implementation
// ================================================================================================

size_t MixPrecisionMoeFCRunner::getMixedPrecisionExtraWorkspaceSize(
    int64_t num_rows, int64_t hidden_size, int64_t inter_size, int num_experts, int experts_per_token) const
{
    using namespace tensorrt_llm::common;

    size_t const num_moe_inputs = static_cast<size_t>(experts_per_token) * num_rows;

    // Cap SF rows at num_rows * num_experts, matching the base class pattern.
    // When topk > num_experts (large EP), we don't need to allocate for the full tokens*topk.
    size_t const act_sf_rows = std::min(num_moe_inputs, static_cast<size_t>(num_rows * num_experts));

    size_t const sf_elem_size = sizeof(TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF);

    // Each buffer size (unaligned — calculateTotalWorkspaceSize handles alignment).
    size_t sizes[] = {
        // Dual expert-first-token offset arrays: [E+1] each
        static_cast<size_t>(num_experts + 1) * sizeof(int64_t), // bf16_expert_first_token_offset
        static_cast<size_t>(num_experts + 1) * sizeof(int64_t), // fp4_expert_first_token_offset

        // Per-group token counts (device scalars)
        sizeof(int64_t), // bf16_num_tokens
        sizeof(int64_t), // fp4_num_tokens

        // FP4 expanded input data — worst case all tokens routed to fp4.
        // Allocated at bf16 element size for the intermediate expand step.
        num_moe_inputs * hidden_size * sizeof(__nv_bfloat16), // mp_fp4_expanded_data

        // NVFP4 scaling factor for the expanded input [NK × H]
        static_cast<size_t>(computeNvfp4SfOffset(num_experts, act_sf_rows, hidden_size))
            * sf_elem_size, // mp_fp4_expand_sf

        // FP4 activation output — worst case, allocated at bf16 element size.
        num_moe_inputs * inter_size * sizeof(__nv_bfloat16), // mp_fp4_activation_output

        // NVFP4 scaling factor for the activation output [NK × I]
        static_cast<size_t>(computeNvfp4SfOffset(num_experts, act_sf_rows, inter_size))
            * sf_elem_size, // mp_fp4_activation_sf
    };

    return calculateTotalWorkspaceSize(sizes, sizeof(sizes) / sizeof(sizes[0]));
}

size_t MixPrecisionMoeFCRunner::getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size,
    int64_t const inter_size, int const num_experts, int const experts_per_token, ActivationType activation_type,
    MOEParallelismConfig parallelism_config, bool use_lora, bool use_deepseek_fp8_block_scale, bool min_latency_mode,
    bool use_awq)
{
    // Base workspace from CutlassMoeFCRunner<bf16, bf16>
    size_t base_size = Base::getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, experts_per_token,
        activation_type, parallelism_config, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);

    int const ep_size = parallelism_config.ep_size;
    TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0, "Number of experts must be a multiple of ep size");
    int const num_experts_per_node = num_experts / ep_size;

    // Extra mixed-precision buffers appended after the base workspace
    size_t extra_size
        = getMixedPrecisionExtraWorkspaceSize(num_rows, hidden_size, inter_size, num_experts_per_node, experts_per_token);

    return base_size + extra_size;
}

void MixPrecisionMoeFCRunner::configureMixedPrecisionWsPtrs(
    char* extra_ws_ptr, int64_t num_rows, int64_t hidden_size, int64_t inter_size, int num_experts, int experts_per_token)
{
    using namespace tensorrt_llm::common;

    size_t const num_moe_inputs = static_cast<size_t>(experts_per_token) * num_rows;
    size_t const act_sf_rows = std::min(num_moe_inputs, static_cast<size_t>(num_rows * num_experts));
    size_t const sf_elem_size = sizeof(TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF);

    auto* base = reinterpret_cast<int8_t*>(extra_ws_ptr);
    uintptr_t offset = 0;

    // Layout order matches getMixedPrecisionExtraWorkspaceSize exactly.
    bf16_expert_first_token_offset_ = reinterpret_cast<int64_t*>(
        nextWorkspacePtr(base, offset, static_cast<size_t>(num_experts + 1) * sizeof(int64_t)));

    fp4_expert_first_token_offset_ = reinterpret_cast<int64_t*>(
        nextWorkspacePtr(base, offset, static_cast<size_t>(num_experts + 1) * sizeof(int64_t)));

    bf16_num_tokens_
        = reinterpret_cast<int64_t*>(nextWorkspacePtr(base, offset, sizeof(int64_t)));

    fp4_num_tokens_
        = reinterpret_cast<int64_t*>(nextWorkspacePtr(base, offset, sizeof(int64_t)));

    mp_fp4_expanded_data_ = reinterpret_cast<void*>(
        nextWorkspacePtr(base, offset, num_moe_inputs * hidden_size * sizeof(__nv_bfloat16)));

    mp_fp4_expand_sf_ = reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(nextWorkspacePtr(
        base, offset, static_cast<size_t>(computeNvfp4SfOffset(num_experts, act_sf_rows, hidden_size)) * sf_elem_size));

    mp_fp4_activation_output_ = reinterpret_cast<void*>(
        nextWorkspacePtr(base, offset, num_moe_inputs * inter_size * sizeof(__nv_bfloat16)));

    mp_fp4_activation_sf_ = reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(nextWorkspacePtr(
        base, offset, static_cast<size_t>(computeNvfp4SfOffset(num_experts, act_sf_rows, inter_size)) * sf_elem_size));
}

void MixPrecisionMoeFCRunner::mixedSort(
    int const* expert_precision_assignment, int num_experts, int64_t expanded_num_rows, cudaStream_t stream)
{
    // TODO: Partition expert_first_token_offset_ into dual offset arrays
    // based on expert_precision_assignment. Set bf16/fp4 token counts.
    TLLM_THROW("MixPrecisionMoeFCRunner::mixedSort not implemented");
}

void MixPrecisionMoeFCRunner::mixedExpandInputRows(void const* input_activations, void const* input_sf,
    int64_t num_rows, int64_t hidden_size, int num_experts, int experts_per_token, cudaStream_t stream)
{
    // TODO: Expand input rows into dual buffers:
    //   bf16 group → Base::permuted_data_
    //   fp4  group → mp_fp4_expanded_data_ + mp_fp4_expand_sf_
    TLLM_THROW("MixPrecisionMoeFCRunner::mixedExpandInputRows not implemented");
}

void MixPrecisionMoeFCRunner::mixedDoActivation(
    ActivationParams activation_type, int64_t inter_size, int64_t hidden_size, int num_experts, cudaStream_t stream)
{
    // TODO: Apply activation on glu_inter_result_ for both groups:
    //   bf16 group: activation → fc1_result_
    //   fp4  group: activation + quantize → mp_fp4_activation_output_ + mp_fp4_activation_sf_
    TLLM_THROW("MixPrecisionMoeFCRunner::mixedDoActivation not implemented");
}

// ================================================================================================
// MixPrecisionMoEFCWrapper implementation
// ================================================================================================

MixPrecisionMoEFCWrapper::MixPrecisionMoEFCWrapper()
{
    // Both runners (bf16_runner_ and fp4_runner_) are default-constructed.
    //
    // Required configuration before calling runMoe():
    //   1. setTactic()                      — sets GEMM configs on both runners
    //   2. setExpertPrecisionAssignment()    — sets per-expert precision map
    //   3. setFp4Weights()                   — sets fp4 weight pointers and quant params
}

size_t MixPrecisionMoEFCWrapper::getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size,
    int64_t const inter_size, int const num_experts, int const experts_per_token, ActivationType activation_type,
    MOEParallelismConfig parallelism_config, bool use_lora, bool use_deepseek_fp8_block_scale, bool min_latency_mode,
    bool use_awq)
{
    // bf16 runner workspace includes the mixed-precision extra buffers.
    size_t bf16_ws = bf16_runner_.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, experts_per_token,
        activation_type, parallelism_config, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);

    // fp4 runner workspace (used for the all-fp4 delegation path).
    size_t fp4_ws = fp4_runner_.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, experts_per_token,
        activation_type, parallelism_config, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);

    // Use the max because only one runner operates at a time in the bare-minimum paths
    // (all-bf16 or all-fp4). For the full mixed-precision pipeline, the bf16 runner workspace
    // serves as the master layout and the extra buffers are appended.
    return std::max(bf16_ws, fp4_ws);
}

void MixPrecisionMoEFCWrapper::setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
    std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config)
{
    bf16_runner_.setTactic(gemm1_config, gemm2_config);
    fp4_runner_.setTactic(gemm1_config, gemm2_config);
}

std::vector<cutlass_extensions::CutlassGemmConfig> MixPrecisionMoEFCWrapper::getTactics(MoeGemmId gemm_id)
{
    // Delegate to bf16 runner. FP4 runner may need separate tactic selection later.
    return bf16_runner_.getTactics(gemm_id);
}

void MixPrecisionMoEFCWrapper::runMoe(void const* input_activations, void const* input_sf,
    bool const swizzled_input_sf, int const* token_selected_experts, float const* token_final_scales,
    void const* fc1_expert_weights, void const* fc1_expert_biases, ActivationParams fc1_activation_type,
    void const* fc2_expert_weights, void const* fc2_expert_biases, QuantParams quant_params,
    int64_t const num_rows, int64_t const num_valid_rows, int64_t const hidden_size,
    int64_t const unpadded_hidden_size, int64_t const inter_size, int const num_experts,
    int const experts_per_token, char* workspace_ptr, void* final_output,
    int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config,
    bool const enable_alltoall, bool use_lora, LoraParams& lora_params,
    bool use_deepseek_fp8_block_scale, bool min_latency_mode, MoeMinLatencyParams& min_latency_params,
    cudaStream_t stream)
{
    // ---- Precondition checks ----
    TLLM_CHECK_WITH_INFO(expert_precision_assignment_ != nullptr,
        "Expert precision assignment not set. Call setExpertPrecisionAssignment() before runMoe().");
    TLLM_CHECK_WITH_INFO(!min_latency_mode, "Mixed-precision MoE does not support min-latency mode");
    TLLM_CHECK_WITH_INFO(!use_lora, "Mixed-precision MoE does not support LoRA");
    TLLM_CHECK_WITH_INFO(
        !use_deepseek_fp8_block_scale, "Mixed-precision MoE does not support DeepSeek FP8 block scale");

    int const num_experts_per_node = num_experts / parallelism_config.ep_size;
    bool const all_bf16 = (num_high_precision_experts_ == num_experts_per_node);
    bool const all_fp4 = (num_high_precision_experts_ == 0);

    if (all_bf16)
    {
        // All experts are bf16 — delegate entirely to the bf16 runner.
        // The bf16 runner handles the full pipeline: configureWsPtrs, sort, expand, gemm1, gemm2.
        bf16_runner_.runMoe(input_activations, input_sf, swizzled_input_sf, token_selected_experts, token_final_scales,
            fc1_expert_weights, fc1_expert_biases, fc1_activation_type, fc2_expert_weights, fc2_expert_biases,
            quant_params, num_rows, num_valid_rows, hidden_size, unpadded_hidden_size, inter_size, num_experts,
            experts_per_token, workspace_ptr, final_output, unpermuted_row_to_permuted_row, parallelism_config,
            enable_alltoall, use_lora, lora_params, use_deepseek_fp8_block_scale, min_latency_mode, min_latency_params,
            stream);
    }
    else if (all_fp4)
    {
        // All experts are fp4 — delegate entirely to the fp4 runner.
        // Use fp4 weights and quant params set via setFp4Weights().
        // Biases are shared (bf16) — the fp4 runner's ScaleBiasType is bf16 since backbone is bf16.
        TLLM_CHECK_WITH_INFO(
            fp4_fc1_expert_weights_ != nullptr, "FP4 FC1 weights not set. Call setFp4Weights() before runMoe().");
        TLLM_CHECK_WITH_INFO(
            fp4_fc2_expert_weights_ != nullptr, "FP4 FC2 weights not set. Call setFp4Weights() before runMoe().");

        fp4_runner_.runMoe(input_activations, input_sf, swizzled_input_sf, token_selected_experts, token_final_scales,
            fp4_fc1_expert_weights_, fc1_expert_biases, fc1_activation_type, fp4_fc2_expert_weights_, fc2_expert_biases,
            fp4_quant_params_, num_rows, num_valid_rows, hidden_size, unpadded_hidden_size, inter_size, num_experts,
            experts_per_token, workspace_ptr, final_output, unpermuted_row_to_permuted_row, parallelism_config,
            enable_alltoall, use_lora, lora_params, use_deepseek_fp8_block_scale, min_latency_mode, min_latency_params,
            stream);
    }
    else
    {
        // Full mixed-precision pipeline: sort → expand → 2×gemm1 → activation → 2×gemm2.
        // Not yet implemented — requires mixedSort, mixedExpandInputRows, mixedGemm1,
        // mixedDoActivation, mixedGemm2.
        TLLM_THROW("Full mixed-precision MoE pipeline not yet implemented. "
                    "Currently only all-bf16 or all-fp4 expert assignment is supported.");
    }
}

std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
MixPrecisionMoEFCWrapper::computeStridesTmaWarpSpecializedDispatch(
    int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n,
    int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k, int const num_experts_per_node, void const* gemm1_in,
    void const* gemm2_in, void const* weights1, void const* weights2, float const* alpha_scale_flat1,
    float const* alpha_scale_flat2, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params, void const* bias1,
    void const* bias2, void* gemm1_output, void* gemm2_output, float const* router_scales,
    int const* permuted_row_to_unpermuted_row, cudaStream_t stream)
{
    // Delegate to bf16 runner
    return bf16_runner_.computeStridesTmaWarpSpecializedDispatch(expert_first_token_offset, layout_info1, layout_info2,
        num_tokens, expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts_per_node, gemm1_in, gemm2_in,
        weights1, weights2, alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2, quant_params, bias1,
        bias2, gemm1_output, gemm2_output, router_scales, permuted_row_to_unpermuted_row, stream);
}

std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
MixPrecisionMoEFCWrapper::computeStridesTmaWarpSpecializedLowLatencyDispatch(
    TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
    int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k, int const num_experts,
    void const* input1, void const* input2, void const* weights1, void const* weights2, float const* fp8_dequant1,
    float const* fp8_dequant2, TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat, QuantParams quant_params, void const* bias1,
    void const* bias2, void* output1, void* output2, int const* num_active_experts_per,
    int const* active_expert_global_ids, int start_expert, cudaStream_t stream)
{
    // Delegate to bf16 runner
    return bf16_runner_.computeStridesTmaWarpSpecializedLowLatencyDispatch(layout_info1, layout_info2, num_tokens,
        gemm1_n, gemm1_k, gemm2_n, gemm2_k, num_experts, input1, input2, weights1, weights2, fp8_dequant1,
        fp8_dequant2, fc1_fp4_act_flat, fc2_fp4_act_flat, quant_params, bias1, bias2, output1, output2,
        num_active_experts_per, active_expert_global_ids, start_expert, stream);
}

size_t MixPrecisionMoEFCWrapper::getGemmWorkspaceSize(int num_experts_per_node) const
{
    // Use the max of both runners' GEMM workspace sizes (they share workspace sequentially)
    return std::max(
        bf16_runner_.getGemmWorkspaceSize(num_experts_per_node), fp4_runner_.getGemmWorkspaceSize(num_experts_per_node));
}

void MixPrecisionMoEFCWrapper::setExpertPrecisionAssignment(int const* assignment, int num_high_precision_experts)
{
    expert_precision_assignment_ = assignment;
    num_high_precision_experts_ = num_high_precision_experts;
}

void MixPrecisionMoEFCWrapper::setFp4Weights(
    void const* fp4_fc1_expert_weights, void const* fp4_fc2_expert_weights, QuantParams fp4_quant_params)
{
    fp4_fc1_expert_weights_ = fp4_fc1_expert_weights;
    fp4_fc2_expert_weights_ = fp4_fc2_expert_weights;
    fp4_quant_params_ = fp4_quant_params;
}

void MixPrecisionMoEFCWrapper::mixedGemm1(void const* fc1_bf16_weights, void const* fc1_bf16_biases,
    QuantParams bf16_quant_params, ActivationParams fc1_activation_type, int64_t hidden_size, int64_t inter_size,
    int num_experts, int64_t expanded_num_rows, cudaStream_t stream)
{
    // TODO: Mixed-precision GEMM1 dispatch:
    //
    // 1. Call bf16_runner_.getMoeGemmRunner().moeGemm() with:
    //    - input:   bf16_runner_.getPermutedData()                (bf16 expanded tokens)
    //    - weights: fc1_bf16_weights                               (bf16 weights)
    //    - offset:  bf16_runner_.getBf16ExpertFirstTokenOffset()   (bf16 group offsets)
    //    - output:  glu_inter_result_[0 .. bf16_boundary)          (bf16 region)
    //
    // 2. Call fp4_runner_.getMoeGemmRunner().moeGemm() with:
    //    - input:   bf16_runner_.getFp4ExpandedData()              (fp4 expanded tokens)
    //    - weights: fp4_fc1_expert_weights_                        (fp4 weights)
    //    - offset:  bf16_runner_.getFp4ExpertFirstTokenOffset()    (fp4 group offsets)
    //    - sf:      bf16_runner_.getFp4ExpandSf()                  (fp4 scaling factor)
    //    - output:  glu_inter_result_[bf16_boundary .. end)        (fp4 region, output is bf16)
    //
    TLLM_THROW("MixPrecisionMoEFCWrapper::mixedGemm1 not implemented");
}

void MixPrecisionMoEFCWrapper::mixedGemm2(void const* fc2_bf16_weights, void const* fc2_bf16_biases,
    QuantParams bf16_quant_params, void* final_output, int64_t hidden_size, int64_t unpadded_hidden_size,
    int64_t inter_size, int num_experts, int64_t num_rows, int64_t expanded_num_rows, int64_t experts_per_token,
    MOEParallelismConfig parallelism_config, bool enable_alltoall, cudaStream_t stream)
{
    // TODO: Mixed-precision GEMM2 dispatch:
    //
    // 1. cudaMemsetAsync(final_output, 0, num_rows * hidden_size * sizeof(__nv_bfloat16))
    //
    // 2. bf16 group: gemm2 + finalize (overwrite mode)
    //    - input:   bf16_runner_.getFc1Result()                    (bf16 activation output)
    //    - weights: fc2_bf16_weights
    //    - offset:  bf16_runner_.getBf16ExpertFirstTokenOffset()
    //    - output:  final_output                                   (overwrite)
    //
    // 3. fp4 group: gemm2 + finalize (accumulate mode)
    //    - input:   bf16_runner_.getFp4ActivationOutput()          (fp4 activation output)
    //    - weights: fp4_fc2_expert_weights_
    //    - offset:  bf16_runner_.getFp4ExpertFirstTokenOffset()
    //    - sf:      bf16_runner_.getFp4ActivationSf()
    //    - output:  final_output                                   (accumulate / add)
    //
    TLLM_THROW("MixPrecisionMoEFCWrapper::mixedGemm2 not implemented");
}

} // namespace kernels::cutlass_kernels

TRTLLM_NAMESPACE_END

#endif // ENABLE_FP4 && ENABLE_BF16
