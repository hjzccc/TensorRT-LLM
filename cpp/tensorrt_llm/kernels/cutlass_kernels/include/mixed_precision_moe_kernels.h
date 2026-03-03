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

#include "moe_kernels.h"

#if defined(ENABLE_FP4) && defined(ENABLE_BF16)

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cutlass_kernels
{

// ================================================================================================
// MixPrecisionMoeFCRunner
// ================================================================================================
//
// A bf16 CutlassMoeFCRunner extended with additional workspace buffers and pipeline stages
// for mixed-precision MoE (some experts use bf16, others use NVFP4).
//
// Responsibilities:
//   - Extra workspace buffers: dual offset arrays, fp4 expanded data, fp4 activation output
//   - New pipeline stages: mixedSort, mixedExpandInputRows, mixedDoActivation
//   - Public accessors for MixPrecisionMoEFCWrapper to reach internal state
//
// Note: This class does NOT override runMoe/gemm1/gemm2. The Wrapper orchestrates those.
//
class MixPrecisionMoeFCRunner : public CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>
{
    using Base = CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;

public:
    MixPrecisionMoeFCRunner() = default;
    ~MixPrecisionMoeFCRunner() override = default;

    // ---------- Override: workspace sizing includes mixed-precision buffers ----------

    size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
        int const num_experts, int const experts_per_token, ActivationType activation_type,
        MOEParallelismConfig parallelism_config, bool use_lora, bool use_deepseek_fp8_block_scale,
        bool min_latency_mode, bool use_awq) override;

    // ---------- Mixed-precision pipeline stages ----------

    // Partition the sorted expert_first_token_offset_ into per-precision offset arrays.
    // Input:
    //   expert_precision_assignment[E]: 1 = bf16, 0 = fp4
    //   Base::expert_first_token_offset_: [E+1] from fusedBuildExpertMapsSortFirstToken
    // Output:
    //   bf16_expert_first_token_offset_: [E+1] offset array for bf16-assigned experts
    //   fp4_expert_first_token_offset_:  [E+1] offset array for fp4-assigned experts
    //   bf16_num_tokens_: total tokens routed to bf16 group (device scalar)
    //   fp4_num_tokens_:  total tokens routed to fp4 group  (device scalar)
    void mixedSort(int const* expert_precision_assignment, int num_experts,
        int64_t expanded_num_rows, cudaStream_t stream);

    // Expand input rows into dual-precision buffers.
    // bf16 group → Base::permuted_data_ (__nv_bfloat16)
    // fp4 group  → mp_fp4_expanded_data_ (quantized fp4) + mp_fp4_expand_sf_ (scaling factor)
    void mixedExpandInputRows(void const* input_activations, void const* input_sf,
        int64_t num_rows, int64_t hidden_size, int num_experts, int experts_per_token,
        cudaStream_t stream);

    // Activation + optional quantization for both groups.
    // Input:
    //   glu_inter_result_ (contiguous [bf16 region | fp4 region], both in bf16 after gemm1)
    // bf16 group: activation → fc1_result_ (__nv_bfloat16)
    // fp4 group:  activation + quantize → mp_fp4_activation_output_ + mp_fp4_activation_sf_
    void mixedDoActivation(ActivationParams activation_type,
        int64_t inter_size, int64_t hidden_size, int num_experts, cudaStream_t stream);

    // ---------- Public accessors for MixPrecisionMoEFCWrapper ----------

    // Base class gemm runner
    auto& getMoeGemmRunner() { return Base::moe_gemm_runner_; }
    auto const& getMoeGemmRunner() const { return Base::moe_gemm_runner_; }

    // Base class buffer pointers
    __nv_bfloat16* getPermutedData() { return Base::permuted_data_; }
    int64_t* getExpertFirstTokenOffset() { return Base::expert_first_token_offset_; }
    void* getGluInterResult() { return Base::glu_inter_result_; }
    __nv_bfloat16* getFc1Result() { return Base::fc1_result_; }
    void* getFc2Result() { return Base::fc2_result_; }
    float* getPermutedTokenFinalScales() { return Base::permuted_token_final_scales_; }
    int* getPermutedRowToUnpermutedRow() { return Base::permuted_row_to_unpermuted_row_; }
    int* getPermutedTokenSelectedExperts() { return Base::permuted_token_selected_experts_; }

    // Base class TMA inputs and configs
    auto& getTmaWsGroupedGemm1Input() { return Base::tma_ws_grouped_gemm1_input_; }
    auto& getTmaWsGroupedGemm2Input() { return Base::tma_ws_grouped_gemm2_input_; }
    auto const& getGemm1Config() const { return Base::gemm1_config_; }
    auto const& getGemm2Config() const { return Base::gemm2_config_; }

    // Base class FP4 scaling and alpha pointers
    auto* getFc1Fp4ActScale() { return Base::fc1_fp4_act_scale_; }
    auto* getFc2Fp4ActScale() { return Base::fc2_fp4_act_scale_; }
    float const** getAlphaScalePtrArrayFc1() { return Base::alpha_scale_ptr_array_fc1_; }
    float const** getAlphaScalePtrArrayFc2() { return Base::alpha_scale_ptr_array_fc2_; }

    // Mixed-precision specific buffers
    int64_t* getBf16ExpertFirstTokenOffset() { return bf16_expert_first_token_offset_; }
    int64_t* getFp4ExpertFirstTokenOffset() { return fp4_expert_first_token_offset_; }
    int64_t* getBf16NumTokens() { return bf16_num_tokens_; }
    int64_t* getFp4NumTokens() { return fp4_num_tokens_; }
    void* getFp4ExpandedData() { return mp_fp4_expanded_data_; }
    auto* getFp4ExpandSf() { return mp_fp4_expand_sf_; }
    void* getFp4ActivationOutput() { return mp_fp4_activation_output_; }
    auto* getFp4ActivationSf() { return mp_fp4_activation_sf_; }

    // ---------- Workspace configuration ----------

    // Configure mixed-precision workspace pointers from the extra workspace region.
    // Called by Wrapper after base class configureWsPtrs.
    void configureMixedPrecisionWsPtrs(char* extra_ws_ptr, int64_t num_rows,
        int64_t hidden_size, int64_t inter_size, int num_experts, int experts_per_token);

    // Get extra workspace size needed for mixed-precision buffers (beyond base workspace).
    size_t getMixedPrecisionExtraWorkspaceSize(int64_t num_rows, int64_t hidden_size,
        int64_t inter_size, int num_experts, int experts_per_token) const;

protected:
    // ===== Mixed-precision workspace pointers =====

    // Dual expert-first-token offset arrays [E+1] each
    int64_t* bf16_expert_first_token_offset_{};
    int64_t* fp4_expert_first_token_offset_{};

    // Per-group token counts (device scalars)
    int64_t* bf16_num_tokens_{};
    int64_t* fp4_num_tokens_{};

    // FP4 expanded input tokens (quantized to fp4) and scaling factor
    void* mp_fp4_expanded_data_{};
    TmaWarpSpecializedGroupedGemmInput::ElementSF* mp_fp4_expand_sf_{};

    // FP4 activation output (activation + quantize to fp4) and scaling factor
    void* mp_fp4_activation_output_{};
    TmaWarpSpecializedGroupedGemmInput::ElementSF* mp_fp4_activation_sf_{};
};

// ================================================================================================
// MixPrecisionMoEFCWrapper
// ================================================================================================
//
// Orchestrates mixed-precision MoE by holding a bf16 runner and an fp4 runner.
// Dispatches 2x gemm1 + 2x gemm2 for the two precision groups.
//
// Weight convention:
//   - bf16 weights are passed through the standard runMoe() interface params
//   - fp4 weights are set via setFp4Weights() before calling runMoe()
//
// Pipeline (runMoe):
//   1. Base configureWsPtrs + Runner configureMixedPrecisionWsPtrs
//   2. fusedBuildExpertMapsSortFirstToken()       [free function, unchanged]
//   3. bf16_runner_.mixedSort()                   → dual offset arrays
//   4. bf16_runner_.mixedExpandInputRows()        → bf16 + fp4 expanded data
//   5. mixedGemm1()                               → 2x moeGemm (GEMM only, no activation)
//   6. bf16_runner_.mixedDoActivation()           → bf16 + fp4 activation outputs
//   7. mixedGemm2()                               → 2x gemm2+finalize
//        bf16: memset(final_output, 0) + overwrite
//        fp4:  accumulate into final_output
//
class MixPrecisionMoEFCWrapper : public CutlassMoeFCRunnerInterface
{
    using Bf16Runner = MixPrecisionMoeFCRunner;
    using Fp4Runner = CutlassMoeFCRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16, __nv_bfloat16>;

public:
    MixPrecisionMoEFCWrapper();
    ~MixPrecisionMoEFCWrapper() override = default;

    // ---------- CutlassMoeFCRunnerInterface overrides ----------

    size_t getWorkspaceSize(int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
        int const num_experts, int const experts_per_token, ActivationType activation_type,
        MOEParallelismConfig parallelism_config, bool use_lora, bool use_deepseek_fp8_block_scale,
        bool min_latency_mode, bool use_awq) override;

    void setTactic(std::optional<cutlass_extensions::CutlassGemmConfig> gemm1_config,
        std::optional<cutlass_extensions::CutlassGemmConfig> gemm2_config) override;

    std::vector<cutlass_extensions::CutlassGemmConfig> getTactics(MoeGemmId gemm_id) override;

    // Main entry point: orchestrates the full mixed-precision MoE pipeline.
    // fc1/fc2_expert_weights and quant_params carry the bf16 weight set.
    // fp4 weights must be set via setFp4Weights() before this call.
    void runMoe(void const* input_activations, void const* input_sf, bool const swizzled_input_sf,
        int const* token_selected_experts, float const* token_final_scales,
        void const* fc1_expert_weights, void const* fc1_expert_biases,
        ActivationParams fc1_activation_type,
        void const* fc2_expert_weights, void const* fc2_expert_biases,
        QuantParams quant_params,
        int64_t const num_rows, int64_t const num_valid_rows,
        int64_t const hidden_size, int64_t const unpadded_hidden_size, int64_t const inter_size,
        int const num_experts, int const experts_per_token,
        char* workspace_ptr, void* final_output,
        int* unpermuted_row_to_permuted_row,
        MOEParallelismConfig parallelism_config, bool const enable_alltoall,
        bool use_lora, LoraParams& lora_params,
        bool use_deepseek_fp8_block_scale, bool min_latency_mode,
        MoeMinLatencyParams& min_latency_params,
        cudaStream_t stream) override;

    // gemm1/gemm2: not used for mixed-precision — required by interface, throws if called.
    void gemm1(void const* const input, void* const output, void* const intermediate_result,
        int64_t const* const expert_first_token_offset,
        TmaWarpSpecializedGroupedGemmInput tma_ws_input_template,
        void const* const fc1_expert_weights, void const* const fc1_expert_biases,
        int64_t const* const num_valid_tokens_ptr, void const* const fc1_int_scales,
        float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
        TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat,
        QuantParams quant_params,
        int64_t const num_rows, int64_t const expanded_num_rows, int64_t const expected_tokens_per_expert,
        int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node,
        ActivationParams fc1_activation_type, float const** alpha_scale_ptr_array, bool bias_is_broadcast,
        bool use_deepseek_fp8_block_scale, cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config,
        bool min_latency_mode, int* num_active_experts_per, int* active_expert_global_ids) override
    {
        TLLM_THROW("gemm1 not supported on MixPrecisionMoEFCWrapper");
    }

    void gemm2(void const* const input, void* const gemm_output, void* const final_output,
        int64_t const* const expert_first_token_offset,
        TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
        void const* const fc2_expert_weights, void const* const fc2_expert_biases,
        void const* const fc2_int_scales, float const* const fc2_fp8_dequant,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
        QuantParams quant_params,
        float const* const token_topk_unpermuted_scales, float const* const token_topk_permuted_scales,
        int const* const unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
        int const* const token_selected_experts, int64_t const* const num_valid_tokens_ptr,
        int64_t const num_rows, int64_t const expanded_num_rows, int64_t const expected_tokens_per_expert,
        int64_t const hidden_size, int64_t const unpadded_hidden_size, int64_t const inter_size,
        int const num_experts_per_node, int64_t const experts_per_token,
        float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
        bool use_deepseek_fp8_block_scale,
        cudaStream_t stream, MOEParallelismConfig parallelism_config, bool const enable_alltoall,
        cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
        int* num_active_experts_per, int* active_expert_global_ids) override
    {
        TLLM_THROW("gemm2 not supported on MixPrecisionMoEFCWrapper");
    }

    std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
    computeStridesTmaWarpSpecializedDispatch(int64_t const* expert_first_token_offset,
        TmaWarpSpecializedGroupedGemmInput layout_info1, TmaWarpSpecializedGroupedGemmInput layout_info2,
        int64_t num_tokens, int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k,
        int64_t gemm2_n, int64_t gemm2_k, int const num_experts_per_node,
        void const* gemm1_in, void const* gemm2_in,
        void const* weights1, void const* weights2,
        float const* alpha_scale_flat1, float const* alpha_scale_flat2,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2,
        QuantParams quant_params, void const* bias1, void const* bias2,
        void* gemm1_output, void* gemm2_output, float const* router_scales,
        int const* permuted_row_to_unpermuted_row, cudaStream_t stream) override;

    std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
    computeStridesTmaWarpSpecializedLowLatencyDispatch(
        TmaWarpSpecializedGroupedGemmInput layout_info1,
        TmaWarpSpecializedGroupedGemmInput layout_info2,
        int64_t num_tokens, int64_t gemm1_n, int64_t gemm1_k,
        int64_t gemm2_n, int64_t gemm2_k, int const num_experts,
        void const* input1, void const* input2,
        void const* weights1, void const* weights2,
        float const* fp8_dequant1, float const* fp8_dequant2,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
        QuantParams quant_params, void const* bias1, void const* bias2,
        void* output1, void* output2, int const* num_active_experts_per,
        int const* active_expert_global_ids, int start_expert,
        cudaStream_t stream) override;

    size_t getGemmWorkspaceSize(int num_experts_per_node) const override;

    // ---------- Mixed-precision configuration ----------

    // Set per-expert precision assignment before runMoe.
    // assignment[i]: 1 = bf16, 0 = fp4.  num_high_precision_experts = count of bf16 experts.
    void setExpertPrecisionAssignment(int const* assignment, int num_high_precision_experts);

    // Set fp4-specific weights and quant params before runMoe.
    // bf16 weights are passed through the standard runMoe interface params.
    void setFp4Weights(void const* fp4_fc1_expert_weights, void const* fp4_fc2_expert_weights,
        QuantParams fp4_quant_params);

    // Access the underlying runners
    Bf16Runner& getBf16Runner() { return bf16_runner_; }
    Bf16Runner const& getBf16Runner() const { return bf16_runner_; }
    Fp4Runner& getFp4Runner() { return fp4_runner_; }
    Fp4Runner const& getFp4Runner() const { return fp4_runner_; }

private:
    // ---------- Internal mixed-precision GEMM dispatch ----------

    // Dispatches 2x moeGemm for gemm1 (bf16 + fp4, GEMM only — no activation).
    // Both write into glu_inter_result_: [bf16 region | fp4 region] (contiguous, both bf16).
    void mixedGemm1(void const* fc1_bf16_weights, void const* fc1_bf16_biases,
        QuantParams bf16_quant_params,
        ActivationParams fc1_activation_type,
        int64_t hidden_size, int64_t inter_size, int num_experts,
        int64_t expanded_num_rows, cudaStream_t stream);

    // Dispatches 2x gemm2+finalize.
    // bf16 group: memset(final_output, 0) + finalize (overwrite)
    // fp4 group:  finalize (accumulate into existing final_output)
    void mixedGemm2(void const* fc2_bf16_weights, void const* fc2_bf16_biases,
        QuantParams bf16_quant_params,
        void* final_output,
        int64_t hidden_size, int64_t unpadded_hidden_size, int64_t inter_size,
        int num_experts, int64_t num_rows, int64_t expanded_num_rows,
        int64_t experts_per_token,
        MOEParallelismConfig parallelism_config, bool enable_alltoall,
        cudaStream_t stream);

    // ---------- Members ----------

    Bf16Runner bf16_runner_;
    Fp4Runner fp4_runner_;

    // Per-expert precision assignment (host pointer, set before runMoe)
    int const* expert_precision_assignment_{};  // [E]: 1 = bf16, 0 = fp4
    int num_high_precision_experts_{};

    // FP4 weights (set via setFp4Weights before runMoe)
    void const* fp4_fc1_expert_weights_{};
    void const* fp4_fc2_expert_weights_{};
    QuantParams fp4_quant_params_{};
};

} // namespace cutlass_kernels
} // namespace kernels

TRTLLM_NAMESPACE_END

#endif // ENABLE_FP4 && ENABLE_BF16
