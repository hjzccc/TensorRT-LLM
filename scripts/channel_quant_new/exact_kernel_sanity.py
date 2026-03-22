#!/usr/bin/env python3
"""TRT-LLM exact kernel sanity checks.

This script mirrors TRT-LLM's own unit tests:
- test_quant_linear_nvfp4_matches_fused_op
- test_quant_linear_fp8_matches_fused_op

Run it inside the `trtllm-dual-tile` docker container.
"""

import json
from pathlib import Path

import torch


def run_nvfp4_case(dtype=torch.float16, bias_enabled=True):
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

    x = torch.rand(3, 32, device="cuda", dtype=dtype)
    W = torch.rand(32, 32, device="cuda", dtype=dtype)
    bias = (torch.rand(32, device="cuda") * 10).to(dtype) if bias_enabled else None

    s_in2 = fp4_global_scale(x).to(torch.float32)
    s_w2 = fp4_global_scale(W).to(torch.float32)

    weight_fp4, weight_scale_cutlass = torch.ops.trtllm.fp4_quantize(W, s_w2, 16, False)
    alpha_fused = (1.0 / (s_in2 * s_w2)).to(torch.float32)

    out_fused = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        x,
        weight_fp4,
        bias=bias,
        input_scale=s_in2,
        weight_scale=weight_scale_cutlass,
        alpha=alpha_fused,
    )

    out_ref = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        x,
        weight_fp4,
        bias,
        [s_in2],
        [weight_scale_cutlass, alpha_fused],
        [],
        [],
    )

    diff = (out_fused.float() - out_ref.float()).abs()
    return {
        "dtype": str(dtype),
        "bias": bias_enabled,
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "within_trtllm_tolerance": bool(torch.allclose(out_fused, out_ref, rtol=1e-3, atol=5e-3)),
        "tolerance": {"rtol": 1e-3, "atol": 5e-3},
        "out_shape": list(out_fused.shape),
    }


def run_fp8_case(dtype=torch.float16, bias_enabled=True):
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401

    x = torch.rand(3, 16, device="cuda", dtype=dtype)
    W = torch.rand(32, 16, device="cuda", dtype=dtype)
    bias = (torch.rand(32, device="cuda") * 10).to(dtype) if bias_enabled else None

    # EXACT TRT-LLM unit-test recipe from test_quant.py
    # Match TRT-LLM test_quant.py exactly: keep scales in FP32
    weight_scale = (torch.max(torch.abs(W.float())) / 448).to(torch.float32).to("cuda")
    weight_fp8 = (W.float() / weight_scale).to(torch.float8_e4m3fn)
    input_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    out_fused = torch.ops.auto_deploy.torch_quant_fp8_linear(
        x,
        weight_fp8,
        bias=bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    out_ref = torch.ops.auto_deploy.torch_fake_quant_fp8_linear(
        x,
        weight_fp8,
        bias,
        [input_scale],
        [weight_scale],
        [],
        [],
    )

    diff = (out_fused.float() - out_ref.float()).abs()
    return {
        "dtype": str(dtype),
        "bias": bias_enabled,
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "within_trtllm_tolerance": bool(torch.allclose(out_fused, out_ref, rtol=5e-4, atol=5e-4)),
        "tolerance": {"rtol": 5e-4, "atol": 5e-4},
        "out_shape": list(out_fused.shape),
    }


def main():
    assert torch.cuda.is_available(), "CUDA required"

    results = {
        "nvfp4": [
            run_nvfp4_case(torch.float16, True),
            run_nvfp4_case(torch.float16, False),
            run_nvfp4_case(torch.bfloat16, True),
            run_nvfp4_case(torch.bfloat16, False),
        ],
        "fp8": [
            run_fp8_case(torch.float16, True),
            run_fp8_case(torch.float16, False),
            run_fp8_case(torch.bfloat16, True),
            run_fp8_case(torch.bfloat16, False),
        ],
    }

    summary = {
        "all_nvfp4_passed": all(r["within_trtllm_tolerance"] for r in results["nvfp4"]),
        "all_fp8_passed": all(r["within_trtllm_tolerance"] for r in results["fp8"]),
    }

    payload = {"summary": summary, "results": results}
    out = Path('/tmp/exact_kernel_sanity_results.json')
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
