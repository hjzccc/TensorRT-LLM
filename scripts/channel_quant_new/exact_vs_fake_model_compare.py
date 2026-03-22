#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/channel_quant_new')
sys.path.insert(0, '/workspace/channel_quant')

import torch
import exact_docker_eval as ede


def fake_nvfp4_linear(input, weight, bias=None):
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
    from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale
    input_2d, prefix_shape = ede.flatten_for_linear(input)
    s_in2 = fp4_global_scale(input_2d).to(torch.float32)
    s_w2 = fp4_global_scale(weight).to(torch.float32)
    weight_fp4, weight_scale_cutlass = torch.ops.trtllm.fp4_quantize(weight, s_w2, ede.SCALING_VECTOR_SIZE, False)
    alpha_fused = (1.0 / (s_in2 * s_w2)).to(torch.float32)
    out = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        input_2d,
        weight_fp4,
        bias,
        [s_in2],
        [weight_scale_cutlass, alpha_fused],
        [],
        [],
    )
    return ede.restore_linear_shape(out, prefix_shape)


def fake_fp8_linear(input, weight, bias=None):
    import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
    input_2d, prefix_shape = ede.flatten_for_linear(input)
    weight_scale = (torch.max(torch.abs(weight.float())) / 448).to(torch.float32).to('cuda')
    weight_fp8 = (weight.float() / weight_scale).to(torch.float8_e4m3fn)
    input_scale = torch.tensor(1.0, device='cuda', dtype=torch.float32)
    out = torch.ops.auto_deploy.torch_fake_quant_fp8_linear(
        input_2d,
        weight_fp8,
        bias,
        [input_scale],
        [weight_scale],
        [],
        [],
    )
    return ede.restore_linear_shape(out, prefix_shape)


def run_compare():
    tokenizer = ede.AutoTokenizer.from_pretrained(ede.MODEL_ID, trust_remote_code=True)
    eval_ids, nsamples, seqlen = ede.load_eval_data(tokenizer, 2048)
    nsamples = 4
    snapshot_dir, root_config, weight_map = ede.load_root_config(ede.MODEL_ID)
    config = ede.build_text_config(root_config)
    device = torch.device('cuda')
    dtype = torch.float16

    # Patch exact_docker_eval's linear functions to fake wrappers
    orig_nvfp4 = ede.nvfp4_linear
    orig_fp8 = ede.fp8_linear
    try:
        ede.nvfp4_linear = fake_nvfp4_linear
        ede.fp8_linear = fake_fp8_linear

        fake_results = {}
        for run_config in ede.default_run_configs():
            ppl = ede.evaluate_ppl(eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir, device, dtype, run_config)
            fake_results[run_config.label] = ppl

    finally:
        ede.nvfp4_linear = orig_nvfp4
        ede.fp8_linear = orig_fp8

    exact_path = Path('/workspace/exact_docker_eval_small.json')
    exact = json.loads(exact_path.read_text())['results']

    rows = []
    for label in ['uniform_bf16','uniform_nvfp4','uniform_fp8']:
        rows.append({
            'label': label,
            'ppl_exact': exact[label]['ppl'],
            'ppl_fake': round(fake_results[label], 4),
            'delta_exact_minus_fake': round(exact[label]['ppl'] - fake_results[label], 4),
        })

    out = {'metadata': {'nsamples': nsamples, 'seqlen': seqlen}, 'rows': rows}
    Path('/workspace/exact_vs_fake_model_compare.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    run_compare()
