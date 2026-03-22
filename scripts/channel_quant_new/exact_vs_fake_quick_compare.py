#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/channel_quant_new')
sys.path.insert(0, '/workspace/channel_quant')

import exact_docker_eval as exact_eval
import real_eval_pipeline as fake_eval
from spike1_ground_truth import build_text_config, load_root_config
from transformers import AutoTokenizer


def run_fake_subset(eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir, device, dtype, mode, label):
    return fake_eval.evaluate_ppl(
        eval_ids=eval_ids,
        nsamples=nsamples,
        seqlen=seqlen,
        config=config,
        weight_map=weight_map,
        snapshot_dir=snapshot_dir,
        device=device,
        dtype=dtype,
        mode=mode,
        label=label,
    )


def main():
    device = torch.device('cuda')
    dtype = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(exact_eval.MODEL_ID, trust_remote_code=True)
    eval_ids, full_nsamples, seqlen = fake_eval.load_eval_data(tokenizer, 2048)
    nsamples = 4  # quick comparison only
    snapshot_dir, root_config, weight_map = load_root_config(exact_eval.MODEL_ID)
    config = build_text_config(root_config)

    rows = []
    for label, mode in [('uniform_bf16','bf16'), ('uniform_nvfp4','nvfp4'), ('uniform_fp8','fp8')]:
        ppl_exact = exact_eval.evaluate_ppl(eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir, device, dtype, exact_eval.EvalConfig(label, mode, 'reference_fp8' if mode=='fp8' else ('reference_nvfp4' if mode=='nvfp4' else 'moe_only')))
        fake_mode = 'bf16' if mode=='bf16' else ('fp4' if mode=='nvfp4' else 'fp8')
        ppl_fake = run_fake_subset(eval_ids, nsamples, seqlen, config, weight_map, snapshot_dir, device, torch.bfloat16, fake_mode, label+'_fake')
        rows.append((label, ppl_exact, ppl_fake, ppl_exact - ppl_fake))

    out = {
        'metadata': {'nsamples': nsamples, 'seqlen': seqlen, 'eval_tokens': nsamples * seqlen},
        'rows': [
            {'label': l, 'ppl_exact': pe, 'ppl_fake': pf, 'delta_exact_minus_fake': d}
            for l, pe, pf, d in rows
        ]
    }
    Path('/tmp/exact_vs_fake_quick_compare.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
