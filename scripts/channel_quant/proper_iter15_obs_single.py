#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch

from proper_iter15_obs_remainder import build_plans
from proper_iter13_assignment import evaluate_compensated_plan
from proper_iter01 import load_cache
from proper_iter10_novel_perchannel import load_metric_cache
from proper_eval import load_gptq_standard_data, dtype_from_name, atomic_json_dump
from baselines_comparison import resolve_non_expert_bytes
from spike1_ground_truth import MODEL_ID, load_root_config, build_text_config

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model-id', default=MODEL_ID)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--dtype', default='bfloat16', choices=['bfloat16','float16','float32'])
    p.add_argument('--cache-path', type=Path, default=RESULTS_DIR / 'proper_iter01_calibration_cache.pt')
    p.add_argument('--metric-cache-path', type=Path, default=RESULTS_DIR / 'proper_iter10_novel_perchannel_metric_cache.pt')
    p.add_argument('--obs-damp-percent', type=float, default=0.01)
    p.add_argument('--calib-screen-chunks', type=int, default=4)
    p.add_argument('--plan-name', default='obs_union_router_affinity_topup_5pct')
    p.add_argument('--output-json', type=Path, default=RESULTS_DIR / 'proper_iter15_obs_single.json')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.set_grad_enabled(False)

    _tok, calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(args.model_id)
    calib_chunks = calib_chunks[: args.calib_screen_chunks]
    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    text_config = build_text_config(root_config)
    total_bf16_bytes = int(root_config.get('total_size', 0) or 0)
    non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

    calibration, _hot_experts, _hot_scores = load_cache(args.cache_path, args.model_id)
    router_affinity_cache, _hessian_cache, _micromix_mean_abs, _micromix_thresholds, _metric_meta = load_metric_cache(args.metric_cache_path, args.model_id)
    if calibration is None or router_affinity_cache is None:
        raise RuntimeError('failed to load caches')

    plans = build_plans(calibration, router_affinity_cache, text_config, non_expert_bytes, total_expert_elems)
    chosen = None
    chosen_extras = None
    for ap in plans:
        if ap.plan.name == args.plan_name:
            chosen = ap.plan
            chosen_extras = ap.extras
            break
    if chosen is None:
        raise RuntimeError(f'plan not found: {args.plan_name}')

    ppl, nll, nsamples = evaluate_compensated_plan(
        args.model_id, chosen, calib_chunks, test_ids, text_config, weight_map, snapshot_dir, device, dtype, args.obs_damp_percent
    )
    out = {
        'metadata': {
            'model': args.model_id,
            'screening': True,
            'screen_calibration_chunks': args.calib_screen_chunks,
            'evaluation': eval_info,
        },
        'result': {
            'name': chosen.name,
            'ppl': round(ppl, 4),
            'nll': round(nll, 6),
            'memory_gb': round(float(chosen.memory_gb), 3),
            'fp8_fraction': round(float(chosen.fp8_weights) / max(total_expert_elems, 1), 4),
            **(chosen_extras or {}),
        }
    }
    atomic_json_dump(args.output_json, out)
    print(json.dumps(out, indent=2))
    print(f'Saved results -> {args.output_json}', flush=True)

if __name__ == '__main__':
    main()
