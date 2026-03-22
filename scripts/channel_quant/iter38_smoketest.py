#!/usr/bin/env python3
"""Quick smoke test for 4/6 adaptive block scaling.
Tests on 4 chunks to verify improvement before full run.
"""
import sys, time
sys.path.insert(0, '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant')

import torch
from spike1_ground_truth import quantize_to_nvfp4_columns, round_to_e2m1_grid, EPS
import baselines_comparison as bc
import spike1_ground_truth as sgt
from proper_eval import atomic_json_dump, dtype_from_name, evaluate_plan, load_gptq_standard_data
from proper_iter01 import build_plan_from_masks, load_cache
from proper_iter07 import JOINT_MEDIUM_TOPUP_FRACTION, build_joint_with_topup_masks
from baselines_comparison import resolve_non_expert_bytes
from spike1_ground_truth import MODEL_ID, build_text_config, load_root_config
from pathlib import Path

RESULTS_DIR = Path('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/channel_quant/results')
CACHE_PATH = RESULTS_DIR / 'proper_iter01_calibration_cache.pt'
MACA_CACHE_PATH = RESULTS_DIR / 'proper_iter29_maca_uniform4k_cache.pt'

def quantize_to_nvfp4_columns_46(weight):
    if weight.ndim == 1:
        weight = weight.unsqueeze(-1); squeeze = True
    else:
        squeeze = False
    blocks = weight.reshape(weight.shape[0] // 16, 16, weight.shape[1])
    absmax = blocks.abs().amax(dim=1, keepdim=True)
    scale6 = absmax / 6.0
    q6 = round_to_e2m1_grid(blocks / (scale6 + EPS)) * scale6
    mse6 = ((blocks - q6) ** 2).mean(dim=1, keepdim=True)
    scale4 = absmax / 4.0
    q4 = round_to_e2m1_grid(blocks / (scale4 + EPS)) * scale4
    mse4 = ((blocks - q4) ** 2).mean(dim=1, keepdim=True)
    dq = torch.where(mse4 < mse6, q4, q6).reshape_as(weight)
    return dq.squeeze(-1) if squeeze else dq

device = torch.device('cuda')
dtype = torch.bfloat16

snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
text_config = build_text_config(root_config)
import json as _json
index_path = snapshot_dir / 'model.safetensors.index.json'
with index_path.open() as f:
    index_payload = _json.load(f)
total_bf16_bytes = int(root_config.get('total_size', 0) or index_payload['metadata']['total_size'])
non_expert_bytes, total_expert_elems = resolve_non_expert_bytes(total_bf16_bytes, text_config)

tokenizer, calib_chunks, test_ids, calib_info, eval_info = load_gptq_standard_data(MODEL_ID)
# Use only 4 chunks for smoke test
test_ids_small = test_ids[:, :4 * 2048]

calibration, _, _ = load_cache(CACHE_PATH, MODEL_ID)
w1_masks, w2_masks = build_joint_with_topup_masks(calibration, text_config, JOINT_MEDIUM_TOPUP_FRACTION)

# Test 1: Standard M=6
t0 = time.time()
plan_std = build_plan_from_masks('standard_m6', 'Standard M=6', text_config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks)
ppl_std, _, _ = evaluate_plan(plan_std, test_ids_small, text_config, weight_map, snapshot_dir, device, dtype)
print(f'Standard M=6: PPL={ppl_std:.4f} ({time.time()-t0:.1f}s)')

# Test 2: 4/6 adaptive
sgt.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns_46
bc.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns_46
t0 = time.time()
plan_46 = build_plan_from_masks('adaptive_46', '4/6 adaptive', text_config, non_expert_bytes, total_expert_elems, w1_masks, w2_masks)
ppl_46, _, _ = evaluate_plan(plan_46, test_ids_small, text_config, weight_map, snapshot_dir, device, dtype)
sgt.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns
bc.quantize_to_nvfp4_columns = quantize_to_nvfp4_columns
print(f'4/6 adaptive: PPL={ppl_46:.4f} ({time.time()-t0:.1f}s)')
print(f'Improvement: {ppl_std - ppl_46:.4f} PPL')
