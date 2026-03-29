import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
import tensorrt_llm._torch.auto_deploy.custom_ops
import exact_docker_eval as ee
from spike1_ground_truth import (
    build_text_config, layer_keys, load_root_config,
    shorten_layer_tensors, move_tensor, release_tensors,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.quantization.torch_quant import _quantize_nvfp4, _dequantize_nvfp4

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
CAL_DIR = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/calibration_layers")
OUT_DIR = Path("/code/tensorrt_llm/scripts/channel_quant_new/profiling/calibration_layers")

sd, rc, wm = load_root_config(MODEL_ID)
cfg = build_text_config(rc)
store = ee.WeightStore(MODEL_ID, sd, wm)
t0 = time.time()

for li in range(40):
    lt = cfg.layer_types[li]
    raw = store.load_tensors(layer_keys(li, lt))
    ld = shorten_layer_tensors(li, raw, torch.device("cuda"), torch.float16)
    del raw

    gup = ld["mlp.experts.gate_up_proj"]
    dp = ld["mlp.experts.down_proj"]

    with open(CAL_DIR / f"layer_{li}.json") as f:
        cal = json.load(f)

    for e in cal.get("experts", []):
        ei = e["expert"]
        for proj, w_all in [("w1", gup), ("w2", dp)]:
            w = w_all[ei]
            ref = w.float()
            nvfp4_out = ee.nvfp4_linear(
                torch.eye(w.shape[1], dtype=torch.float16, device="cuda")[:32],
                w
            )
            quant_err_per_channel = []
            for ch_start in range(0, w.shape[0], 32):
                ch_end = min(ch_start + 32, w.shape[0])
                w_block = w[ch_start:ch_end]
                if w_block.shape[0] < 32:
                    w_block = torch.nn.functional.pad(w_block, (0, 0, 0, 32 - w_block.shape[0]))
                eye_in = torch.eye(w_block.shape[1], dtype=torch.float16, device="cuda")
                nvfp4_block = ee.nvfp4_linear(eye_in, w_block)
                ref_block = torch.nn.functional.linear(eye_in, w_block)
                block_err = (nvfp4_block[:, :ch_end-ch_start] - ref_block[:, :ch_end-ch_start]).float().pow(2).sum(dim=0)
                quant_err_per_channel.extend(block_err.cpu().tolist())

            e[proj]["quant_error"] = quant_err_per_channel

    with open(OUT_DIR / f"layer_{li}.json", "w") as f:
        json.dump(cal, f)

    release_tensors(ld)
    torch.cuda.empty_cache()
    if (li + 1) % 10 == 0:
        print(f"Layer {li+1}/40 ({time.time()-t0:.0f}s)", flush=True)

print(f"Done in {time.time()-t0:.0f}s")
