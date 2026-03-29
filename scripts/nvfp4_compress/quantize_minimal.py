#!/usr/bin/env python3
"""Memory-minimal NVFP4 quantization: one safetensors shard per projection."""
import ctypes
import gc
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

CKPT = "/code/tensorrt_llm/scripts/nvfp4_compress/nvfp4_checkpoint"
SNAP = "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
SKIP = ["layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate", "embed_tokens",
        "lm_head", "A_log", "dt_bias", "conv1d", "linear_attn", "self_attn", "mtp.", "model.visual."]

libc = ctypes.CDLL("libc.so.6")


def quantize_expert_block(block_3d, proj_name, layer_prefix, shard_idx, weight_map):
    shard_data = {}
    for ei in range(block_3d.shape[0]):
        w = block_3d[ei].contiguous().cuda()
        gs = fp4_global_scale(w).to(torch.float32)
        p, s = torch.ops.trtllm.fp4_quantize(w, gs, 16, False)
        base = f"{layer_prefix}{ei}.{proj_name}"
        shard_data[f"{base}.weight"] = p.cpu()
        shard_data[f"{base}.weight_scale"] = s.cpu()
        shard_data[f"{base}.weight_scale_2"] = gs.cpu().reshape(1)
        shard_data[f"{base}.input_scale"] = torch.ones(1, dtype=torch.float32)
        del w, gs, p, s
    torch.cuda.empty_cache()

    sn = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
    save_file(shard_data, f"{CKPT}/{sn}")
    for k in shard_data:
        weight_map[k] = sn
    count = len(shard_data) // 4
    del shard_data
    gc.collect()
    libc.malloc_trim(0)
    return count


def main():
    if os.path.exists(CKPT):
        shutil.rmtree(CKPT)
    os.makedirs(CKPT)

    with open(f"{SNAP}/model.safetensors.index.json") as f:
        src_wm = json.load(f)["weight_map"]

    shard_idx = 0
    weight_map = {}
    total_q = 0
    total_bf16 = 0
    total_skip = 0
    t0 = time.time()

    for key in sorted(src_wm.keys()):
        if any(p in key for p in ["mtp.", "model.visual."]):
            total_skip += 1
            continue

        new_key = key.replace("model.language_model.", "model.")
        sf = src_wm[key]

        with safe_open(f"{SNAP}/{sf}", framework="pt", device="cpu") as f:
            tensor = f.get_tensor(key)

        is_fused = "mlp.experts." in key and tensor.dim() == 3 and ("gate_up_proj" in key or "down_proj" in key)

        if is_fused:
            fn = key.split("mlp.experts.")[-1]
            lp = new_key.split("mlp.experts.")[0] + "mlp.experts."
            is_gu = fn == "gate_up_proj"

            if is_gu:
                half = tensor.shape[1] // 2
                gate_block = tensor[:, :half, :].contiguous()
                up_block = tensor[:, half:, :].contiguous()
                del tensor
                gc.collect()
                libc.malloc_trim(0)

                n = quantize_expert_block(gate_block, "gate_proj", lp, shard_idx, weight_map)
                total_q += n
                shard_idx += 1
                del gate_block
                gc.collect()
                libc.malloc_trim(0)
                print(f"  Shard {shard_idx-1}: gate_proj {n} experts ({time.time()-t0:.0f}s)", flush=True)

                n = quantize_expert_block(up_block, "up_proj", lp, shard_idx, weight_map)
                total_q += n
                shard_idx += 1
                del up_block
                gc.collect()
                libc.malloc_trim(0)
                print(f"  Shard {shard_idx-1}: up_proj {n} experts ({time.time()-t0:.0f}s)", flush=True)
            else:
                n = quantize_expert_block(tensor, fn, lp, shard_idx, weight_map)
                total_q += n
                shard_idx += 1
                del tensor
                gc.collect()
                libc.malloc_trim(0)
                print(f"  Shard {shard_idx-1}: {fn} {n} experts ({time.time()-t0:.0f}s)", flush=True)

        elif any(p in key for p in SKIP) or not key.endswith(".weight"):
            sn = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            save_file({new_key: tensor}, f"{CKPT}/{sn}")
            weight_map[new_key] = sn
            shard_idx += 1
            total_bf16 += 1
            del tensor

        else:
            w = tensor.cuda()
            gs = fp4_global_scale(w).to(torch.float32)
            p, s = torch.ops.trtllm.fp4_quantize(w, gs, 16, False)
            base = new_key.replace(".weight", "")
            d = {
                f"{base}.weight": p.cpu(),
                f"{base}.weight_scale": s.cpu(),
                f"{base}.weight_scale_2": gs.cpu().reshape(1),
                f"{base}.input_scale": torch.ones(1, dtype=torch.float32),
            }
            sn = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
            save_file(d, f"{CKPT}/{sn}")
            for k in d:
                weight_map[k] = sn
            shard_idx += 1
            total_q += 1
            del tensor, w, gs, p, s, d
            gc.collect()
            torch.cuda.empty_cache()

    ts = shard_idx
    for old in set(weight_map.values()):
        new = old.replace("PLACEHOLDER", f"{ts:05d}")
        if old != new:
            os.rename(f"{CKPT}/{old}", f"{CKPT}/{new}")
    weight_map = {k: v.replace("PLACEHOLDER", f"{ts:05d}") for k, v in weight_map.items()}

    json.dump({"metadata": {}, "weight_map": weight_map},
              open(f"{CKPT}/model.safetensors.index.json", "w"), indent=2)

    with open(f"{SNAP}/config.json") as f:
        orig_cfg = json.load(f)
    tc = orig_cfg.get("text_config", {})
    tc["model_type"] = "qwen3_next"
    tc["architectures"] = ["Qwen3NextForCausalLM"]
    tc["torch_dtype"] = "bfloat16"
    for k in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        if k in orig_cfg and k not in tc:
            tc[k] = orig_cfg[k]
    json.dump(tc, open(f"{CKPT}/config.json", "w"), indent=2)

    hf_qc = {"producer": {"name": "modelopt", "version": "0.37.0"},
              "quantization": {"quant_algo": "NVFP4", "group_size": 16,
                               "exclude_modules": [f"model.layers.{i}.mlp.gate" for i in range(tc["num_hidden_layers"])] + ["lm_head"]}}
    json.dump(hf_qc, open(f"{CKPT}/hf_quant_config.json", "w"), indent=2)

    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "generation_config.json"]:
        src = f"{SNAP}/{fn}"
        if os.path.exists(src):
            shutil.copy2(src, f"{CKPT}/{fn}")

    total_gb = sum(os.path.getsize(f"{CKPT}/{f}") for f in os.listdir(CKPT) if not f.startswith(".")) / 1e9
    print(f"\nDone: {ts} shards, {total_q} quantized, {total_bf16} bf16, {total_skip} skipped, {total_gb:.1f}GB ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
