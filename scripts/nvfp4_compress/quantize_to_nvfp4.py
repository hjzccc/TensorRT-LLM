#!/usr/bin/env python3
"""Quantize Qwen3.5-35B-A3B BF16 → NVFP4 checkpoint, layer by layer.

Loads BF16 weights one layer at a time from safetensors (memory-mapped),
quantizes linear weights with torch.ops.trtllm.fp4_quantize, and saves
a sharded NVFP4 checkpoint that TRT-LLM can load.

Reference format: nvidia/Qwen3-30B-A3B-NVFP4
Output format: model_type=qwen3_next, hf_quant_config.json with quant_algo=NVFP4

Usage:
    docker exec trtllm-dual-tile bash -c \
        "cd /code/tensorrt_llm && python3 -u scripts/nvfp4_compress/quantize_to_nvfp4.py"
"""
import gc
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# TRT-LLM quantization ops
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 — registers ops
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale

# ── Config ────────────────────────────────────────────────────────────────

SRC_MODEL = "Qwen/Qwen3.5-35B-A3B"
SRC_SNAP = None  # auto-detect from HF cache
DST_DIR = "/code/tensorrt_llm/scripts/nvfp4_compress/nvfp4_checkpoint"
BLOCK_SIZE = 16
SHARD_SIZE_GB = 1

# Keys that should NOT be quantized (norms, gates, embeddings, biases, MTP)
SKIP_QUANT_PATTERNS = [
    "layernorm",
    "norm.weight",
    "mlp.gate.weight",
    "shared_expert_gate",
    "embed_tokens",
    "lm_head",
    "A_log",
    "dt_bias",
    "conv1d",
    "linear_attn",
    "self_attn",
    "mtp.",
    "model.visual.",
]


def find_src_snapshot():
    """Find the HF cache snapshot directory for the source model."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache_dir, f"models--{SRC_MODEL.replace('/', '--')}")
    snap_dir = os.path.join(model_dir, "snapshots")
    if not os.path.exists(snap_dir):
        raise FileNotFoundError(f"Model not found in cache: {snap_dir}")
    snaps = os.listdir(snap_dir)
    if not snaps:
        raise FileNotFoundError(f"No snapshots in {snap_dir}")
    return os.path.join(snap_dir, snaps[0])


def should_quantize(key: str) -> bool:
    """Check if a weight key should be NVFP4 quantized."""
    for pattern in SKIP_QUANT_PATTERNS:
        if pattern in key:
            return False
    # Only quantize weight tensors (not biases, not scales)
    if not key.endswith(".weight"):
        return False
    return True


def quantize_weight(weight_bf16: torch.Tensor) -> dict:
    """Quantize a single BF16 weight tensor to NVFP4.

    Returns dict with weight (uint8 packed), weight_scale (uint8/FP8),
    weight_scale_2 (float32 global scale), input_scale (float32 placeholder).
    """
    weight_gpu = weight_bf16.cuda()
    global_scale = fp4_global_scale(weight_gpu).to(torch.float32)
    packed_fp4, block_scales = torch.ops.trtllm.fp4_quantize(
        weight_gpu, global_scale, BLOCK_SIZE, False
    )
    # Move to CPU immediately to free GPU memory
    packed_cpu = packed_fp4.cpu()
    scales_cpu = block_scales.cpu()
    gs_cpu = global_scale.cpu()
    del weight_gpu, packed_fp4, block_scales, global_scale
    torch.cuda.empty_cache()

    return {
        "weight": packed_cpu,                # uint8 packed FP4
        "weight_scale": scales_cpu,          # uint8 (FP8 E4M3 block scales)
        "weight_scale_2": gs_cpu.reshape(1), # float32 global scale
        "input_scale": torch.ones(1, dtype=torch.float32),  # placeholder
    }


def strip_prefix(key: str) -> str:
    """Strip 'model.language_model.' prefix from weight keys."""
    return key.replace("model.language_model.", "model.")


def build_config(src_snap: str) -> dict:
    """Build a flat qwen3_next config from the multimodal wrapper config."""
    with open(os.path.join(src_snap, "config.json")) as f:
        orig = json.load(f)

    text_cfg = orig.get("text_config", {})
    flat = dict(text_cfg)
    flat["model_type"] = "qwen3_next"
    flat["architectures"] = ["Qwen3NextForCausalLM"]
    flat["torch_dtype"] = "bfloat16"

    # Carry over top-level token IDs
    for k in ["bos_token_id", "eos_token_id", "pad_token_id"]:
        if k in orig and k not in flat:
            flat[k] = orig[k]

    # Add quantization_config (compressed-tensors style, for HF compatibility)
    flat["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "kv_cache_scheme": {"num_bits": 8, "type": "float", "dynamic": False},
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False},
                "input_activations": {"num_bits": 4, "type": "float", "group_size": 16, "dynamic": False},
                "targets": ["Linear"],
            }
        },
        "ignore": [f"model.layers.{i}.mlp.gate" for i in range(flat["num_hidden_layers"])] + ["lm_head"],
    }

    return flat


def build_hf_quant_config(num_layers: int) -> dict:
    """Build the hf_quant_config.json that TRT-LLM reads."""
    return {
        "producer": {"name": "modelopt", "version": "0.37.0"},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": "FP8",
            "group_size": 16,
            "exclude_modules": (
                [f"model.layers.{i}.mlp.gate" for i in range(num_layers)]
                + ["lm_head"]
            ),
        },
    }


def main():
    global SRC_SNAP
    t0 = time.time()

    if SRC_SNAP is None:
        SRC_SNAP = find_src_snapshot()
    print(f"Source: {SRC_SNAP}", flush=True)
    print(f"Output: {DST_DIR}", flush=True)

    # Clean previous output
    if os.path.exists(DST_DIR):
        shutil.rmtree(DST_DIR)
    os.makedirs(DST_DIR)

    # Load weight index
    with open(os.path.join(SRC_SNAP, "model.safetensors.index.json")) as f:
        src_index = json.load(f)
    src_weight_map = src_index["weight_map"]

    # Group keys by source shard file for efficient reading
    shard_to_keys = {}
    for key, shard_file in src_weight_map.items():
        shard_to_keys.setdefault(shard_file, []).append(key)

    # Process all shards
    dst_weight_map = {}
    dst_shard_data = {}
    dst_shard_bytes = 0
    dst_shard_idx = 0
    total_quantized = 0
    total_kept_bf16 = 0
    total_skipped = 0

    all_keys_ordered = []
    for shard_file in sorted(shard_to_keys.keys()):
        for key in shard_to_keys[shard_file]:
            all_keys_ordered.append((key, shard_file))

    for key, shard_file in all_keys_ordered:
        new_key = strip_prefix(key)

        if any(p in key for p in ["mtp.", "model.visual."]):
            total_skipped += 1
            continue

        shard_path = os.path.join(SRC_SNAP, shard_file)
        with safe_open(shard_path, framework="pt", device="cpu") as sf:
            tensor = sf.get_tensor(key)

        is_fused_expert = (
            "mlp.experts." in key
            and tensor.dim() == 3
            and ("gate_up_proj" in key or "down_proj" in key)
        )

        if is_fused_expert:
            num_experts = tensor.shape[0]
            fused_name = key.split("mlp.experts.")[-1]
            is_gate_up = fused_name == "gate_up_proj"
            layer_prefix = new_key.split("mlp.experts.")[0] + "mlp.experts."

            if is_gate_up:
                half = tensor.shape[1] // 2
                gate_block = tensor[:, :half, :].contiguous()
                up_block = tensor[:, half:, :].contiguous()
                del tensor
                gc.collect()
                proj_blocks = [("gate_proj", gate_block), ("up_proj", up_block)]
            else:
                proj_blocks = [(fused_name, tensor)]

            for proj_name, block_3d in proj_blocks:
                for expert_idx in range(num_experts):
                    expert_weight = block_3d[expert_idx].contiguous()
                    q = quantize_weight(expert_weight)
                    base = f"{layer_prefix}{expert_idx}.{proj_name}"
                    for suffix, value in q.items():
                        dst_shard_data[f"{base}.{suffix}"] = value
                        dst_shard_bytes += value.nelement() * value.element_size()
                    total_quantized += 1
                    del expert_weight, q

                    if dst_shard_bytes >= SHARD_SIZE_GB * 1e9:
                        shard_name = f"model-{dst_shard_idx:05d}-of-PLACEHOLDER.safetensors"
                        shard_out = os.path.join(DST_DIR, shard_name)
                        print(f"  Writing shard {dst_shard_idx}: {len(dst_shard_data)} tensors, "
                              f"{dst_shard_bytes/1e9:.1f} GB ({time.time()-t0:.0f}s)", flush=True)
                        save_file(dst_shard_data, shard_out)
                        for k in dst_shard_data:
                            dst_weight_map[k] = shard_name
                        dst_shard_data = {}
                        dst_shard_bytes = 0
                        dst_shard_idx += 1
                        gc.collect()
                        torch.cuda.empty_cache()
                del block_3d
            del proj_blocks
            gc.collect()
            try:
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("1")
            except PermissionError:
                pass

        elif should_quantize(key):
            q = quantize_weight(tensor)
            base_key = new_key.replace(".weight", "")
            for suffix, value in q.items():
                dst_shard_data[f"{base_key}.{suffix}"] = value
                dst_shard_bytes += value.nelement() * value.element_size()
            total_quantized += 1
            del tensor
        else:
            dst_shard_data[new_key] = tensor
            dst_shard_bytes += tensor.nelement() * tensor.element_size()
            total_kept_bf16 += 1

        if dst_shard_bytes >= SHARD_SIZE_GB * 1e9:
                    shard_name = f"model-{dst_shard_idx:05d}-of-PLACEHOLDER.safetensors"
                    shard_out = os.path.join(DST_DIR, shard_name)
                    print(f"  Writing shard {dst_shard_idx}: {len(dst_shard_data)} tensors, "
                          f"{dst_shard_bytes/1e9:.1f} GB ({time.time()-t0:.0f}s)", flush=True)
                    save_file(dst_shard_data, shard_out)
                    for k in dst_shard_data:
                        dst_weight_map[k] = shard_name
                    dst_shard_data = {}
                    dst_shard_bytes = 0
                    dst_shard_idx += 1
                    gc.collect()
                    torch.cuda.empty_cache()

        gc.collect()

    # Write remaining data
    if dst_shard_data:
        shard_name = f"model-{dst_shard_idx:05d}-of-PLACEHOLDER.safetensors"
        shard_out = os.path.join(DST_DIR, shard_name)
        print(f"  Writing shard {dst_shard_idx}: {len(dst_shard_data)} tensors, "
              f"{dst_shard_bytes/1e9:.1f} GB ({time.time()-t0:.0f}s)", flush=True)
        save_file(dst_shard_data, shard_out)
        for k in dst_shard_data:
            dst_weight_map[k] = shard_name
        dst_shard_idx += 1
        dst_shard_data = {}
        gc.collect()

    total_shards = dst_shard_idx
    for old_name in set(dst_weight_map.values()):
        new_name = old_name.replace("PLACEHOLDER", f"{total_shards:05d}")
        old_path = os.path.join(DST_DIR, old_name)
        new_path = os.path.join(DST_DIR, new_name)
        if old_path != new_path and os.path.exists(old_path):
            os.rename(old_path, new_path)
    dst_weight_map = {k: v.replace("PLACEHOLDER", f"{total_shards:05d}")
                      for k, v in dst_weight_map.items()}

    # Write index
    index = {"metadata": {"total_size": 0}, "weight_map": dst_weight_map}
    with open(os.path.join(DST_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Write config
    config = build_config(SRC_SNAP)
    with open(os.path.join(DST_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Write hf_quant_config.json
    hf_qc = build_hf_quant_config(config["num_hidden_layers"])
    with open(os.path.join(DST_DIR, "hf_quant_config.json"), "w") as f:
        json.dump(hf_qc, f, indent=2)

    # Copy tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                  "generation_config.json", "chat_template.jinja", "merges.txt"]:
        src = os.path.join(SRC_SNAP, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DST_DIR, fname))

    elapsed = time.time() - t0
    total_gb = sum(
        os.path.getsize(os.path.join(DST_DIR, f))
        for f in os.listdir(DST_DIR)
        if not f.startswith(".")
    ) / 1e9

    print(f"\n{'='*60}", flush=True)
    print(f"Done in {elapsed:.0f}s", flush=True)
    print(f"Quantized: {total_quantized} weights", flush=True)
    print(f"Kept BF16: {total_kept_bf16} tensors", flush=True)
    print(f"Skipped:   {total_skipped} tensors (MTP, vision)", flush=True)
    print(f"Shards:    {total_shards}", flush=True)
    print(f"Total:     {total_gb:.1f} GB", flush=True)
    print(f"Output:    {DST_DIR}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
