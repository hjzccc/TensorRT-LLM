#!/usr/bin/env python3
"""Apply Hadamard rotation folded into weights, then quantize to NVFP4.

QuaRot-style approach: rotate the entire residual stream by folding Hadamard
into all weight matrices and norm scales. No runtime changes needed.
"""
import json, time, gc, shutil, os, math
import torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

FP4_E2M1_MAX = 6.0
FP8_E4M3_MAX = 448.0
NVFP_GROUP_SIZE = 16
HADAMARD_GROUP_SIZE = 16


def cast_to_fp4(x):
    sign = torch.sign(x)
    xa = torch.abs(x)
    r = torch.zeros_like(xa)
    r[(xa > 0.25) & (xa < 0.75)] = 0.5
    r[(xa >= 0.75) & (xa <= 1.25)] = 1.0
    r[(xa > 1.25) & (xa < 1.75)] = 1.5
    r[(xa >= 1.75) & (xa <= 2.5)] = 2.0
    r[(xa > 2.5) & (xa < 3.5)] = 3.0
    r[(xa >= 3.5) & (xa <= 5.0)] = 4.0
    r[xa > 5.0] = 6.0
    return r * sign


FP4_GRID_SORTED = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_BITPACKING_PERM = [15, 14, 13, 12, 11, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7]


def pack_fp4_to_uint8(x):
    grid = torch.tensor(FP4_GRID_SORTED, device=x.device)
    perm = torch.tensor(FP4_BITPACKING_PERM, device=x.device)
    grid_ids = torch.bucketize(x, grid)
    lo = (grid_ids - 1).clamp(min=0, max=15)
    hi = grid_ids.clamp(min=0, max=15)
    g_lo, g_hi = grid[lo], grid[hi]
    pick_hi = (g_hi - x < x - g_lo) | ((g_hi - x == x - g_lo) & (perm[hi] % 2 == 0))
    q = torch.where(pick_hi, perm[hi], perm[lo])
    return (q[:, 1::2] << 4 | q[:, ::2]).to(torch.uint8)


def get_block_hadamard(hidden_size, group_size):
    from scipy.linalg import hadamard as scipy_hadamard
    h_block = torch.tensor(scipy_hadamard(group_size), dtype=torch.float32) / math.sqrt(group_size)
    return torch.block_diag(*[h_block] * (hidden_size // group_size))


def quantize_to_nvfp4(weight, global_scale=None):
    out_f, in_f = weight.shape
    if global_scale is None:
        global_scale = ((FP8_E4M3_MAX * FP4_E2M1_MAX) / weight.abs().max().clamp(min=1e-12)).to(torch.float32)
    scaled = weight * global_scale
    blocks = scaled.reshape(-1, NVFP_GROUP_SIZE)
    block_max = blocks.abs().amax(dim=1, keepdim=True)
    block_scale = (block_max / FP4_E2M1_MAX).clamp(min=1e-12)
    fp4_vals = cast_to_fp4(blocks / block_scale)
    packed = pack_fp4_to_uint8(fp4_vals.reshape(out_f, in_f))
    block_scale_2d = block_scale.reshape(out_f, in_f // NVFP_GROUP_SIZE).to(torch.float8_e4m3fn)
    return packed, block_scale_2d, global_scale.reshape(1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-path", type=str, default="/tmp/qwen3_30b_bf16")
    parser.add_argument("--ref-path", type=str, default="/root/.cache/huggingface/hub/models--nvidia--Qwen3-30B-A3B-NVFP4/snapshots/2538ded2a4edb247b4d2b4a8ba24e44bd4c017c3")
    parser.add_argument("--output", type=str, default="/tmp/hadamard_fold_nvfp4_30b")
    parser.add_argument("--group-size", type=int, default=HADAMARD_GROUP_SIZE)
    parser.add_argument("--bf16-only", action="store_true")
    parser.add_argument("--bf16-only-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    BF16_PATH = Path(args.bf16_path)
    NVFP4_REF = Path(args.ref_path)
    OUT = Path(args.output)
    save_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.bf16_only_dtype]

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                if free_mem < 4 * 1024**3:
                    device = "cpu"
            except Exception:
                pass
    else:
        device = args.device
    dev = torch.device(device)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()

    bf16_wm = json.loads((BF16_PATH / "model.safetensors.index.json").read_text())["weight_map"]
    nvfp4_wm = json.loads((NVFP4_REF / "model.safetensors.index.json").read_text())["weight_map"]
    nvfp4_qcfg = json.loads((NVFP4_REF / "hf_quant_config.json").read_text())
    cfg = json.loads((BF16_PATH / "config.json").read_text())

    num_layers = cfg["num_hidden_layers"]
    hidden_size = cfg["hidden_size"]

    quant_bases = set()
    for k in nvfp4_wm:
        if k.endswith(".weight_scale"):
            quant_bases.add(k.removesuffix(".weight_scale"))
    if args.bf16_only:
        quant_bases = set()

    H = get_block_hadamard(hidden_size, args.group_size).to(dev)
    print(f"hidden={hidden_size}, layers={num_layers}, quant={len(quant_bases)}, device={dev}, H={H.shape}", flush=True)

    # Load all norm weights
    norms = {}
    for key in bf16_wm:
        if "layernorm.weight" in key or key == "model.norm.weight":
            with safe_open(str(BF16_PATH / bf16_wm[key]), framework="pt", device="cpu") as sf:
                norms[key] = sf.get_tensor(key).float()
    print(f"Loaded {len(norms)} norms", flush=True)

    t0 = time.time()
    output_wm = {}
    shard_data = {}
    shard_idx = 0
    current_shard_size = 0
    MAX_SHARD = 5 * 1024**3
    processed = 0
    processed_keys = set()

    def flush():
        nonlocal shard_data, shard_idx, current_shard_size
        if not shard_data:
            return
        sn = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        save_file({k: v.clone().contiguous() for k, v in shard_data.items()}, str(OUT / sn))
        for k in shard_data:
            output_wm[k] = sn
        shard_idx += 1
        shard_data = {}
        current_shard_size = 0
        gc.collect()

    def add(key, tensor):
        nonlocal current_shard_size
        if args.bf16_only and tensor.is_floating_point():
            tensor = tensor.to(save_dtype)
        sz = tensor.numel() * tensor.element_size()
        if current_shard_size + sz > MAX_SHARD:
            flush()
        shard_data[key] = tensor
        current_shard_size += sz

    def add_q(base, packed, bs, gs):
        add(f"{base}.weight", packed)
        add(f"{base}.weight_scale", bs)
        add(f"{base}.weight_scale_2", gs)
        add(f"{base}.input_scale", torch.ones(1, dtype=torch.float32))

    def load_w(key):
        with safe_open(str(BF16_PATH / bf16_wm[key]), framework="pt", device="cpu") as sf:
            return sf.get_tensor(key).float()

    def rotate_input(W, g):
        return (W.to(dev) * g.to(dev).unsqueeze(0) @ H.T).cpu()

    def rotate_output(W):
        return (H @ W.to(dev)).cpu()

    # Embed: rotate residual
    embed = load_w("model.embed_tokens.weight")
    add("model.embed_tokens.weight", (embed.to(dev) @ H.T).cpu())
    processed_keys.add("model.embed_tokens.weight")
    del embed
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # lm_head: absorb final norm + rotation
    g_f = norms["model.norm.weight"]
    lm = load_w("lm_head.weight")
    add("lm_head.weight", (lm.to(dev) * g_f.to(dev).unsqueeze(0) @ H.T).cpu())
    processed_keys.add("lm_head.weight")
    del lm
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Final norm -> ones
    add("model.norm.weight", torch.ones_like(g_f))
    processed_keys.add("model.norm.weight")
    print("Embed + lm_head done", flush=True)

    # Per-layer processing
    for li in range(num_layers):
        pfx = f"model.layers.{li}"
        g1 = norms[f"{pfx}.input_layernorm.weight"]
        g2 = norms[f"{pfx}.post_attention_layernorm.weight"]

        # Norms -> ones
        add(f"{pfx}.input_layernorm.weight", torch.ones_like(g1))
        processed_keys.add(f"{pfx}.input_layernorm.weight")
        add(f"{pfx}.post_attention_layernorm.weight", torch.ones_like(g2))
        processed_keys.add(f"{pfx}.post_attention_layernorm.weight")

        # QKV: W_new = W @ diag(g1) @ H^T, shared global scale
        qkv_ws = {}
        qkv_amax = 0.0
        for proj in ["q_proj", "k_proj", "v_proj"]:
            key = f"{pfx}.self_attn.{proj}.weight"
            if key not in bf16_wm:
                continue
            W_new = rotate_input(load_w(key), g1)
            base = key.removesuffix(".weight")
            qkv_ws[base] = W_new
            qkv_amax = max(qkv_amax, W_new.abs().max().item())
            processed_keys.add(key)

        if qkv_ws:
            shared_gs = torch.tensor((FP8_E4M3_MAX * FP4_E2M1_MAX) / max(qkv_amax, 1e-12), dtype=torch.float32)
            for base, W_new in qkv_ws.items():
                if base in quant_bases:
                    p, bs, gs = quantize_to_nvfp4(W_new, global_scale=shared_gs)
                    add_q(base, p, bs, gs)
                    processed += 1
                else:
                    add(f"{base}.weight", W_new)
            del qkv_ws
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        # O: W_new = H @ W
        o_key = f"{pfx}.self_attn.o_proj.weight"
        if o_key in bf16_wm:
            W_new = rotate_output(load_w(o_key))
            base = o_key.removesuffix(".weight")
            if base in quant_bases:
                p, bs, gs = quantize_to_nvfp4(W_new)
                add_q(base, p, bs, gs)
                processed += 1
            else:
                add(o_key, W_new)
            processed_keys.add(o_key)
            del W_new
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        # MLP: gate+up get rotate_input, down gets rotate_output
        # Process all expert + shared expert keys for this layer
        mlp_keys = sorted(k for k in bf16_wm if k.startswith(f"{pfx}.mlp.") and k.endswith(".weight") and k not in processed_keys)

        gate_up_cache = {}
        for key in mlp_keys:
            base = key.removesuffix(".weight")
            is_input = (
                ".mlp.gate.weight" in key
                or ".mlp.shared_expert_gate.weight" in key
                or ".gate_proj." in key
                or ".up_proj." in key
            )
            is_output = ".down_proj." in key

            W = load_w(key)
            if is_input:
                W_new = rotate_input(W, g2)
            elif is_output:
                W_new = rotate_output(W)
            else:
                W_new = W

            if base in quant_bases:
                if ".gate_proj." in key:
                    gate_up_cache[base] = W_new
                elif ".up_proj." in key:
                    gate_base = base.replace(".up_proj", ".gate_proj")
                    if gate_base in gate_up_cache:
                        combined_amax = max(gate_up_cache[gate_base].abs().max().item(), W_new.abs().max().item())
                        shared_gs = torch.tensor((FP8_E4M3_MAX * FP4_E2M1_MAX) / max(combined_amax, 1e-12), dtype=torch.float32)
                        for gu_base, gu_w in [(gate_base, gate_up_cache.pop(gate_base)), (base, W_new)]:
                            p, bs, gs = quantize_to_nvfp4(gu_w, global_scale=shared_gs)
                            add_q(gu_base, p, bs, gs)
                            processed += 1
                    else:
                        p, bs, gs = quantize_to_nvfp4(W_new)
                        add_q(base, p, bs, gs)
                        processed += 1
                else:
                    p, bs, gs = quantize_to_nvfp4(W_new)
                    add_q(base, p, bs, gs)
                    processed += 1
            else:
                add(key, W_new)

            processed_keys.add(key)
            del W, W_new
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        # Flush any unpaired gate weights
        for base, W_new in gate_up_cache.items():
            p, bs, gs = quantize_to_nvfp4(W_new)
            add_q(base, p, bs, gs)
            processed += 1
        gate_up_cache.clear()

        if (li + 1) % 4 == 0:
            print(f"  Layer {li+1}/{num_layers}, {processed} quant ({time.time()-t0:.0f}s)", flush=True)

    # Copy remaining unprocessed keys
    for bf16_shard in sorted(set(bf16_wm.values())):
        remaining = [k for k, v in bf16_wm.items() if v == bf16_shard and k not in processed_keys]
        if not remaining:
            continue
        with safe_open(str(BF16_PATH / bf16_shard), framework="pt", device="cpu") as sf:
            for key in remaining:
                add(key, sf.get_tensor(key))
                processed_keys.add(key)

    flush()

    # Fix shard names
    total = shard_idx
    for key in list(output_wm.keys()):
        output_wm[key] = output_wm[key].replace("PLACEHOLDER", f"{total:05d}")
    for f in OUT.glob("model-*-of-PLACEHOLDER.safetensors"):
        f.rename(OUT / f.name.replace("PLACEHOLDER", f"{total:05d}"))

    json.dump({"metadata": {}, "weight_map": output_wm}, (OUT / "model.safetensors.index.json").open("w"), indent=2)

    aux_src = BF16_PATH if args.bf16_only else NVFP4_REF
    for fn in os.listdir(str(aux_src)):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = aux_src / fn
        dst = OUT / fn
        if not dst.exists() and (src.is_file() or src.is_symlink()):
            shutil.copy2(str(src.resolve()), str(dst))

    print(f"Done. {processed} quantized, {total} shards, {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
