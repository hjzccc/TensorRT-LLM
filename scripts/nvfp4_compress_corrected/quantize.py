"""Step 1: BF16 -> NVFP4 quantization.

Quantizes a BF16 HF checkpoint to NVFP4 format (per-16-block FP8 scale + E2M1 codes).
Output is stored as a side artifact (not HF-loadable directly).

Supports two methods:
  - rtn: simple round-to-nearest (fast, no calibration needed)
  - gptq: GPTQ error compensation (better quality, needs Hessian)

Usage:
    # RTN (simple)
    python quantize.py \
        --model-path /data/junzhou/models/Qwen3-30B-A3B \
        --output-path /data/junzhou/artifacts/nvfp4 \
        --method rtn

    # GPTQ (needs Hessian dir from calibration step)
    python quantize.py \
        --model-path /data/junzhou/models/Qwen3-30B-A3B \
        --output-path /data/junzhou/artifacts/nvfp4_gptq \
        --method gptq \
        --hessian-dir /data/junzhou/hessians
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_utils import is_expert_weight, is_linear_weight, load_index, get_shard_order
from nvfp4_utils import quantize_bf16_to_nvfp4


def _load_hessian(hessian_dir: Path, layer_name: str) -> torch.Tensor | None:
    """Try to load a Hessian matrix for the given layer name."""
    # Hessian files are per-block safetensors with layer names as keys
    import re
    match = re.search(r"layers\.(\d+)\.", layer_name)
    if not match:
        return None
    block_idx = int(match.group(1))
    hessian_file = hessian_dir / f"block_{block_idx:02d}.safetensors"
    if not hessian_file.exists():
        return None
    from safetensors import safe_open
    f = safe_open(str(hessian_file), framework="pt")
    if layer_name in [k for k in f.keys()]:
        return f.get_tensor(layer_name)
    # Try base name without .weight suffix
    base = layer_name.removesuffix(".weight")
    if base in [k for k in f.keys()]:
        return f.get_tensor(base)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 -> NVFP4 quantization")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to BF16 HF model directory")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to write NVFP4 artifact directory")
    parser.add_argument("--method", type=str, default="rtn",
                        choices=["rtn", "gptq"],
                        help="Quantization method")
    parser.add_argument("--hessian-dir", type=str, default=None,
                        help="Path to Hessian directory (required for gptq)")
    parser.add_argument("--scope", type=str, default="all",
                        choices=["all", "experts-only"],
                        help="Which layers to quantize")
    parser.add_argument("--activation-order", action="store_true", default=False,
                        help="Process columns by Hessian diagonal importance (MR-GPTQ Ingredient 2)")
    parser.add_argument("--hadamard-group-size", type=int, default=None,
                        help="Block-local Hadamard rotation group size (e.g. 128). "
                             "Only used with --method gptq. Default: None (no rotation)")
    parser.add_argument("--hadamard-mode", type=str, default="undo",
                        choices=["undo", "rotated"],
                        help="'undo' = rotate back to original basis (TRT-LLM compatible). "
                             "'rotated' = keep rotated basis (MR-GPTQ style, needs activation rotation at eval)")
    parser.add_argument("--shard-start", type=int, default=0,
                        help="First shard index to process (0-based). For parallel runs across GPUs.")
    parser.add_argument("--shard-end", type=int, default=None,
                        help="Last shard index (exclusive). None = all remaining shards.")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.method == "gptq" and args.hessian_dir is None:
        parser.error("--hessian-dir required for gptq method")
    if args.hadamard_group_size is not None and args.method != "gptq":
        parser.error("--hadamard-group-size only works with --method gptq")
    if args.activation_order and args.method != "gptq":
        parser.error("--activation-order only works with --method gptq")

    model_dir = Path(args.model_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    hessian_dir = Path(args.hessian_dir) if args.hessian_dir else None

    weight_map = load_index(model_dir)
    shard_order = get_shard_order(weight_map)

    should_quantize = is_linear_weight if args.scope == "all" else is_expert_weight

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    # Import GPTQ if needed
    gptq_quantize_fn = None
    if args.method == "gptq":
        from gptq import gptq_quantize_to_nvfp4
        gptq_quantize_fn = gptq_quantize_to_nvfp4

    manifest: dict[str, dict] = {}
    stats = {"rtn": 0, "gptq": 0, "skipped": 0}
    t0 = time.time()

    shard_start = args.shard_start
    shard_end = args.shard_end if args.shard_end is not None else len(shard_order)
    print(f"Processing shards [{shard_start}, {shard_end}) of {len(shard_order)} on {args.device}")

    for si, shard_name in enumerate(shard_order):
        if si < shard_start or si >= shard_end:
            continue
        input_path = model_dir / shard_name
        tensors = load_file(str(input_path), device="cpu")
        artifact_tensors: dict[str, torch.Tensor] = {}

        for key in shard_to_keys.get(shard_name, []):
            if key not in tensors:
                continue
            w = tensors[key]
            if w.dim() != 2 or not should_quantize(key):
                continue
            if w.shape[1] % 16 != 0:
                print(f"  SKIP (in_features % 16 != 0): {key} {w.shape}")
                stats["skipped"] += 1
                continue

            w_gpu = w.to(device)
            method_used = "rtn"

            if args.method == "gptq" and hessian_dir is not None:
                base = key.removesuffix(".weight")
                hessian = _load_hessian(hessian_dir, base)
                if hessian is not None:
                    packed, block_scales, global_scale, _meta = gptq_quantize_fn(
                        w_gpu, hessian.to(device),
                        activation_order=args.activation_order,
                        hadamard_group_size=args.hadamard_group_size,
                        hadamard_mode=args.hadamard_mode)
                    method_used = "gptq"
                else:
                    # Fall back to RTN if no Hessian available
                    packed, block_scales, global_scale = quantize_bf16_to_nvfp4(w_gpu)
                    method_used = "rtn"
            else:
                packed, block_scales, global_scale = quantize_bf16_to_nvfp4(w_gpu)

            base = key.removesuffix(".weight")
            artifact_tensors[f"{base}.weight_packed"] = packed.cpu()
            artifact_tensors[f"{base}.weight_scale"] = block_scales.cpu()
            artifact_tensors[f"{base}.weight_global_scale"] = global_scale.cpu()

            manifest[key] = {
                "shape": list(w.shape),
                "shard": shard_name,
                "method": method_used,
            }
            stats[method_used] += 1
            print(f"  [{si+1}/{len(shard_order)}] [{method_used}] {key}: {w.shape}")

        if artifact_tensors:
            save_file(artifact_tensors, str(output_dir / shard_name))

    manifest_path = output_dir / "manifest.json"
    manifest_data = {
        "source": str(model_dir),
        "method": args.method,
        "scope": args.scope,
        "activation_order": args.activation_order,
        "hadamard_group_size": args.hadamard_group_size,
        "hadamard_mode": args.hadamard_mode if args.hadamard_group_size else "none",
        "num_quantized": len(manifest),
        "stats": stats,
        "layers": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2))

    elapsed = time.time() - t0
    print(f"\nDone. {len(manifest)} layers quantized in {elapsed:.1f}s")
    print(f"  RTN: {stats['rtn']}, GPTQ: {stats['gptq']}, Skipped: {stats['skipped']}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
