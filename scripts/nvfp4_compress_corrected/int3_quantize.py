"""BF16 -> INT3 artifact writer.

This script mirrors the corrected NVFP4 quantize flow but targets INT3.
It writes sidecar artifacts, not a directly loadable HF checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_utils import get_shard_order, is_expert_weight, is_linear_weight, load_index
from int3_utils import gptq_quantize_to_int3, rtn_quantize_to_int3


def _load_hessian(hessian_dir: Path, layer_name: str) -> torch.Tensor | None:
    match = re.search(r"layers\.(\d+)\.", layer_name)
    if not match:
        return None
    block_idx = int(match.group(1))
    hessian_file = hessian_dir / f"block_{block_idx:02d}.safetensors"
    if not hessian_file.exists():
        return None
    f = safe_open(str(hessian_file), framework="pt")
    if layer_name in list(f.keys()):
        return f.get_tensor(layer_name)
    base = layer_name.removesuffix(".weight")
    if base in list(f.keys()):
        return f.get_tensor(base)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="BF16 -> INT3 quantization")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--method", type=str, default="gptq-sym",
                        choices=["rtn-sym", "rtn-asym", "gptq-sym", "gptq-asym"])
    parser.add_argument("--hessian-dir", type=str, default=None)
    parser.add_argument("--scope", type=str, default="all",
                        choices=["all", "experts-only"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--activation-order", action="store_true", default=False)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    if args.method.startswith("gptq") and args.hessian_dir is None:
        parser.error("--hessian-dir required for gptq methods")

    model_dir = Path(args.model_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    hessian_dir = Path(args.hessian_dir) if args.hessian_dir else None

    if args.method in ("rtn-sym", "gptq-sym"):
        scheme = "sym"
    elif args.method in ("rtn-asym", "gptq-asym"):
        scheme = "asym"
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    should_quantize = is_linear_weight if args.scope == "all" else is_expert_weight

    weight_map = load_index(model_dir)
    shard_order = get_shard_order(weight_map)
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    manifest: dict[str, dict] = {}
    stats = {"rtn": 0, "gptq": 0, "skipped": 0}
    shard_end = args.shard_end if args.shard_end is not None else len(shard_order)
    print(f"Processing shards [{args.shard_start}, {shard_end}) of {len(shard_order)} on {args.device}")

    t0 = time.time()
    for si, shard_name in enumerate(shard_order):
        if si < args.shard_start or si >= shard_end:
            continue

        tensors = load_file(str(model_dir / shard_name), device="cpu")
        artifact_tensors: dict[str, torch.Tensor] = {}

        for key in shard_to_keys.get(shard_name, []):
            if key not in tensors:
                continue
            w = tensors[key]
            if w.dim() != 2 or not should_quantize(key):
                continue
            if w.shape[1] % args.group_size != 0:
                print(f"  SKIP (in_features % group_size != 0): {key} {w.shape}")
                stats["skipped"] += 1
                continue

            w_gpu = w.to(device)
            method_used = "rtn"
            if args.method.startswith("gptq"):
                if hessian_dir is None:
                    raise RuntimeError("hessian_dir is required for gptq methods")
                hessian = _load_hessian(hessian_dir, key.removesuffix(".weight"))
                if hessian is not None:
                    qweight, scale, zero, meta = gptq_quantize_to_int3(
                        w_gpu,
                        hessian.to(device),
                        scheme=scheme,
                        group_size=args.group_size,
                        activation_order=args.activation_order,
                    )
                    method_used = "gptq"
                else:
                    qweight, scale, zero = rtn_quantize_to_int3(
                        w_gpu, scheme=scheme, group_size=args.group_size)
                    meta = {"scheme": scheme, "group_size": args.group_size, "activation_order": False}
                    method_used = "rtn"
            else:
                qweight, scale, zero = rtn_quantize_to_int3(
                    w_gpu, scheme=scheme, group_size=args.group_size)
                meta = {"scheme": scheme, "group_size": args.group_size, "activation_order": False}

            base = key.removesuffix(".weight")
            artifact_tensors[f"{base}.weight_q"] = qweight.cpu()
            artifact_tensors[f"{base}.weight_scale"] = scale.cpu()
            if zero is not None:
                artifact_tensors[f"{base}.weight_zero"] = zero.cpu()

            manifest[key] = {
                "shape": list(w.shape),
                "shard": shard_name,
                "method": method_used,
            }
            stats[method_used] += 1
            print(f"  [{si+1}/{len(shard_order)}] [{method_used}] {key}: {w.shape}")

        if artifact_tensors:
            save_file(artifact_tensors, str(output_dir / shard_name))

    manifest_data = {
        "source": str(model_dir),
        "format": "int3",
        "method": args.method,
        "scheme": scheme,
        "scope": args.scope,
        "group_size": args.group_size,
        "activation_order": args.activation_order,
        "num_quantized": len(manifest),
        "stats": stats,
        "layers": manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest_data, indent=2))

    elapsed = time.time() - t0
    print(f"\nDone. {len(manifest)} layers quantized in {elapsed:.1f}s")
    print(f"  RTN: {stats['rtn']}, GPTQ: {stats['gptq']}, Skipped: {stats['skipped']}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
