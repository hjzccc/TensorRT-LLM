"""Extract artifact-format NVFP4 tensors from an existing NVFP4 HF checkpoint.

This is used for checkpoints like nvidia/Qwen3-30B-A3B-NVFP4 that already store
packed FP4 weights plus scales. The output artifact format matches quantize.py:

  <base>.weight_packed
  <base>.weight_scale
  <base>.weight_global_scale

The purpose is to let the rest of the corrected pipeline (compression,
dequantize, evaluation) treat "official vanilla NVFP4" and "our generated
NVFP4 artifacts" the same way.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from checkpoint_utils import get_shard_order, is_expert_weight, is_linear_weight, load_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract NVFP4 artifact format from an existing checkpoint")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                        help="Path to an existing NVFP4 HF checkpoint")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to write artifact-format files")
    parser.add_argument("--scope", type=str, default="all",
                        choices=["all", "experts-only"],
                        help="Which layers to extract")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_path)
    out_dir = Path(args.output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    should_take = is_linear_weight if args.scope == "all" else is_expert_weight

    weight_map = load_index(ckpt_dir)
    shard_order = get_shard_order(weight_map)
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    manifest: dict[str, dict] = {}
    t0 = time.time()

    for si, shard_name in enumerate(shard_order, 1):
        tensors = load_file(str(ckpt_dir / shard_name), device="cpu")
        out_tensors: dict[str, torch.Tensor] = {}

        for key in shard_to_keys.get(shard_name, []):
            if key not in tensors or not key.endswith(".weight"):
                continue
            if not should_take(key):
                continue

            weight = tensors[key]
            if weight.dtype != torch.uint8:
                continue

            base = key.removesuffix(".weight")
            scale_key = f"{base}.weight_scale"
            gscale_key = f"{base}.weight_scale_2"
            if scale_key not in tensors or gscale_key not in tensors:
                continue

            out_tensors[f"{base}.weight_packed"] = weight
            out_tensors[f"{base}.weight_scale"] = tensors[scale_key]
            out_tensors[f"{base}.weight_global_scale"] = tensors[gscale_key]
            manifest[key] = {"shape": list(weight.shape), "shard": shard_name}

        if out_tensors:
            save_file(out_tensors, str(out_dir / shard_name))
            print(f"[{si}/{len(shard_order)}] {shard_name}: {len(out_tensors)//3} layers extracted")

    (out_dir / "manifest.json").write_text(json.dumps({
        "source": str(ckpt_dir),
        "method": "vanilla_nvfp4_checkpoint",
        "scope": args.scope,
        "hadamard_mode": "none",
        "num_quantized": len(manifest),
        "layers": manifest,
    }, indent=2))

    elapsed = time.time() - t0
    print(f"Done. Extracted {len(manifest)} layers in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
