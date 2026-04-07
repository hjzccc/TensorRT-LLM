"""Step 2: NVFP4 -> sub-NVFP4 codebook compression.

Takes NVFP4 artifacts from quantize.py and applies per-block codebook
compression: each 16-element block is restricted to the best 4 of 15
unique E2M1 values. The per-block scale is preserved unchanged.

Can also take a vanilla NVFP4 checkpoint (e.g. nvidia/Qwen3-30B-A3B-NVFP4)
directly by specifying --input-format=checkpoint.

Usage:
    python compress.py \
        --input-path /data/junzhou/artifacts/nvfp4 \
        --output-path /data/junzhou/artifacts/nvfp4_compressed \
        --input-format artifact     # or "checkpoint"
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
from codebook import build_all_codebooks, compress_packed_tensor
from checkpoint_utils import is_expert_weight, is_linear_weight
from nvfp4_utils import unpack_fp4, nibbles_to_floats


def compress_from_artifacts(input_dir: Path, output_dir: Path,
                            device: torch.device, include_zero: bool,
                            scope: str) -> None:
    """Compress NVFP4 artifacts produced by quantize.py."""
    manifest = json.loads((input_dir / "manifest.json").read_text())
    codebooks = build_all_codebooks(include_zero=include_zero).to(device)
    print(f"Using {codebooks.shape[0]} codebooks (include_zero={include_zero})")
    should_compress = is_linear_weight if scope == "all" else is_expert_weight

    output_dir.mkdir(parents=True, exist_ok=True)
    shards = sorted(input_dir.glob("model-*.safetensors"))
    t0 = time.time()

    compressed_layers: dict[str, dict] = {}

    for si, shard_path in enumerate(shards):
        tensors = load_file(str(shard_path), device="cpu")

        # Group by layer base name
        bases: set[str] = set()
        for k in tensors:
            if k.endswith(".weight_packed"):
                bases.add(k.removesuffix(".weight_packed"))

        out_tensors: dict[str, torch.Tensor] = {}
        for base in sorted(bases):
            weight_key = f"{base}.weight"
            if not should_compress(weight_key):
                continue

            packed = tensors[f"{base}.weight_packed"].to(device)
            block_scales = tensors[f"{base}.weight_scale"].to(device)
            global_scale = tensors[f"{base}.weight_global_scale"].to(device)

            c_packed, c_scales, c_gscale = compress_packed_tensor(
                packed, block_scales, global_scale,
                codebooks=codebooks, include_zero=include_zero,
            )

            out_tensors[f"{base}.weight_packed"] = c_packed.cpu()
            out_tensors[f"{base}.weight_scale"] = c_scales.cpu()
            out_tensors[f"{base}.weight_global_scale"] = c_gscale.cpu()
            if weight_key in manifest.get("layers", {}):
                compressed_layers[weight_key] = manifest["layers"][weight_key]
            print(f"  [{si+1}/{len(shards)}] {base}")

        if out_tensors:
            save_file(out_tensors, str(output_dir / shard_path.name))

    # Copy manifest
    manifest["compressed"] = True
    manifest["compression_scope"] = scope
    manifest["include_zero"] = include_zero
    manifest["num_codebooks"] = int(codebooks.shape[0])
    manifest["base_artifact_path"] = str(input_dir)
    manifest["layers"] = compressed_layers
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    print(f"\nDone. {len(compressed_layers)} layers compressed in {elapsed:.1f}s")


def compress_from_checkpoint(input_dir: Path, output_dir: Path,
                             device: torch.device, include_zero: bool,
                             scope: str) -> None:
    """Compress an existing NVFP4 checkpoint (e.g. nvidia's) directly.

    Reads packed uint8 weights + FP8 scales from the checkpoint shards,
    applies codebook compression, writes compressed artifacts.
    """
    from checkpoint_utils import load_index, get_shard_order, is_expert_weight, is_linear_weight

    weight_map = load_index(input_dir)
    shard_order = get_shard_order(weight_map)
    codebooks = build_all_codebooks(include_zero=include_zero).to(device)
    should_compress = is_linear_weight if scope == "all" else is_expert_weight

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    t0 = time.time()

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    for si, shard_name in enumerate(shard_order):
        tensors = load_file(str(input_dir / shard_name), device="cpu")
        out_tensors: dict[str, torch.Tensor] = {}

        for key in shard_to_keys.get(shard_name, []):
            if key not in tensors:
                continue
            t = tensors[key]
            if t.dtype != torch.uint8 or not should_compress(key):
                continue

            base = key.removesuffix(".weight")
            scale_key = f"{base}.weight_scale"
            gscale_key = f"{base}.weight_scale_2"
            if gscale_key not in tensors:
                gscale_key = f"{base}.weight_global_scale"

            scale = tensors.get(scale_key)
            gscale = tensors.get(gscale_key, torch.tensor([1.0]))
            if scale is None:
                continue

            c_packed, c_scales, c_gscale = compress_packed_tensor(
                t.to(device), scale.to(device), gscale.to(device),
                codebooks=codebooks, include_zero=include_zero,
            )

            out_tensors[f"{base}.weight_packed"] = c_packed.cpu()
            out_tensors[f"{base}.weight_scale"] = c_scales.cpu()
            out_tensors[f"{base}.weight_global_scale"] = c_gscale.cpu()
            manifest[key] = {"shape": list(t.shape), "shard": shard_name}
            print(f"  [{si+1}/{len(shard_order)}] {key}")

        if out_tensors:
            save_file(out_tensors, str(output_dir / shard_name))

    (output_dir / "manifest.json").write_text(json.dumps({
        "source": str(input_dir),
        "scope": scope,
        "num_quantized": len(manifest),
        "compressed": True,
        "include_zero": include_zero,
        "num_codebooks": int(codebooks.shape[0]),
        "layers": manifest,
    }, indent=2))

    elapsed = time.time() - t0
    print(f"\nDone. {len(manifest)} layers compressed in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="NVFP4 -> sub-NVFP4 codebook compression")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--input-format", type=str, default="artifact",
                        choices=["artifact", "checkpoint"],
                        help="'artifact' = from quantize.py output, "
                             "'checkpoint' = from an existing NVFP4 HF checkpoint")
    parser.add_argument("--scope", type=str, default="experts-only",
                        choices=["all", "experts-only"])
    parser.add_argument("--include-zero", action="store_true", default=True,
                        help="Force 0.0 in every codebook (default: True)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)

    if args.input_format == "artifact":
        compress_from_artifacts(input_dir, output_dir, device, args.include_zero, args.scope)
    else:
        compress_from_checkpoint(input_dir, output_dir, device, args.include_zero, args.scope)


if __name__ == "__main__":
    main()
