"""Step 3+4: NVFP4 artifacts -> BF16 HF checkpoint for TRT-LLM evaluation.

Takes NVFP4 (or compressed sub-NVFP4) artifacts and rebuilds a clean
HF-compatible BF16 checkpoint by:
  1. Loading original HF model shard-by-shard
  2. Replacing quantized tensors with dequantized BF16 values
  3. Saving with original shard structure

The output is a standard HF checkpoint that TRT-LLM can load directly.

Usage:
    python dequantize.py \
        --original-model /data/junzhou/models/Qwen3-30B-A3B \
        --artifact-path /data/junzhou/artifacts/nvfp4_compressed \
        --output-path /data/junzhou/models/Qwen3-30B-A3B-compressed-eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_utils import rebuild_checkpoint
from nvfp4_utils import dequantize_nvfp4_to_bf16


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dequantize NVFP4 artifacts to BF16 HF checkpoint")
    parser.add_argument("--original-model", type=str, required=True,
                        help="Path to original BF16 HF model (for shard structure)")
    parser.add_argument("--artifact-path", type=str, required=True,
                        help="Path to NVFP4 artifact directory (from quantize.py or compress.py)")
    parser.add_argument("--base-artifact-path", type=str, default=None,
                        help="Optional base NVFP4 artifact directory. If provided, tensors not present in --artifact-path are loaded from here. This is required when compressed artifacts only cover a subset of layers (e.g. experts-only compression on top of all-layer NVFP4).")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path to write BF16 HF checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    original_dir = Path(args.original_model)
    artifact_dir = Path(args.artifact_path)
    base_artifact_dir = Path(args.base_artifact_path) if args.base_artifact_path else None
    output_dir = Path(args.output_path)
    device = torch.device(args.device)

    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    quantized_layers = set(manifest.get("layers", {}).keys())
    had_mode = manifest.get("hadamard_mode", "none")
    had_gs = manifest.get("hadamard_group_size", None)

    print(f"Artifact source: {manifest.get('source')}")
    print(f"Layers to dequantize: {len(quantized_layers)}")
    print(f"Compressed: {manifest.get('compressed', False)}")
    print(f"Hadamard mode: {had_mode}, group_size: {had_gs}")
    if base_artifact_dir is not None:
        print(f"Base artifacts: {base_artifact_dir}")

    # Only rotated-mode artifacts stay in the rotated basis.
    # Undo-mode artifacts were already rotated back to the original basis inside
    # gptq.py before being re-quantized and saved, so applying Hadamard again
    # here would be incorrect.
    need_inverse_hadamard = (had_mode == "rotated" and had_gs is not None)
    if need_inverse_hadamard:
        from nvfp4_utils import hadamard_rotate
        print(f"  Will apply inverse Hadamard (group_size={had_gs}) after dequant")

    print("Loading artifact tensors...")
    artifacts: dict[str, torch.Tensor] = {}
    for shard_path in sorted(artifact_dir.glob("model-*.safetensors")):
        tensors = load_file(str(shard_path), device="cpu")
        artifacts.update(tensors)
    print(f"  {len(artifacts)} artifact tensors loaded")

    base_artifacts: dict[str, torch.Tensor] = {}
    if base_artifact_dir is not None:
        print("Loading base artifact tensors...")
        for shard_path in sorted(base_artifact_dir.glob("model-*.safetensors")):
            tensors = load_file(str(shard_path), device="cpu")
            base_artifacts.update(tensors)
        print(f"  {len(base_artifacts)} base artifact tensors loaded")

    def transform_fn(key: str, original: torch.Tensor) -> torch.Tensor | None:
        if key not in quantized_layers:
            return None

        base = key.removesuffix(".weight")
        packed_key = f"{base}.weight_packed"
        scale_key = f"{base}.weight_scale"
        gscale_key = f"{base}.weight_global_scale"

        tensor_source = artifacts
        if packed_key not in tensor_source:
            if base_artifact_dir is not None and packed_key in base_artifacts:
                tensor_source = base_artifacts
            else:
                print(f"  WARNING: artifact missing for {key}, keeping original")
                return None

        packed = tensor_source[packed_key].to(device)
        block_scales = tensor_source[scale_key].to(device)
        global_scale = tensor_source[gscale_key].to(device)

        dequantized = dequantize_nvfp4_to_bf16(packed, block_scales, global_scale)

        if need_inverse_hadamard and dequantized.dim() == 2 and dequantized.shape[-1] % had_gs == 0:
            dequantized = hadamard_rotate(dequantized.float(), had_gs).to(torch.bfloat16)

        return dequantized

    t0 = time.time()
    rebuild_checkpoint(original_dir, output_dir, transform_fn, device)

    # Preserve artifact metadata so the evaluator can decide whether runtime
    # activation rotation is required.
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
