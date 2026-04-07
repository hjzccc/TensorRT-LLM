"""INT3 artifacts -> BF16 HF checkpoint rebuild."""

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
from int3_utils import dequantize_int3


def main() -> None:
    parser = argparse.ArgumentParser(description="Dequantize INT3 artifacts to BF16 HF checkpoint")
    parser.add_argument("--original-model", type=str, required=True)
    parser.add_argument("--artifact-path", type=str, required=True)
    parser.add_argument("--base-artifact-path", type=str, default=None,
                        help="Optional base INT3 artifact dir for subset rebuilds")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    original_dir = Path(args.original_model)
    artifact_dir = Path(args.artifact_path)
    base_artifact_dir = Path(args.base_artifact_path) if args.base_artifact_path else None
    output_dir = Path(args.output_path)
    device = torch.device(args.device)

    manifest = json.loads((artifact_dir / 'manifest.json').read_text())
    quantized_layers = set(manifest.get('layers', {}).keys())
    scheme = manifest['scheme']
    group_size = int(manifest['group_size'])

    print(f"Artifact source: {manifest.get('source')}")
    print(f"Layers to dequantize: {len(quantized_layers)}")
    print(f"Scheme: {scheme}, group_size: {group_size}")
    if base_artifact_dir is not None:
        print(f"Base artifacts: {base_artifact_dir}")

    artifacts = {}
    for shard_path in sorted(artifact_dir.glob('model-*.safetensors')):
        artifacts.update(load_file(str(shard_path), device='cpu'))
    print(f"  {len(artifacts)} artifact tensors loaded")

    base_artifacts = {}
    if base_artifact_dir is not None:
        for shard_path in sorted(base_artifact_dir.glob('model-*.safetensors')):
            base_artifacts.update(load_file(str(shard_path), device='cpu'))
        print(f"  {len(base_artifacts)} base artifact tensors loaded")

    def transform_fn(key: str, original: torch.Tensor) -> torch.Tensor | None:
        if key not in quantized_layers:
            return None
        base = key.removesuffix('.weight')
        q_key = f'{base}.weight_q'
        scale_key = f'{base}.weight_scale'
        zero_key = f'{base}.weight_zero'

        source = artifacts
        if q_key not in source:
            if base_artifact_dir is not None and q_key in base_artifacts:
                source = base_artifacts
            else:
                print(f"  WARNING: artifact missing for {key}, keeping original")
                return None

        qweight = source[q_key].to(device)
        scale = source[scale_key].to(device)
        zero = source[zero_key].to(device) if zero_key in source else None
        return dequantize_int3(qweight, scale, zero, scheme, group_size)

    t0 = time.time()
    rebuild_checkpoint(original_dir, output_dir, transform_fn, device)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
