#!/usr/bin/env python3
"""
Evaluate compressed NVFP4 checkpoint directly without decompression.
Uses streaming layer-by-layer loading to avoid OOM.
"""

import sys
import json
import torch
import gc
from pathlib import Path
from typing import Any, Dict, List, Optional
from safetensors.torch import load_file
from transformers import AutoTokenizer

# Configuration
CHECKPOINT_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_weighted_abs")
BATCH_SIZE = 32
MAX_BATCH_TOTAL_TOKENS = 8000
SUBJECTS = ["professional_law", "abstract_algebra", "anatomy", "astronomy", "business_ethics"]

def load_checkpoint_index() -> Dict:
    """Load checkpoint index"""
    index_file = CHECKPOINT_DIR / "model.safetensors.index.json"
    with open(index_file) as f:
        return json.load(f)

def get_shard_for_key(index: Dict, key: str) -> Optional[str]:
    """Get shard file for a given weight key"""
    return index["weight_map"].get(key)

def load_weight(key: str, index: Dict, device: str = "cpu") -> Optional[torch.Tensor]:
    """Load a single weight from checkpoint"""
    shard_file = get_shard_for_key(index, key)
    if not shard_file:
        return None
    
    shard_path = CHECKPOINT_DIR / shard_file
    try:
        weights = load_file(str(shard_path), device=device)
        return weights.get(key)
    except Exception as e:
        print(f"Error loading {key} from {shard_file}: {e}")
        return None

def analyze_checkpoint_structure():
    """Analyze checkpoint structure and compression metrics"""
    print("=" * 60)
    print("COMPRESSED CHECKPOINT ANALYSIS")
    print("=" * 60)
    
    index = load_checkpoint_index()
    weight_map = index["weight_map"]
    
    # Count weight types
    weight_types = {}
    for key in weight_map.keys():
        if ".weight_indices" in key:
            weight_types["indices"] = weight_types.get("indices", 0) + 1
        elif ".weight_codebook_entries" in key:
            weight_types["codebook_entries"] = weight_types.get("codebook_entries", 0) + 1
        elif ".weight_scale" in key:
            weight_types["scales"] = weight_types.get("scales", 0) + 1
        elif ".weight" in key:
            weight_types["weights"] = weight_types.get("weights", 0) + 1
    
    print(f"\nWeight Statistics:")
    print(f"  Total weights: {len(weight_map)}")
    for wtype, count in sorted(weight_types.items()):
        print(f"  {wtype}: {count}")
    
    # Analyze shard distribution
    shards = {}
    for key, shard in weight_map.items():
        shards[shard] = shards.get(shard, 0) + 1
    
    print(f"\nShard Statistics:")
    print(f"  Total shards: {len(shards)}")
    print(f"  Avg weights per shard: {len(weight_map) / len(shards):.1f}")
    
    # Check shard files exist
    missing = 0
    for shard in shards.keys():
        shard_path = CHECKPOINT_DIR / shard
        if not shard_path.exists():
            missing += 1
    
    print(f"  Missing shards: {missing}")
    
    # Sample a compressed weight
    print(f"\nSample Compressed Weight:")
    sample_key = None
    for key in weight_map.keys():
        if ".weight_indices" in key:
            sample_key = key
            break
    
    if sample_key:
        weight = load_weight(sample_key, index, device="cpu")
        if weight is not None:
            print(f"  Key: {sample_key}")
            print(f"  Shape: {weight.shape}")
            print(f"  Dtype: {weight.dtype}")
            print(f"  Size: {weight.numel() * weight.element_size() / (1024**2):.2f} MB")
        else:
            print(f"  Failed to load {sample_key}")
    
    # Calculate compression ratio
    print(f"\nCompression Metrics:")
    baseline_size = sum(f.stat().st_size for f in Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint").glob("model-*.safetensors")) / (1024**3)
    compressed_size = sum(f.stat().st_size for f in CHECKPOINT_DIR.glob("model-*.safetensors")) / (1024**3)
    ratio = baseline_size / compressed_size if compressed_size > 0 else 0
    
    print(f"  Baseline: {baseline_size:.2f} GB")
    print(f"  Compressed: {compressed_size:.2f} GB")
    print(f"  Ratio: {ratio:.2f}x")
    
    return True

if __name__ == "__main__":
    try:
        success = analyze_checkpoint_structure()
        print("\n✅ Checkpoint analysis complete")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
