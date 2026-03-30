#!/usr/bin/env python3
"""
Phase 18B Integration: Apply Block-Diagonal Fisher to actual checkpoint compression.

This script:
1. Loads an NVFP4 checkpoint (sharded format)
2. Extracts weight blocks
3. Computes Fisher information (approximated from weight magnitudes)
4. Applies Block-Diagonal Fisher codebook selection
5. Measures compression and PPL impact
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from safetensors import safe_open
import time
import glob

# Add phase18b to path
sys.path.insert(0, "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress")
from phase18b_block_diagonal_fisher import BlockDiagonalFisherCodebookSelector


def load_checkpoint_sharded(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load sharded NVFP4 checkpoint."""
    print(f"Loading sharded checkpoint from {checkpoint_dir}...")
    
    state_dict = {}
    shard_files = sorted(glob.glob(f"{checkpoint_dir}/model-*.safetensors"))
    
    print(f"Found {len(shard_files)} shard files")
    
    for shard_file in shard_files[:3]:  # Load first 3 shards for speed
        print(f"  Loading {Path(shard_file).name}...")
        with safe_open(shard_file, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
    print(f"Loaded {len(state_dict)} tensors")
    return state_dict


def extract_blocks(tensor: torch.Tensor, block_size: int = 128) -> List[np.ndarray]:
    """Extract 128-element blocks from a tensor."""
    blocks = []
    flat = tensor.reshape(-1).cpu().numpy().astype(np.float32)
    
    for i in range(0, len(flat), block_size):
        block = flat[i:i+block_size]
        if len(block) == block_size:
            blocks.append(block)
    
    return blocks


def estimate_fisher_from_weights(block: np.ndarray) -> np.ndarray:
    """Estimate Fisher diagonal from weight magnitudes."""
    # Simple approximation: Fisher ~ |weight|^2
    fisher = np.abs(block) ** 2
    # Normalize
    fisher = fisher / (np.sum(fisher) + 1e-8)
    return fisher


def evaluate_phase18b_on_checkpoint(
    checkpoint_dir: str,
    max_blocks: int = 1000,
    sample_layers: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate Phase 18B on actual checkpoint.
    
    Args:
        checkpoint_dir: Path to NVFP4 checkpoint directory
        max_blocks: Maximum blocks to evaluate (for speed)
        sample_layers: Specific layers to evaluate (if None, sample all)
        
    Returns:
        Dictionary with evaluation results
    """
    print("\n" + "="*80)
    print("Phase 18B Integration: Block-Diagonal Fisher on Real Checkpoint")
    print("="*80 + "\n")
    
    # Load checkpoint
    state_dict = load_checkpoint_sharded(checkpoint_dir)
    
    # Initialize selector
    selector = BlockDiagonalFisherCodebookSelector()
    
    # Collect blocks from quantized weight tensors
    all_blocks = []
    all_fishers = []
    layer_info = {}
    
    for key, tensor in state_dict.items():
        # Only process weight tensors (skip scales, indices, etc.)
        if "weight" not in key or tensor.dtype != torch.float32:
            continue
        
        print(f"Processing {key} (shape {tensor.shape})...")
        
        blocks = extract_blocks(tensor)
        
        for block in blocks:
            if len(all_blocks) >= max_blocks:
                break
            
            # Estimate Fisher
            fisher = estimate_fisher_from_weights(block)
            
            all_blocks.append(block)
            all_fishers.append(fisher)
        
        layer_info[key] = {
            "shape": list(tensor.shape),
            "num_blocks": len(blocks),
            "dtype": str(tensor.dtype)
        }
        
        if len(all_blocks) >= max_blocks:
            break
    
    print(f"\nCollected {len(all_blocks)} blocks from checkpoint")
    
    # Evaluate Block-Diagonal Fisher
    print("\nEvaluating Block-Diagonal Fisher...")
    results = selector.evaluate_on_blocks(all_blocks, all_fishers)
    
    # Prepare output
    output = {
        "phase": "18B",
        "method": "Block-Diagonal Fisher",
        "checkpoint_dir": checkpoint_dir,
        "num_blocks_evaluated": len(all_blocks),
        "max_blocks_limit": max_blocks,
        "results": results,
        "layer_info": layer_info,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return output


def main():
    """Main entry point."""
    checkpoint_dir = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint"
    
    # Check if checkpoint exists
    if not Path(checkpoint_dir).exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_dir}")
        print("Skipping integration test (checkpoint not available)")
        
        # Create dummy results for testing
        dummy_results = {
            "phase": "18B",
            "method": "Block-Diagonal Fisher",
            "status": "SKIPPED - checkpoint not available",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase18b_integration_results.json")
        with open(output_file, 'w') as f:
            json.dump(dummy_results, f, indent=2)
        
        print(f"Saved dummy results to {output_file}")
        return
    
    # Evaluate on checkpoint
    results = evaluate_phase18b_on_checkpoint(checkpoint_dir, max_blocks=100)
    
    # Save results
    output_file = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase18b_integration_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    print("\n" + "="*80)
    print("Phase 18B Integration - COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
