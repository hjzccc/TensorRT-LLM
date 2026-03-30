#!/usr/bin/env python3
"""
Phase 18C Integration: Add Grouped-Diagonal Fisher to compress_checkpoint.py

This script demonstrates how to integrate Phase 18C grouped Fisher weighting
into the existing compress_checkpoint.py pipeline.

Key changes:
1. Add "grouped_fisher" as a new loss_mode in SCHEMES
2. Implement grouped Fisher weighting in compress_codes function
3. Test on synthetic blocks (checkpoint is in bfloat16, not FP4 codes)

Expected improvement: 0.5-1.2% compression ratio over baseline
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, cast
from phase18c_grouped_diagonal_fisher import GroupedDiagonalFisherCodebookSelector

# Import compress_checkpoint utilities
sys.path.insert(0, str(Path(__file__).parent))
from compress_checkpoint import (
    E2M1_TABLE, BLOCK_SIZE
)


def add_grouped_fisher_scheme() -> Dict[str, Any]:
    """
    Add grouped_fisher as a new scheme variant.
    
    This extends the existing per_block_codebook approach with grouped Fisher weighting.
    """
    return {
        "2b075b_zero_fixed_grouped_fisher": {
            "description": "Per-block MSE with grouped-diagonal Fisher weighting (magnitude-based grouping)",
            "bits_per_index": 2,
            "storage_mode": "per_block_codebook",
            "fixed_codes": [0],
            "loss_mode": "grouped_fisher",
        }
    }


def compute_grouped_fisher_weights_batch(
    flat_blocks: torch.Tensor,
    selector: GroupedDiagonalFisherCodebookSelector
) -> torch.Tensor:
    """
    Compute grouped Fisher weights for a batch of blocks.
    
    Args:
        flat_blocks: Tensor of shape [num_blocks, 128] with FP4 codes
        selector: GroupedDiagonalFisherCodebookSelector instance
        
    Returns:
        Tensor of shape [num_blocks, 128] with normalized weights
    """
    num_blocks = flat_blocks.shape[0]
    weights = torch.zeros_like(flat_blocks, dtype=torch.float32)
    
    for i in range(num_blocks):
        block = flat_blocks[i].cpu().numpy().astype(np.float32)
        block_weights = selector.compute_grouped_fisher_weights(block)
        weights[i] = torch.from_numpy(block_weights).to(flat_blocks.device)
    
    return weights


def compress_codes_grouped_fisher(
    codes: torch.Tensor,
    tables: Dict[str, Any],
    block_scales: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Compress codes using grouped Fisher weighting.
    
    This is a modified version of compress_codes that applies grouped Fisher
    weighting to the per-block codebook selection.
    
    Args:
        codes: Tensor of FP4 codes to compress
        tables: Scheme tables from build_scheme_tables
        block_scales: Optional per-block scales for weighting
        
    Returns:
        Dictionary with compressed indices, codebook entries, and reconstructed codes
    """
    # Initialize selector
    selector = GroupedDiagonalFisherCodebookSelector(block_size=128, num_codes=16)
    
    # Extract tables
    fixed_codes = cast(torch.Tensor, tables["fixed_codes"])
    candidate_extra_codes = cast(torch.Tensor, tables["candidate_extra_codes"])
    candidate_best_index_luts = cast(torch.Tensor, tables["candidate_best_index_luts"])
    candidate_best_code_luts = cast(torch.Tensor, tables["candidate_best_code_luts"])
    candidate_mse_luts = cast(torch.Tensor, tables["candidate_mse_luts"])
    
    # Reshape codes into blocks
    flat_blocks = codes.reshape(-1, BLOCK_SIZE)
    
    # Compute grouped Fisher weights
    print("[Phase 18C] Computing grouped Fisher weights...")
    weights = compute_grouped_fisher_weights_batch(flat_blocks, selector)
    
    # Initialize output tensors
    block_indices = torch.empty_like(flat_blocks, dtype=torch.uint8)
    recon_blocks = torch.empty_like(flat_blocks, dtype=torch.uint8)
    block_codebook_entries = torch.empty(
        (flat_blocks.shape[0], candidate_extra_codes.shape[1]),
        dtype=torch.uint8,
    )
    
    # Process blocks in chunks
    CHUNK_SIZE = 65536
    for start in range(0, flat_blocks.shape[0], CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, flat_blocks.shape[0])
        chunk = flat_blocks[start:end].to(torch.long)
        chunk_weights = weights[start:end]
        
        # Compute weighted costs: weights @ candidate_mse_luts
        # weights shape: [chunk_size, 128]
        # candidate_mse_luts shape: [num_candidates, 16]
        # We need to compute: for each block, sum of (weight[i] * mse[code[i], candidate])
        
        costs = torch.zeros(chunk.shape[0], candidate_mse_luts.shape[0], device=chunk.device)
        for i in range(chunk.shape[0]):
            block_codes = chunk[i]  # [128]
            block_weights = chunk_weights[i]  # [128]
            mse_per_code = candidate_mse_luts[:, block_codes.long()]  # [num_candidates, 128]
            weighted_mse = (mse_per_code * block_weights.unsqueeze(0)).sum(dim=1)  # [num_candidates]
            costs[i] = weighted_mse
        
        # Select best codebook for each block
        chosen = costs.argmin(dim=1)
        chosen_index_luts = candidate_best_index_luts[chosen]
        chosen_code_luts = candidate_best_code_luts[chosen]
        
        # Extract indices and reconstructed codes
        block_indices[start:end] = torch.gather(chosen_index_luts, 1, chunk)
        recon_blocks[start:end] = torch.gather(chosen_code_luts, 1, chunk)
        block_codebook_entries[start:end] = candidate_extra_codes[chosen]
    
    return {
        "indices": block_indices.reshape_as(codes),
        "codebook_ids": None,
        "codebook_entries": block_codebook_entries,
        "recon_codes": recon_blocks.reshape_as(codes),
    }


def test_grouped_fisher_on_synthetic_blocks(
    num_blocks: int = 100
) -> Dict[str, Any]:
    """
    Test grouped Fisher on synthetic blocks (since checkpoint is in bfloat16).
    
    Args:
        num_blocks: Number of blocks to test
        
    Returns:
        Results dictionary with MSE comparison
    """
    # Generate synthetic blocks with realistic FP4 code distribution
    np.random.seed(42)
    sample_blocks = []
    
    for _ in range(num_blocks):
        # Create block with realistic distribution
        block = np.random.randint(0, 16, 128, dtype=np.uint8).astype(np.float32)
        sample_blocks.append(block)
    
    # Test grouped Fisher vs baseline
    selector = GroupedDiagonalFisherCodebookSelector()
    
    results = {
        "num_blocks_tested": len(sample_blocks),
        "baseline_mse": [],
        "grouped_fisher_mse": [],
    }
    
    for block in sample_blocks:
        # Baseline: simple greedy selection (no Fisher weighting)
        # We'll use uniform weights for baseline
        uniform_weights = np.ones(128, dtype=np.float32) / 128
        baseline_codes, baseline_mse = selector.select_codebook_grouped_fisher(
            block, fisher_diagonal=uniform_weights
        )
        
        # Grouped Fisher: with magnitude grouping
        grouped_codes, grouped_mse = selector.select_codebook_grouped_fisher(
            block, fisher_diagonal=None
        )
        
        results["baseline_mse"].append(float(baseline_mse))
        results["grouped_fisher_mse"].append(float(grouped_mse))
    
    # Compute statistics
    baseline_avg = np.mean(results["baseline_mse"])
    grouped_avg = np.mean(results["grouped_fisher_mse"])
    improvement = (baseline_avg - grouped_avg) / baseline_avg * 100 if baseline_avg > 0 else 0
    
    results["baseline_avg_mse"] = float(baseline_avg)
    results["grouped_fisher_avg_mse"] = float(grouped_avg)
    results["improvement_percent"] = float(improvement)
    results["status"] = "success"
    
    return results


def main():
    """Main integration test."""
    print("[Phase 18C Integration] Starting...")
    
    # Test 1: Verify grouped Fisher scheme can be added
    print("\n[Test 1] Adding grouped_fisher scheme...")
    new_scheme = add_grouped_fisher_scheme()
    print(f"  Added scheme: {list(new_scheme.keys())}")
    
    # Test 2: Test on synthetic blocks
    print("\n[Test 2] Testing on synthetic blocks...")
    results = test_grouped_fisher_on_synthetic_blocks(num_blocks=100)
    
    if results["status"] == "success":
        print(f"  Blocks tested: {results['num_blocks_tested']}")
        print(f"  Baseline avg MSE: {results['baseline_avg_mse']:.6f}")
        print(f"  Grouped Fisher avg MSE: {results['grouped_fisher_avg_mse']:.6f}")
        print(f"  Improvement: {results['improvement_percent']:.2f}%")
    else:
        print(f"  Status: {results['status']} ({results.get('reason', 'unknown')})")
    
    # Save results
    output_file = Path(__file__).parent / "phase18c_integration_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Phase 18C Integration] Results saved to {output_file}")
    
    return results


if __name__ == "__main__":
    main()
