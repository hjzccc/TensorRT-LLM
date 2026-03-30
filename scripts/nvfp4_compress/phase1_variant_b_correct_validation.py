#!/usr/bin/env python3
"""
Phase 1 Final: Variant B Correct Validation

Properly quantizes bfloat16 weights to FP4 codes, then applies Variant B.
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple
import logging

import torch
import numpy as np
from safetensors import safe_open
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FP4 E2M1 code table
E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

# Configuration
SRC_CHECKPOINT = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
OUTPUT_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress")
BLOCK_SIZE = 128
NUM_CODEWORDS = 4
MAX_WEIGHTS_TO_LOAD = 20

SKIP_PATTERNS = [
    "layernorm", "norm.weight", "mlp.gate.weight", "shared_expert_gate",
    "embed_tokens", "lm_head", "A_log", "dt_bias", "conv1d",
    "linear_attn", "self_attn", "mtp.", "model.visual.",
]

def should_compress(key: str) -> bool:
    """Check if a weight should be compressed."""
    for pattern in SKIP_PATTERNS:
        if pattern in key:
            return False
    return key.endswith(".weight")

def quantize_to_fp4(weight: torch.Tensor) -> np.ndarray:
    """
    Quantize bfloat16 weight to FP4 codes (0-15).
    Uses simple nearest-neighbor quantization to E2M1 table.
    """
    weight_np = weight.float().cpu().numpy().flatten()
    
    # Find nearest FP4 code for each weight value
    codes = np.zeros(len(weight_np), dtype=np.uint8)
    for i, val in enumerate(weight_np):
        distances = np.abs(E2M1_TABLE - val)
        codes[i] = np.argmin(distances)
    
    return codes

class VariantBValidator:
    """Variant B frequency-weighted MSE validator."""
    
    def __init__(self, block_size: int = 128, num_codewords: int = 4):
        self.block_size = block_size
        self.num_codewords = num_codewords
    
    def select_codebook_for_block_greedy(self, block: np.ndarray) -> Tuple[Tuple, float]:
        """
        Select best codebook using greedy frequency-weighted MSE.
        """
        code_counts = Counter(block)
        
        # Get the 4 most frequent codes
        most_common = code_counts.most_common(self.num_codewords)
        
        # If we have fewer than 4 unique codes, pad with zeros
        if len(most_common) < self.num_codewords:
            used_codes = set(code for code, _ in most_common)
            for code in range(16):
                if code not in used_codes and len(most_common) < self.num_codewords:
                    most_common.append((code, 0))
        
        selected_codes = tuple(code for code, _ in most_common[:self.num_codewords])
        
        # Compute weighted MSE for this codebook
        code_weights = np.array([code_counts.get(i, 0) for i in range(16)]) / len(block)
        
        weighted_mse = 0.0
        for code in range(16):
            if code_weights[code] > 0:
                code_value = E2M1_TABLE[code]
                codebook_values = np.array([E2M1_TABLE[c] for c in selected_codes])
                distances = np.abs(codebook_values - code_value)
                nearest_idx = np.argmin(distances)
                error = (code_value - codebook_values[nearest_idx]) ** 2
                weighted_mse += code_weights[code] * error
        
        return selected_codes, weighted_mse
    
    def compress_weight(self, weight_np: np.ndarray) -> Dict:
        """Compress a single weight using Variant B."""
        num_blocks = len(weight_np) // self.block_size
        if len(weight_np) % self.block_size != 0:
            num_blocks += 1
        
        total_mse = 0.0
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(weight_np))
            block = weight_np[start:end]
            
            _, weighted_mse = self.select_codebook_for_block_greedy(block)
            total_mse += weighted_mse * len(block)
        
        avg_mse = total_mse / len(weight_np)
        
        # Estimate compression
        original_size = len(weight_np) * 2  # bfloat16 = 2 bytes
        # Compressed: 2 bits per element + codebook overhead
        compressed_size = (len(weight_np) * 2 / 8) + (num_blocks * 4 * 4)
        
        return {
            "avg_mse": float(avg_mse),
            "num_blocks": num_blocks,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": original_size / compressed_size if compressed_size > 0 else 0,
        }

def main():
    """Main validation pipeline."""
    print("=" * 80)
    print("PHASE 1 FINAL: VARIANT B CORRECT VALIDATION")
    print("(Quantizes bfloat16 to FP4, then applies Variant B)")
    print("=" * 80)
    
    # Load sample weights
    logger.info(f"Step 1: Loading sample weights (max {MAX_WEIGHTS_TO_LOAD})")
    
    weights = {}
    model_files = sorted(SRC_CHECKPOINT.glob("model-*.safetensors"))
    
    for model_file in model_files:
        if len(weights) >= MAX_WEIGHTS_TO_LOAD:
            break
        
        logger.info(f"  Loading {model_file.name}...")
        try:
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if len(weights) >= MAX_WEIGHTS_TO_LOAD:
                        break
                    if should_compress(key):
                        tensor = f.get_tensor(key)
                        if tensor.dtype != torch.float32:
                            tensor = tensor.float()
                        weights[key] = tensor
        except Exception as e:
            logger.warning(f"  Error loading {model_file.name}: {e}")
            continue
    
    logger.info(f"Loaded {len(weights)} sample weights")
    
    if not weights:
        logger.error("No weights loaded!")
        return
    
    # Initialize Variant B
    logger.info("Step 2: Initializing Variant B validator")
    validator = VariantBValidator(block_size=BLOCK_SIZE, num_codewords=NUM_CODEWORDS)
    
    # Validate weights
    logger.info("Step 3: Quantizing and validating compression on sample weights")
    start_time = time.time()
    
    results = []
    total_original_size = 0
    total_compressed_size = 0
    
    for idx, (weight_name, weight_tensor) in enumerate(weights.items(), 1):
        print(f"  [{idx:3d}/{len(weights)}] {weight_name}...", end=" ", flush=True)
        
        try:
            # Step 1: Quantize bfloat16 to FP4 codes
            fp4_codes = quantize_to_fp4(weight_tensor)
            
            # Step 2: Apply Variant B compression
            result = validator.compress_weight(fp4_codes)
            
            total_original_size += result["original_size"]
            total_compressed_size += result["compressed_size"]
            
            results.append({
                "weight": weight_name,
                "original_size": result["original_size"],
                "compressed_size": result["compressed_size"],
                "compression_ratio": result["compression_ratio"],
                "avg_mse": result["avg_mse"],
            })
            
            print(f"✓ {result['compression_ratio']:.2f}x")
        
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append({
                "weight": weight_name,
                "error": str(e),
            })
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    successful = [r for r in results if "compression_ratio" in r]
    
    if successful:
        avg_compression = np.mean([r["compression_ratio"] for r in successful])
        overall_compression = total_original_size / total_compressed_size if total_compressed_size > 0 else 0
        overall_compression_percent = (1 - total_compressed_size / total_original_size) * 100 if total_original_size > 0 else 0
        
        print(f"\nSample weights: {len(weights)}")
        print(f"Successfully compressed: {len(successful)}")
        print(f"Average compression ratio: {avg_compression:.2f}x")
        print(f"Overall compression ratio: {overall_compression:.2f}x")
        print(f"Overall compression: {overall_compression_percent:.1f}%")
        print(f"\nOriginal size: {total_original_size/1e9:.2f}GB")
        print(f"Compressed size: {total_compressed_size/1e9:.2f}GB")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        # Compare with baseline
        baseline_compression = 97.5
        print(f"\n" + "=" * 80)
        print("COMPARISON WITH BASELINE")
        print("=" * 80)
        print(f"Baseline (97.5%): 97.5% compression")
        print(f"Variant B: {overall_compression_percent:.1f}% compression")
        
        if overall_compression_percent >= baseline_compression:
            print(f"✅ VARIANT B MEETS OR EXCEEDS BASELINE")
            print(f"   Improvement: +{overall_compression_percent - baseline_compression:.1f}%")
        else:
            print(f"⚠️  VARIANT B BELOW BASELINE")
            print(f"   Degradation: {baseline_compression - overall_compression_percent:.1f}%")
    
    # Save results
    logger.info("Step 4: Saving results")
    metadata = {
        "variant": "B_frequency_weighted_mse",
        "validation_type": "sample_with_quantization",
        "quantization_method": "nearest_neighbor_to_e2m1",
        "selection_method": "greedy_most_frequent",
        "block_size": BLOCK_SIZE,
        "num_codewords": NUM_CODEWORDS,
        "sample_size": len(weights),
        "successfully_compressed": len(successful),
        "total_original_size": total_original_size,
        "total_compressed_size": total_compressed_size,
        "overall_compression_ratio": float(total_original_size / total_compressed_size) if total_compressed_size > 0 else 0,
        "overall_compression_percent": float(overall_compression_percent) if total_original_size > 0 else 0,
        "baseline_compression_percent": 97.5,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    
    output_file = OUTPUT_DIR / "phase1_variant_b_correct_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("PHASE 1 VALIDATION COMPLETE ✅")
    print("=" * 80)
    
    return metadata

if __name__ == "__main__":
    main()
