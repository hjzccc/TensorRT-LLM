#!/usr/bin/env python3
"""
Phase 1: Variant B Proper Validation

Uses the same compression calculation as the baseline (Two-Level Quantization).
Compares bits per element, not file sizes.
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
from itertools import combinations

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
MAX_WEIGHTS_TO_LOAD = 10

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
    """Quantize bfloat16 weight to FP4 codes (0-15)."""
    weight_np = weight.float().cpu().numpy().flatten()
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
        
        # Precompute all possible codebooks
        self.all_codes = np.arange(16)
        self.all_codebooks = list(combinations(self.all_codes, num_codewords))
        logger.info(f"Initialized with {len(self.all_codebooks)} possible codebooks")
        
        # Precompute codebook values
        self.codebook_values_cache = {}
        for codebook in self.all_codebooks:
            self.codebook_values_cache[codebook] = np.array([
                E2M1_TABLE[c] for c in codebook
            ])
    
    def select_codebook_for_block(self, block: np.ndarray) -> Tuple[Tuple, float]:
        """Select best codebook for a block using frequency-weighted MSE."""
        code_counts = Counter(block)
        code_weights = np.array([code_counts.get(i, 0) for i in range(16)]) / len(block)
        
        best_weighted_mse = float('inf')
        best_codebook = None
        
        for codebook in self.all_codebooks:
            codebook_values = self.codebook_values_cache[codebook]
            
            weighted_mse = 0.0
            for code in range(16):
                if code_weights[code] > 0:
                    code_value = E2M1_TABLE[code]
                    distances = np.abs(codebook_values - code_value)
                    nearest_idx = np.argmin(distances)
                    error = (code_value - codebook_values[nearest_idx]) ** 2
                    weighted_mse += code_weights[code] * error
            
            if weighted_mse < best_weighted_mse:
                best_weighted_mse = weighted_mse
                best_codebook = codebook
        
        return best_codebook, best_weighted_mse
    
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
            
            best_codebook, weighted_mse = self.select_codebook_for_block(block)
            total_mse += weighted_mse * len(block)
        
        avg_mse = total_mse / len(weight_np)
        
        # Compression calculation (matching baseline methodology):
        # Original: FP32 = 32 bits per element
        # Compressed: 2 bits per element (indices) + codebook overhead
        # Codebook: 4 codes × 4 bits (FP4) = 16 bits per block
        
        original_bits = len(weight_np) * 32  # FP32
        index_bits = len(weight_np) * 2  # 2-bit indices
        codebook_bits = num_blocks * 16  # 4 FP4 codes per block
        total_bits = index_bits + codebook_bits
        
        compression = 1.0 - (total_bits / original_bits)
        bits_per_elem = total_bits / len(weight_np)
        
        return {
            "avg_mse": float(avg_mse),
            "num_blocks": num_blocks,
            "original_bits": original_bits,
            "total_bits": total_bits,
            "compression": float(compression),
            "bits_per_elem": float(bits_per_elem),
        }

def main():
    """Main validation pipeline."""
    print("=" * 80)
    print("PHASE 1: VARIANT B PROPER VALIDATION")
    print("(Using baseline compression calculation: bits per element)")
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
    logger.info("Step 3: Validating compression on sample weights")
    start_time = time.time()
    
    results = []
    total_compression = 0.0
    total_bits_per_elem = 0.0
    total_mse = 0.0
    
    for idx, (weight_name, weight_tensor) in enumerate(weights.items(), 1):
        print(f"  [{idx:3d}/{len(weights)}] {weight_name}...", end=" ", flush=True)
        
        try:
            # Quantize bfloat16 to FP4 codes
            fp4_codes = quantize_to_fp4(weight_tensor)
            
            # Apply Variant B compression
            result = validator.compress_weight(fp4_codes)
            
            total_compression += result["compression"]
            total_bits_per_elem += result["bits_per_elem"]
            total_mse += result["avg_mse"]
            
            results.append({
                "weight": weight_name,
                "shape": list(weight_tensor.shape),
                "numel": weight_tensor.numel(),
                "compression": result["compression"],
                "bits_per_elem": result["bits_per_elem"],
                "avg_mse": result["avg_mse"],
            })
            
            print(f"✓ {result['compression']*100:.1f}% compression, {result['bits_per_elem']:.2f} bits/elem")
        
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
    
    successful = [r for r in results if "compression" in r]
    
    if successful:
        avg_compression = np.mean([r["compression"] for r in successful])
        avg_bits_per_elem = np.mean([r["bits_per_elem"] for r in successful])
        avg_mse = np.mean([r["avg_mse"] for r in successful])
        
        print(f"\nSample weights: {len(weights)}")
        print(f"Successfully compressed: {len(successful)}")
        print(f"Average compression: {avg_compression*100:.1f}%")
        print(f"Average bits per element: {avg_bits_per_elem:.4f}")
        print(f"Average MSE: {avg_mse:.6f}")
        print(f"Elapsed time: {elapsed:.2f}s")
        
        # Compare with baseline
        baseline_compression = 0.975
        baseline_bits_per_elem = 0.8125
        
        print(f"\n" + "=" * 80)
        print("COMPARISON WITH BASELINE (Two-Level Quantization)")
        print("=" * 80)
        print(f"Baseline compression: {baseline_compression*100:.1f}%")
        print(f"Variant B compression: {avg_compression*100:.1f}%")
        print(f"Baseline bits/elem: {baseline_bits_per_elem:.4f}")
        print(f"Variant B bits/elem: {avg_bits_per_elem:.4f}")
        
        if avg_compression >= baseline_compression:
            print(f"✅ VARIANT B MEETS OR EXCEEDS BASELINE")
            print(f"   Improvement: +{(avg_compression - baseline_compression)*100:.1f}%")
        else:
            print(f"⚠️  VARIANT B BELOW BASELINE")
            print(f"   Degradation: {(baseline_compression - avg_compression)*100:.1f}%")
    
    # Save results
    logger.info("Step 4: Saving results")
    metadata = {
        "variant": "B_frequency_weighted_mse",
        "validation_type": "sample_proper_calculation",
        "block_size": BLOCK_SIZE,
        "num_codewords": NUM_CODEWORDS,
        "sample_size": len(weights),
        "successfully_compressed": len(successful),
        "avg_compression": float(avg_compression) if successful else 0,
        "avg_bits_per_elem": float(avg_bits_per_elem) if successful else 0,
        "avg_mse": float(avg_mse) if successful else 0,
        "baseline_compression": 0.975,
        "baseline_bits_per_elem": 0.8125,
        "elapsed_seconds": elapsed,
        "results": results,
    }
    
    output_file = OUTPUT_DIR / "phase1_variant_b_proper_validation_results.json"
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("PHASE 1 VALIDATION COMPLETE ✅")
    print("=" * 80)
    
    return metadata

if __name__ == "__main__":
    main()
