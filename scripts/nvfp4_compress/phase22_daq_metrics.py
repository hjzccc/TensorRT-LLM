#!/usr/bin/env python3
"""
Phase 22: DAQ-Inspired Delta-Aware Quantization Metrics

Implements Sign Preservation Rate (SPR) and Cosine Similarity (CS) metrics
as alternatives to MSE for codebook selection.

Paper: DAQ (arXiv:2603.22324, March 2026)
- SPR: Measures how well quantization preserves sign of weights
- CS: Measures directional fidelity of quantized weights

Expected improvement: +0.1-0.3% compression over Phase 21
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple
import logging

import torch
import numpy as np
from safetensors import safe_open
from itertools import combinations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FP4 E2M1 code table
E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

SRC_CHECKPOINT = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")
OUTPUT_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress")
BLOCK_SIZE = 128
NUM_CODEWORDS = 4
MAX_WEIGHTS_TO_LOAD = 15

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

class DAQMetricsValidator:
    """Phase 22: DAQ-Inspired Delta-Aware Metrics."""
    
    def __init__(self, block_size: int = 128, num_codewords: int = 4):
        self.block_size = block_size
        self.num_codewords = num_codewords
        
        # Generate all possible 4-code codebooks
        self.all_codebooks = list(combinations(range(16), num_codewords))
        logger.info(f"Generated {len(self.all_codebooks)} possible codebooks")
        
        # Precompute codebook values
        self.codebook_values_cache = {}
        for codebook in self.all_codebooks:
            self.codebook_values_cache[codebook] = np.array([
                E2M1_TABLE[c] for c in codebook
            ])
    
    def compute_spr(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """
        Compute Sign Preservation Rate (SPR).
        Measures fraction of elements where sign is preserved.
        """
        # Avoid division by zero for zero elements
        nonzero_mask = original != 0
        if not np.any(nonzero_mask):
            return 1.0  # All zeros, perfect preservation
        
        sign_match = np.sign(original[nonzero_mask]) == np.sign(quantized[nonzero_mask])
        return float(np.mean(sign_match))
    
    def compute_cosine_similarity(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """
        Compute Cosine Similarity (CS).
        Measures directional fidelity of quantized weights.
        """
        orig_norm = np.linalg.norm(original)
        quant_norm = np.linalg.norm(quantized)
        
        if orig_norm < 1e-8 or quant_norm < 1e-8:
            return 1.0  # Both near zero, perfect similarity
        
        dot_product = np.dot(original, quantized)
        cs = dot_product / (orig_norm * quant_norm)
        return float(np.clip(cs, -1.0, 1.0))
    
    def compute_daq_score(self, original: np.ndarray, quantized: np.ndarray, 
                         alpha: float = 0.5) -> float:
        """
        Compute combined DAQ score.
        DAQ_score = alpha * SPR + (1 - alpha) * CS
        
        alpha=0.5 gives equal weight to sign preservation and directional fidelity
        """
        spr = self.compute_spr(original, quantized)
        cs = self.compute_cosine_similarity(original, quantized)
        
        # DAQ score: higher is better (opposite of MSE)
        # We want to maximize this, so we'll use (1 - score) as loss
        daq_score = alpha * spr + (1 - alpha) * cs
        return daq_score
    
    def select_codebook_for_block_daq(self, block: np.ndarray, 
                                      original_block: np.ndarray) -> Tuple[Tuple, float]:
        """Select best codebook for a block using DAQ metrics."""
        best_daq_score = -float('inf')
        best_codebook = None
        
        for codebook in self.all_codebooks:
            codebook_values = self.codebook_values_cache[codebook]
            
            # Quantize block using this codebook
            quantized_block = np.zeros_like(original_block)
            for i, code in enumerate(block):
                code_value = E2M1_TABLE[code]
                distances = np.abs(codebook_values - code_value)
                nearest_idx = np.argmin(distances)
                quantized_block[i] = codebook_values[nearest_idx]
            
            # Compute DAQ score
            daq_score = self.compute_daq_score(original_block, quantized_block)
            
            if daq_score > best_daq_score:
                best_daq_score = daq_score
                best_codebook = codebook
        
        return best_codebook, best_daq_score
    
    def select_codebook_for_block_mse(self, block: np.ndarray) -> Tuple[Tuple, float]:
        """Select best codebook for a block using MSE (baseline)."""
        best_mse = float('inf')
        best_codebook = None
        
        for codebook in self.all_codebooks:
            codebook_values = self.codebook_values_cache[codebook]
            
            # Compute MSE
            mse = 0.0
            for code in block:
                code_value = E2M1_TABLE[code]
                distances = np.abs(codebook_values - code_value)
                nearest_idx = np.argmin(distances)
                error = (code_value - codebook_values[nearest_idx]) ** 2
                mse += error
            
            mse /= len(block)
            
            if mse < best_mse:
                best_mse = mse
                best_codebook = codebook
        
        return best_codebook, best_mse
    
    def compress_weight_daq(self, weight_np: np.ndarray, original_weight: np.ndarray) -> Dict:
        """Compress a single weight using DAQ metrics."""
        num_blocks = len(weight_np) // self.block_size
        if len(weight_np) % self.block_size != 0:
            num_blocks += 1
        
        total_daq_score = 0.0
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(weight_np))
            block = weight_np[start:end]
            original_block = original_weight[start:end]
            
            _, daq_score = self.select_codebook_for_block_daq(block, original_block)
            total_daq_score += daq_score * len(block)
        
        avg_daq_score = total_daq_score / len(weight_np)
        
        # Compression calculation (matching baseline)
        original_bits = len(weight_np) * 32  # FP32
        index_bits = len(weight_np) * 2  # 2-bit indices
        codebook_bits = num_blocks * 16  # 4 FP4 codes per block
        total_bits = index_bits + codebook_bits
        
        compression = 1.0 - (total_bits / original_bits)
        bits_per_elem = total_bits / len(weight_np)
        
        return {
            "avg_daq_score": float(avg_daq_score),
            "num_blocks": num_blocks,
            "compression": float(compression),
            "bits_per_elem": float(bits_per_elem),
        }
    
    def compress_weight_mse(self, weight_np: np.ndarray) -> Dict:
        """Compress a single weight using MSE (baseline)."""
        num_blocks = len(weight_np) // self.block_size
        if len(weight_np) % self.block_size != 0:
            num_blocks += 1
        
        total_mse = 0.0
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(weight_np))
            block = weight_np[start:end]
            
            _, mse = self.select_codebook_for_block_mse(block)
            total_mse += mse * len(block)
        
        avg_mse = total_mse / len(weight_np)
        
        # Compression calculation
        original_bits = len(weight_np) * 32
        index_bits = len(weight_np) * 2
        codebook_bits = num_blocks * 16
        total_bits = index_bits + codebook_bits
        
        compression = 1.0 - (total_bits / original_bits)
        bits_per_elem = total_bits / len(weight_np)
        
        return {
            "avg_mse": float(avg_mse),
            "num_blocks": num_blocks,
            "compression": float(compression),
            "bits_per_elem": float(bits_per_elem),
        }

def main():
    """Main validation pipeline."""
    print("=" * 80)
    print("PHASE 22: DAQ-INSPIRED DELTA-AWARE QUANTIZATION METRICS")
    print("=" * 80)
    
    # Load sample weights
    logger.info(f"Step 1: Loading sample weights (max {MAX_WEIGHTS_TO_LOAD})")
    
    weights = {}
    original_weights = {}
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
                        original_weights[key] = tensor.cpu().numpy().flatten()
                        weights[key] = quantize_to_fp4(tensor)
        except Exception as e:
            logger.warning(f"  Error loading {model_file.name}: {e}")
            continue
    
    logger.info(f"Loaded {len(weights)} sample weights")
    
    if not weights:
        logger.error("No weights loaded!")
        return
    
    # Initialize validator
    logger.info("Step 2: Initializing DAQ validator")
    validator = DAQMetricsValidator(block_size=BLOCK_SIZE, num_codewords=NUM_CODEWORDS)
    
    # Validate compression on sample weights
    logger.info("Step 3: Validating compression on sample weights")
    print("\n" + "=" * 80)
    print("PHASE 22: DAQ-INSPIRED DELTA-AWARE QUANTIZATION METRICS")
    print("=" * 80)
    
    results_daq = []
    results_mse = []
    
    for idx, (key, weight_codes) in enumerate(weights.items(), 1):
        original_weight = original_weights[key]
        
        # Compress with DAQ metrics
        result_daq = validator.compress_weight_daq(weight_codes, original_weight)
        results_daq.append(result_daq)
        
        # Compress with MSE (baseline)
        result_mse = validator.compress_weight_mse(weight_codes)
        results_mse.append(result_mse)
        
        daq_score = result_daq["avg_daq_score"]
        mse = result_mse["avg_mse"]
        compression = result_daq["compression"]
        
        print(f"  [{idx:2d}/{len(weights)}] {key[:50]:50s} | "
              f"DAQ: {daq_score:.4f} | MSE: {mse:.4f} | Compression: {compression*100:.2f}%")
    
    # Compute statistics
    avg_daq_score = np.mean([r["avg_daq_score"] for r in results_daq])
    avg_mse = np.mean([r["avg_mse"] for r in results_mse])
    avg_compression_daq = np.mean([r["compression"] for r in results_daq])
    avg_compression_mse = np.mean([r["compression"] for r in results_mse])
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nDAQ Metrics (Phase 22):")
    print(f"  Avg DAQ Score: {avg_daq_score:.4f}")
    print(f"  Avg Compression: {avg_compression_daq*100:.2f}%")
    
    print(f"\nMSE Baseline (Phase 21):")
    print(f"  Avg MSE: {avg_mse:.4f}")
    print(f"  Avg Compression: {avg_compression_mse*100:.2f}%")
    
    improvement = (avg_compression_daq - avg_compression_mse) * 100
    print(f"\nImprovement (DAQ vs MSE):")
    print(f"  Compression delta: {improvement:+.3f}%")
    print(f"  Status: {'✅ POSITIVE' if improvement > 0 else '❌ NEGATIVE'}")
    
    # Save results
    results = {
        "phase": "22",
        "step": "daq_metrics_validation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "daq_metrics": {
            "avg_daq_score": float(avg_daq_score),
            "avg_compression": float(avg_compression_daq),
            "num_weights": len(weights),
        },
        "mse_baseline": {
            "avg_mse": float(avg_mse),
            "avg_compression": float(avg_compression_mse),
            "num_weights": len(weights),
        },
        "improvement": {
            "compression_delta_percent": float(improvement),
            "status": "positive" if improvement > 0 else "negative",
        }
    }
    
    output_file = OUTPUT_DIR / "phase22_daq_metrics_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()
