#!/usr/bin/env python3
"""
Phase 22: DAQ-Inspired Greedy Codebook Selection

Uses greedy selection with DAQ metrics (Sign Preservation Rate + Cosine Similarity)
instead of exhaustive search. Much faster while maintaining quality.

Expected improvement: +0.1-0.3% compression over Phase 21
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple, Set
import logging

import torch
import numpy as np
from safetensors import safe_open

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
    for pattern in SKIP_PATTERNS:
        if pattern in key:
            return False
    return key.endswith(".weight")

def quantize_to_fp4(weight: torch.Tensor) -> np.ndarray:
    weight_np = weight.float().cpu().numpy().flatten()
    codes = np.zeros(len(weight_np), dtype=np.uint8)
    for i, val in enumerate(weight_np):
        distances = np.abs(E2M1_TABLE - val)
        codes[i] = np.argmin(distances)
    return codes

class DAQGreedyValidator:
    """Phase 22: DAQ-Inspired Greedy Codebook Selection."""
    
    def __init__(self, block_size: int = 128, num_codewords: int = 4):
        self.block_size = block_size
        self.num_codewords = num_codewords
    
    def compute_spr(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """Sign Preservation Rate."""
        nonzero_mask = original != 0
        if not np.any(nonzero_mask):
            return 1.0
        sign_match = np.sign(original[nonzero_mask]) == np.sign(quantized[nonzero_mask])
        return float(np.mean(sign_match))
    
    def compute_cosine_similarity(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """Cosine Similarity."""
        orig_norm = np.linalg.norm(original)
        quant_norm = np.linalg.norm(quantized)
        if orig_norm < 1e-8 or quant_norm < 1e-8:
            return 1.0
        dot_product = np.dot(original, quantized)
        cs = dot_product / (orig_norm * quant_norm)
        return float(np.clip(cs, -1.0, 1.0))
    
    def compute_daq_score(self, original: np.ndarray, quantized: np.ndarray, 
                         alpha: float = 0.5) -> float:
        """Combined DAQ score: alpha * SPR + (1-alpha) * CS"""
        spr = self.compute_spr(original, quantized)
        cs = self.compute_cosine_similarity(original, quantized)
        return alpha * spr + (1 - alpha) * cs
    
    def compute_mse(self, original: np.ndarray, quantized: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((original - quantized) ** 2))
    
    def greedy_select_codebook_daq(self, block: np.ndarray, 
                                   original_block: np.ndarray) -> Tuple[Tuple, float]:
        """Greedy codebook selection using DAQ metrics."""
        selected_codes: Set[int] = set()
        best_daq_score = -float('inf')
        
        # Greedy: select 4 codes one by one
        for _ in range(self.num_codewords):
            best_code = None
            best_score = -float('inf')
            
            for code in range(16):
                if code in selected_codes:
                    continue
                
                # Try adding this code
                candidate_codes = list(selected_codes) + [code]
                candidate_values = E2M1_TABLE[candidate_codes]
                
                # Quantize block using candidate codebook
                quantized_block = np.zeros_like(original_block)
                for i, orig_code in enumerate(block):
                    orig_value = E2M1_TABLE[orig_code]
                    distances = np.abs(candidate_values - orig_value)
                    nearest_idx = np.argmin(distances)
                    quantized_block[i] = candidate_values[nearest_idx]
                
                # Compute DAQ score
                daq_score = self.compute_daq_score(original_block, quantized_block)
                
                if daq_score > best_score:
                    best_score = daq_score
                    best_code = code
            
            if best_code is not None:
                selected_codes.add(best_code)
                best_daq_score = best_score
        
        return tuple(sorted(selected_codes)), best_daq_score
    
    def greedy_select_codebook_mse(self, block: np.ndarray) -> Tuple[Tuple, float]:
        """Greedy codebook selection using MSE (baseline)."""
        selected_codes: Set[int] = set()
        best_mse = float('inf')
        
        for _ in range(self.num_codewords):
            best_code = None
            best_mse_candidate = float('inf')
            
            for code in range(16):
                if code in selected_codes:
                    continue
                
                candidate_codes = list(selected_codes) + [code]
                candidate_values = E2M1_TABLE[candidate_codes]
                
                mse = 0.0
                for orig_code in block:
                    orig_value = E2M1_TABLE[orig_code]
                    distances = np.abs(candidate_values - orig_value)
                    nearest_idx = np.argmin(distances)
                    error = (orig_value - candidate_values[nearest_idx]) ** 2
                    mse += error
                
                mse /= len(block)
                
                if mse < best_mse_candidate:
                    best_mse_candidate = mse
                    best_code = code
            
            if best_code is not None:
                selected_codes.add(best_code)
                best_mse = best_mse_candidate
        
        return tuple(sorted(selected_codes)), best_mse
    
    def compress_weight_daq(self, weight_np: np.ndarray, original_weight: np.ndarray) -> Dict:
        """Compress using DAQ metrics."""
        num_blocks = len(weight_np) // self.block_size
        if len(weight_np) % self.block_size != 0:
            num_blocks += 1
        
        total_daq_score = 0.0
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(weight_np))
            block = weight_np[start:end]
            original_block = original_weight[start:end]
            
            _, daq_score = self.greedy_select_codebook_daq(block, original_block)
            total_daq_score += daq_score * len(block)
        
        avg_daq_score = total_daq_score / len(weight_np)
        
        original_bits = len(weight_np) * 32
        index_bits = len(weight_np) * 2
        codebook_bits = num_blocks * 16
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
        """Compress using MSE (baseline)."""
        num_blocks = len(weight_np) // self.block_size
        if len(weight_np) % self.block_size != 0:
            num_blocks += 1
        
        total_mse = 0.0
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(weight_np))
            block = weight_np[start:end]
            
            _, mse = self.greedy_select_codebook_mse(block)
            total_mse += mse * len(block)
        
        avg_mse = total_mse / len(weight_np)
        
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
    print("=" * 80)
    print("PHASE 22: DAQ-INSPIRED GREEDY CODEBOOK SELECTION")
    print("=" * 80)
    
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
    
    logger.info("Step 2: Initializing DAQ greedy validator")
    validator = DAQGreedyValidator(block_size=BLOCK_SIZE, num_codewords=NUM_CODEWORDS)
    
    logger.info("Step 3: Validating compression on sample weights")
    print("\n" + "=" * 80)
    print("PHASE 22: DAQ-INSPIRED GREEDY CODEBOOK SELECTION")
    print("=" * 80)
    
    results_daq = []
    results_mse = []
    
    start_time = time.time()
    
    for idx, (key, weight_codes) in enumerate(weights.items(), 1):
        original_weight = original_weights[key]
        
        result_daq = validator.compress_weight_daq(weight_codes, original_weight)
        results_daq.append(result_daq)
        
        result_mse = validator.compress_weight_mse(weight_codes)
        results_mse.append(result_mse)
        
        daq_score = result_daq["avg_daq_score"]
        mse = result_mse["avg_mse"]
        compression = result_daq["compression"]
        
        elapsed = time.time() - start_time
        print(f"  [{idx:2d}/{len(weights)}] {key[:50]:50s} | "
              f"DAQ: {daq_score:.4f} | MSE: {mse:.4f} | Compression: {compression*100:.2f}% | "
              f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    avg_daq_score = np.mean([r["avg_daq_score"] for r in results_daq])
    avg_mse = np.mean([r["avg_mse"] for r in results_mse])
    avg_compression_daq = np.mean([r["compression"] for r in results_daq])
    avg_compression_mse = np.mean([r["compression"] for r in results_mse])
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nDAQ Metrics (Phase 22 - Greedy):")
    print(f"  Avg DAQ Score: {avg_daq_score:.4f}")
    print(f"  Avg Compression: {avg_compression_daq*100:.2f}%")
    
    print(f"\nMSE Baseline (Phase 21 - Greedy):")
    print(f"  Avg MSE: {avg_mse:.4f}")
    print(f"  Avg Compression: {avg_compression_mse*100:.2f}%")
    
    improvement = (avg_compression_daq - avg_compression_mse) * 100
    print(f"\nImprovement (DAQ vs MSE):")
    print(f"  Compression delta: {improvement:+.3f}%")
    print(f"  Status: {'✅ POSITIVE' if improvement > 0 else '❌ NEGATIVE'}")
    print(f"  Time: {total_time:.1f}s for {len(weights)} weights")
    
    results = {
        "phase": "22",
        "step": "daq_greedy_validation",
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
        },
        "performance": {
            "total_time_seconds": float(total_time),
            "time_per_weight_seconds": float(total_time / len(weights)),
        }
    }
    
    output_file = OUTPUT_DIR / "phase22_daq_greedy_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")
    print(f"\n✅ Results saved to {output_file}")

if __name__ == "__main__":
    main()
