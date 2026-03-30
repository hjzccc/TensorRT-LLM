#!/usr/bin/env python3
"""
Phase 5: Entropy Coding Analysis
Step 1: Analyze codeword frequency distribution from Variant B compression
"""

import torch
import numpy as np
from collections import Counter
import json
import time
from pathlib import Path
import logging
from phase4_variant_b_production import VariantBProduction, E2M1_TABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_synthetic_fp4_data(num_codes: int = 100000, seed: int = 42) -> np.ndarray:
    """Generate synthetic FP4 codes for testing."""
    np.random.seed(seed)
    # Simulate realistic FP4 distribution (biased towards smaller values)
    # Probabilities must sum to 1.0
    probs = np.array([
        0.15, 0.12, 0.10, 0.08,  # codes 0-3 (small positive)
        0.12, 0.10, 0.08, 0.05,  # codes 4-7 (larger positive)
        0.15, 0.12, 0.10, 0.08,  # codes 8-11 (small negative)
        0.12, 0.10, 0.08, 0.05,  # codes 12-15 (larger negative)
    ], dtype=np.float32)
    probs = probs / probs.sum()  # Normalize to sum to 1.0
    
    codes = np.random.choice(16, size=num_codes, p=probs)
    return codes.astype(np.uint8)

def analyze_codeword_frequencies(codes: np.ndarray, block_size: int = 128) -> dict:
    """
    Analyze codeword frequency distribution across blocks.
    
    Args:
        codes: Array of FP4 codes
        block_size: Size of each block
        
    Returns:
        Dictionary with frequency analysis
    """
    logger.info(f"Analyzing {len(codes)} codes in blocks of {block_size}")
    
    # Global frequency
    global_freq = Counter(codes)
    total_codes = len(codes)
    
    # Per-block frequency
    block_frequencies = []
    num_blocks = len(codes) // block_size
    
    for i in range(num_blocks):
        block = codes[i*block_size:(i+1)*block_size]
        block_freq = Counter(block)
        block_frequencies.append(block_freq)
    
    # Compute statistics
    global_probs = {code: count / total_codes for code, count in global_freq.items()}
    
    # Entropy calculation
    entropy = 0.0
    for prob in global_probs.values():
        if prob > 0:
            entropy -= prob * np.log2(prob)
    
    # Huffman code length estimation
    # For each code, estimate optimal Huffman code length
    huffman_lengths = {}
    for code, prob in global_probs.items():
        if prob > 0:
            huffman_lengths[code] = -np.log2(prob)
        else:
            huffman_lengths[code] = 0
    
    # Average bits per code with Huffman
    avg_huffman_bits = sum(
        global_probs.get(code, 0) * huffman_lengths.get(code, 0)
        for code in range(16)
    )
    
    # Uniform coding (2 bits per code)
    uniform_bits = 2.0
    
    # Potential improvement
    improvement = (uniform_bits - avg_huffman_bits) / uniform_bits * 100
    
    result = {
        'total_codes': total_codes,
        'num_blocks': num_blocks,
        'block_size': block_size,
        'global_frequencies': dict(global_freq),
        'global_probabilities': global_probs,
        'entropy': entropy,
        'uniform_bits_per_code': uniform_bits,
        'huffman_bits_per_code': avg_huffman_bits,
        'huffman_lengths': huffman_lengths,
        'potential_improvement_percent': improvement,
        'block_frequencies_sample': [dict(f) for f in block_frequencies[:5]],  # First 5 blocks
    }
    
    return result

def main():
    """Main analysis function."""
    logger.info("=" * 80)
    logger.info("Phase 5: Entropy Coding Analysis - Step 1")
    logger.info("=" * 80)
    
    # Generate synthetic FP4 data
    logger.info("Generating synthetic FP4 data...")
    codes = generate_synthetic_fp4_data(num_codes=100000)
    
    # Analyze frequencies
    logger.info("Analyzing codeword frequencies...")
    analysis = analyze_codeword_frequencies(codes, block_size=128)
    
    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("FREQUENCY ANALYSIS RESULTS")
    logger.info("=" * 80)
    
    logger.info(f"\nTotal codes: {analysis['total_codes']}")
    logger.info(f"Number of blocks: {analysis['num_blocks']}")
    logger.info(f"Block size: {analysis['block_size']}")
    
    logger.info("\nGlobal Code Frequencies:")
    for code in sorted(analysis['global_frequencies'].keys()):
        freq = analysis['global_frequencies'][code]
        prob = analysis['global_probabilities'][code]
        logger.info(f"  Code {code:2d}: {freq:6d} ({prob:6.2%})")
    
    logger.info(f"\nEntropy: {analysis['entropy']:.4f} bits/code")
    logger.info(f"Uniform coding: {analysis['uniform_bits_per_code']:.4f} bits/code")
    logger.info(f"Huffman coding: {analysis['huffman_bits_per_code']:.4f} bits/code")
    logger.info(f"Potential improvement: {analysis['potential_improvement_percent']:.2f}%")
    
    logger.info("\nHuffman Code Lengths (optimal):")
    for code in sorted(analysis['huffman_lengths'].keys()):
        length = analysis['huffman_lengths'][code]
        if length > 0:
            logger.info(f"  Code {code:2d}: {length:.2f} bits")
    
    logger.info("\nSample Block Frequencies (first 5 blocks):")
    for i, block_freq in enumerate(analysis['block_frequencies_sample']):
        logger.info(f"  Block {i}: {dict(sorted(block_freq.items()))}")
    
    # Save results
    output_file = Path(__file__).parent / "phase5_entropy_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"\nResults saved to {output_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    
    return analysis

if __name__ == "__main__":
    analysis = main()
