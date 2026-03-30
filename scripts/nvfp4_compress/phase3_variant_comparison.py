#!/usr/bin/env python3
"""
Phase 3: Codebook Variant Comparison Framework

Systematically compares all 4 codebook-selection variants:
- Variant A: Exact MSE (brute-force baseline)
- Variant B: Weighted MSE (EM-based)
- Variant C: Frequency-regularized MSE
- Variant D: Signed-pair constrained (fastest)
"""

import torch
import numpy as np
from itertools import combinations
from collections import Counter
import json
import time
from pathlib import Path
from typing import Tuple, List, Dict

# FP4 E2M1 code table
E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)

class CodebookVariantTester:
    """Framework for testing codebook selection variants."""
    
    def __init__(self, block_size: int = 128, num_codewords: int = 4):
        self.block_size = block_size
        self.num_codewords = num_codewords
        self.all_codes = np.arange(16)
        self.all_codebooks = list(combinations(self.all_codes, num_codewords))
        print(f"Total possible {num_codewords}-entry codebooks: {len(self.all_codebooks)}")
    
    def code_to_value(self, code: int) -> float:
        """Convert single FP4 code to float value."""
        return E2M1_TABLE[code]
    
    def codes_to_values(self, codes: np.ndarray) -> np.ndarray:
        """Convert FP4 codes to float values."""
        return np.array([E2M1_TABLE[c] for c in codes])
    
    def quantize_block(self, block: np.ndarray, codebook: Tuple) -> Tuple[np.ndarray, float]:
        """Quantize block using given codebook, return quantized block and MSE."""
        codebook_array = np.array(codebook)
        codebook_values = self.codes_to_values(codebook_array)
        
        quantized = np.zeros_like(block, dtype=np.float32)
        for i, code in enumerate(block):
            code_value = self.code_to_value(code)
            distances = np.abs(codebook_values - code_value)
            nearest_idx = np.argmin(distances)
            quantized[i] = codebook_values[nearest_idx]
        
        mse = np.mean((self.codes_to_values(block) - quantized) ** 2)
        return quantized, mse
    
    def variant_a_exact_mse(self, fp4_codes: np.ndarray) -> Dict:
        """Variant A: Brute-force exact MSE baseline."""
        print("\n[Variant A] Exact MSE - Brute-force enumeration")
        
        num_blocks = len(fp4_codes) // self.block_size
        total_mse = 0.0
        codebook_usage = Counter()
        
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(fp4_codes))
            block = fp4_codes[start:end]
            
            best_mse = float('inf')
            best_codebook = None
            
            for codebook in self.all_codebooks:
                _, mse = self.quantize_block(block, codebook)
                if mse < best_mse:
                    best_mse = mse
                    best_codebook = codebook
            
            total_mse += best_mse * len(block)
            codebook_usage[best_codebook] += 1
            
            if (block_idx + 1) % 10 == 0:
                print(f"  Processed {block_idx + 1}/{num_blocks} blocks")
        
        elapsed = time.time() - start_time
        avg_mse = total_mse / len(fp4_codes)
        
        return {
            'variant': 'A_exact_mse',
            'method': 'Brute-force enumeration',
            'avg_mse': float(avg_mse),
            'num_blocks': num_blocks,
            'unique_codebooks': len(codebook_usage),
            'elapsed_sec': elapsed,
            'codes_per_sec': len(fp4_codes) / elapsed,
        }
    
    def variant_b_weighted_mse(self, fp4_codes: np.ndarray) -> Dict:
        """Variant B: Weighted MSE using code frequency."""
        print("\n[Variant B] Weighted MSE - Frequency-weighted optimization")
        
        num_blocks = len(fp4_codes) // self.block_size
        total_mse = 0.0
        codebook_usage = Counter()
        
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(fp4_codes))
            block = fp4_codes[start:end]
            
            # Compute code frequencies in block
            code_counts = Counter(block)
            code_weights = np.array([code_counts.get(i, 0) for i in range(16)]) / len(block)
            
            best_weighted_mse = float('inf')
            best_codebook = None
            
            for codebook in self.all_codebooks:
                codebook_array = np.array(codebook)
                codebook_values = self.codes_to_values(codebook_array)
                
                weighted_mse = 0.0
                for code in range(16):
                    if code_weights[code] > 0:
                        code_value = self.code_to_value(code)
                        distances = np.abs(codebook_values - code_value)
                        nearest_idx = np.argmin(distances)
                        error = (code_value - codebook_values[nearest_idx]) ** 2
                        weighted_mse += code_weights[code] * error
                
                if weighted_mse < best_weighted_mse:
                    best_weighted_mse = weighted_mse
                    best_codebook = codebook
            
            total_mse += best_weighted_mse * len(block)
            codebook_usage[best_codebook] += 1
            
            if (block_idx + 1) % 10 == 0:
                print(f"  Processed {block_idx + 1}/{num_blocks} blocks")
        
        elapsed = time.time() - start_time
        avg_mse = total_mse / len(fp4_codes)
        
        return {
            'variant': 'B_weighted_mse',
            'method': 'Frequency-weighted MSE',
            'avg_mse': float(avg_mse),
            'num_blocks': num_blocks,
            'unique_codebooks': len(codebook_usage),
            'elapsed_sec': elapsed,
            'codes_per_sec': len(fp4_codes) / elapsed,
        }
    
    def variant_c_frequency_regularized(self, fp4_codes: np.ndarray, lambda_reg: float = 0.01) -> Dict:
        """Variant C: Frequency-regularized MSE with code utilization penalty."""
        print(f"\n[Variant C] Frequency-regularized MSE (λ={lambda_reg})")
        
        num_blocks = len(fp4_codes) // self.block_size
        total_mse = 0.0
        codebook_usage = Counter()
        
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(fp4_codes))
            block = fp4_codes[start:end]
            
            code_counts = Counter(block)
            code_weights = np.array([code_counts.get(i, 0) for i in range(16)]) / len(block)
            
            best_loss = float('inf')
            best_codebook = None
            
            for codebook in self.all_codebooks:
                codebook_array = np.array(codebook)
                codebook_values = self.codes_to_values(codebook_array)
                
                # MSE term
                weighted_mse = 0.0
                for code in range(16):
                    if code_weights[code] > 0:
                        code_value = self.code_to_value(code)
                        distances = np.abs(codebook_values - code_value)
                        nearest_idx = np.argmin(distances)
                        error = (code_value - codebook_values[nearest_idx]) ** 2
                        weighted_mse += code_weights[code] * error
                
                # Regularization: penalize unused codes
                num_used_codes = len(codebook)
                regularization = lambda_reg * (self.num_codewords - num_used_codes)
                
                loss = weighted_mse + regularization
                
                if loss < best_loss:
                    best_loss = loss
                    best_codebook = codebook
            
            total_mse += best_loss * len(block)
            codebook_usage[best_codebook] += 1
            
            if (block_idx + 1) % 10 == 0:
                print(f"  Processed {block_idx + 1}/{num_blocks} blocks")
        
        elapsed = time.time() - start_time
        avg_mse = total_mse / len(fp4_codes)
        
        return {
            'variant': 'C_frequency_regularized',
            'method': f'Frequency-regularized MSE (λ={lambda_reg})',
            'avg_mse': float(avg_mse),
            'num_blocks': num_blocks,
            'unique_codebooks': len(codebook_usage),
            'elapsed_sec': elapsed,
            'codes_per_sec': len(fp4_codes) / elapsed,
        }
    
    def variant_d_signed_pair_constrained(self, fp4_codes: np.ndarray) -> Dict:
        """Variant D: Signed-pair constrained search (symmetric codebooks)."""
        print("\n[Variant D] Signed-pair constrained - Symmetric codebook search")
        
        # Generate only symmetric codebooks: if code c is selected, -c must also be selected
        # For FP4 E2M1: codes 0-7 are positive, 8-15 are negative (8=0, 9=-0.5, etc.)
        symmetric_codebooks = []
        for combo in combinations(range(8), self.num_codewords // 2):
            # Add both positive and negative codes
            full_combo = list(combo) + [c + 8 for c in combo]
            symmetric_codebooks.append(tuple(sorted(full_combo)))
        
        symmetric_codebooks = list(set(symmetric_codebooks))
        print(f"  Symmetric codebooks: {len(symmetric_codebooks)} (vs {len(self.all_codebooks)} total)")
        
        num_blocks = len(fp4_codes) // self.block_size
        total_mse = 0.0
        codebook_usage = Counter()
        
        start_time = time.time()
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = min(start + self.block_size, len(fp4_codes))
            block = fp4_codes[start:end]
            
            best_mse = float('inf')
            best_codebook = None
            
            for codebook in symmetric_codebooks:
                _, mse = self.quantize_block(block, codebook)
                if mse < best_mse:
                    best_mse = mse
                    best_codebook = codebook
            
            total_mse += best_mse * len(block)
            codebook_usage[best_codebook] += 1
            
            if (block_idx + 1) % 10 == 0:
                print(f"  Processed {block_idx + 1}/{num_blocks} blocks")
        
        elapsed = time.time() - start_time
        avg_mse = total_mse / len(fp4_codes)
        
        return {
            'variant': 'D_signed_pair_constrained',
            'method': 'Signed-pair constrained search',
            'avg_mse': float(avg_mse),
            'num_blocks': num_blocks,
            'unique_codebooks': len(codebook_usage),
            'elapsed_sec': elapsed,
            'codes_per_sec': len(fp4_codes) / elapsed,
            'search_space_reduction': len(symmetric_codebooks) / len(self.all_codebooks),
        }
    
    def run_all_variants(self, fp4_codes: np.ndarray) -> Dict:
        """Run all 4 variants and compare."""
        results = {}
        
        results['A'] = self.variant_a_exact_mse(fp4_codes)
        results['B'] = self.variant_b_weighted_mse(fp4_codes)
        results['C'] = self.variant_c_frequency_regularized(fp4_codes)
        results['D'] = self.variant_d_signed_pair_constrained(fp4_codes)
        
        return results

def main():
    print("=" * 80)
    print("Phase 3: Codebook Variant Comparison")
    print("=" * 80)
    
    # Generate synthetic FP4 codes
    print("\nGenerating synthetic FP4 codes...")
    np.random.seed(42)
    
    # Simulate realistic FP4 code distribution
    codes_list = []
    for _ in range(100):  # 100 blocks
        # Positive codes (0-7) are more common
        block = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=100, 
                                p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05])
        # Negative codes (8-15) are less common
        block = np.concatenate([block, np.random.choice([8, 9, 10, 11, 12, 13, 14, 15], size=28)])
        codes_list.extend(block)
    
    fp4_codes = np.array(codes_list, dtype=np.uint8)
    print(f"Generated {len(fp4_codes)} codes")
    
    # Run comparison
    tester = CodebookVariantTester(block_size=128, num_codewords=4)
    results = tester.run_all_variants(fp4_codes)
    
    # Print comparison
    print("\n" + "=" * 80)
    print("VARIANT COMPARISON RESULTS")
    print("=" * 80)
    
    print("\n{:<10} {:<30} {:<12} {:<12} {:<12}".format(
        "Variant", "Method", "MSE", "Elapsed (s)", "Codes/sec"))
    print("-" * 80)
    
    for variant in ['A', 'B', 'C', 'D']:
        result = results[variant]
        print("{:<10} {:<30} {:<12.6f} {:<12.2f} {:<12.0f}".format(
            variant,
            result['method'][:28],
            result['avg_mse'],
            result['elapsed_sec'],
            result['codes_per_sec']
        ))
    
    # Compression estimates
    print("\n" + "=" * 80)
    print("COMPRESSION ESTIMATES (4-entry codebook)")
    print("=" * 80)
    
    bits_per_elem = 2.0 + 0.5/128  # 2 bits for code index + 0.5 bits for codebook overhead
    
    print(f"\nBits per element: {bits_per_elem:.4f}")
    print(f"Compression ratio: {4.0 / bits_per_elem:.2f}x")
    
    print("\nVariant Performance:")
    for variant in ['A', 'B', 'C', 'D']:
        result = results[variant]
        print(f"  {variant}: MSE={result['avg_mse']:.6f}, "
              f"Time={result['elapsed_sec']:.1f}s, "
              f"Codebooks={result['unique_codebooks']}")
    
    # Save results
    output_file = Path('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase3_variant_comparison_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("Phase 3 Complete")
    print("=" * 80)

if __name__ == '__main__':
    main()
