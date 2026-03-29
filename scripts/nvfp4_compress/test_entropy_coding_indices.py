"""
Test entropy coding on codebook indices.

Hypothesis: Codebook indices follow non-uniform distributions. We can compress
them using entropy coding (Huffman or arithmetic coding) to reduce storage.

Expected improvement: 1-2% storage reduction on index data.
"""

import json
import numpy as np
from collections import Counter
import math

def generate_test_data(num_layers=95, layer_size=1024):
    """Generate synthetic layer data with realistic index distributions."""
    layers = []
    for i in range(num_layers):
        # Simulate different layer types with different distributions
        if i < 30:  # Attention layers - more uniform
            indices = np.random.randint(0, 256, layer_size)
        elif i < 60:  # FFN layers - more skewed (some entries used more)
            # Use Zipfian distribution to simulate skewed usage
            indices = np.random.zipf(1.5, layer_size) % 256
        else:  # Output layers - concentrated (few entries used)
            # Use only 50% of codebook entries
            indices = np.random.randint(0, 128, layer_size)
        
        layers.append(indices)
    return layers

def compute_entropy(indices):
    """Compute Shannon entropy of indices."""
    counts = Counter(indices)
    total = len(indices)
    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def estimate_huffman_compression(indices):
    """Estimate Huffman compression ratio."""
    # Huffman coding achieves compression close to entropy
    entropy = compute_entropy(indices)
    # Huffman typically achieves entropy + small overhead
    # For simplicity, assume it achieves entropy + 0.1 bits/symbol
    bits_per_symbol = entropy + 0.1
    return bits_per_symbol / 8  # Convert to bytes per symbol

def estimate_arithmetic_compression(indices):
    """Estimate arithmetic coding compression ratio."""
    # Arithmetic coding achieves compression very close to entropy
    entropy = compute_entropy(indices)
    # Arithmetic coding overhead is negligible
    bits_per_symbol = entropy
    return bits_per_symbol / 8  # Convert to bytes per symbol

def main():
    print("=" * 70)
    print("ENTROPY CODING ON CODEBOOK INDICES")
    print("=" * 70)
    print()
    
    layers = generate_test_data(num_layers=95, layer_size=1024)
    print(f"Generated {len(layers)} layers\n")
    
    # Analyze entropy distribution
    print("Analyzing index distributions...\n")
    
    total_entropy = 0
    total_huffman_bytes = 0
    total_arithmetic_bytes = 0
    total_uncompressed_bytes = 0
    
    entropy_values = []
    
    for i, indices in enumerate(layers):
        entropy = compute_entropy(indices)
        huffman_bytes = estimate_huffman_compression(indices) * len(indices)
        arithmetic_bytes = estimate_arithmetic_compression(indices) * len(indices)
        uncompressed_bytes = len(indices)  # 1 byte per index (0-255)
        
        total_entropy += entropy
        total_huffman_bytes += huffman_bytes
        total_arithmetic_bytes += arithmetic_bytes
        total_uncompressed_bytes += uncompressed_bytes
        entropy_values.append(entropy)
        
        if (i + 1) % 20 == 0:
            print(f"  Layer {i+1:2d}: entropy = {entropy:.3f} bits/symbol, "
                  f"Huffman = {huffman_bytes:.0f} bytes, "
                  f"Arithmetic = {arithmetic_bytes:.0f} bytes")
    
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    avg_entropy = total_entropy / len(layers)
    
    print(f"\nEntropy analysis:")
    print(f"  Average entropy:             {avg_entropy:.4f} bits/symbol")
    print(f"  Min entropy:                 {min(entropy_values):.4f} bits/symbol")
    print(f"  Max entropy:                 {max(entropy_values):.4f} bits/symbol")
    
    print(f"\nStorage analysis (95 layers × 1024 indices):")
    print(f"  Uncompressed (1 byte/index): {int(total_uncompressed_bytes):,} bytes")
    print(f"  Huffman coding:              {int(total_huffman_bytes):,} bytes")
    print(f"  Arithmetic coding:           {int(total_arithmetic_bytes):,} bytes")
    
    huffman_reduction = (1 - total_huffman_bytes / total_uncompressed_bytes) * 100
    arithmetic_reduction = (1 - total_arithmetic_bytes / total_uncompressed_bytes) * 100
    
    print(f"\nCompression ratios:")
    print(f"  Huffman reduction:           {huffman_reduction:.2f}%")
    print(f"  Arithmetic reduction:        {arithmetic_reduction:.2f}%")
    
    # Save results
    results = {
        'test': 'entropy_coding_indices',
        'num_layers': len(layers),
        'average_entropy_bits_per_symbol': float(avg_entropy),
        'min_entropy': float(min(entropy_values)),
        'max_entropy': float(max(entropy_values)),
        'uncompressed_bytes': int(total_uncompressed_bytes),
        'huffman_bytes': int(total_huffman_bytes),
        'arithmetic_bytes': int(total_arithmetic_bytes),
        'huffman_reduction_percent': float(huffman_reduction),
        'arithmetic_reduction_percent': float(arithmetic_reduction)
    }
    
    with open('test_entropy_coding_indices_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to test_entropy_coding_indices_results.json")
    
    # Recommendation
    print("\n" + "=" * 70)
    if arithmetic_reduction > 1.0:
        print("✅ ENTROPY CODING RECOMMENDED")
        print(f"   - Compression: {arithmetic_reduction:.2f}% reduction")
        print(f"   - Recommended: Arithmetic coding (better compression)")
    else:
        print("⚠️  ENTROPY CODING NOT RECOMMENDED")
        print(f"   - Compression too low: {arithmetic_reduction:.2f}%")
        print(f"   - Decompression overhead may exceed savings")

if __name__ == '__main__':
    main()
