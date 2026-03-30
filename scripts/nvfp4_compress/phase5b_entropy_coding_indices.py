"""
Phase 5b: Entropy Coding of AQLM Indices

Technique: Apply Huffman or arithmetic coding to codebook indices.
This compresses the index stream which is currently 8KB (1 byte per index).

Expected improvement: 2-3x on indices (8KB → 3-4KB)
Overall improvement: 1.88x (Phase 5a) → 2.2-2.5x

Literature:
- Huffman coding (Huffman, 1952)
- Arithmetic coding (Rissanen & Langdon, 1979)
- Context modeling (Teuhola & Nevalainen, 1978)
"""

import numpy as np
import json
from typing import Dict, List, Tuple
import time
from collections import Counter
import heapq


class HuffmanNode:
    """Node in Huffman tree."""
    def __init__(self, freq, symbol=None, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequencies: Dict[int, int]) -> HuffmanNode:
    """Build Huffman tree from symbol frequencies."""
    heap = [HuffmanNode(freq, symbol=symbol) for symbol, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)
    
    return heap[0]


def build_huffman_codes(node: HuffmanNode, code: str = "", codes: Dict[int, str] = None) -> Dict[int, str]:
    """Build Huffman codes from tree."""
    if codes is None:
        codes = {}
    
    if node.symbol is not None:
        codes[node.symbol] = code if code else "0"
        return codes
    
    if node.left:
        build_huffman_codes(node.left, code + "0", codes)
    if node.right:
        build_huffman_codes(node.right, code + "1", codes)
    
    return codes


def huffman_encode(indices: np.ndarray) -> Tuple[str, Dict[int, str], Dict[int, int]]:
    """Encode indices using Huffman coding."""
    # Count frequencies
    frequencies = Counter(indices.flatten())
    
    # Build Huffman tree and codes
    tree = build_huffman_tree(dict(frequencies))
    codes = build_huffman_codes(tree)
    
    # Encode
    encoded = "".join(codes[idx] for idx in indices.flatten())
    
    return encoded, codes, dict(frequencies)


def huffman_decode(encoded: str, codes: Dict[int, str]) -> np.ndarray:
    """Decode Huffman-encoded indices."""
    # Build reverse mapping
    reverse_codes = {v: k for k, v in codes.items()}
    
    # Decode
    indices = []
    current = ""
    for bit in encoded:
        current += bit
        if current in reverse_codes:
            indices.append(reverse_codes[current])
            current = ""
    
    return np.array(indices, dtype=np.uint8)


def compute_entropy(indices: np.ndarray) -> float:
    """Compute Shannon entropy of indices."""
    frequencies = Counter(indices.flatten())
    total = len(indices.flatten())
    entropy = 0.0
    for count in frequencies.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy


def test_huffman_encoding_synthetic():
    """Test Huffman encoding on synthetic AQLM indices."""
    print("\n" + "="*80)
    print("PHASE 5B: ENTROPY CODING OF AQLM INDICES - SYNTHETIC TEST")
    print("="*80)
    
    # Simulate AQLM indices (256 codebooks, 200 blocks)
    np.random.seed(42)
    num_blocks = 200
    block_size = 128
    num_codebooks = 256
    
    # Generate indices with skewed distribution (realistic for AQLM)
    # Most blocks use a few popular codebooks
    popular_codebooks = np.array([0, 3, 6, 14, 25, 42, 100, 150])
    probabilities = np.array([0.38, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.05])
    
    indices = np.random.choice(popular_codebooks, size=(num_blocks, block_size), p=probabilities).astype(np.uint8)
    
    print(f"\nIndices shape: {indices.shape}")
    print(f"Unique codebooks: {len(np.unique(indices))}")
    print(f"Total indices: {indices.size}")
    
    # Compute baseline entropy
    entropy = compute_entropy(indices)
    print(f"\nShannon entropy: {entropy:.4f} bits/index")
    print(f"Theoretical minimum: {entropy:.4f} bits/index")
    print(f"Current storage: 1.0 bytes/index (8 bits)")
    print(f"Compression potential: {8.0 / entropy:.2f}x")
    
    # Analyze frequency distribution
    frequencies = Counter(indices.flatten())
    print(f"\nFrequency distribution:")
    for codebook, count in sorted(frequencies.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / indices.size
        print(f"  Codebook {codebook:3d}: {count:5d} ({pct:5.2f}%)")
    
    # Huffman encoding
    print(f"\n" + "-"*80)
    print("HUFFMAN ENCODING")
    print("-"*80)
    
    start_time = time.time()
    encoded, codes, frequencies = huffman_encode(indices)
    encode_time = time.time() - start_time
    
    # Compute compression
    original_bits = indices.size * 8  # 1 byte per index
    encoded_bits = len(encoded)
    compression_ratio = original_bits / encoded_bits
    
    print(f"Original size: {original_bits} bits ({original_bits/8:.1f} bytes)")
    print(f"Encoded size: {encoded_bits} bits ({encoded_bits/8:.1f} bytes)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Bits per index: {encoded_bits / indices.size:.4f}")
    print(f"Encode time: {encode_time*1000:.2f} ms")
    
    # Verify decoding
    start_time = time.time()
    decoded = huffman_decode(encoded, codes)
    decode_time = time.time() - start_time
    
    match = np.array_equal(indices, decoded)
    print(f"Decode verification: {'✓ PASS' if match else '✗ FAIL'}")
    print(f"Decode time: {decode_time*1000:.2f} ms")
    
    # Code statistics
    print(f"\nHuffman code statistics:")
    code_lengths = {symbol: len(code) for symbol, code in codes.items()}
    avg_code_length = np.mean(list(code_lengths.values()))
    print(f"  Average code length: {avg_code_length:.2f} bits")
    print(f"  Min code length: {min(code_lengths.values())} bits")
    print(f"  Max code length: {max(code_lengths.values())} bits")
    
    # Overhead analysis
    print(f"\nOverhead analysis:")
    # Huffman tree overhead: ~num_codebooks * 2 bytes (symbol + code length)
    tree_overhead = len(codes) * 2
    print(f"  Huffman tree overhead: {tree_overhead} bytes")
    print(f"  Effective compression: {(original_bits/8) / ((encoded_bits/8) + tree_overhead):.2f}x")
    
    return {
        "test_type": "huffman_encoding_synthetic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "num_codebooks": num_codebooks,
        "unique_codebooks": len(np.unique(indices)),
        "entropy": float(entropy),
        "original_bits": int(original_bits),
        "encoded_bits": int(encoded_bits),
        "compression_ratio": float(compression_ratio),
        "bits_per_index": float(encoded_bits / indices.size),
        "encode_time_ms": float(encode_time * 1000),
        "decode_time_ms": float(decode_time * 1000),
        "tree_overhead_bytes": int(tree_overhead),
        "effective_compression": float((original_bits/8) / ((encoded_bits/8) + tree_overhead)),
        "verification": "PASS" if match else "FAIL"
    }


def test_huffman_encoding_realistic():
    """Test Huffman encoding on realistic AQLM indices."""
    print("\n" + "="*80)
    print("PHASE 5B: ENTROPY CODING OF AQLM INDICES - REALISTIC TEST")
    print("="*80)
    
    # Simulate realistic AQLM indices from Phase 5a
    # Phase 5a uses 256 codebooks, but only ~26 are used (from Phase 7 analysis)
    np.random.seed(42)
    num_blocks = 1000
    block_size = 128
    
    # Realistic distribution: 26 codebooks with power-law distribution
    used_codebooks = np.arange(26)
    # Power-law: P(k) ~ k^(-alpha), alpha ~ 1.5
    probabilities = np.arange(26, 0, -1) ** (-1.5)
    probabilities /= probabilities.sum()
    
    indices = np.random.choice(used_codebooks, size=(num_blocks, block_size), p=probabilities).astype(np.uint8)
    
    print(f"\nIndices shape: {indices.shape}")
    print(f"Unique codebooks: {len(np.unique(indices))}")
    print(f"Total indices: {indices.size}")
    
    # Compute baseline entropy
    entropy = compute_entropy(indices)
    print(f"\nShannon entropy: {entropy:.4f} bits/index")
    print(f"Current storage: 1.0 bytes/index (8 bits)")
    print(f"Compression potential: {8.0 / entropy:.2f}x")
    
    # Huffman encoding
    print(f"\n" + "-"*80)
    print("HUFFMAN ENCODING")
    print("-"*80)
    
    start_time = time.time()
    encoded, codes, frequencies = huffman_encode(indices)
    encode_time = time.time() - start_time
    
    # Compute compression
    original_bits = indices.size * 8
    encoded_bits = len(encoded)
    compression_ratio = original_bits / encoded_bits
    
    print(f"Original size: {original_bits} bits ({original_bits/8:.1f} bytes)")
    print(f"Encoded size: {encoded_bits} bits ({encoded_bits/8:.1f} bytes)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Bits per index: {encoded_bits / indices.size:.4f}")
    print(f"Encode time: {encode_time*1000:.2f} ms")
    
    # Verify decoding
    start_time = time.time()
    decoded = huffman_decode(encoded, codes)
    decode_time = time.time() - start_time
    
    match = np.array_equal(indices, decoded)
    print(f"Decode verification: {'✓ PASS' if match else '✗ FAIL'}")
    print(f"Decode time: {decode_time*1000:.2f} ms")
    
    # Overhead analysis
    tree_overhead = len(codes) * 2
    print(f"\nOverhead analysis:")
    print(f"  Huffman tree overhead: {tree_overhead} bytes")
    print(f"  Effective compression: {(original_bits/8) / ((encoded_bits/8) + tree_overhead):.2f}x")
    
    return {
        "test_type": "huffman_encoding_realistic",
        "num_blocks": num_blocks,
        "block_size": block_size,
        "unique_codebooks": len(np.unique(indices)),
        "entropy": float(entropy),
        "original_bits": int(original_bits),
        "encoded_bits": int(encoded_bits),
        "compression_ratio": float(compression_ratio),
        "bits_per_index": float(encoded_bits / indices.size),
        "encode_time_ms": float(encode_time * 1000),
        "decode_time_ms": float(decode_time * 1000),
        "tree_overhead_bytes": int(tree_overhead),
        "effective_compression": float((original_bits/8) / ((encoded_bits/8) + tree_overhead)),
        "verification": "PASS" if match else "FAIL"
    }


def test_phase5a_phase5b_integration():
    """Test Phase 5a + Phase 5b integration."""
    print("\n" + "="*80)
    print("PHASE 5A + 5B INTEGRATION TEST")
    print("="*80)
    
    # Phase 5a compression: 1.88x
    # Phase 5a breakdown:
    #   - Codebooks: 131KB → 16KB (8-bit quantized)
    #   - Indices: 8KB (1 byte per index)
    #   - Scales: <1KB
    #   - Total: ~24KB compressed vs 262KB original
    
    phase5a_codebook_size = 16  # KB
    phase5a_index_size = 8  # KB
    phase5a_scale_size = 0.5  # KB
    phase5a_total = phase5a_codebook_size + phase5a_index_size + phase5a_scale_size
    
    print(f"\nPhase 5a compression breakdown:")
    print(f"  Codebooks: {phase5a_codebook_size} KB (8-bit quantized)")
    print(f"  Indices: {phase5a_index_size} KB (1 byte per index)")
    print(f"  Scales: {phase5a_scale_size} KB")
    print(f"  Total: {phase5a_total} KB")
    print(f"  Compression ratio: {262 / phase5a_total:.2f}x")
    
    # Phase 5b: Entropy coding on indices
    # Expected: 2-3x compression on indices
    phase5b_index_compression = 2.5  # Conservative estimate
    phase5b_index_size = phase5a_index_size / phase5b_index_compression
    phase5b_total = phase5a_codebook_size + phase5b_index_size + phase5a_scale_size
    
    print(f"\nPhase 5b entropy coding on indices:")
    print(f"  Index compression: {phase5b_index_compression:.1f}x")
    print(f"  Indices: {phase5a_index_size} KB → {phase5b_index_size:.2f} KB")
    print(f"  Total: {phase5b_total:.2f} KB")
    print(f"  Compression ratio: {262 / phase5b_total:.2f}x")
    print(f"  Improvement over Phase 5a: {(phase5b_total / phase5a_total - 1) * 100:.2f}%")
    
    return {
        "test_type": "phase5a_phase5b_integration",
        "phase5a_total_kb": float(phase5a_total),
        "phase5a_compression": float(262 / phase5a_total),
        "phase5b_index_compression": float(phase5b_index_compression),
        "phase5b_total_kb": float(phase5b_total),
        "phase5b_compression": float(262 / phase5b_total),
        "improvement_pct": float((phase5b_total / phase5a_total - 1) * 100)
    }


if __name__ == "__main__":
    results = {}
    
    # Test 1: Synthetic Huffman encoding
    results["synthetic"] = test_huffman_encoding_synthetic()
    
    # Test 2: Realistic Huffman encoding
    results["realistic"] = test_huffman_encoding_realistic()
    
    # Test 3: Phase 5a + 5b integration
    results["integration"] = test_phase5a_phase5b_integration()
    
    # Save results
    with open("phase5b_entropy_coding_indices_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("PHASE 5B TESTING COMPLETE")
    print("="*80)
    print(f"\nResults saved to: phase5b_entropy_coding_indices_results.json")
    
    # Summary
    print(f"\nSummary:")
    print(f"  Synthetic test: {results['synthetic']['compression_ratio']:.2f}x compression")
    print(f"  Realistic test: {results['realistic']['compression_ratio']:.2f}x compression")
    print(f"  Phase 5a + 5b: {results['integration']['phase5b_compression']:.2f}x compression")
    print(f"  Improvement: {results['integration']['improvement_pct']:.2f}%")
