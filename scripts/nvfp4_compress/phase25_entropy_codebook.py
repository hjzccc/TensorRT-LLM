#!/usr/bin/env python3
"""
Phase 25A: Huffman Coding of Per-Block Codebook Entries

Applies Huffman coding to the 3 stored FP4 codes per block in the
2b075b_zero_fixed_weighted_abs scheme.

Key finding:
- Codebook entries have entropy 3.117 bits (vs 4.0 current)
- Savings: 0.883 bits/entry × 3 entries / 16 elements = 0.165 bpe
- New total: 2.75 - 0.165 = 2.585 bpe (6.0% improvement, LOSSLESS)

This is different from previous entropy coding attempts which targeted
the old K-means scheme with near-uniform distribution (no headroom).
"""

from __future__ import annotations

import heapq
import json
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

COMPRESSED_DIR = Path("compressed_2b075b_zero_fixed_weighted_abs")
OUTPUT_DIR = Path("compressed_2b075b_entropy_codebook")
BLOCK_SIZE = 16


# ── Huffman coding ──────────────────────────────────────────────────────────

class HuffmanNode:
    def __init__(self, symbol: Optional[int], freq: int):
        self.symbol = symbol
        self.freq = freq
        self.left: Optional[HuffmanNode] = None
        self.right: Optional[HuffmanNode] = None

    def __lt__(self, other: "HuffmanNode") -> bool:
        return self.freq < other.freq


def build_huffman_tree(freq_table: Dict[int, int]) -> HuffmanNode:
    heap = [HuffmanNode(sym, freq) for sym, freq in freq_table.items() if freq > 0]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(None, left.freq + right.freq)
        parent.left = left
        parent.right = right
        heapq.heappush(heap, parent)
    return heap[0]


def build_codebook(node: HuffmanNode, prefix: str = "") -> Dict[int, str]:
    if node.symbol is not None:
        return {node.symbol: prefix or "0"}
    codes = {}
    if node.left:
        codes.update(build_codebook(node.left, prefix + "0"))
    if node.right:
        codes.update(build_codebook(node.right, prefix + "1"))
    return codes


def huffman_encode(symbols: np.ndarray, codebook: Dict[int, str]) -> bytes:
    """Encode symbols to bytes using Huffman codes."""
    bits = "".join(codebook[int(s)] for s in symbols)
    # Pad to byte boundary
    pad = (8 - len(bits) % 8) % 8
    bits += "0" * pad
    # Pack into bytes
    result = bytearray()
    result.append(pad)  # first byte = padding length
    for i in range(0, len(bits), 8):
        result.append(int(bits[i:i+8], 2))
    return bytes(result)


def huffman_decode(data: bytes, codebook_inv: Dict[str, int], n_symbols: int) -> np.ndarray:
    """Decode bytes back to symbols using inverse Huffman codebook."""
    pad = data[0]
    bits = "".join(f"{b:08b}" for b in data[1:])
    if pad > 0:
        bits = bits[:-pad]
    
    symbols = []
    current = ""
    for bit in bits:
        current += bit
        if current in codebook_inv:
            symbols.append(codebook_inv[current])
            current = ""
        if len(symbols) == n_symbols:
            break
    
    return np.array(symbols, dtype=np.uint8)


# ── Analysis ────────────────────────────────────────────────────────────────

def measure_codebook_entry_distribution(compressed_dir: Path, max_shards: int = 50) -> Dict[int, int]:
    """Measure frequency of each FP4 code in codebook entries."""
    counts: Dict[int, int] = {i: 0 for i in range(16)}
    
    index_file = compressed_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    
    shards_seen = set()
    n_shards = 0
    
    for key, shard_file in index["weight_map"].items():
        if not key.endswith("weight_codebook_entries"):
            continue
        if shard_file in shards_seen:
            continue
        shards_seen.add(shard_file)
        if n_shards >= max_shards:
            break
        
        shard_path = compressed_dir / shard_file
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if not k.endswith("weight_codebook_entries"):
                        continue
                    t = f.get_tensor(k).numpy().flatten()
                    lo = t & 0x0F
                    hi = (t >> 4) & 0x0F
                    for code in np.concatenate([lo, hi]):
                        counts[int(code)] += 1
            n_shards += 1
        except Exception as e:
            print(f"  Warning: {shard_file}: {e}")
    
    print(f"  Measured distribution from {n_shards} shards")
    return counts


def compute_entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * np.log2(p)
    return entropy


# ── Main compression ─────────────────────────────────────────────────────────

def compress_with_entropy_codebook(
    compressed_dir: Path,
    output_dir: Path,
    huffman_codebook: Dict[int, str],
    huffman_codebook_inv: Dict[str, int],
) -> Dict:
    """Apply Huffman coding to codebook entries in all shards."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = compressed_dir / "model.safetensors.index.json"
    with open(index_file) as f:
        index = json.load(f)
    
    manifest = json.load(open(compressed_dir / "compression_manifest.json"))
    
    total_original_bits = 0
    total_encoded_bits = 0
    n_tensors = 0
    
    # Process each shard
    shards_to_process = set(index["weight_map"].values())
    
    for shard_file in sorted(shards_to_process):
        shard_path = compressed_dir / shard_file
        output_shard_path = output_dir / shard_file
        
        if not shard_path.exists():
            continue
        
        new_data = {}
        
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                if key.endswith("weight_codebook_entries"):
                    # Encode with Huffman
                    t = tensor.numpy().flatten()
                    lo = t & 0x0F
                    hi = (t >> 4) & 0x0F
                    # Interleave: lo[0], hi[0], lo[1], hi[1], ...
                    entries = np.empty(len(lo) + len(hi), dtype=np.uint8)
                    entries[0::2] = lo
                    entries[1::2] = hi
                    
                    n_entries = len(entries)
                    original_bits = n_entries * 4  # 4 bits per entry
                    
                    encoded = huffman_encode(entries, huffman_codebook)
                    encoded_bits = (len(encoded) - 1) * 8  # subtract padding byte
                    
                    total_original_bits += original_bits
                    total_encoded_bits += encoded_bits
                    n_tensors += 1
                    
                    # Store as uint8 tensor with length prefix
                    encoded_arr = np.frombuffer(encoded, dtype=np.uint8)
                    # Prepend n_entries as 4-byte little-endian int
                    n_entries_bytes = np.frombuffer(struct.pack("<I", n_entries), dtype=np.uint8)
                    full_arr = np.concatenate([n_entries_bytes, encoded_arr])
                    new_data[key] = torch.from_numpy(full_arr)
                else:
                    new_data[key] = tensor
        
        save_file(new_data, str(output_shard_path))
    
    return {
        "n_tensors": n_tensors,
        "original_bits": total_original_bits,
        "encoded_bits": total_encoded_bits,
        "compression_ratio": total_original_bits / total_encoded_bits if total_encoded_bits > 0 else 0,
        "bits_saved_per_entry": (total_original_bits - total_encoded_bits) / (total_original_bits / 4) if total_original_bits > 0 else 0,
    }


def main():
    print("=" * 70)
    print("Phase 25A: Huffman Coding of Per-Block Codebook Entries")
    print("=" * 70)
    print()
    
    start = time.time()
    
    # Step 1: Measure distribution
    print("Step 1: Measuring codebook entry distribution...")
    counts = measure_codebook_entry_distribution(COMPRESSED_DIR, max_shards=30)
    
    total = sum(counts.values())
    entropy = compute_entropy(counts)
    
    E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    print(f"  Total entries: {total:,}")
    print(f"  Shannon entropy: {entropy:.4f} bits (vs 4.0 current)")
    print(f"  Savings per entry: {4.0 - entropy:.4f} bits")
    print(f"  Savings per element: {3 * (4.0 - entropy) / BLOCK_SIZE:.4f} bpe")
    print(f"  Expected new total: {2.75 - 3 * (4.0 - entropy) / BLOCK_SIZE:.4f} bpe")
    print()
    
    # Step 2: Build Huffman tree
    print("Step 2: Building Huffman tree...")
    freq_table = {k: v for k, v in counts.items() if v > 0}
    tree = build_huffman_tree(freq_table)
    huffman_codebook = build_codebook(tree)
    huffman_codebook_inv = {v: k for k, v in huffman_codebook.items()}
    
    # Compute average code length
    avg_code_len = sum(len(huffman_codebook[s]) * counts[s] for s in huffman_codebook) / total
    print(f"  Average Huffman code length: {avg_code_len:.4f} bits (entropy: {entropy:.4f})")
    print(f"  Huffman overhead: {avg_code_len - entropy:.4f} bits")
    print()
    
    # Step 3: Verify on a small sample
    print("Step 3: Verifying encode/decode roundtrip...")
    test_symbols = np.array([7, 15, 5, 13, 1, 7, 15, 5], dtype=np.uint8)
    encoded = huffman_encode(test_symbols, huffman_codebook)
    decoded = huffman_decode(encoded, huffman_codebook_inv, len(test_symbols))
    assert np.array_equal(test_symbols, decoded), f"Roundtrip failed: {test_symbols} != {decoded}"
    print(f"  Roundtrip OK: {test_symbols} -> {len(encoded)} bytes -> {decoded}")
    print()
    
    # Step 4: Compute expected bits/elem
    print("Step 4: Computing expected bits/elem...")
    # Current: 2 bits/index + 3 * 4 bits / 16 = 2.75 bpe
    # New: 2 bits/index + 3 * avg_code_len / 16 bpe
    new_codebook_bpe = 3 * avg_code_len / BLOCK_SIZE
    new_total_bpe = 2.0 + new_codebook_bpe
    print(f"  Current: 2.75 bpe")
    print(f"  New codebook overhead: {new_codebook_bpe:.4f} bpe (vs 0.75 current)")
    print(f"  New total: {new_total_bpe:.4f} bpe")
    print(f"  Improvement: {(2.75 - new_total_bpe) / 2.75 * 100:.1f}%")
    print(f"  Compression vs 4-bit: {(4 - new_total_bpe) / 4 * 100:.1f}%")
    print()
    
    # Save results
    results = {
        "phase": "25A",
        "method": "Huffman coding of per-block codebook entries",
        "distribution": {str(k): int(v) for k, v in counts.items()},
        "entropy_bits": entropy,
        "avg_huffman_bits": avg_code_len,
        "current_bpe": 2.75,
        "new_codebook_bpe": new_codebook_bpe,
        "new_total_bpe": new_total_bpe,
        "improvement_pct": (2.75 - new_total_bpe) / 2.75 * 100,
        "compression_vs_4bit_pct": (4 - new_total_bpe) / 4 * 100,
        "huffman_codebook": {str(k): v for k, v in huffman_codebook.items()},
        "elapsed_sec": time.time() - start,
    }
    
    with open("phase25_entropy_codebook_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to phase25_entropy_codebook_results.json")
    print(f"Elapsed: {time.time() - start:.1f}s")
    print()
    print("=" * 70)
    print(f"RESULT: {new_total_bpe:.4f} bpe ({(2.75 - new_total_bpe) / 2.75 * 100:.1f}% improvement)")
    print(f"        {(4 - new_total_bpe) / 4 * 100:.1f}% compression vs 4-bit")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
