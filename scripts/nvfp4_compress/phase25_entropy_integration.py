"""
Phase 25: Entropy Codebook Integration

Integrates Huffman coding of codebook entries into the compression pipeline.
Expected improvement: 5.4% (2.75 bpe → 2.6004 bpe)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
import heapq
from collections import defaultdict


class HuffmanNode:
    """Node in Huffman tree"""
    def __init__(self, freq: int, value: Optional[int] = None, left=None, right=None):
        self.freq = freq
        self.value = value
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCodebook:
    """Huffman coding for codebook entries"""
    
    def __init__(self):
        self.codes = {}  # value -> (code, length)
        self.tree = None
        self.root = None
    
    def build_from_distribution(self, distribution: Dict[int, int]):
        """Build Huffman tree from frequency distribution"""
        if not distribution:
            return
        
        # Create leaf nodes
        heap = [HuffmanNode(freq, value) for value, freq in distribution.items()]
        heapq.heapify(heap)
        
        # Build tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, parent)
        
        self.root = heap[0] if heap else None
        self._generate_codes(self.root, "")
    
    def _generate_codes(self, node: Optional[HuffmanNode], code: str):
        """Generate Huffman codes from tree"""
        if node is None:
            return
        
        if node.value is not None:  # Leaf node
            self.codes[node.value] = (code if code else "0", len(code) if code else 1)
        else:
            self._generate_codes(node.left, code + "0")
            self._generate_codes(node.right, code + "1")
    
    def encode(self, values: np.ndarray) -> Tuple[bytes, Dict]:
        """Encode values using Huffman codes"""
        if not self.codes:
            return b"", {}
        
        # Build bit string
        bit_string = ""
        for val in values.flat:
            code, _ = self.codes.get(int(val), ("0", 1))
            bit_string += code
        
        # Pad to byte boundary
        padding = (8 - len(bit_string) % 8) % 8
        bit_string += "0" * padding
        
        # Convert to bytes
        encoded = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))
        
        return encoded, {
            "padding": padding,
            "original_length": len(values),
            "encoded_length": len(encoded)
        }
    
    def decode(self, encoded: bytes, metadata: Dict, num_values: int) -> np.ndarray:
        """Decode Huffman-encoded values"""
        if not self.root:
            return np.zeros(num_values, dtype=np.int32)
        
        # Convert bytes to bit string
        bit_string = "".join(format(byte, "08b") for byte in encoded)
        
        # Remove padding
        padding = metadata.get("padding", 0)
        if padding:
            bit_string = bit_string[:-padding]
        
        # Decode
        values = []
        node = self.root
        for bit in bit_string:
            if bit == "0":
                node = node.left
            else:
                node = node.right
            
            if node.value is not None:  # Leaf node
                values.append(node.value)
                node = self.root
                if len(values) >= num_values:
                    break
        
        return np.array(values[:num_values], dtype=np.int32)
    
    def get_average_code_length(self) -> float:
        """Get average code length (bits per symbol)"""
        if not self.codes:
            return 0.0
        
        total_length = sum(length for _, length in self.codes.values())
        return total_length / len(self.codes)


def measure_codebook_distribution(codebook_entries: torch.Tensor) -> Dict[int, int]:
    """Measure distribution of codebook entry values"""
    distribution = defaultdict(int)
    
    for val in codebook_entries.flatten():
        distribution[int(val.item())] += 1
    
    return dict(distribution)


def apply_entropy_codebook(
    codebook_entries: torch.Tensor,
    distribution: Optional[Dict[int, int]] = None
) -> Tuple[torch.Tensor, Dict, HuffmanCodebook]:
    """
    Apply Huffman coding to codebook entries
    
    Args:
        codebook_entries: Original codebook entries (shape: [num_codebooks, codebook_size])
        distribution: Pre-computed distribution (optional)
    
    Returns:
        encoded_entries: Huffman-encoded entries
        metadata: Encoding metadata
        huffman: HuffmanCodebook object for decoding
    """
    
    # Measure distribution if not provided
    if distribution is None:
        distribution = measure_codebook_distribution(codebook_entries)
    
    # Build Huffman codebook
    huffman = HuffmanCodebook()
    huffman.build_from_distribution(distribution)
    
    # Encode
    encoded, metadata = huffman.encode(codebook_entries.cpu().numpy())
    
    # Store metadata
    metadata.update({
        "huffman_codes": {str(k): v for k, v in huffman.codes.items()},
        "original_shape": list(codebook_entries.shape),
        "average_code_length": huffman.get_average_code_length()
    })
    
    return torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8)), metadata, huffman


def decode_entropy_codebook(
    encoded_entries: torch.Tensor,
    metadata: Dict,
    huffman: HuffmanCodebook
) -> torch.Tensor:
    """
    Decode Huffman-encoded codebook entries
    
    Args:
        encoded_entries: Huffman-encoded entries
        metadata: Encoding metadata
        huffman: HuffmanCodebook object
    
    Returns:
        decoded_entries: Original codebook entries
    """
    
    num_values = np.prod(metadata["original_shape"])
    decoded = huffman.decode(encoded_entries.cpu().numpy().tobytes(), metadata, num_values)
    
    return torch.from_numpy(decoded.reshape(metadata["original_shape"])).float()


def estimate_entropy_improvement(
    codebook_entries: torch.Tensor,
    current_bits_per_entry: float = 4.0
) -> Dict:
    """
    Estimate improvement from entropy coding
    
    Args:
        codebook_entries: Codebook entries
        current_bits_per_entry: Current bits per entry (default 4.0 for 4-bit)
    
    Returns:
        Improvement statistics
    """
    
    distribution = measure_codebook_distribution(codebook_entries)
    
    # Calculate Shannon entropy
    total = sum(distribution.values())
    entropy = 0.0
    for count in distribution.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    # Build Huffman and get average code length
    huffman = HuffmanCodebook()
    huffman.build_from_distribution(distribution)
    avg_code_length = huffman.get_average_code_length()
    
    # Calculate improvement
    improvement_percent = (1 - avg_code_length / current_bits_per_entry) * 100
    
    return {
        "shannon_entropy": entropy,
        "huffman_avg_length": avg_code_length,
        "current_bits": current_bits_per_entry,
        "improvement_percent": improvement_percent,
        "compression_ratio": current_bits_per_entry / avg_code_length,
        "num_unique_values": len(distribution)
    }


if __name__ == "__main__":
    # Test
    print("Phase 25: Entropy Codebook Integration")
    print("=" * 80)
    
    # Create test data
    codebook = torch.randint(0, 256, (8, 256), dtype=torch.int32)
    
    # Estimate improvement
    stats = estimate_entropy_improvement(codebook)
    print(f"Shannon entropy: {stats['shannon_entropy']:.4f} bits")
    print(f"Huffman avg length: {stats['huffman_avg_length']:.4f} bits")
    print(f"Improvement: {stats['improvement_percent']:.2f}%")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Test encode/decode
    encoded, metadata, huffman = apply_entropy_codebook(codebook)
    decoded = decode_entropy_codebook(encoded, metadata, huffman)
    
    # Verify
    match = torch.allclose(codebook.float(), decoded)
    print(f"\nEncode/decode verification: {'PASS' if match else 'FAIL'}")
    print(f"Original size: {codebook.numel() * 4} bytes")
    print(f"Encoded size: {encoded.numel()} bytes")
    print(f"Compression: {codebook.numel() * 4 / encoded.numel():.2f}x")

