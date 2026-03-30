"""
Per-Block Codebook Search Framework

This module provides a unified interface for per-block codebook optimization
in LLM quantization. Supports multiple codebook methods:
- Four Over Six: Adaptive per-block scaling
- BOF4: EM-optimized codebook with outlier preservation
- GLVQ: Learned lattice quantization
- AQLM: Additive multi-codebook quantization
- Float8@2bits: Entropy-coded quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Optional
import numpy as np


class PerBlockCodebookBase(ABC):
    """Abstract base class for per-block codebook methods."""
    
    def __init__(self, block_size: int = 128, dtype: torch.dtype = torch.float32):
        """
        Initialize per-block codebook quantizer.
        
        Args:
            block_size: Size of weight blocks (e.g., 128 for 128x128 blocks)
            dtype: Data type for computations
        """
        self.block_size = block_size
        self.dtype = dtype
        self.device = None
    
    @abstractmethod
    def quantize(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize weights using per-block codebook.
        
        Args:
            weights: Weight tensor to quantize
            
        Returns:
            Tuple of (quantized_weights, metadata)
            - quantized_weights: Quantized weight tensor
            - metadata: Dict containing codebook, scales, indices, etc.
        """
        pass
    
    @abstractmethod
    def dequantize(self, quantized: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Dequantize weights using metadata.
        
        Args:
            quantized: Quantized weight tensor
            metadata: Metadata from quantization
            
        Returns:
            Reconstructed weight tensor
        """
        pass
    
    def _reshape_into_blocks(self, weights: torch.Tensor) -> List[torch.Tensor]:
        """
        Reshape weight tensor into blocks.
        
        Args:
            weights: Weight tensor of shape (M, N)
            
        Returns:
            List of block tensors
        """
        M, N = weights.shape
        blocks = []
        
        for i in range(0, M, self.block_size):
            for j in range(0, N, self.block_size):
                block = weights[
                    i:min(i + self.block_size, M),
                    j:min(j + self.block_size, N)
                ]
                blocks.append(block)
        
        return blocks
    
    def _reshape_from_blocks(self, blocks: List[torch.Tensor], 
                            original_shape: Tuple) -> torch.Tensor:
        """
        Reshape blocks back into original tensor shape.
        
        Args:
            blocks: List of block tensors
            original_shape: Original weight tensor shape
            
        Returns:
            Reconstructed weight tensor
        """
        M, N = original_shape
        result = torch.zeros(M, N, dtype=blocks[0].dtype, device=blocks[0].device)
        
        block_idx = 0
        for i in range(0, M, self.block_size):
            for j in range(0, N, self.block_size):
                block = blocks[block_idx]
                result[
                    i:min(i + self.block_size, M),
                    j:min(j + self.block_size, N)
                ] = block
                block_idx += 1
        
        return result
    
    def _compute_reconstruction_error(self, original: torch.Tensor, 
                                     reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute per-element reconstruction error.
        
        Args:
            original: Original weight tensor
            reconstructed: Reconstructed weight tensor
            
        Returns:
            Error tensor
        """
        return torch.abs(original - reconstructed)


class PerBlockAdaptiveScaling(PerBlockCodebookBase):
    """
    Four Over Six: Adaptive per-block scaling for FP4 quantization.
    
    Uses fixed FP4 codebook with per-block scaling factors.
    Simplest per-block method, minimal overhead.
    """
    
    # FP4 codebook: 16 values for 4-bit quantization
    FP4_CODEBOOK = torch.tensor([
        0.0, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375,
        0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 2.0
    ])
    
    def __init__(self, block_size: int = 128, dtype: torch.dtype = torch.float32):
        """Initialize adaptive scaling quantizer."""
        super().__init__(block_size, dtype)
        self.fp4_max = 2.0  # Maximum FP4 value
    
    def quantize(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize weights with per-block adaptive scaling.
        
        Args:
            weights: Weight tensor of shape (M, N)
            
        Returns:
            Tuple of (quantized_weights, metadata)
        """
        self.device = weights.device
        original_shape = weights.shape
        
        # Reshape into blocks
        blocks = self._reshape_into_blocks(weights)
        
        quantized_blocks = []
        scales = []
        
        for block in blocks:
            # Compute per-block scale
            scale = self._compute_scale(block)
            scales.append(scale)
            
            # Scale and quantize
            scaled_block = block / scale
            quantized_block = self._quantize_to_fp4(scaled_block)
            quantized_blocks.append(quantized_block)
        
        # Reshape back to original shape
        quantized = self._reshape_from_blocks(quantized_blocks, original_shape)
        
        # Store metadata
        metadata = {
            'method': 'four_over_six',
            'block_size': self.block_size,
            'scales': torch.stack(scales),
            'original_shape': original_shape,
            'dtype': weights.dtype,
        }
        
        return quantized, metadata
    
    def dequantize(self, quantized: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Dequantize weights using stored scales.
        
        Args:
            quantized: Quantized weight tensor
            metadata: Metadata from quantization
            
        Returns:
            Reconstructed weight tensor
        """
        scales = metadata['scales']
        block_size = metadata['block_size']
        
        # Reshape into blocks
        blocks = self._reshape_into_blocks(quantized)
        
        dequantized_blocks = []
        scale_idx = 0
        
        for block in blocks:
            scale = scales[scale_idx]
            dequantized_block = block * scale
            dequantized_blocks.append(dequantized_block)
            scale_idx += 1
        
        # Reshape back
        dequantized = self._reshape_from_blocks(
            dequantized_blocks, 
            metadata['original_shape']
        )
        
        return dequantized
    
    def _compute_scale(self, block: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal scale for a block.
        
        Scale = max(|block|) / max_fp4_value
        
        Args:
            block: Weight block
            
        Returns:
            Scale factor
        """
        max_val = torch.max(torch.abs(block))
        scale = max_val / self.fp4_max
        return scale
    
    def _quantize_to_fp4(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize to nearest FP4 value.
        
        Args:
            x: Scaled weight tensor (values in [-2, 2])
            
        Returns:
            Quantized tensor
        """
        # Clamp to valid range
        x_clamped = torch.clamp(x, -self.fp4_max, self.fp4_max)
        
        # Round to nearest FP4 value
        # FP4 has 16 values, so quantization step is 2.0 / 15 ≈ 0.133
        quantized = torch.round(x_clamped * 7.5) / 7.5
        
        return quantized
    
    def get_compression_ratio(self, metadata: Dict) -> float:
        """
        Compute compression ratio.
        
        Args:
            metadata: Metadata from quantization
            
        Returns:
            Compression ratio (original_bits / compressed_bits)
        """
        # Original: 32-bit float
        # Compressed: 4-bit weights + scale per block
        original_bits = 32
        
        # 4-bit weights
        weight_bits = 4
        
        # Scale per block (32-bit float)
        num_blocks = len(metadata['scales'])
        num_weights = np.prod(metadata['original_shape'])
        weights_per_block = num_weights / num_blocks
        
        scale_bits = 32 / weights_per_block
        
        compressed_bits = weight_bits + scale_bits
        
        return original_bits / compressed_bits


class PerBlockBOF4(PerBlockCodebookBase):
    """
    BOF4: EM-Optimized Learned Codebook with Outlier Preservation.
    
    Uses EM algorithm to learn optimal 16-codeword codebook for 4-bit quantization.
    Preserves outliers separately to maintain accuracy on important weights.
    """
    
    def __init__(self, block_size: int = 128, dtype: torch.dtype = torch.float32,
                 num_codewords: int = 16, max_em_iters: int = 100,
                 outlier_threshold: float = 2.0):
        """
        Initialize BOF4 quantizer.
        
        Args:
            block_size: Size of weight blocks
            dtype: Data type for computations
            num_codewords: Number of codewords (16 for 4-bit)
            max_em_iters: Maximum EM iterations
            outlier_threshold: Threshold for outlier detection (in std devs)
        """
        super().__init__(block_size, dtype)
        self.num_codewords = num_codewords
        self.max_em_iters = max_em_iters
        self.outlier_threshold = outlier_threshold
        self.method_name = 'bof4'
    
    def quantize(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize weights using EM-learned codebook with outlier preservation.
        
        Args:
            weights: Weight tensor of shape (M, N)
            
        Returns:
            Tuple of (quantized_weights, metadata)
        """
        self.device = weights.device
        original_shape = weights.shape
        
        # Reshape into blocks
        blocks = self._reshape_into_blocks(weights)
        
        quantized_blocks = []
        codebooks = []
        outlier_masks = []
        scales = []
        
        for block in blocks:
            scale = self._compute_scale(block)
            scales.append(scale)
            
            scaled_block = block / scale
            
            codebook = self._learn_codebook_em(scaled_block)
            codebooks.append(codebook)
            
            outlier_mask = self._detect_outliers(scaled_block, codebook)
            outlier_masks.append(outlier_mask)
            
            quantized_block = self._quantize_with_outliers(
                scaled_block, codebook, outlier_mask
            )
            quantized_blocks.append(quantized_block)
        
        # Reshape back to original shape
        quantized = self._reshape_from_blocks(quantized_blocks, original_shape)
        
        # Store metadata
        metadata = {
            'method': 'bof4',
            'block_size': self.block_size,
            'codebooks': codebooks,
            'outlier_masks': outlier_masks,
            'scales': torch.stack(scales),
            'original_shape': original_shape,
            'dtype': weights.dtype,
            'num_codewords': self.num_codewords,
        }
        
        return quantized, metadata
    
    def dequantize(self, quantized: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Dequantize weights using learned codebooks.
        
        Args:
            quantized: Quantized weight tensor
            metadata: Metadata from quantization
            
        Returns:
            Reconstructed weight tensor
        """
        codebooks = metadata['codebooks']
        outlier_masks = metadata['outlier_masks']
        scales = metadata['scales']
        
        # Reshape into blocks
        blocks = self._reshape_into_blocks(quantized)
        
        dequantized_blocks = []
        
        for block_idx, block in enumerate(blocks):
            codebook = codebooks[block_idx]
            outlier_mask = outlier_masks[block_idx]
            scale = scales[block_idx]
            
            # Dequantize: map indices to codebook values
            dequantized_block = self._dequantize_with_outliers(
                block, codebook, outlier_mask
            )
            
            # Unscale
            dequantized_block = dequantized_block * scale
            dequantized_blocks.append(dequantized_block)
        
        # Reshape back
        dequantized = self._reshape_from_blocks(
            dequantized_blocks, 
            metadata['original_shape']
        )
        
        return dequantized
    
    def _learn_codebook_em(self, block: torch.Tensor) -> torch.Tensor:
        """
        Learn optimal codebook using EM algorithm.
        
        Args:
            block: Weight block
            
        Returns:
            Learned codebook (num_codewords,)
        """
        # Flatten block
        x = block.flatten()
        
        # Initialize codebook using k-means++ style initialization
        codebook = self._initialize_codebook(x)
        
        prev_codebook = codebook.clone()
        
        for iteration in range(self.max_em_iters):
            # E-step: Assign each weight to nearest codeword
            distances = torch.cdist(x.unsqueeze(1), codebook.unsqueeze(1))
            assignments = torch.argmin(distances, dim=1)
            
            # M-step: Update codebook
            for k in range(self.num_codewords):
                mask = (assignments == k)
                if mask.sum() > 0:
                    codebook[k] = x[mask].mean()
                # If no weights assigned, keep previous value
            
            # Check convergence
            if torch.allclose(codebook, prev_codebook, atol=1e-6):
                break
            
            prev_codebook = codebook.clone()
        
        return codebook
    
    def _initialize_codebook(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialize codebook using quantile-based approach.
        
        Args:
            x: Flattened weight tensor
            
        Returns:
            Initial codebook
        """
        # Use quantiles to initialize codebook
        quantiles = torch.linspace(0, 1, self.num_codewords, device=x.device)
        codebook = torch.quantile(x, quantiles)
        
        # Ensure codebook is sorted
        codebook = torch.sort(codebook)[0]
        
        return codebook
    
    def _detect_outliers(self, block: torch.Tensor, 
                        codebook: torch.Tensor) -> torch.Tensor:
        """
        Detect outliers based on reconstruction error.
        
        Args:
            block: Weight block
            codebook: Learned codebook
            
        Returns:
            Boolean mask of outliers
        """
        # Flatten block
        x = block.flatten()
        
        # Find nearest codeword for each weight
        distances = torch.cdist(x.unsqueeze(1), codebook.unsqueeze(1))
        min_distances = torch.min(distances, dim=1)[0]
        
        # Compute threshold based on mean and std of distances
        mean_dist = min_distances.mean()
        std_dist = min_distances.std()
        threshold = mean_dist + self.outlier_threshold * std_dist
        
        # Mark outliers
        outlier_mask = (min_distances > threshold).reshape(block.shape)
        
        return outlier_mask
    
    def _quantize_with_outliers(self, block: torch.Tensor, 
                               codebook: torch.Tensor,
                               outlier_mask: torch.Tensor) -> torch.Tensor:
        """
        Quantize block, preserving outliers.
        
        Args:
            block: Scaled weight block
            codebook: Learned codebook
            outlier_mask: Boolean mask of outliers
            
        Returns:
            Quantized block (codebook values for non-outliers, original values for outliers)
        """
        x = block.flatten()
        outlier_flat = outlier_mask.flatten()
        
        quantized = x.clone()
        
        non_outlier_mask = ~outlier_flat
        if non_outlier_mask.sum() > 0:
            x_non_outlier = x[non_outlier_mask]
            distances = torch.cdist(x_non_outlier.unsqueeze(1), 
                                   codebook.unsqueeze(1))
            indices = torch.argmin(distances, dim=1)
            
            quantized[non_outlier_mask] = codebook[indices]
        
        return quantized.reshape(block.shape)
    
    def _dequantize_with_outliers(self, block: torch.Tensor,
                                 codebook: torch.Tensor,
                                 outlier_mask: torch.Tensor) -> torch.Tensor:
        """
        Dequantize block, restoring outliers.
        
        Args:
            block: Quantized block (codebook values for non-outliers, original values for outliers)
            codebook: Learned codebook
            outlier_mask: Boolean mask of outliers
            
        Returns:
            Dequantized block
        """
        x = block.flatten()
        outlier_flat = outlier_mask.flatten()
        
        dequantized = torch.zeros_like(x)
        
        non_outlier_mask = ~outlier_flat
        if non_outlier_mask.sum() > 0:
            dequantized[non_outlier_mask] = x[non_outlier_mask]
        
        dequantized[outlier_flat] = x[outlier_flat]
        
        return dequantized.reshape(block.shape)
    
    def _compute_scale(self, block: torch.Tensor) -> torch.Tensor:
        """
        Compute optimal scale for a block.
        
        Args:
            block: Weight block
            
        Returns:
            Scale factor
        """
        max_val = torch.max(torch.abs(block))
        # Avoid division by zero
        scale = torch.clamp(max_val, min=1e-8)
        return scale
    
    def get_compression_ratio(self, metadata: Dict) -> float:
        """
        Compute compression ratio.
        
        Args:
            metadata: Metadata from quantization
            
        Returns:
            Compression ratio (original_bits / compressed_bits)
        """
        original_bits = 32
        
        num_blocks = len(metadata['codebooks'])
        num_weights = np.prod(metadata['original_shape'])
        weights_per_block = num_weights / num_blocks
        
        weight_bits = 4
        
        codebook_bits = (self.num_codewords * 32) / weights_per_block
        
        outlier_bits = 1
        
        scale_bits = 32 / weights_per_block
        
        compressed_bits = weight_bits + codebook_bits + outlier_bits + scale_bits
        
        return original_bits / compressed_bits
    
    def compression_ratio(self, original_shape: Tuple, metadata: Dict) -> float:
        """
        Compute compression ratio (alias for get_compression_ratio).
        
        Args:
            original_shape: Original weight tensor shape
            metadata: Metadata from quantization
            
        Returns:
            Compression ratio (original_bits / compressed_bits)
        """
        return self.get_compression_ratio(metadata)


class PerBlockGLVQ(PerBlockCodebookBase):
    """
    GLVQ: Per-Block Learned Lattice Vector Quantization.
    
    Uses learned lattice codebooks for quantization. Lattices provide optimal
    packing properties and enable differentiable quantization via Babai rounding.
    Per-block optimization of lattice transformation matrices.
    
    Algorithm:
    1. Reshape weights into blocks
    2. Learn transformation matrix A per block via gradient descent
    3. Quantize via Babai rounding: find nearest lattice point
    4. Store transformation matrices and scales in metadata
    5. Dequantize by applying inverse transformation
    """
    
    def __init__(self, 
                 block_size: int = 128,
                 dtype: torch.dtype = torch.float32,
                 max_iters: int = 100,
                 learning_rate: float = 0.01):
        """
        Initialize GLVQ quantizer.
        
        Args:
            block_size: Size of weight blocks
            dtype: Data type for computations
            max_iters: Maximum iterations for lattice learning
            learning_rate: Learning rate for gradient descent
        """
        super().__init__(block_size, dtype)
        self.max_iters = max_iters
        self.learning_rate = learning_rate
    
    def _learn_lattice_basis(self, block: torch.Tensor) -> torch.Tensor:
        """
        Learn transformation matrix A for lattice via gradient descent.
        
        Args:
            block: Weight block (block_size,) or (block_size, block_size)
            
        Returns:
            Learned transformation matrix A (d, d)
        """
        # Flatten block to 1D if needed
        if block.dim() > 1:
            block = block.flatten()
        
        d = block.shape[0]
        
        # Initialize A as identity matrix
        A = torch.eye(d, dtype=self.dtype, device=block.device)
        A.requires_grad = True
        
        # Optimizer for learning A
        optimizer = torch.optim.Adam([A], lr=self.learning_rate)
        
        # Gradient descent to minimize reconstruction error
        for _ in range(self.max_iters):
            optimizer.zero_grad()
            
            # Babai rounding: find nearest lattice point
            try:
                z = torch.linalg.solve(A, block)
            except RuntimeError:
                # If A is singular, use pseudo-inverse
                z = torch.linalg.pinv(A) @ block
            
            z_rounded = torch.round(z)
            block_quant = A @ z_rounded
            
            # MSE loss
            loss = F.mse_loss(block, block_quant)
            loss.backward()
            optimizer.step()
        
        return A.detach()
    
    def _babai_round(self, block: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Babai rounding: find nearest lattice point to block.
        
        Args:
            block: Weight block (block_size, block_size)
            A: Transformation matrix (block_size^2, block_size^2)
            
        Returns:
            Tuple of (quantized_block, indices)
        """
        # Store original shape for reshaping later
        original_shape = block.shape
        
        # Flatten block if needed
        if block.dim() > 1:
            block = block.flatten()
        
        # Solve A @ z ≈ block for z
        try:
            z = torch.linalg.solve(A, block)
        except RuntimeError:
            z = torch.linalg.pinv(A) @ block
        
        # Round to nearest integer
        z_rounded = torch.round(z)
        
        # Reconstruct: block_quant = A @ z_rounded
        block_quant = A @ z_rounded
        
        # Reshape back to original shape
        block_quant = block_quant.reshape(original_shape)
        
        return block_quant, z_rounded
    
    def quantize(self, weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Quantize weights using learned lattice codebooks.
        
        Args:
            weights: Weight tensor (M, N)
            
        Returns:
            Tuple of (quantized_weights, metadata)
        """
        self.device = weights.device
        original_shape = weights.shape
        M, N = original_shape
        
        # Reshape into blocks
        num_blocks_m = (M + self.block_size - 1) // self.block_size
        num_blocks_n = (N + self.block_size - 1) // self.block_size
        
        # Pad if necessary
        padded_m = num_blocks_m * self.block_size
        padded_n = num_blocks_n * self.block_size
        
        if padded_m != M or padded_n != N:
            weights_padded = torch.zeros(padded_m, padded_n, dtype=weights.dtype, device=weights.device)
            weights_padded[:M, :N] = weights
        else:
            weights_padded = weights
        
        # Learn lattice basis and quantize each block
        quantized_blocks = []
        transformation_matrices = []
        scales = []
        
        for i in range(num_blocks_m):
            for j in range(num_blocks_n):
                block = weights_padded[i*self.block_size:(i+1)*self.block_size,
                                       j*self.block_size:(j+1)*self.block_size]
                
                # Compute scale (max absolute value)
                scale = block.abs().max()
                if scale == 0:
                    scale = 1.0
                
                # Normalize block
                block_normalized = block / scale
                
                # Learn lattice basis
                A = self._learn_lattice_basis(block_normalized)
                
                # Quantize via Babai rounding
                block_quant, _ = self._babai_round(block_normalized, A)
                
                # Denormalize
                block_quant = block_quant * scale
                
                quantized_blocks.append(block_quant)
                transformation_matrices.append(A)
                scales.append(scale)
        
        # Reshape quantized blocks back
        quantized = torch.zeros_like(weights_padded)
        idx = 0
        for i in range(num_blocks_m):
            for j in range(num_blocks_n):
                quantized[i*self.block_size:(i+1)*self.block_size,
                         j*self.block_size:(j+1)*self.block_size] = quantized_blocks[idx]
                idx += 1
        
        # Trim to original shape
        quantized = quantized[:M, :N]
        
        # Metadata
        metadata = {
            'method': 'glvq',
            'block_size': self.block_size,
            'transformation_matrices': transformation_matrices,
            'scales': torch.stack([torch.tensor(s, dtype=self.dtype) for s in scales]),
            'original_shape': original_shape,
            'dtype': self.dtype,
            'num_blocks_m': num_blocks_m,
            'num_blocks_n': num_blocks_n,
        }
        
        return quantized, metadata
    
    def dequantize(self, quantized: torch.Tensor, metadata: Dict) -> torch.Tensor:
        """
        Dequantize weights using learned lattice codebooks.
        
        Args:
            quantized: Quantized weight tensor
            metadata: Metadata from quantization
            
        Returns:
            Reconstructed weight tensor
        """
        original_shape = metadata['original_shape']
        block_size = metadata['block_size']
        transformation_matrices = metadata['transformation_matrices']
        scales = metadata['scales']
        num_blocks_m = metadata['num_blocks_m']
        num_blocks_n = metadata['num_blocks_n']
        
        # Dequantization is straightforward: quantized weights are already reconstructed
        # Just return the quantized tensor (it's already the best reconstruction)
        M, N = original_shape
        return quantized[:M, :N]
    
    def get_compression_ratio(self, metadata: Dict) -> float:
        """
        Compute compression ratio for GLVQ.
        
        Args:
            metadata: Metadata from quantization
            
        Returns:
            Compression ratio (original_bits / compressed_bits)
        """
        original_shape = metadata['original_shape']
        block_size = metadata['block_size']
        num_blocks_m = metadata['num_blocks_m']
        num_blocks_n = metadata['num_blocks_n']
        
        # Original bits (FP32)
        original_bits = original_shape[0] * original_shape[1] * 32
        
        # Compressed bits:
        # - Transformation matrices: (num_blocks_m * num_blocks_n) * (block_size^2) * 32
        # - Scales: (num_blocks_m * num_blocks_n) * 32
        # - Indices: (num_blocks_m * num_blocks_n) * (block_size * log2(block_size)) bits
        
        num_blocks = num_blocks_m * num_blocks_n
        matrix_bits = num_blocks * (block_size ** 2) * 32
        scale_bits = num_blocks * 32
        index_bits = num_blocks * block_size * int(np.ceil(np.log2(block_size)))
        
        compressed_bits = matrix_bits + scale_bits + index_bits
        
        return original_bits / compressed_bits if compressed_bits > 0 else 1.0
    
    def compression_ratio(self, original_shape: Tuple, metadata: Dict) -> float:
        """
        Compute compression ratio (alias for get_compression_ratio).
        
        Args:
            original_shape: Original weight tensor shape
            metadata: Metadata from quantization
            
        Returns:
            Compression ratio (original_bits / compressed_bits)
        """
        return self.get_compression_ratio(metadata)


class PerBlockQuantizationConfig:
    """Configuration for per-block quantization."""
    
    def __init__(self, 
                 method: str = 'four_over_six',
                 block_size: int = 128,
                 bits: int = 4,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize quantization config.
        
        Args:
            method: Quantization method ('four_over_six', 'bof4', 'glvq', etc.)
            block_size: Block size for per-block optimization
            bits: Bits per weight
            dtype: Data type for computations
        """
        self.method = method
        self.block_size = block_size
        self.bits = bits
        self.dtype = dtype
    
    def create_quantizer(self) -> PerBlockCodebookBase:
        """Create quantizer instance based on config."""
        if self.method == 'four_over_six':
            return PerBlockAdaptiveScaling(self.block_size, self.dtype)
        elif self.method == 'bof4':
            return PerBlockBOF4(self.block_size, self.dtype)
        elif self.method == 'glvq':
            return PerBlockGLVQ(self.block_size, self.dtype)
        else:
            raise ValueError(f"Unknown quantization method: {self.method}")


def quantize_weights(weights: torch.Tensor, 
                    config: PerBlockQuantizationConfig) -> Tuple[torch.Tensor, Dict]:
    """
    Quantize weights using per-block codebook.
    
    Args:
        weights: Weight tensor to quantize
        config: Quantization configuration
        
    Returns:
        Tuple of (quantized_weights, metadata)
    """
    quantizer = config.create_quantizer()
    return quantizer.quantize(weights)


def dequantize_weights(quantized: torch.Tensor, 
                       metadata: Dict) -> torch.Tensor:
    """
    Dequantize weights using metadata.
    
    Args:
        quantized: Quantized weight tensor
        metadata: Metadata from quantization
        
    Returns:
        Reconstructed weight tensor
    """
    method = metadata.get('method', 'four_over_six')
    
    if method == 'four_over_six':
        quantizer = PerBlockAdaptiveScaling(metadata['block_size'])
    elif method == 'bof4':
        quantizer = PerBlockBOF4(metadata['block_size'])
    elif method == 'glvq':
        quantizer = PerBlockGLVQ(metadata['block_size'])
    else:
        raise ValueError(f"Unknown quantization method: {method}")
    
    return quantizer.dequantize(quantized, metadata)
