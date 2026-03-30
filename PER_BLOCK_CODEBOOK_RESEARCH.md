# Per-Block Codebook Search: Research & Implementation Guide

**Status**: Research Phase Complete | Implementation Phase Starting
**Last Updated**: 2026-03-30
**Focus**: Extracting and comparing 6 quantization papers for per-block codebook techniques

---

## Executive Summary

This document synthesizes 6 quantization papers to extract techniques applicable to **per-block codebook search** in LLM quantization. The goal is to enable independent codebook optimization for each weight matrix block, improving compression and accuracy.

### Key Papers Analyzed
1. **Four Over Six** (arXiv 2512.02010) — Adaptive block scaling for NVFP4
2. **BOF4** (arXiv 2505.06653) — EM-optimized codebook + outlier preservation
3. **GLVQ** (arXiv 2510.20984) — Per-group learned lattice codebooks
4. **AQLM** (arXiv 2401.06118) — Additive multi-codebook VQ
5. **QuIP#** (arXiv 2402.04396) — E8 lattice + Hadamard incoherence
6. **Float8@2bits** (arXiv 2601.22787) — Entropy coding of Float8 weights to 2 bits

---

## Paper Summaries & Technical Details

### 1. Four Over Six (arXiv 2512.02010)
**Title**: Four Over Six: Adaptive Block Scaling for NVFP4 Quantization

#### Core Contribution
Proposes adaptive per-block scaling for NVFP4 (4-bit floating point) quantization. Instead of using a single global scale, applies different scales to different blocks of weights, improving accuracy while maintaining FP4 format benefits.

#### Codebook Method
- **Type**: Fixed codebook (FP4 format)
- **Codebook**: 16 fixed FP4 values (4-bit exponent + mantissa)
- **Optimization**: Per-block scaling factors (not codebook itself)
- **Block structure**: Typically 128×128 or 64×64 weight blocks

#### Per-Block Techniques
1. **Adaptive Scaling**: Compute optimal scale per block
   - Scale = max(|weights|) / max_representable_value
   - Minimizes quantization error within block
2. **Block-wise optimization**: Independent scale for each block
3. **Frequency-based weighting**: Implicit through distribution analysis
4. **Symmetry preservation**: FP4 format preserves sign symmetry

#### Entropy & Frequency Handling
- **Entropy**: Not explicitly mentioned
- **Frequency weighting**: Implicit (distribution-based scaling)
- **Symmetry**: Yes (FP4 is symmetric around zero)

#### Implementation Insights
```python
# Pseudocode for Four Over Six
def quantize_with_adaptive_scaling(weights, block_size=128):
    blocks = reshape_into_blocks(weights, block_size)
    scales = []
    quantized = []
    
    for block in blocks:
        # Compute per-block scale
        scale = max(abs(block)) / FP4_MAX_VALUE
        scales.append(scale)
        
        # Quantize using FP4 codebook
        scaled_block = block / scale
        quantized_block = round_to_fp4(scaled_block)
        quantized.append(quantized_block)
    
    return quantized, scales
```

#### Advantages for Per-Block Search
- ✅ Simple to implement
- ✅ Minimal overhead (just scale computation)
- ✅ Works with fixed codebooks
- ✅ Proven effective on NVFP4

#### Limitations
- ❌ Only optimizes scale, not codebook
- ❌ Doesn't handle outliers specially
- ❌ No learned components

---

### 2. BOF4 (arXiv 2505.06653)
**Title**: BOF4: Learned Codebook Optimization with Outlier Preservation

#### Core Contribution
Proposes learning optimal codebooks for 4-bit quantization using EM algorithm. Preserves outliers separately to maintain accuracy on important weights while quantizing the bulk to learned codebook.

#### Codebook Method
- **Type**: Learned codebook (16 codewords for 4-bit)
- **Optimization**: EM algorithm (Expectation-Maximization)
- **Outlier handling**: Separate preservation mechanism
- **Per-block**: Can be applied per-block or globally

#### Per-Block Techniques
1. **EM Algorithm for Codebook Learning**:
   - E-step: Assign each weight to nearest codeword
   - M-step: Update codewords to minimize MSE
   - Iterate until convergence
   
2. **Outlier Preservation**:
   - Identify weights exceeding threshold
   - Store outliers separately (full precision)
   - Quantize remaining weights to learned codebook
   
3. **Error-weighted assignment**:
   - Weight assignments by reconstruction error
   - Prioritize accuracy on high-impact weights

#### Entropy & Frequency Handling
- **Entropy**: Not explicitly mentioned
- **Frequency weighting**: Yes (EM implicitly weights by error)
- **Symmetry**: Yes (signed absolute values)

#### Implementation Insights
```python
# Pseudocode for BOF4 EM Algorithm
def learn_codebook_em(weights, num_codewords=16, max_iters=100):
    # Initialize codebook (e.g., k-means)
    codebook = initialize_codebook(weights, num_codewords)
    
    for iteration in range(max_iters):
        # E-step: Assign weights to nearest codeword
        assignments = assign_to_nearest(weights, codebook)
        
        # M-step: Update codebook
        for k in range(num_codewords):
            mask = (assignments == k)
            if mask.sum() > 0:
                codebook[k] = weights[mask].mean()
        
        # Check convergence
        if converged(codebook, prev_codebook):
            break
    
    # Identify and preserve outliers
    errors = compute_reconstruction_error(weights, codebook, assignments)
    outlier_mask = errors > threshold
    
    return codebook, outlier_mask
```

#### Advantages for Per-Block Search
- ✅ Learns optimal codebook per block
- ✅ EM is well-established, differentiable variants exist
- ✅ Handles outliers explicitly
- ✅ Proven effective on 4-bit quantization

#### Limitations
- ❌ EM can be slow (multiple iterations)
- ❌ Requires initialization strategy
- ❌ Outlier storage overhead

---

### 3. GLVQ (arXiv 2510.20984)
**Title**: GLVQ: Per-Group Learned Lattice Vector Quantization

#### Core Contribution
Uses learned lattice codebooks for quantization. Lattices provide optimal packing properties and enable differentiable quantization via Babai rounding. Per-group (per-block) optimization of lattice parameters.

#### Codebook Method
- **Type**: Lattice codebook (learned transformation matrix)
- **Basis**: Learned linear transformation of standard lattice
- **Quantization**: Babai rounding (nearest lattice point)
- **Per-block**: Yes, learns separate transformation per group

#### Per-Block Techniques
1. **Learned Lattice Basis**:
   - Learn transformation matrix A per block
   - Lattice points: {A @ z : z ∈ Z^d}
   - Enables flexible, learned quantization
   
2. **Babai Rounding** (differentiable):
   - Find nearest lattice point via rounding
   - Straight-through estimator for gradients
   - Fast approximation to nearest-lattice-point problem
   
3. **Gradient-based optimization**:
   - Backprop through Babai rounding
   - Learn A to minimize reconstruction error
   - Can use standard SGD/Adam

#### Entropy & Frequency Handling
- **Entropy**: Not explicitly mentioned
- **Frequency weighting**: Implicit (learned matrices adapt to data)
- **Symmetry**: Yes (lattices are symmetric)

#### Implementation Insights
```python
# Pseudocode for GLVQ
class LearnedLatticeQuantizer(nn.Module):
    def __init__(self, dim, block_size):
        super().__init__()
        # Learn transformation matrix per block
        self.A = nn.Parameter(torch.eye(dim))
    
    def babai_round(self, x):
        """Find nearest lattice point via Babai rounding."""
        # Solve A @ z ≈ x for z
        z = torch.linalg.solve(self.A, x)
        z_rounded = torch.round(z)
        return self.A @ z_rounded
    
    def forward(self, x):
        # Quantize via Babai rounding
        x_quant = self.babai_round(x)
        # Straight-through estimator for gradients
        return x + (x_quant - x).detach()
    
    def loss(self, x):
        x_quant = self.babai_round(x)
        return F.mse_loss(x, x_quant)
```

#### Advantages for Per-Block Search
- ✅ Learned codebooks (flexible)
- ✅ Differentiable quantization (Babai rounding)
- ✅ Lattice properties (optimal packing)
- ✅ Per-group optimization built-in

#### Limitations
- ❌ More complex than fixed codebooks
- ❌ Requires learning transformation matrix
- ❌ Babai rounding approximation error

---

### 4. AQLM (arXiv 2401.06118)
**Title**: AQLM: Additive Quantization with Learned Multi-Codebooks

#### Core Contribution
Uses multiple codebooks per weight, where each weight is represented as sum of selections from different codebooks. Enables finer quantization granularity with fewer bits per codebook.

#### Codebook Method
- **Type**: Multiple learned codebooks (e.g., 2-4 codebooks)
- **Representation**: weight ≈ codebook_1[idx_1] + codebook_2[idx_2] + ...
- **Bits**: Distributed across codebooks (e.g., 2+2 bits = 4 bits total)
- **Per-block**: Can be applied per-block

#### Per-Block Techniques
1. **Additive Quantization**:
   - Decompose weight into sum of codebook selections
   - Each codebook has fewer codewords (e.g., 4 for 2-bit)
   - Total bits = sum of individual codebook bits
   
2. **Learned Codebooks**:
   - Learn each codebook independently or jointly
   - Optimize to minimize reconstruction error
   - Can use EM or gradient-based methods
   
3. **Input-adaptive selection**:
   - Select codebook entries based on input distribution
   - Adapt to specific weight patterns per block

#### Entropy & Frequency Handling
- **Entropy**: Not explicitly mentioned
- **Frequency weighting**: Implicit (input-adaptive)
- **Symmetry**: Flexible (can be symmetric or asymmetric)

#### Implementation Insights
```python
# Pseudocode for AQLM
class AdditiveQuantizer(nn.Module):
    def __init__(self, dim, num_codebooks=2, bits_per_codebook=2):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = 2 ** bits_per_codebook
        
        # Learn multiple codebooks
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(self.codebook_size, dim))
            for _ in range(num_codebooks)
        ])
    
    def quantize(self, x):
        """Quantize x as sum of codebook selections."""
        residual = x.clone()
        indices = []
        reconstructed = torch.zeros_like(x)
        
        for codebook in self.codebooks:
            # Find nearest codeword for residual
            distances = torch.cdist(residual, codebook)
            idx = distances.argmin(dim=1)
            indices.append(idx)
            
            # Update residual and reconstruction
            selected = codebook[idx]
            reconstructed += selected
            residual = x - reconstructed
        
        return reconstructed, indices
    
    def forward(self, x):
        x_quant, _ = self.quantize(x)
        return x + (x_quant - x).detach()
```

#### Advantages for Per-Block Search
- ✅ Flexible bit allocation
- ✅ Multiple codebooks enable finer granularity
- ✅ Can be applied per-block
- ✅ Proven effective on LLMs

#### Limitations
- ❌ More complex (multiple codebooks)
- ❌ Higher search cost (product of codebook sizes)
- ❌ Requires careful initialization

---

### 5. QuIP# (arXiv 2402.04396)
**Title**: QuIP#: E8 Lattice Quantization with Hadamard Incoherence

#### Core Contribution
Uses E8 lattice (optimal 8D packing) for quantization. Applies Hadamard transform to reduce weight incoherence, enabling better lattice quantization. Combines fixed lattice with preprocessing.

#### Codebook Method
- **Type**: Fixed lattice codebook (E8 lattice)
- **Lattice**: E8 (optimal packing in 8D)
- **Preprocessing**: Hadamard transform for incoherence
- **Quantization**: Babai rounding on transformed weights

#### Per-Block Techniques
1. **Hadamard Transform**:
   - Apply orthogonal Hadamard matrix to weights
   - Reduces incoherence (max weight magnitude)
   - Enables better lattice quantization
   
2. **E8 Lattice Quantization**:
   - Quantize to nearest E8 lattice point
   - Babai rounding for efficiency
   - Optimal packing properties
   
3. **Gaussian assumption**:
   - Assumes weights are Gaussian-distributed
   - Optimal for Gaussian data

#### Entropy & Frequency Handling
- **Entropy**: Not explicitly mentioned
- **Frequency weighting**: Implicit (Gaussian assumption)
- **Symmetry**: Yes (E8 lattice is symmetric)

#### Implementation Insights
```python
# Pseudocode for QuIP#
def hadamard_transform(x):
    """Apply Hadamard transform for incoherence."""
    # Hadamard matrix (orthogonal)
    H = construct_hadamard_matrix(x.shape[-1])
    return x @ H.T

def quantize_e8_lattice(x):
    """Quantize to nearest E8 lattice point."""
    # E8 lattice basis (precomputed)
    e8_basis = get_e8_basis()
    
    # Babai rounding
    z = torch.linalg.solve(e8_basis, x)
    z_rounded = torch.round(z)
    return e8_basis @ z_rounded

def quip_quantize(weights):
    # Apply Hadamard transform
    weights_transformed = hadamard_transform(weights)
    
    # Quantize to E8 lattice
    weights_quant = quantize_e8_lattice(weights_transformed)
    
    # Inverse Hadamard transform
    weights_quant = hadamard_transform(weights_quant)
    
    return weights_quant
```

#### Advantages for Per-Block Search
- ✅ Fixed lattice (no learning needed)
- ✅ Optimal packing properties
- ✅ Hadamard preprocessing improves quantization
- ✅ Fast (Babai rounding)

#### Limitations
- ❌ Fixed codebook (not adaptive)
- ❌ Hadamard transform overhead
- ❌ Assumes Gaussian distribution

---

### 6. Float8@2bits (arXiv 2601.22787)
**Title**: Float8@2bits: Entropy Coding of Float8 Weights to 2 Bits

#### Core Contribution
Applies entropy coding to Float8 quantized weights, achieving 2-bit compression. Uses frequency-based variable-length codes to exploit non-uniform weight distributions.

#### Codebook Method
- **Type**: Entropy-coded codebook (variable-length codes)
- **Base format**: Float8 (8-bit floating point)
- **Compression**: Huffman or arithmetic coding
- **Per-block**: Can be applied per-block with block-specific codes

#### Per-Block Techniques
1. **Entropy Coding**:
   - Compute frequency histogram of Float8 values
   - Generate variable-length codes (Huffman)
   - Shorter codes for frequent values
   
2. **Block-specific codebooks**:
   - Different blocks may have different distributions
   - Generate separate codebook per block
   - Enables better compression
   
3. **Frequency-based optimization**:
   - Explicit frequency weighting
   - Optimize code lengths based on value frequency
   - Minimize expected code length

#### Entropy & Frequency Handling
- **Entropy**: Yes (explicit entropy coding)
- **Frequency weighting**: Yes (Huffman/arithmetic coding)
- **Symmetry**: No (variable-length codes break symmetry)

#### Implementation Insights
```python
# Pseudocode for Float8@2bits
def compute_frequency_histogram(weights):
    """Compute frequency of each Float8 value."""
    unique_vals, counts = torch.unique(weights, return_counts=True)
    frequencies = counts / counts.sum()
    return unique_vals, frequencies

def generate_huffman_codes(frequencies):
    """Generate Huffman codes for values."""
    # Build Huffman tree
    tree = build_huffman_tree(frequencies)
    
    # Generate variable-length codes
    codes = {}
    for val, freq in zip(frequencies.keys(), frequencies.values()):
        codes[val] = huffman_encode(tree, val)
    
    return codes

def entropy_encode_block(weights, block_size=128):
    """Entropy encode a block of weights."""
    blocks = reshape_into_blocks(weights, block_size)
    encoded = []
    codebooks = []
    
    for block in blocks:
        # Compute histogram
        unique_vals, frequencies = compute_frequency_histogram(block)
        
        # Generate Huffman codes
        codes = generate_huffman_codes(frequencies)
        codebooks.append(codes)
        
        # Encode block
        encoded_block = [codes[val] for val in block.flatten()]
        encoded.append(encoded_block)
    
    return encoded, codebooks

def entropy_decode_block(encoded, codebooks):
    """Decode entropy-encoded block."""
    decoded = []
    for encoded_block, codebook in zip(encoded, codebooks):
        # Reverse codebook (code -> value)
        reverse_codebook = {v: k for k, v in codebook.items()}
        
        # Decode
        decoded_block = [reverse_codebook[code] for code in encoded_block]
        decoded.append(decoded_block)
    
    return decoded
```

#### Advantages for Per-Block Search
- ✅ Explicit entropy optimization
- ✅ Frequency-based weighting
- ✅ Per-block codebooks natural
- ✅ Proven effective (2-bit compression)

#### Limitations
- ❌ Variable-length codes (decoding complexity)
- ❌ Breaks symmetry
- ❌ Requires codebook storage/transmission

---

## Comparative Analysis

### Codebook Type Comparison

| Paper | Type | Learned | Fixed | Lattice | Entropy |
|-------|------|---------|-------|---------|---------|
| Four Over Six | Fixed FP4 | ❌ | ✅ | ❌ | ❌ |
| BOF4 | Learned | ✅ | ❌ | ❌ | ❌ |
| GLVQ | Learned Lattice | ✅ | ❌ | ✅ | ❌ |
| AQLM | Learned Multi | ✅ | ❌ | ❌ | ❌ |
| QuIP# | Fixed Lattice | ❌ | ✅ | ✅ | ❌ |
| Float8@2bits | Entropy-coded | ✅ | ❌ | ❌ | ✅ |

### Optimization Method Comparison

| Paper | Method | Complexity | Speed | Differentiable |
|-------|--------|-----------|-------|---|
| Four Over Six | Scaling | Low | Fast | ✅ |
| BOF4 | EM | Medium | Slow | ⚠️ (variants exist) |
| GLVQ | Gradient-based | Medium | Medium | ✅ |
| AQLM | Gradient-based | Medium | Medium | ✅ |
| QuIP# | Hadamard + Babai | Low | Fast | ✅ |
| Float8@2bits | Huffman | Medium | Medium | ❌ |

### Per-Block Suitability

| Paper | Per-Block Native | Adaptation Cost | Overhead |
|-------|---|---|---|
| Four Over Six | ✅ | Low (scale only) | Minimal |
| BOF4 | ✅ | Medium (EM per block) | Codebook storage |
| GLVQ | ✅ | Medium (learn A per block) | Matrix storage |
| AQLM | ✅ | Medium (learn codebooks) | Multiple codebooks |
| QuIP# | ⚠️ | Low (Hadamard fixed) | Minimal |
| Float8@2bits | ✅ | Medium (Huffman per block) | Codebook storage |

---

## Recommended Implementation Strategy

### Phase 1: Baseline (Four Over Six)
**Goal**: Implement simplest per-block technique
- ✅ Minimal code changes
- ✅ Proven effective
- ✅ Foundation for comparison

**Implementation**:
```python
# tensorrt_llm/quantization/per_block_codebook.py
class PerBlockAdaptiveScaling:
    """Four Over Six: Adaptive per-block scaling for FP4."""
    
    def __init__(self, block_size=128):
        self.block_size = block_size
    
    def quantize(self, weights):
        """Quantize weights with per-block scaling."""
        # Reshape into blocks
        blocks = self._reshape_blocks(weights)
        
        # Compute per-block scales
        scales = []
        quantized = []
        
        for block in blocks:
            scale = self._compute_scale(block)
            scales.append(scale)
            
            # Quantize to FP4
            scaled = block / scale
            quantized_block = self._quantize_fp4(scaled)
            quantized.append(quantized_block)
        
        return torch.cat(quantized), torch.stack(scales)
    
    def _compute_scale(self, block):
        """Compute optimal scale for block."""
        return torch.max(torch.abs(block)) / FP4_MAX_VALUE
    
    def _quantize_fp4(self, x):
        """Quantize to FP4 codebook."""
        # Round to nearest FP4 value
        return torch.round(x * 8) / 8  # Simplified
```

### Phase 2: EM-based Learning (BOF4)
**Goal**: Implement learned codebook optimization
- ✅ Better compression than fixed codebooks
- ✅ Handles outliers
- ⚠️ Slower (EM iterations)

**Implementation**:
```python
class PerBlockEMCodebook:
    """BOF4: EM-optimized codebook per block."""
    
    def __init__(self, block_size=128, num_codewords=16, max_iters=100):
        self.block_size = block_size
        self.num_codewords = num_codewords
        self.max_iters = max_iters
    
    def learn_codebook(self, weights):
        """Learn codebook using EM algorithm."""
        blocks = self._reshape_blocks(weights)
        codebooks = []
        outlier_masks = []
        
        for block in blocks:
            codebook, outlier_mask = self._em_algorithm(block)
            codebooks.append(codebook)
            outlier_masks.append(outlier_mask)
        
        return codebooks, outlier_masks
    
    def _em_algorithm(self, block):
        """EM algorithm for codebook learning."""
        # Initialize codebook (k-means)
        codebook = self._initialize_codebook(block)
        
        for iteration in range(self.max_iters):
            # E-step: assign to nearest codeword
            assignments = self._assign_nearest(block, codebook)
            
            # M-step: update codebook
            codebook = self._update_codebook(block, assignments)
        
        # Identify outliers
        errors = self._compute_errors(block, codebook, assignments)
        outlier_mask = errors > self._compute_threshold(errors)
        
        return codebook, outlier_mask
```

### Phase 3: Lattice-based (GLVQ)
**Goal**: Implement learned lattice quantization
- ✅ Optimal packing properties
- ✅ Differentiable (Babai rounding)
- ⚠️ More complex

**Implementation**:
```python
class PerBlockLearnedLattice(nn.Module):
    """GLVQ: Learned lattice quantization per block."""
    
    def __init__(self, block_size=128, dim=128):
        super().__init__()
        self.block_size = block_size
        self.dim = dim
        
        # Learn transformation matrix per block
        self.A = nn.Parameter(torch.eye(dim))
    
    def babai_round(self, x):
        """Babai rounding for nearest lattice point."""
        z = torch.linalg.solve(self.A, x)
        z_rounded = torch.round(z)
        return self.A @ z_rounded
    
    def forward(self, x):
        """Quantize via Babai rounding."""
        x_quant = self.babai_round(x)
        # Straight-through estimator
        return x + (x_quant - x).detach()
    
    def loss(self, x):
        """Reconstruction loss."""
        x_quant = self.babai_round(x)
        return F.mse_loss(x, x_quant)
```

### Phase 4: Entropy Coding (Float8@2bits)
**Goal**: Implement entropy-based compression
- ✅ Explicit frequency optimization
- ✅ Proven 2-bit compression
- ⚠️ Variable-length codes (decoding complexity)

**Implementation**:
```python
class PerBlockEntropyCodebook:
    """Float8@2bits: Entropy coding per block."""
    
    def __init__(self, block_size=128):
        self.block_size = block_size
    
    def encode_block(self, weights):
        """Entropy encode a block."""
        blocks = self._reshape_blocks(weights)
        encoded = []
        codebooks = []
        
        for block in blocks:
            # Compute frequency histogram
            unique_vals, frequencies = torch.unique(
                block, return_counts=True
            )
            frequencies = frequencies / frequencies.sum()
            
            # Generate Huffman codes
            codes = self._generate_huffman_codes(unique_vals, frequencies)
            codebooks.append(codes)
            
            # Encode block
            encoded_block = self._encode_with_codes(block, codes)
            encoded.append(encoded_block)
        
        return encoded, codebooks
    
    def _generate_huffman_codes(self, values, frequencies):
        """Generate Huffman codes for values."""
        # Build Huffman tree and generate codes
        # (Implementation details omitted)
        pass
```

---

## Integration with TensorRT-LLM

### File Structure
```
tensorrt_llm/
├── quantization/
│   ├── per_block_codebook.py          # New: Per-block codebook base
│   ├── per_block_adaptive_scaling.py  # New: Four Over Six
│   ├── per_block_em_codebook.py       # New: BOF4
│   ├── per_block_lattice.py           # New: GLVQ
│   ├── per_block_entropy.py           # New: Float8@2bits
│   └── per_block_aqlm.py              # New: AQLM
└── _torch/auto_deploy/
    └── transform/library/
        └── per_block_quantization.py  # New: Integration layer
```

### Integration Points
1. **Quantization pipeline**: Add per-block codebook as option
2. **Weight loading**: Support per-block codebook storage
3. **Inference**: Use per-block codebooks during dequantization
4. **Training**: Support gradient-based codebook learning

---

## Experimental Validation Plan

### Benchmark Setup
- **Model**: Llama-7B (or smaller for quick iteration)
- **Dataset**: WikiText-2 (perplexity), Hellaswag (accuracy)
- **Metrics**: 
  - Bits per parameter
  - Perplexity (WikiText)
  - Task accuracy (Hellaswag, MMLU)
  - Inference speed (tokens/sec)
  - Codebook size (memory overhead)

### Comparison Matrix
```
Method              | Bits | PPL  | Accuracy | Speed | Overhead
Four Over Six       | 4.0  | ?    | ?        | ?     | Minimal
BOF4                | 4.0  | ?    | ?        | ?     | Codebook
GLVQ                | 4.0  | ?    | ?        | ?     | Matrix
AQLM                | 4.0  | ?    | ?        | ?     | Codebooks
QuIP#               | 4.0  | ?    | ?        | ?     | Minimal
Float8@2bits        | 2.0  | ?    | ?        | ?     | Codebook
```

---

## Next Steps

1. **Implement Phase 1** (Four Over Six adaptive scaling)
   - Create `per_block_codebook.py` base class
   - Implement adaptive scaling
   - Test on small model
   - Benchmark vs. global scaling

2. **Implement Phase 2** (BOF4 EM)
   - Add EM algorithm
   - Implement outlier preservation
   - Compare with Phase 1

3. **Implement Phase 3** (GLVQ lattice)
   - Add Babai rounding
   - Learn transformation matrices
   - Compare with Phase 1-2

4. **Implement Phase 4** (Entropy coding)
   - Add Huffman/arithmetic coding
   - Per-block codebook generation
   - Compare with all phases

5. **Integration & Deployment**
   - Integrate into TensorRT-LLM pipeline
   - Support model loading/saving
   - Benchmark on full models

---

## References

1. Four Over Six (2512.02010) — Adaptive block scaling for NVFP4
2. BOF4 (2505.06653) — EM-optimized codebook + outlier preservation
3. GLVQ (2510.20984) — Per-group learned lattice codebooks
4. AQLM (2401.06118) — Additive multi-codebook VQ
5. QuIP# (2402.04396) — E8 lattice + Hadamard incoherence
6. Float8@2bits (2601.22787) — Entropy coding of Float8 weights to 2 bits

---

**Document Status**: Ready for implementation
**Last Review**: 2026-03-30
**Next Review**: After Phase 1 implementation
