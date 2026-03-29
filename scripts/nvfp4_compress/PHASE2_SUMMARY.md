# Phase 2 Summary: Fixed Codebook Compression

## Critical Bug Discovery & Fix

### The Bug
The `apply_code_lut` function in `sub_fp4_compress.py` was performing tensor indexing with operands on different devices:
```python
def apply_code_lut(codes: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    return lut[codes.long()]  # BUG: codes on GPU, lut on CPU
```

This caused incorrect indexing behavior, resulting in catastrophic PPL failures:
- **3bit_uniform**: PPL = 1,817,424.66 (vs expected ~6.7)
- **3bit_dense**: PPL = 31,365,116.58 (vs expected ~6.7)

### The Fix
```python
def apply_code_lut(codes: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    # Move to CPU for indexing
    codes_cpu = codes.cpu() if codes.device.type == 'cuda' else codes
    lut_cpu = lut.cpu() if lut.device.type == 'cuda' else lut
    
    # Apply LUT on CPU
    remapped_cpu = lut_cpu[codes_cpu.long()]
    
    # Move back to original device
    return remapped_cpu.to(codes.device)
```

## Phase 2 Results

### Baseline (Identity Mapping)
- **Codebook**: Full NVFP4 [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6]
- **PPL**: 6.6974 ✓
- **Time**: 618 seconds
- **Status**: VERIFIED CORRECT

### Buggy Experiments (Before Fix)
- **3bit_uniform**: PPL = 1,817,424.66 ✗
- **3bit_dense**: PPL = 31,365,116.58 ✗
- **Root Cause**: Device mismatch in LUT indexing

### Fixed Experiments (After Fix)
- **3bit_uniform (fixed)**: Running... (ETA 37 minutes)
- **Expected**: PPL ~6.7-6.9 (small degradation from baseline)

## Next Steps

1. **Wait for 3bit_uniform (fixed) to complete** (~37 minutes)
2. **Run remaining codebooks** with fixed script:
   - 3bit_dense
   - 3bit_truncate
   - 2bit_opt1, 2bit_opt3, 2bit_uniform
3. **Analyze results** and select best codebook
4. **Proceed to Phase 3** (per-block optimal codebook selection)

## Key Learnings

1. **Device handling is critical** in PyTorch tensor operations
2. **Tensor indexing requires same device** for both operands
3. **Testing on minimal examples** helped identify the bug
4. **Partial evaluation** showed the bug was cumulative (3 layers OK, 40 layers failed)

## Timeline

- **Phase 1**: Complete ✓ (FP4 pack/unpack utilities)
- **Phase 2**: In Progress (Fixed codebook experiments)
  - Baseline: 618s
  - 3bit_uniform (fixed): ~1330s (ETA 37 min)
  - Remaining: ~6 × 1330s = ~8000s (~2.2 hours)
- **Phase 3**: Ready (Per-block optimal codebook)
- **Phase 4**: Ready (Learned codebooks)

**Total estimated time for Phase 2**: ~3 hours

