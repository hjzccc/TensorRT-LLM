# Phase 28: Lossless zstd Compression — Completion Report

**Date**: March 30, 2026
**Status**: ✅ COMPLETE

## Summary

Applied lossless zstd-3 compression to the per-block codebook compressed checkpoint,
achieving **14.95% additional size reduction** with **zero accuracy impact**.

## Results

| Metric | Value |
|--------|-------|
| Input size | 16.62 GB |
| Output size | 14.13 GB |
| Savings | 2.48 GB (15.0%) |
| Compression ratio vs NVFP4 | **1.51x** |
| Time | 53s |
| Tensors compressed | 92,520 |
| Tensors passthrough | 62,063 |

## Compression Breakdown

| Data Type | Savings |
|-----------|---------|
| weight_indices (2-bit) | ~5.0% |
| weight_codebook_entries | ~39.2% |
| weight_scale (FP8) | ~50.7% |
| **Overall compressible** | **20.4%** |

## Why This Works

1. **Codebook entries** (38.8% savings): Many blocks choose similar FP4 codes.
   Repeated patterns across blocks are captured by zstd's LZ77 + Huffman.

2. **Scale data** (50.7% savings): The FP8 E4M3 scales have only 64 unique values
   in this model. zstd efficiently encodes this low-entropy data.

3. **Index data** (5.0% savings): 2-bit indices are near-uniform but have some
   structural patterns (runs of zeros for sparse experts).

## Verification

- Decompression verified lossless on 20 tensors from shard 12
- All tensors match exactly after compress/decompress cycle

## Files

-  — Compression script
-  — Results
-  — Output checkpoint (14.13 GB)

## Compression History

| Phase | Method | Size | Ratio vs NVFP4 |
|-------|--------|------|----------------|
| Baseline | NVFP4 | 21.28 GB | 1.00x |
| Phase 4-22 | 2b075b codebook | 16.62 GB | 1.28x |
| **Phase 28** | **+ zstd-3** | **14.13 GB** | **1.51x** |
