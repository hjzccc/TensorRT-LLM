#!/bin/bash
# Run all remaining compressions and evaluations sequentially
# This script is designed to run AFTER eval_A completes

set -e
SCRIPT_DIR="/code/tensorrt_llm/scripts/nvfp4_compress"
COMPRESS="python3 $SCRIPT_DIR/compress_checkpoint.py"
DECOMPRESS="python3 $SCRIPT_DIR/decompress_checkpoint.py"
EVAL="python3 $SCRIPT_DIR/run_mmlu_direct.py --offline --subject professional_law --batch-size 64 --max-batch-total-tokens 12000"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code/tensorrt_llm

echo "=== PHASE 1: Compress remaining schemes ==="

# D: scale_weighted (BOF4 block-scale-weighted MSE)
echo "[D] Compressing 2b075b_zero_fixed_scale_weighted..."
$COMPRESS --scheme 2b075b_zero_fixed_scale_weighted \
  > $SCRIPT_DIR/compress_D_scale_weighted.log 2>&1
echo "[D] Compress done"

# E: freq_sq (frequency-squared weighting)
echo "[E] Compressing 2b075b_zero_fixed_freq_sq..."
$COMPRESS --scheme 2b075b_zero_fixed_freq_sq \
  > $SCRIPT_DIR/compress_E_freq_sq.log 2>&1
echo "[E] Compress done"

# F: 3-bit exact (no fixed zero, 4 free codes)
echo "[F] Compressing 3b1b_4free_exact..."
$COMPRESS --scheme 3b1b_4free_exact \
  > $SCRIPT_DIR/compress_F_3bit_exact.log 2>&1
echo "[F] Compress done"

# G: 3-bit weighted_abs
echo "[G] Compressing 3b1b_4free_weighted_abs..."
$COMPRESS --scheme 3b1b_4free_weighted_abs \
  > $SCRIPT_DIR/compress_G_3bit_weighted.log 2>&1
echo "[G] Compress done"

echo "=== PHASE 2: Decompress all ==="

$DECOMPRESS --input $SCRIPT_DIR/compressed_2b075b_zero_fixed_scale_weighted \
  --output $SCRIPT_DIR/decompressed_D_scale_weighted \
  > $SCRIPT_DIR/decompress_D.log 2>&1
echo "[D] Decompress done"

$DECOMPRESS --input $SCRIPT_DIR/compressed_2b075b_zero_fixed_freq_sq \
  --output $SCRIPT_DIR/decompressed_E_freq_sq \
  > $SCRIPT_DIR/decompress_E.log 2>&1
echo "[E] Decompress done"

$DECOMPRESS --input $SCRIPT_DIR/compressed_3b1b_4free_exact \
  --output $SCRIPT_DIR/decompressed_F_3bit_exact \
  > $SCRIPT_DIR/decompress_F.log 2>&1
echo "[F] Decompress done"

$DECOMPRESS --input $SCRIPT_DIR/compressed_3b1b_4free_weighted_abs \
  --output $SCRIPT_DIR/decompressed_G_3bit_weighted \
  > $SCRIPT_DIR/decompress_G.log 2>&1
echo "[G] Decompress done"

echo "=== PHASE 3: Evaluate B (weighted_abs) ==="
$EVAL \
  --ckpt-dir $SCRIPT_DIR/decompressed_2b075b_zero_fixed_weighted_abs \
  --output $SCRIPT_DIR/result_B_weighted_abs.json \
  > $SCRIPT_DIR/eval_B.log 2>&1
echo "[B] Eval done"

echo "=== PHASE 4: Evaluate C (freq_symmetric) ==="
$EVAL \
  --ckpt-dir $SCRIPT_DIR/decompressed_2b1b_freq_symmetric \
  --output $SCRIPT_DIR/result_C_freq_sym.json \
  > $SCRIPT_DIR/eval_C.log 2>&1
echo "[C] Eval done"

echo "=== PHASE 5: Evaluate 2b1b_demo ==="
$EVAL \
  --ckpt-dir $SCRIPT_DIR/decompressed_2b1b_demo \
  --output $SCRIPT_DIR/result_2b1b_demo.json \
  > $SCRIPT_DIR/eval_2b1b_demo.log 2>&1
echo "[2b1b_demo] Eval done"

echo "=== PHASE 6: Evaluate D (scale_weighted) ==="
$EVAL \
  --ckpt-dir $SCRIPT_DIR/decompressed_D_scale_weighted \
  --output $SCRIPT_DIR/result_D_scale_weighted.json \
  > $SCRIPT_DIR/eval_D.log 2>&1
echo "[D] Eval done"

echo "=== PHASE 7: Evaluate E (freq_sq) ==="
$EVAL \
  --ckpt-dir $SCRIPT_DIR/decompressed_E_freq_sq \
  --output $SCRIPT_DIR/result_E_freq_sq.json \
  > $SCRIPT_DIR/eval_E.log 2>&1
echo "[E] Eval done"

echo "=== PHASE 8: Evaluate F (3bit exact) ==="
$EVAL \
  --ckpt-dir $SCRIPT_DIR/decompressed_F_3bit_exact \
  --output $SCRIPT_DIR/result_F_3bit_exact.json \
  > $SCRIPT_DIR/eval_F.log 2>&1
echo "[F] Eval done"

echo "=== PHASE 9: Evaluate G (3bit weighted) ==="
$EVAL \
  --ckpt-dir $SCRIPT_DIR/decompressed_G_3bit_weighted \
  --output $SCRIPT_DIR/result_G_3bit_weighted.json \
  > $SCRIPT_DIR/eval_G.log 2>&1
echo "[G] Eval done"

echo "=== ALL DONE ==="
