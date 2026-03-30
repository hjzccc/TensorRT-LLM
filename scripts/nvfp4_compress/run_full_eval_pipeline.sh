#!/bin/bash
set -e
SCRIPT_DIR="/code/tensorrt_llm/scripts/nvfp4_compress"
DECOMPRESS="python3 $SCRIPT_DIR/decompress_checkpoint.py"
EVAL="python3 $SCRIPT_DIR/run_mmlu_direct.py --offline --batch-size 64 --max-batch-total-tokens 12000"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code/tensorrt_llm

echo "=== STEP 1: Decompress scale_weighted (D) ==="
if [ ! -d "$SCRIPT_DIR/decompressed_2b075b_zero_fixed_scale_weighted" ]; then
    $DECOMPRESS \
        --input $SCRIPT_DIR/compressed_2b075b_zero_fixed_scale_weighted \
        --output $SCRIPT_DIR/decompressed_2b075b_zero_fixed_scale_weighted \
        > $SCRIPT_DIR/decompress_D_scale_weighted.log 2>&1
    echo "[D] Decompress done"
else
    echo "[D] Already decompressed, skipping"
fi

echo "=== STEP 2: Evaluate all compressed variants ==="

# A: Original NVFP4 (baseline - already done, skip)
echo "[BASELINE] Already evaluated: 77.98% MMLU"

# B: weighted_abs (decompressed_2b075b_zero_fixed_weighted_abs)
echo "[B] Evaluating weighted_abs..."
$EVAL \
    --ckpt-dir $SCRIPT_DIR/decompressed_2b075b_zero_fixed_weighted_abs \
    --output $SCRIPT_DIR/result_B_weighted_abs.json \
    > $SCRIPT_DIR/eval_B_weighted_abs.log 2>&1
echo "[B] Done"

# C: exact MSE (decompressed_2b075b_zero_fixed_exact = B+D)
echo "[C/B+D] Evaluating exact MSE (B+D)..."
$EVAL \
    --ckpt-dir $SCRIPT_DIR/decompressed_2b075b_zero_fixed_exact \
    --output $SCRIPT_DIR/result_BD_exact.json \
    > $SCRIPT_DIR/eval_BD_exact.log 2>&1
echo "[C/B+D] Done"

# D: scale_weighted
echo "[D] Evaluating scale_weighted..."
$EVAL \
    --ckpt-dir $SCRIPT_DIR/decompressed_2b075b_zero_fixed_scale_weighted \
    --output $SCRIPT_DIR/result_D_scale_weighted.json \
    > $SCRIPT_DIR/eval_D_scale_weighted.log 2>&1
echo "[D] Done"

# E: freq_symmetric (global 2-codebook)
echo "[E] Evaluating freq_symmetric..."
$EVAL \
    --ckpt-dir $SCRIPT_DIR/decompressed_2b1b_freq_symmetric \
    --output $SCRIPT_DIR/result_E_freq_sym.json \
    > $SCRIPT_DIR/eval_E_freq_sym.log 2>&1
echo "[E] Done"

echo "=== ALL EVALS COMPLETE ==="
