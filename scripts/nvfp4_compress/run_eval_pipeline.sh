#!/bin/bash
# Sequential evaluation pipeline - runs after C finishes
# Waits for GPU to be free, then runs all remaining evals

set -e
SCRIPTS="/code/tensorrt_llm/scripts/nvfp4_compress"
EVAL_CMD="python3 $SCRIPTS/run_mmlu_direct.py --offline --subject professional_law --batch-size 64 --max-batch-total-tokens 12000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code/tensorrt_llm

wait_for_gpu() {
    echo "Waiting for GPU to be free..."
    while true; do
        GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$GPU_PROCS" -eq 0 ]; then
            echo "GPU is free"
            break
        fi
        sleep 10
    done
}

run_eval() {
    local name=$1
    local ckpt=$2
    local output=$3
    local log=$4
    echo "=== Evaluating $name ==="
    wait_for_gpu
    $EVAL_CMD --ckpt-dir $ckpt --output $output > $log 2>&1
    echo "EXIT:$?" >> $log
    echo "=== $name done ==="
    # Show result
    python3 -c "import json; d=json.load(open('$output')); print(f'$name: MMLU={d[\"mmlu_acc\"]:.4f}')" 2>/dev/null || echo "$name: result file not found"
}

# A: exact MSE (2.75 bits)
run_eval "A_exact_mse" \
    "$SCRIPTS/decompressed_2b075b_zero_fixed_exact" \
    "$SCRIPTS/result_A_exact_mse.json" \
    "$SCRIPTS/eval_A.log"

# B: weighted_abs MSE (2.75 bits)
run_eval "B_weighted_abs" \
    "$SCRIPTS/decompressed_2b075b_zero_fixed_weighted_abs" \
    "$SCRIPTS/result_B_weighted_abs.json" \
    "$SCRIPTS/eval_B.log"

# 2b1b_demo (2.0625 bits)
run_eval "2b1b_demo" \
    "$SCRIPTS/decompressed_2b1b_demo" \
    "$SCRIPTS/result_2b1b_demo.json" \
    "$SCRIPTS/eval_2b1b_demo.log"

# Wait for D compression to finish, then decompress and eval
echo "Waiting for D compression..."
while [ ! -f "$SCRIPTS/compress_all_done.flag" ]; do
    sleep 30
    echo "Still waiting for compressions..."
done
echo "Compressions done"

# Decompress D, E, F, G
python3 $SCRIPTS/decompress_checkpoint.py \
    --input $SCRIPTS/compressed_2b075b_zero_fixed_scale_weighted \
    --output $SCRIPTS/decompressed_D_scale_weighted \
    > $SCRIPTS/decompress_D.log 2>&1
echo "D decompressed"

python3 $SCRIPTS/decompress_checkpoint.py \
    --input $SCRIPTS/compressed_2b075b_zero_fixed_freq_sq \
    --output $SCRIPTS/decompressed_E_freq_sq \
    > $SCRIPTS/decompress_E.log 2>&1
echo "E decompressed"

python3 $SCRIPTS/decompress_checkpoint.py \
    --input $SCRIPTS/compressed_3b1b_4free_exact \
    --output $SCRIPTS/decompressed_F_3bit_exact \
    > $SCRIPTS/decompress_F.log 2>&1
echo "F decompressed"

python3 $SCRIPTS/decompress_checkpoint.py \
    --input $SCRIPTS/compressed_3b1b_4free_weighted_abs \
    --output $SCRIPTS/decompressed_G_3bit_weighted \
    > $SCRIPTS/decompress_G.log 2>&1
echo "G decompressed"

# D: scale_weighted (2.75 bits, BOF4-style)
run_eval "D_scale_weighted" \
    "$SCRIPTS/decompressed_D_scale_weighted" \
    "$SCRIPTS/result_D_scale_weighted.json" \
    "$SCRIPTS/eval_D.log"

# E: freq_sq (2.75 bits)
run_eval "E_freq_sq" \
    "$SCRIPTS/decompressed_E_freq_sq" \
    "$SCRIPTS/result_E_freq_sq.json" \
    "$SCRIPTS/eval_E.log"

# F: 3bit exact (3.0 bits, no fixed zero)
run_eval "F_3bit_exact" \
    "$SCRIPTS/decompressed_F_3bit_exact" \
    "$SCRIPTS/result_F_3bit_exact.json" \
    "$SCRIPTS/eval_F.log"

# G: 3bit weighted_abs (3.0 bits)
run_eval "G_3bit_weighted" \
    "$SCRIPTS/decompressed_G_3bit_weighted" \
    "$SCRIPTS/result_G_3bit_weighted.json" \
    "$SCRIPTS/eval_G.log"

echo "=== ALL EVALUATIONS COMPLETE ==="
echo "PIPELINE_DONE" > $SCRIPTS/eval_pipeline_done.flag

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
echo "Baseline (NVFP4): 59.78%"
for f in $SCRIPTS/result_*.json; do
    name=$(basename $f .json)
    python3 -c "import json; d=json.load(open('$f')); print(f'$name: MMLU={d[\"mmlu_acc\"]:.4f} ({d[\"mmlu_acc\"]*100:.2f}%)')" 2>/dev/null || true
done
