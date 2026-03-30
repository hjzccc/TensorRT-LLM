#!/bin/bash
# Sequential pipeline for remaining experiments
# Run inside trtllm-dual-tile container
# Usage: docker exec trtllm-dual-tile bash -lc "cd /code/tensorrt_llm && bash scripts/nvfp4_compress/run_next_pipeline.sh"

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code/tensorrt_llm
BASE="scripts/nvfp4_compress"
NVFP4_CKPT="$BASE/nvfp4_checkpoint"

run_eval() {
    local name="$1"
    local ckpt_dir="$2"
    local output="$BASE/result_${name}.json"
    local log="$BASE/eval_${name}.log"
    echo "=== EVAL: $name ===" | tee -a "$log"
    python3 -u "$BASE/run_mmlu_direct.py" \
        --offline \
        --subject professional_law \
        --batch-size 64 \
        --max-batch-total-tokens 12000 \
        --ckpt-dir "$ckpt_dir" \
        --output "$output" \
        >> "$log" 2>&1
    echo "EXIT:$?" >> "$log"
    python3 -c "import json; d=json.load(open('$output')); print('$name: acc=', d['mmlu_acc'])" 2>/dev/null || echo "$name: FAILED"
}

compress_and_eval() {
    local scheme="$1"
    local compressed="$BASE/compressed_${scheme}"
    local decompressed="$BASE/decompressed_${scheme}"
    
    if [ ! -f "$compressed/compression_manifest.json" ]; then
        echo "=== COMPRESS: $scheme ==="
        python3 -u "$BASE/compress_checkpoint.py" \
            --scheme "$scheme" \
            --input "$NVFP4_CKPT" \
            --output "$compressed" \
            > "$BASE/compress_${scheme}.log" 2>&1
        echo "Compress exit: $?"
    else
        echo "=== SKIP COMPRESS: $scheme (already done) ==="
    fi
    
    if [ ! -d "$decompressed" ] || [ ! -f "$decompressed/config.json" ]; then
        echo "=== DECOMPRESS: $scheme ==="
        python3 -u "$BASE/decompress_checkpoint.py" \
            --input "$compressed" \
            --output "$decompressed" \
            > "$BASE/decompress_${scheme}.log" 2>&1
        echo "Decompress exit: $?"
    else
        echo "=== SKIP DECOMPRESS: $scheme (already done) ==="
    fi
    
    run_eval "$scheme" "$decompressed"
}

echo "=== PIPELINE START ==="
echo "Baseline (nvfp4_checkpoint): professional_law = 0.6271"
echo ""

# Step 1: Eval exact MSE (already decompressed)
echo "--- Step 1: exact MSE ---"
run_eval "exact_mse" "$BASE/decompressed_2b075b_zero_fixed_exact"

# Step 2: Eval weighted_abs (already decompressed)
echo "--- Step 2: weighted_abs ---"
run_eval "weighted_abs" "$BASE/decompressed_2b075b_zero_fixed_weighted_abs"

# Step 3: Wait for scale_weighted compression, then eval
echo "--- Step 3: scale_weighted ---"
while [ ! -f "$BASE/compressed_2b075b_zero_fixed_scale_weighted/compression_manifest.json" ]; do
    echo "Waiting for scale_weighted compression... ($(ls $BASE/compressed_2b075b_zero_fixed_scale_weighted/ 2>/dev/null | wc -l)/733 shards)"
    sleep 60
done
echo "scale_weighted compression done!"
compress_and_eval "2b075b_zero_fixed_scale_weighted"

# Step 4: freq_sq
echo "--- Step 4: freq_sq ---"
compress_and_eval "2b075b_zero_fixed_freq_sq"

# Step 5: scale_linear (new)
echo "--- Step 5: scale_linear ---"
compress_and_eval "2b075b_zero_fixed_scale_linear"

# Step 6: 3b1b_4free_exact (3.0 bits/elem)
echo "--- Step 6: 3b1b_4free_exact ---"
compress_and_eval "3b1b_4free_exact"

# Step 7: 3b1b_4free_weighted_abs
echo "--- Step 7: 3b1b_4free_weighted_abs ---"
compress_and_eval "3b1b_4free_weighted_abs"

# Step 8: 3b1b_4free_scale_weighted (new)
echo "--- Step 8: 3b1b_4free_scale_weighted ---"
compress_and_eval "3b1b_4free_scale_weighted"

echo "=== PIPELINE COMPLETE ==="
