#!/bin/bash
# Sequential evaluation pipeline for all compression variants
# Run inside trtllm-dual-tile container
set -e

CKPT_BASE="/code/tensorrt_llm/scripts/nvfp4_compress"
EVAL_SCRIPT="$CKPT_BASE/run_mmlu_direct.py"
COMPRESS_SCRIPT="$CKPT_BASE/compress_checkpoint.py"
DECOMPRESS_SCRIPT="$CKPT_BASE/decompress_checkpoint.py"
NVFP4_CKPT="$CKPT_BASE/nvfp4_checkpoint"

run_eval() {
    local name="$1"
    local ckpt_dir="$2"
    local output_json="$CKPT_BASE/mmlu_${name}_results.json"
    local log="$CKPT_BASE/mmlu_${name}.log"
    echo "=== EVAL: $name ==="
    python3 -u "$EVAL_SCRIPT" \
        --offline \
        --ckpt-dir "$ckpt_dir" \
        --subject professional_law \
        --batch-size 64 \
        --max-batch-total-tokens 12000 \
        --output "$output_json" \
        > "$log" 2>&1
    echo "EXIT:$?" >> "$log"
    # Extract result
    python3 -c "import json; d=json.load(open('$output_json')); print('$name: acc=', d['mmlu_acc'])" 2>/dev/null || echo "$name: FAILED"
}

compress_and_decompress() {
    local scheme="$1"
    local compressed_dir="$CKPT_BASE/compressed_${scheme}"
    local decompressed_dir="$CKPT_BASE/decompressed_${scheme}"
    echo "=== COMPRESS: $scheme ==="
    python3 -u "$COMPRESS_SCRIPT" \
        --input "$NVFP4_CKPT" \
        --output "$compressed_dir" \
        --scheme "$scheme" \
        > "$CKPT_BASE/compress_${scheme}.log" 2>&1
    echo "=== DECOMPRESS: $scheme ==="
    python3 -u "$DECOMPRESS_SCRIPT" \
        --input "$compressed_dir" \
        --output "$decompressed_dir" \
        > "$CKPT_BASE/decompress_${scheme}.log" 2>&1
}

# Variant B: weighted_abs (already compressed, need to finish decompression)
echo "=== Finishing decompression of weighted_abs ==="
rm -rf "$CKPT_BASE/decompressed_2b075b_zero_fixed_weighted_abs"
python3 -u "$DECOMPRESS_SCRIPT" \
    --input "$CKPT_BASE/compressed_2b075b_zero_fixed_weighted_abs" \
    --output "$CKPT_BASE/decompressed_2b075b_zero_fixed_weighted_abs" \
    > "$CKPT_BASE/decompress_2b075b_weighted.log" 2>&1
echo "Decompression done"

# Eval B
run_eval "variantB_weighted_abs" "$CKPT_BASE/decompressed_2b075b_zero_fixed_weighted_abs"

# Variant C: scale_weighted (new)
compress_and_decompress "2b075b_zero_fixed_scale_weighted"
run_eval "variantC_scale_weighted" "$CKPT_BASE/decompressed_2b075b_zero_fixed_scale_weighted"

# Variant D: freq_sq (new)
compress_and_decompress "2b075b_zero_fixed_freq_sq"
run_eval "variantD_freq_sq" "$CKPT_BASE/decompressed_2b075b_zero_fixed_freq_sq"

echo "=== ALL EVALS COMPLETE ==="
