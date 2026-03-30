#!/bin/bash
# Sequential pipeline: compress remaining schemes, decompress, evaluate all variants
# Designed to run inside Docker container, survives interruptions via result file checks

set -euo pipefail
SCRIPTS="/code/tensorrt_llm/scripts/nvfp4_compress"
CKPT="$SCRIPTS/nvfp4_checkpoint"
EVAL_CMD="python3 -u $SCRIPTS/run_mmlu_direct.py --offline --subject professional_law --batch-size 64 --max-batch-total-tokens 12000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code/tensorrt_llm

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$SCRIPTS/sequential_pipeline.log"; }

# ── HELPERS ──────────────────────────────────────────────────────────────────

compress_if_needed() {
    local scheme=$1; local out_dir="$SCRIPTS/compressed_${scheme}"
    local manifest="$out_dir/compression_manifest.json"
    if [ -f "$manifest" ]; then
        log "SKIP compress $scheme (manifest exists)"
        return 0
    fi
    log "Compressing $scheme ..."
    python3 "$SCRIPTS/compress_checkpoint.py" --scheme "$scheme" --input "$CKPT" \
        > "$SCRIPTS/compress_${scheme}.log" 2>&1
    log "Done compress $scheme"
}

decompress_if_needed() {
    local src_dir=$1; local dst_dir=$2
    if [ -f "$dst_dir/config.json" ]; then
        log "SKIP decompress $dst_dir (config.json exists)"
        return 0
    fi
    log "Decompressing $src_dir -> $dst_dir ..."
    python3 "$SCRIPTS/decompress_checkpoint.py" --input "$src_dir" --output "$dst_dir" \
        > "$SCRIPTS/decompress_$(basename $dst_dir).log" 2>&1
    # Copy config files from reference checkpoint
    local ref="$SCRIPTS/decompressed_2b075b_zero_fixed_exact"
    cp "$ref/config.json" "$dst_dir/config.json"
    cp "$ref/hf_quant_config.json" "$dst_dir/hf_quant_config.json"
    cp "$ref/model.safetensors.index.json" "$dst_dir/model.safetensors.index.json"
    log "Done decompress $dst_dir"
}

eval_if_needed() {
    local name=$1; local ckpt_dir=$2; local result_file=$3
    if [ -f "$result_file" ]; then
        local acc
        acc=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d[\"mmlu_acc\"]:.4f}')" 2>/dev/null || echo "?")
        log "SKIP eval $name (result exists: acc=$acc)"
        return 0
    fi
    log "Evaluating $name ..."
    $EVAL_CMD --ckpt-dir "$ckpt_dir" --output "$result_file" \
        > "$SCRIPTS/eval_${name}.log" 2>&1
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        local acc
        acc=$(python3 -c "import json; d=json.load(open('$result_file')); print(f'{d[\"mmlu_acc\"]:.4f}')" 2>/dev/null || echo "?")
        log "DONE $name: acc=$acc"
    else
        log "FAILED $name (exit $exit_code)"
    fi
}

# ── PHASE 1: Wait for Variant B to finish (if running) ───────────────────────
log "=== PHASE 1: Check Variant B ==="
# Variant B is running as a separate Docker exec; wait for its result file
while [ ! -f "$SCRIPTS/mmlu_variantB_results.json" ]; do
    log "Waiting for Variant B eval to finish..."
    sleep 60
done
log "Variant B done"

# ── PHASE 2: Compressions (CPU-only, can overlap with GPU evals) ─────────────
log "=== PHASE 2: Compressions ==="
compress_if_needed "2b075b_zero_fixed_scale_weighted"
compress_if_needed "2b075b_zero_fixed_freq_sq"
compress_if_needed "3b1b_4free_exact"
compress_if_needed "3b1b_4free_weighted_abs"

# ── PHASE 3: Decompressions ───────────────────────────────────────────────────
log "=== PHASE 3: Decompressions ==="
decompress_if_needed "$SCRIPTS/compressed_2b075b_zero_fixed_scale_weighted" \
                     "$SCRIPTS/decompressed_D_scale_weighted"
decompress_if_needed "$SCRIPTS/compressed_2b075b_zero_fixed_freq_sq" \
                     "$SCRIPTS/decompressed_E_freq_sq"
decompress_if_needed "$SCRIPTS/compressed_3b1b_4free_exact" \
                     "$SCRIPTS/decompressed_F_3bit_exact"
decompress_if_needed "$SCRIPTS/compressed_3b1b_4free_weighted_abs" \
                     "$SCRIPTS/decompressed_G_3bit_weighted"

# ── PHASE 4: Evaluations (sequential, GPU) ───────────────────────────────────
log "=== PHASE 4: Evaluations ==="

# Already-decompressed variants
eval_if_needed "A_exact_mse"    "$SCRIPTS/decompressed_2b075b_zero_fixed_exact"    "$SCRIPTS/result_A_exact_mse.json"
eval_if_needed "B_weighted_abs" "$SCRIPTS/decompressed_2b075b_zero_fixed_weighted_abs" "$SCRIPTS/result_B_weighted_abs.json"
eval_if_needed "C_freq_sym"     "$SCRIPTS/decompressed_2b1b_freq_symmetric"        "$SCRIPTS/result_C_freq_sym.json"
eval_if_needed "2b1b_demo"      "$SCRIPTS/decompressed_2b1b_demo"                  "$SCRIPTS/result_2b1b_demo.json"

# New variants
eval_if_needed "D_scale_weighted" "$SCRIPTS/decompressed_D_scale_weighted" "$SCRIPTS/result_D_scale_weighted.json"
eval_if_needed "E_freq_sq"        "$SCRIPTS/decompressed_E_freq_sq"        "$SCRIPTS/result_E_freq_sq.json"
eval_if_needed "F_3bit_exact"     "$SCRIPTS/decompressed_F_3bit_exact"     "$SCRIPTS/result_F_3bit_exact.json"
eval_if_needed "G_3bit_weighted"  "$SCRIPTS/decompressed_G_3bit_weighted"  "$SCRIPTS/result_G_3bit_weighted.json"

# ── SUMMARY ──────────────────────────────────────────────────────────────────
log "=== RESULTS SUMMARY ==="
log "Baseline (NVFP4 professional_law): 59.78%"
for f in "$SCRIPTS"/result_*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .json)
    acc=$(python3 -c "
import json
try:
    d=json.load(open('$f'))
    acc=d.get('mmlu_acc', 0)
    print(f'{acc:.4f}')
except: print('?')
" 2>/dev/null)
    log "  $name: $acc"
done
log "=== ALL DONE ==="
