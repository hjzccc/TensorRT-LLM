#!/bin/bash
# Full pipeline: compress D/E/F/G, decompress, then evaluate all variants
# Designed to run sequentially and survive interruptions

set -e
SCRIPTS="/code/tensorrt_llm/scripts/nvfp4_compress"
CKPT="$SCRIPTS/nvfp4_checkpoint"
EVAL="python3 $SCRIPTS/run_mmlu_direct.py --offline --subject professional_law --batch-size 64 --max-batch-total-tokens 12000"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /code/tensorrt_llm

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $SCRIPTS/full_pipeline.log; }

run_eval() {
    local name=$1; local ckpt=$2; local output=$3; local logf=$4
    log "=== Evaluating $name ==="
    $EVAL --ckpt-dir $ckpt --output $output > $logf 2>&1
    local exit_code=$?
    echo "EXIT:$exit_code" >> $logf
    if [ $exit_code -eq 0 ]; then
        python3 -c "import json; d=json.load(open('$output')); log=open('$SCRIPTS/full_pipeline.log','a'); log.write(f'RESULT $name: MMLU={d[\"mmlu_acc\"]:.4f} ({d[\"mmlu_acc\"]*100:.2f}%)\n')" 2>/dev/null || true
        log "DONE $name"
    else
        log "FAILED $name (exit $exit_code)"
    fi
}

# ── PHASE 1: Compressions (CPU-only) ──────────────────────────────────────────
log "=== PHASE 1: Compressions ==="

if [ ! -f "$SCRIPTS/compressed_2b075b_zero_fixed_scale_weighted/compression_manifest.json" ]; then
    log "Compressing D (scale_weighted)..."
    python3 $SCRIPTS/compress_checkpoint.py --scheme 2b075b_zero_fixed_scale_weighted --input $CKPT \
        > $SCRIPTS/compress_D.log 2>&1
    log "D compressed"
fi

if [ ! -f "$SCRIPTS/compressed_2b075b_zero_fixed_freq_sq/compression_manifest.json" ]; then
    log "Compressing E (freq_sq)..."
    python3 $SCRIPTS/compress_checkpoint.py --scheme 2b075b_zero_fixed_freq_sq --input $CKPT \
        > $SCRIPTS/compress_E.log 2>&1
    log "E compressed"
fi

if [ ! -f "$SCRIPTS/compressed_3b1b_4free_exact/compression_manifest.json" ]; then
    log "Compressing F (3bit exact)..."
    python3 $SCRIPTS/compress_checkpoint.py --scheme 3b1b_4free_exact --input $CKPT \
        > $SCRIPTS/compress_F.log 2>&1
    log "F compressed"
fi

if [ ! -f "$SCRIPTS/compressed_3b1b_4free_weighted_abs/compression_manifest.json" ]; then
    log "Compressing G (3bit weighted_abs)..."
    python3 $SCRIPTS/compress_checkpoint.py --scheme 3b1b_4free_weighted_abs --input $CKPT \
        > $SCRIPTS/compress_G.log 2>&1
    log "G compressed"
fi

# ── PHASE 2: Decompressions ────────────────────────────────────────────────────
log "=== PHASE 2: Decompressions ==="

decompress_if_needed() {
    local src=$1; local dst=$2; local logf=$3
    if [ ! -f "$dst/config.json" ]; then
        log "Decompressing $src -> $dst..."
        python3 $SCRIPTS/decompress_checkpoint.py --input $src --output $dst > $logf 2>&1
        # Copy config files
        cp $SCRIPTS/decompressed_2b075b_zero_fixed_exact/config.json $dst/config.json
        cp $SCRIPTS/decompressed_2b075b_zero_fixed_exact/hf_quant_config.json $dst/hf_quant_config.json
        cp $SCRIPTS/decompressed_2b075b_zero_fixed_exact/model.safetensors.index.json $dst/model.safetensors.index.json
        log "Decompressed $dst"
    else
        log "Already decompressed: $dst"
    fi
}

decompress_if_needed \
    "$SCRIPTS/compressed_2b075b_zero_fixed_scale_weighted" \
    "$SCRIPTS/decompressed_D_scale_weighted" \
    "$SCRIPTS/decompress_D.log"

decompress_if_needed \
    "$SCRIPTS/compressed_2b075b_zero_fixed_freq_sq" \
    "$SCRIPTS/decompressed_E_freq_sq" \
    "$SCRIPTS/decompress_E.log"

decompress_if_needed \
    "$SCRIPTS/compressed_3b1b_4free_exact" \
    "$SCRIPTS/decompressed_F_3bit_exact" \
    "$SCRIPTS/decompress_F.log"

decompress_if_needed \
    "$SCRIPTS/compressed_3b1b_4free_weighted_abs" \
    "$SCRIPTS/decompressed_G_3bit_weighted" \
    "$SCRIPTS/decompress_G.log"

# ── PHASE 3: Evaluations (sequential, GPU) ────────────────────────────────────
log "=== PHASE 3: Evaluations ==="

# A: exact MSE (2.75 bits)
if [ ! -f "$SCRIPTS/result_A_exact_mse.json" ]; then
    run_eval "A_exact_mse" \
        "$SCRIPTS/decompressed_2b075b_zero_fixed_exact" \
        "$SCRIPTS/result_A_exact_mse.json" \
        "$SCRIPTS/eval_A_final.log"
fi

# B: weighted_abs MSE (2.75 bits)
if [ ! -f "$SCRIPTS/result_B_weighted_abs.json" ]; then
    run_eval "B_weighted_abs" \
        "$SCRIPTS/decompressed_2b075b_zero_fixed_weighted_abs" \
        "$SCRIPTS/result_B_weighted_abs.json" \
        "$SCRIPTS/eval_B_final.log"
fi

# C: freq_symmetric (2.0625 bits)
if [ ! -f "$SCRIPTS/result_C_freq_sym.json" ]; then
    run_eval "C_freq_sym" \
        "$SCRIPTS/decompressed_2b1b_freq_symmetric" \
        "$SCRIPTS/result_C_freq_sym.json" \
        "$SCRIPTS/eval_C_final.log"
fi

# 2b1b_demo (2.0625 bits)
if [ ! -f "$SCRIPTS/result_2b1b_demo.json" ]; then
    run_eval "2b1b_demo" \
        "$SCRIPTS/decompressed_2b1b_demo" \
        "$SCRIPTS/result_2b1b_demo.json" \
        "$SCRIPTS/eval_2b1b_demo_final.log"
fi

# D: scale_weighted (2.75 bits, BOF4-style)
if [ ! -f "$SCRIPTS/result_D_scale_weighted.json" ]; then
    run_eval "D_scale_weighted" \
        "$SCRIPTS/decompressed_D_scale_weighted" \
        "$SCRIPTS/result_D_scale_weighted.json" \
        "$SCRIPTS/eval_D_final.log"
fi

# E: freq_sq (2.75 bits)
if [ ! -f "$SCRIPTS/result_E_freq_sq.json" ]; then
    run_eval "E_freq_sq" \
        "$SCRIPTS/decompressed_E_freq_sq" \
        "$SCRIPTS/result_E_freq_sq.json" \
        "$SCRIPTS/eval_E_final.log"
fi

# F: 3bit exact (3.0 bits)
if [ ! -f "$SCRIPTS/result_F_3bit_exact.json" ]; then
    run_eval "F_3bit_exact" \
        "$SCRIPTS/decompressed_F_3bit_exact" \
        "$SCRIPTS/result_F_3bit_exact.json" \
        "$SCRIPTS/eval_F_final.log"
fi

# G: 3bit weighted_abs (3.0 bits)
if [ ! -f "$SCRIPTS/result_G_3bit_weighted.json" ]; then
    run_eval "G_3bit_weighted" \
        "$SCRIPTS/decompressed_G_3bit_weighted" \
        "$SCRIPTS/result_G_3bit_weighted.json" \
        "$SCRIPTS/eval_G_final.log"
fi

log "=== ALL DONE ==="
log ""
log "=== RESULTS SUMMARY ==="
log "Baseline (NVFP4 professional_law): 59.78%"
for f in $SCRIPTS/result_*.json; do
    name=$(basename $f .json)
    python3 -c "
import json, sys
try:
    d=json.load(open('$f'))
    acc=d.get('mmlu_acc', d.get('subjects',{}).get('professional_law',{}).get('acc',0))
    print(f'  $name: {acc:.4f} ({acc*100:.2f}%)')
except: pass
" 2>/dev/null || true
done
