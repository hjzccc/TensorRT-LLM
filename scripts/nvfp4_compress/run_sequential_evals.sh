#!/bin/bash
# Sequential MMLU evaluations - one at a time to avoid OOM
# Run in trtllm-dual-tile container

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH

cd /code/tensorrt_llm/scripts/nvfp4_compress

SUBJECTS="abstract_algebra anatomy astronomy business_ethics"
BATCH=64
MAX_TOKENS=12000

echo "=== SEQUENTIAL EVAL PLAN ==="
echo "Subjects: $SUBJECTS"
echo "Batch: $BATCH, MaxTokens: $MAX_TOKENS"
echo ""

# Helper function
run_eval() {
    local name=$1
    local ckpt=$2
    local output=$3
    local log=$4
    
    echo ">>> Starting eval: $name"
    echo "    Checkpoint: $ckpt"
    echo "    Output: $output"
    
    # Run all subjects in one call (no --subject flag = all cached subjects)
    # But we want specific subjects, so run each separately and merge
    local all_correct=0
    local all_total=0
    local subjects_json="{}"
    
    for subj in $SUBJECTS; do
        local subj_out="${output%.json}_${subj}.json"
        python3 -u run_mmlu_direct.py \
            --offline \
            --subject "$subj" \
            --ckpt-dir "$ckpt" \
            --batch-size $BATCH \
            --max-batch-total-tokens $MAX_TOKENS \
            --output "$subj_out" \
            2>&1 | tee "${log%.log}_${subj}.log"
        echo "  Done: $subj"
    done
    
    # Merge results
    python3 -c "
import json, glob, sys
files = ['${output%.json}_' + s + '.json' for s in '$SUBJECTS'.split()]
merged = {'subjects': {}, 'config': {}}
total_correct = 0
total_total = 0
for f in files:
    try:
        d = json.load(open(f))
        for k, v in d.get('subjects', {}).items():
            merged['subjects'][k] = v
            total_correct += v['correct']
            total_total += v['total']
        merged['config'] = d.get('config', {})
    except Exception as e:
        print(f'Warning: {f}: {e}', file=sys.stderr)
merged['mmlu_acc'] = total_correct / total_total if total_total > 0 else 0
merged['total_correct'] = total_correct
merged['total_total'] = total_total
json.dump(merged, open('$output', 'w'), indent=2)
print(f'Merged: {total_correct}/{total_total} = {merged[\"mmlu_acc\"]:.4f}')
"
    echo ">>> Done: $name -> $output"
    echo ""
}

# 1. Baseline NVFP4
if [ ! -f "result_baseline_4subj.json" ]; then
    run_eval "BASELINE_NVFP4" "nvfp4_checkpoint" "result_baseline_4subj.json" "eval_baseline_4subj.log"
else
    echo "SKIP: result_baseline_4subj.json already exists"
fi

# 2. Variant B (weighted_abs, 2.75 bits)
if [ ! -f "result_B_weighted_abs_4subj.json" ]; then
    run_eval "B_WEIGHTED_ABS" "decompressed_2b075b_zero_fixed_weighted_abs" "result_B_weighted_abs_4subj.json" "eval_B_4subj.log"
else
    echo "SKIP: result_B_weighted_abs_4subj.json already exists"
fi

echo "=== ALL EVALS COMPLETE ==="
echo "Results:"
for f in result_baseline_4subj.json result_BD_exact_full.json result_B_weighted_abs_4subj.json; do
    if [ -f "$f" ]; then
        acc=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d.get(\"mmlu_acc\",0):.4f}')" 2>/dev/null)
        echo "  $f: $acc"
    fi
done
