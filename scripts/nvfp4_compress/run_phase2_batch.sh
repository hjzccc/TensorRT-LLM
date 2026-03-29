#!/bin/bash
# Run all Phase 2 experiments sequentially

CODEBOOKS=(
    "3bit_uniform"
    "3bit_dense"
    "3bit_truncate"
    "2bit_opt1"
    "2bit_opt3"
    "2bit_uniform"
)

RESULTS_DIR="/code/tensorrt_llm/scripts/nvfp4_compress/phase2_results"
mkdir -p "$RESULTS_DIR"

echo "Starting Phase 2 batch experiments..."
echo "========================================"

for cb in "${CODEBOOKS[@]}"; do
    echo ""
    echo "Running: $cb"
    echo "Time: $(date)"
    
    docker exec trtllm-dual-tile bash -c "cd /code/tensorrt_llm && python scripts/nvfp4_compress/sub_fp4_compress.py --codebook $cb" 2>&1 | tee "$RESULTS_DIR/${cb}.log"
    
    # Check if result file was created
    if [ -f "/tmp/result_${cb}.json" ]; then
        cp "/tmp/result_${cb}.json" "$RESULTS_DIR/"
        echo "✓ Result saved"
    else
        echo "✗ No result file for $cb"
    fi
done

echo ""
echo "========================================"
echo "Phase 2 batch complete!"
echo "Results in: $RESULTS_DIR"
