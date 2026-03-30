#!/bin/bash
# Monitor compression progress every 5 minutes

OUTPUT_DIR="/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_weighted_abs_rerun2"
BASELINE_SIZE=$(du -sb /home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint 2>/dev/null | awk '{print $1}')
BASELINE_GB=$((BASELINE_SIZE / 1024 / 1024 / 1024))

echo "Starting compression monitor..."
echo "Baseline: ${BASELINE_GB}GB"
echo "Target: 733 files"
echo ""

while ps aux | grep -q "compress_checkpoint.py.*weighted_abs_rerun2"; do
    COUNT=$(find "$OUTPUT_DIR" -name "model-*.safetensors" -type f 2>/dev/null | wc -l)
    SIZE=$(du -sb "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
    SIZE_GB=$((SIZE / 1024 / 1024 / 1024))
    
    if [ $SIZE_GB -gt 0 ]; then
        RATIO=$(echo "scale=2; $BASELINE_GB / $SIZE_GB" | bc)
    else
        RATIO="0.00"
    fi
    
    PERCENT=$((COUNT * 100 / 733))
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$TIMESTAMP] $COUNT/733 ($PERCENT%) | ${SIZE_GB}GB | Ratio: ${RATIO}x"
    
    sleep 300  # Check every 5 minutes
done

echo "✅ Compression complete!"
FINAL_COUNT=$(find "$OUTPUT_DIR" -name "model-*.safetensors" -type f 2>/dev/null | wc -l)
FINAL_SIZE=$(du -sb "$OUTPUT_DIR" 2>/dev/null | awk '{print $1}')
FINAL_GB=$((FINAL_SIZE / 1024 / 1024 / 1024))
FINAL_RATIO=$(echo "scale=2; $BASELINE_GB / $FINAL_GB" | bc)

echo "Final: $FINAL_COUNT/733 | ${FINAL_GB}GB | Ratio: ${FINAL_RATIO}x"
