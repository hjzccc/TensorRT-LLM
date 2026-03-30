#!/bin/bash
# Monitor MMLU evaluation progress

LOG_FILE="phase4_3_mmlu_eval.log"
RESULTS_FILE="/tmp/lm_eval_mmlu_results.json"

echo "Monitoring MMLU evaluation..."
echo "Log file: $LOG_FILE"
echo "Results file: $RESULTS_FILE"
echo ""

while true; do
    echo "=== $(date) ==="
    
    # Check if process is still running
    if pgrep -f "phase4_3_full_accuracy_validation.py" > /dev/null; then
        echo "✓ Process running"
    else
        echo "✗ Process NOT running"
        break
    fi
    
    # Show last few lines of log
    echo ""
    echo "Last 10 lines of log:"
    tail -10 "$LOG_FILE"
    
    # Check if results file exists
    echo ""
    if [ -f "$RESULTS_FILE" ]; then
        echo "✓ Results file exists"
        echo "  Size: $(du -h $RESULTS_FILE | cut -f1)"
    else
        echo "✗ Results file not yet created"
    fi
    
    echo ""
    echo "Waiting 5 minutes before next check..."
    sleep 300
done

echo ""
echo "=== Evaluation Complete ==="
echo "Final log:"
tail -30 "$LOG_FILE"
