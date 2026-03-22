#!/usr/bin/env python3
"""
Master Orchestration Script: 5-Step Activation Sensitivity Pipeline
Runs all 5 steps sequentially with monitoring and progress tracking
"""

import subprocess
import time
import json
from pathlib import Path
import sys

OUTPUT_DIR = Path("/workspace/channel_quant_new/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS = [
    ("activation_sensitivity_step1.py", "Compute True Activation Sensitivity", 3600),
    ("activation_sensitivity_step2.py", "Identify Channel Dominance", 1800),
    ("activation_sensitivity_step3.py", "Allocate Precision Per-Channel", 3600),
    ("activation_sensitivity_step4.py", "Implement Token-Aware Compensation", 5400),
    ("activation_sensitivity_step5.py", "Preserve Attention Structure", 3600)
]

pipeline_log = {
    "metadata": {
        "pipeline": "5-Step Activation Sensitivity",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "expected_total_time_hours": 5,
        "expected_improvement_ppl": "0.05-0.12",
        "target_ppl": "< 6.45"
    },
    "steps": {}
}

print("=" * 60)
print("5-STEP ACTIVATION SENSITIVITY PIPELINE")
print("=" * 60)
print(f"Start time: {pipeline_log['metadata']['start_time']}")
print(f"Expected duration: 5 hours")
print(f"Expected improvement: 0.05-0.12 PPL")
print(f"Target: < 6.45 PPL")
print("=" * 60)
sys.stdout.flush()

total_start = time.time()

for step_num, (script_name, description, timeout) in enumerate(STEPS, 1):
    print(f"\n[PIPELINE] STEP {step_num}/5: {description}")
    print(f"[PIPELINE] Script: {script_name}")
    print(f"[PIPELINE] Timeout: {timeout}s ({timeout/60:.0f} minutes)")
    sys.stdout.flush()
    
    step_start = time.time()
    
    try:
        # Run the step script
        result = subprocess.run(
            ["python3", script_name],
            cwd="/workspace/channel_quant_new",
            timeout=timeout,
            capture_output=True,
            text=True
        )
        
        step_elapsed = time.time() - step_start
        
        pipeline_log["steps"][f"step_{step_num}"] = {
            "name": description,
            "script": script_name,
            "status": "COMPLETED" if result.returncode == 0 else "FAILED",
            "return_code": result.returncode,
            "elapsed_seconds": step_elapsed,
            "stdout_lines": len(result.stdout.split('\n')),
            "stderr_lines": len(result.stderr.split('\n'))
        }
        
        if result.returncode == 0:
            print(f"[PIPELINE] ✓ STEP {step_num} COMPLETED in {step_elapsed:.1f}s")
            # Print last few lines of output
            output_lines = result.stdout.split('\n')
            for line in output_lines[-5:]:
                if line.strip():
                    print(f"[PIPELINE]   {line}")
        else:
            print(f"[PIPELINE] ✗ STEP {step_num} FAILED (return code: {result.returncode})")
            print(f"[PIPELINE] Error output:")
            for line in result.stderr.split('\n')[-10:]:
                if line.strip():
                    print(f"[PIPELINE]   {line}")
        
        sys.stdout.flush()
        
    except subprocess.TimeoutExpired:
        print(f"[PIPELINE] ✗ STEP {step_num} TIMEOUT after {timeout}s")
        pipeline_log["steps"][f"step_{step_num}"] = {
            "name": description,
            "script": script_name,
            "status": "TIMEOUT",
            "timeout_seconds": timeout
        }
        sys.stdout.flush()
        break
    except Exception as e:
        print(f"[PIPELINE] ✗ STEP {step_num} ERROR: {e}")
        pipeline_log["steps"][f"step_{step_num}"] = {
            "name": description,
            "script": script_name,
            "status": "ERROR",
            "error": str(e)
        }
        sys.stdout.flush()
        break

total_elapsed = time.time() - total_start
pipeline_log["metadata"]["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
pipeline_log["metadata"]["total_elapsed_seconds"] = total_elapsed

# Save pipeline log
log_file = OUTPUT_DIR / "activation_sensitivity_pipeline.json"
with open(log_file, "w") as f:
    json.dump(pipeline_log, f, indent=2)

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60)
print(f"Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f} hours)")
print(f"Completed steps: {sum(1 for s in pipeline_log['steps'].values() if s.get('status') == 'COMPLETED')}/5")
print(f"Log saved to: {log_file}")
print("=" * 60)
sys.stdout.flush()

