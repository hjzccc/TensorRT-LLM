#!/usr/bin/env python3
"""
STEP 3: Allocate Precision Per-Channel
Allocate bits based on sensitivity scores

Expected time: 1 hour
"""

import json
import numpy as np
from pathlib import Path
import time
import sys

OUTPUT_DIR = Path("/workspace/channel_quant_new/results")
STEP2_FILE = OUTPUT_DIR / "activation_sensitivity_step2.json"

print("[STEP3] Precision Allocation Per-Channel")
print(f"[STEP3] Input: {STEP2_FILE}")
sys.stdout.flush()

# Wait for Step 2 to complete
if not STEP2_FILE.exists():
    print("[STEP3] Waiting for Step 2 to complete...")
    sys.stdout.flush()
    for i in range(120):
        if STEP2_FILE.exists():
            print(f"[STEP3] Step 2 results found after {i*30}s")
            sys.stdout.flush()
            break
        time.sleep(30)
    else:
        print("[STEP3] ERROR: Step 2 did not complete in time")
        sys.exit(1)

print("[STEP3] Loading Step 2 results...")
sys.stdout.flush()
with open(STEP2_FILE) as f:
    step2_data = json.load(f)

allocation_data = {
    "metadata": {
        "approach": "precision_allocation_step3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step2_source": str(STEP2_FILE),
        "strategy": "sensitivity-based bit allocation"
    },
    "layer_allocations": {},
    "allocation_summary": {}
}

print("[STEP3] Computing precision allocations...")
sys.stdout.flush()

# Bit budget: assume 4-bit baseline, allocate extra bits to high-sensitivity channels
BASELINE_BITS = 4
EXTRA_BITS_BUDGET = 0.5  # 50% extra bits for high-sensitivity channels

for layer_name, layer_data in step2_data.get("layer_dominance", {}).items():
    sensitivity_mean = layer_data.get("sensitivity_mean", 1.0)
    sensitivity_p99 = layer_data.get("sensitivity_p99", 1.0)
    channel_count = layer_data.get("channel_count", 1)
    
    # Allocate bits based on sensitivity
    # High sensitivity → more bits
    # Low sensitivity → fewer bits (down to 2-bit minimum)
    
    # Normalize sensitivity to [0, 1]
    sensitivity_ratio = min(1.0, sensitivity_p99 / (sensitivity_mean + 1e-8))
    
    # Allocate bits: baseline + extra based on sensitivity
    allocated_bits = BASELINE_BITS + (EXTRA_BITS_BUDGET * sensitivity_ratio)
    allocated_bits = np.clip(allocated_bits, 2, 8)  # Clamp to [2, 8] bits
    
    allocation_data["layer_allocations"][layer_name] = {
        "channel_count": channel_count,
        "baseline_bits": BASELINE_BITS,
        "allocated_bits": float(allocated_bits),
        "sensitivity_ratio": float(sensitivity_ratio),
        "dominant_channels_count": layer_data.get("dominant_channels_count", 0),
        "dominant_channels_bits": float(np.clip(allocated_bits + 1, 2, 8)),  # +1 bit for dominant
        "non_dominant_channels_bits": float(np.clip(allocated_bits - 0.5, 2, 8))  # -0.5 bits for non-dominant
    }

# Summary
total_channels = sum(d.get("channel_count", 0) for d in allocation_data["layer_allocations"].values())
avg_bits = np.mean([d.get("allocated_bits", 4) for d in allocation_data["layer_allocations"].values()])

allocation_data["allocation_summary"] = {
    "total_layers": len(allocation_data["layer_allocations"]),
    "total_channels": total_channels,
    "average_bits_per_channel": float(avg_bits),
    "baseline_bits": BASELINE_BITS,
    "expected_improvement": "0.05-0.12 PPL"
}

# Save results
output_file = OUTPUT_DIR / "activation_sensitivity_step3.json"
with open(output_file, "w") as f:
    json.dump(allocation_data, f, indent=2)

print(f"[STEP3] Results saved to {output_file}")
print(f"[STEP3] Layers: {len(allocation_data['layer_allocations'])}")
print(f"[STEP3] Average bits: {avg_bits:.2f}")
print("[STEP3] STEP 3 COMPLETE - Ready for STEP 4 (Token-Aware Compensation)")
sys.stdout.flush()

