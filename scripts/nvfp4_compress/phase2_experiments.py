#!/usr/bin/env python3
"""Phase 2: Run all codebook experiments in sequence."""
import subprocess
import json
import sys
from pathlib import Path

CODEBOOKS = [
    '3bit_uniform',
    '3bit_dense',
    '3bit_truncate',
    '2bit_opt1',
    '2bit_opt3',
    '2bit_uniform',
]

RESULTS_DIR = Path("/code/tensorrt_llm/scripts/nvfp4_compress/phase2_results")
RESULTS_DIR.mkdir(exist_ok=True)

results = {}

for codebook in CODEBOOKS:
    print(f"\n{'='*70}")
    print(f"PHASE 2: Testing {codebook}")
    print(f"{'='*70}")
    
    cmd = [
        "python", "/code/tensorrt_llm/scripts/nvfp4_compress/sub_fp4_compress.py",
        "--codebook", codebook
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    # Load result JSON
    result_file = Path(f"/code/tensorrt_llm/scripts/nvfp4_compress/result_{codebook}.json")
    if result_file.exists():
        with open(result_file) as f:
            results[codebook] = json.load(f)
        print(f"✓ {codebook}: {results[codebook]['ppl']:.4f} PPL")
    else:
        print(f"✗ {codebook}: No result file")

# Save summary
summary_file = RESULTS_DIR / "summary.json"
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print(f"PHASE 2 SUMMARY")
print(f"{'='*70}")
for cb, res in sorted(results.items(), key=lambda x: x[1]['ppl']):
    print(f"{cb:20s}: PPL={res['ppl']:.4f}, bits={res['effective_bpe']:.2f}")

print(f"\nResults saved to {summary_file}")
