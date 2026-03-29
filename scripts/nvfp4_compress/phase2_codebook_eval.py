"""
Phase 2: Per-Block-16 Codebook Compression Evaluation

Tests different fixed codebook strategies:
- 2.1: Identity (baseline validation)
- 2.2: 3-bit uniform
- 2.3: 3-bit adaptive
- 2.4: 2-bit uniform
- 2.5: 2-bit optimal

This script is designed to be run in the TRT-LLM docker container
with access to the Qwen3.5-35B-A3B model and WikiText-2 dataset.
"""

import torch
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
import time
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new" / "profiling"))

from fp4_utils import (
    unpack_fp4, repack_fp4, FP4Codebook,
    CODEBOOK_IDENTITY, CODEBOOK_3BIT_UNIFORM, CODEBOOK_3BIT_ADAPTIVE,
    CODEBOOK_2BIT_UNIFORM, CODEBOOK_2BIT_OPTIMAL
)
from eval_pipeline import apply_codebook_mapping, validate_identity_mapping


def create_test_plan() -> List[Dict]:
    """Create the Phase 2 test plan"""
    return [
        {
            "name": "2.1_identity_baseline",
            "codebook": CODEBOOK_IDENTITY,
            "description": "Identity mapping (all 16 codes) - baseline validation",
            "expected_ppl_delta": 0.0,
        },
        {
            "name": "2.2_3bit_uniform",
            "codebook": CODEBOOK_3BIT_UNIFORM,
            "description": "3-bit uniform {0,2,4,5,6,7,14,15}",
            "expected_ppl_delta": 0.017,
        },
        {
            "name": "2.3_3bit_adaptive",
            "codebook": CODEBOOK_3BIT_ADAPTIVE,
            "description": "3-bit adaptive {0,1,2,4,6,7,14,15}",
            "expected_ppl_delta": -0.009,
        },
        {
            "name": "2.4_2bit_uniform",
            "codebook": CODEBOOK_2BIT_UNIFORM,
            "description": "2-bit uniform {0,4,6,15}",
            "expected_ppl_delta": 0.6,
        },
        {
            "name": "2.5_2bit_optimal",
            "codebook": CODEBOOK_2BIT_OPTIMAL,
            "description": "2-bit optimal {0,2,6,15}",
            "expected_ppl_delta": 0.55,
        },
    ]


def analyze_codebook_coverage(codebook: FP4Codebook) -> Dict:
    """Analyze how well a codebook covers the FP4 space"""
    from fp4_utils import E2M1_LOOKUP
    
    coverage = {
        "codebook_codes": codebook.codebook_codes,
        "num_codes": codebook.num_codes,
        "bits_per_code": codebook.bits_per_code,
        "code_values": [E2M1_LOOKUP[c].item() for c in codebook.codebook_codes],
        "mapping": codebook.get_mapping(),
    }
    
    # Analyze mapping quality
    max_error = 0.0
    total_error = 0.0
    for code in range(16):
        nearest = codebook.code_to_nearest[code]
        error = abs(E2M1_LOOKUP[code].item() - E2M1_LOOKUP[nearest].item())
        max_error = max(max_error, error)
        total_error += error
    
    coverage["max_mapping_error"] = max_error
    coverage["avg_mapping_error"] = total_error / 16.0
    
    return coverage


def main():
    """Main evaluation loop"""
    print("=" * 80)
    print("NVFP4 Sub-Format Compression — Phase 2: Codebook Evaluation")
    print("=" * 80)
    
    # Create test plan
    test_plan = create_test_plan()
    
    print(f"\nTest Plan: {len(test_plan)} experiments")
    for i, test in enumerate(test_plan, 1):
        print(f"  {i}. {test['name']}: {test['description']}")
        print(f"     Expected PPL delta: {test['expected_ppl_delta']:+.4f}")
    
    # Analyze each codebook
    print("\n" + "=" * 80)
    print("Codebook Analysis")
    print("=" * 80)
    
    results = {}
    for test in test_plan:
        codebook = test["codebook"]
        analysis = analyze_codebook_coverage(codebook)
        
        print(f"\n{test['name']}:")
        print(f"  Codebook: {codebook}")
        print(f"  Codes: {analysis['codebook_codes']}")
        print(f"  Values: {[f'{v:6.2f}' for v in analysis['code_values']]}")
        print(f"  Max mapping error: {analysis['max_mapping_error']:.4f}")
        print(f"  Avg mapping error: {analysis['avg_mapping_error']:.4f}")
        
        results[test['name']] = analysis
    
    # Save analysis
    output_file = Path(__file__).parent / "phase2_codebook_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Analysis saved to {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("\nPhase 2 is ready to run. Next steps:")
    print("1. Run identity baseline (2.1) to validate pipeline")
    print("2. Run 3-bit experiments (2.2, 2.3) to validate compression")
    print("3. Run 2-bit experiments (2.4, 2.5) to understand limits")
    print("\nTo run full evaluation on Qwen3.5-35B-A3B:")
    print("  docker exec trtllm-phase12 bash -c 'cd /code/tensorrt_llm && python3 -u scripts/nvfp4_compress/phase2_full_eval.py'")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
