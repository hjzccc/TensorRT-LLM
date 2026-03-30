#!/usr/bin/env python3
"""
Run MMLU evaluation on Variant B compressed checkpoint.
"""

import sys
import json
from pathlib import Path

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "nvfp4_compress"))

from lm_eval_nvfp4 import NVFP4LM
from lm_eval import evaluator as lm_eval_evaluator

# Configuration
CHECKPOINT_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_weighted_abs")
SUBJECTS = ["professional_law", "abstract_algebra", "anatomy", "astronomy", "business_ethics"]
BATCH_SIZE = 4
MAX_BATCH_TOTAL_TOKENS = 4096

def main():
    print("=" * 70)
    print("VARIANT B MMLU EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Subjects: {SUBJECTS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max batch tokens: {MAX_BATCH_TOTAL_TOKENS}")
    print()
    
    # Load model
    print("Loading model...")
    model = NVFP4LM(
        ckpt_dir=CHECKPOINT_DIR,
        batch_size=BATCH_SIZE,
        max_batch_total_tokens=MAX_BATCH_TOTAL_TOKENS,
    )
    
    print(f"Model loaded: {model.get_model_info()}")
    print()
    
    # Run evaluation
    print("Running MMLU evaluation...")
    results = lm_eval_evaluator.simple_evaluate(
        model=model,
        tasks=["mmlu"],
        num_fewshot=0,
        batch_size=BATCH_SIZE,
        max_batch_total_tokens=MAX_BATCH_TOTAL_TOKENS,
        limit=None,
        write_out=False,
        log_samples=False,
    )
    
    # Save results
    output_file = Path(__file__).parent / "variant_b_mmlu_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print(f"Overall accuracy: {results.get('results', {}).get('mmlu', {}).get('acc', 'N/A')}")

if __name__ == "__main__":
    main()
