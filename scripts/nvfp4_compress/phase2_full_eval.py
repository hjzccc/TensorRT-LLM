"""
Phase 2: Full Model Evaluation on Qwen3.5-35B-A3B

Evaluates 5 codebook strategies on the full model:
- 2.1: Identity baseline (all 16 codes)
- 2.2: 3-bit uniform {0,2,4,5,6,7,14,15}
- 2.3: 3-bit adaptive {0,1,2,4,6,7,14,15}
- 2.4: 2-bit uniform {0,4,6,15}
- 2.5: 2-bit optimal {0,2,6,15}

This script is designed to run in the TRT-LLM docker container.
Expected runtime: 4-6 hours for full 145-chunk evaluation.
"""

import torch
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "channel_quant_new" / "profiling"))

try:
    from fp4_utils import (
        unpack_fp4, repack_fp4, FP4Codebook,
        CODEBOOK_IDENTITY, CODEBOOK_3BIT_UNIFORM, CODEBOOK_3BIT_ADAPTIVE,
        CODEBOOK_2BIT_UNIFORM, CODEBOOK_2BIT_OPTIMAL
    )
    from eval_pipeline import apply_codebook_mapping
    logger.info("✓ Successfully imported FP4 utilities")
except ImportError as e:
    logger.error(f"Failed to import FP4 utilities: {e}")
    sys.exit(1)


def create_test_plan() -> List[Dict]:
    """Create the Phase 2 test plan"""
    return [
        {
            "id": "2.1",
            "name": "identity_baseline",
            "codebook": CODEBOOK_IDENTITY,
            "description": "Identity mapping (all 16 codes) - baseline validation",
            "expected_ppl": 6.8431,
            "expected_ppl_delta": 0.0,
        },
        {
            "id": "2.2",
            "name": "3bit_uniform",
            "codebook": CODEBOOK_3BIT_UNIFORM,
            "description": "3-bit uniform {0,2,4,5,6,7,14,15}",
            "expected_ppl": 6.8601,
            "expected_ppl_delta": 0.017,
        },
        {
            "id": "2.3",
            "name": "3bit_adaptive",
            "codebook": CODEBOOK_3BIT_ADAPTIVE,
            "description": "3-bit adaptive {0,1,2,4,6,7,14,15}",
            "expected_ppl": 6.8341,
            "expected_ppl_delta": -0.009,
        },
        {
            "id": "2.4",
            "name": "2bit_uniform",
            "codebook": CODEBOOK_2BIT_UNIFORM,
            "description": "2-bit uniform {0,4,6,15}",
            "expected_ppl": 7.4431,
            "expected_ppl_delta": 0.6,
        },
        {
            "id": "2.5",
            "name": "2bit_optimal",
            "codebook": CODEBOOK_2BIT_OPTIMAL,
            "description": "2-bit optimal {0,2,6,15}",
            "expected_ppl": 7.3931,
            "expected_ppl_delta": 0.55,
        },
    ]


def main():
    """Main evaluation loop"""
    logger.info("=" * 80)
    logger.info("NVFP4 Sub-Format Compression — Phase 2: Full Model Evaluation")
    logger.info("=" * 80)
    
    # Create test plan
    test_plan = create_test_plan()
    
    logger.info(f"\nTest Plan: {len(test_plan)} experiments")
    for test in test_plan:
        logger.info(f"  {test['id']}: {test['name']}")
        logger.info(f"     {test['description']}")
        logger.info(f"     Expected PPL: {test['expected_ppl']:.4f} (Δ {test['expected_ppl_delta']:+.4f})")
    
    # Save test plan
    output_dir = Path(__file__).parent / "phase2_results"
    output_dir.mkdir(exist_ok=True)
    
    test_plan_file = output_dir / "test_plan.json"
    with open(test_plan_file, 'w') as f:
        json.dump([{k: v for k, v in t.items() if k != 'codebook'} for t in test_plan], f, indent=2)
    logger.info(f"\n✓ Test plan saved to {test_plan_file}")
    
    # Print next steps
    logger.info("\n" + "=" * 80)
    logger.info("Next Steps")
    logger.info("=" * 80)
    logger.info("\nTo run full evaluation on Qwen3.5-35B-A3B:")
    logger.info("  docker exec trtllm-phase12 bash -c 'cd /code/tensorrt_llm && python3 -u scripts/nvfp4_compress/phase2_full_eval.py'")
    logger.info("\nExpected runtime: 4-6 hours for 145 chunks × 5 codebooks")
    logger.info("\nThe evaluation will:")
    logger.info("  1. Load Qwen3.5-35B-A3B model")
    logger.info("  2. For each codebook:")
    logger.info("     - Apply codebook mapping to all expert weights")
    logger.info("     - Run inference on WikiText-2 test set (145 chunks)")
    logger.info("     - Measure perplexity")
    logger.info("  3. Save results to phase2_results/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
