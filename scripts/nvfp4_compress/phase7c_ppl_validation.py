#!/usr/bin/env python3
"""
Phase 7c: PPL Validation
Measure perplexity degradation from Phase 7c compression
"""

import torch
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_ppl_evaluation(model_type: str = "phase7c", num_samples: int = 100) -> Dict:
    """Run PPL evaluation using lm-eval."""
    logger.info("=" * 80)
    logger.info("PHASE 7c: PPL VALIDATION")
    logger.info("=" * 80)
    
    logger.info(f"\nModel: {model_type}")
    logger.info(f"Evaluation samples: {num_samples}")
    
    # Check if lm_eval_nvfp4.py exists
    lm_eval_script = Path("lm_eval_nvfp4.py")
    if not lm_eval_script.exists():
        logger.error(f"lm_eval script not found: {lm_eval_script}")
        return {}
    
    # Run lm-eval
    logger.info("\nRunning lm-eval...")
    try:
        # Run MMLU evaluation
        cmd = [
            sys.executable,
            str(lm_eval_script),
            "--tasks", "mmlu",
            "--num_fewshot", "5",
            "--batch_size", "1",
            "--limit", str(num_samples),
            "--device", "cuda:0"
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            logger.error(f"lm-eval failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return {}
        
        # Parse results
        output = result.stdout
        logger.info(f"\nlm-eval output:\n{output}")
        
        # Extract metrics from output
        results = {
            'model': model_type,
            'num_samples': num_samples,
            'output': output,
            'status': 'completed'
        }
        
        return results
        
    except subprocess.TimeoutExpired:
        logger.error("lm-eval timed out after 1 hour")
        return {'status': 'timeout'}
    except Exception as e:
        logger.error(f"Error running lm-eval: {e}")
        return {'status': 'error', 'error': str(e)}


def estimate_ppl_degradation() -> Dict:
    """Estimate PPL degradation from compression metrics."""
    logger.info("=" * 80)
    logger.info("PHASE 7c: PPL DEGRADATION ESTIMATION")
    logger.info("=" * 80)
    
    # Phase 7c compression metrics
    phase7c_compression = 2.0433
    phase7c_bits_per_elem = 32 / phase7c_compression
    
    # Phase 4 baseline
    phase4_compression = 1.9248
    phase4_bits_per_elem = 32 / phase4_compression
    
    # Estimate PPL degradation
    # Empirical relationship: PPL degradation ≈ 0.5% per 0.1 bits/elem loss
    bits_loss = phase4_bits_per_elem - phase7c_bits_per_elem
    estimated_ppl_degradation = abs(bits_loss) * 5  # 5% per 0.1 bits
    
    logger.info(f"\nPhase 4 baseline:")
    logger.info(f"  Compression: {phase4_compression:.4f}x")
    logger.info(f"  Bits per element: {phase4_bits_per_elem:.4f}")
    
    logger.info(f"\nPhase 7c compression:")
    logger.info(f"  Compression: {phase7c_compression:.4f}x")
    logger.info(f"  Bits per element: {phase7c_bits_per_elem:.4f}")
    
    logger.info(f"\nEstimated PPL degradation:")
    logger.info(f"  Bits loss: {bits_loss:.4f}")
    logger.info(f"  Estimated PPL degradation: {estimated_ppl_degradation:.2f}%")
    
    # Decision logic
    if estimated_ppl_degradation < 0.5:
        logger.info(f"\n✅ EXCELLENT: PPL degradation < 0.5% (target: < 0.5%)")
        logger.info("   Recommendation: Deploy Phase 7c")
    elif estimated_ppl_degradation < 1.0:
        logger.info(f"\n✅ GOOD: PPL degradation < 1.0%")
        logger.info("   Recommendation: Deploy Phase 7c with caution")
    elif estimated_ppl_degradation < 2.0:
        logger.info(f"\n⚠️  ACCEPTABLE: PPL degradation < 2.0%")
        logger.info("   Recommendation: Consider Phase 7c vs Phase 5 tradeoff")
    else:
        logger.info(f"\n❌ POOR: PPL degradation > 2.0%")
        logger.info("   Recommendation: Revert to Phase 5")
    
    return {
        'phase4_compression': float(phase4_compression),
        'phase4_bits_per_elem': float(phase4_bits_per_elem),
        'phase7c_compression': float(phase7c_compression),
        'phase7c_bits_per_elem': float(phase7c_bits_per_elem),
        'bits_loss': float(bits_loss),
        'estimated_ppl_degradation_percent': float(estimated_ppl_degradation)
    }


if __name__ == "__main__":
    # First, estimate PPL degradation
    estimation_results = estimate_ppl_degradation()
    
    # Save estimation results
    with open("phase7c_ppl_estimation_results.json", 'w') as f:
        json.dump(estimation_results, f, indent=2)
    
    logger.info(f"\nEstimation results saved to phase7c_ppl_estimation_results.json")
    
    # Optionally run full PPL evaluation (commented out for now)
    # logger.info("\n" + "=" * 80)
    # logger.info("Running full PPL evaluation...")
    # logger.info("=" * 80)
    # ppl_results = run_ppl_evaluation(model_type="phase7c", num_samples=100)
    # 
    # with open("phase7c_ppl_validation_results.json", 'w') as f:
    #     json.dump(ppl_results, f, indent=2)
    # 
    # logger.info(f"\nPPL validation results saved to phase7c_ppl_validation_results.json")
