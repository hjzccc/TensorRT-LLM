"""
Phase 2 Corrected: Full Model Evaluation with Proper Codebook Mapping

This version correctly applies codebook mapping AFTER FP4 quantization,
not before weight loading.

Key insight: FP4 quantization happens in the forward pass via
torch.ops.trtllm.fp4_quantize(), so we need to intercept there.
"""

import torch
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional
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
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")

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

try:
    # import tensorrt_llm._torch.auto_deploy.custom_ops  # Not needed, exact_docker_eval handles it
    import exact_docker_eval as ee
    from spike1_ground_truth import (
        build_text_config,
        load_root_config,
    )
    logger.info("✓ Successfully imported TRT-LLM evaluation utilities")
except ImportError as e:
    logger.error(f"Failed to import TRT-LLM utilities: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"

# Baseline PPL values
BF16_BASELINE_PPL = 6.5896
NVFP4_BASELINE_PPL = 6.8431


# Global codebook for monkey-patching
_CURRENT_CODEBOOK: Optional[FP4Codebook] = None
_ORIGINAL_FP4_QUANTIZE = None


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


def apply_codebook_to_fp4_codes(weight_fp4: torch.Tensor, codebook: FP4Codebook) -> torch.Tensor:
    """
    Apply codebook mapping to FP4 packed codes.
    
    This is called AFTER fp4_quantize(), so weight_fp4 is already packed FP4 bytes.
    """
    if codebook is None or codebook.name == "identity":
        return weight_fp4
    
    # Unpack FP4 codes
    codes = unpack_fp4(weight_fp4)
    
    # Encode to codebook indices
    indices = codebook.encode(codes)
    
    # Decode back to FP4 codes (constrained to codebook)
    codes_transformed = codebook.decode(indices)
    
    # Repack
    weight_fp4_transformed = repack_fp4(codes_transformed)
    
    return weight_fp4_transformed


def patched_fp4_quantize(weight, s_w, scaling_vector_size, flag):
    """Patched version of fp4_quantize that applies codebook mapping"""
    global _ORIGINAL_FP4_QUANTIZE, _CURRENT_CODEBOOK
    
    # Call original quantization
    weight_fp4, weight_scale = _ORIGINAL_FP4_QUANTIZE(weight, s_w, scaling_vector_size, flag)
    
    # Apply codebook mapping if codebook is set
    if _CURRENT_CODEBOOK is not None:
        weight_fp4 = apply_codebook_to_fp4_codes(weight_fp4, _CURRENT_CODEBOOK)
    
    return weight_fp4, weight_scale


def evaluate_codebook(
    test: Dict,
    snapshot_dir: Path,
    weight_map: Dict,
    model_config,
    eval_ids,
    num_samples,
    seqlen,
    device,
    dtype,
    output_dir: Path,
) -> Dict:
    """
    Evaluate a single codebook strategy.
    
    Strategy: Monkey-patch torch.ops.trtllm.fp4_quantize to apply codebook mapping.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {test['id']}: {test['name']}")
    logger.info(f"{'='*80}")
    logger.info(f"Description: {test['description']}")
    logger.info(f"Expected PPL: {test['expected_ppl']:.4f} (Δ {test['expected_ppl_delta']:+.4f})")
    
    start_time = time.time()
    
    try:
        global _ORIGINAL_FP4_QUANTIZE, _CURRENT_CODEBOOK
        
        # Set global codebook for monkey-patching
        _CURRENT_CODEBOOK = test['codebook']
        
        # Save original and apply monkey-patch
        _ORIGINAL_FP4_QUANTIZE = torch.ops.trtllm.fp4_quantize
        torch.ops.trtllm.fp4_quantize = patched_fp4_quantize
        
        # Evaluate PPL using exact_docker_eval
        ppl = ee.evaluate_ppl(
            eval_ids=eval_ids,
            nsamples=num_samples,
            seqlen=seqlen,
            config=model_config,
            weight_map=weight_map,
            snapshot_dir=snapshot_dir,
            device=device,
            dtype=dtype,
            run_config=ee.EvalConfig(
                label=test['name'],
                mode="nvfp4",
                quant_scope="moe",
            ),
            layer_batch_size=1,
        )
        
        # Restore original function
        torch.ops.trtllm.fp4_quantize = _ORIGINAL_FP4_QUANTIZE
        _CURRENT_CODEBOOK = None
        
        elapsed = time.time() - start_time
        
        result = {
            "id": test['id'],
            "name": test['name'],
            "description": test['description'],
            "codebook_name": test['codebook'].name,
            "num_codes": test['codebook'].num_codes,
            "bits_per_code": test['codebook'].bits_per_code,
            "expected_ppl": test['expected_ppl'],
            "expected_ppl_delta": test['expected_ppl_delta'],
            "measured_ppl": float(ppl),
            "measured_ppl_delta": float(ppl - NVFP4_BASELINE_PPL),
            "ppl_error": float(abs(ppl - test['expected_ppl'])),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"✓ PPL: {ppl:.4f} (Δ {ppl - NVFP4_BASELINE_PPL:+.4f})")
        logger.info(f"  Expected: {test['expected_ppl']:.4f} (error: {result['ppl_error']:.4f})")
        logger.info(f"  Elapsed: {elapsed:.1f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Restore original function
        if _ORIGINAL_FP4_QUANTIZE is not None:
            torch.ops.trtllm.fp4_quantize = _ORIGINAL_FP4_QUANTIZE
        _CURRENT_CODEBOOK = None
        
        return {
            "id": test['id'],
            "name": test['name'],
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """Main evaluation loop"""
    logger.info("=" * 80)
    logger.info("NVFP4 Sub-Format Compression — Phase 2: Full Model Evaluation (Corrected)")
    logger.info("=" * 80)
    logger.info("Note: Codebook mapping applied AFTER FP4 quantization (correct approach)")
    
    # Create test plan
    test_plan = create_test_plan()
    
    logger.info(f"\nTest Plan: {len(test_plan)} experiments")
    for test in test_plan:
        logger.info(f"  {test['id']}: {test['name']}")
        logger.info(f"     {test['description']}")
        logger.info(f"     Expected PPL: {test['expected_ppl']:.4f} (Δ {test['expected_ppl_delta']:+.4f})")
    
    # Setup output directory
    output_dir = Path(__file__).parent / "phase2_results_corrected"
    output_dir.mkdir(exist_ok=True)
    
    # Save test plan
    test_plan_file = output_dir / "test_plan.json"
    with open(test_plan_file, 'w') as f:
        json.dump([{k: v for k, v in t.items() if k != 'codebook'} for t in test_plan], f, indent=2)
    logger.info(f"\n✓ Test plan saved to {test_plan_file}")
    
    # Load model and weights
    logger.info("\n" + "=" * 80)
    logger.info("Loading model and weights...")
    logger.info("=" * 80)
    
    try:
        # Load root config
        snapshot_dir, root_config, weight_map = load_root_config(MODEL_ID)
        logger.info(f"✓ Loaded root config for {MODEL_ID}")
        logger.info(f"  Snapshot dir: {snapshot_dir}")
        logger.info(f"  Weight map entries: {len(weight_map)}")
        
        # Build model config
        model_config = build_text_config(root_config)
        logger.info(f"✓ Built model config")
        
        # Load evaluation data
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        eval_ids, max_samples, seqlen = ee.load_eval_data(tokenizer, 2048)
        num_samples = min(145, max_samples)
        logger.info(f"✓ Loaded evaluation data: {num_samples} chunks, seqlen={seqlen}")
        
        # Setup device and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16
        logger.info(f"✓ Device: {device}, dtype: {dtype}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run evaluations
    logger.info("\n" + "=" * 80)
    logger.info("Running evaluations...")
    logger.info("=" * 80)
    
    results = []
    for test in test_plan:
        result = evaluate_codebook(
            test, snapshot_dir, weight_map, model_config,
            eval_ids, num_samples, seqlen, device, dtype, output_dir
        )
        results.append(result)
        
        # Save intermediate results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Saved intermediate results to {results_file}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    for result in results:
        if "error" in result:
            logger.info(f"  {result['id']}: {result['name']} — ERROR: {result['error']}")
        else:
            logger.info(f"  {result['id']}: {result['name']}")
            logger.info(f"     Measured PPL: {result['measured_ppl']:.4f} (Δ {result['measured_ppl_delta']:+.4f})")
            logger.info(f"     Expected PPL: {result['expected_ppl']:.4f} (error: {result['ppl_error']:.4f})")
    
    # Save final results
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\n✓ Final results saved to {results_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
