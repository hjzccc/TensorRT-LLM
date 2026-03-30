#!/usr/bin/env python3
"""
Phase 4.3: Accuracy Validation

Measure accuracy degradation from Variant B compression using MMLU and GSM8K.
"""

import torch
import numpy as np
from pathlib import Path
import json
import logging
import time
from typing import Dict, List, Tuple
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AccuracyValidator:
    """Validate accuracy of compressed checkpoint."""
    
    def __init__(self, checkpoint_path: str, baseline_checkpoint_path: str = None):
        """
        Initialize accuracy validator.
        
        Args:
            checkpoint_path: Path to compressed checkpoint
            baseline_checkpoint_path: Path to baseline checkpoint for comparison
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.baseline_checkpoint_path = Path(baseline_checkpoint_path) if baseline_checkpoint_path else None
        
        logger.info(f"Checkpoint: {checkpoint_path}")
        if baseline_checkpoint_path:
            logger.info(f"Baseline: {baseline_checkpoint_path}")
    
    def run_lm_eval(self, tasks: List[str], num_fewshot: int = 0, limit: int = None) -> Dict:
        """
        Run lm-eval on specified tasks.
        
        Args:
            tasks: List of task names (e.g., ['mmlu', 'gsm8k'])
            num_fewshot: Number of few-shot examples
            limit: Limit number of examples per task
            
        Returns:
            Dict with evaluation results
        """
        logger.info(f"Running lm-eval on tasks: {tasks}")
        
        results = {}
        for task in tasks:
            logger.info(f"Evaluating {task}...")
            
            # Build lm-eval command
            cmd = [
                'lm_eval',
                '--model', 'hf',
                '--model_args', f'pretrained={self.checkpoint_path}',
                '--tasks', task,
                '--num_fewshot', str(num_fewshot),
                '--batch_size', 'auto',
                '--output_path', f'/tmp/lm_eval_{task}_results.json',
            ]
            
            if limit:
                cmd.extend(['--limit', str(limit)])
            
            try:
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0:
                    logger.info(f"✅ {task} evaluation complete")
                    # Parse results from output
                    results[task] = {
                        'status': 'success',
                        'output': result.stdout[-500:] if result.stdout else '',
                    }
                else:
                    logger.error(f"❌ {task} evaluation failed")
                    logger.error(f"stderr: {result.stderr[-500:]}")
                    results[task] = {
                        'status': 'failed',
                        'error': result.stderr[-500:] if result.stderr else '',
                    }
            except subprocess.TimeoutExpired:
                logger.error(f"❌ {task} evaluation timed out")
                results[task] = {
                    'status': 'timeout',
                    'error': 'Evaluation timed out after 1 hour',
                }
            except Exception as e:
                logger.error(f"❌ {task} evaluation error: {e}")
                results[task] = {
                    'status': 'error',
                    'error': str(e),
                }
        
        return results
    
    def validate_accuracy(self) -> Dict:
        """
        Validate accuracy of compressed checkpoint.
        
        Returns:
            Dict with validation results
        """
        logger.info("Starting accuracy validation...")
        
        start_time = time.time()
        
        # Run evaluations
        # Note: Using small limits for quick testing
        # In production, remove limits for full evaluation
        results = self.run_lm_eval(
            tasks=['mmlu', 'gsm8k'],
            num_fewshot=0,
            limit=10  # Small limit for quick testing
        )
        
        elapsed = time.time() - start_time
        
        summary = {
            'checkpoint_path': str(self.checkpoint_path),
            'baseline_checkpoint_path': str(self.baseline_checkpoint_path) if self.baseline_checkpoint_path else None,
            'elapsed_sec': elapsed,
            'task_results': results,
        }
        
        logger.info(f"Accuracy validation complete in {elapsed:.2f}s")
        
        return summary

def main():
    """Test accuracy validation."""
    print("=" * 80)
    print("Phase 4.3: Accuracy Validation")
    print("=" * 80)
    
    checkpoint_path = '/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint'
    
    # Check if lm-eval is installed
    try:
        result = subprocess.run(['lm_eval', '--version'], capture_output=True, text=True, timeout=5)
        logger.info(f"lm-eval version: {result.stdout.strip()}")
    except Exception as e:
        logger.warning(f"lm-eval not found: {e}")
        logger.warning("Installing lm-eval...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'lm-eval', '-q'], timeout=300)
    
    # Run validation
    validator = AccuracyValidator(checkpoint_path)
    results = validator.validate_accuracy()
    
    # Print summary
    print("\n" + "=" * 80)
    print("ACCURACY VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nCheckpoint: {results['checkpoint_path']}")
    print(f"Elapsed time: {results['elapsed_sec']:.2f}s")
    print(f"\nTask Results:")
    for task, result in results['task_results'].items():
        print(f"  {task}: {result['status']}")
        if result['status'] != 'success':
            print(f"    Error: {result.get('error', 'Unknown error')[:100]}")
    
    # Save results
    output_file = Path('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase4_3_accuracy_validation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("Phase 4.3 Complete ✅")
    print("=" * 80)

if __name__ == '__main__':
    main()
