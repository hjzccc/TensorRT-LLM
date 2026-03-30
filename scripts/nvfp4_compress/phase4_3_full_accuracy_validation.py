#!/usr/bin/env python3
"""
Phase 4.3: Full Accuracy Validation (Production)

Measure accuracy degradation from Variant B compression using MMLU and GSM8K.
This is the production version with full evaluation (no limits).
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
import argparse

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
    
    def run_lm_eval(self, task: str, num_fewshot: int = 5, batch_size: str = 'auto', limit: int = None) -> Dict:
        """
        Run lm-eval on specified task.
        
        Args:
            task: Task name (e.g., 'mmlu', 'gsm8k')
            num_fewshot: Number of few-shot examples
            batch_size: Batch size ('auto' or integer)
            limit: Limit number of examples per task (None = full evaluation)
            
        Returns:
            Dict with evaluation results
        """
        logger.info(f"Running lm-eval on task: {task}")
        logger.info(f"  Few-shot: {num_fewshot}")
        logger.info(f"  Batch size: {batch_size}")
        if limit:
            logger.info(f"  Limit: {limit} examples")
        else:
            logger.info(f"  Limit: FULL EVALUATION (no limit)")
        
        # Build lm-eval command
        cmd = [
            'lm_eval',
            '--model', 'hf',
            '--model_args', f'pretrained={self.checkpoint_path}',
            '--tasks', task,
            '--num_fewshot', str(num_fewshot),
            '--batch_size', str(batch_size),
            '--output_path', f'/tmp/lm_eval_{task}_results.json',
            '--log_samples',
        ]
        
        if limit:
            cmd.extend(['--limit', str(limit)])
        
        try:
            logger.info(f"Running: {' '.join(cmd)}")
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=86400)  # 24 hour timeout
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✅ {task} evaluation complete in {elapsed:.2f}s")
                
                # Try to parse results from JSON file
                try:
                    with open(f'/tmp/lm_eval_{task}_results.json', 'r') as f:
                        eval_results = json.load(f)
                    return {
                        'status': 'success',
                        'elapsed_sec': elapsed,
                        'results': eval_results,
                    }
                except Exception as e:
                    logger.warning(f"Could not parse results JSON: {e}")
                    return {
                        'status': 'success',
                        'elapsed_sec': elapsed,
                        'output': result.stdout[-1000:] if result.stdout else '',
                    }
            else:
                logger.error(f"❌ {task} evaluation failed")
                logger.error(f"stderr: {result.stderr[-1000:]}")
                return {
                    'status': 'failed',
                    'elapsed_sec': elapsed,
                    'error': result.stderr[-1000:] if result.stderr else '',
                }
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {task} evaluation timed out")
            return {
                'status': 'timeout',
                'error': 'Evaluation timed out after 24 hours',
            }
        except Exception as e:
            logger.error(f"❌ {task} evaluation error: {e}")
            return {
                'status': 'error',
                'error': str(e),
            }
    
    def validate_accuracy(self, tasks: List[str] = None, num_fewshot: int = 5, batch_size: str = 'auto', limit: int = None) -> Dict:
        """
        Validate accuracy of compressed checkpoint.
        
        Args:
            tasks: List of tasks to evaluate (default: ['mmlu', 'gsm8k'])
            num_fewshot: Number of few-shot examples
            batch_size: Batch size for evaluation
            limit: Limit number of examples per task
            
        Returns:
            Dict with validation results
        """
        if tasks is None:
            tasks = ['mmlu', 'gsm8k']
        
        logger.info("Starting accuracy validation...")
        
        start_time = time.time()
        
        # Run evaluations
        results = {}
        for task in tasks:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating {task.upper()}")
            logger.info(f"{'='*80}")
            results[task] = self.run_lm_eval(task, num_fewshot=num_fewshot, batch_size=batch_size, limit=limit)
        
        elapsed = time.time() - start_time
        
        summary = {
            'checkpoint_path': str(self.checkpoint_path),
            'baseline_checkpoint_path': str(self.baseline_checkpoint_path) if self.baseline_checkpoint_path else None,
            'num_fewshot': num_fewshot,
            'batch_size': batch_size,
            'limit': limit,
            'elapsed_sec': elapsed,
            'task_results': results,
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Accuracy validation complete in {elapsed:.2f}s ({elapsed/3600:.2f} hours)")
        logger.info(f"{'='*80}")
        
        return summary

def main():
    """Run full accuracy validation."""
    parser = argparse.ArgumentParser(description='Phase 4.3: Full Accuracy Validation')
    parser.add_argument('--task', type=str, default='mmlu', choices=['mmlu', 'gsm8k', 'both'],
                        help='Task to evaluate')
    parser.add_argument('--num_fewshot', type=int, default=5, help='Number of few-shot examples')
    parser.add_argument('--batch_size', type=str, default='auto', help='Batch size')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples (None = full)')
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint',
                        help='Path to checkpoint')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 4.3: Full Accuracy Validation (Production)")
    print("=" * 80)
    
    # Determine tasks
    if args.task == 'both':
        tasks = ['mmlu', 'gsm8k']
    else:
        tasks = [args.task]
    
    # Run validation
    validator = AccuracyValidator(args.checkpoint)
    results = validator.validate_accuracy(
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("ACCURACY VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\nCheckpoint: {results['checkpoint_path']}")
    print(f"Few-shot: {results['num_fewshot']}")
    print(f"Batch size: {results['batch_size']}")
    print(f"Limit: {results['limit'] if results['limit'] else 'FULL EVALUATION'}")
    print(f"Total elapsed time: {results['elapsed_sec']:.2f}s ({results['elapsed_sec']/3600:.2f} hours)")
    print(f"\nTask Results:")
    for task, result in results['task_results'].items():
        print(f"\n  {task.upper()}:")
        print(f"    Status: {result['status']}")
        if result['status'] == 'success':
            print(f"    Elapsed: {result.get('elapsed_sec', 'N/A'):.2f}s")
            if 'results' in result:
                print(f"    Results: {json.dumps(result['results'], indent=6)}")
        else:
            print(f"    Error: {result.get('error', 'Unknown error')[:200]}")
    
    # Save results
    output_file = Path('/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/phase4_3_full_accuracy_validation_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {output_file}")
    
    print("\n" + "=" * 80)
    print("Phase 4.3 Complete ✅")
    print("=" * 80)

if __name__ == '__main__':
    main()
