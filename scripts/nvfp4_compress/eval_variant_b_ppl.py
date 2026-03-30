#!/usr/bin/env python3
"""
Evaluate Variant B checkpoint perplexity vs baseline
"""
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_ppl(model, tokenizer, dataset, max_samples=100, batch_size=4):
    """Compute perplexity on a dataset"""
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, min(len(dataset), max_samples), batch_size):
            batch = dataset[i:i+batch_size]
            texts = batch["text"] if isinstance(batch["text"], list) else [batch["text"]]
            
            # Tokenize
            tokens = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Forward pass
            outputs = model(**tokens, labels=tokens.input_ids)
            loss = outputs.loss
            
            # Accumulate
            total_loss += loss.item() * tokens.input_ids.shape[0]
            total_tokens += tokens.input_ids.shape[0]
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"  Processed {min(i+batch_size, max_samples)} samples")
    
    # Compute perplexity
    ppl = np.exp(total_loss / total_tokens)
    return ppl

def main():
    baseline_path = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint"
    variant_b_path = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_weighted_abs"
    
    logger.info("Loading validation dataset...")
    # Use a small subset of wikitext for quick evaluation
    try:
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="validation")
        logger.info(f"Loaded {len(dataset)} validation samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Using dummy dataset for testing")
        dataset = [{"text": "This is a test sentence."} for _ in range(100)]
    
    results = {}
    
    # Evaluate baseline
    logger.info("\n" + "="*60)
    logger.info("Evaluating BASELINE checkpoint")
    logger.info("="*60)
    
    try:
        logger.info(f"Loading baseline from {baseline_path}")
        baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
        baseline_model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Computing baseline perplexity...")
        baseline_ppl = evaluate_ppl(baseline_model, baseline_tokenizer, dataset, max_samples=100)
        results["baseline_ppl"] = baseline_ppl
        logger.info(f"Baseline PPL: {baseline_ppl:.4f}")
        
        del baseline_model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Failed to evaluate baseline: {e}")
        results["baseline_error"] = str(e)
    
    # Evaluate Variant B
    logger.info("\n" + "="*60)
    logger.info("Evaluating VARIANT B checkpoint")
    logger.info("="*60)
    
    try:
        logger.info(f"Loading Variant B from {variant_b_path}")
        variant_b_tokenizer = AutoTokenizer.from_pretrained(variant_b_path, trust_remote_code=True)
        variant_b_model = AutoModelForCausalLM.from_pretrained(
            variant_b_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Computing Variant B perplexity...")
        variant_b_ppl = evaluate_ppl(variant_b_model, variant_b_tokenizer, dataset, max_samples=100)
        results["variant_b_ppl"] = variant_b_ppl
        logger.info(f"Variant B PPL: {variant_b_ppl:.4f}")
        
        del variant_b_model
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"Failed to evaluate Variant B: {e}")
        results["variant_b_error"] = str(e)
    
    # Compute degradation
    if "baseline_ppl" in results and "variant_b_ppl" in results:
        baseline_ppl = results["baseline_ppl"]
        variant_b_ppl = results["variant_b_ppl"]
        degradation = (variant_b_ppl - baseline_ppl) / baseline_ppl
        
        logger.info("\n" + "="*60)
        logger.info("RESULTS")
        logger.info("="*60)
        logger.info(f"Baseline PPL: {baseline_ppl:.4f}")
        logger.info(f"Variant B PPL: {variant_b_ppl:.4f}")
        logger.info(f"Degradation: {degradation:.2%}")
        logger.info(f"Constraint: <0.03 (3%)")
        logger.info(f"Status: {'✓ PASS' if degradation < 0.03 else '✗ FAIL'}")
        
        results["degradation_percent"] = degradation * 100
        results["constraint_met"] = degradation < 0.03
    
    # Save results
    output_file = "/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/variant_b_ppl_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
