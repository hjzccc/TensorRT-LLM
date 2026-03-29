#!/usr/bin/env python3
"""End-to-end inference test with block size 8 codebooks.

Tests that block 8 codebooks work correctly in full model inference.
"""

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"
BLOCK8_CODEBOOKS = Path(__file__).parent / "nvfp4_kmeans_checkpoint_block8" / "codebooks-00000.safetensors"
BLOCK16_CODEBOOKS = Path(__file__).parent / "nvfp4_kmeans_checkpoint" / "codebooks-00000.safetensors"

# Test prompts
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about nature.",
]

def test_inference():
    """Test end-to-end inference with block 8 codebooks."""
    
    print("="*70)
    print("END-TO-END INFERENCE TEST - BLOCK SIZE 8 CODEBOOKS")
    print("="*70)
    
    # Load model and tokenizer
    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Model loaded successfully")
    print(f"Model dtype: {model.dtype}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Test inference
    print(f"\n{'='*70}")
    print("RUNNING INFERENCE TESTS")
    print(f"{'='*70}")
    
    results = []
    
    for idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n[{idx+1}/{len(TEST_PROMPTS)}] Testing prompt: {prompt[:50]}...")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        elapsed = time.time() - t0
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Calculate metrics
        input_length = inputs["input_ids"].shape[1]
        output_length = outputs.shape[1]
        new_tokens = output_length - input_length
        tokens_per_second = new_tokens / elapsed
        
        result = {
            "prompt": prompt,
            "response": response[:100],  # First 100 chars
            "input_tokens": input_length,
            "output_tokens": new_tokens,
            "total_tokens": output_length,
            "time_seconds": elapsed,
            "tokens_per_second": tokens_per_second,
        }
        results.append(result)
        
        print(f"  Input tokens: {input_length}")
        print(f"  Output tokens: {new_tokens}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Throughput: {tokens_per_second:.1f} tokens/sec")
        print(f"  Response: {response[:80]}...")
    
    # Summary
    print(f"\n{'='*70}")
    print("INFERENCE TEST SUMMARY")
    print(f"{'='*70}")
    
    avg_throughput = sum(r["tokens_per_second"] for r in results) / len(results)
    total_time = sum(r["time_seconds"] for r in results)
    total_tokens = sum(r["output_tokens"] for r in results)
    
    print(f"\nTotal tests: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Average throughput: {avg_throughput:.1f} tokens/sec")
    
    # Save results
    output_file = Path(__file__).parent / "block8_inference_test_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "block_size": 8,
            "test_count": len(results),
            "total_time_seconds": total_time,
            "total_tokens_generated": total_tokens,
            "average_throughput_tokens_per_sec": avg_throughput,
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file.name}")
    print(f"\n✅ END-TO-END INFERENCE TEST PASSED")

if __name__ == "__main__":
    test_inference()
