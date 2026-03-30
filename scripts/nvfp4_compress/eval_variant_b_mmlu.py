#!/usr/bin/env python3
"""
MMLU evaluation for Variant B compressed checkpoint.
Loads compressed weights directly without decompression.
"""

import sys
import json
import torch
import gc
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoConfig
import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

# Configuration
CHECKPOINT_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/compressed_2b075b_zero_fixed_weighted_abs")
BASELINE_DIR = Path("/home/jerry/Documents/fork_new/TensorRT-LLM-dual-tile/scripts/nvfp4_compress/nvfp4_checkpoint")

BATCH_SIZE = 4
MAX_BATCH_TOTAL_TOKENS = 4096
SUBJECTS = ["professional_law", "abstract_algebra", "anatomy", "astronomy", "business_ethics"]

class CompressedNVFP4LM(LM):
    """Language model wrapper for compressed NVFP4 checkpoint"""
    
    def __init__(self, checkpoint_dir: Path, device: str = "cuda"):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        self.index = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from {self.checkpoint_dir}")
        
        # Load config
        config_path = self.checkpoint_dir / "config.json"
        with open(config_path) as f:
            self.config = json.load(f)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        
        # Load checkpoint index
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            self.index = json.load(f)
        
        print(f"Model config loaded: {self.config.get('model_type', 'unknown')}")
        print(f"Checkpoint has {len(self.index['weight_map'])} weights across {len(set(self.index['weight_map'].values()))} shards")
    
    def _get_shard_path(self, key: str) -> Optional[Path]:
        """Get shard file path for a weight key"""
        shard_name = self.index["weight_map"].get(key)
        if not shard_name:
            return None
        return self.checkpoint_dir / shard_name
    
    def _load_weight(self, key: str) -> Optional[torch.Tensor]:
        """Load a single weight from checkpoint"""
        shard_path = self._get_shard_path(key)
        if not shard_path or not shard_path.exists():
            return None
        
        try:
            weights = load_file(str(shard_path), device="cpu")
            return weights.get(key)
        except Exception as e:
            print(f"Error loading {key}: {e}")
            return None
    
    def _decompress_weight(self, weight_key: str) -> torch.Tensor:
        """Decompress a weight from compressed format"""
        # Load compression components
        indices = self._load_weight(f"{weight_key}_indices")
        codebook = self._load_weight(f"{weight_key}_codebook_entries")
        scales = self._load_weight(f"{weight_key}_scale")
        
        if indices is None or codebook is None:
            # Not compressed, load directly
            return self._load_weight(weight_key)
        
        # Decompress: weight = codebook[indices] * scales
        decompressed = codebook[indices.long()]
        if scales is not None:
            decompressed = decompressed * scales.unsqueeze(-1)
        
        return decompressed
    
    def generate_until(self, requests, until=None, max_length=None, **kwargs):
        """Generate completions for requests"""
        if until is None:
            until = ["\n"]
        
        results = []
        for request in requests:
            prompt = request.args[0]
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(self.device)
            
            # Generate (placeholder - would need full model implementation)
            # For now, return dummy response
            results.append("dummy response")
        
        return results
    
    def loglikelihood(self, requests, **kwargs):
        """Compute log-likelihood for requests"""
        results = []
        
        for request in requests:
            context, completion = request.args
            
            # Tokenize context and completion
            context_ids = self.tokenizer(context, return_tensors="pt")["input_ids"]
            full_ids = self.tokenizer(context + completion, return_tensors="pt")["input_ids"]
            
            # Compute log-likelihood (placeholder)
            # Would need full model forward pass
            results.append((0.0, False))  # (log_likelihood, is_greedy)
        
        return results
    
    def loglikelihood_rolling(self, requests, **kwargs):
        """Compute rolling log-likelihood"""
        return self.loglikelihood(requests, **kwargs)

def analyze_checkpoint():
    """Analyze checkpoint structure and compression metrics"""
    print("=" * 70)
    print("VARIANT B CHECKPOINT ANALYSIS")
    print("=" * 70)
    
    # Load index
    index_path = CHECKPOINT_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    
    weight_map = index["weight_map"]
    
    # Count weight types
    weight_types = {}
    for key in weight_map.keys():
        if "_indices" in key:
            weight_types["indices"] = weight_types.get("indices", 0) + 1
        elif "_codebook_entries" in key:
            weight_types["codebook_entries"] = weight_types.get("codebook_entries", 0) + 1
        elif "_scale" in key:
            weight_types["scales"] = weight_types.get("scales", 0) + 1
        elif ".weight" in key:
            weight_types["weights"] = weight_types.get("weights", 0) + 1
    
    print(f"\nWeight Statistics:")
    print(f"  Total weights: {len(weight_map)}")
    for wtype, count in sorted(weight_types.items()):
        print(f"  {wtype}: {count}")
    
    # Shard statistics
    shards = {}
    for key, shard in weight_map.items():
        shards[shard] = shards.get(shard, 0) + 1
    
    print(f"\nShard Statistics:")
    print(f"  Total shards: {len(shards)}")
    print(f"  Avg weights per shard: {len(weight_map) / len(shards):.1f}")
    
    # Check shard files
    missing = 0
    total_size = 0
    for shard in shards.keys():
        shard_path = CHECKPOINT_DIR / shard
        if shard_path.exists():
            total_size += shard_path.stat().st_size
        else:
            missing += 1
    
    print(f"  Missing shards: {missing}")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    
    # Compression ratio
    baseline_size = sum((BASELINE_DIR / f).stat().st_size 
                       for f in BASELINE_DIR.glob("model-*.safetensors") 
                       if f.is_file())
    
    print(f"\nCompression Metrics:")
    print(f"  Baseline size: {baseline_size / (1024**3):.2f} GB")
    print(f"  Compressed size: {total_size / (1024**3):.2f} GB")
    print(f"  Compression ratio: {baseline_size / total_size:.2f}x")
    
    return index

def main():
    """Main evaluation function"""
    print("Starting Variant B MMLU Evaluation")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Subjects: {SUBJECTS}")
    
    # Analyze checkpoint
    index = analyze_checkpoint()
    
    print("\n" + "=" * 70)
    print("CHECKPOINT VALIDATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Implement full model forward pass")
    print("2. Run MMLU evaluation on subset")
    print("3. Compare with baseline")

if __name__ == "__main__":
    main()
