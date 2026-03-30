#!/usr/bin/env python3
"""Fixed MMLU evaluation with memory-efficient expert loading"""

import sys
from pathlib import Path
import torch
import gc

# Import original module
sys.path.insert(0, str(Path(__file__).parent))
from lm_eval_nvfp4 import NVFP4LM as OriginalNVFP4LM

class NVFP4LMFixed(OriginalNVFP4LM):
    """Memory-efficient version that doesn't stack all experts at once"""
    
    def _load_layer(self, layer_idx: int) -> dict:
        """Load layer with memory-efficient expert handling"""
        layer = {}
        
        # Load non-expert weights first
        for shard_idx, shard_path in enumerate(self.shard_paths):
            with safe_open(shard_path, framework="pt", device="cpu") as sf:
                for key in sf.keys():
                    if f"layers.{layer_idx}." not in key:
                        continue
                    if "mlp.experts" in key:
                        continue  # Handle experts separately
                    
                    tensor = sf.get_tensor(key)
                    if key not in layer:
                        layer[key] = tensor
        
        # Load experts WITHOUT stacking (keep as dict of individual tensors)
        experts = {}
        for proj_name in ["up_proj", "down_proj", "gate_proj"]:
            experts[proj_name] = {}
            
            for expert_idx in range(self.model_config.num_experts):
                expert_weights = []
                expert_scales = []
                expert_scales_2 = []
                
                for shard_idx, shard_path in enumerate(self.shard_paths):
                    with safe_open(shard_path, framework="pt", device="cpu") as sf:
                        expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{proj_name}"
                        
                        weight_key = f"{expert_prefix}.weight"
                        scale_key = f"{expert_prefix}.weight_scale"
                        scale_2_key = f"{expert_prefix}.weight_scale_2"
                        
                        if weight_key in sf:
                            expert_weights.append(sf.get_tensor(weight_key))
                        if scale_key in sf:
                            expert_scales.append(sf.get_tensor(scale_key))
                        if scale_2_key in sf:
                            expert_scales_2.append(sf.get_tensor(scale_2_key))
                
                # Store individual expert tensors (don't stack)
                if expert_weights:
                    experts[proj_name][expert_idx] = {
                        "weight": expert_weights[0] if len(expert_weights) == 1 else torch.cat(expert_weights, dim=1),
                        "weight_scale": expert_scales[0] if expert_scales else None,
                        "weight_scale_2": expert_scales_2[0] if expert_scales_2 else None,
                    }
        
        layer["experts"] = experts
        return layer
    
    def _forward_experts(self, hidden_states, layer_experts, gate_logits):
        """Forward pass through experts without requiring stacked tensors"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        num_experts = self.model_config.num_experts
        
        # Compute expert routing
        expert_weights = torch.softmax(gate_logits, dim=-1)  # [batch, seq, num_experts]
        
        # Process each expert individually to save memory
        output = torch.zeros_like(hidden_states)
        
        for expert_idx in range(num_experts):
            expert_weight = expert_weights[..., expert_idx:expert_idx+1]  # [batch, seq, 1]
            
            # Get expert tensors
            up_weight = layer_experts["up_proj"][expert_idx]["weight"]
            down_weight = layer_experts["down_proj"][expert_idx]["weight"]
            
            # Forward through expert
            expert_out = torch.nn.functional.linear(hidden_states, up_weight)
            expert_out = torch.nn.functional.gelu(expert_out)
            expert_out = torch.nn.functional.linear(expert_out, down_weight)
            
            # Accumulate with routing weight
            output = output + expert_out * expert_weight
        
        return output

# For backward compatibility
NVFP4LM = NVFP4LMFixed

if __name__ == "__main__":
    print("Fixed MMLU evaluation module loaded")
