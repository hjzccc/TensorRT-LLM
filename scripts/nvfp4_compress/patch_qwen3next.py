#!/usr/bin/env python3
"""Patch transformers qwen3_next MoE forward for meta tensor traceability."""
import sys

FILE = "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_next/modeling_qwen3_next.py"

with open(FILE) as f:
    content = f.read()

# Patch 1: MoE forward — replace data-dependent expert loop with fixed-range loop
OLD_MOE = (
    "        # Loop over all available experts in the model and perform the computation on each expert\n"
    "        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()\n"
    "        for expert_idx in expert_hit:\n"
    "            expert_layer = self.experts[expert_idx]\n"
    "            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))\n"
)

NEW_MOE = (
    "        for expert_idx_int in range(self.num_experts):\n"
    "            expert_layer = self.experts[expert_idx_int]\n"
    "            idx, top_x = torch.where(expert_mask[expert_idx_int].squeeze(0))\n"
    "            if top_x.shape[0] == 0:\n"
    "                continue\n"
)

if OLD_MOE in content:
    content = content.replace(OLD_MOE, NEW_MOE)
    print("Patched MoE forward: fixed-range expert loop")
else:
    print("WARNING: MoE patch target not found (may already be patched)")

# Patch 2: cache_position meta tensor check
OLD_CACHE = "if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):"
NEW_CACHE = (
    "try:\n"
    "            _cp_check = cache_position[0].item() > 0\n"
    "        except (RuntimeError, NotImplementedError):\n"
    "            _cp_check = False\n"
    "        if _cp_check or (attention_mask is not None and torch.all(attention_mask == 1)):"
)

if OLD_CACHE in content:
    content = content.replace(OLD_CACHE, NEW_CACHE)
    print("Patched cache_position meta tensor check")
else:
    print("WARNING: cache_position patch target not found (may already be patched)")

with open(FILE, "w") as f:
    f.write(content)
print("Done")
