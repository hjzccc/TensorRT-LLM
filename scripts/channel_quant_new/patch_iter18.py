#!/usr/bin/env python3
"""Patch iter02 copy to create integrated 3-tier experiment."""

path = "/workspace/channel_quant_new/exact_explore_iter18_integrated_3tier.py"
with open(path, "r") as f:
    code = f.read()

# 1. Update docstring
code = code.replace(
    "Exact-path Exploration Iteration 2: FP8 Budget Sweep.",
    "Exact-path Iter18: Integrated Three-Tier (BF16 rescue + FP8 + NVFP4) with Global Allocation."
)

# 2. Update output filename
code = code.replace(
    "exact_explore_iter02_budget_sweep.json",
    "exact_explore_iter18_integrated_3tier.json"
)

# 3. Add bf16 fraction to BudgetConfig
code = code.replace(
    '''@dataclass(frozen=True)
class BudgetConfig:
    """FP8 budget fractions for W1 and W2."""
    label: str
    w1_fp8_fraction: float
    w2_fp8_fraction: float

    @property
    def description(self) -> str:
        return f"W1={self.w1_fp8_fraction:.0%} FP8, W2={self.w2_fp8_fraction:.0%} FP8"''',
    '''@dataclass(frozen=True)
class BudgetConfig:
    """Budget fractions for W1 and W2 across tiers."""
    label: str
    w1_fp8_fraction: float
    w2_fp8_fraction: float
    w1_bf16_fraction: float = 0.0
    w2_bf16_fraction: float = 0.0

    @property
    def description(self) -> str:
        parts = []
        if self.w1_bf16_fraction > 0 or self.w2_bf16_fraction > 0:
            parts.append(f"BF16: W1={self.w1_bf16_fraction:.1%} W2={self.w2_bf16_fraction:.1%}")
        parts.append(f"FP8: W1={self.w1_fp8_fraction:.0%} W2={self.w2_fp8_fraction:.0%}")
        return ", ".join(parts)'''
)

# 4. Replace configs
old_configs = '''ALL_BUDGET_CONFIGS = [
    BudgetConfig("budget_10pct",       0.02, 0.08),
    BudgetConfig("budget_20pct",       0.04, 0.16),   # current best
    BudgetConfig("budget_30pct",       0.06, 0.24),
    BudgetConfig("budget_40pct",       0.08, 0.32),
    BudgetConfig("budget_50pct",       0.10, 0.40),
    BudgetConfig("budget_w2only_30pct", 0.00, 0.30),  # W2-only budget
    BudgetConfig("budget_w2only_50pct", 0.00, 0.50),  # W2-only budget
]'''

new_configs = '''ALL_BUDGET_CONFIGS = [
    # Baseline: FP8+NVFP4 only (matches iter02 budget_50pct)
    BudgetConfig("fp8_50pct",              0.10, 0.40, 0.00, 0.00),
    # BF16 rescue: promote top channels from FP8 to BF16
    BudgetConfig("rescue_2_fp8_48",        0.095, 0.385, 0.005, 0.015),
    BudgetConfig("rescue_5_fp8_45",        0.09, 0.36, 0.01, 0.04),
    BudgetConfig("rescue_10_fp8_40",       0.08, 0.32, 0.02, 0.08),
    # BF16 rescue + extra FP8 budget
    BudgetConfig("rescue_2_fp8_58",        0.115, 0.465, 0.005, 0.015),
    BudgetConfig("rescue_5_fp8_55",        0.11, 0.44, 0.01, 0.04),
    BudgetConfig("rescue_10_fp8_50",       0.10, 0.40, 0.02, 0.08),
    # More aggressive BF16
    BudgetConfig("rescue_15_fp8_50",       0.10, 0.40, 0.03, 0.12),
]'''

code = code.replace(old_configs, new_configs)

# 5. Modify mixed_exact_linear to support bf16 tier
# Add a new function right before mixed_exact_linear
bf16_extension = '''
def mixed_3tier_linear(
    input_tensor: torch.Tensor, weight: torch.Tensor,
    fp8_row_mask: torch.Tensor, bf16_row_mask: torch.Tensor
) -> torch.Tensor:
    """Three-tier linear: BF16 (rescue) + FP8 + NVFP4 with row-moving alignment."""
    bf16_row_mask = bf16_row_mask.to(device=weight.device, dtype=torch.bool)
    fp8_row_mask = fp8_row_mask.to(device=weight.device, dtype=torch.bool)
    # BF16 mask overrides FP8
    fp8_only = fp8_row_mask & ~bf16_row_mask
    nvfp4_mask = ~fp8_row_mask & ~bf16_row_mask

    # If no BF16, fall back to existing 2-tier
    if not bf16_row_mask.any():
        return mixed_exact_linear(input_tensor, weight, fp8_row_mask)

    input_2d, prefix_shape = exact_eval.flatten_for_linear(input_tensor)
    out = torch.empty(
        (input_2d.shape[0], weight.shape[0]), dtype=input_2d.dtype, device=input_2d.device
    )

    bf16_rows = torch.nonzero(bf16_row_mask, as_tuple=False).flatten()
    fp8_rows = torch.nonzero(fp8_only, as_tuple=False).flatten()
    fp4_rows = torch.nonzero(nvfp4_mask, as_tuple=False).flatten()

    # Alignment: NVFP4 needs rows % 32 == 0
    if fp4_rows.numel() > 0 and fp4_rows.numel() % 32 != 0:
        deficit = fp4_rows.numel() % 32
        moved = fp4_rows[-deficit:]
        fp8_rows = torch.cat([fp8_rows, moved])
        fp4_rows = fp4_rows[:-deficit]

    # FP8 needs rows % 16 == 0
    if fp8_rows.numel() > 0 and fp8_rows.numel() % 16 != 0:
        excess = fp8_rows.numel() % 16
        moved = fp8_rows[-excess:]
        bf16_rows = torch.cat([bf16_rows, moved])
        fp8_rows = fp8_rows[:-excess]

    if bf16_rows.numel() > 0:
        out[:, bf16_rows] = exact_eval.bf16_linear(input_2d, weight[bf16_rows])
    if fp8_rows.numel() > 0:
        out[:, fp8_rows] = exact_eval.fp8_linear(input_2d, weight[fp8_rows])
    if fp4_rows.numel() > 0:
        out[:, fp4_rows] = exact_eval.nvfp4_linear(input_2d, weight[fp4_rows])

    return exact_eval.restore_linear_shape(out, prefix_shape)

'''

code = code.replace(
    "def mixed_exact_linear(",
    bf16_extension + "def mixed_exact_linear("
)

# 6. Build BF16 masks in build_budget_masks
# After building the FP8 masks, build BF16 masks from the top of the same scores
old_build_end = '''    return mixed_w1_masks, mixed_w2_masks, metadata'''

new_build_end = '''    # Build BF16 rescue masks: top channels from the FP8 mask get promoted to BF16
    bf16_w1_masks: dict[int, dict[int, torch.Tensor]] = {}
    bf16_w2_masks: dict[int, dict[int, torch.Tensor]] = {}
    if budget_config.w1_bf16_fraction > 0 or budget_config.w2_bf16_fraction > 0:
        bf16_w1_raw, bf16_w2_raw, _ = build_global_fraction_masks(
            prioritized_cache,
            config,
            budget_config.w1_bf16_fraction,
            budget_config.w2_bf16_fraction,
            budget_source=f"bf16_rescue_{budget_config.label}",
        )
        bf16_w1_masks = bf16_w1_raw
        bf16_w2_masks = bf16_w2_raw
    else:
        for li in range(config.num_hidden_layers):
            bf16_w1_masks[li] = {ei: torch.zeros_like(mixed_w1_masks[li][ei]) for ei in mixed_w1_masks[li]}
            bf16_w2_masks[li] = {ei: torch.zeros_like(mixed_w2_masks[li][ei]) for ei in mixed_w2_masks[li]}

    metadata["w1_bf16_fraction_target"] = budget_config.w1_bf16_fraction
    metadata["w2_bf16_fraction_target"] = budget_config.w2_bf16_fraction

    return mixed_w1_masks, mixed_w2_masks, bf16_w1_masks, bf16_w2_masks, metadata'''

code = code.replace(old_build_end, new_build_end, 1)

# 7. Update the caller to unpack 5 values and pass bf16 masks to MoE forward
code = code.replace(
    "mixed_w1_masks, mixed_w2_masks, budget_meta = build_budget_masks(",
    "mixed_w1_masks, mixed_w2_masks, bf16_w1_masks, bf16_w2_masks, budget_meta = build_budget_masks("
)

# 8. Update mixed_moe_forward calls to pass bf16 masks
code = code.replace(
    "moe_out = mixed_moe_forward(\n                    hidden_states, moe_tensors, config, w1_masks, w2_masks, layer_idx,\n                )",
    "moe_out = mixed_moe_forward(\n                    hidden_states, moe_tensors, config, w1_masks, w2_masks, bf16_w1, bf16_w2, layer_idx,\n                )"
)

# 9. Update mixed_moe_forward signature to accept bf16 masks
code = code.replace(
    '''def mixed_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    w1_masks: dict[int, torch.Tensor],
    w2_masks: dict[int, torch.Tensor],
    layer_idx: int,''',
    '''def mixed_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    w1_masks: dict[int, torch.Tensor],
    w2_masks: dict[int, torch.Tensor],
    bf16_w1_masks: dict[int, torch.Tensor],
    bf16_w2_masks: dict[int, torch.Tensor],
    layer_idx: int,'''
)

# 10. Update the expert loop to use 3-tier linear
code = code.replace(
    "gate_up = mixed_exact_linear(current_state, gate_up_proj[expert_idx], w1_fp8_mask)",
    '''w1_bf16_mask = expand_w1_pair_mask(bf16_w1_masks.get(expert_idx, torch.zeros(w1_fp8_mask.numel(), dtype=torch.bool)), gate_up_proj[expert_idx])
        gate_up = mixed_3tier_linear(current_state, gate_up_proj[expert_idx], w1_fp8_mask, w1_bf16_mask)'''
)

code = code.replace(
    "current_hidden = mixed_exact_linear(hidden, down_proj[expert_idx], w2_masks[expert_idx])",
    '''w2_bf16_mask = bf16_w2_masks.get(expert_idx, torch.zeros(down_proj[expert_idx].shape[0], dtype=torch.bool))
        current_hidden = mixed_3tier_linear(hidden, down_proj[expert_idx], w2_masks[expert_idx], w2_bf16_mask)'''
)

# 11. Update the main loop to pass bf16 masks
code = code.replace(
    "w1_masks = budget_w1_masks[layer_idx]\n                w2_masks = budget_w2_masks[layer_idx]",
    "w1_masks = budget_w1_masks[layer_idx]\n                w2_masks = budget_w2_masks[layer_idx]\n                bf16_w1 = budget_bf16_w1[layer_idx]\n                bf16_w2 = budget_bf16_w2[layer_idx]"
)

# 12. Store bf16 masks
code = code.replace(
    "budget_w1_masks = mixed_w1_masks\n        budget_w2_masks = mixed_w2_masks",
    "budget_w1_masks = mixed_w1_masks\n        budget_w2_masks = mixed_w2_masks\n        budget_bf16_w1 = bf16_w1_masks\n        budget_bf16_w2 = bf16_w2_masks"
)

with open(path, "w") as f:
    f.write(code)
print("Patched iter18 successfully")
