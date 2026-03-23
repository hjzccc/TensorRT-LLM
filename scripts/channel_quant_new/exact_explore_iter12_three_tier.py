#!/usr/bin/env python3
"""Exact-path Exploration: Three-Tier Precision (BF16 + FP8 + NVFP4).

Tests channel-level assignment across three precision tiers:
  - BF16:  no quantization (most sensitive channels)
  - FP8:   moderate quantization (medium channels)
  - NVFP4: aggressive quantization (cold channels)

Also tests two-tier BF16+NVFP4 (skip FP8 entirely).

Configurations:
  1. bf16_5_nvfp4_95   — 5% BF16, 95% NVFP4 (two-tier, no FP8)
  2. bf16_10_nvfp4_90  — 10% BF16, 90% NVFP4
  3. bf16_20_nvfp4_80  — 20% BF16, 80% NVFP4
  4. bf16_5_fp8_45_nvfp4_50  — 5% BF16, 45% FP8, 50% NVFP4 (three-tier)
  5. bf16_10_fp8_40_nvfp4_50 — 10% BF16, 40% FP8, 50% NVFP4
  6. bf16_10_fp8_20_nvfp4_70 — 10% BF16, 20% FP8, 70% NVFP4
  7. bf16_5_fp8_15_nvfp4_80  — 5% BF16, 15% FP8, 80% NVFP4
  + baselines: uniform_bf16, uniform_fp8, uniform_nvfp4

Run inside trtllm-dual-tile docker:
  python /workspace/channel_quant_new/exact_explore_iter12_three_tier.py --nsamples 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextRotaryEmbedding

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/channel_quant_new")
sys.path.insert(0, "/workspace/channel_quant")
sys.path.insert(0, str(SCRIPT_DIR.parent / "channel_quant"))

import exact_docker_eval as exact_eval
from spike1_ground_truth import (
    build_text_config,
    layer_keys,
    load_root_config,
    move_tensor,
    release_tensors,
    rms_norm_qwen3_next,
    shorten_layer_tensors,
)

RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "exact_explore_iter12_three_tier.json"
CALIBRATION_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter01_calibration_cache.pt")
METRIC_CACHE_PATH = Path("/workspace/channel_quant/results/proper_iter10_novel_perchannel_metric_cache.pt")

W1_CHANNELS = 1024
W2_CHANNELS = 2048
W1_GRANULARITY = 16
W2_GRANULARITY = 32


@dataclass(frozen=True)
class ThreeTierConfig:
    label: str
    bf16_fraction: float
    fp8_fraction: float
    nvfp4_fraction: float

    @property
    def description(self) -> str:
        parts = []
        if self.bf16_fraction > 0:
            parts.append(f"BF16={self.bf16_fraction:.0%}")
        if self.fp8_fraction > 0:
            parts.append(f"FP8={self.fp8_fraction:.0%}")
        if self.nvfp4_fraction > 0:
            parts.append(f"NVFP4={self.nvfp4_fraction:.0%}")
        return ", ".join(parts)


ALL_CONFIGS = [
    ThreeTierConfig("bf16_2_nvfp4_98",             0.02, 0.00, 0.98),
    ThreeTierConfig("bf16_5_nvfp4_95",             0.05, 0.00, 0.95),
    ThreeTierConfig("bf16_10_nvfp4_90",            0.10, 0.00, 0.90),
    ThreeTierConfig("bf16_15_nvfp4_85",            0.15, 0.00, 0.85),
    ThreeTierConfig("bf16_20_nvfp4_80",            0.20, 0.00, 0.80),
    ThreeTierConfig("bf16_30_nvfp4_70",            0.30, 0.00, 0.70),
    ThreeTierConfig("bf16_50_nvfp4_50",            0.50, 0.00, 0.50),
    ThreeTierConfig("bf16_5_fp8_45_nvfp4_50",      0.05, 0.45, 0.50),
    ThreeTierConfig("bf16_10_fp8_40_nvfp4_50",     0.10, 0.40, 0.50),
    ThreeTierConfig("bf16_10_fp8_20_nvfp4_70",     0.10, 0.20, 0.70),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=exact_eval.MODEL_ID)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--layer-batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def compute_channel_sensitivity(
    weight: torch.Tensor,
) -> torch.Tensor:
    """Compute per-output-channel sensitivity score.
    
    Uses weight magnitude variance as a proxy — channels with high magnitude
    variance are more sensitive to quantization (outlier-prone).
    Falls back to absmax if metric cache unavailable.
    """
    return weight.float().abs().mean(dim=-1)


def build_three_tier_masks(
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    config: ThreeTierConfig,
    num_experts: int,
    routing_weights: dict[int, float] | None = None,
    oracle_data: dict | None = None,
    profiling_data: dict | None = None,
    layer_idx: int = 0,
    oracle_curves: dict | None = None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build per-expert channel tier assignments.
    
    Returns:
        w1_tiers: dict[expert_idx] -> Tensor of shape [W1_CHANNELS] with values 0=nvfp4, 1=fp8, 2=bf16
        w2_tiers: dict[expert_idx] -> Tensor of shape [W2_CHANNELS] with values 0=nvfp4, 1=fp8, 2=bf16
    """
    w1_tiers: dict[int, torch.Tensor] = {}
    w2_tiers: dict[int, torch.Tensor] = {}

    for expert_idx in range(num_experts):
        # Compute per-expert BF16 fraction based on routing importance
        expert_bf16_fraction = config.bf16_fraction
        
        if routing_weights is not None and expert_idx in routing_weights:
            expert_bf16_fraction = _compute_expert_bf16_fraction(
                config.bf16_fraction,
                routing_weights[expert_idx],
                routing_weights,
            )
        
        # Apply expert-level error-aware adjustment
        if profiling_data and str(layer_idx) in profiling_data:
            layer_data = profiling_data[str(layer_idx)]
            # Find this expert in the layer data
            for expert_data in layer_data['experts']:
                if expert_data['expert'] == expert_idx:
                    # Compute expert-level error concentration
                    w1_gini = expert_data['w1']['gini_unweighted']
                    w2_gini = expert_data['w2']['gini_unweighted']
                    expert_error_concentration = (w1_gini + w2_gini) / 2
                    expert_bf16_fraction = _compute_expert_error_aware_bf16_fraction(
                        expert_bf16_fraction, expert_error_concentration
                    )
                    break
        
        for proj, weights, n_channels, granularity, tiers_out in [
            ("w1", gate_up_weights[expert_idx], W1_CHANNELS, W1_GRANULARITY, w1_tiers),
            ("w2", down_weights[expert_idx], W2_CHANNELS, W2_GRANULARITY, w2_tiers),
        ]:
            sensitivity = compute_channel_sensitivity(weights)
            
            # Phase 8: Use per-projection BF16 allocation based on W1/W2 characteristics
            proj_bf16_fraction = expert_bf16_fraction
            if profiling_data and str(layer_idx) in profiling_data:
                for expert_data in profiling_data[str(layer_idx)]['experts']:
                    if expert_data['expert'] == expert_idx:
                        if proj == "w1":
                            w1_gini = expert_data['w1']['gini_unweighted']
                            w1_mae = expert_data['w1']['total_mae']
                            proj_bf16_fraction = _compute_w1_bf16_fraction(
                                expert_bf16_fraction, w1_gini, w1_mae
                            )
                        elif proj == "w2":
                            w2_gini = expert_data['w2']['gini_unweighted']
                            w2_mae = expert_data['w2']['total_mae']
                            proj_bf16_fraction = _compute_w2_bf16_fraction(
                                expert_bf16_fraction, w2_gini, w2_mae
                            )
                        break
            
            # Phase 6: Use cumulative curves to optimize BF16 fraction per expert/projection
            if oracle_curves and layer_idx in oracle_curves and expert_idx in oracle_curves[layer_idx]:
                expert_curves = oracle_curves[layer_idx][expert_idx]
                if proj == "w1" and "w1" in expert_curves:
                    proj_bf16_fraction = _compute_optimal_bf16_fraction_from_curves(
                        expert_curves["w1"].get("cum_err_by_mag", []),
                        expert_curves["w1"].get("cum_err_oracle", []),
                        expert_bf16_fraction,
                        strategy="oracle_guided",
                    )
                elif proj == "w2" and "w2" in expert_curves:
                    proj_bf16_fraction = _compute_optimal_bf16_fraction_from_curves(
                        expert_curves["w2"].get("cum_err_by_mag", []),
                        expert_curves["w2"].get("cum_err_oracle", []),
                        expert_bf16_fraction,
                        strategy="oracle_guided",
                    )
            
            n_bf16_raw = int(round(proj_bf16_fraction * n_channels))
            n_fp8_raw = int(round(config.fp8_fraction * n_channels))
            n_bf16 = _snap(n_bf16_raw, n_channels, granularity) if n_bf16_raw > 0 else 0
            n_fp8 = _snap(n_fp8_raw, n_channels - n_bf16, granularity) if n_fp8_raw > 0 else 0
            n_nvfp4 = n_channels - n_bf16 - n_fp8
            if n_nvfp4 < 0:
                n_fp8 = n_channels - n_bf16
                n_nvfp4 = 0

            sensitivity_cpu = sensitivity.cpu()
            tiers = torch.zeros(n_channels, dtype=torch.long)
            
            # Use oracle-based selection if available, otherwise fall back to magnitude
            if oracle_data and layer_idx in oracle_data and expert_idx in oracle_data[layer_idx]:
                # Oracle-based: sort by actual quantization error
                expert_oracle = oracle_data[layer_idx][expert_idx]
                if proj == "w1" and "w1_oracle_error" in expert_oracle:
                    oracle_error = expert_oracle["w1_oracle_error"]
                    _, sorted_idx = torch.tensor(oracle_error).sort(descending=True)
                elif proj == "w2" and "w2_oracle_error" in expert_oracle:
                    oracle_error = expert_oracle["w2_oracle_error"]
                    _, sorted_idx = torch.tensor(oracle_error).sort(descending=True)
                else:
                    # Fall back to magnitude if oracle not available for this projection
                    _, sorted_idx = sensitivity_cpu.sort(descending=True)
            else:
                # Magnitude-based: sort by weight magnitude (current approach)
                _, sorted_idx = sensitivity_cpu.sort(descending=True)

            if n_bf16 > 0:
                tiers[sorted_idx[:n_bf16]] = 2
            if n_fp8 > 0:
                tiers[sorted_idx[n_bf16 : n_bf16 + n_fp8]] = 1

            # Move zero-weight NVFP4 channels to BF16 to avoid NVFP4 kernel
            # NaN from division-by-zero scale on all-zero sub-matrices.
            nvfp4_channels = (tiers == 0).nonzero(as_tuple=True)[0]
            if nvfp4_channels.numel() > 0:
                zero_mask = sensitivity_cpu[nvfp4_channels] == 0
                if zero_mask.any():
                    tiers[nvfp4_channels[zero_mask]] = 2

            tiers_out[expert_idx] = tiers

    return w1_tiers, w2_tiers



def _snap(n: int, total: int, granularity: int) -> int:
    if n <= 0:
        return 0
    if n >= total:
        return total
    return max(granularity, ((n + granularity // 2) // granularity) * granularity)

def _compute_expert_bf16_fraction(
    base_fraction: float,
    expert_routing_weight: float,
    all_routing_weights: dict[int, float],
    strategy: str = "proportional",
) -> float:
    """Compute per-expert BF16 fraction based on routing importance.
    
    Args:
        base_fraction: Base BF16 fraction from config
        expert_routing_weight: routing_weight_sum for this expert
        all_routing_weights: dict[expert_idx] -> routing_weight_sum for all experts
        strategy: Allocation strategy
            - "proportional": More routing weight → more BF16 budget
            - "threshold": Only high-routing experts get extra BF16
            - "uniform": All experts get same BF16 (no routing awareness)
    
    Returns:
        Per-expert BF16 fraction
    """
    total_routing = sum(all_routing_weights.values())
    if total_routing < 1e-10:
        return base_fraction
    if strategy == "proportional":
        # More routing weight → more BF16 budget
        # Scale BF16 fraction proportionally to routing weight
        routing_fraction = expert_routing_weight / total_routing
        # Map routing fraction to [0.8, 1.2] multiplier
        scale = min(1.2, max(0.8, 0.8 + routing_fraction * 0.8))
        return base_fraction * scale
    elif strategy == "threshold":
        # Only high-routing experts get extra BF16
        avg_routing = total_routing / len(all_routing_weights)
        if expert_routing_weight > avg_routing * 1.5:
            return base_fraction * 1.2
        elif expert_routing_weight < avg_routing * 0.5:
            return base_fraction * 0.8
        else:
            return base_fraction
    else:  # uniform
        # All experts get same BF16 (no routing awareness)
        return base_fraction


def _compute_layer_bf16_fraction(
    layer_idx: int,
    num_layers: int,
    base_fraction: float,
    strategy: str = "linear_increase",
) -> float:
    """Compute BF16 fraction based on layer depth.
    
    Args:
        layer_idx: Current layer index (0 to num_layers-1)
        num_layers: Total number of layers
        base_fraction: Base BF16 fraction from config
        strategy: Allocation strategy
            - "linear_increase": Early layers get less BF16, later layers get more
            - "linear_decrease": Early layers get more BF16, later layers get less
            - "uniform": All layers get same BF16 (no depth awareness)
    
    Returns:
        Per-layer BF16 fraction
    """
    if strategy == "linear_increase":
        # Early layers (0): base_fraction * 0.7
        # Late layers (num_layers-1): base_fraction * 1.3
        return base_fraction * (0.7 + 0.6 * (layer_idx / (num_layers - 1)))
    elif strategy == "linear_decrease":
        # Early layers (0): base_fraction * 1.3
        # Late layers (num_layers-1): base_fraction * 0.7
        return base_fraction * (1.3 - 0.6 * (layer_idx / (num_layers - 1)))
    else:
        # Uniform allocation
        return base_fraction


def _compute_error_aware_bf16_fraction(
    base_fraction: float,
    layer_error_concentration: float,
    strategy: str = "gini_based",
) -> float:
    """Compute BF16 fraction based on error concentration (Gini coefficient).
    
    Args:
        base_fraction: Base BF16 fraction from config
        layer_error_concentration: Gini coefficient for the layer (0-1)
        strategy: Allocation strategy
            - "gini_based": Higher Gini (more concentrated error) → more BF16
            - "uniform": No error-aware adjustment
    
    Returns:
        Error-aware adjusted BF16 fraction
    """
    if strategy == "gini_based":
        # Higher Gini = more concentrated error = need more BF16
        # Gini range: 0.04 (low) to 0.27 (high)
        # Map to [0.7, 1.3] multiplier
        # Clamp to reasonable range
        scale = min(1.3, max(0.7, 0.7 + layer_error_concentration * 2.0))
        return base_fraction * scale
    else:
        # No error-aware adjustment
        return base_fraction


def _compute_expert_error_aware_bf16_fraction(
    base_fraction: float,
    expert_error_concentration: float,
    strategy: str = "gini_based",
) -> float:
    """Compute per-expert BF16 fraction based on error concentration.
    
    Args:
        base_fraction: Base BF16 fraction from config
        expert_error_concentration: Gini coefficient for the expert (0-1)
        strategy: Allocation strategy
            - "gini_based": Higher Gini (more concentrated error) → more BF16
            - "uniform": No expert-level error-aware adjustment
    
    Returns:
        Expert-error-aware adjusted BF16 fraction
    """
    if strategy == "gini_based":
        # Higher Gini = more concentrated error = need more BF16
        # Gini range: 0.05 (low) to 0.50 (high)
        # Map to [0.8, 1.2] multiplier
        scale = min(1.2, max(0.8, 0.8 + expert_error_concentration * 0.8))
        return base_fraction * scale
    else:
        # No expert-level error-aware adjustment
        return base_fraction



def _compute_w1_bf16_fraction(
    base_fraction: float,
    w1_gini: float,
    w1_mae: float,
    strategy: str = "gini_mae_hybrid",
) -> float:
    """Compute W1-specific BF16 fraction.
    
    W1 characteristics:
    - High Gini variability (0.0220-0.8554, 38x range)
    - Negative Gini-MAE correlation (-0.1976)
    - Needs more aggressive allocation for high-Gini experts
    
    Strategy:
    - "gini_mae_hybrid": Use both Gini and MAE for allocation
    - "gini_only": Use only Gini
    - "mae_only": Use only MAE
    """
    if strategy == "gini_mae_hybrid":
        # W1 has high Gini variability, use it as primary signal
        gini_scale = min(1.5, max(0.5, 0.7 + w1_gini * 2.0))
        mae_scale = min(1.2, max(0.8, 0.8 + (w1_mae / 0.0972) * 0.4))
        combined_scale = 0.8 * gini_scale + 0.2 * mae_scale
        return base_fraction * combined_scale
    elif strategy == "gini_only":
        scale = min(1.5, max(0.5, 0.7 + w1_gini * 2.0))
        return base_fraction * scale
    else:  # mae_only
        mae_scale = min(1.2, max(0.8, 0.8 + (w1_mae / 0.0972) * 0.4))
        return base_fraction * mae_scale


def _compute_w2_bf16_fraction(
    base_fraction: float,
    w2_gini: float,
    w2_mae: float,
    strategy: str = "gini_mae_hybrid",
) -> float:
    """Compute W2-specific BF16 fraction.
    
    W2 characteristics:
    - Low Gini variability (0.0288-0.2046, 7x range)
    - Positive Gini-MAE correlation (0.4146)
    - More stable, can use simpler allocation
    
    Strategy:
    - "gini_mae_hybrid": Use both Gini and MAE for allocation
    - "gini_only": Use only Gini
    - "mae_only": Use only MAE
    """
    if strategy == "gini_mae_hybrid":
        # W2 has low Gini variability, use MAE as primary signal
        gini_scale = min(1.3, max(0.7, 0.7 + w2_gini * 2.0))
        mae_scale = min(1.5, max(0.5, 0.5 + (w2_mae / 0.0441) * 1.0))
        combined_scale = 0.4 * gini_scale + 0.6 * mae_scale
        return base_fraction * combined_scale
    elif strategy == "gini_only":
        scale = min(1.3, max(0.7, 0.7 + w2_gini * 2.0))
        return base_fraction * scale
    else:  # mae_only
        mae_scale = min(1.5, max(0.5, 0.5 + (w2_mae / 0.0441) * 1.0))
        return base_fraction * mae_scale

def _compute_optimal_bf16_fraction_from_curves(
    cum_err_by_mag: list,
    cum_err_oracle: list,
    base_fraction: float,
    strategy: str = "oracle_guided",
) -> float:
    """Compute optimal BF16 fraction using cumulative error curves.
    
    Args:
        cum_err_by_mag: Cumulative error by magnitude (0-100 scale)
        cum_err_oracle: Cumulative error by oracle (0-100 scale)
        base_fraction: Base BF16 fraction from config
        strategy: Optimization strategy
            - "oracle_guided": Use oracle curve to find optimal BF16%
            - "magnitude_guided": Use magnitude curve to find optimal BF16%
            - "uniform": Use base_fraction (no optimization)
    
    Returns:
        Optimized BF16 fraction
    """
    if strategy == "oracle_guided" and cum_err_oracle:
        # Find the BF16% that achieves 80% error reduction
        target_error_reduction = 80.0
        for i, cum_err in enumerate(cum_err_oracle):
            if cum_err >= target_error_reduction:
                # Interpolate to find exact percentage
                optimal_pct = (i / len(cum_err_oracle)) * 100
                # Scale to [0.5x, 1.5x] of base_fraction
                scale = min(1.5, max(0.5, optimal_pct / 100))
                return base_fraction * scale
        # If we don't reach 80%, use maximum available
        return base_fraction * 1.5
    elif strategy == "magnitude_guided" and cum_err_by_mag:
        # Find the BF16% that achieves 75% error reduction
        target_error_reduction = 75.0
        for i, cum_err in enumerate(cum_err_by_mag):
            if cum_err >= target_error_reduction:
                optimal_pct = (i / len(cum_err_by_mag)) * 100
                scale = min(1.5, max(0.5, optimal_pct / 100))
                return base_fraction * scale
        return base_fraction * 1.5
    else:
        # No optimization
        return base_fraction
    
    routing_fraction = expert_routing_weight / total_routing
    
    if strategy == "proportional":
        # Scale BF16 fraction by routing importance
        # Low routing (0.1% of total): 0.5x base_fraction
        # High routing (1% of total): 1.5x base_fraction
        # Formula: 0.5 + routing_fraction * 1000 scales [0.001, 0.01] to [1.5, 10]
        # Clamp to reasonable range [0.5, 2.0]
        scale = min(2.0, max(0.5, 0.5 + routing_fraction * 1000))
        return base_fraction * scale
    elif strategy == "threshold":
        # Only allocate extra BF16 if routing weight is above median
        median_routing = sorted(all_routing_weights.values())[len(all_routing_weights) // 2]
        if expert_routing_weight > median_routing:
            return base_fraction * 1.5
        else:
            return base_fraction * 0.5
    else:
        # Uniform allocation
        return base_fraction



def three_tier_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    tiers: torch.Tensor,
) -> torch.Tensor:
    n_out = weight.shape[0]
    device = input_tensor.device
    bf16_mask = tiers == 2
    fp8_mask = tiers == 1
    nvfp4_mask = tiers == 0

    output = torch.zeros(
        input_tensor.shape[0], n_out, dtype=input_tensor.dtype, device=device
    )

    if bf16_mask.any():
        idx = bf16_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        out_sub = exact_eval.bf16_linear(input_tensor, w_sub)
        output.index_copy_(1, idx, out_sub)

    if fp8_mask.any():
        idx = fp8_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        n_rows = w_sub.shape[0]
        pad_n = (16 - n_rows % 16) % 16
        if pad_n:
            w_sub = F.pad(w_sub, (0, 0, 0, pad_n))
        out_sub = exact_eval.fp8_linear(input_tensor, w_sub)
        if pad_n:
            out_sub = out_sub[:, :n_rows]
        output.index_copy_(1, idx, out_sub)

    if nvfp4_mask.any():
        idx = nvfp4_mask.nonzero(as_tuple=True)[0].to(device)
        w_sub = weight[idx]
        n_rows = w_sub.shape[0]
        pad_n = (32 - n_rows % 32) % 32
        if pad_n:
            w_sub = F.pad(w_sub, (0, 0, 0, pad_n))
        out_sub = exact_eval.nvfp4_linear(input_tensor, w_sub)
        if pad_n:
            out_sub = out_sub[:, :n_rows]
        output.index_copy_(1, idx, out_sub)

    return output


def three_tier_moe_forward(
    hidden_states: torch.Tensor,
    tensors: dict[str, torch.Tensor],
    config: Any,
    w1_tiers: dict[int, torch.Tensor],
    w2_tiers: dict[int, torch.Tensor],
) -> torch.Tensor:
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    flat = hidden_states.view(-1, hidden_dim)

    router_logits = exact_eval.bf16_linear(flat, tensors["gate.weight"]).float()
    routing_weights = torch.softmax(router_logits, dim=1)
    routing_weights, selected_experts = torch.topk(
        routing_weights, config.num_experts_per_tok, dim=-1
    )
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)

    final_hidden_states = torch.zeros(
        (batch_size * sequence_length, hidden_dim),
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    gate_up_proj = tensors["experts.gate_up_proj"]
    down_proj = tensors["experts.down_proj"]
    expert_counts = torch.bincount(
        selected_experts.reshape(-1), minlength=config.num_experts
    )

    for expert_idx in torch.nonzero(expert_counts > 0, as_tuple=False).flatten().tolist():
        token_idx, route_pos = torch.where(selected_experts == expert_idx)
        current_state = flat[token_idx]

        w1_tier = w1_tiers.get(expert_idx)
        w2_tier = w2_tiers.get(expert_idx)

        if w1_tier is not None and (w1_tier > 0).any():
            gate_up = three_tier_linear(current_state, gate_up_proj[expert_idx], w1_tier)
        else:
            gate_up = exact_eval.nvfp4_linear(current_state, gate_up_proj[expert_idx])

        gate, up = gate_up.chunk(2, dim=-1)
        hidden = F.silu(gate) * up

        if w2_tier is not None and (w2_tier > 0).any():
            current_hidden = three_tier_linear(hidden, down_proj[expert_idx], w2_tier)
        else:
            current_hidden = exact_eval.nvfp4_linear(hidden, down_proj[expert_idx])

        current_hidden = current_hidden * routing_weights[token_idx, route_pos].unsqueeze(-1)
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(hidden_states.dtype))

    shared = exact_eval.bf16_linear(flat, tensors["shared_expert.gate_proj.weight"])
    shared = F.silu(shared) * exact_eval.bf16_linear(flat, tensors["shared_expert.up_proj.weight"])
    shared = exact_eval.bf16_linear(shared, tensors["shared_expert.down_proj.weight"])
    shared_gate = torch.sigmoid(exact_eval.bf16_linear(flat, tensors["shared_expert_gate.weight"]))
    final_hidden_states = final_hidden_states + shared_gate * shared
    return final_hidden_states.view(batch_size, sequence_length, hidden_dim)


def evaluate_three_tier_ppl(
    eval_ids: torch.Tensor,
    nsamples: int,
    seqlen: int,
    model_config: Any,
    weight_map: dict[str, str],
    snapshot_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    tier_config: ThreeTierConfig,
    layer_batch_size: int,
    profiling_data: dict | None = None,
    oracle_data: dict | None = None,
    oracle_curves: dict | None = None,
) -> float:
    store = exact_eval.WeightStore(exact_eval.MODEL_ID, snapshot_dir, weight_map)
    embed_key = "model.language_model.embed_tokens.weight"
    norm_key = "model.language_model.norm.weight"
    lm_head_key = "lm_head.weight"
    root_t = store.load_tensors([embed_key, norm_key, lm_head_key])
    embed_w = move_tensor(root_t[embed_key], device, dtype)
    final_norm_w = move_tensor(root_t[norm_key], device, dtype)
    lm_head_w = move_tensor(root_t[lm_head_key], device, dtype)
    del root_t

    eval_chunks = eval_ids[:, : nsamples * seqlen].view(nsamples, seqlen).contiguous()
    hidden_bank = torch.empty(
        (nsamples, seqlen, model_config.hidden_size), dtype=dtype, device="cpu"
    )
    for i in range(nsamples):
        hidden_bank[i].copy_(F.embedding(eval_chunks[i : i + 1].to(device), embed_w).squeeze(0).cpu())

    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    rotary = Qwen3NextRotaryEmbedding(config=model_config, device=device)
    rotary_input = torch.empty((1, seqlen, model_config.hidden_size), device=device, dtype=dtype)
    position_embeddings = rotary(rotary_input, position_ids)
    del rotary_input

    nlls: list[torch.Tensor] = []

    with torch.inference_mode():
        for layer_idx in range(model_config.num_hidden_layers):
            layer_type = model_config.layer_types[layer_idx]
            raw = store.load_tensors(layer_keys(layer_idx, layer_type))
            layer_tensors = shorten_layer_tensors(layer_idx, raw, device, dtype)
            del raw

            moe_t = {k.replace("mlp.", "", 1): v for k, v in layer_tensors.items() if k.startswith("mlp.")}
            
            # Compute per-layer BF16 fraction based on depth
            layer_bf16_fraction = _compute_layer_bf16_fraction(
                layer_idx, model_config.num_hidden_layers, tier_config.bf16_fraction
            )
            
            # Compute error concentration (Gini coefficient) for this layer
            if profiling_data and str(layer_idx) in profiling_data:
                layer_data = profiling_data[str(layer_idx)]
                w1_gini = sum(e['w1']['gini_unweighted'] for e in layer_data['experts']) / len(layer_data['experts'])
                w2_gini = sum(e['w2']['gini_unweighted'] for e in layer_data['experts']) / len(layer_data['experts'])
                layer_error_concentration = (w1_gini + w2_gini) / 2
                # Apply error-aware adjustment
                layer_bf16_fraction = _compute_error_aware_bf16_fraction(
                    layer_bf16_fraction, layer_error_concentration
                )
            layer_config = ThreeTierConfig(
                label=tier_config.label,
                bf16_fraction=layer_bf16_fraction,
                fp8_fraction=tier_config.fp8_fraction,
                nvfp4_fraction=tier_config.nvfp4_fraction,
            )
            
            
            # Load routing weights from profiling data
            layer_routing_weights = None
            if profiling_data and str(layer_idx) in profiling_data:
                layer_routing_weights = {
                    expert["expert"]: expert["routing_weight_sum"]
                    for expert in profiling_data[str(layer_idx)]["experts"]
                }
            
            w1_tiers, w2_tiers = build_three_tier_masks(
                moe_t.get("experts.gate_up_proj", torch.zeros(1)),
                moe_t.get("experts.down_proj", torch.zeros(1)),
                layer_config,
                model_config.num_experts,
                routing_weights=layer_routing_weights,
                oracle_data=oracle_data,
                profiling_data=profiling_data,
                layer_idx=layer_idx,
                oracle_curves=oracle_curves,
            )

            for sample_idx in range(nsamples):
                hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["input_layernorm.weight"], model_config.rms_norm_eps
                )
                if layer_type == "full_attention":
                    attn_t = {k.replace("self_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("self_attn.")}
                    hidden_states = exact_eval.full_attention_forward_exact(
                        hidden_states, attn_t, model_config, position_embeddings,
                        exact_eval.build_causal_mask(seqlen, device), mode="bf16", quantized=False,
                    )
                else:
                    attn_t = {k.replace("linear_attn.", ""): v for k, v in layer_tensors.items() if k.startswith("linear_attn.")}
                    hidden_states = exact_eval.linear_attention_forward_exact(
                        hidden_states, attn_t, model_config, "bf16", "moe_only"
                    )
                hidden_states = residual + hidden_states
                residual = hidden_states
                hidden_states = rms_norm_qwen3_next(
                    hidden_states, layer_tensors["post_attention_layernorm.weight"], model_config.rms_norm_eps
                )
                moe_out = three_tier_moe_forward(
                    hidden_states, moe_t, model_config, w1_tiers, w2_tiers,
                )
                hidden_states = residual + moe_out
                hidden_bank[sample_idx : sample_idx + 1].copy_(hidden_states.cpu())
                del hidden_states, residual, moe_out

            release_tensors(layer_tensors)
            if (layer_idx + 1) % 10 == 0:
                print(f"  [{tier_config.label}] layer {layer_idx + 1}/{model_config.num_hidden_layers}", flush=True)

        for sample_idx in range(nsamples):
            torch.cuda.empty_cache()
            chunk = eval_chunks[sample_idx : sample_idx + 1].to(device)
            hidden_states = hidden_bank[sample_idx : sample_idx + 1].to(device)
            hidden_states = rms_norm_qwen3_next(hidden_states, final_norm_w, model_config.rms_norm_eps)
            logits = F.linear(hidden_states, lm_head_w)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = chunk[:, 1:]
            loss = F.cross_entropy(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            nlls.append(loss.float() * seqlen)
            del logits, shift_logits, hidden_states

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen)).item()
    del embed_w, final_norm_w, lm_head_w, store
    torch.cuda.empty_cache()
    return ppl


def main() -> None:
    args = parse_args()
    exact_eval.ensure_runtime_available()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    eval_ids, full_nsamples, seqlen = exact_eval.load_eval_data(tokenizer, args.seqlen)
    nsamples = min(args.nsamples, full_nsamples)
    print(f"Eval: {nsamples} chunks of {seqlen} tokens", flush=True)

    snapshot_dir, root_config, weight_map = load_root_config(args.model_id)
    model_config = build_text_config(root_config)

    baselines = [
        exact_eval.EvalConfig("uniform_bf16", "bf16", "moe_only"),
        exact_eval.EvalConfig("uniform_fp8", "fp8", "moe_only"),
        exact_eval.EvalConfig("uniform_nvfp4", "nvfp4", "moe_only"),
    ]

    results: dict[str, dict[str, Any]] = {}

    # Load profiling data for routing-aware allocation
    profiling_path = SCRIPT_DIR / "profiling" / "error_profile.json"
    profiling_data = None
    if profiling_path.exists():
        with profiling_path.open("r") as f:
            profiling_data = json.load(f)
        print(f"Loaded profiling data from {profiling_path}", flush=True)
    else:
        print(f"Warning: Profiling data not found at {profiling_path}", flush=True)
    
    # Load oracle data for oracle-based channel selection
    oracle_path = SCRIPT_DIR / "profiling" / "oracle_data.json"
    oracle_data = None
    if oracle_path.exists():
        with oracle_path.open("r") as f:
            oracle_data = json.load(f)
        # Convert string keys back to integers
        oracle_data = {int(k): {int(ek): v for ek, v in experts.items()} for k, experts in oracle_data.items()}
        print(f"Loaded oracle data from {oracle_path}", flush=True)
    else:
        print(f"Warning: Oracle data not found at {oracle_path}", flush=True)
    
    # Load oracle_curves for cumulative curve optimization (Phase 6)
    oracle_curves_path = SCRIPT_DIR / "profiling" / "oracle_curves.json"
    oracle_curves = None
    if oracle_curves_path.exists():
        with oracle_curves_path.open("r") as f:
            oracle_curves_raw = json.load(f)
        # Convert string keys back to integers
        oracle_curves = {int(k): {int(ek): v for ek, v in experts.items()} for k, experts in oracle_curves_raw.items()}
        print(f"Loaded oracle_curves from {oracle_curves_path}", flush=True)
    else:
        print(f"Warning: Oracle curves not found at {oracle_curves_path}", flush=True)


    for run_config in baselines:
        print(f"\n=== {run_config.label} ===", flush=True)
        start_time = time.time()
        ppl = exact_eval.evaluate_ppl(
            eval_ids, nsamples, seqlen, model_config, weight_map, snapshot_dir,
            device, dtype, run_config, args.layer_batch_size,
        )
        elapsed = time.time() - start_time
        results[run_config.label] = {
            "tiers": run_config.mode,
            "bf16_pct": 1.0 if run_config.mode == "bf16" else 0.0,
            "fp8_pct": 1.0 if run_config.mode == "fp8" else 0.0,
            "nvfp4_pct": 1.0 if run_config.mode == "nvfp4" else 0.0,
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    for tier_config in ALL_CONFIGS:
        print(f"\n=== {tier_config.label} ({tier_config.description}) ===", flush=True)
        start_time = time.time()
        ppl = evaluate_three_tier_ppl(
            eval_ids, nsamples, seqlen, model_config, weight_map, snapshot_dir,
            device, dtype, tier_config, args.layer_batch_size,
            profiling_data=profiling_data,
            oracle_data=oracle_data,
            oracle_curves=oracle_curves,
        )
        elapsed = time.time() - start_time
        results[tier_config.label] = {
            "tiers": tier_config.description,
            "bf16_pct": tier_config.bf16_fraction,
            "fp8_pct": tier_config.fp8_fraction,
            "nvfp4_pct": tier_config.nvfp4_fraction,
            "ppl": round(ppl, 4),
            "time_s": round(elapsed, 1),
        }
        print(f"  -> PPL={ppl:.4f} ({elapsed:.0f}s)", flush=True)

    payload = {
        "metadata": {
            "model": args.model_id,
            "nsamples": nsamples,
            "seqlen": seqlen,
            "eval_tokens": nsamples * seqlen,
            "experiment": "three_tier_precision",
            "scope": "moe_only",
            "sensitivity_metric": "per-channel weight absmax (simple proxy)",
            "tiers": "BF16 (tier 2) > FP8 (tier 1) > NVFP4 (tier 0)",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    print(f"\nSaved -> {args.output}", flush=True)
    print(f"\n{'Label':<35s} {'BF16%':>6s} {'FP8%':>6s} {'FP4%':>6s} {'PPL':>10s}")
    print("-" * 70)
    for label, row in sorted(results.items(), key=lambda item: item[1]["ppl"]):
        print(
            f"{label:<35s} {row['bf16_pct']:>5.0%} {row['fp8_pct']:>5.0%} "
            f"{row['nvfp4_pct']:>5.0%} {row['ppl']:>10.4f}"
        )


if __name__ == "__main__":
    main()
