#!/usr/bin/env python3
"""Deterministic single-tile SwiGLU fusion A/B harness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
LIB_CANDIDATES = (
    ROOT_DIR / "tensorrt_llm" / "libs" / "libth_common.so",
    Path("/code/tensorrt_llm/tensorrt_llm/libs/libth_common.so"),
)


def load_trtllm_library() -> None:
    for candidate in LIB_CANDIDATES:
        if candidate.exists():
            torch.ops.load_library(str(candidate))
            return
    raise FileNotFoundError(
        f"Could not find libth_common.so in: {', '.join(str(path) for path in LIB_CANDIDATES)}"
    )


DTYPE = torch.bfloat16
DEVICE = "cuda"
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN = 2048
INTER = 768
FC1_ROWS = INTER * 2
SWIGLU = 5
NUM_BASE_TACTICS = 4
INTERLEAVE_GROUP_SIZE = 64
SUPPORTED_FUSION_TACTICS = {0, 1, 2}
ACTIVATION_CHOICES = (
    "identity",
    "gelu",
    "relu",
    "silu",
    "swiglu",
    "geglu",
    "swiglu_bias",
    "relu2",
)
QUANTIZATION_CHOICES = ("nvfp4", "mxfp4", "bf16")
DEFAULT_ACTIVATION = "swiglu"
DEFAULT_QUANTIZATION = "nvfp4"
TACTIC_NAMES = {
    0: "M128",
    1: "M64",
    2: "M32",
    3: "M256",
}


def interleave_linear_and_gate(x: torch.Tensor,
                               group_size: int = INTERLEAVE_GROUP_SIZE,
                               dim: int = -1) -> torch.Tensor:
    sizes = x.size()
    dim = dim % x.dim()
    if sizes[dim] % (group_size * 2) != 0:
        raise ValueError(
            f"Dimension {dim} with size {sizes[dim]} is not divisible by {group_size * 2}"
        )
    prev_sizes = sizes[:dim]
    post_sizes = sizes[dim + 1:]
    x = x.view(*prev_sizes, 2, sizes[dim] // (group_size * 2), group_size,
               *post_sizes)
    x = x.transpose(dim, dim + 1).contiguous().view(*sizes)
    return x


def get_profile_ids(tactic: int) -> list[int]:
    if tactic not in TACTIC_NAMES:
        raise ValueError(f"Unsupported tactic {tactic}; expected one of {sorted(TACTIC_NAMES)}")
    if tactic == 3:
        return [3, 4]
    gemm1 = tactic
    gemm2 = tactic + NUM_BASE_TACTICS
    return [gemm1, gemm2]


def get_bool_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def compute_fusion_gate_report(tactic: int, activation: str = DEFAULT_ACTIVATION,
                               quantization: str = DEFAULT_QUANTIZATION,
                               use_prequant_scale: bool = False) -> dict[str, str | bool]:
    reasons = []
    if not get_bool_env("ENABLE_SINGLE_TILE_SWIGLU_FUSION"):
        reasons.append("ENABLE_SINGLE_TILE_SWIGLU_FUSION=0")
    if get_bool_env("FORCE_UNFUSED_SWIGLU"):
        reasons.append("FORCE_UNFUSED_SWIGLU=1")
    if activation != DEFAULT_ACTIVATION:
        reasons.append(f"activation={activation}")
    if quantization != DEFAULT_QUANTIZATION:
        reasons.append(f"quantization={quantization}")
    if use_prequant_scale:
        reasons.append("prequant_scale_kernel=1")
    if tactic not in SUPPORTED_FUSION_TACTICS:
        reasons.append(f"tactic={TACTIC_NAMES[tactic]}")

    enabled = not reasons
    reason = "supported_single_tile_swiglu_nvfp4" if enabled else ", ".join(reasons)
    return {"enabled": enabled, "reason": reason}


def compute_fusion_gate_state(tactic: int, activation: str = DEFAULT_ACTIVATION,
                              quantization: str = DEFAULT_QUANTIZATION,
                              use_prequant_scale: bool = False) -> bool:
    return bool(compute_fusion_gate_report(tactic, activation, quantization, use_prequant_scale)["enabled"])


def resolve_fc1_layout(tactic: int, activation: str = DEFAULT_ACTIVATION,
                       quantization: str = DEFAULT_QUANTIZATION,
                       use_prequant_scale: bool = False) -> str:
    return "fused" if compute_fusion_gate_state(tactic, activation, quantization, use_prequant_scale) else "unfused"


def validate_runtime_config(activation: str, quantization: str, use_prequant_scale: bool) -> None:
    problems = []
    if activation != DEFAULT_ACTIVATION:
        problems.append(f"activation={activation}")
    if quantization != DEFAULT_QUANTIZATION:
        problems.append(f"quantization={quantization}")
    if use_prequant_scale:
        problems.append("prequant_scale_kernel=1")
    if problems:
        joined = ", ".join(problems)
        raise RuntimeError(
            "Kernel execution mode only supports the deterministic SWIGLU/NVFP4 non-prequant path; "
            + f"got {joined}. Use --print-gate-state to inspect unsupported fallback cases."
        )


def quantize_for_trtllm(weight_bf16: torch.Tensor,
                        global_sf: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight_bf16.shape
    global_sf_tensor = torch.tensor(global_sf, dtype=torch.float32, device=weight_bf16.device)
    packed_u8, sf_u8 = torch.ops.trtllm.fp4_quantize(weight_bf16, global_sf_tensor, 16, False,
                                                     False)
    weight_int64 = packed_u8.view(torch.int64).reshape(rows, cols // 16)
    scale_u8 = sf_u8.reshape(rows, cols // 16)
    scale_interleaved = torch.ops.trtllm.block_scale_interleave(scale_u8)
    scale_int32 = scale_interleaved.view(torch.int32).reshape(rows, cols // 64)
    return weight_int64, scale_int32


def unswizzle_scale_int32(scale_int32: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    scale_u8 = scale_int32.view(torch.uint8).reshape(rows, cols // 16)
    return torch.ops.trtllm.block_scale_interleave_reverse(scale_u8.view(1, rows, cols // 16)).view(rows, cols // 16)


def swizzle_scale_u8(scale_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    scale_interleaved = torch.ops.trtllm.block_scale_interleave(scale_u8.view(1, rows, cols // 16))
    return scale_interleaved.view(torch.int32).reshape(rows, cols // 64)


def materialize_unfused_fc1(fc1_up: torch.Tensor, fc1_gate: torch.Tensor) -> torch.Tensor:
    return torch.cat([fc1_up, fc1_gate], dim=-2).contiguous()


def materialize_fused_fc1(fc1_up: torch.Tensor, fc1_gate: torch.Tensor) -> torch.Tensor:
    contiguous = materialize_unfused_fc1(fc1_up, fc1_gate)
    return interleave_linear_and_gate(contiguous, group_size=INTERLEAVE_GROUP_SIZE, dim=-2)


def quantize_experts(weight_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed_weights = []
    packed_scales = []
    for expert_id in range(weight_bf16.shape[0]):
        weight_int64, scale_int32 = quantize_for_trtllm(weight_bf16[expert_id])
        packed_weights.append(weight_int64)
        packed_scales.append(scale_int32)
    return torch.stack(packed_weights), torch.stack(packed_scales)


def make_deterministic_inputs(seed: int, num_tokens: int) -> dict[str, Any]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    token_offsets = torch.arange(num_tokens, device=DEVICE, dtype=torch.int32).unsqueeze(1) * TOP_K
    topk_offsets = torch.arange(TOP_K, device=DEVICE, dtype=torch.int32).unsqueeze(0)
    topk_idx = (token_offsets + topk_offsets) % NUM_EXPERTS

    topk_scores = torch.linspace(1.0, 1.0 + 0.05 * (TOP_K - 1), TOP_K, device=DEVICE,
                                 dtype=torch.float32)
    topk_scores = topk_scores.repeat(num_tokens, 1)
    topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

    input_tensor = torch.randn(num_tokens, HIDDEN, generator=generator, device=DEVICE,
                               dtype=DTYPE) * 0.25
    fc1_up = torch.randn(NUM_EXPERTS, INTER, HIDDEN, generator=generator, device=DEVICE,
                         dtype=DTYPE) * 0.0125
    fc1_gate = torch.randn(NUM_EXPERTS, INTER, HIDDEN, generator=generator, device=DEVICE,
                           dtype=DTYPE) * 0.0225
    fc2_bf16 = torch.randn(NUM_EXPERTS, HIDDEN, INTER, generator=generator, device=DEVICE,
                           dtype=DTYPE) * 0.0125

    return {
        "seed": seed,
        "num_tokens": num_tokens,
        "input_tensor": input_tensor,
        "topk_idx": topk_idx.int(),
        "topk_scores": topk_scores,
        "fc1_up": fc1_up,
        "fc1_gate": fc1_gate,
        "fc2_bf16": fc2_bf16,
    }


def save_shared_inputs(shared_inputs: dict[str, Any], path: Path) -> None:
    cpu_payload = {}
    for key, value in shared_inputs.items():
        cpu_payload[key] = value.cpu() if isinstance(value, torch.Tensor) else value
    torch.save(cpu_payload, path)


def load_shared_inputs(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    for key, value in list(payload.items()):
        if isinstance(value, torch.Tensor):
            payload[key] = value.to(DEVICE)
    return payload


def validate_fc1_layout_contract(shared_inputs: dict[str, Any]) -> None:
    sample_up = shared_inputs["fc1_up"][0]
    sample_gate = shared_inputs["fc1_gate"][0]

    contiguous_bf16 = materialize_unfused_fc1(sample_up, sample_gate)
    fused_bf16 = materialize_fused_fc1(sample_up, sample_gate)
    contiguous_weight, contiguous_scale = quantize_for_trtllm(contiguous_bf16)
    fused_weight, fused_scale = quantize_for_trtllm(fused_bf16)

    expected_weight = interleave_linear_and_gate(contiguous_weight, group_size=INTERLEAVE_GROUP_SIZE,
                                                 dim=0)
    contiguous_scale_unswizzled = unswizzle_scale_int32(contiguous_scale, FC1_ROWS, HIDDEN)
    expected_scale_unswizzled = interleave_linear_and_gate(contiguous_scale_unswizzled,
                                                           group_size=INTERLEAVE_GROUP_SIZE, dim=0)
    expected_scale = swizzle_scale_u8(expected_scale_unswizzled, FC1_ROWS, HIDDEN)
    if not torch.equal(fused_weight, expected_weight):
        raise AssertionError("FC1 fused packed weights do not follow the expected [up_i, gate_i] order")
    if not torch.equal(fused_scale, expected_scale):
        raise AssertionError("FC1 fused block scales do not follow the same permutation as the weights")

    row_ids = torch.arange(FC1_ROWS, device=DEVICE, dtype=torch.int32).view(FC1_ROWS, 1)
    fused_row_ids = interleave_linear_and_gate(row_ids, group_size=INTERLEAVE_GROUP_SIZE, dim=0).flatten()
    num_groups = INTER // INTERLEAVE_GROUP_SIZE
    for group_id in range(num_groups):
        start = group_id * 2 * INTERLEAVE_GROUP_SIZE
        up_base = group_id * INTERLEAVE_GROUP_SIZE
        gate_base = INTER + group_id * INTERLEAVE_GROUP_SIZE
        expected_up = torch.arange(up_base, up_base + INTERLEAVE_GROUP_SIZE, device=DEVICE,
                                   dtype=torch.int32)
        expected_gate = torch.arange(gate_base, gate_base + INTERLEAVE_GROUP_SIZE, device=DEVICE,
                                     dtype=torch.int32)
        if not torch.equal(fused_row_ids[start:start + INTERLEAVE_GROUP_SIZE], expected_up):
            raise AssertionError("FC1 fused layout does not place the up chunk before the gate chunk")
        if not torch.equal(
                fused_row_ids[start + INTERLEAVE_GROUP_SIZE:start + 2 * INTERLEAVE_GROUP_SIZE],
                expected_gate):
            raise AssertionError("FC1 fused layout does not preserve the [up_i, gate_i] chunk pattern")


def make_quant_scales(fc1_scale: torch.Tensor,
                      fc2_scale: torch.Tensor) -> list[torch.Tensor]:
    fc1_act_global = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    fc1_global = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
    fc2_act_global = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    fc2_global = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=DEVICE)
    return [fc1_act_global, fc1_scale, fc1_global, fc2_act_global, fc2_scale, fc2_global]


def build_single_tile_case(shared_inputs: dict[str, Any], tactic: int, layout: str) -> dict[str, Any]:
    if layout == "fused":
        fc1_bf16 = materialize_fused_fc1(shared_inputs["fc1_up"], shared_inputs["fc1_gate"])
    elif layout == "unfused":
        fc1_bf16 = materialize_unfused_fc1(shared_inputs["fc1_up"], shared_inputs["fc1_gate"])
    else:
        raise ValueError(f"Unsupported layout '{layout}'")

    fc1_weight, fc1_scale = quantize_experts(fc1_bf16)
    fc2_weight, fc2_scale = quantize_experts(shared_inputs["fc2_bf16"])
    quant_scales = make_quant_scales(fc1_scale, fc2_scale)
    input_scale = torch.ones(shared_inputs["num_tokens"], HIDDEN // 16,
                             dtype=torch.float8_e4m3fn, device=DEVICE)
    return {
        "fc1_weight": fc1_weight,
        "fc2_weight": fc2_weight,
        "quant_scales": quant_scales,
        "input_scale": input_scale,
        "profile_ids": get_profile_ids(tactic),
    }


def create_runner() -> Any:
    return torch.classes.trtllm.FusedMoeRunner(
        DTYPE, torch.int64, DTYPE, False, False, False, False, True)


def call_run_moe(runner: Any, shared_inputs: dict[str, Any], case: dict[str, Any]) -> torch.Tensor:
    return runner.run_moe(
        shared_inputs["input_tensor"],
        shared_inputs["topk_idx"],
        shared_inputs["topk_scores"],
        case["fc1_weight"],
        None,
        case["fc2_weight"],
        None,
        case["quant_scales"],
        case["input_scale"],
        False,
        None,
        None,
        None,
        1,
        0,
        1,
        0,
        1,
        0,
        False,
        False,
        case["profile_ids"],
        SWIGLU,
        None,
        None,
        None,
    )


def summarize_output(tag: str, output: torch.Tensor) -> None:
    head = output[0, :8].float().tolist() if output.shape[0] > 0 else []
    print(f"[{tag}] shape={tuple(output.shape)}")
    print(f"[{tag}] abs_mean={output.abs().mean().item():.6f}")
    print(f"[{tag}] abs_max={output.abs().max().item():.6f}")
    print(f"[{tag}] head={head}")


def run_single_tile(shared_inputs: dict[str, Any], tactic: int, layout: str,
                    tag: str) -> torch.Tensor:
    case = build_single_tile_case(shared_inputs, tactic, layout)
    runner = create_runner()
    output = call_run_moe(runner, shared_inputs, case)
    torch.cuda.synchronize()

    env_flags = {
        "ENABLE_SINGLE_TILE_SWIGLU_FUSION": os.environ.get("ENABLE_SINGLE_TILE_SWIGLU_FUSION", "0"),
        "FORCE_UNFUSED_SWIGLU": os.environ.get("FORCE_UNFUSED_SWIGLU", "0"),
    }
    print(f"[{tag}] tactic={tactic} ({TACTIC_NAMES[tactic]}) profile_ids={case['profile_ids']} layout={layout}")
    print(f"[{tag}] env={env_flags}")
    summarize_output(tag, output)
    return output


def print_filtered_stderr(stderr: str) -> None:
    lines = []
    for line in stderr.splitlines():
        if not line:
            continue
        if "FutureWarning" in line:
            continue
        if "pynvml" in line:
            continue
        lines.append(line)
    if lines:
        print("STDERR:")
        for line in lines[:20]:
            print(line)


def run_branch_subprocess(inputs_path: Path, tactic: int, layout: str,
                          tag: str, env_overrides: dict[str, str | None],
                          activation: str = DEFAULT_ACTIVATION,
                          quantization: str = DEFAULT_QUANTIZATION,
                          use_prequant_scale: bool = False) -> torch.Tensor:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / f"{tag}.pt"
        env = os.environ.copy()
        for key, value in env_overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-branch",
            "--inputs-path",
            str(inputs_path),
            "--output-path",
            str(output_path),
            "--tactic",
            str(tactic),
            "--layout",
            layout,
            "--tag",
            tag,
            "--activation",
            activation,
            "--quantization",
            quantization,
        ]
        if use_prequant_scale:
            command.append("--use-prequant-scale")
        result = subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print_filtered_stderr(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Subprocess '{tag}' failed with code {result.returncode}")
        return torch.load(output_path, map_location=DEVICE)


def run_gate_state_subprocess(tactic: int, env_overrides: dict[str, str | None],
                              activation: str = DEFAULT_ACTIVATION,
                              quantization: str = DEFAULT_QUANTIZATION,
                              use_prequant_scale: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--tactic",
        str(tactic),
        "--activation",
        activation,
        "--quantization",
        quantization,
        "--print-gate-state",
    ]
    if use_prequant_scale:
        command.append("--use-prequant-scale")

    result = subprocess.run(command, cwd=ROOT_DIR, env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print_filtered_stderr(result.stderr)
        raise RuntimeError(f"Gate-state subprocess failed with code {result.returncode}")

    report = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        report[key.strip()] = value.strip()
    if "FUSION_GATE" not in report:
        raise RuntimeError("Gate-state subprocess did not print FUSION_GATE")
    return report


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float,
                    label: str) -> None:
    diff = (candidate.float() - reference.float()).abs()
    rel_diff = diff / (reference.float().abs() + 1e-6)
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"{label}: max_abs_diff={max_abs_diff:.8f}")
    print(f"{label}: mean_abs_diff={mean_abs_diff:.8f}")
    print(f"{label}: max_rel_diff={max_rel_diff:.8f}")
    print(f"{label}: mean_rel_diff={mean_rel_diff:.8f}")

    if torch.allclose(candidate, reference, atol=atol, rtol=rtol):
        print(f"{label}: PASS (atol={atol}, rtol={rtol})")
        return

    within_tol = ((diff <= atol) | (rel_diff <= rtol)).float().mean().item()
    print(f"{label}: within_tol={within_tol * 100.0:.2f}%")
    flat_diff = diff.flatten()
    worst_indices = flat_diff.topk(min(5, flat_diff.numel())).indices
    for flat_index in worst_indices:
        flat_idx = int(flat_index.item())
        row, col = divmod(flat_idx, int(candidate.shape[1]))
        print(
            f"{label}: mismatch[{row},{col}] fused={candidate[row, col].item():.6f} "
            + f"unfused={reference[row, col].item():.6f} diff={flat_diff[flat_index].item():.6f}")
    raise AssertionError(f"{label}: outputs diverged beyond tolerance")


def branch_main(args: argparse.Namespace) -> None:
    validate_runtime_config(args.activation, args.quantization, args.use_prequant_scale)
    load_trtllm_library()
    shared_inputs = load_shared_inputs(Path(args.inputs_path))
    output = run_single_tile(shared_inputs, args.tactic, args.layout, args.tag)
    torch.save(output.cpu(), args.output_path)


def harness_main(args: argparse.Namespace) -> None:
    validate_runtime_config(args.activation, args.quantization, args.use_prequant_scale)
    gate_report = compute_fusion_gate_report(
        args.tactic,
        activation=args.activation,
        quantization=args.quantization,
        use_prequant_scale=args.use_prequant_scale,
    )
    if not gate_report["enabled"]:
        raise RuntimeError(
            "Harness execution requires the supported fused gate. Use `--print-gate-state` for unsupported cases. "
            + f"Current gate reason: {gate_report['reason']}"
        )

    load_trtllm_library()
    print("=" * 70)
    print("Single-tile SwiGLU fusion harness")
    print("=" * 70)
    print(f"seed={args.seed} num_tokens={args.num_tokens} tactic={args.tactic} ({TACTIC_NAMES[args.tactic]})")
    print(f"profile_ids={get_profile_ids(args.tactic)}")
    print("FUSION_GATE=1")
    print(f"FUSION_REASON={gate_report['reason']}")

    shared_inputs = make_deterministic_inputs(seed=args.seed, num_tokens=args.num_tokens)
    validate_fc1_layout_contract(shared_inputs)
    print("Layout check: PASS ([up_i, gate_i] chunk order and matching scale permutation)")

    with tempfile.TemporaryDirectory() as tmpdir:
        inputs_path = Path(tmpdir) / "shared_inputs.pt"
        save_shared_inputs(shared_inputs, inputs_path)

        print("\n" + "=" * 70)
        print("Step 1: Running fused/default branch")
        print("=" * 70)
        if os.environ.get("FORCE_UNFUSED_SWIGLU"):
            raise RuntimeError("FORCE_UNFUSED_SWIGLU is set in the parent environment; clear it for the fused/default branch")
        output_fused = run_single_tile(shared_inputs, args.tactic, "fused", "FUSED")

        print("\n" + "=" * 70)
        print("Step 2: Running forced-unfused branch in a subprocess")
        print("=" * 70)
        output_unfused = run_branch_subprocess(
            inputs_path,
            tactic=args.tactic,
            layout="unfused",
            tag="UNFUSED",
            env_overrides={
                "FORCE_UNFUSED_SWIGLU": "1",
            },
            activation=args.activation,
            quantization=args.quantization,
            use_prequant_scale=args.use_prequant_scale,
        ).to(DEVICE)

    print("\n" + "=" * 70)
    print("Step 3: Comparing fused vs unfused")
    print("=" * 70)
    compare_outputs(output_unfused, output_fused, atol=args.atol, rtol=args.rtol,
                    label="Fused vs unfused")
    print("PASS: single-tile fused matches unfused within tolerance")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic single-tile SwiGLU fusion harness")
    parser.add_argument("--tactic", type=int, default=0, choices=sorted(TACTIC_NAMES))
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.15)
    parser.add_argument("--run-branch", action="store_true")
    parser.add_argument("--inputs-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--layout", choices=("fused", "unfused"))
    parser.add_argument("--tag", type=str, default="SUBPROCESS")
    parser.add_argument("--activation", type=str, default=DEFAULT_ACTIVATION,
                        choices=ACTIVATION_CHOICES)
    parser.add_argument("--quantization", type=str, default=DEFAULT_QUANTIZATION,
                        choices=QUANTIZATION_CHOICES)
    parser.add_argument("--use-prequant-scale", action="store_true")
    parser.add_argument("--print-gate-state", action="store_true",
                        help="Print fusion gate state and exit")
    args = parser.parse_args()

    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive")
    if args.run_branch:
        if not args.inputs_path or not args.output_path or not args.layout:
            parser.error("--run-branch requires --inputs-path, --output-path, and --layout")
    return args


def main() -> None:
    args = parse_args()
    
    # Handle --print-gate-state early, before any library loading
    if args.print_gate_state:
        gate_report = compute_fusion_gate_report(
            args.tactic,
            activation=args.activation,
            quantization=args.quantization,
            use_prequant_scale=args.use_prequant_scale,
        )
        gate_state = bool(gate_report["enabled"])
        gate_value = 1 if gate_state else 0
        print(f"FUSION_GATE={gate_value}")
        print(f"FUSION_REASON={gate_report['reason']}")
        print(
            f"FC1_LAYOUT={resolve_fc1_layout(args.tactic, args.activation, args.quantization, args.use_prequant_scale)}"
        )
        return
    
    if args.run_branch:
        branch_main(args)
    else:
        harness_main(args)


if __name__ == "__main__":
    main()
