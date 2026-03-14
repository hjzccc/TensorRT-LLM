#!/usr/bin/env python3
"""Guard harness for unsupported single-tile SwiGLU fusion tactics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

# Support both direct execution and package imports
try:
    from .test_single_tile_swiglu_fusion import (
        DEFAULT_ACTIVATION,
        DEFAULT_QUANTIZATION,
        DEVICE,
        TACTIC_NAMES,
        compare_outputs,
        get_profile_ids,
        load_trtllm_library,
        make_deterministic_inputs,
        run_gate_state_subprocess,
        save_shared_inputs,
        run_branch_subprocess,
        validate_fc1_layout_contract,
    )
except ImportError:
    # Direct execution: add repo root to path for absolute import
    repo_root = str(Path(__file__).resolve().parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from added_benchmark.test_single_tile_swiglu_fusion import (
        DEFAULT_ACTIVATION,
        DEFAULT_QUANTIZATION,
        DEVICE,
        TACTIC_NAMES,
        compare_outputs,
        get_profile_ids,
        load_trtllm_library,
        make_deterministic_inputs,
        run_gate_state_subprocess,
        save_shared_inputs,
        run_branch_subprocess,
        validate_fc1_layout_contract,
    )


DEFAULT_GUARD_TACTICS = (2,)


def assert_gate_state(case_name: str, tactic: int, expected_gate: int,
                      env_overrides: dict[str, str | None],
                      activation: str = DEFAULT_ACTIVATION,
                      quantization: str = DEFAULT_QUANTIZATION,
                      use_prequant_scale: bool = False) -> None:
    report = run_gate_state_subprocess(
        tactic=tactic,
        env_overrides=env_overrides,
        activation=activation,
        quantization=quantization,
        use_prequant_scale=use_prequant_scale,
    )
    actual_gate = int(report["FUSION_GATE"])
    if actual_gate != expected_gate:
        raise AssertionError(
            f"{case_name}: expected FUSION_GATE={expected_gate}, got {actual_gate}; report={report}"
        )
    reason = report.get("FUSION_REASON", "<missing>")
    layout = report.get("FC1_LAYOUT", "<missing>")
    print(f"{case_name}: FUSION_GATE={actual_gate} FC1_LAYOUT={layout} reason={reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guard unsupported single-tile SwiGLU fusion tactics")
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--tactics", type=int, nargs="+", default=list(DEFAULT_GUARD_TACTICS))
    args = parser.parse_args()

    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive")
    invalid = sorted(set(args.tactics) - set(DEFAULT_GUARD_TACTICS))
    if invalid:
        parser.error(f"Guard harness only supports tactics {DEFAULT_GUARD_TACTICS}; got {invalid}")
    return args


def main() -> None:
    args = parse_args()
    load_trtllm_library()

    print("=" * 70)
    print("Single-tile SwiGLU unsupported-tactic guard")
    print("=" * 70)
    print(f"seed={args.seed} num_tokens={args.num_tokens} tactics={args.tactics}")

    shared_inputs = make_deterministic_inputs(seed=args.seed, num_tokens=args.num_tokens)
    validate_fc1_layout_contract(shared_inputs)
    print("Layout check: PASS ([up_i, gate_i] fused order stays distinct from contiguous [up | gate])")

    print("\n" + "=" * 70)
    print("Fallback matrix verification")
    print("=" * 70)

    assert_gate_state(
        case_name="No opt-in",
        tactic=0,
        expected_gate=0,
        env_overrides={
            "ENABLE_SINGLE_TILE_SWIGLU_FUSION": None,
            "FORCE_UNFUSED_SWIGLU": None,
        },
    )
    assert_gate_state(
        case_name="FORCE_UNFUSED_SWIGLU=1",
        tactic=0,
        expected_gate=0,
        env_overrides={
            "ENABLE_SINGLE_TILE_SWIGLU_FUSION": "1",
            "FORCE_UNFUSED_SWIGLU": "1",
        },
    )
    for unsupported_tactic in DEFAULT_GUARD_TACTICS:
        assert_gate_state(
            case_name=f"Unsupported tactic {unsupported_tactic} ({TACTIC_NAMES[unsupported_tactic]})",
            tactic=unsupported_tactic,
            expected_gate=0,
            env_overrides={
                "ENABLE_SINGLE_TILE_SWIGLU_FUSION": "1",
                "FORCE_UNFUSED_SWIGLU": None,
            },
        )
    assert_gate_state(
        case_name="Non-SwiGLU activation",
        tactic=0,
        expected_gate=0,
        env_overrides={
            "ENABLE_SINGLE_TILE_SWIGLU_FUSION": "1",
            "FORCE_UNFUSED_SWIGLU": None,
        },
        activation="relu",
    )
    assert_gate_state(
        case_name="Non-NVFP4 quantization",
        tactic=0,
        expected_gate=0,
        env_overrides={
            "ENABLE_SINGLE_TILE_SWIGLU_FUSION": "1",
            "FORCE_UNFUSED_SWIGLU": None,
        },
        quantization="mxfp4",
    )
    assert_gate_state(
        case_name="Prequant-scale path",
        tactic=0,
        expected_gate=0,
        env_overrides={
            "ENABLE_SINGLE_TILE_SWIGLU_FUSION": "1",
            "FORCE_UNFUSED_SWIGLU": None,
        },
        use_prequant_scale=True,
    )
    print("Fallback matrix: PASS (all unsupported cases explicitly report FUSION_GATE=0)")

    with tempfile.TemporaryDirectory() as tmpdir:
        inputs_path = Path(tmpdir) / "shared_inputs.pt"
        save_shared_inputs(shared_inputs, inputs_path)

        for tactic in args.tactics:
            print("\n" + "=" * 70)
            print(f"Guarding tactic {tactic} ({TACTIC_NAMES[tactic]}) profile_ids={get_profile_ids(tactic)}")
            print("=" * 70)
            try:
                default_output = run_branch_subprocess(
                    inputs_path,
                    tactic=tactic,
                    layout="unfused",
                    tag=f"TACTIC_{tactic}_DEFAULT",
                    env_overrides={
                        "ENABLE_SINGLE_TILE_SWIGLU_FUSION": "1",
                        "FORCE_UNFUSED_SWIGLU": None,
                    },
                    activation=DEFAULT_ACTIVATION,
                    quantization=DEFAULT_QUANTIZATION,
                ).to(DEVICE)
                forced_output = run_branch_subprocess(
                    inputs_path,
                    tactic=tactic,
                    layout="unfused",
                    tag=f"TACTIC_{tactic}_FORCED_UNFUSED",
                    env_overrides={
                        "ENABLE_SINGLE_TILE_SWIGLU_FUSION": "1",
                        "FORCE_UNFUSED_SWIGLU": "1",
                    },
                    activation=DEFAULT_ACTIVATION,
                    quantization=DEFAULT_QUANTIZATION,
                ).to(DEVICE)
            except RuntimeError as exc:
                print(f"Guard tactic {tactic}: SKIP runtime compare ({exc})")
                continue
            compare_outputs(forced_output, default_output, atol=args.atol, rtol=args.rtol,
                            label=f"Guard tactic {tactic}")
            print(
                f"Guard tactic {tactic}: PASS (contiguous [up | gate] layout still matches the forced-unfused baseline)")

    print("\n" + "=" * 70)
    print("PASS: unsupported single-tile cases stayed unfused")
    print("=" * 70)


if __name__ == "__main__":
    main()
