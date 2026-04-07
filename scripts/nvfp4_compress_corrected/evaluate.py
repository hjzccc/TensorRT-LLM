"""TRT-LLM evaluation entrypoint for corrected NVFP4 experiments.

This evaluator stays entirely on the TRT-LLM PyTorch backend.  It does not
shell out to ``python -m tensorrt_llm.commands.eval`` because we need one extra
runtime step for the true MR-GPTQ path:

1. build TRT-LLM ``LLM`` with backend="pytorch"
2. inspect the model directory for ``manifest.json``
3. if the checkpoint is in rotated basis, patch every worker model so the
   selected linear modules rotate activations online before compute
4. run MMLU / GSM8K through TRT-LLM evaluators

This matches the real MR-GPTQ runtime pattern more closely than the old
"undo rotation at dequant time" workaround.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ["TRTLLM_ENABLE_PDL"] = "0"

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.evaluate.lm_eval import GSM8K
from tensorrt_llm.evaluate.mmlu import MMLU


def _ensure_base_worker_rotation_api() -> None:
    """Monkeypatch BaseWorker.setup_engine so workers auto-install Hadamard
    activation rotation after model loading, triggered by env vars."""
    import torch
    from tensorrt_llm.executor.base_worker import BaseWorker

    if hasattr(BaseWorker, "_nvfp4_patched"):
        return

    original_setup_engine = BaseWorker.setup_engine

    def patched_setup_engine(self):
        original_setup_engine(self)

        config_path = os.environ.get("NVFP4_HADAMARD_CONFIG")
        if not config_path:
            return

        config = json.loads(Path(config_path).read_text())
        group_size = config["group_size"]
        layer_names = config["layer_names"]

        from fast_hadamard_transform import hadamard_transform
        scale = 1.0 / math.sqrt(group_size)

        def rotate_activation(x: torch.Tensor) -> torch.Tensor:
            return hadamard_transform(
                x.reshape(-1, group_size), scale=scale
            ).reshape(x.shape)

        if not hasattr(self, "engine") or self.engine is None:
            return
        if not hasattr(self.engine, "model_engine"):
            return

        model = self.engine.model_engine.model
        module_map = dict(model.named_modules())
        patched = 0
        for name in layer_names:
            module = module_map.get(name)
            if module is None or getattr(module, "_nvfp4_hadamard_wrapped", False):
                continue
            orig_fwd = module.forward

            def make_wrapper(_orig_fwd=orig_fwd):
                def wrapped(*args, **kwargs):
                    if args and isinstance(args[0], torch.Tensor):
                        args = (rotate_activation(args[0]),) + args[1:]
                    return _orig_fwd(*args, **kwargs)
                return wrapped

            module.forward = make_wrapper()
            module._nvfp4_hadamard_wrapped = True
            patched += 1

        print(f"[Worker rank {getattr(self, 'rank', '?')}] "
              f"Hadamard activation rotation installed on {patched}/{len(layer_names)} modules")

    BaseWorker.setup_engine = patched_setup_engine
    BaseWorker._nvfp4_patched = True


def _load_rotation_spec(model_path: Path) -> dict[str, Any] | None:
    """Load runtime rotation metadata from the rebuilt checkpoint directory.

    Returns None when the checkpoint is already in original basis.
    """
    manifest_path = model_path / "manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text())
    hadamard_mode = manifest.get("hadamard_mode", "none")
    hadamard_group_size = manifest.get("hadamard_group_size", None)
    layer_keys = sorted(manifest.get("layers", {}).keys())

    if hadamard_mode != "rotated" or hadamard_group_size is None:
        return None

    # Manifest stores HF checkpoint tensor keys.  TRT-LLM runtime modules are
    # fused differently, so we map checkpoint names -> runtime module names.
    runtime_layer_names: set[str] = set()
    for key in layer_keys:
        name = key.removesuffix(".weight")
        if ".self_attn.q_proj" in name or ".self_attn.k_proj" in name or ".self_attn.v_proj" in name:
            runtime_layer_names.add(
                name.replace(".self_attn.q_proj", ".self_attn.qkv_proj")
                    .replace(".self_attn.k_proj", ".self_attn.qkv_proj")
                    .replace(".self_attn.v_proj", ".self_attn.qkv_proj"))
        elif ".self_attn.o_proj" in name:
            runtime_layer_names.add(name)
        elif ".mlp.experts." in name or ".mlp.gate" in name:
            # TRT-LLM fuses MoE internals. The clean Python hook point is the
            # Qwen3MoE module itself, which receives the hidden_states used by
            # both the gate and the expert up/gate projection path.
            prefix = name.split(".mlp.", 1)[0]
            runtime_layer_names.add(f"{prefix}.mlp")

    return {
        "group_size": int(hadamard_group_size),
        "layer_names": sorted(runtime_layer_names),
        "manifest_path": str(manifest_path),
    }


def _install_runtime_rotation(rotation_spec: dict[str, Any]) -> Path:
    """Write rotation config to a temp file and set the env var that workers read."""
    config_path = Path("/tmp/nvfp4_hadamard_config.json")
    config_path.write_text(json.dumps(rotation_spec))
    os.environ["NVFP4_HADAMARD_CONFIG"] = str(config_path)
    return config_path


def _remove_runtime_rotation() -> None:
    os.environ.pop("NVFP4_HADAMARD_CONFIG", None)


def _run_task(llm: LLM, task: str, output_dir: Path | None,
              num_samples: int | None) -> float:
    """Run one TRT-LLM evaluator and return its score."""
    if task == "mmlu":
        evaluator = MMLU(num_fewshot=5,
                         num_samples=num_samples)
        sampling_params = SamplingParams(max_tokens=8)
    elif task == "gsm8k":
        evaluator = GSM8K(num_samples=num_samples,
                          output_path=str(output_dir) if output_dir else None)
        sampling_params = SamplingParams(max_tokens=256)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return evaluator.evaluate(llm, sampling_params)


def main() -> None:
    parser = argparse.ArgumentParser(description="TRT-LLM evaluation")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", default=["mmlu", "gsm8k"])
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Optional subset size for a fast smoke test")
    parser.add_argument("--output-path",
                        type=str,
                        default=None,
                        help="Directory to save result JSONs")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    output_dir = Path(args.output_path) if args.output_path else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    _ensure_base_worker_rotation_api()

    llm = LLM(model=str(model_path),
              backend="pytorch",
              tensor_parallel_size=args.tp_size,
              trust_remote_code=True)

    runtime_rotation = _load_rotation_spec(model_path)
    if runtime_rotation is not None:
        print(
            f"Hadamard runtime rotation config: group_size={runtime_rotation['group_size']}, "
            f"layers={len(runtime_rotation['layer_names'])}")
        _install_runtime_rotation(runtime_rotation)
    else:
        print("No runtime activation rotation needed for this checkpoint")

    results: dict[str, Any] = {}
    try:
        for task in args.tasks:
            print(f"\n{'=' * 60}")
            print(f"Evaluating: {task}")
            print(f"{'=' * 60}")

            t0 = time.time()
            task_output_dir = output_dir / task if output_dir else None
            if task_output_dir:
                task_output_dir.mkdir(parents=True, exist_ok=True)
            score = _run_task(llm, task, task_output_dir, args.num_samples)
            elapsed = time.time() - t0

            results[task] = {
                "score": score,
                "elapsed_seconds": elapsed,
                "runtime_rotation": runtime_rotation is not None,
            }
            print(f"{task}: {score:.2f} (elapsed {elapsed:.1f}s)")

            if output_dir:
                (output_dir / f"{task}.json").write_text(
                    json.dumps(results[task], indent=2))
    finally:
        if runtime_rotation is not None:
            _remove_runtime_rotation()
        llm.shutdown()

    if output_dir:
        (output_dir / "all_results.json").write_text(
            json.dumps(results, indent=2))
        print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
