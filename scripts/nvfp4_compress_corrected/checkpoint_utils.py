# pyright: reportMissingImports=false

"""HF-compatible checkpoint I/O utilities.

This file rebuilds Hugging Face safetensors checkpoints while preserving the
original file layout exactly.

PIPELINE POSITION:
    BF16 -> NVFP4 -> sub-NVFP4 -> NVFP4 -> BF16
                                         ^^^^
                                         This file sits at the checkpoint I/O
                                         boundary and writes the final tensors
                                         back into a BF16-compatible HF layout.

The main design rule is that the rebuilt checkpoint must be structurally
indistinguishable from the original one:
    - same shard filenames
    - same tensor keys
    - same key-to-shard assignment
    - same copied non-weight files

That rule matters because downstream HF loading code expects the index and shard
layout to match. Changing the structure would make it harder to compare results
against the original checkpoint and would complicate evaluation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file, save_file


# -----------------------------------------------------------------------------
# Index helpers
# -----------------------------------------------------------------------------

def load_index(model_dir: Path) -> dict[str, str]:
    """Load the checkpoint index and return its ``weight_map``.

    What:
        Reads ``model.safetensors.index.json`` from a Hugging Face model
        directory and extracts the mapping from tensor name to shard filename.

    Why:
        The index is the source of truth for how tensors are distributed across
        shards. Rebuild logic should follow that map exactly rather than guessing
        the layout from filenames on disk.

    Math / logic:
        There is no tensor math here. The important invariant is that the return
        value is a dictionary

            tensor_name -> shard_filename

        and every later write decision follows that mapping.

    Why the branches exist:
        - Missing index file means the directory is not a complete sharded HF
          checkpoint, so execution stops immediately.
        - Missing or malformed ``weight_map`` means the index cannot be trusted,
          so execution stops before any output is written.

    Args:
        model_dir: Directory containing the HF checkpoint.

    Returns:
        Dictionary mapping tensor names to shard filenames.
    """
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")

    data = json.loads(index_path.read_text())
    wm = data.get("weight_map")
    if not isinstance(wm, dict):
        raise ValueError("Invalid index: missing weight_map")

    return wm


def get_shard_order(weight_map: dict[str, str]) -> list[str]:
    """Return unique shard filenames in first-seen order.

    What:
        Walks the ``weight_map`` values and returns each shard filename once, in
        the order in which it first appears.

    Why:
        Python dictionaries preserve insertion order. Reusing that order keeps the
        rebuild progress output stable and makes shard processing deterministic.

    Logic:
        This is an ordered deduplication pass:

            if shard not seen yet:
                append shard to output

    Args:
        weight_map: Mapping from tensor name to shard filename.

    Returns:
        Ordered list of unique shard filenames.
    """
    seen: set[str] = set()
    order: list[str] = []
    for shard in weight_map.values():
        if shard not in seen:
            seen.add(shard)
            order.append(shard)
    return order


# -----------------------------------------------------------------------------
# Checkpoint rebuild
# -----------------------------------------------------------------------------

def rebuild_checkpoint(
    original_dir: Path,
    output_dir: Path,
    transform_fn: Callable[[str, torch.Tensor], torch.Tensor | None],
    device: torch.device = torch.device("cpu"),
) -> None:
    """Rebuild a sharded HF checkpoint while transforming selected tensors.

    What:
        Reads each original shard, optionally replaces individual tensors via
        ``transform_fn``, writes a new shard with the same filename, and copies
        all non-safetensors files unchanged.

    Why:
        Research code often wants to modify only a subset of weights while still
        producing a checkpoint that standard HF loading code can open without any
        custom logic. Preserving the original shard structure makes that possible.

    Logic:
        For each tensor ``(name, tensor)`` in the original checkpoint:

            result = transform_fn(name, tensor.to(device))

        If ``result is None``:
            keep the original tensor as-is

        Else:
            validate result shape and NaN status
            write result back under the same tensor key

        The index layout is not recomputed. It is reused directly from the
        original checkpoint.

    Why the branches exist:
        - ``result is None`` means the caller chose not to transform that tensor.
        - Shape mismatch is rejected because changing tensor shape would break the
          checkpoint contract and likely the model architecture.
        - NaN values are rejected because they would silently poison evaluation.
        - Non-safetensors files are copied unchanged because configs, tokenizer
          files, and the index file belong to the HF artifact structure.

    Args:
        original_dir: Directory containing the source checkpoint.
        output_dir: Directory that will receive the rebuilt checkpoint.
        transform_fn: Callback of the form ``(tensor_name, tensor_value) -> new
            tensor or None``.
        device: Device used when calling ``transform_fn``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = load_index(original_dir)
    shard_order = get_shard_order(weight_map)

    # Invert the index so each shard knows which tensor keys should be present in
    # it. This avoids scanning every loaded shard for unrelated keys.
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    total_transformed = 0

    for si, shard_name in enumerate(shard_order):
        input_path = original_dir / shard_name
        output_path = output_dir / shard_name

        tensors = load_file(str(input_path), device="cpu")
        keys_in_shard = shard_to_keys.get(shard_name, [])

        shard_transformed = 0
        for key in keys_in_shard:
            if key not in tensors:
                # The index says the key should be here, but the code preserves the
                # original behavior and simply skips missing entries.
                continue

            original_tensor = tensors[key].to(device)
            result = transform_fn(key, original_tensor)
            if result is not None:
                if result.shape != original_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: "
                        f"original {original_tensor.shape} vs result {result.shape}"
                    )
                if result.isnan().any():
                    raise ValueError(f"NaN detected in transformed {key}")

                tensors[key] = result.to("cpu")
                shard_transformed += 1

        save_file(tensors, str(output_path))
        total_transformed += shard_transformed
        print(
            f"  [{si+1}/{len(shard_order)}] {shard_name}: "
            f"{shard_transformed}/{len(keys_in_shard)} tensors transformed"
        )

    # Copy files such as config.json, tokenizer files, and the JSON index without
    # modification so the rebuilt directory keeps the same HF metadata surface.
    for f in original_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, output_dir / f.name)

    print(f"Done. {total_transformed} tensors transformed across {len(shard_order)} shards.")


# -----------------------------------------------------------------------------
# Tensor-name filters
# -----------------------------------------------------------------------------

def is_expert_weight(name: str) -> bool:
    """Return ``True`` when a tensor name looks like an MoE expert weight.

    What:
        Matches Hugging Face-style tensor names for expert MLP weights.

    Why:
        Many research passes need to target only expert weights without touching
        shared attention or embedding tensors. A string filter is enough because
        shard rebuilding is driven by checkpoint key names.

    Logic:
        The current convention identifies expert weights by two conditions:
            1. the name contains ``.mlp.experts.``
            2. the name ends with ``.weight``

    Args:
        name: Tensor key from the checkpoint index.

    Returns:
        ``True`` if the name matches the expert-weight pattern.
    """
    return ".mlp.experts." in name and name.endswith(".weight")


def is_linear_weight(name: str) -> bool:
    """Return ``True`` when a tensor name likely belongs to an ``nn.Linear``.

    What:
        Uses checkpoint key naming patterns to include ordinary linear weights and
        exclude common non-linear-weight tensors such as norms and embeddings.

    Why:
        Quantization passes usually target linear layers only. Model checkpoints do
        not carry module class information in the key string, so the selection must
        be done by name heuristics.

    Logic:
        A tensor is considered a candidate linear weight only if:
            - its name ends with ``.weight``
            - its lowercase name does not contain any exclusion token from
              ``["layernorm", "norm", "embed", "lm_head"]``

    Why the branch exists:
        The early ``False`` for names not ending in ``.weight`` avoids unnecessary
        exclusion checks and keeps the intended rule explicit.

    Args:
        name: Tensor key from the checkpoint index.

    Returns:
        ``True`` if the name matches the linear-weight heuristic.
    """
    if not name.endswith(".weight"):
        return False

    exclude = ["layernorm", "norm", "embed", "lm_head"]
    return not any(ex in name.lower() for ex in exclude)
