#!/usr/bin/env python3
"""Compare fused vs unfused nsys gpukernsum CSV outputs for single-tile SwiGLU."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import TypedDict


ACTIVATION_KERNEL_RE = re.compile(r"do(gated)?activationkernel", re.IGNORECASE)
GEMM_KERNEL_RE = re.compile(r"gemmuniversal|groupedgemm|\bgemm\b", re.IGNORECASE)


KernelRow = dict[str, str]


class KernelSummary(TypedDict):
    activation_rows: list[KernelRow]
    gemm_rows: list[KernelRow]
    activation_instances: int
    gemm_instances: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fused/unfused single-tile SwiGLU profile CSVs")
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--unfused", type=Path, required=True)
    return parser.parse_args()


def _parse_int(value: str) -> int:
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return 0
    return int(float(cleaned))


def load_gpukernsum_rows(path: Path) -> list[KernelRow]:
    if not path.exists():
        raise FileNotFoundError(f"Missing gpukernsum CSV: {path}")

    header = None
    rows: list[KernelRow] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            normalized = [cell.strip() for cell in row]
            if not normalized or all(not cell for cell in normalized):
                continue
            if "Name" in normalized:
                header = normalized
                continue
            if header is None:
                continue
            if normalized[0].startswith("=="):
                continue
            if len(normalized) < len(header):
                normalized.extend([""] * (len(header) - len(normalized)))
            if len(normalized) > len(header):
                normalized = normalized[:len(header)]
            rows.append(dict(zip(header, normalized)))

    if header is None:
        raise RuntimeError(f"Could not find a gpukernsum header row in {path}")
    return rows


def get_name(row: KernelRow) -> str:
    return row.get("Name", "")


def get_instances(row: KernelRow) -> int:
    for key in ("Instances", "Calls", "Count"):
        if key in row:
            return _parse_int(row[key])
    return 0


def summarize(rows: list[KernelRow]) -> KernelSummary:
    activation_rows = [row for row in rows if ACTIVATION_KERNEL_RE.search(get_name(row))]
    gemm_rows = [row for row in rows if GEMM_KERNEL_RE.search(get_name(row))]
    return {
        "activation_rows": activation_rows,
        "gemm_rows": gemm_rows,
        "activation_instances": sum(get_instances(row) for row in activation_rows),
        "gemm_instances": sum(get_instances(row) for row in gemm_rows),
    }


def format_names(rows: list[KernelRow]) -> str:
    names = sorted({get_name(row) for row in rows if get_name(row)})
    return "; ".join(names) if names else "<none>"


def main() -> None:
    args = parse_args()
    fused_rows = load_gpukernsum_rows(args.fused)
    unfused_rows = load_gpukernsum_rows(args.unfused)
    fused = summarize(fused_rows)
    unfused = summarize(unfused_rows)

    print(f"Fused activation kernel instances: {fused['activation_instances']}")
    print(f"Fused activation kernels: {format_names(fused['activation_rows'])}")
    print(f"Unfused activation kernel instances: {unfused['activation_instances']}")
    print(f"Unfused activation kernels: {format_names(unfused['activation_rows'])}")
    print(f"Fused GEMM kernel instances: {fused['gemm_instances']}")
    print(f"Fused GEMM kernels: {format_names(fused['gemm_rows'])}")
    print(f"Unfused GEMM kernel instances: {unfused['gemm_instances']}")
    print(f"Unfused GEMM kernels: {format_names(unfused['gemm_rows'])}")

    failures: list[str] = []
    if fused["activation_instances"] != 0:
        failures.append("fused run still contains a standalone activation kernel")
    if unfused["activation_instances"] == 0:
        failures.append("unfused run is missing the standalone activation kernel")
    if fused["gemm_instances"] == 0:
        failures.append("fused run is missing GEMM kernels")
    if unfused["gemm_instances"] == 0:
        failures.append("unfused run is missing GEMM kernels")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print("PASS: fused run removed standalone gated activation kernel")


if __name__ == "__main__":
    main()
