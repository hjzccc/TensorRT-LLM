#!/usr/bin/env python3
"""Categorize SASS instructions from nvdisasm -g output and map to source lines."""
import re
import sys
from collections import defaultdict

categories = {
    "OMMA": re.compile(r"OMMA"),
    "SYNCS.PHASECHK": re.compile(r"SYNCS\.PHASECHK"),
    "NANOSLEEP": re.compile(r"NANOSLEEP"),
    "LDSM": re.compile(r"\bLDSM\b"),
    "STSM": re.compile(r"\bSTSM\b"),
    "LDS": re.compile(r"\bLDS[\. ]"),
    "STS": re.compile(r"\bSTS[\. ]"),
    "LDG": re.compile(r"\bLDG[\. ]"),
    "STG": re.compile(r"\bSTG[\. ]"),
    "BAR.SYNC": re.compile(r"BAR\.SYNC"),
    "SYNCS.ARRIVE": re.compile(r"SYNCS\.ARRIVE"),
    "DEPBAR": re.compile(r"\bDEPBAR\b"),
}

src_pattern = re.compile(r'//## File "([^"]+)", line (\d+)')
instr_pattern = re.compile(r'/\*([0-9a-f]+)\*/')

# category -> {source_loc: count}
counts = {cat: defaultdict(int) for cat in categories}
# Also track SASS addresses per category
addr_list = {cat: [] for cat in categories}

current_src = "unknown"

input_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nvdisasm_lineinfo.txt"

with open(input_file) as f:
    for line in f:
        m = src_pattern.search(line)
        if m:
            fpath = m.group(1)
            # Shorten path
            fpath = fpath.replace("/code/tensorrt_llm/cpp/build/_deps/cutlass-src/include/", "cutlass:")
            fpath = fpath.replace("/code/tensorrt_llm/cpp/build/_deps/flashmla-src/csrc/cutlass/include/", "flashmla:")
            fpath = fpath.replace("/usr/local/cuda/targets/x86_64-linux/include/", "cuda:")
            current_src = f"{fpath}:{m.group(2)}"
            continue

        for cat, pat in categories.items():
            if pat.search(line):
                counts[cat][current_src] += 1
                addr_m = instr_pattern.search(line)
                if addr_m:
                    addr_list[cat].append((int(addr_m.group(1), 16), current_src))

# Print sorted results
for cat in categories:
    if not counts[cat]:
        continue
    total = sum(counts[cat].values())
    print(f"\n{'='*60}")
    print(f"  {cat} -- {total} instructions total")
    print(f"{'='*60}")
    for src, cnt in sorted(counts[cat].items(), key=lambda x: -x[1]):
        pct = 100*cnt/total
        print(f"  {cnt:4d} ({pct:5.1f}%)  {src}")

# Print OMMA address range
print(f"\n{'='*60}")
print(f"  OMMA Address Ranges")
print(f"{'='*60}")
if addr_list["OMMA"]:
    addrs = sorted(addr_list["OMMA"])
    print(f"  First OMMA: 0x{addrs[0][0]:x}  ({addrs[0][1]})")
    print(f"  Last OMMA:  0x{addrs[-1][0]:x}  ({addrs[-1][1]})")
    print(f"  Span: {addrs[-1][0] - addrs[0][0]} bytes")

# Print overall kernel structure: address ranges by section
print(f"\n{'='*60}")
print(f"  Kernel Structure (by SASS address ranges)")
print(f"{'='*60}")
for cat in categories:
    if not addr_list[cat]:
        continue
    addrs = sorted(addr_list[cat])
    lo = addrs[0][0]
    hi = addrs[-1][0]
    print(f"  {cat:20s}: 0x{lo:05x} - 0x{hi:05x}  ({len(addrs)} instructions)")
