#!/usr/bin/env python3
"""Cross-reference NCU stall data with nvdisasm source line mapping."""
import re, csv, sys
from collections import defaultdict

NVDIS_FILE = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nvdisasm_lineinfo.txt"
NCU_CSV_FILE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ncu_source.csv"

# Step 1: Build nvdisasm offset -> source line mapping
src_pattern = re.compile(r'//## File "([^"]+)", line (\d+)')
instr_pattern = re.compile(r'/\*([0-9a-f]+)\*/\s+(.+)')

nvdis_map = {}
current_src = ("unknown", "0")

with open(NVDIS_FILE) as f:
    for line in f:
        m = src_pattern.search(line)
        if m:
            fpath = m.group(1)
            fpath = fpath.replace("/code/tensorrt_llm/cpp/build/_deps/cutlass-src/include/", "cutlass:")
            fpath = fpath.replace("/code/tensorrt_llm/cpp/build/_deps/flashmla-src/csrc/cutlass/include/", "flashmla:")
            fpath = fpath.replace("/usr/local/cuda/targets/x86_64-linux/include/", "cuda:")
            current_src = (fpath, m.group(2))
            continue
        m = instr_pattern.search(line)
        if m:
            offset = m.group(1)
            instr_text = m.group(2).strip().rstrip(";").strip()
            instr_text = re.sub(r"\s+", " ", instr_text)
            nvdis_map[offset] = (current_src[0], current_src[1], instr_text)

print("Loaded %d instructions from nvdisasm" % len(nvdis_map))

# Build lookup by normalized instruction text
nvdis_by_instr = defaultdict(list)
for offset, (src_file, src_line, instr_text) in nvdis_map.items():
    norm = instr_text.split(";")[0].strip()
    norm = re.sub(r"\s+", " ", norm)
    nvdis_by_instr[norm].append((offset, src_file, src_line))

# Step 2: Parse NCU CSV
stall_cols = ["stall_barrier", "stall_branch_resolving", "stall_dispatch", "stall_drain",
              "stall_lg", "stall_long_sb", "stall_math", "stall_membar", "stall_mio",
              "stall_misc", "stall_no_inst", "stall_not_selected", "stall_selected",
              "stall_short_sb", "stall_sleep", "stall_tex", "stall_wait"]

stall_by_source = defaultdict(lambda: defaultdict(int))
total_samples_by_source = defaultdict(int)
matched = 0
unmatched = 0
total_stall_samples = 0

with open(NCU_CSV_FILE) as f:
    reader = csv.reader(f)
    header = next(reader)
    header = next(reader)
    
    col_indices = {}
    for i, col in enumerate(header):
        col_indices[col.strip('"')] = i
    
    stall_col_idx = {}
    for col_name in stall_cols:
        for key, idx in col_indices.items():
            if key == col_name:
                stall_col_idx[col_name] = idx
                break
    
    samples_idx = col_indices.get("# Samples", None)
    
    for row in reader:
        if len(row) < 10:
            continue
        
        ncu_addr = row[0].strip('"')
        ncu_instr = row[1].strip('"').strip()
        norm_ncu = re.sub(r"\s+", " ", ncu_instr)
        
        samples = 0
        if samples_idx is not None:
            try:
                samples = int(row[samples_idx].strip('"'))
            except (ValueError, IndexError):
                pass
        
        if samples == 0:
            continue
        
        source_loc = None
        matches = nvdis_by_instr.get(norm_ncu, [])
        if matches:
            offset, src_file, src_line = matches[0]
            source_loc = "%s:%s" % (src_file, src_line)
            matched += 1
        else:
            source_loc = "UNMATCHED:%s" % ncu_addr
            unmatched += 1
        
        total_samples_by_source[source_loc] += samples
        total_stall_samples += samples
        
        for stall_name, idx in stall_col_idx.items():
            try:
                val = int(row[idx].strip('"'))
                if val > 0:
                    stall_by_source[source_loc][stall_name] += val
            except (ValueError, IndexError):
                pass

print("Matched: %d, Unmatched: %d" % (matched, unmatched))
print("Total stall samples: %d" % total_stall_samples)
print()

def print_top(title, stall_name, limit=15):
    print("=" * 80)
    print("  %s" % title)
    print("=" * 80)
    total = sum(v.get(stall_name, 0) for v in stall_by_source.values())
    for src in sorted(stall_by_source.keys(), key=lambda s: -stall_by_source[s].get(stall_name, 0))[:limit]:
        cnt = stall_by_source[src].get(stall_name, 0)
        if cnt == 0:
            break
        pct = 100.0 * cnt / total if total > 0 else 0
        print("  %6d (%5.1f%%)  %s" % (cnt, pct, src))
    print()

# Overall top sources
print("=" * 80)
print("  TOP STALL SOURCES BY TOTAL SAMPLES")
print("=" * 80)
for src, count in sorted(total_samples_by_source.items(), key=lambda x: -x[1])[:20]:
    pct = 100.0 * count / total_stall_samples if total_stall_samples > 0 else 0
    print("  %6d (%5.1f%%)  %s" % (count, pct, src))
print()

print_top("TOP stall_long_sb SOURCES (scoreboard waits)", "stall_long_sb")
print_top("TOP stall_sleep SOURCES (nanosleep / producer wait)", "stall_sleep")
print_top("TOP stall_barrier SOURCES (BAR.SYNC)", "stall_barrier")
print_top("TOP stall_wait SOURCES (memory fence / data wait)", "stall_wait")
print_top("TOP stall_mio SOURCES (MIO throttle)", "stall_mio")
print_top("TOP stall_math SOURCES (math pipe busy)", "stall_math")
print_top("TOP stall_lg SOURCES (local/global memory)", "stall_lg")
print_top("TOP stall_short_sb SOURCES (short scoreboard)", "stall_short_sb")

# Detailed breakdown per top source
print("=" * 80)
print("  STALL BREAKDOWN PER TOP SOURCE")
print("=" * 80)
for src, count in sorted(total_samples_by_source.items(), key=lambda x: -x[1])[:15]:
    print("\n  %s  (total samples: %d)" % (src, count))
    stalls = stall_by_source[src]
    for stall_name in sorted(stalls.keys(), key=lambda s: -stalls[s]):
        if stalls[stall_name] > 0:
            print("    %-30s: %5d" % (stall_name, stalls[stall_name]))
