#!/usr/bin/env python3
"""Single-process pipeline: compress D/E/F/G, decompress, evaluate all variants.
Designed to run as a single persistent process with checkpointing.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path("/code/tensorrt_llm/scripts/nvfp4_compress")
CKPT = SCRIPTS / "nvfp4_checkpoint"
COMPRESS = sys.executable + " " + str(SCRIPTS / "compress_checkpoint.py")
DECOMPRESS = sys.executable + " " + str(SCRIPTS / "decompress_checkpoint.py")
EVAL = sys.executable + " -u " + str(SCRIPTS / "run_mmlu_direct.py")
EVAL_ARGS = "--offline --subject professional_law --batch-size 64 --max-batch-total-tokens 12000"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir("/code/tensorrt_llm")

LOG = SCRIPTS / "python_pipeline.log"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

def run(cmd, logfile, check=True):
    log(f"Running: {cmd[:80]}...")
    with open(logfile, "w") as f:
        result = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
    if check and result.returncode != 0:
        log(f"FAILED (exit {result.returncode}): {cmd[:80]}")
        return False
    log(f"Done (exit {result.returncode})")
    return True

def compress_if_needed(scheme, logfile):
    manifest = SCRIPTS / f"compressed_{scheme}" / "compression_manifest.json"
    if manifest.exists():
        log(f"Already compressed: {scheme}")
        return True
    log(f"Compressing {scheme}...")
    return run(f"{COMPRESS} --scheme {scheme} --input {CKPT}", logfile)

def decompress_if_needed(src_scheme, dst_name, logfile):
    dst = SCRIPTS / dst_name
    if (dst / "config.json").exists():
        log(f"Already decompressed: {dst_name}")
        return True
    src = SCRIPTS / f"compressed_{src_scheme}"
    log(f"Decompressing {src_scheme} -> {dst_name}...")
    ok = run(f"{DECOMPRESS} --input {src} --output {dst}", logfile)
    if ok:
        # Copy config files
        ref = SCRIPTS / "decompressed_2b075b_zero_fixed_exact"
        for fn in ["config.json", "hf_quant_config.json", "model.safetensors.index.json"]:
            shutil.copy(ref / fn, dst / fn)
    return ok

def eval_if_needed(name, ckpt_dir, output_json, logfile):
    if Path(output_json).exists():
        try:
            d = json.load(open(output_json))
            acc = d.get("mmlu_acc", d.get("subjects", {}).get("professional_law", {}).get("acc", 0))
            log(f"Already evaluated {name}: {acc:.4f} ({acc*100:.2f}%)")
            return True
        except:
            pass
    log(f"Evaluating {name}...")
    ok = run(
        f"{EVAL} {EVAL_ARGS} --ckpt-dir {ckpt_dir} --output {output_json}",
        logfile
    )
    if ok and Path(output_json).exists():
        try:
            d = json.load(open(output_json))
            acc = d.get("mmlu_acc", d.get("subjects", {}).get("professional_law", {}).get("acc", 0))
            log(f"RESULT {name}: {acc:.4f} ({acc*100:.2f}%)")
        except:
            pass
    return ok

# ── PHASE 1: Compressions ──────────────────────────────────────────────────────
log("=== PHASE 1: Compressions ===")
compress_if_needed("2b075b_zero_fixed_scale_weighted", SCRIPTS / "compress_D.log")
compress_if_needed("2b075b_zero_fixed_freq_sq", SCRIPTS / "compress_E.log")
compress_if_needed("3b1b_4free_exact", SCRIPTS / "compress_F.log")
compress_if_needed("3b1b_4free_weighted_abs", SCRIPTS / "compress_G.log")
compress_if_needed("2b075b_zero_fixed_grouped_fisher", SCRIPTS / "compress_H.log")

# ── PHASE 2: Decompressions ────────────────────────────────────────────────────
log("=== PHASE 2: Decompressions ===")
decompress_if_needed("2b075b_zero_fixed_scale_weighted", "decompressed_D_scale_weighted", SCRIPTS / "decompress_D.log")
decompress_if_needed("2b075b_zero_fixed_freq_sq", "decompressed_E_freq_sq", SCRIPTS / "decompress_E.log")
decompress_if_needed("3b1b_4free_exact", "decompressed_F_3bit_exact", SCRIPTS / "decompress_F.log")
decompress_if_needed("3b1b_4free_weighted_abs", "decompressed_G_3bit_weighted", SCRIPTS / "decompress_G.log")
decompress_if_needed("2b075b_zero_fixed_grouped_fisher", "decompressed_H_grouped_fisher", SCRIPTS / "decompress_H.log")

# ── PHASE 3: Evaluations ───────────────────────────────────────────────────────
log("=== PHASE 3: Evaluations ===")

# A: exact MSE (2.75 bits)
eval_if_needed("A_exact_mse",
    SCRIPTS / "decompressed_2b075b_zero_fixed_exact",
    SCRIPTS / "result_A_exact_mse.json",
    SCRIPTS / "eval_A_final.log")

# B: weighted_abs MSE (2.75 bits)
eval_if_needed("B_weighted_abs",
    SCRIPTS / "decompressed_2b075b_zero_fixed_weighted_abs",
    SCRIPTS / "result_B_weighted_abs.json",
    SCRIPTS / "eval_B_final.log")

# C: freq_symmetric (2.0625 bits)
eval_if_needed("C_freq_sym",
    SCRIPTS / "decompressed_2b1b_freq_symmetric",
    SCRIPTS / "result_C_freq_sym.json",
    SCRIPTS / "eval_C_final.log")

# 2b1b_demo (2.0625 bits)
eval_if_needed("2b1b_demo",
    SCRIPTS / "decompressed_2b1b_demo",
    SCRIPTS / "result_2b1b_demo.json",
    SCRIPTS / "eval_2b1b_demo_final.log")

# D: scale_weighted (2.75 bits, BOF4-style)
eval_if_needed("D_scale_weighted",
    SCRIPTS / "decompressed_D_scale_weighted",
    SCRIPTS / "result_D_scale_weighted.json",
    SCRIPTS / "eval_D_final.log")

# E: freq_sq (2.75 bits)
eval_if_needed("E_freq_sq",
    SCRIPTS / "decompressed_E_freq_sq",
    SCRIPTS / "result_E_freq_sq.json",
    SCRIPTS / "eval_E_final.log")

# F: 3bit exact (3.0 bits)
eval_if_needed("F_3bit_exact",
    SCRIPTS / "decompressed_F_3bit_exact",
    SCRIPTS / "result_F_3bit_exact.json",
    SCRIPTS / "eval_F_final.log")

# G: 3bit weighted_abs (3.0 bits)
eval_if_needed("G_3bit_weighted",
    SCRIPTS / "decompressed_G_3bit_weighted",
    SCRIPTS / "result_G_3bit_weighted.json",
    SCRIPTS / "eval_G_final.log")

# H: grouped_fisher (2.75 bits)
eval_if_needed("H_grouped_fisher",
    SCRIPTS / "decompressed_H_grouped_fisher",
    SCRIPTS / "result_H_grouped_fisher.json",
    SCRIPTS / "eval_H_final.log")

# ── SUMMARY ────────────────────────────────────────────────────────────────────
log("=== RESULTS SUMMARY ===")
log("Baseline (NVFP4 professional_law): 59.78%")
for f in sorted(SCRIPTS.glob("result_*.json")):
    try:
        d = json.load(open(f))
        acc = d.get("mmlu_acc", d.get("subjects", {}).get("professional_law", {}).get("acc", 0))
        log(f"  {f.stem}: {acc:.4f} ({acc*100:.2f}%)")
    except:
        pass

log("=== ALL DONE ===")
