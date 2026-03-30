#!/usr/bin/env python3

from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.nvfp4_compress.run_mmlu_direct import main


if __name__ == "__main__":
    main()
