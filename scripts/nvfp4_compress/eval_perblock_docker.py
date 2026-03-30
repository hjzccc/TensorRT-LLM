#!/usr/bin/env python3
"""Docker-compatible wrapper for eval_perblock_fast.py"""
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
# Add Docker paths
sys.path.insert(0, '/code/tensorrt_llm/scripts/channel_quant_new')
sys.path.insert(0, '/code/tensorrt_llm/scripts/channel_quant')
sys.path.insert(0, '/code/tensorrt_llm/scripts/channel_quant_new/profiling')

# Now import and run the main evaluation
import importlib.util
spec = importlib.util.spec_from_file_location(
    'eval_perblock_fast',
    '/code/tensorrt_llm/scripts/nvfp4_compress/eval_perblock_fast.py'
)
mod = importlib.util.module_from_spec(spec)
# Override the sys.path inserts in the module
spec.loader.exec_module(mod)
