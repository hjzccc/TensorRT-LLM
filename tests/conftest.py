"""
conftest.py - pytest configuration for per-block codebook tests.

Provides a sys.modules stub so that tests/test_per_block_codebook.py can
import tensorrt_llm.quantization.per_block_codebook directly without
triggering the full TensorRT-LLM C++ binding chain (which requires a
compiled TRT-LLM installation).
"""
import sys
import types
import importlib
import importlib.util
import os

# Only apply the stub if the real tensorrt_llm package is broken
def _try_real_import():
    try:
        import tensorrt_llm.quantization.per_block_codebook  # noqa
        return True
    except Exception:
        return False

if not _try_real_import():
    # Build a minimal stub hierarchy: tensorrt_llm -> tensorrt_llm.quantization
    # then load per_block_codebook directly from the file.
    
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create stub for tensorrt_llm top-level package
    trtllm_stub = types.ModuleType("tensorrt_llm")
    trtllm_stub.__path__ = [os.path.join(repo_root, "tensorrt_llm")]
    trtllm_stub.__package__ = "tensorrt_llm"
    trtllm_stub.__spec__ = None
    sys.modules["tensorrt_llm"] = trtllm_stub
    
    # Create stub for tensorrt_llm.quantization sub-package
    quant_stub = types.ModuleType("tensorrt_llm.quantization")
    quant_stub.__path__ = [os.path.join(repo_root, "tensorrt_llm", "quantization")]
    quant_stub.__package__ = "tensorrt_llm.quantization"
    quant_stub.__spec__ = None
    sys.modules["tensorrt_llm.quantization"] = quant_stub
    trtllm_stub.quantization = quant_stub
    
    # Load per_block_codebook directly from file
    pbc_path = os.path.join(repo_root, "tensorrt_llm", "quantization", "per_block_codebook.py")
    spec = importlib.util.spec_from_file_location(
        "tensorrt_llm.quantization.per_block_codebook", pbc_path
    )
    pbc_module = importlib.util.module_from_spec(spec)
    pbc_module.__package__ = "tensorrt_llm.quantization"
    sys.modules["tensorrt_llm.quantization.per_block_codebook"] = pbc_module
    spec.loader.exec_module(pbc_module)
    quant_stub.per_block_codebook = pbc_module
