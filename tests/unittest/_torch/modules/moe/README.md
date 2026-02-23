# MoE Unit Tests

Unit tests for the Mixture-of-Experts (MoE) module in the PyTorch backend.

## Files

### Utilities (not run directly)

| File | Description |
|------|-------------|
| `quantize_utils.py` | Quantization helpers — weight generation, FP8/NVFP4 casting, tolerance calculation |
| `heter_moe_utils.py` | Shared helpers for heterogeneous MoE tests — config factories, backend creation, forward pass runners |

### Tests

| File | Description |
|------|-------------|
| `test_moe_module.py` | End-to-end MoE module tests via the high-level `forward()` interface (multi-process, MPI) |
| `test_moe_backend.py` | Backend-level tests via `quantize_input` + `run_moe` across all quant/backend combos |
| `test_heter_moe_correctness.py` | Forward-equivalence tests — verifies heterogeneous backends match homogeneous baselines |
| `test_heter_moe_config_policy.py` | Config validation (9 tests) and dispatch-policy logic (10 tests) for `HeterMoeConfig` |
| `test_heter_moe_benchmark.py` | Runtime performance benchmarks for heterogeneous MoE (latency, L2 cache, CUDA graphs) |

## Running

```bash
# All MoE unit tests
pytest tests/unittest/_torch/modules/moe/

# Single file
pytest tests/unittest/_torch/modules/moe/test_heter_moe_correctness.py

# Pattern match
pytest tests/unittest -k "heter_moe"
```

Most tests require a CUDA GPU. NVFP4 tests additionally require SM100+ (Blackwell).

## Related

- `tests/microbenchmarks/bench_heter_moe_ratio.py` — BF16/NVFP4 ratio sweep benchmark (imports `heter_moe_utils`)
- `tensorrt_llm/_torch/modules/moe/` — production MoE implementation
