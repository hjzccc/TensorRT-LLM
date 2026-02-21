# Building TensorRT-LLM with Docker

## Prerequisites

- NVIDIA GPU with drivers installed (`nvidia-smi` works)
- Docker + NVIDIA Container Toolkit (run `./install_docker.sh` if not installed)
- After installing Docker, either log out/in or use `sg docker -c "..."` wrapper for docker commands

## Quick Start

### 1. Pull the devel image

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc3
```

### 2. Ensure LFS files are downloaded (one-time)

The repo uses Git LFS for prebuilt libraries (XQA cubins, internal CUTLASS kernels, nvshmem).
If you cloned without LFS or see small pointer files instead of real binaries:

```bash
git lfs install
git lfs pull
```

Verify with `git lfs ls-files` — all entries should show `*` (downloaded).

### 3. Build

**Start a persistent container:**

```bash
docker run -d --name trtllm-build \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume $(pwd):/code/tensorrt_llm \
  --workdir /code/tensorrt_llm \
  --tmpfs /tmp:exec \
  nvcr.io/nvidia/tensorrt-llm/devel:1.3.0rc3 \
  sleep infinity
```

**Run the build:**

fresh build:
```bash
docker exec trtllm-build \
  python3 scripts/build_wheel.py \
    --clean --fast_build \
    -a '120-real' \
    --use_ccache --no-venv \
    -j 14
```

rerun build:
```bash
docker exec -it trtllm-build \
  python3 scripts/build_wheel.py \
    --fast_build \
    -a '120-real' \
    --use_ccache --no-venv \
    -j 14
```

```bash
docker exec -it trtllm-build \
  python3 scripts/build_wheel.py \
    --build_type Debug \
    --fast_build \
    -a '120-real' \
    --use_ccache --no-venv \
    -j 8 \
    --configure_cmake \
```
### 4. Cleanup

```bash
docker stop trtllm-build && docker rm trtllm-build
```

## Memory Guide

CUDA compilation is extremely memory-hungry. The `-j` flag controls parallel compilation jobs.

| RAM   | Safe `-j` value | Notes                          |
|-------|-----------------|--------------------------------|
| 16 GB | `-j 1` or `-j 2`| Very slow, but won't OOM       |
| 32 GB | `-j 4`          | Good balance of speed vs memory|
| 64 GB | `-j 8`          | Comfortable                    |
| 128GB+| `-j` (no limit) | Use all cores                  |

**Rule of thumb:** each nvcc job can use 2–4 GB. Divide your available RAM by 4 to get a safe `-j` value.

> ⚠️ **Do NOT use `-j` without a number on ≤32 GB systems.** It defaults to all CPU cores and will OOM.

## Key Build Options

| Flag | Description |
|------|-------------|
| `-a '90-real'` | Target architecture. Use `90-real` for Hopper, `120-real` for Blackwell, `'90-real;120-real'` for both |
| `--fast_build` | Skip some kernels for faster dev builds |
| `--use_ccache` | Cache compilations (huge speedup on rebuilds) |
| `--no-venv` | Use container's Python directly |
| `--clean` | Start fresh (remove old build dir) |
| `-j N` | Parallel jobs. **Set this based on your RAM** |
| `--cpp_only` | Skip Python bindings (**broken** — needs `torch_python`) |

## Notes

- Build output goes to `cpp/build/`. It's owned by root (Docker runs as root). Use `docker exec` to clean it: `docker exec trtllm-build rm -rf /code/tensorrt_llm/cpp/build`
- First build takes 30–90 min depending on job count and architecture targets. Rebuilds with ccache are much faster.
- If your shell doesn't have the docker group, prefix commands with `sg docker -c "..."`.
