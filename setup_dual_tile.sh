#!/bin/bash
set -euo pipefail

REPO_DIR="/code/tensorrt_llm"
BUILD_DIR="$REPO_DIR/cpp/build"
PARALLEL=${1:-8}

echo "============================================="
echo " TRT-LLM Dual-Tile (M32/M64) Setup"
echo " SM120 NVFP4 MoE Grouped GEMM"
echo "============================================="
echo ""
echo "Build parallelism: $PARALLEL (pass as arg to change, e.g. ./setup_dual_tile.sh 4)"
echo ""

cd "$REPO_DIR"

# Step 1: cmake configure only — this fetches CUTLASS via FetchContent
echo "[1/5] Running cmake configure (fetches CUTLASS)..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAST_BUILD=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120-real" \
  -DBUILD_PYT=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_BENCHMARKS=ON \
  -S "$REPO_DIR/cpp"
cd "$REPO_DIR"

# Step 2: apply CUTLASS patches for M32/M64 SFA TMA fix
echo ""
echo "[2/5] Applying CUTLASS patches..."
./cpp/patches/apply_cutlass_patches.sh

# Step 3: full build via build_wheel.py (reuses existing configure, builds + creates wheel)
echo ""
echo "[3/5] Building TRT-LLM (this takes a while)..."
python3 scripts/build_wheel.py \
  --fast_build \
  -a '120-real' \
  --use_ccache \
  --no-venv \
  -j "$PARALLEL"

# Step 4: install the wheel
echo ""
echo "[4/5] Installing wheel..."
pip install "$REPO_DIR"/build/tensorrt_llm-*.whl

# Step 5: fix library paths
echo ""
echo "[5/5] Configuring library paths..."
echo "/usr/local/tensorrt/lib" > /etc/ld.so.conf.d/tensorrt.conf
ldconfig 2>/dev/null

# Verify
echo ""
echo "============================================="
echo " Verification"
echo "============================================="
python3 -c "
import tensorrt_llm
print(f'tensorrt_llm {tensorrt_llm.__version__} OK')
runner = __import__('torch').classes.trtllm.FusedMoeRunner(
    __import__('torch').int64, __import__('torch').int64, __import__('torch').bfloat16,
    False, False, False, False, True)
n = runner.get_tactic_num(1)
print(f'GEMM1 tactics: {n}')
assert n >= 6, f'Expected >= 6 tactics (M128+M64+M32), got {n}'
print('M128 + M64 + M32 tiles available — setup complete!')
" 2>&1 | grep -v "^/" | grep -v "Warning" | grep -v "self.m"

echo "============================================="
echo " Done! Run tests with:"
echo "   python3 added_benchmark/test_m64_tile.py"
echo "============================================="
