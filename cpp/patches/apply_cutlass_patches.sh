#!/bin/bash
# Apply CUTLASS patches for M32/M64 CTA tile support on SM120.
#
# These patches fix TMA descriptor incompatibility when CTA_M < 128 with
# block-scaled FP4 on Blackwell. Without them, only CTA_M=128 compiles.
#
# Usage:
#   ./cpp/patches/apply_cutlass_patches.sh [build_dir]
#
# Arguments:
#   build_dir  Path to the cmake build directory (default: cpp/build)
#
# Run AFTER cmake configure (which fetches CUTLASS via FetchContent)
# but BEFORE make/ninja compile. If building with build_wheel.py,
# run this between the cmake and build steps, or simply re-run after
# build_wheel.py completes and rebuild:
#
#   python3 scripts/build_wheel.py --fast_build -a '120-real' --use_ccache --no-venv -j 8
#   ./cpp/patches/apply_cutlass_patches.sh
#   cmake --build cpp/build -j 8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${1:-$REPO_ROOT/cpp/build}"

CUTLASS_SRC="$BUILD_DIR/_deps/cutlass-src"

if [ ! -d "$CUTLASS_SRC" ]; then
    echo "ERROR: CUTLASS source not found at $CUTLASS_SRC"
    echo "Run cmake configure first to fetch CUTLASS via FetchContent."
    exit 1
fi

PATCH_FILE="$SCRIPT_DIR/cutlass_m32_m64_sfa_tma.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

echo "Applying CUTLASS M32/M64 SFA TMA patch to $CUTLASS_SRC ..."

# Check if patch is already applied by looking for our signature line
if grep -q "TileM_SFA" "$CUTLASS_SRC/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl" 2>/dev/null; then
    echo "Patch already applied. Skipping."
    exit 0
fi

cd "$CUTLASS_SRC"
git apply "$PATCH_FILE"

echo "Patch applied successfully."
echo "Files modified:"
echo "  - include/cutlass/epilogue/collective/builders/sm120_builder.inl (mode change)"
echo "  - include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl (+29 lines)"
echo "  - include/cutlass/gemm/collective/sm120_blockscaled_mma_array_tma.hpp (+28 lines)"
