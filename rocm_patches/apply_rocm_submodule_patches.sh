#!/usr/bin/env bash
# Apply the ROCm/HIP source fixes that live inside the git submodules of the
# masked DROID-SLAM (lietorch, eigen). Submodule internals cannot be committed
# from this repo, so run this once after `git submodule update --init --recursive`
# and before building `thirdparty/DROID-SLAM`.
#
# The in-tree fixes (masked DROID-SLAM setup.py gencode gate + .type()->.scalar_type()
# in src/*.cu) are already committed on this branch and need no action here.
#
# Idempotent: safe to re-run.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DROID="${HERE}/../thirdparty/DROID-SLAM"
LIET="${DROID}/thirdparty/lietorch"
EIGEN="${DROID}/thirdparty/eigen"

echo "[rocm-patch] lietorch/eigen submodule fixes"

# P2: lietorch's own setup.py hardcodes NVIDIA -gencode flags -> hipcc/clang++
# rejects them. Strip on any build (arch comes from PYTORCH_ROCM_ARCH on ROCm).
if [ -f "${LIET}/setup.py" ]; then
  sed -i '/gencode=arch=compute/d' "${LIET}/setup.py"
  echo "  P2 patched ${LIET}/setup.py (removed -gencode)"
fi

# P3: dependent member-template call needs the `template` disambiguator under
# clang/hipcc: Matrix4x4().block<3,3>(...) -> Matrix4x4().template block<3,3>(...)
if [ -f "${LIET}/lietorch/src/lietorch_gpu.cu" ]; then
  sed -i 's/Matrix4x4()\.block<3,3>/Matrix4x4().template block<3,3>/' \
    "${LIET}/lietorch/src/lietorch_gpu.cu"
  echo "  P3 patched lietorch_gpu.cu (.template)"
fi

# P5: bundled Eigen uses EIGEN_USING_STD(fill_n); under hipcc (__HIPCC__/__CUDACC__
# defined) that macro expands to `using ::fill_n;` (global namespace, which has no
# fill_n) -> force the std:: qualified name.
if [ -d "${EIGEN}/Eigen" ]; then
  grep -rl 'EIGEN_USING_STD(fill_n)' "${EIGEN}/Eigen" 2>/dev/null | \
    xargs -r sed -i 's/EIGEN_USING_STD(fill_n);/using std::fill_n;/g'
  echo "  P5 patched eigen EIGEN_USING_STD(fill_n) -> using std::fill_n"
fi

echo "[rocm-patch] done"
