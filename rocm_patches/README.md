# Running HaWoR's masked DROID-SLAM on AMD ROCm (gfx942 / MI300X)

The masked DROID-SLAM CUDA extension (`thirdparty/DROID-SLAM`) and its `lietorch`
sub-dependency build and run on AMD ROCm via PyTorch's HIP/`hipify` path. This
branch (`amd_support`) carries the minimal, upstreamable fixes.

## Validated environment

- Node: AMD Instinct MI300X (`gfx942`)
- Container: `rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.10.0`
  (torch 2.10.0+rocm7.2, HIP 7.2)
- `torch_scatter`: prebuilt ROCm wheel from
  [pyg-rocm-build](https://github.com/Looong01/pyg-rocm-build) release 15
  (`torch-2.10.0-rocm-*-py310`); avoids building `torch-scatter` from source.
- `PYTORCH_ROCM_ARCH=gfx942`; all visualization run headless.

## The five ROCm fixes

Committed **in-tree** on this branch:

- **P1** `thirdparty/DROID-SLAM/setup.py` — gate the NVIDIA `-gencode=arch=compute_*`
  flags behind `torch.version.hip is None`. On ROCm the target arch comes from
  `PYTORCH_ROCM_ARCH`; clang++/hipcc rejects `-gencode`. CUDA builds are unchanged.
- **P4** `thirdparty/DROID-SLAM/src/{correlation_kernels,altcorr_kernel}.cu` —
  `AT_DISPATCH_*` was passed the deprecated `Tensor.type()`. Newer torch's
  `DeprecatedTypeProperties` no longer converts to `ScalarType`; use
  `.scalar_type()` (correct on both CUDA and ROCm).

Applied to the **submodules** by `rocm_patches/apply_rocm_submodule_patches.sh`
(submodule internals cannot be committed from this repo):

- **P2** `lietorch/setup.py` — same `-gencode` strip.
- **P3** `lietorch/lietorch/src/lietorch_gpu.cu` — add the `template`
  disambiguator: `Matrix4x4().block<3,3>` → `Matrix4x4().template block<3,3>`
  (required by clang/hipcc for dependent member templates).
- **P5** `eigen/.../SparseCore/SparseMatrix.h` — `EIGEN_USING_STD(fill_n)` expands
  to `using ::fill_n;` under `__HIPCC__/__CUDACC__`; force `using std::fill_n;`.

## Build steps

```bash
git submodule update --init --recursive
bash rocm_patches/apply_rocm_submodule_patches.sh   # P2/P3/P5 (submodules)
cd thirdparty/DROID-SLAM
PYTORCH_ROCM_ARCH=gfx942 python setup.py install     # builds droid_backends + lietorch
```

## Functional GPU smoke

`droid_backends` was validated on `gfx942` by exercising the ported kernels on the
GPU: `iproj` / `projmap` / `frame_distance` (from `droid_kernels.cu`, the P5 Eigen
path) and `corr_index_forward` / `altcorr_forward` (the P4 `.scalar_type()` path),
alongside `lietorch.SE3` and `torch_scatter.scatter_mean`.
