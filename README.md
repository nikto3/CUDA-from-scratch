# CUDA from scratch

CUDA kernels implemented and profiled from scratch while working through
the PMPP ("Programming Massively Parallel Processors") book.

Every kernel is checked for correctness against a CPU reference and, where
relevant, profiled to verify the optimization it demonstrates.

## Kernels

| Kernel | Concept |
|---|---|
| VecAdd | basics: grid/block indexing, boundary guard |
| RGB2Gray | 2D indexing over an image |
| Blur | 2D stencil |
| MatMul | naive matrix multiply |
| TiledMatMul | shared-memory tiling |
| CoarsTiledMatMul | tiling + thread coarsening |
| CornerTurningMatMul | coalesced loads of a column-major matrix |
| MatTranspose | naive vs. shared-memory + bank-conflict-free |

## Verification & profiling

Correctness: each kernel is diffed against a CPU implementation.
Performance: profiled with Nsight Compute on a Turing T4.
