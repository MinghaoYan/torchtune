#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

template <typename DType>
bool sgmv_backprop(DType *grad_y, DType *x, DType **w_A, DType **w_B,
                   DType **grad_A, DType **grad_B, int32_t *s,
                   void *tmp_d, int num_problems, int d_in, int d_out,
                   int layer_idx, cudaStream_t stream) {
  using cutlass_t = typename cutlass_dtype<DType>::type;

  // Allocate temporary buffers
  auto ptr_GradY = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_X = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_V = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_GradV = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_B = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_A = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_GradB = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);
  auto ptr_GradA = alloc_from_buf<cutlass_t *>(&tmp_d, num_problems);

  auto ld_GradY = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_X = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_V = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_GradV = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_B = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_A = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_GradB = alloc_from_buf<int64_t>(&tmp_d, num_problems);
  auto ld_GradA = alloc_from_buf<int64_t>(&tmp_d, num_problems);

  auto all_problems =
      alloc_from_buf<cutlass::gemm::GemmCoord>(&tmp_d, num_problems);

  // Set up the problem sizes and pointers
  precompute_backprop_args<<<num_problems, 1, 0, stream>>>(
      all_problems, ptr_GradY, ptr_X, ptr_V, ptr_GradV, ptr_B, ptr_A,
      ptr_GradB, ptr_GradA, ld_GradY, ld_X, ld_V, ld_GradV,
      ld_B, ld_A, ld_GradB, ld_GradA,
      (cutlass_t *)grad_y, (cutlass_t *)x, (cutlass_t **)w_A,
      (cutlass_t **)w_B, (cutlass_t **)grad_A, (cutlass_t **)grad_B,
      s, d_in, d_out, layer_idx);

  // Compute GradV = GradY * B^T
  // For expand mode only
  if (d_in < d_out) {
    // Compute GradV = GradY * B^T
    // GEMM: (m x n) x (n x r) -> (m x r)
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,                                      // Element A (GradY)
        cutlass::layout::RowMajor,                      // Layout A
        cutlass::ComplexTransform::kNone,
        8,                                              // Granularity A
        cutlass_t,                                      // Element B (B^T)
        cutlass::layout::ColumnMajor,                   // Layout B (B^T)
        cutlass::ComplexTransform::kNone,
        8,                                              // Granularity B
        cutlass_t,                                      // Element C&D (GradV)
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        cutlass::arch::Sm80,                            // Architecture
        cutlass::gemm::GemmShape<128, 128, 32>,         // Thread Block Shape
        cutlass::gemm::GemmShape<64, 64, 32>,           // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 8>,             // Instruction Shape
        LinearCombination<cutlass_t, 8, float, float>,  // Epilogue
        GemmIdentityThreadblockSwizzle<>,               // Swizzling Operator
        2                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(1.0, 0.0);

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(
        all_problems, num_problems, 512, epilogue_op,
        ptr_GradY, ptr_B, ptr_GradV, ptr_GradV,
        ld_GradY, ld_B, ld_GradV, ld_GradV);

    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "GradV gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
    status = gemm(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "GradV gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
  } else {
    // For shrink mode, GradV = GradY
    // Copy GradY to GradV
    for (int i = 0; i < num_problems; ++i) {
      cudaMemcpyAsync(ptr_GradV[i], ptr_GradY[i],
                      all_problems[i].m() * all_problems[i].n() * sizeof(cutlass_t),
                      cudaMemcpyDeviceToDevice, stream);
    }
  }

  // Compute GradA = X^T * GradV
  // GEMM: (k x m) x (m x r) -> (k x r)
  {
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,                                      // Element A (X^T)
        cutlass::layout::ColumnMajor,                   // Layout A (X^T)
        cutlass::ComplexTransform::kNone,
        8,                                              // Granularity A
        cutlass_t,                                      // Element B (GradV)
        cutlass::layout::RowMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,
        8,                                              // Granularity B
        cutlass_t,                                      // Element C&D (GradA)
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        cutlass::arch::Sm80,                            // Architecture
        cutlass::gemm::GemmShape<128, 128, 32>,         // Thread Block Shape
        cutlass::gemm::GemmShape<64, 64, 32>,           // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 8>,             // Instruction Shape
        LinearCombination<cutlass_t, 8, float, float>,  // Epilogue
        GemmIdentityThreadblockSwizzle<>,               // Swizzling Operator
        2                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0); // Accumulate

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(
        all_problems, num_problems, 512, epilogue_op,
        ptr_X, ptr_GradV, ptr_GradA, ptr_GradA,
        ld_X, ld_GradV, ld_GradA, ld_GradA);

    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "GradA gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
    status = gemm(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "GradA gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
  }

  // For expand mode, compute GradB = V^T * GradY
  if (d_in < d_out) {
    // Compute GradB = V^T * GradY
    // GEMM: (r x m) x (m x n) -> (r x n)
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,                                      // Element A (V^T)
        cutlass::layout::ColumnMajor,                   // Layout A (V^T)
        cutlass::ComplexTransform::kNone,
        8,                                              // Granularity A
        cutlass_t,                                      // Element B (GradY)
        cutlass::layout::RowMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,
        8,                                              // Granularity B
        cutlass_t,                                      // Element C&D (GradB)
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        cutlass::arch::Sm80,                            // Architecture
        cutlass::gemm::GemmShape<128, 128, 32>,         // Thread Block Shape
        cutlass::gemm::GemmShape<64, 64, 32>,           // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 8>,             // Instruction Shape
        LinearCombination<cutlass_t, 8, float, float>,  // Epilogue
        GemmIdentityThreadblockSwizzle<>,               // Swizzling Operator
        2                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp = typename GemmKernel::Epilogue::OutputOp;
    typename EpilogueOutputOp::Params epilogue_op(1.0, 1.0); // Accumulate

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;
    typename GemmGrouped::Arguments args(
        all_problems, num_problems, 512, epilogue_op,
        ptr_V, ptr_GradY, ptr_GradB, ptr_GradB,
        ld_V, ld_GradY, ld_GradB, ld_GradB);

    GemmGrouped gemm;
    auto status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "GradB gemm.initialize failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
    status = gemm(stream);
    if (status != cutlass::Status::kSuccess) {
      fprintf(stderr, "GradB gemm.run failed: %s\n",
              cutlassGetStatusString(status));
      return false;
    }
  }

  return true;
}
