#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

template <typename T>
struct cutlass_dtype {
    using type = T;
};

template <>
struct cutlass_dtype<half> {
    using type = cutlass::half_t;
};

template <>
struct cutlass_dtype<nv_bfloat16> {
    using type = cutlass::bfloat16_t;
};

template <typename T>
__global__ void precompute_sgmv_args_backward(cutlass::gemm::GemmCoord* all_problems,
                                              T** ptr_Y, T** ptr_X, T** ptr_W,
                                              int64_t* ld_Y, int64_t* ld_X,
                                              int64_t* ld_W, T* Y, T* X, T** W,
                                              int32_t* s, int d_in, int d_out,
                                              int layer_idx) {
    int i = blockIdx.x;
    int m = s[i + 1] - s[i];
    int k = (d_in < d_out) ? d_in : d_out; // Assuming r = min(d_in, d_out)
    int n = (d_in < d_out) ? d_out : d_in;
    all_problems[i] = cutlass::gemm::GemmCoord(m, n, k);
    ptr_W[i] = W[i] + layer_idx * d_in * d_out;
    ptr_X[i] = X + s[i] * d_in;
    ptr_Y[i] = Y + s[i] * n; // Adjust based on whether computing grad_A or grad_B
    ld_X[i] = k;
    ld_W[i] = n;
    ld_Y[i] = n;
}

size_t sgmv_tmp_size_backward(int num_problems) {
    constexpr auto sz = sizeof(void*) * 3 + sizeof(int64_t) * 3 +
                        sizeof(cutlass::gemm::GemmCoord);
    return sz * num_problems;
}

template <typename T>
inline T* alloc_from_buf_backward(void** buf, int n) {
    auto* p = (T*)*buf;
    *buf = (void*)(p + n);
    return p;
}

template <typename DType>
bool sgmv_backprop(DType* grad_y, DType* x, DType* grad_A, DType* grad_B, DType** A, DType** B,
                  int32_t* s, void* tmp_d, int num_problems, int d_in, int d_out,
                  int layer_idx, cudaStream_t stream) {
    using cutlass_t = typename cutlass_dtype<DType>::type;

    // Allocate temporary buffers
    void* tmp_ptr = tmp_d;
    auto ptr_GradY = alloc_from_buf_backward<cutlass_t*>(tmp_ptr, num_problems);
    auto ptr_X = alloc_from_buf_backward<cutlass_t*>(tmp_ptr, num_problems);
    auto ptr_W = alloc_from_buf_backward<cutlass_t*>(tmp_ptr, num_problems);
    auto ld_GradY = alloc_from_buf_backward<int64_t>(tmp_ptr, num_problems);
    auto ld_X = alloc_from_buf_backward<int64_t>(tmp_ptr, num_problems);
    auto ld_W = alloc_from_buf_backward<int64_t>(tmp_ptr, num_problems);
    auto all_problems = alloc_from_buf_backward<cutlass::gemm::GemmCoord>(tmp_ptr, num_problems);

    // Step 1: Compute grad_v = grad_y * B^T
    precompute_sgmv_args_backward<<<num_problems, 1, 0, stream>>>(
        all_problems, ptr_GradY, ptr_X, ptr_W, ld_GradY, ld_X, ld_W,
        (cutlass_t*)grad_y, (cutlass_t*)x, (cutlass_t**)B, s, d_in, d_out, layer_idx);

    // Define GEMM configuration for grad_v = grad_y * B^T
    using GemmKernel_grad_v = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,                                      // Element A
        cutlass::layout::RowMajor,                      // Layout A
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity A
        cutlass_t,                                      // Element B
        cutlass::layout::RowMajor,                      // Layout B
        cutlass::ComplexTransform::kNone,               //
        8,                                              // Granularity B
        cutlass_t,                                      // Element C&D
        cutlass::layout::RowMajor,                      // Layout C&D
        float,                                          // Element Accumulator
        cutlass::arch::OpClassTensorOp,                 // Operator Class Tag
        cutlass::arch::Sm80,                            // Architecture
        cutlass::gemm::GemmShape<32, 128, 16>,          // Thread Block Shape
        cutlass::gemm::GemmShape<32, 64, 16>,           // Warp Shape
        cutlass::gemm::GemmShape<16, 8, 8>,             // Instruction Shape
        cutlass::epilogue::thread::LinearCombination<cutlass_t, 8, float, float>,  // Epilogue
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,              // Swizzling Operator
        2                                               // Stages
        >::GemmKernel;

    using EpilogueOutputOp_grad_v = typename GemmKernel_grad_v::Epilogue::OutputOp;
    typename EpilogueOutputOp_grad_v::Params epilogue_op_grad_v(1.0, 0.0); // Scale grad_v appropriately

    using GemmGrouped_grad_v = cutlass::gemm::device::GemmGrouped<GemmKernel_grad_v>;
    typename GemmGrouped_grad_v::Arguments args_grad_v(all_problems, num_problems, 512,
                                                       epilogue_op_grad_v, ptr_X, ptr_W, ptr_GradY,
                                                       ptr_GradY, ld_X, ld_W, ld_GradY, ld_GradY);

    GemmGrouped_grad_v gemm_grad_v;
    auto status_grad_v = gemm_grad_v.initialize(args_grad_v, nullptr, stream);
    if (status_grad_v != cutlass::Status::kSuccess) {
        fprintf(stderr, "sgmv_backprop grad_v gemm.initialize failed: %s\n",
                cutlassGetStatusString(status_grad_v));
        return false;
    }
    status_grad_v = gemm_grad_v.run(stream);
    if (status_grad_v != cutlass::Status::kSuccess) {
        fprintf(stderr, "sgmv_backprop grad_v gemm.run failed: %s\n",
                cutlassGetStatusString(status_grad_v));
        return false;
    }

    // Step 2: Compute grad_A = x^T * grad_v
    // Reuse precompute_sgmv_args_backward or create a new precompute function if necessary
    // Adjust pointers for grad_A computation
    // Assuming grad_A is stored in a separate buffer
    // Adjust ld_Y and ld_W accordingly

    // Precompute arguments for grad_A
    precompute_sgmv_args_backward<<<num_problems, 1, 0, stream>>>(
        all_problems, ptr_GradY, ptr_X, ptr_W, ld_GradY, ld_X, ld_W,
        (cutlass_t*)grad_v, (cutlass_t*)x, (cutlass_t**)A, s, d_in, d_out, layer_idx);

    // Define GEMM configuration for grad_A = x^T * grad_v
    using GemmKernel_grad_A = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        8,
        cutlass_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        8,
        cutlass_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32, 128, 16>,
        cutlass::gemm::GemmShape<32, 64, 16>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<cutlass_t, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        2
        >::GemmKernel;

    using EpilogueOutputOp_grad_A = typename GemmKernel_grad_A::Epilogue::OutputOp;
    typename EpilogueOutputOp_grad_A::Params epilogue_op_grad_A(1.0, 1.0); // Accumulate gradients

    using GemmGrouped_grad_A = cutlass::gemm::device::GemmGrouped<GemmKernel_grad_A>;
    typename GemmGrouped_grad_A::Arguments args_grad_A(all_problems, num_problems, 512,
                                                       epilogue_op_grad_A, ptr_X, ptr_GradY, grad_A,
                                                       grad_A, ld_X, ld_GradY, ld_GradY, ld_GradY);

    GemmGrouped_grad_A gemm_grad_A;
    auto status_grad_A = gemm_grad_A.initialize(args_grad_A, nullptr, stream);
    if (status_grad_A != cutlass::Status::kSuccess) {
        fprintf(stderr, "sgmv_backprop grad_A gemm.initialize failed: %s\n",
                cutlassGetStatusString(status_grad_A));
        return false;
    }
    status_grad_A = gemm_grad_A.run(stream);
    if (status_grad_A != cutlass::Status::kSuccess) {
        fprintf(stderr, "sgmv_backprop grad_A gemm.run failed: %s\n",
                cutlassGetStatusString(status_grad_A));
        return false;
    }

    // Step 3: Compute v = x * A (if needed for grad_B)
    // Alternatively, store v during the forward pass to avoid recomputation

    // For this example, we'll recompute v
    // Allocate buffer for v
    DType* v;
    cudaMalloc(&v, sizeof(DType) * num_problems * d_out); // Adjust size as needed

    // Compute v = x * A
    // Precompute arguments for v computation
    precompute_sgmv_args_backward<<<num_problems, 1, 0, stream>>>(
        all_problems, ptr_X, ptr_X, ptr_W, ld_Y, ld_X, ld_W,
        (cutlass_t*)x, (cutlass_t*)x, (cutlass_t**)A, s, d_in, d_out, layer_idx);

    using GemmKernel_v = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        8,
        cutlass_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        8,
        cutlass_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32, 128, 16>,
        cutlass::gemm::GemmShape<32, 64, 16>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<cutlass_t, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        2
        >::GemmKernel;

    using EpilogueOutputOp_v = typename GemmKernel_v::Epilogue::OutputOp;
    typename EpilogueOutputOp_v::Params epilogue_op_v(1.0, 0.0); // Overwrite v

    using GemmGrouped_v = cutlass::gemm::device::GemmGrouped<GemmKernel_v>;
    typename GemmGrouped_v::Arguments args_v(all_problems, num_problems, 512,
                                           epilogue_op_v, ptr_X, ptr_W, v,
                                           v, ld_X, ld_W, ld_Y, ld_Y);

    GemmGrouped_v gemm_v;
    auto status_v = gemm_v.initialize(args_v, nullptr, stream);
    if (status_v != cutlass::Status::kSuccess) {
        fprintf(stderr, "sgmv_backprop v gemm.initialize failed: %s\n",
                cutlassGetStatusString(status_v));
        cudaFree(v);
        return false;
    }
    status_v = gemm_v.run(stream);
    if (status_v != cutlass::Status::kSuccess) {
        fprintf(stderr, "sgmv_backprop v gemm.run failed: %s\n",
                cutlassGetStatusString(status_v));
        cudaFree(v);
        return false;
    }

    // Step 4: Compute grad_B = v^T * grad_y
    precompute_sgmv_args_backward<<<num_problems, 1, 0, stream>>>(
        all_problems, ptr_GradY, ptr_X, ptr_W, ld_GradY, ld_X, ld_W,
        (cutlass_t*)v, (cutlass_t*)v, (cutlass_t**)B, s, d_in, d_out, layer_idx);

    using GemmKernel_grad_B = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        cutlass_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        8,
        cutlass_t,
        cutlass::layout::RowMajor,
        cutlass::ComplexTransform::kNone,
        8,
        cutlass_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32, 128, 16>,
        cutlass::gemm::GemmShape<32, 64, 16>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<cutlass_t, 8, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        2
        >::GemmKernel;

    using EpilogueOutputOp_grad_B = typename GemmKernel_grad_B::Epilogue::OutputOp;
    typename EpilogueOutputOp_grad_B::Params epilogue_op_grad_B(1.0, 1.0); // Accumulate gradients

    using GemmGrouped_grad_B = cutlass::gemm::device::GemmGrouped<GemmKernel_grad_B>;
    typename GemmGrouped_grad_B::Arguments args_grad_B(all_problems, num_problems, 512,
                                                       epilogue_op_grad_B, ptr_X, ptr_W, grad_B,
                                                       grad_B, ld_X, ld_W, ld_Y, ld_Y);

    GemmGrouped_grad_B gemm_grad_B;
    auto status_grad_B = gemm_grad_B.initialize(args_grad_B, nullptr, stream);
    if (status_grad_B != cutlass::Status::kSuccess) {
        fprintf(stderr, "sgmv_backprop grad_B gemm.initialize failed: %s\n",
                cutlassGetStatusString(status_grad_B));
        cudaFree(v);
        return false;
    }
    status_grad_B = gemm_grad_B.run(stream);
    if (status_grad_B != cutlass::Status::kSuccess) {
        fprintf(stderr, "sgmv_backprop grad_B gemm.run failed: %s\n",
                cutlassGetStatusString(status_grad_B));
        cudaFree(v);
        return false;
    }

    // Free temporary buffer for v
    cudaFree(v);

    return true;
}

// Kernel to compute grad_v = grad_y * B^T
template <typename DType>
__global__ void compute_grad_v(DType* grad_v, DType* grad_y, DType** B, int32_t* s, int d_in, int d_out, int layer_idx) {
    int i = blockIdx.x;
    int m = s[i + 1] - s[i];
    // Each block handles one LoRA model
    // Compute grad_v[i] = grad_y[i] * B^T
    // This requires a matrix-vector multiplication
    // Implementation details depend on the specific layout and batching
    // For simplicity, you can use CUTLASS or other libraries here as well
}

// Kernel to compute v = x * A
template <typename DType>
__global__ void compute_v(DType* v, DType* x, DType** A, int32_t* s, int d_in, int r, int layer_idx) {
    int i = blockIdx.x;
    int m = s[i + 1] - s[i];
    // Each block handles one LoRA model
    // Compute v[i] = x[i] * A
    // This requires a matrix-vector multiplication
    // Implementation details depend on the specific layout and batching
    // For simplicity, you can use CUTLASS or other libraries here as well
}
