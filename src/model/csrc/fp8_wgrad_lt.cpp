// FP8 (e4m3) tensorwise scaled GEMM with beta=1 accumulation via cuBLASLt.
//
// torch._scaled_mm / F.scaled_mm は beta=0 固定なので、dW を一旦別 tensor に書いて
// から param.grad へ eager add する (950M param × micro-step 数の帯域往復) しかない。
// cuBLASLt は FP8 A/B + BF16 C/D で beta=1 を受けるので、dW GEMM の epilogue で
// 直接 gradient accumulation buffer へ足し込む。
//
// 計算 (row-major 表記): out[N, K] (+)= b_nm[N, M] @ a_km[K, M]^T
//   a_km: [K, M] e4m3、M 連続 (dW の X 側、転置済み)
//   b_nm: [N, M] e4m3、M 連続 (dW の dY 側、転置済み)
//   out : [N, K] bf16
// cuBLASLt (column-major) では D_cm[K, N] = op(A)[K, M] @ op(B)[M, N] で
//   A = a_km を col-major [M, K] (ld=M) と見て transa=T
//   B = b_nm を col-major [M, N] (ld=M) と見て transb=N
// という FP8 が要求する TN layout になる。
//
// このファイルは nvcc 不要 (host API のみ)。headers/libs は pip の nvidia/cu13 から。
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>

#include <cstdint>
#include <mutex>
#include <string>

namespace {

void check_lt(cublasStatus_t status, const char* what) {
    TORCH_CHECK(
        status == CUBLAS_STATUS_SUCCESS,
        "cublasLt ", what, " failed: ", cublasGetStatusString(status)
    );
}

struct LtDescs {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    ~LtDescs() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c) cublasLtMatrixLayoutDestroy(c);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (op) cublasLtMatmulDescDestroy(op);
    }
};

}  // namespace

void fp8_wgrad_lt(
    const torch::Tensor& a_km,
    const torch::Tensor& scale_a,
    const torch::Tensor& b_nm,
    const torch::Tensor& scale_b,
    torch::Tensor& out,
    bool accumulate
) {
    TORCH_CHECK(a_km.is_cuda() && b_nm.is_cuda() && out.is_cuda(), "fp8_wgrad_lt: CUDA tensors required");
    TORCH_CHECK(a_km.scalar_type() == at::kFloat8_e4m3fn, "fp8_wgrad_lt: a_km must be float8_e4m3fn");
    TORCH_CHECK(b_nm.scalar_type() == at::kFloat8_e4m3fn, "fp8_wgrad_lt: b_nm must be float8_e4m3fn");
    TORCH_CHECK(out.scalar_type() == at::kBFloat16, "fp8_wgrad_lt: out must be bfloat16");
    TORCH_CHECK(scale_a.scalar_type() == at::kFloat && scale_b.scalar_type() == at::kFloat,
                "fp8_wgrad_lt: scales must be float32");
    TORCH_CHECK(scale_a.is_cuda() && scale_b.is_cuda() && scale_a.numel() == 1 && scale_b.numel() == 1,
                "fp8_wgrad_lt: scales must be 0-d CUDA tensors");
    TORCH_CHECK(a_km.dim() == 2 && b_nm.dim() == 2 && out.dim() == 2, "fp8_wgrad_lt: 2-D tensors required");
    TORCH_CHECK(a_km.is_contiguous() && b_nm.is_contiguous() && out.is_contiguous(),
                "fp8_wgrad_lt: contiguous tensors required");
    const int64_t k = a_km.size(0);
    const int64_t m = a_km.size(1);
    const int64_t n = b_nm.size(0);
    TORCH_CHECK(b_nm.size(1) == m, "fp8_wgrad_lt: contraction dim mismatch: a_km ", a_km.sizes(), " b_nm ", b_nm.sizes());
    TORCH_CHECK(out.size(0) == n && out.size(1) == k, "fp8_wgrad_lt: out must be [N, K]=[", n, ",", k, "], got ", out.sizes());
    TORCH_CHECK(m % 16 == 0 && n % 16 == 0 && k % 16 == 0, "fp8_wgrad_lt: M/N/K must be multiples of 16");

    const c10::cuda::CUDAGuard guard(out.device());
    cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    LtDescs d;
    check_lt(cublasLtMatmulDescCreate(&d.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MatmulDescCreate");
    const cublasOperation_t ta = CUBLAS_OP_T;
    const cublasOperation_t tb = CUBLAS_OP_N;
    check_lt(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "TRANSA");
    check_lt(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "TRANSB");
    const void* sa = scale_a.data_ptr();
    const void* sb = scale_b.data_ptr();
    check_lt(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)), "A_SCALE");
    check_lt(cublasLtMatmulDescSetAttribute(d.op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)), "B_SCALE");

    // A: stored [M x K] col-major (ld=M), op(A)=A^T is [K x M]
    check_lt(cublasLtMatrixLayoutCreate(&d.a, CUDA_R_8F_E4M3, m, k, m), "layout A");
    // B: stored [M x N] col-major (ld=M), op(B)=B is [M x N]
    check_lt(cublasLtMatrixLayoutCreate(&d.b, CUDA_R_8F_E4M3, m, n, m), "layout B");
    // C/D: [K x N] col-major (ld=K) == out row-major [N, K]
    check_lt(cublasLtMatrixLayoutCreate(&d.c, CUDA_R_16BF, k, n, k), "layout C");

    const size_t ws_size = at::cuda::getCUDABlasLtWorkspaceSize();
    void* ws = at::cuda::getCUDABlasLtWorkspace();
    check_lt(cublasLtMatmulPreferenceCreate(&d.pref), "PreferenceCreate");
    check_lt(cublasLtMatmulPreferenceSetAttribute(
        d.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)), "MAX_WORKSPACE");

    cublasLtMatmulHeuristicResult_t heur{};
    int returned = 0;
    check_lt(cublasLtMatmulAlgoGetHeuristic(
        handle, d.op, d.a, d.b, d.c, d.c, d.pref, 1, &heur, &returned), "AlgoGetHeuristic");
    TORCH_CHECK(returned > 0, "fp8_wgrad_lt: no cublasLt algorithm for M=", m, " N=", n, " K=", k);

    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    check_lt(cublasLtMatmul(
        handle, d.op,
        &alpha,
        a_km.data_ptr(), d.a,
        b_nm.data_ptr(), d.b,
        &beta,
        out.data_ptr(), d.c,
        out.data_ptr(), d.c,
        &heur.algo, ws, ws_size, stream), "Matmul");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def(
        "fp8_wgrad_lt",
        &fp8_wgrad_lt,
        "out[N,K] (+)= b_nm[N,M] @ a_km[K,M]^T with tensorwise FP8 scales (cuBLASLt, beta=accumulate)"
    );
}
