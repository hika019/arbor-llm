#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

__global__ void patch_starts_kernel(
    const bool* __restrict__ raw,
    bool* __restrict__ starts,
    int64_t rows,
    int64_t cols,
    int64_t min_len,
    int64_t max_len
) {
    int64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) {
        return;
    }

    const int64_t base = row * cols;
    int64_t i = 0;
    while (i < cols) {
        starts[base + i] = true;

        const int64_t lo = i + min_len;
        if (lo >= cols) {
            break;
        }

        const int64_t hi = min(i + max_len, cols);
        int64_t next = hi;
        for (int64_t p = lo; p < hi; ++p) {
            if (raw[base + p]) {
                next = p;
                break;
            }
        }
        i = next;
    }
}

}  // namespace

torch::Tensor patch_starts_cuda(torch::Tensor raw, int64_t min_len, int64_t max_len) {
    TORCH_CHECK(raw.is_cuda(), "raw must be a CUDA tensor");
    TORCH_CHECK(raw.scalar_type() == torch::kBool, "raw must be bool");
    TORCH_CHECK(raw.dim() == 2, "raw must be rank-2 (B, T)");
    TORCH_CHECK(raw.is_contiguous(), "raw must be contiguous");
    TORCH_CHECK(min_len > 0, "min_len must be positive");
    TORCH_CHECK(max_len >= min_len, "max_len must be >= min_len");

    const auto rows = raw.size(0);
    const auto cols = raw.size(1);
    auto starts = torch::zeros_like(raw);
    if (rows == 0 || cols == 0) {
        return starts;
    }

    const c10::cuda::CUDAGuard device_guard(raw.device());
    patch_starts_kernel<<<rows, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        raw.data_ptr<bool>(),
        starts.data_ptr<bool>(),
        rows,
        cols,
        min_len,
        max_len
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return starts;
}
