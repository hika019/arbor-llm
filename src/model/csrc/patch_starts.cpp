#include <torch/extension.h>

torch::Tensor patch_starts_cuda(torch::Tensor raw, int64_t min_len, int64_t max_len);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "patch_starts_cuda",
        &patch_starts_cuda,
        "Compute dynamic patch start positions from raw boundary candidates (CUDA)"
    );
}
