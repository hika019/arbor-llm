#include <torch/extension.h>

torch::Tensor patch_starts_cuda(
    torch::Tensor raw, torch::Tensor force, int64_t min_len, int64_t max_len,
    int64_t budget, int64_t horizon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "patch_starts_cuda",
        &patch_starts_cuda,
        "Compute dynamic patch start positions (min/max length, forced starts, causal budget) (CUDA)"
    );
}
