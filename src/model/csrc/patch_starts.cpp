#include <torch/extension.h>

torch::Tensor patch_starts_cuda(
    torch::Tensor raw, torch::Tensor force, torch::Tensor char_start, torch::Tensor info,
    int64_t min_len, int64_t max_len, int64_t budget, int64_t horizon, int64_t soft_len,
    int64_t reserve, double info_min, double info_max);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "patch_starts_cuda",
        &patch_starts_cuda,
        "Compute dynamic patch start positions (min/max length, forced starts, causal budget) (CUDA)"
    );
}
