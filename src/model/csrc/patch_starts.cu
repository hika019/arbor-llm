#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace {

// 境界規則は src/model/arbor.py の _patch_starts_reference (CPU) と完全に同じ。
//   - patch は最長 max_len。force[p] (文書先頭) は min_len を無視して区切る候補。
//   - raw[p] は run >= min_len のときだけ候補。
//   - budget > 0 なら、位置 p で新 patch を開けるのは
//       count + ceil((horizon - p) / max_len) <= budget
//     のときだけ (count = それまでに開いた patch 数)。残りを max_len ずつ区切っても
//     上限に収まることを常に保証するので、patch 数は budget を超えない。判定は
//     位置と過去の境界だけで決まる (因果的)。max_len 到達の強制境界はこの不変条件から
//     常に許される。
__global__ void patch_starts_kernel(
    const bool* __restrict__ raw,
    const bool* __restrict__ force,
    bool* __restrict__ starts,
    int64_t rows,
    int64_t cols,
    int64_t min_len,
    int64_t max_len,
    int64_t budget,
    int64_t horizon
) {
    int64_t row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) {
        return;
    }

    const int64_t base = row * cols;
    int64_t i = 0;
    int64_t count = 0;
    while (i < cols) {
        starts[base + i] = true;
        ++count;

        const int64_t hi = min(i + max_len, cols);
        const int64_t lo = i + min_len;
        int64_t next = hi;
        for (int64_t p = i + 1; p < hi; ++p) {
            const bool cand = force[base + p] || (p >= lo && raw[base + p]);
            if (!cand) {
                continue;
            }
            if (budget > 0) {
                const int64_t rem = max(horizon - p, (int64_t)0);
                if (count + (rem + max_len - 1) / max_len > budget) {
                    continue;
                }
            }
            next = p;
            break;
        }
        i = next;
    }
}

}  // namespace

torch::Tensor patch_starts_cuda(
    torch::Tensor raw, torch::Tensor force, int64_t min_len, int64_t max_len,
    int64_t budget, int64_t horizon
) {
    TORCH_CHECK(raw.is_cuda() && force.is_cuda(), "raw/force must be CUDA tensors");
    TORCH_CHECK(raw.scalar_type() == torch::kBool && force.scalar_type() == torch::kBool,
                "raw/force must be bool");
    TORCH_CHECK(raw.dim() == 2, "raw must be rank-2 (B, T)");
    TORCH_CHECK(raw.sizes() == force.sizes(), "raw/force shape mismatch");
    TORCH_CHECK(raw.is_contiguous() && force.is_contiguous(), "raw/force must be contiguous");
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
        force.data_ptr<bool>(),
        starts.data_ptr<bool>(),
        rows,
        cols,
        min_len,
        max_len,
        budget,
        horizon
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return starts;
}
