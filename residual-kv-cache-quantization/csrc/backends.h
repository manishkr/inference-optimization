#pragma once

#include <cuda_runtime.h>

#include "params.h"

namespace rkv {

enum class InputType { kHalf, kBFloat16 };

namespace fast {
// Tensor-core kernels; need sm_75 or newer.
void launch(const Params& p, InputType type, cudaStream_t stream);
}  // namespace fast

namespace portable {
// CUDA-core FP32 kernels.
void launch(const Params& p, InputType type, cudaStream_t stream);
}  // namespace portable

// Combines the partitions of every row with the new tokens (splits > 1).
void launch_merge(const Params& p, InputType type, cudaStream_t stream);

}  // namespace rkv
