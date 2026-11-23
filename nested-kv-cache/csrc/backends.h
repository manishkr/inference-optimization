#pragma once

#include <torch/extension.h>

#include <cstddef>

namespace fast {
size_t shared_bytes_required();
torch::Tensor scores(const torch::Tensor& keys);
}  // namespace fast

namespace portable {
size_t shared_bytes_required();
torch::Tensor scores(const torch::Tensor& keys);
}  // namespace portable
