#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "backends.h"

namespace {

constexpr int kMaxDevices = 64;

int optin_shared_bytes(int device) {
  static std::mutex mutex;
  static std::array<int, kMaxDevices> cache = [] {
    std::array<int, kMaxDevices> values;
    values.fill(-1);
    return values;
  }();
  TORCH_CHECK(device >= 0 && device < kMaxDevices, "unsupported device index ", device);
  std::lock_guard<std::mutex> lock(mutex);
  if (cache[device] < 0) {
    int value = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &value, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    cache[device] = value;
  }
  return cache[device];
}

bool fast_supported(int device) {
  return static_cast<size_t>(optin_shared_bytes(device)) >= fast::shared_bytes_required();
}

bool portable_supported(int device) {
  return static_cast<size_t>(optin_shared_bytes(device)) >= portable::shared_bytes_required();
}

bool aligned16(const torch::Tensor& keys) {
  return reinterpret_cast<std::uintptr_t>(keys.data_ptr()) % 16 == 0;
}

void check_keys(const torch::Tensor& keys) {
  TORCH_CHECK(keys.is_cuda(), "keys must be a CUDA tensor");
  TORCH_CHECK(keys.dim() == 4, "keys must have shape [B, H, N, 128]");
  TORCH_CHECK(keys.size(0) == 1 || keys.size(0) == 2, "batch size must be 1 or 2");
  TORCH_CHECK(keys.size(1) == 4 || keys.size(1) == 8, "head count must be 4 or 8");
  TORCH_CHECK(keys.size(2) >= 1 && keys.size(2) <= 32768,
              "sequence length must be in [1, 32768]");
  TORCH_CHECK(keys.size(3) == 128, "head dimension must be 128");
  TORCH_CHECK(keys.scalar_type() == at::kHalf || keys.scalar_type() == at::kBFloat16,
              "keys must be float16 or bfloat16");
  TORCH_CHECK(keys.is_contiguous(), "keys must be contiguous");
}

std::vector<std::string> available_backends(int device) {
  std::vector<std::string> names;
  if (fast_supported(device)) names.emplace_back("fast");
  if (portable_supported(device)) names.emplace_back("portable");
  return names;
}

std::string backend_name(int device) {
  if (fast_supported(device)) return "fast";
  if (portable_supported(device)) return "portable";
  return "unsupported";
}

torch::Tensor scores_fast(const torch::Tensor& keys) {
  check_keys(keys);
  const int device = keys.get_device();
  TORCH_CHECK(fast_supported(device), "fast backend needs ", fast::shared_bytes_required(),
              " bytes of shared memory per block; device ", device, " allows ",
              optin_shared_bytes(device));
  TORCH_CHECK(aligned16(keys), "fast backend needs a 16-byte aligned data pointer");
  const c10::cuda::CUDAGuard guard(keys.device());
  return fast::scores(keys);
}

torch::Tensor scores_portable(const torch::Tensor& keys) {
  check_keys(keys);
  const int device = keys.get_device();
  TORCH_CHECK(portable_supported(device), "portable backend needs ",
              portable::shared_bytes_required(), " bytes of shared memory per block; device ",
              device, " allows ", optin_shared_bytes(device));
  const c10::cuda::CUDAGuard guard(keys.device());
  return portable::scores(keys);
}

torch::Tensor scores(const torch::Tensor& keys) {
  check_keys(keys);
  if (fast_supported(keys.get_device()) && aligned16(keys)) {
    const c10::cuda::CUDAGuard guard(keys.device());
    return fast::scores(keys);
  }
  return scores_portable(keys);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scores", &scores, "Global, block and local scores using the best backend for the device");
  m.def("scores_fast", &scores_fast, "Scores using the large shared-memory kernels");
  m.def("scores_portable", &scores_portable, "Scores using the portable kernels");
  m.def("backend", &backend_name, "Backend that scores() uses on a device");
  m.def("available_backends", &available_backends, "Backends that can run on a device");
}
