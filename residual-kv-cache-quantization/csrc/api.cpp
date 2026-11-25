#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "backends.h"

namespace {

using rkv::kDim;
using rkv::kHeads;

constexpr int kMaxDevices = 64;
constexpr int kMaxBatch = 16;
constexpr int kPartitionGrain = 64;     // tokens; a multiple of every kernel's block tile
constexpr int kMinPartition = 256;      // shortest partition when a row is split
constexpr int kBlocksPerSm = 3;         // resident blocks per SM of the fast kernel
constexpr int kWaves = 2;

struct DeviceInfo {
  int major = 0;
  int minor = 0;
  int sms = 0;
};

DeviceInfo device_info(int device) {
  static std::mutex mutex;
  static std::array<DeviceInfo, kMaxDevices> cache{};
  TORCH_CHECK(device >= 0 && device < kMaxDevices, "unsupported device index ", device);
  std::lock_guard<std::mutex> lock(mutex);
  DeviceInfo& info = cache[device];
  if (info.sms == 0) {
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&info.major, cudaDevAttrComputeCapabilityMajor, device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&info.minor, cudaDevAttrComputeCapabilityMinor, device));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&info.sms, cudaDevAttrMultiProcessorCount, device));
  }
  return info;
}

bool fast_supported(int device) {
  const DeviceInfo info = device_info(device);
  return info.major * 10 + info.minor >= 75;
}

std::vector<std::string> available_backends(int device) {
  std::vector<std::string> names;
  if (fast_supported(device)) names.emplace_back("fast");
  names.emplace_back("portable");
  return names;
}

std::string backend_name(int device) { return fast_supported(device) ? "fast" : "portable"; }

// `alignment` is the data-pointer alignment the kernels' vector loads need:
// 16 bytes for code rows, 4 bytes for the packed scale and offset words.
void check_tensor(const at::Tensor& t, const char* name, at::ScalarType dtype,
                  const std::vector<int64_t>& shape, const at::Device& device, int alignment = 1) {
  TORCH_CHECK(t.defined(), name, " is undefined");
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.device() == device, name, " is on ", t.device(), " but query is on ", device);
  TORCH_CHECK(t.scalar_type() == dtype, name, " must have dtype ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.sizes() == at::IntArrayRef(shape), name, " must have shape ", at::IntArrayRef(shape),
              ", got ", t.sizes());
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(reinterpret_cast<std::uintptr_t>(t.data_ptr()) % alignment == 0, name,
              " must have a ", alignment, "-byte aligned data pointer");
}

rkv::Cache check_cache(const std::vector<at::Tensor>& cache, const char* name, int64_t batch,
                       int64_t n, const at::Device& device) {
  TORCH_CHECK(cache.size() == 7, name,
              " must be (anchor_codes, anchor_scale, anchor_offset, residual_codes, "
              "residual_scale, residual_offset, alpha), got ", cache.size(), " tensors");
  const std::string prefix(name);
  const auto f8 = at::kFloat8_e4m3fn;
  check_tensor(cache[0], (prefix + ".anchor_codes").c_str(), at::kByte, {batch, kHeads, n, 64}, device,
               16);
  check_tensor(cache[1], (prefix + ".anchor_scale").c_str(), f8, {batch, kHeads, n, 4}, device, 4);
  check_tensor(cache[2], (prefix + ".anchor_offset").c_str(), f8, {batch, kHeads, n, 4}, device, 4);
  check_tensor(cache[3], (prefix + ".residual_codes").c_str(), at::kByte, {batch, kHeads, 3, n, 32},
               device, 16);
  check_tensor(cache[4], (prefix + ".residual_scale").c_str(), f8, {batch, kHeads, 3, n, 4}, device, 4);
  check_tensor(cache[5], (prefix + ".residual_offset").c_str(), f8, {batch, kHeads, 3, n, 4}, device, 4);
  check_tensor(cache[6], (prefix + ".alpha").c_str(), at::kBFloat16, {batch, kHeads, 3, n}, device);
  rkv::Cache view;
  view.anchor_codes = static_cast<const uint8_t*>(cache[0].data_ptr());
  view.anchor_scale = static_cast<const uint8_t*>(cache[1].data_ptr());
  view.anchor_offset = static_cast<const uint8_t*>(cache[2].data_ptr());
  view.residual_codes = static_cast<const uint8_t*>(cache[3].data_ptr());
  view.residual_scale = static_cast<const uint8_t*>(cache[4].data_ptr());
  view.residual_offset = static_cast<const uint8_t*>(cache[5].data_ptr());
  view.alpha = static_cast<const uint16_t*>(cache[6].data_ptr());
  return view;
}

// Splits each row into partitions so that about kWaves waves of blocks cover
// the GPU, with partitions of at least kMinPartition tokens. Rows that fit in
// one partition are finished by a single kernel.
std::pair<int, int> plan_partitions(int rows, int n, int sms) {
  const int max_splits = (n + kMinPartition - 1) / kMinPartition;
  const int wanted = (kWaves * kBlocksPerSm * sms + rows - 1) / rows;
  int splits = std::max(1, std::min(wanted, max_splits));
  int part_len = (n + splits - 1) / splits;
  part_len = (part_len + kPartitionGrain - 1) / kPartitionGrain * kPartitionGrain;
  splits = (n + part_len - 1) / part_len;
  return {splits, part_len};
}

std::vector<at::Tensor> attention(const at::Tensor& query, const at::Tensor& new_key,
                                  const at::Tensor& new_value, int64_t loop,
                                  const std::vector<at::Tensor>& key_cache,
                                  const std::vector<at::Tensor>& value_cache,
                                  const at::Tensor& key_mean, const std::string& backend) {
  TORCH_CHECK(query.defined() && query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(query.dim() == 4, "query must have shape [B, 16, Q, 128], got ", query.sizes());
  const int64_t batch = query.size(0), queries = query.size(2);
  TORCH_CHECK(batch >= 1 && batch <= kMaxBatch, "batch size must be in [1, 16], got ", batch);
  TORCH_CHECK(query.size(1) == kHeads, "head count must be 16, got ", query.size(1));
  TORCH_CHECK(queries >= 1 && queries <= rkv::kMaxQueries, "query count must be in [1, 8], got ", queries);
  TORCH_CHECK(query.size(3) == kDim, "head dimension must be 128, got ", query.size(3));
  const auto dtype = query.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "query must be float16 or bfloat16, got ", dtype);
  TORCH_CHECK(loop >= 0 && loop <= rkv::kAnchorLoop, "loop must be 0, 1, 2 or 3, got ", loop);
  const at::Device device = query.device();
  const std::vector<int64_t> qshape{batch, kHeads, queries, kDim};
  check_tensor(query, "query", dtype, qshape, device);
  check_tensor(new_key, "new_key", dtype, qshape, device);
  check_tensor(new_value, "new_value", dtype, qshape, device);
  TORCH_CHECK(key_cache.size() == 7 && key_cache[0].defined() && key_cache[0].dim() == 4,
              "key_cache must be a tuple of 7 tensors with anchor_codes of shape [B, 16, N, 64]");
  const int64_t n = key_cache[0].size(2);
  TORCH_CHECK(n >= 1 && n <= rkv::kMaxLength, "cached length must be in [1, 32768], got ", n);
  rkv::Params p{};
  p.key = check_cache(key_cache, "key_cache", batch, n, device);
  p.value = check_cache(value_cache, "value_cache", batch, n, device);
  check_tensor(key_mean, "key_mean", at::kFloat, {kHeads, 4, kDim}, device);

  const int device_index = device.index();
  const bool use_fast = backend == "fast" || (backend.empty() && fast_supported(device_index));
  TORCH_CHECK(backend.empty() || backend == "fast" || backend == "portable",
              "unknown backend '", backend, "'");
  TORCH_CHECK(!use_fast || fast_supported(device_index),
              "fast backend needs compute capability 7.5 or newer");

  const c10::cuda::CUDAGuard guard(device);
  const auto f32 = query.options().dtype(at::kFloat);
  at::Tensor out = at::empty({batch, kHeads, queries, kDim}, f32);
  at::Tensor lse = at::empty({batch, kHeads, queries}, f32);

  const int rows = static_cast<int>(batch * kHeads);
  const auto [splits, part_len] = plan_partitions(rows, static_cast<int>(n), device_info(device_index).sms);
  at::Tensor partial_o, partial_ml;
  if (splits > 1) {
    partial_o = at::empty({rows, splits, queries, kDim}, f32);
    partial_ml = at::empty({rows, splits, queries, 2}, f32);
  }

  p.query = query.data_ptr();
  p.new_key = new_key.data_ptr();
  p.new_value = new_value.data_ptr();
  p.key_mean = key_mean.data_ptr<float>();
  p.out = out.data_ptr<float>();
  p.lse = lse.data_ptr<float>();
  p.partial_o = splits > 1 ? partial_o.data_ptr<float>() : nullptr;
  p.partial_ml = splits > 1 ? reinterpret_cast<float2*>(partial_ml.data_ptr<float>()) : nullptr;
  p.rows = rows;
  p.n = static_cast<int>(n);
  p.q = static_cast<int>(queries);
  p.loop = static_cast<int>(loop);
  p.splits = splits;
  p.part_len = part_len;

  const auto type = dtype == at::kHalf ? rkv::InputType::kHalf : rkv::InputType::kBFloat16;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device_index).stream();
  if (use_fast) {
    rkv::fast::launch(p, type, stream);
  } else {
    rkv::portable::launch(p, type, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (splits > 1) {
    rkv::launch_merge(p, type, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {out, lse};
}

std::vector<int64_t> partitions(int64_t batch, int64_t n, int64_t device) {
  TORCH_CHECK(batch >= 1 && batch <= kMaxBatch && n >= 1 && n <= rkv::kMaxLength,
              "batch must be in [1, 16] and length in [1, 32768]");
  const auto [splits, part_len] = plan_partitions(static_cast<int>(batch * kHeads), static_cast<int>(n),
                                                  device_info(static_cast<int>(device)).sms);
  return {splits, part_len};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention", &attention, "ResidualQuant attention; backend '' picks the best for the device");
  m.def("backend", &backend_name, "Backend that attention() uses on a device by default");
  m.def("available_backends", &available_backends, "Backends that can run on a device");
  m.def("partitions", &partitions, "(splits, tokens per partition) chosen for a batch and length");
}
