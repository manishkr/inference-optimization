#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <mutex>

#include "backends.h"

namespace portable {

constexpr int D = 128;
constexpr int BLOCK = 256;
constexpr int WINDOW = 64;
constexpr float EPS = 1e-12f;

constexpr int TILE = 128;
constexpr int HALF = TILE / 2;
constexpr int SLOTS = 3;
constexpr int WARPS = 16;
constexpr int THREADS = WARPS * 32;
constexpr int CHUNK = TILE / WARPS;
constexpr int CHUNKS_PER_TILE = TILE / CHUNK;
constexpr int WINDOW_CHUNKS = WINDOW / CHUNK;
constexpr int CHUNK_SLOTS = (SLOTS * HALF) / CHUNK;
constexpr int LOADS = TILE * D * 2 / 16 / THREADS;
static_assert(HALF == WINDOW && CHUNK == 8, "a half tile spans the window; chunks are scored in batches of 8");
static_assert(BLOCK % TILE == 0 && HALF % CHUNK == 0, "a tile lies within one block; chunks within a half");
static_assert(TILE * D * 2 / 16 % THREADS == 0 && THREADS % 32 == 0, "tile rows split evenly");

__device__ __forceinline__ void convert4(uint2 raw, c10::BFloat16, float (&v)[4]) {
  v[0] = __uint_as_float(raw.x << 16);
  v[1] = __uint_as_float(raw.x & 0xffff0000u);
  v[2] = __uint_as_float(raw.y << 16);
  v[3] = __uint_as_float(raw.y & 0xffff0000u);
}

__device__ __forceinline__ void convert4(uint2 raw, c10::Half, float (&v)[4]) {
  float2 low = __half22float2(*reinterpret_cast<const __half2*>(&raw.x));
  float2 high = __half22float2(*reinterpret_cast<const __half2*>(&raw.y));
  v[0] = low.x; v[1] = low.y; v[2] = high.x; v[3] = high.y;
}

template <typename T>
__device__ __forceinline__ void load4(const T* row, int lane, bool vector, float (&v)[4]) {
  const T* p = row + lane * 4;
  if (vector) {
    convert4(*reinterpret_cast<const uint2*>(p), T{}, v);
  } else {
    for (int j = 0; j < 4; ++j) v[j] = static_cast<float>(p[j]);
  }
}

__device__ __forceinline__ float warp_all(float value) {
  for (int offset = 16; offset; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  return value;
}

__device__ __forceinline__ float feature_norm(float value, float* scratch) {
  if (threadIdx.x < D) {
    float square = warp_all(value * value);
    if ((threadIdx.x & 31) == 0) scratch[threadIdx.x >> 5] = square;
  }
  __syncthreads();
  return sqrtf(scratch[0] + scratch[1] + scratch[2] + scratch[3]);
}

template <int LANES>
__device__ __forceinline__ void scatter(float* values, int lane) {
  bool upper = lane & LANES;
#pragma unroll
  for (int j = 0; j < LANES; ++j) {
    float send = upper ? values[j] : values[j + LANES];
    float keep = upper ? values[j + LANES] : values[j];
    values[j] = keep + __shfl_xor_sync(0xffffffff, send, LANES);
  }
}

constexpr int SUM_WARPS = 16;
template <typename T>
__global__ void __launch_bounds__(SUM_WARPS * 32) kv_block_sums(const T* keys, float* blocks, float* inverse,
                                                                int n, int nb, bool vector) {
  __shared__ float partial[SUM_WARPS][D];
  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int group = blockIdx.x / nb, start = (blockIdx.x % nb) * BLOCK;
  int first = start + warp * (BLOCK / SUM_WARPS), last = min(n, first + BLOCK / SUM_WARPS);
  float sum[4] = {0, 0, 0, 0};
  const T* row = keys + (static_cast<int64_t>(group) * n + first) * D;
  float* scales = inverse + static_cast<int64_t>(group) * n + first;

  constexpr int AHEAD = 8;
  for (int base = 0; base < last - first; base += AHEAD) {
    int count = min(AHEAD, last - first - base);
    float v[AHEAD][4], squares[AHEAD];
#pragma unroll
    for (int k = 0; k < AHEAD; ++k) {
      if (k < count) load4(row + (base + k) * D, lane, vector, v[k]);
      else for (int j = 0; j < 4; ++j) v[k][j] = 0.0f;
      squares[k] = v[k][0] * v[k][0] + v[k][1] * v[k][1] + v[k][2] * v[k][2] + v[k][3] * v[k][3];
    }
    // After the transposed reduction lane l holds the squared norm of row l % AHEAD.
    scatter<4>(squares, lane);
    scatter<2>(squares, lane);
    scatter<1>(squares, lane);
    float total = squares[0];
    total += __shfl_xor_sync(0xffffffff, total, 8);
    total += __shfl_xor_sync(0xffffffff, total, 16);
    float inv = 1.0f / fmaxf(sqrtf(total), EPS);
    if (lane < count) scales[base + lane] = inv;
#pragma unroll
    for (int k = 0; k < AHEAD; ++k) {
      float scale = __shfl_sync(0xffffffff, inv, k);
      for (int j = 0; j < 4; ++j) sum[j] = fmaf(v[k][j], scale, sum[j]);
    }
  }
  for (int j = 0; j < 4; ++j) partial[warp][lane * 4 + j] = sum[j];
  __syncthreads();
  if (threadIdx.x < D) {
    float total = 0;
    for (int w = 0; w < SUM_WARPS; ++w) total += partial[w][threadIdx.x];
    blocks[static_cast<int64_t>(blockIdx.x) * D + threadIdx.x] = total;
  }
}

// Scores CHUNK consecutive rows given as converted values `v` with reciprocal norms `s`; `old`
// holds the rows WINDOW positions earlier. `window` enters as the sum of the WINDOW - 1 normalized
// rows before the first one.
template <bool FULL, typename T>
__device__ __forceinline__ void score_chunk(const float (&v)[CHUNK][4], const float (&s)[CHUNK], const T* old,
                                            const float* old_scale, int count, int first,
                                            const float (&g)[4], const float (&b)[4],
                                            float (&window)[4], float* out, int total_count, int lane) {
  float values[CHUNK * 4];
#pragma unroll
  for (int t = 0; t < CHUNK; ++t) {
    float gs = 0, bs = 0, ls = 0, ww = 0;
    if (FULL || t < count) {
      for (int j = 0; j < 4; ++j) window[j] = fmaf(v[t][j], s[t], window[j]);
      if (t > 0 && (FULL || first + t - WINDOW >= 0)) {
        float o[4], so = old_scale[t];
        load4(old + t * D, lane, true, o);
        for (int j = 0; j < 4; ++j) window[j] = fmaf(-o[j], so, window[j]);
      }
      for (int j = 0; j < 4; ++j) {
        gs += v[t][j] * g[j];
        bs += v[t][j] * b[j];
        ls += v[t][j] * window[j];
        ww += window[j] * window[j];
      }
    }
    values[4 * t] = gs; values[4 * t + 1] = bs; values[4 * t + 2] = ls; values[4 * t + 3] = ww;
  }
  scatter<16>(values, lane);
  scatter<8>(values, lane);
  scatter<4>(values, lane);
  scatter<2>(values, lane);
  scatter<1>(values, lane);
  float total = values[0];
  float ww = __shfl_down_sync(0xffffffff, total, 1);
  int t = lane / 4, quantity = lane % 4;
  float scale = s[0];
#pragma unroll
  for (int k = 1; k < CHUNK; ++k) scale = t == k ? s[k] : scale;
  if ((FULL || t < count) && quantity < 3) {
    total *= scale;
    float score = total;
    if (quantity == 2) score = total / fmaxf(sqrtf(ww), min(WINDOW, first + t + 1) * EPS);
    out[quantity * static_cast<int64_t>(total_count) + t] = -score;
  }
}

constexpr int SLICES = THREADS / D;
constexpr int RING_BYTES = SLOTS * HALF * D * 2 + SLOTS * HALF * 4 + CHUNK_SLOTS * D * 4;
constexpr int STATIC_BYTES = (D + SLICES * D + 4) * 4;

// One CTA scores a contiguous run of tiles, one tile per iteration: place the prefetched rows,
// prefetch the next tile, publish chunk sums, then score. Rows live in a ring of three half-tile
// slots, so the WINDOW rows before each tile are already on chip and every key is read once.
template <typename T>
__global__ void __launch_bounds__(THREADS, 1) kv_stream_scores(const T* keys, const float* inverse,
                                                               const float* blocks, float* out, int n,
                                                               int nb, int nt, int tiles_per_cta,
                                                               int total_tiles, int total_count, bool vector) {
  extern __shared__ __align__(16) unsigned char storage[];
  T* rows = reinterpret_cast<T*>(storage);
  float* scales = reinterpret_cast<float*>(storage + SLOTS * HALF * D * 2);
  float* chunks = scales + SLOTS * HALF;
  __shared__ float global_unit[D], partial[SLICES][D], scratch[4];

  int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int begin = blockIdx.x * tiles_per_cta, finish = min(total_tiles, begin + tiles_per_cta);
  if (begin >= finish) return;

  auto slot_rows = [&](int half) { return rows + (half % SLOTS) * HALF * D; };
  auto slot_scales = [&](int half) { return scales + (half % SLOTS) * HALF; };
  auto chunk_at = [&](int sequence) { return chunks + (sequence % CHUNK_SLOTS) * D + lane * 4; };

  uint4 held[LOADS];
  float held_scale = 0.0f;
  auto fetch = [&](int tile) {
    int group = tile / nt, start = (tile % nt) * TILE, count = min(TILE, n - start);
    int64_t base = static_cast<int64_t>(group) * n + start;
    if (vector) {
      const uint4* from = reinterpret_cast<const uint4*>(keys + base * D);
      int limit = count * D * 2 / 16;
#pragma unroll
      for (int k = 0; k < LOADS; ++k) {
        int at = threadIdx.x + k * THREADS;
        if (at < limit) held[k] = from[at];
      }
    }
    if (threadIdx.x < count) held_scale = inverse[base + threadIdx.x];
  };
  // Writes the fetched tile into halves 2 * step and 2 * step + 1 of the ring.
  auto place = [&](int tile, int step) {
    int group = tile / nt, start = (tile % nt) * TILE, count = min(TILE, n - start);
    if (vector) {
      int limit = count * D * 2 / 16;
#pragma unroll
      for (int k = 0; k < LOADS; ++k) {
        int at = threadIdx.x + k * THREADS, row = at / (D * 2 / 16);
        uint4* to = reinterpret_cast<uint4*>(slot_rows(2 * step + row / HALF));
        if (at < limit) to[at - (row / HALF) * HALF * (D * 2 / 16)] = held[k];
      }
    } else {
      const T* from = keys + (static_cast<int64_t>(group) * n + start) * D;
      for (int k = threadIdx.x; k < count * D; k += THREADS) {
        int row = k / D;
        slot_rows(2 * step + row / HALF)[k - (row / HALF) * HALF * D] = from[k];
      }
    }
    if (threadIdx.x < count)
      slot_scales(2 * step + threadIdx.x / HALF)[threadIdx.x % HALF] = held_scale;
  };

  float g[4], b[4];
  auto load_global = [&](int group) {
    int d = threadIdx.x % D, slice = threadIdx.x / D;
    const float* sums = blocks + static_cast<int64_t>(group) * nb * D + d;
    constexpr int AHEAD = 8;
    float acc[AHEAD] = {};
    int block = slice;
    for (; block + (AHEAD - 1) * SLICES < nb; block += AHEAD * SLICES)
#pragma unroll
      for (int k = 0; k < AHEAD; ++k) acc[k] += sums[(block + k * SLICES) * D];
    for (; block < nb; block += SLICES) acc[0] += sums[block * D];
    partial[slice][d] = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    __syncthreads();
    float total = 0;
    for (int s = 0; s < SLICES; ++s) total += partial[s][d];
    float norm = feature_norm(total, scratch);
    if (threadIdx.x < D) global_unit[d] = total / fmaxf(norm, n * EPS);
    __syncthreads();
    for (int j = 0; j < 4; ++j) g[j] = global_unit[lane * 4 + j];
  };

  // Step 0 is reserved for the WINDOW rows before the first tile; half 1 holds them.
  if (begin % nt > 0) {
    int64_t base = static_cast<int64_t>(begin / nt) * n + (begin % nt) * TILE - HALF;
    T* to = slot_rows(1);
    if (vector) {
      const uint4* from = reinterpret_cast<const uint4*>(keys + base * D);
      for (int k = threadIdx.x; k < HALF * D * 2 / 16; k += THREADS) reinterpret_cast<uint4*>(to)[k] = from[k];
    } else {
      for (int k = threadIdx.x; k < HALF * D; k += THREADS) to[k] = keys[base * D + k];
    }
    if (threadIdx.x < HALF) slot_scales(1)[threadIdx.x] = inverse[base + threadIdx.x];
    __syncthreads();
    if (warp < WINDOW_CHUNKS) {
      const T* cur = to + warp * CHUNK * D;
      const float* scale = slot_scales(1) + warp * CHUNK;
      float total[4] = {0, 0, 0, 0};
#pragma unroll
      for (int k = 0; k < CHUNK; ++k) {
        float v[4];
        load4(cur + k * D, lane, true, v);
        for (int j = 0; j < 4; ++j) total[j] = fmaf(v[j], scale[k], total[j]);
      }
      float* sum = chunk_at(CHUNKS_PER_TILE - WINDOW_CHUNKS + warp);
      for (int j = 0; j < 4; ++j) sum[j] = total[j];
    }
  }
  fetch(begin);

  int current_group = -1, current_block = -1;
  for (int tile = begin, step = 1; tile < finish; ++tile, ++step) {
    int group = tile / nt, t = tile % nt;
    place(tile, step);
    if (tile + 1 < finish) fetch(tile + 1);
    int index = t * TILE / BLOCK;
    bool new_block = group != current_group || index != current_block;
    float block_sum[4];
    if (new_block) {
      const float* from = blocks + (static_cast<int64_t>(group) * nb + index) * D + lane * 4;
      for (int j = 0; j < 4; ++j) block_sum[j] = from[j];
    }
    if (group != current_group) {
      current_group = group;
      load_global(group);
    }
    current_block = index;
    __syncthreads();
    int first = t * TILE + warp * CHUNK, count = max(0, min(CHUNK, n - first));
    int half = 2 * step + warp * CHUNK / HALF, offset = warp * CHUNK % HALF;
    // Each warp converts its chunk once, keeps it in registers for scoring and publishes its sum.
    float v[CHUNK][4], s[CHUNK];
    {
      const T* cur = slot_rows(half) + offset * D;
      const float* cur_scale = slot_scales(half) + offset;
      float total[4] = {0, 0, 0, 0};
#pragma unroll
      for (int k = 0; k < CHUNK; ++k) {
        bool valid = k < count;
        s[k] = valid ? cur_scale[k] : 0.0f;
        if (valid) load4(cur + k * D, lane, true, v[k]);
        else for (int j = 0; j < 4; ++j) v[k][j] = 0.0f;
        for (int j = 0; j < 4; ++j) total[j] = fmaf(v[k][j], s[k], total[j]);
      }
      float* to = chunk_at(CHUNKS_PER_TILE * step + warp);
      for (int j = 0; j < 4; ++j) to[j] = total[j];
    }
    __syncthreads();

    if (new_block) {
      int members = min(BLOCK, n - index * BLOCK);
      float square = 0;
      for (int j = 0; j < 4; ++j) square += block_sum[j] * block_sum[j];
      float inv = 1.0f / fmaxf(sqrtf(warp_all(square)), members * EPS);
      for (int j = 0; j < 4; ++j) b[j] = block_sum[j] * inv;
    }
    if (count > 0) {
      // The window before the chunk is the sum of the previous WINDOW_CHUNKS chunks minus its oldest row.
      int sequence = CHUNKS_PER_TILE * step + warp;
      int available = min(WINDOW_CHUNKS, first / CHUNK);
      float window[4] = {0, 0, 0, 0};
      for (int k = 1; k <= available; ++k) {
        const float* from = chunk_at(sequence - k);
        for (int j = 0; j < 4; ++j) window[j] += from[j];
      }
      const T* old = slot_rows(half + SLOTS - 1) + offset * D;
      const float* old_scale = slot_scales(half + SLOTS - 1) + offset;
      if (first >= WINDOW) {
        float o[4];
        load4(old, lane, true, o);
        for (int j = 0; j < 4; ++j) window[j] = fmaf(-o[j], old_scale[0], window[j]);
      }
      float* to = out + static_cast<int64_t>(group) * n + first;
      if (first >= WINDOW && count == CHUNK)
        score_chunk<true>(v, s, old, old_scale, CHUNK, first, g, b, window, to, total_count, lane);
      else
        score_chunk<false>(v, s, old, old_scale, count, first, g, b, window, to, total_count, lane);
    }
    __syncthreads();
  }
}

size_t shared_bytes_required() { return RING_BYTES + STATIC_BYTES; }

// Number of scoring CTAs that are resident at once; each gets an equal share of the tiles.
template <typename T>
int stream_capacity(int device) {
  static std::once_flag flags[64];
  static int capacity[64];
  TORCH_CHECK(device >= 0 && device < 64, "unsupported device index ", device);
  std::call_once(flags[device], [device] {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kv_stream_scores<T>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, RING_BYTES));
    int per_sm = 0, sms = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kv_stream_scores<T>,
                                                                 THREADS, RING_BYTES));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    capacity[device] = std::max(1, per_sm * sms);
  });
  return capacity[device];
}

torch::Tensor scores(const torch::Tensor& keys) {
  int n = keys.size(2), groups = keys.size(0) * keys.size(1);
  int nb = (n + BLOCK - 1) / BLOCK, nt = (n + TILE - 1) / TILE;
  int device = keys.get_device();
  auto options = keys.options().dtype(at::kFloat);
  auto out = torch::empty({3, keys.size(0), keys.size(1), n}, options);
  auto blocks = torch::empty({groups, nb, D}, options);
  auto inverse = torch::empty({groups, n}, options);
  auto stream = c10::cuda::getCurrentCUDAStream(device);
  bool vector = reinterpret_cast<std::uintptr_t>(keys.data_ptr()) % 16 == 0;
  auto run = [&](auto tag) {
    using T = decltype(tag);
    int capacity = stream_capacity<T>(device);
    int total_tiles = groups * nt;
    int per_cta = (total_tiles + capacity - 1) / capacity;
    int ctas = (total_tiles + per_cta - 1) / per_cta;
    const T* data = keys.data_ptr<T>();
    float *b = blocks.data_ptr<float>(), *s = inverse.data_ptr<float>(), *o = out.data_ptr<float>();
    kv_block_sums<T><<<groups * nb, SUM_WARPS * 32, 0, stream>>>(data, b, s, n, nb, vector);
    kv_stream_scores<T><<<ctas, THREADS, RING_BYTES, stream>>>(
        data, s, b, o, n, nb, nt, per_cta, total_tiles, groups * n, vector);
  };
  if (keys.scalar_type() == at::kHalf) run(c10::Half{});
  else run(c10::BFloat16{});
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace portable
