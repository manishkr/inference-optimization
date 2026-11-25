#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>

#include "params.h"

namespace rkv {

// Scores are kept in the base-2 domain: s2 = (q . k) * log2(e) / sqrt(128).
constexpr float kScoreScale = 0.12751743082459868f;
constexpr float kLn2 = 0.6931471805599453f;

// Exact E4M3 decoding: move the 7 magnitude bits under the FP32 exponent and
// rebias by 2^120. E4M3 subnormals become FP32 subnormals, so this relies on
// denormals being preserved (no fast-math).
__device__ __forceinline__ float e4m3_to_float(uint32_t byte) {
  return __uint_as_float(((byte & 0x80u) << 24) | ((byte & 0x7Fu) << 20)) * 0x1p120f;
}

__device__ __forceinline__ float4 e4m3x4_to_float4(uint32_t word) {
  return make_float4(e4m3_to_float(word), e4m3_to_float(word >> 8), e4m3_to_float(word >> 16),
                     e4m3_to_float(word >> 24));
}

__device__ __forceinline__ float bf16_bits_to_float(uint32_t bits) {
  return __uint_as_float(bits << 16);
}

template <typename T>
__device__ __forceinline__ float load_float(const void* base, size_t index);

template <>
__device__ __forceinline__ float load_float<__half>(const void* base, size_t index) {
  return __half2float(static_cast<const __half*>(base)[index]);
}

template <>
__device__ __forceinline__ float load_float<__nv_bfloat16>(const void* base, size_t index) {
  return bf16_bits_to_float(static_cast<const uint16_t*>(base)[index]);
}

__device__ __forceinline__ float fast_exp2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_log2(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);
  return x;
}

// Base-2 scores of query i against new tokens j <= i for i in [i_begin, i_end),
// written to scores[i][j]. Called by the whole block.
template <typename T>
__device__ void new_token_scores(const Params& p, int row, int i_begin, int i_end,
                             float (*scores)[kMaxQueries]) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int pair = 0;
  for (int i = i_begin; i < i_end; ++i) {
    for (int j = 0; j <= i; ++j, ++pair) {
      if (pair % kWarps != warp) continue;
      const size_t qi = (static_cast<size_t>(row) * p.q + i) * kDim;
      const size_t kj = (static_cast<size_t>(row) * p.q + j) * kDim;
      float dot = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const int d = lane * 4 + c;
        dot = fmaf(load_float<T>(p.query, qi + d), load_float<T>(p.new_key, kj + d), dot);
      }
      dot = warp_sum(dot);
      if (lane == 0) scores[i][j] = dot * kScoreScale;
    }
  }
}

// Adds new tokens 0..i to the merged cached-token state of query i and
// writes the final output channel `ch` (and the log-sum-exp from channel 0).
template <typename T>
__device__ __forceinline__ void finish_query(const Params& p, int row, int i, int ch, float m,
                                               float l, float o, const float* scores_i) {
  float top = m;
  for (int j = 0; j <= i; ++j) top = fmaxf(top, scores_i[j]);
  const float carry = fast_exp2(m - top);
  float num = o * carry, den = l * carry;
  for (int j = 0; j <= i; ++j) {
    const float w = fast_exp2(scores_i[j] - top);
    num = fmaf(w, load_float<T>(p.new_value, (static_cast<size_t>(row) * p.q + j) * kDim + ch), num);
    den += w;
  }
  const size_t idx = static_cast<size_t>(row) * p.q + i;
  p.out[idx * kDim + ch] = num / den;
  if (ch == 0) p.lse[idx] = (top + fast_log2(den)) * kLn2;
}

// Per-warp online-softmax state of one partition: unnormalised outputs
// relative to each warp's running maximum.
struct BlockState {
  float o[kWarps][kMaxQueries][kDim];
  float m[kWarps][kMaxQueries];
  float l[kWarps][kMaxQueries];
};

// Merges the warps of a block, then either finishes the row (single
// partition) or stores the partition's partial result for the merge kernel.
template <typename T>
__device__ void finish_block(const Params& p, int row, int part, const BlockState& st,
                             const float (*scores)[kMaxQueries]) {
  const int ch = threadIdx.x;
  for (int i = 0; i < p.q; ++i) {
    float m = -INFINITY;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) m = fmaxf(m, st.m[w][i]);
    float l = 0.f, o = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      const float e = st.m[w][i] == -INFINITY ? 0.f : fast_exp2(st.m[w][i] - m);
      l = fmaf(st.l[w][i], e, l);
      o = fmaf(st.o[w][i][ch], e, o);
    }
    if (p.splits == 1) {
      finish_query<T>(p, row, i, ch, m, l, o, scores[i]);
    } else {
      const size_t slot = (static_cast<size_t>(row) * p.splits + part) * p.q + i;
      p.partial_o[slot * kDim + ch] = o;
      if (ch == 0) p.partial_ml[slot] = make_float2(m, l);
    }
  }
}

}  // namespace rkv
