// CUDA-core backend. Each (row, partition) gets a block of four warps; warp w
// handles tokens w, w + 4, w + 8, ... of the partition. A lane rebuilds four
// consecutive channels of each key and value in FP32, and the query-key dot
// products are summed with warp shuffles.
#include "backends.h"
#include "common.cuh"

namespace rkv {
namespace portable {
namespace {

template <typename T, bool kResidual, int Q>
__global__ void __launch_bounds__(kThreads) attention_kernel(Params p) {
  const int part = blockIdx.x, row = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int head = row % kHeads, group = lane >> 3, ch0 = lane * 4;
  const int begin = part * p.part_len, end = min(p.n, begin + p.part_len);

  __shared__ BlockState state;
  __shared__ float scores[kMaxQueries][kMaxQueries];
  if (p.splits == 1) new_token_scores<T>(p, row, 0, Q, scores);

  float q[Q][4], acc[Q][4], m[Q], l[Q], mean3[4], meanr[4];
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    mean3[c] = p.key_mean[(head * 4 + kAnchorLoop) * kDim + ch0 + c];
    meanr[c] = kResidual ? p.key_mean[(head * 4 + p.loop) * kDim + ch0 + c] : 0.f;
  }
#pragma unroll
  for (int i = 0; i < Q; ++i) {
    m[i] = -INFINITY;
    l[i] = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      q[i][c] = load_float<T>(p.query, (static_cast<size_t>(row) * Q + i) * kDim + ch0 + c);
      acc[i][c] = 0.f;
    }
  }

  const size_t row_tokens = static_cast<size_t>(row) * p.n;
  const size_t loop_tokens = (static_cast<size_t>(row) * kResidualLoops + p.loop) * p.n;
#pragma unroll 1
  for (int n = begin + warp; n < end; n += kWarps) {
    const size_t at = row_tokens + n, rt = loop_tokens + n;
    const uint32_t kcode = *reinterpret_cast<const uint16_t*>(p.key.anchor_codes + at * 64 + lane * 2);
    const uint32_t vcode = *reinterpret_cast<const uint16_t*>(p.value.anchor_codes + at * 64 + lane * 2);
    const float ks = e4m3_to_float(p.key.anchor_scale[at * 4 + group]);
    const float ko = e4m3_to_float(p.key.anchor_offset[at * 4 + group]);
    const float vs = e4m3_to_float(p.value.anchor_scale[at * 4 + group]);
    const float vo = e4m3_to_float(p.value.anchor_offset[at * 4 + group]);
    float k[4], v[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      k[c] = fmaf(ks, static_cast<float>((kcode >> (4 * c)) & 0xF), ko) + mean3[c];
      v[c] = fmaf(vs, static_cast<float>((vcode >> (4 * c)) & 0xF), vo);
    }
    if constexpr (kResidual) {
      const uint32_t krc = p.key.residual_codes[rt * 32 + lane];
      const uint32_t vrc = p.value.residual_codes[rt * 32 + lane];
      const float ka = bf16_bits_to_float(p.key.alpha[rt]);
      const float va = bf16_bits_to_float(p.value.alpha[rt]);
      const float krs = e4m3_to_float(p.key.residual_scale[rt * 4 + group]);
      const float kro = e4m3_to_float(p.key.residual_offset[rt * 4 + group]);
      const float vrs = e4m3_to_float(p.value.residual_scale[rt * 4 + group]);
      const float vro = e4m3_to_float(p.value.residual_offset[rt * 4 + group]);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const float kr = fmaf(krs, static_cast<float>((krc >> (2 * c)) & 3), kro);
        const float vr = fmaf(vrs, static_cast<float>((vrc >> (2 * c)) & 3), vro);
        k[c] = fmaf(ka, k[c], kr) + meanr[c];
        v[c] = fmaf(va, v[c], vr);
      }
    }
#pragma unroll
    for (int i = 0; i < Q; ++i) {
      float dot = 0.f;
#pragma unroll
      for (int c = 0; c < 4; ++c) dot = fmaf(q[i][c], k[c], dot);
      const float s = warp_sum(dot) * kScoreScale;
      const float top = fmaxf(m[i], s);
      const float carry = fast_exp2(m[i] - top), w = fast_exp2(s - top);
      l[i] = fmaf(l[i], carry, w);
#pragma unroll
      for (int c = 0; c < 4; ++c) acc[i][c] = fmaf(acc[i][c], carry, w * v[c]);
      m[i] = top;
    }
  }

#pragma unroll
  for (int i = 0; i < Q; ++i) {
#pragma unroll
    for (int c = 0; c < 4; ++c) state.o[warp][i][ch0 + c] = acc[i][c];
    if (lane == 0) {
      state.m[warp][i] = m[i];
      state.l[warp][i] = l[i];
    }
  }
  __syncthreads();
  finish_block<T>(p, row, part, state, scores);
}

template <typename T, bool kResidual>
void launch_typed(const Params& p, cudaStream_t stream) {
  const dim3 grid(p.splits, p.rows);
  switch (p.q) {
#define RKV_CASE(QV) \
  case QV: attention_kernel<T, kResidual, QV><<<grid, kThreads, 0, stream>>>(p); break;
    RKV_CASE(1) RKV_CASE(2) RKV_CASE(3) RKV_CASE(4) RKV_CASE(5) RKV_CASE(6) RKV_CASE(7) RKV_CASE(8)
#undef RKV_CASE
  }
}

}  // namespace

void launch(const Params& p, InputType type, cudaStream_t stream) {
  const bool residual = p.loop != kAnchorLoop;
  if (type == InputType::kHalf) {
    residual ? launch_typed<__half, true>(p, stream) : launch_typed<__half, false>(p, stream);
  } else {
    residual ? launch_typed<__nv_bfloat16, true>(p, stream)
             : launch_typed<__nv_bfloat16, false>(p, stream);
  }
}

}  // namespace portable
}  // namespace rkv
