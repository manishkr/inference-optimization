#include "backends.h"
#include "common.cuh"

namespace rkv {
namespace {

// One block per (query, row); thread `ch` owns one output channel.
template <typename T>
__global__ void __launch_bounds__(kThreads) merge_kernel(Params p) {
  const int i = blockIdx.x, row = blockIdx.y, ch = threadIdx.x;
  __shared__ float scores[kMaxQueries][kMaxQueries];
  new_token_scores<T>(p, row, i, i + 1, scores);

  const size_t base = static_cast<size_t>(row) * p.splits;
  float m = -INFINITY;
  for (int part = 0; part < p.splits; ++part) m = fmaxf(m, p.partial_ml[(base + part) * p.q + i].x);
  float l = 0.f, o = 0.f;
  for (int part = 0; part < p.splits; ++part) {
    const size_t slot = (base + part) * p.q + i;
    const float2 ml = p.partial_ml[slot];
    const float e = ml.x == -INFINITY ? 0.f : fast_exp2(ml.x - m);
    l = fmaf(ml.y, e, l);
    o = fmaf(p.partial_o[slot * kDim + ch], e, o);
  }
  __syncthreads();
  finish_query<T>(p, row, i, ch, m, l, o, scores[i]);
}

}  // namespace

void launch_merge(const Params& p, InputType type, cudaStream_t stream) {
  const dim3 grid(p.q, p.rows);
  if (type == InputType::kHalf) {
    merge_kernel<__half><<<grid, kThreads, 0, stream>>>(p);
  } else {
    merge_kernel<__nv_bfloat16><<<grid, kThreads, 0, stream>>>(p);
  }
}

}  // namespace rkv
