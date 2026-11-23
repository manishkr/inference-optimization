#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <mutex>

#include "backends.h"
#include "common.cuh"

namespace fast {

constexpr int min_blocks_for_arch(int blocks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return blocks < 4 ? blocks : 4;
#else
  return blocks;
#endif
}

struct BlockSmem256 {
  uint4 raw[20][16][16];
  Float8Row16 S[19];
  Float8Row16 halo_odd[3];
  Float8Row16 bunit;
  float inv_norm[20][16];
};

struct BlockSmem512 {
  uint4 raw[16][16][16];
  Float8Row16 S[19];
  Float8Row16 E_mid[16];
  Float8Row16 halo_q[16];
  Float8Row16 bunit;
  float inv_norm[16][16];
};

__device__ __forceinline__ float8 warp_pair_sum_float8(float8 v) {
  float8 other;
  other.lo.x = __shfl_xor_sync(0xffffffffu, v.lo.x, 16);
  other.lo.y = __shfl_xor_sync(0xffffffffu, v.lo.y, 16);
  other.lo.z = __shfl_xor_sync(0xffffffffu, v.lo.z, 16);
  other.lo.w = __shfl_xor_sync(0xffffffffu, v.lo.w, 16);
  other.hi.x = __shfl_xor_sync(0xffffffffu, v.hi.x, 16);
  other.hi.y = __shfl_xor_sync(0xffffffffu, v.hi.y, 16);
  other.hi.z = __shfl_xor_sync(0xffffffffu, v.hi.z, 16);
  other.hi.w = __shfl_xor_sync(0xffffffffu, v.hi.w, 16);
  return add_float8(v, other);
}

__device__ __forceinline__ float butterfly_reduce4(
    float v0, float v1, float v2, float v3, int sub_lane, uint32_t hw_mask) {
  bool odd1 = (sub_lane & 1) != 0;
  float s01 = odd1 ? v0 : v1;
  float m01 = (odd1 ? v1 : v0) + __shfl_xor_sync(hw_mask, s01, 1);

  float s23 = odd1 ? v2 : v3;
  float m23 = (odd1 ? v3 : v2) + __shfl_xor_sync(hw_mask, s23, 1);

  bool odd2 = (sub_lane & 2) != 0;
  float s0123 = odd2 ? m01 : m23;
  float m = (odd2 ? m23 : m01) + __shfl_xor_sync(hw_mask, s0123, 2);

  m += __shfl_xor_sync(hw_mask, m, 4);
  m += __shfl_xor_sync(hw_mask, m, 8);
  return m;
}

__device__ __forceinline__ float butterfly_reduce4_warp(
    float v0, float v1, float v2, float v3, int sub_lane) {
  bool odd1 = (sub_lane & 1) != 0;
  float s01 = odd1 ? v0 : v1;
  float m01 = (odd1 ? v1 : v0) + __shfl_xor_sync(0xffffffffu, s01, 1);

  float s23 = odd1 ? v2 : v3;
  float m23 = (odd1 ? v3 : v2) + __shfl_xor_sync(0xffffffffu, s23, 1);

  bool odd2 = (sub_lane & 2) != 0;
  float s0123 = odd2 ? m01 : m23;
  float m = (odd2 ? m23 : m01) + __shfl_xor_sync(0xffffffffu, s0123, 2);

  m += __shfl_xor_sync(0xffffffffu, m, 4);
  m += __shfl_xor_sync(0xffffffffu, m, 8);
  return m;
}

__device__ __forceinline__ void compute_inv4(
    float sq0, float sq1, float sq2, float sq3,
    int sub_lane, int base_lane, uint32_t hw_mask,
    float& inv0, float& inv1, float& inv2, float& inv3, float& my_inv) {
  float my_sq = butterfly_reduce4_warp(sq0, sq1, sq2, sq3, sub_lane);
  my_inv = fast_inv_sqrt(my_sq, EPS_SQ);
  inv0 = __shfl_sync(0xffffffffu, my_inv, base_lane + 0);
  inv1 = __shfl_sync(0xffffffffu, my_inv, base_lane + 1);
  inv2 = __shfl_sync(0xffffffffu, my_inv, base_lane + 2);
  inv3 = __shfl_sync(0xffffffffu, my_inv, base_lane + 3);
}

template <bool IS_BF16>
__global__ __launch_bounds__(256, 2) void kv_block_local_kernel(
    const uint4* __restrict__ keys,
    float* __restrict__ blocks,
    float* __restrict__ out,
    int n,
    int nb,
    int total) {
  extern __shared__ char smem_buf[];
  BlockSmem256* smem = reinterpret_cast<BlockSmem256*>(smem_buf);

  const int b_idx = blockIdx.x;
  const int g = blockIdx.y;
  const int block_start = b_idx * BLOCK;
  const int count = min(BLOCK, n - block_start);
  const int w = threadIdx.x >> 5;
  const int hw = threadIdx.x >> 4;
  const int sub_lane = threadIdx.x & 15;
  const int base_lane = threadIdx.x & 16;
  const uint32_t hw_mask = base_lane ? 0xffff0000u : 0x0000ffffu;

  const uint4* keys_group = keys + static_cast<size_t>(g) * n * 16;
  const bool full_warp = ((w << 5) + 31 < count);

  const int hc = hw >> 2;
  const int hs = hw & 3;
  const int hk_base = hs << 2;
  const int idx_main = hw + 4;

  uint4 hr0, hr1, hr2, hr3;
  if (b_idx > 0) {
    int h_base_row = block_start - 64 + (hw << 2);
    hr0 = __ldg(&keys_group[(h_base_row + 0) * 16 + sub_lane]);
    hr1 = __ldg(&keys_group[(h_base_row + 1) * 16 + sub_lane]);
    hr2 = __ldg(&keys_group[(h_base_row + 2) * 16 + sub_lane]);
    hr3 = __ldg(&keys_group[(h_base_row + 3) * 16 + sub_lane]);
  }

  uint4 r0, r1, r2, r3;
  if (full_warp) {
    int base_row0 = block_start + (hw << 4);
    r0 = __ldg(&keys_group[(base_row0 + 0) * 16 + sub_lane]);
    r1 = __ldg(&keys_group[(base_row0 + 1) * 16 + sub_lane]);
    r2 = __ldg(&keys_group[(base_row0 + 2) * 16 + sub_lane]);
    r3 = __ldg(&keys_group[(base_row0 + 3) * 16 + sub_lane]);
  }

  if (b_idx == 0) {
    uint4 zero_u4 = make_uint4(0, 0, 0, 0);
    smem->raw[hc][hk_base + 0][sub_lane] = zero_u4;
    smem->raw[hc][hk_base + 1][sub_lane] = zero_u4;
    smem->raw[hc][hk_base + 2][sub_lane] = zero_u4;
    smem->raw[hc][hk_base + 3][sub_lane] = zero_u4;
    if (sub_lane < 4) {
      smem->inv_norm[hc][hk_base + sub_lane] = 0.0f;
    }
    if (hc >= 1) {
      if (hs == 0) store_f8(smem->S[hc - 1], sub_lane, zero_float8());
      if (hs == 2) store_f8(smem->halo_odd[hc - 1], sub_lane, zero_float8());
    }
  } else {
    smem->raw[hc][hk_base + 0][sub_lane] = hr0;
    smem->raw[hc][hk_base + 1][sub_lane] = hr1;
    smem->raw[hc][hk_base + 2][sub_lane] = hr2;
    smem->raw[hc][hk_base + 3][sub_lane] = hr3;

    float8 hv0 = raw_to_float8<IS_BF16>(hr0);
    float8 hv1 = raw_to_float8<IS_BF16>(hr1);
    float8 hv2 = raw_to_float8<IS_BF16>(hr2);
    float8 hv3 = raw_to_float8<IS_BF16>(hr3);

    float hinv0, hinv1, hinv2, hinv3, my_hinv;
    compute_inv4(
        dot_float8(hv0, hv0), dot_float8(hv1, hv1), dot_float8(hv2, hv2), dot_float8(hv3, hv3),
        sub_lane, base_lane, hw_mask, hinv0, hinv1, hinv2, hinv3, my_hinv);

    if (sub_lane < 4) {
      smem->inv_norm[hc][hk_base + sub_lane] = my_hinv;
    }

    if (hc >= 1) {
      float8 h_sum = scale_float8(hv0, hinv0);
      h_sum = fma_float8(hv1, hinv1, h_sum);
      h_sum = fma_float8(hv2, hinv2, h_sum);
      h_sum = fma_float8(hv3, hinv3, h_sum);
      h_sum = warp_pair_sum_float8(h_sum);
      if (hs == 0) store_f8(smem->S[hc - 1], sub_lane, h_sum);
      if (hs == 2) store_f8(smem->halo_odd[hc - 1], sub_lane, h_sum);
    }
  }

  float8 running_P = zero_float8();
  float8 P_ckpt0 = zero_float8();
  float8 P_ckpt1 = zero_float8();
  float8 P_ckpt2 = zero_float8();

  if (full_warp) {
    const uint4* p_main = keys_group + ((block_start + (hw << 4)) * 16 + sub_lane);
    #pragma unroll
    for (int s = 0; s < 4; ++s) {
      int k_base = s << 2;
      uint4 nr0, nr1, nr2, nr3;
      if (s < 3) {
        const uint4* p_next = p_main + ((s + 1) * 64);
        nr0 = __ldg(p_next + 0);
        nr1 = __ldg(p_next + 16);
        nr2 = __ldg(p_next + 32);
        nr3 = __ldg(p_next + 48);
      }

      smem->raw[idx_main][k_base + 0][sub_lane] = r0;
      smem->raw[idx_main][k_base + 1][sub_lane] = r1;
      smem->raw[idx_main][k_base + 2][sub_lane] = r2;
      smem->raw[idx_main][k_base + 3][sub_lane] = r3;

      float8 v0 = raw_to_float8<IS_BF16>(r0);
      float8 v1 = raw_to_float8<IS_BF16>(r1);
      float8 v2 = raw_to_float8<IS_BF16>(r2);
      float8 v3 = raw_to_float8<IS_BF16>(r3);

      float inv0, inv1, inv2, inv3, my_inv;
      compute_inv4(
          dot_float8(v0, v0), dot_float8(v1, v1), dot_float8(v2, v2), dot_float8(v3, v3),
          sub_lane, base_lane, hw_mask, inv0, inv1, inv2, inv3, my_inv);

      if (sub_lane < 4) {
        smem->inv_norm[idx_main][k_base + sub_lane] = my_inv;
      }

      running_P = fma_float8(v0, inv0, running_P);
      running_P = fma_float8(v1, inv1, running_P);
      running_P = fma_float8(v2, inv2, running_P);
      running_P = fma_float8(v3, inv3, running_P);

      if (s == 0) P_ckpt0 = running_P;
      if (s == 1) P_ckpt1 = running_P;
      if (s == 2) P_ckpt2 = running_P;

      if (s < 3) {
        r0 = nr0; r1 = nr1; r2 = nr2; r3 = nr3;
      }
    }
  } else if ((w << 5) < count) {
    for (int s = 0; s < 4; ++s) {
      int k_base = s << 2;
      #pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        int k = k_base + sub;
        int local_t = (hw << 4) + k;
        uint4 r_val = make_uint4(0, 0, 0, 0);
        float inv_n = 0.0f;
        if (local_t < count) {
          r_val = __ldg(&keys_group[(block_start + local_t) * 16 + sub_lane]);
          float8 v = raw_to_float8<IS_BF16>(r_val);
          float sq = halfwarp_allreduce_sum(dot_float8(v, v), hw_mask);
          inv_n = fast_inv_sqrt(sq, EPS_SQ);
          running_P = add_float8(running_P, scale_float8(v, inv_n));
        }
        smem->raw[idx_main][k][sub_lane] = r_val;
        if (sub_lane == 0) {
          smem->inv_norm[idx_main][k] = inv_n;
        }
      }
      if (s == 0) P_ckpt0 = running_P;
      if (s == 1) P_ckpt1 = running_P;
      if (s == 2) P_ckpt2 = running_P;
    }
  }
  store_f8(smem->S[hw + 3], sub_lane, running_P);

  __syncthreads();

  const int lane = threadIdx.x & 31;
  if (w == 0) {
    float4 s0 = add_float4(
        add_float4(load_f4_lane(smem->S[3], lane), load_f4_lane(smem->S[4], lane)),
        add_float4(load_f4_lane(smem->S[5], lane), load_f4_lane(smem->S[6], lane)));
    float4 s1 = add_float4(
        add_float4(load_f4_lane(smem->S[7], lane), load_f4_lane(smem->S[8], lane)),
        add_float4(load_f4_lane(smem->S[9], lane), load_f4_lane(smem->S[10], lane)));
    float4 s2 = add_float4(
        add_float4(load_f4_lane(smem->S[11], lane), load_f4_lane(smem->S[12], lane)),
        add_float4(load_f4_lane(smem->S[13], lane), load_f4_lane(smem->S[14], lane)));
    float4 s3 = add_float4(
        add_float4(load_f4_lane(smem->S[15], lane), load_f4_lane(smem->S[16], lane)),
        add_float4(load_f4_lane(smem->S[17], lane), load_f4_lane(smem->S[18], lane)));
    float4 bsum4 = add_float4(add_float4(s0, s1), add_float4(s2, s3));
    if (nb > 1) {
      store_f4_lane(reinterpret_cast<Float8Row16*>(blocks)[g * nb + b_idx], lane, bsum4);
    }
    float b_sq = warp_allreduce_sum(dot_float4(bsum4, bsum4));
    float b_thresh = static_cast<float>(count) * EPS;
    float inv_bn = fast_inv_sqrt(b_sq, b_thresh * b_thresh);
    store_f4_lane(smem->bunit, lane, scale_float4(bsum4, inv_bn));
  } else if (w == 1 && b_idx > 0) {
    #pragma unroll
    for (int h_idx = 0; h_idx < 3; ++h_idx) {
      store_f4_lane(
          smem->S[h_idx],
          lane,
          add_float4(load_f4_lane(smem->S[h_idx], lane), load_f4_lane(smem->halo_odd[h_idx], lane)));
    }
  }

  __syncthreads();

  if ((w << 5) < count) {
    float8 running_E = add_float8(
        add_float8(load_f8(smem->S[hw], sub_lane), load_f8(smem->S[hw + 1], sub_lane)),
        load_f8(smem->S[hw + 2], sub_lane));
    float8 b_unit = load_f8(smem->bunit, sub_lane);

    float my_norm_l_sq = 0.0f;
    float my_dot_ul = 0.0f;
    float my_dot_ub = 0.0f;
    const int my_strip = sub_lane >> 2;

    #pragma unroll
    for (int s = 3; s >= 0; --s) {
      int k_base = s << 2;

      float8 E3 = running_E;
      running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[hw][k_base + 3][sub_lane]), smem->inv_norm[hw][k_base + 3], running_E);
      float8 E2 = running_E;
      running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[hw][k_base + 2][sub_lane]), smem->inv_norm[hw][k_base + 2], running_E);
      float8 E1 = running_E;
      running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[hw][k_base + 1][sub_lane]), smem->inv_norm[hw][k_base + 1], running_E);
      float8 E0 = running_E;

      float8 E_arr[4] = {E0, E1, E2, E3};
      float8 P_cur = (s == 3) ? P_ckpt2 : ((s == 2) ? P_ckpt1 : ((s == 1) ? P_ckpt0 : zero_float8()));

      float nsq[4], dul[4], dub[4];
      #pragma unroll
      for (int sub = 0; sub < 4; ++sub) {
        int k = k_base + sub;
        float8 v = raw_to_float8<IS_BF16>(smem->raw[idx_main][k][sub_lane]);
        float inv = smem->inv_norm[idx_main][k];
        P_cur = fma_float8(v, inv, P_cur);
        float8 loc = add_float8(P_cur, E_arr[sub]);

        nsq[sub] = dot_float8(loc, loc);
        dul[sub] = dot_float8(v, loc);
        dub[sub] = dot_float8(v, b_unit);
      }

      if (s > 0) {
        running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[hw][k_base + 0][sub_lane]), smem->inv_norm[hw][k_base + 0], E0);
      }

      float r_nsq = butterfly_reduce4(nsq[0], nsq[1], nsq[2], nsq[3], sub_lane, hw_mask);
      float r_dul = butterfly_reduce4(dul[0], dul[1], dul[2], dul[3], sub_lane, hw_mask);
      float r_dub = butterfly_reduce4(dub[0], dub[1], dub[2], dub[3], sub_lane, hw_mask);

      if (my_strip == s) {
        my_norm_l_sq = r_nsq;
        my_dot_ul = r_dul;
        my_dot_ub = r_dub;
      }
    }

    int out_t = (hw << 4) + sub_lane;
    if (out_t < count) {
      float my_u_inv = smem->inv_norm[idx_main][sub_lane];
      int global_i = block_start + out_t;
      int win_count = min(WINDOW, global_i + 1);
      float l_thresh = static_cast<float>(win_count) * EPS;
      float inv_denom_l = fast_inv_sqrt(my_norm_l_sq, l_thresh * l_thresh);
      float bs = (my_dot_ub == 0.0f) ? 0.0f : (-my_dot_ub * my_u_inv);
      float ls = (my_dot_ul == 0.0f) ? 0.0f : (-my_dot_ul * (my_u_inv * inv_denom_l));
      int row = g * n + global_i;
      if (nb == 1) {
        out[row] = bs;
      }
      out[total + row] = bs;
      out[2 * total + row] = ls;
    }
  }
}

template <bool IS_BF16>
__global__ __launch_bounds__(512, 1) void kv_block_local_kernel_512(
    const uint4* __restrict__ keys,
    float* __restrict__ blocks,
    float* __restrict__ out,
    int n,
    int nb,
    int total) {
  extern __shared__ char smem_buf[];
  BlockSmem512* smem = reinterpret_cast<BlockSmem512*>(smem_buf);

  const int b_idx = blockIdx.x;
  const int g = blockIdx.y;
  const int block_start = b_idx * BLOCK;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int hw32 = tid >> 4;
  const int half = hw32 & 1;
  const int sub_lane = tid & 15;
  const int base_lane = tid & 16;
  const uint32_t hw_mask = base_lane ? 0xffff0000u : 0x0000ffffu;
  const int idx_main = w + 4;
  const int k_half_base = half << 3;

  const uint4* keys_group = keys + static_cast<size_t>(g) * n * 16;
  const int hc = hw32 >> 3;
  const int hk_base = (hw32 & 7) << 1;

  uint4 hr0, hr1;
  if (b_idx > 0) {
    int h_base_row = block_start - 64 + (hw32 << 1);
    hr0 = __ldg(&keys_group[(h_base_row + 0) * 16 + sub_lane]);
    hr1 = __ldg(&keys_group[(h_base_row + 1) * 16 + sub_lane]);
  }

  uint4 r0, r1, r2, r3;
  if (n > 0) {
    int base_row0 = block_start + (w << 4) + k_half_base;
    r0 = __ldg(&keys_group[(base_row0 + 0) * 16 + sub_lane]);
    r1 = __ldg(&keys_group[(base_row0 + 1) * 16 + sub_lane]);
    r2 = __ldg(&keys_group[(base_row0 + 2) * 16 + sub_lane]);
    r3 = __ldg(&keys_group[(base_row0 + 3) * 16 + sub_lane]);
  }

  if (b_idx == 0) {
    uint4 zero_u4 = make_uint4(0, 0, 0, 0);
    smem->raw[hc][hk_base + 0][sub_lane] = zero_u4;
    smem->raw[hc][hk_base + 1][sub_lane] = zero_u4;
    if (sub_lane < 2) {
      smem->inv_norm[hc][hk_base + sub_lane] = 0.0f;
    }
    if (half == 0) {
      store_f8(smem->halo_q[w], sub_lane, zero_float8());
    }
  } else {
    smem->raw[hc][hk_base + 0][sub_lane] = hr0;
    smem->raw[hc][hk_base + 1][sub_lane] = hr1;

    float8 hv0 = raw_to_float8<IS_BF16>(hr0);
    float8 hv1 = raw_to_float8<IS_BF16>(hr1);

    float hinv0, hinv1, dummy2, dummy3, my_hinv;
    compute_inv4(
        dot_float8(hv0, hv0), dot_float8(hv1, hv1), 1.0f, 1.0f,
        sub_lane, base_lane, hw_mask, hinv0, hinv1, dummy2, dummy3, my_hinv);

    if (sub_lane < 2) {
      smem->inv_norm[hc][hk_base + sub_lane] = my_hinv;
    }

    float8 h_sum = fma_float8(hv1, hinv1, scale_float8(hv0, hinv0));
    h_sum = warp_pair_sum_float8(h_sum);
    if (half == 0) {
      store_f8(smem->halo_q[w], sub_lane, h_sum);
    }
  }

  const uint4* p_next = keys_group + ((block_start + (w << 4) + k_half_base + 4) * 16 + sub_lane);
  uint4 nr0 = __ldg(p_next + 0);
  uint4 nr1 = __ldg(p_next + 16);
  uint4 nr2 = __ldg(p_next + 32);
  uint4 nr3 = __ldg(p_next + 48);

  float inv0_s0, inv1_s0, inv2_s0, inv3_s0, my_inv_s0;
  float8 P_s0;
  {
    int k_base = k_half_base;
    if (w < 12) {
      smem->raw[idx_main][k_base + 0][sub_lane] = r0;
      smem->raw[idx_main][k_base + 1][sub_lane] = r1;
      smem->raw[idx_main][k_base + 2][sub_lane] = r2;
      smem->raw[idx_main][k_base + 3][sub_lane] = r3;
    }

    float8 v0 = raw_to_float8<IS_BF16>(r0);
    float8 v1 = raw_to_float8<IS_BF16>(r1);
    float8 v2 = raw_to_float8<IS_BF16>(r2);
    float8 v3 = raw_to_float8<IS_BF16>(r3);

    compute_inv4(
        dot_float8(v0, v0), dot_float8(v1, v1), dot_float8(v2, v2), dot_float8(v3, v3),
        sub_lane, base_lane, hw_mask, inv0_s0, inv1_s0, inv2_s0, inv3_s0, my_inv_s0);

    if (w < 12 && sub_lane < 4) {
      smem->inv_norm[idx_main][k_base + sub_lane] = my_inv_s0;
    }

    P_s0 = scale_float8(v0, inv0_s0);
    P_s0 = fma_float8(v1, inv1_s0, P_s0);
    P_s0 = fma_float8(v2, inv2_s0, P_s0);
    P_s0 = fma_float8(v3, inv3_s0, P_s0);
  }

  float inv0_s1, inv1_s1, inv2_s1, inv3_s1, my_inv_s1;
  float8 P_half;
  {
    int k_base = k_half_base + 4;
    if (w < 12) {
      smem->raw[idx_main][k_base + 0][sub_lane] = nr0;
      smem->raw[idx_main][k_base + 1][sub_lane] = nr1;
      smem->raw[idx_main][k_base + 2][sub_lane] = nr2;
      smem->raw[idx_main][k_base + 3][sub_lane] = nr3;
    }

    float8 v0 = raw_to_float8<IS_BF16>(nr0);
    float8 v1 = raw_to_float8<IS_BF16>(nr1);
    float8 v2 = raw_to_float8<IS_BF16>(nr2);
    float8 v3 = raw_to_float8<IS_BF16>(nr3);

    compute_inv4(
        dot_float8(v0, v0), dot_float8(v1, v1), dot_float8(v2, v2), dot_float8(v3, v3),
        sub_lane, base_lane, hw_mask, inv0_s1, inv1_s1, inv2_s1, inv3_s1, my_inv_s1);

    if (w < 12 && sub_lane < 4) {
      smem->inv_norm[idx_main][k_base + sub_lane] = my_inv_s1;
    }

    P_half = fma_float8(v0, inv0_s1, P_s0);
    P_half = fma_float8(v1, inv1_s1, P_half);
    P_half = fma_float8(v2, inv2_s1, P_half);
    P_half = fma_float8(v3, inv3_s1, P_half);
  }

  if (half == 1 && w < 12) {
    store_f8(smem->E_mid[w + 4], sub_lane, P_half);
  }

  float8 P_other;
  P_other.lo.x = __shfl_xor_sync(0xffffffffu, P_half.lo.x, 16);
  P_other.lo.y = __shfl_xor_sync(0xffffffffu, P_half.lo.y, 16);
  P_other.lo.z = __shfl_xor_sync(0xffffffffu, P_half.lo.z, 16);
  P_other.lo.w = __shfl_xor_sync(0xffffffffu, P_half.lo.w, 16);
  P_other.hi.x = __shfl_xor_sync(0xffffffffu, P_half.hi.x, 16);
  P_other.hi.y = __shfl_xor_sync(0xffffffffu, P_half.hi.y, 16);
  P_other.hi.z = __shfl_xor_sync(0xffffffffu, P_half.hi.z, 16);
  P_other.hi.w = __shfl_xor_sync(0xffffffffu, P_half.hi.w, 16);

  float8 S_chunk = add_float8(P_half, P_other);
  if (half == 0) {
    store_f8(smem->S[w + 3], sub_lane, S_chunk);
  }

  float8 P_base0 = (half == 1) ? P_other : zero_float8();
  float8 P_base1 = (half == 1) ? add_float8(P_other, P_s0) : P_s0;

  __syncthreads();

  const int lane = tid & 31;
  if (w == 0) {
    float4 s0 = add_float4(
        add_float4(load_f4_lane(smem->S[3], lane), load_f4_lane(smem->S[4], lane)),
        add_float4(load_f4_lane(smem->S[5], lane), load_f4_lane(smem->S[6], lane)));
    float4 s1 = add_float4(
        add_float4(load_f4_lane(smem->S[7], lane), load_f4_lane(smem->S[8], lane)),
        add_float4(load_f4_lane(smem->S[9], lane), load_f4_lane(smem->S[10], lane)));
    float4 s2 = add_float4(
        add_float4(load_f4_lane(smem->S[11], lane), load_f4_lane(smem->S[12], lane)),
        add_float4(load_f4_lane(smem->S[13], lane), load_f4_lane(smem->S[14], lane)));
    float4 s3 = add_float4(
        add_float4(load_f4_lane(smem->S[15], lane), load_f4_lane(smem->S[16], lane)),
        add_float4(load_f4_lane(smem->S[17], lane), load_f4_lane(smem->S[18], lane)));
    float4 bsum4 = add_float4(add_float4(s0, s1), add_float4(s2, s3));
    store_f4_lane(reinterpret_cast<Float8Row16*>(blocks)[g * nb + b_idx], lane, bsum4);
    float b_sq = warp_allreduce_sum(dot_float4(bsum4, bsum4));
    constexpr float b_thresh = static_cast<float>(BLOCK) * EPS;
    float inv_bn = fast_inv_sqrt(b_sq, b_thresh * b_thresh);
    store_f4_lane(smem->bunit, lane, scale_float4(bsum4, inv_bn));
  } else if (w >= 1 && w <= 4) {
    int hc_idx = w - 1;
    float4 q23 = add_float4(
        load_f4_lane(smem->halo_q[hc_idx * 4 + 2], lane),
        load_f4_lane(smem->halo_q[hc_idx * 4 + 3], lane));
    store_f4_lane(smem->E_mid[hc_idx], lane, q23);
    if (hc_idx >= 1) {
      float4 q01 = add_float4(
          load_f4_lane(smem->halo_q[hc_idx * 4 + 0], lane),
          load_f4_lane(smem->halo_q[hc_idx * 4 + 1], lane));
      store_f4_lane(smem->S[hc_idx - 1], lane, add_float4(q01, q23));
    }
  }

  __syncthreads();

  float8 running_E = add_float8(
      add_float8(load_f8(smem->S[w], sub_lane), load_f8(smem->S[w + 1], sub_lane)),
      load_f8(smem->S[w + 2], sub_lane));
  if (half == 0) {
    running_E = add_float8(running_E, load_f8(smem->E_mid[w], sub_lane));
  }
  float8 b_unit = load_f8(smem->bunit, sub_lane);

  float my_norm_l_sq = 0.0f;
  float my_dot_ul = 0.0f;
  float my_dot_ub = 0.0f;
  const int my_strip = sub_lane >> 2;

  {
    int k_base = k_half_base + 4;
    float8 E3 = running_E;
    running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[w][k_base + 3][sub_lane]), smem->inv_norm[w][k_base + 3], running_E);
    float8 E2 = running_E;
    running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[w][k_base + 2][sub_lane]), smem->inv_norm[w][k_base + 2], running_E);
    float8 E1 = running_E;
    running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[w][k_base + 1][sub_lane]), smem->inv_norm[w][k_base + 1], running_E);
    float8 E0 = running_E;

    float8 P_cur = P_base1;
    float nsq0, dul0, dub0, nsq1, dul1, dub1, nsq2, dul2, dub2, nsq3, dul3, dub3;
    {
      float8 v = raw_to_float8<IS_BF16>(nr0);
      P_cur = fma_float8(v, inv0_s1, P_cur);
      float8 loc = add_float8(P_cur, E0);
      nsq0 = dot_float8(loc, loc); dul0 = dot_float8(v, loc); dub0 = dot_float8(v, b_unit);
    }
    {
      float8 v = raw_to_float8<IS_BF16>(nr1);
      P_cur = fma_float8(v, inv1_s1, P_cur);
      float8 loc = add_float8(P_cur, E1);
      nsq1 = dot_float8(loc, loc); dul1 = dot_float8(v, loc); dub1 = dot_float8(v, b_unit);
    }
    {
      float8 v = raw_to_float8<IS_BF16>(nr2);
      P_cur = fma_float8(v, inv2_s1, P_cur);
      float8 loc = add_float8(P_cur, E2);
      nsq2 = dot_float8(loc, loc); dul2 = dot_float8(v, loc); dub2 = dot_float8(v, b_unit);
    }
    {
      float8 v = raw_to_float8<IS_BF16>(nr3);
      P_cur = fma_float8(v, inv3_s1, P_cur);
      float8 loc = add_float8(P_cur, E3);
      nsq3 = dot_float8(loc, loc); dul3 = dot_float8(v, loc); dub3 = dot_float8(v, b_unit);
    }

    running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[w][k_base + 0][sub_lane]), smem->inv_norm[w][k_base + 0], E0);

    float r_nsq = butterfly_reduce4(nsq0, nsq1, nsq2, nsq3, sub_lane, hw_mask);
    float r_dul = butterfly_reduce4(dul0, dul1, dul2, dul3, sub_lane, hw_mask);
    float r_dub = butterfly_reduce4(dub0, dub1, dub2, dub3, sub_lane, hw_mask);
    if (my_strip == 1) {
      my_norm_l_sq = r_nsq;
      my_dot_ul = r_dul;
      my_dot_ub = r_dub;
    }
  }

  {
    int k_base = k_half_base;
    float8 E3 = running_E;
    running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[w][k_base + 3][sub_lane]), smem->inv_norm[w][k_base + 3], running_E);
    float8 E2 = running_E;
    running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[w][k_base + 2][sub_lane]), smem->inv_norm[w][k_base + 2], running_E);
    float8 E1 = running_E;
    running_E = fma_float8(raw_to_float8<IS_BF16>(smem->raw[w][k_base + 1][sub_lane]), smem->inv_norm[w][k_base + 1], running_E);
    float8 E0 = running_E;

    float8 P_cur = P_base0;
    float nsq0, dul0, dub0, nsq1, dul1, dub1, nsq2, dul2, dub2, nsq3, dul3, dub3;
    {
      float8 v = raw_to_float8<IS_BF16>(r0);
      P_cur = fma_float8(v, inv0_s0, P_cur);
      float8 loc = add_float8(P_cur, E0);
      nsq0 = dot_float8(loc, loc); dul0 = dot_float8(v, loc); dub0 = dot_float8(v, b_unit);
    }
    {
      float8 v = raw_to_float8<IS_BF16>(r1);
      P_cur = fma_float8(v, inv1_s0, P_cur);
      float8 loc = add_float8(P_cur, E1);
      nsq1 = dot_float8(loc, loc); dul1 = dot_float8(v, loc); dub1 = dot_float8(v, b_unit);
    }
    {
      float8 v = raw_to_float8<IS_BF16>(r2);
      P_cur = fma_float8(v, inv2_s0, P_cur);
      float8 loc = add_float8(P_cur, E2);
      nsq2 = dot_float8(loc, loc); dul2 = dot_float8(v, loc); dub2 = dot_float8(v, b_unit);
    }
    {
      float8 v = raw_to_float8<IS_BF16>(r3);
      P_cur = fma_float8(v, inv3_s0, P_cur);
      float8 loc = add_float8(P_cur, E3);
      nsq3 = dot_float8(loc, loc); dul3 = dot_float8(v, loc); dub3 = dot_float8(v, b_unit);
    }

    float r_nsq = butterfly_reduce4(nsq0, nsq1, nsq2, nsq3, sub_lane, hw_mask);
    float r_dul = butterfly_reduce4(dul0, dul1, dul2, dul3, sub_lane, hw_mask);
    float r_dub = butterfly_reduce4(dub0, dub1, dub2, dub3, sub_lane, hw_mask);
    if (my_strip == 0) {
      my_norm_l_sq = r_nsq;
      my_dot_ul = r_dul;
      my_dot_ub = r_dub;
    }
  }

  float my_u_inv = __shfl_sync(0xffffffffu, my_inv_s1, base_lane + (sub_lane & 3));
  if (my_strip == 0) {
    my_u_inv = my_inv_s0;
  }
  if (sub_lane < 8) {
    int out_t = (w << 4) + k_half_base + sub_lane;
    int global_i = block_start + out_t;
    int win_count = min(WINDOW, global_i + 1);
    float l_thresh = static_cast<float>(win_count) * EPS;
    float inv_denom_l = fast_inv_sqrt(my_norm_l_sq, l_thresh * l_thresh);
    float bs = (my_dot_ub == 0.0f) ? 0.0f : (-my_dot_ub * my_u_inv);
    float ls = (my_dot_ul == 0.0f) ? 0.0f : (-my_dot_ul * (my_u_inv * inv_denom_l));
    int row = g * n + global_i;
    out[total + row] = bs;
    out[2 * total + row] = ls;
  }
}

template <bool IS_BF16, int FIXED_NB, int Q_SHIFT, int STRIPS = 4, int MIN_BLOCKS = 4>
__global__ __launch_bounds__(256, min_blocks_for_arch(MIN_BLOCKS)) void kv_global_scores_kernel(
    const uint4* __restrict__ keys,
    const float* __restrict__ blocks,
    float* __restrict__ out,
    int n,
    int nb) {
  __shared__ Float8Row16 smem_g[8];

  constexpr int TILES = FIXED_NB * (4 / STRIPS);
  int tile_idx;
  int g;
  if constexpr (STRIPS < 4) {
    int cta_id = blockIdx.y * TILES + blockIdx.x;
    int slot = (cta_id >> 7) & 3;
    int sm = cta_id & 127;
    int g_rev = sm >> Q_SHIFT;
    int t_rev = (slot << Q_SHIFT) | (sm & ((1 << Q_SHIFT) - 1));
    tile_idx = (TILES - 1) - t_rev;
    g = gridDim.y - 1 - g_rev;
  } else if constexpr (Q_SHIFT == 6) {
    int x_hi = blockIdx.x >> 6;
    int x_lo = blockIdx.x & 63;
    int g_rev = (blockIdx.y & ~1) | x_hi;
    int t_rev = ((blockIdx.y & 1) << 6) | x_lo;
    tile_idx = (FIXED_NB - 1) - t_rev;
    g = gridDim.y - 1 - g_rev;
  } else if constexpr (Q_SHIFT > 0) {
    int x_hi = blockIdx.x >> Q_SHIFT;
    int x_lo = blockIdx.x & ((1 << Q_SHIFT) - 1);
    int g_rev = (blockIdx.y & ~3) | x_hi;
    int t_rev = ((blockIdx.y & 3) << Q_SHIFT) | x_lo;
    tile_idx = (FIXED_NB - 1) - t_rev;
    g = gridDim.y - 1 - g_rev;
  } else {
    tile_idx = (FIXED_NB > 0 ? FIXED_NB : nb) - 1 - blockIdx.x;
    g = gridDim.y - 1 - blockIdx.y;
  }
  const int w = threadIdx.x >> 5;
  const int hw = threadIdx.x >> 4;
  const int sub_lane = threadIdx.x & 15;
  const int my_strip = sub_lane >> 2;

  const int actual_n = (FIXED_NB > 0) ? (FIXED_NB * 256) : n;
  const int actual_nb = (FIXED_NB > 0) ? FIXED_NB : nb;
  const uint4* keys_group = keys + static_cast<size_t>(g) * actual_n * 16;
  float* out_group = out + g * actual_n;
  constexpr int ROWS_PER_CTA = STRIPS * 64;
  constexpr int ROWS_PER_HW = STRIPS * 4;
  const int base_t = tile_idx * ROWS_PER_CTA + hw * ROWS_PER_HW;
  const uint4* p_base = keys_group + (base_t * 16 + sub_lane);
  const bool full_warp = (FIXED_NB > 0) || ((tile_idx * ROWS_PER_CTA + (w << 5) + 31) < actual_n);

  uint4 r0, r1, r2, r3;
  if (full_warp) {
    r0 = ldg_stream_u4(p_base + 0);
    r1 = ldg_stream_u4(p_base + 16);
    r2 = ldg_stream_u4(p_base + 32);
    r3 = ldg_stream_u4(p_base + 48);
  }

  const Float8Row16* __restrict__ blocks_group = reinterpret_cast<const Float8Row16*>(blocks + g * actual_nb * D);
  float8 g_sum;
  if constexpr (FIXED_NB > 0) {
    constexpr int ITERS = FIXED_NB >> 4;
    g_sum = load_f8_keep(blocks_group[hw], sub_lane);
    #pragma unroll
    for (int i = 1; i < ITERS; ++i) {
      g_sum = add_float8(g_sum, load_f8_keep(blocks_group[hw + i * 16], sub_lane));
    }
  } else {
    g_sum = zero_float8();
    #pragma unroll 4
    for (int b = hw; b < actual_nb; b += 16) {
      g_sum = add_float8(g_sum, load_f8_keep(blocks_group[b], sub_lane));
    }
  }
  g_sum = warp_pair_sum_float8(g_sum);
  if ((hw & 1) == 0) {
    store_f8(smem_g[w], sub_lane, g_sum);
  }
  __syncthreads();

  if (w == 0) {
    const int lane = threadIdx.x & 31;
    float4 total_g4 = add_float4(
        add_float4(
            add_float4(load_f4_lane(smem_g[0], lane), load_f4_lane(smem_g[1], lane)),
            add_float4(load_f4_lane(smem_g[2], lane), load_f4_lane(smem_g[3], lane))),
        add_float4(
            add_float4(load_f4_lane(smem_g[4], lane), load_f4_lane(smem_g[5], lane)),
            add_float4(load_f4_lane(smem_g[6], lane), load_f4_lane(smem_g[7], lane))));
    float g_sq = warp_allreduce_sum(dot_float4(total_g4, total_g4));
    float g_thresh = static_cast<float>(actual_n) * EPS;
    float inv_gn = fast_inv_sqrt(g_sq, g_thresh * g_thresh);
    store_f4_lane(smem_g[0], lane, scale_float4(total_g4, inv_gn));
  }
  __syncthreads();

  float8 g_unit = load_f8(smem_g[0], sub_lane);

  if constexpr (STRIPS == 1) {
    float sq0, dvg0, sq1, dvg1, sq2, dvg2, sq3, dvg3;
    dot_raw_u4<IS_BF16>(r0, g_unit, sq0, dvg0);
    dot_raw_u4<IS_BF16>(r1, g_unit, sq1, dvg1);
    dot_raw_u4<IS_BF16>(r2, g_unit, sq2, dvg2);
    dot_raw_u4<IS_BF16>(r3, g_unit, sq3, dvg3);
    float my_sq = butterfly_reduce4_warp(sq0, sq1, sq2, sq3, sub_lane);
    float my_dot_vg = butterfly_reduce4_warp(dvg0, dvg1, dvg2, dvg3, sub_lane);
    if (sub_lane < 4) {
      float inv_denom = fast_inv_sqrt(my_sq, EPS_SQ);
      out_group[base_t + sub_lane] = (my_dot_vg == 0.0f) ? 0.0f : (-my_dot_vg * inv_denom);
    }
  } else if (full_warp) {
    float my_sq = 0.0f;
    float my_dot_vg = 0.0f;

    #pragma unroll
    for (int s = 0; s < 4; ++s) {
      const uint4* p_next = p_base + ((s + 1) * 64);
      float sq0, dvg0, sq1, dvg1, sq2, dvg2, sq3, dvg3;
      {
        float8 v = raw_to_float8<IS_BF16>(r0);
        sq0 = dot_float8(v, v);
        dvg0 = dot_float8(v, g_unit);
        if (s < 3) r0 = ldg_stream_u4(p_next + 0);
      }
      {
        float8 v = raw_to_float8<IS_BF16>(r1);
        sq1 = dot_float8(v, v);
        dvg1 = dot_float8(v, g_unit);
        if (s < 3) r1 = ldg_stream_u4(p_next + 16);
      }
      {
        float8 v = raw_to_float8<IS_BF16>(r2);
        sq2 = dot_float8(v, v);
        dvg2 = dot_float8(v, g_unit);
        if (s < 3) r2 = ldg_stream_u4(p_next + 32);
      }
      {
        float8 v = raw_to_float8<IS_BF16>(r3);
        sq3 = dot_float8(v, v);
        dvg3 = dot_float8(v, g_unit);
        if (s < 3) r3 = ldg_stream_u4(p_next + 48);
      }

      float r_sq = butterfly_reduce4_warp(sq0, sq1, sq2, sq3, sub_lane);
      if (my_strip == s) my_sq = r_sq;
      float r_dvg = butterfly_reduce4_warp(dvg0, dvg1, dvg2, dvg3, sub_lane);
      if (my_strip == s) my_dot_vg = r_dvg;
    }
    float inv_denom = fast_inv_sqrt(my_sq, EPS_SQ);
    out_group[base_t + sub_lane] = (my_dot_vg == 0.0f) ? 0.0f : (-my_dot_vg * inv_denom);
  } else if constexpr (FIXED_NB == 0) {
    if (base_t < actual_n) {
      const uint32_t hw_mask = (threadIdx.x & 16) ? 0xffff0000u : 0x0000ffffu;
      int rem = min(16, actual_n - base_t);
      float my_sq = 0.0f;
      float my_dot_vg = 0.0f;
      for (int k = 0; k < rem; ++k) {
        uint4 r_val = __ldg(&keys_group[(base_t + k) * 16 + sub_lane]);
        float8 v = raw_to_float8<IS_BF16>(r_val);
        float sq = halfwarp_allreduce_sum(dot_float8(v, v), hw_mask);
        if (sub_lane == k) my_sq = sq;
        float dot_vg = halfwarp_allreduce_sum(dot_float8(v, g_unit), hw_mask);
        if (sub_lane == k) my_dot_vg = dot_vg;
      }
      if (sub_lane < rem) {
        float inv_denom = fast_inv_sqrt(my_sq, EPS_SQ);
        out_group[base_t + sub_lane] = (my_dot_vg == 0.0f) ? 0.0f : (-my_dot_vg * inv_denom);
      }
    }
  }
}

template <bool IS_BF16>
void configure_kernels() {
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kv_block_local_kernel<IS_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(BlockSmem256))));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kv_block_local_kernel_512<IS_BF16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(BlockSmem512))));
  constexpr auto carveout = cudaFuncAttributePreferredSharedMemoryCarveout;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kv_global_scores_kernel<IS_BF16, 32, 5, 1, 4>, carveout, 10));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kv_global_scores_kernel<IS_BF16, 32, 3, 4, 4>, carveout, 10));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kv_global_scores_kernel<IS_BF16, 64, 4, 4, 4>, carveout, 10));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kv_global_scores_kernel<IS_BF16, 64, 4, 4, 5>, carveout, 10));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kv_global_scores_kernel<IS_BF16, 128, 6, 4, 4>, carveout, 10));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kv_global_scores_kernel<IS_BF16, 128, 6, 4, 5>, carveout, 10));
  C10_CUDA_CHECK(cudaFuncSetAttribute(kv_global_scores_kernel<IS_BF16, 0, 0, 4, 4>, carveout, 10));
}

constexpr int kMaxDevices = 64;

template <bool IS_BF16>
void configure_once(int device) {
  static std::once_flag flags[kMaxDevices];
  TORCH_CHECK(device >= 0 && device < kMaxDevices, "unsupported device index ", device);
  std::call_once(flags[device], configure_kernels<IS_BF16>);
}

template <bool IS_BF16>
void launch(const uint4* keys, float* out, float* blocks, int groups, int n, int nb,
            cudaStream_t stream) {
  const int total = groups * n;
  constexpr int smem_bytes = sizeof(BlockSmem256);
  constexpr int smem_bytes_512 = sizeof(BlockSmem512);
  const dim3 grid(nb, groups);

  if (n == 8192 && groups == 4) {
    kv_block_local_kernel_512<IS_BF16><<<grid, 512, smem_bytes_512, stream>>>(
        keys, blocks, out, n, nb, total);
  } else {
    kv_block_local_kernel<IS_BF16><<<grid, 256, smem_bytes, stream>>>(
        keys, blocks, out, n, nb, total);
  }

  if (nb == 1) return;
  if (n == 8192 && groups == 4) {
    kv_global_scores_kernel<IS_BF16, 32, 5, 1, 4><<<dim3(128, 4), 256, 0, stream>>>(
        keys, blocks, out, n, nb);
  } else if (n == 8192 && (groups & 3) == 0) {
    kv_global_scores_kernel<IS_BF16, 32, 3, 4, 4><<<grid, 256, 0, stream>>>(
        keys, blocks, out, n, nb);
  } else if (n == 16384 && groups == 16) {
    kv_global_scores_kernel<IS_BF16, 64, 4, 4, 5><<<grid, 256, 0, stream>>>(
        keys, blocks, out, n, nb);
  } else if (n == 16384 && (groups & 3) == 0) {
    kv_global_scores_kernel<IS_BF16, 64, 4, 4, 4><<<grid, 256, 0, stream>>>(
        keys, blocks, out, n, nb);
  } else if (n == 32768 && groups == 8) {
    kv_global_scores_kernel<IS_BF16, 128, 6, 4, 5><<<grid, 256, 0, stream>>>(
        keys, blocks, out, n, nb);
  } else if (n == 32768 && (groups & 3) == 0) {
    kv_global_scores_kernel<IS_BF16, 128, 6, 4, 4><<<grid, 256, 0, stream>>>(
        keys, blocks, out, n, nb);
  } else {
    kv_global_scores_kernel<IS_BF16, 0, 0, 4, 4><<<grid, 256, 0, stream>>>(
        keys, blocks, out, n, nb);
  }
}

size_t shared_bytes_required() {
  return std::max(sizeof(BlockSmem256), sizeof(BlockSmem512));
}

torch::Tensor scores(const torch::Tensor& keys) {
  const int n = static_cast<int>(keys.size(2));
  const int groups = static_cast<int>(keys.size(0) * keys.size(1));
  const int nb = (n + BLOCK - 1) / BLOCK;
  const int device = keys.get_device();
  auto options = keys.options().dtype(at::kFloat);
  auto out = torch::empty({3, keys.size(0), keys.size(1), n}, options);

  torch::Tensor blocks;
  float* blocks_ptr = nullptr;
  if (nb > 1) {
    blocks = torch::empty({groups, nb, D}, options);
    blocks_ptr = blocks.data_ptr<float>();
  }

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  const auto* data = reinterpret_cast<const uint4*>(keys.data_ptr());
  float* out_ptr = out.data_ptr<float>();
  if (keys.scalar_type() == at::kBFloat16) {
    configure_once<true>(device);
    launch<true>(data, out_ptr, blocks_ptr, groups, n, nb, stream);
  } else {
    configure_once<false>(device);
    launch<false>(data, out_ptr, blocks_ptr, groups, n, nb, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace fast
