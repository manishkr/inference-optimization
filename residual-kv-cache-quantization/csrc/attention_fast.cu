// Tensor-core backend.
//
// Each (row, partition) gets a block of four warps. Each warp stages 16-token
// tiles of the packed caches in its own shared-memory slice and runs both
// products on mma.m16n8k8 (FP16 operands, FP32 accumulation) with exact
// operands:
//   * An E4M3 scale times an INT4 or INT2 code has at most 8 significant bits
//     and lies in [2^-9, 6720], so scale * code is exact in FP16. The query-key
//     product uses (scale * code) as B and the query as A; the group offsets,
//     key means and alpha are applied afterwards in FP32 with per-group query
//     sums.
//   * Each query is multiplied by 2^k, with k chosen to bring its largest
//     component into [2^14, 2^15). FP16 queries are then exact. BF16
//     components that FP16 cannot hold move to a second part scaled by a
//     further 2^32.
//   * The value product uses (scale * code) as A and the softmax weights
//     (times alpha) as B, split into two FP16 parts after scaling by 2^13 or
//     2^15; value offsets are accumulated in FP32 per group.
// Rows 0-7 of the query-key A operand hold the queries and rows 8-15 their
// second parts. The score fragment of each thread (query = lane / 4, tokens
// 2 * (lane % 4) + {0, 1}) is then exactly the B fragment the value product
// needs.
//
// The global loads of the next tile are issued before the current tile is
// computed. Staged code rows are XOR-swizzled at 16-byte granularity so that
// fragment reads are free of bank conflicts without padding.
#include <cstring>

#include "backends.h"
#include "common.cuh"

namespace rkv {
namespace fast {
namespace {

constexpr int kTile = 16;  // tokens per warp step
constexpr int kBlockTile = kTile * kWarps;
constexpr int kMaxQueryShift = 60;
constexpr int kMinBlocksPerSm = 3;

struct alignas(16) WarpTile {
  uint8_t key_codes[kTile * 64];
  uint8_t value_codes[kTile * 64];
  uint8_t key_rcodes[kTile * 32];
  uint8_t value_rcodes[kTile * 32];
  uint4 key_scale[kTile];        // per group: half2(scale, scale)
  uint4 key_rscale[kTile];
  float4 key_offset[kTile];
  float4 key_roffset[kTile];
  float key_alpha[kTile];
  uint4 value_scale[kTile / 2];  // per token pair and group: half2(scale[2p], scale[2p+1])
  uint4 value_rscale[kTile / 2];
  float4 value_offset[kTile];    // alpha * anchor offset + residual offset
  float value_alpha[kTile];
};

union SharedTiles {
  WarpTile warp[kWarps];
  BlockState state;
  float query[kMaxQueries][kDim];
};

// Byte offsets of swizzled anchor (64-byte) and residual (32-byte) code rows.
__device__ __forceinline__ int anchor_at(int t, int byte) {
  return t * 64 + ((((byte >> 4) ^ (t >> 1)) & 3) << 4) + (byte & 15);
}

__device__ __forceinline__ int residual_at(int t, int byte) {
  return t * 32 + ((((byte >> 4) ^ (t >> 2)) & 1) << 4) + (byte & 15);
}

__device__ __forceinline__ uint32_t as_u32(__half2 h) {
  uint32_t u;
  memcpy(&u, &h, 4);
  return u;
}

__device__ __forceinline__ __half2 as_h2(uint32_t u) {
  __half2 h;
  memcpy(&h, &u, 4);
  return h;
}

__device__ __forceinline__ uint32_t pack_halves(__half lo, __half hi) {
  return as_u32(__halves2half2(lo, hi));
}

__device__ __forceinline__ uint32_t broadcast_half(float x) {
  return as_u32(__float2half2_rn(x));
}

// (x & kMask) | bias in one LOP3 with the mask as its immediate, so only the
// bias shared by every unpacking needs a register.
template <uint32_t kMask>
__device__ __forceinline__ uint32_t and_or(uint32_t x, uint32_t bias) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(x), "n"(kMask), "r"(bias));
  return d;
}

// Codes j and j + 4 of an 8-code word, biased by 1024 as FP16 (bias = 0x64006400).
__device__ __forceinline__ uint32_t nibble_pair(uint32_t word, int j, uint32_t bias) {
  return and_or<0x000F000Fu>(word >> (4 * j), bias);
}

__device__ __forceinline__ uint32_t crumb_pair(uint32_t word, int j, uint32_t bias) {
  return and_or<0x00030003u>(word >> (2 * j), bias);
}

// (code * scale) for both halves; exact in FP16.
__device__ __forceinline__ uint32_t scaled_code(uint32_t biased, uint32_t scale) {
  const __half2 code = __hsub2(as_h2(biased), as_h2(0x64006400u));
  return as_u32(__hmul2(code, as_h2(scale)));
}

// x = hi + lo with both parts FP16.
__device__ __forceinline__ void split_pair(float a, float b, uint32_t& hi, uint32_t& lo) {
  const __half2 h = __floats2half2_rn(a, b);
  const float2 f = __half22float2(h);
  hi = as_u32(h);
  lo = as_u32(__floats2half2_rn(a - f.x, b - f.y));
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(b));
#else
  __trap();
#endif
}

__device__ __forceinline__ uint32_t ldg32(const uint8_t* base, size_t index) {
  return __ldg(reinterpret_cast<const uint32_t*>(base) + index);
}

__device__ __forceinline__ uint4 ldg128(const uint8_t* address) {
  return __ldg(reinterpret_cast<const uint4*>(address));
}

// One lane's share of a tile's global data: two 16-byte chunks of anchor
// codes and one of residual codes for keys and values, and the metadata words
// of one token (keys on lanes 0-15, values on lanes 16-31).
struct TileData {
  uint4 key[2], value[2], key_r, value_r;
  uint32_t scale, offset, rscale, roffset, alpha;
};

template <bool kResidual>
__device__ __forceinline__ TileData load_tile(const Params& p, int row, int base, int valid, int lane) {
  const size_t row_tokens = static_cast<size_t>(row) * p.n + base;
  const size_t loop_tokens = (static_cast<size_t>(row) * kResidualLoops + p.loop) * p.n + base;
  TileData d;
  const uint4 zero = make_uint4(0, 0, 0, 0);
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int c = lane + 32 * k, t = c >> 2;
    const size_t at = (row_tokens + t) * 64 + (c & 3) * 16;
    d.key[k] = t < valid ? ldg128(p.key.anchor_codes + at) : zero;
    d.value[k] = t < valid ? ldg128(p.value.anchor_codes + at) : zero;
  }
  d.key_r = d.value_r = zero;
  if constexpr (kResidual) {
    const int t = lane >> 1;
    const size_t at = (loop_tokens + t) * 32 + (lane & 1) * 16;
    if (t < valid) {
      d.key_r = ldg128(p.key.residual_codes + at);
      d.value_r = ldg128(p.value.residual_codes + at);
    }
  }
  const int t = lane & 15;
  const Cache& cache = lane >= 16 ? p.value : p.key;
  d.scale = d.offset = d.rscale = d.roffset = d.alpha = 0;
  if (t < valid) {
    d.scale = ldg32(cache.anchor_scale, row_tokens + t);
    d.offset = ldg32(cache.anchor_offset, row_tokens + t);
    if constexpr (kResidual) {
      d.rscale = ldg32(cache.residual_scale, loop_tokens + t);
      d.roffset = ldg32(cache.residual_offset, loop_tokens + t);
      d.alpha = __ldg(cache.alpha + loop_tokens + t);
    }
  }
  return d;
}

template <bool kResidual>
__device__ __forceinline__ void store_tile(const TileData& d, WarpTile& tile, int lane) {
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const int c = lane + 32 * k, t = c >> 2;
    *reinterpret_cast<uint4*>(&tile.key_codes[anchor_at(t, (c & 3) * 16)]) = d.key[k];
    *reinterpret_cast<uint4*>(&tile.value_codes[anchor_at(t, (c & 3) * 16)]) = d.value[k];
  }
  if constexpr (kResidual) {
    const int t = lane >> 1;
    *reinterpret_cast<uint4*>(&tile.key_rcodes[residual_at(t, (lane & 1) * 16)]) = d.key_r;
    *reinterpret_cast<uint4*>(&tile.value_rcodes[residual_at(t, (lane & 1) * 16)]) = d.value_r;
  }
  const int t = lane & 15;
  const float4 s = e4m3x4_to_float4(d.scale), o = e4m3x4_to_float4(d.offset);
  if (lane < 16) {
    tile.key_scale[t] =
        make_uint4(broadcast_half(s.x), broadcast_half(s.y), broadcast_half(s.z), broadcast_half(s.w));
    tile.key_offset[t] = o;
    if constexpr (kResidual) {
      const float4 rs = e4m3x4_to_float4(d.rscale);
      tile.key_rscale[t] = make_uint4(broadcast_half(rs.x), broadcast_half(rs.y),
                                      broadcast_half(rs.z), broadcast_half(rs.w));
      tile.key_roffset[t] = e4m3x4_to_float4(d.roffset);
      tile.key_alpha[t] = bf16_bits_to_float(d.alpha);
    }
  } else {
    __half* vs = reinterpret_cast<__half*>(&tile.value_scale[t >> 1]) + (t & 1);
    vs[0] = __float2half_rn(s.x);
    vs[2] = __float2half_rn(s.y);
    vs[4] = __float2half_rn(s.z);
    vs[6] = __float2half_rn(s.w);
    if constexpr (kResidual) {
      const float4 rs = e4m3x4_to_float4(d.rscale), ro = e4m3x4_to_float4(d.roffset);
      const float alpha = bf16_bits_to_float(d.alpha);
      __half* vr = reinterpret_cast<__half*>(&tile.value_rscale[t >> 1]) + (t & 1);
      vr[0] = __float2half_rn(rs.x);
      vr[2] = __float2half_rn(rs.y);
      vr[4] = __float2half_rn(rs.z);
      vr[6] = __float2half_rn(rs.w);
      tile.value_offset[t] = make_float4(fmaf(alpha, o.x, ro.x), fmaf(alpha, o.y, ro.y),
                                         fmaf(alpha, o.z, ro.z), fmaf(alpha, o.w, ro.w));
      tile.value_alpha[t] = alpha;
    } else {
      tile.value_offset[t] = o;
    }
  }
}

// Per-thread query constants for query lane / 4.
struct QueryRegs {
  uint32_t a[16][2];       // A fragments of the query-key product: query, second part
  float sum[kGroups];      // per-group sums of the query
  float inv;               // undoes the power-of-two query scaling
  float mean3, meanr;      // dot products with the anchor and loop key means
};

// The partition's token loop, the warp merge and the block's final step.
template <typename T, bool kResidual>
__device__ __forceinline__ void run_partition(const Params& p, SharedTiles& tiles,
                                              const float (*scores)[kMaxQueries], const QueryRegs& q,
                                              int row, int part, int begin, int end) {
  constexpr int kWeightShift = kResidual ? 13 : 15;  // keeps |weight| * 2^shift <= 2^15
  constexpr float kWeightScale = static_cast<float>(1 << kWeightShift);
  constexpr float kWeightUnscale = 1.f / kWeightScale;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gu = lane >> 2, gt = lane & 3;

  float o[8][4];
#pragma unroll
  for (int mt = 0; mt < 8; ++mt) o[mt][0] = o[mt][1] = o[mt][2] = o[mt][3] = 0.f;
  float voff[kGroups] = {0.f, 0.f, 0.f, 0.f};
  float m = -INFINITY, l = 0.f;

  // An opaque copy of the FP16 bias keeps it in one register across the loop.
  uint32_t bias;
  asm volatile("mov.b32 %0, 0x64006400;" : "=r"(bias));

  WarpTile& tile = tiles.warp[warp];
  int base = begin + warp * kTile;
  TileData next;
  if (base < end) next = load_tile<kResidual>(p, row, base, min(kTile, end - base), lane);
#pragma unroll 1
  for (; base < end; base += kBlockTile) {
    const int valid = min(kTile, end - base);
    store_tile<kResidual>(next, tile, lane);
    __syncwarp();
    if (base + kBlockTile < end) {
      next = load_tile<kResidual>(p, row, base + kBlockTile, min(kTile, end - base - kBlockTile), lane);
    }

    float score[2][2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int tb = nt * 8 + gu;  // token of this thread's B fragment
      const uint4 ks = tile.key_scale[tb];
      uint4 krs = make_uint4(0, 0, 0, 0);
      if constexpr (kResidual) krs = tile.key_rscale[tb];
      float c4[4] = {0.f, 0.f, 0.f, 0.f}, c2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        const uint32_t w4 = *reinterpret_cast<const uint32_t*>(&tile.key_codes[anchor_at(tb, 16 * g + 4 * gt)]);
        const uint32_t sg = (&ks.x)[g];
        uint32_t w2 = 0, rg = 0;
        if constexpr (kResidual) {
          const uint32_t y =
              *reinterpret_cast<const uint16_t*>(&tile.key_rcodes[residual_at(tb, 8 * g + 2 * gt)]);
          w2 = __byte_perm(y, 0u, 0x4140);  // byte 1 moves to byte 2
          rg = (&krs.x)[g];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t b4 = scaled_code(nibble_pair(w4, j, bias), sg);
          mma(c4, q.a[4 * g + j][0], q.a[4 * g + j][1], b4);
          if constexpr (kResidual) {
            const uint32_t b2 = scaled_code(crumb_pair(w2, j, bias), rg);
            mma(c2, q.a[4 * g + j][0], q.a[4 * g + j][1], b2);
          }
        }
      }
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int t = nt * 8 + 2 * gt + e;
        const float a4 = fmaf(c4[2 + e], 0x1p-32f, c4[e]);
        const float4 ko = tile.key_offset[t];
        float s = fmaf(ko.x, q.sum[0], fmaf(ko.y, q.sum[1], fmaf(ko.z, q.sum[2], fmaf(ko.w, q.sum[3], q.mean3))));
        s = fmaf(a4, q.inv, s);
        if constexpr (kResidual) {
          const float a2 = fmaf(c2[2 + e], 0x1p-32f, c2[e]);
          const float4 kr = tile.key_roffset[t];
          float r = fmaf(kr.x, q.sum[0], fmaf(kr.y, q.sum[1], fmaf(kr.z, q.sum[2], fmaf(kr.w, q.sum[3], q.meanr))));
          r = fmaf(a2, q.inv, r);
          s = fmaf(tile.key_alpha[t], s, r);
        }
        score[nt][e] = t < valid ? s * kScoreScale : -INFINITY;
      }
    }

    // Online softmax for query gu over the 16 tokens held by its four lanes.
    float top = fmaxf(fmaxf(score[0][0], score[0][1]), fmaxf(score[1][0], score[1][1]));
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 1));
    top = fmaxf(top, __shfl_xor_sync(0xffffffffu, top, 2));
    top = fmaxf(m, top);
    const float carry = fast_exp2(m - top);
    m = top;
    float w[2][2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 2; ++e) w[nt][e] = fast_exp2(score[nt][e] - top);
    l = fmaf(l, carry, (w[0][0] + w[0][1]) + (w[1][0] + w[1][1]));
    if (__any_sync(0xffffffffu, carry != 1.f)) {  // some query's maximum moved
#pragma unroll
      for (int g = 0; g < kGroups; ++g) voff[g] *= carry;
      const float carry0 = __shfl_sync(0xffffffffu, carry, 8 * gt);
      const float carry1 = __shfl_sync(0xffffffffu, carry, 8 * gt + 4);
#pragma unroll
      for (int mt = 0; mt < 8; ++mt) {
        o[mt][0] *= carry0;
        o[mt][1] *= carry1;
        o[mt][2] *= carry0;
        o[mt][3] *= carry1;
      }
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const float4 vo = tile.value_offset[nt * 8 + 2 * gt + e];
        voff[0] = fmaf(w[nt][e], vo.x, voff[0]);
        voff[1] = fmaf(w[nt][e], vo.y, voff[1]);
        voff[2] = fmaf(w[nt][e], vo.z, voff[2]);
        voff[3] = fmaf(w[nt][e], vo.w, voff[3]);
      }

    // Value product: rows are channels, k is tokens, columns are queries.
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int t0 = nt * 8 + 2 * gt, t1 = t0 + 1;
      uint32_t wa_hi, wa_lo, wr_hi = 0, wr_lo = 0;
      if constexpr (kResidual) {
        split_pair(w[nt][0] * tile.value_alpha[t0] * kWeightScale,
                   w[nt][1] * tile.value_alpha[t1] * kWeightScale, wa_hi, wa_lo);
        split_pair(w[nt][0] * kWeightScale, w[nt][1] * kWeightScale, wr_hi, wr_lo);
      } else {
        split_pair(w[nt][0] * kWeightScale, w[nt][1] * kWeightScale, wa_hi, wa_lo);
      }
      const uint4 vs = tile.value_scale[nt * 4 + gt];
      uint4 vrs = make_uint4(0, 0, 0, 0);
      if constexpr (kResidual) vrs = tile.value_rscale[nt * 4 + gt];
#pragma unroll
      for (int g = 0; g < kGroups; ++g) {
        const uint32_t x = __byte_perm(
            *reinterpret_cast<const uint16_t*>(&tile.value_codes[anchor_at(t0, 16 * g + 2 * gu)]),
            *reinterpret_cast<const uint16_t*>(&tile.value_codes[anchor_at(t1, 16 * g + 2 * gu)]), 0x5410);
        const uint32_t sg = (&vs.x)[g];
        const uint32_t a0 = scaled_code(nibble_pair(x, 0, bias), sg), a1 = scaled_code(nibble_pair(x, 1, bias), sg);
        const uint32_t a2 = scaled_code(nibble_pair(x, 2, bias), sg), a3 = scaled_code(nibble_pair(x, 3, bias), sg);
        mma(o[2 * g], a0, a1, wa_hi);
        mma(o[2 * g], a0, a1, wa_lo);
        mma(o[2 * g + 1], a2, a3, wa_hi);
        mma(o[2 * g + 1], a2, a3, wa_lo);
        if constexpr (kResidual) {
          const uint32_t y = __byte_perm(tile.value_rcodes[residual_at(t0, 8 * g + gu)],
                                         tile.value_rcodes[residual_at(t1, 8 * g + gu)], 0x5410);
          const uint32_t rg = (&vrs.x)[g];
          const uint32_t r0 = scaled_code(crumb_pair(y, 0, bias), rg), r1 = scaled_code(crumb_pair(y, 1, bias), rg);
          const uint32_t r2 = scaled_code(crumb_pair(y, 2, bias), rg), r3 = scaled_code(crumb_pair(y, 3, bias), rg);
          mma(o[2 * g], r0, r1, wr_hi);
          mma(o[2 * g], r0, r1, wr_lo);
          mma(o[2 * g + 1], r2, r3, wr_hi);
          mma(o[2 * g + 1], r2, r3, wr_lo);
        }
      }
    }
    __syncwarp();
  }

  // Sum the four lanes of each query, then hand query 2 gt and 2 gt + 1's
  // offset sums to the lanes holding those output columns.
  l += __shfl_xor_sync(0xffffffffu, l, 1);
  l += __shfl_xor_sync(0xffffffffu, l, 2);
  float vq0[kGroups], vq1[kGroups];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) {
    voff[g] += __shfl_xor_sync(0xffffffffu, voff[g], 1);
    voff[g] += __shfl_xor_sync(0xffffffffu, voff[g], 2);
    vq0[g] = __shfl_sync(0xffffffffu, voff[g], 8 * gt);
    vq1[g] = __shfl_sync(0xffffffffu, voff[g], 8 * gt + 4);
  }
  __syncthreads();
  BlockState& state = tiles.state;
#pragma unroll
  for (int mt = 0; mt < 8; ++mt) {
    const int g = mt >> 1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int ch = 32 * g + 4 * gu + 2 * (mt & 1) + r;
      state.o[warp][2 * gt][ch] = fmaf(o[mt][2 * r], kWeightUnscale, vq0[g]);
      state.o[warp][2 * gt + 1][ch] = fmaf(o[mt][2 * r + 1], kWeightUnscale, vq1[g]);
    }
  }
  if (gt == 0) {
    state.m[warp][gu] = m;
    state.l[warp][gu] = l;
  }
  __syncthreads();
  finish_block<T>(p, row, part, state, scores);
}

template <typename T, bool kResidual>
__global__ void __launch_bounds__(kThreads, kMinBlocksPerSm) attention_kernel(Params p) {
  const int part = blockIdx.x, row = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int gu = lane >> 2, gt = lane & 3;
  const int head = row % kHeads;
  const int begin = part * p.part_len, end = min(p.n, begin + p.part_len);

  __shared__ SharedTiles tiles;
  __shared__ float s_qsum[kMaxQueries][kGroups];
  __shared__ float s_qmean3[kMaxQueries], s_qmeanr[kMaxQueries], s_qmul[kMaxQueries], s_qinv[kMaxQueries];
  __shared__ float s_scores[kMaxQueries][kMaxQueries];

  float (*s_query)[kDim] = tiles.query;
  for (int idx = threadIdx.x; idx < kMaxQueries * kDim; idx += kThreads) {
    const int i = idx / kDim, d = idx % kDim;
    s_query[i][d] = i < p.q ? load_float<T>(p.query, (static_cast<size_t>(row) * p.q + i) * kDim + d) : 0.f;
  }
  __syncthreads();

  const float* mean3 = p.key_mean + (head * 4 + kAnchorLoop) * kDim;
  const float* meanr = p.key_mean + (head * 4 + (kResidual ? p.loop : kAnchorLoop)) * kDim;
  for (int i = warp; i < kMaxQueries; i += kWarps) {
    float qv[4], peak = 0.f, gsum = 0.f, dot3 = 0.f, dotr = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const int d = lane * 4 + c;
      qv[c] = s_query[i][d];
      peak = fmaxf(peak, fabsf(qv[c]));
      gsum += qv[c];
      dot3 = fmaf(qv[c], mean3[d], dot3);
      dotr = fmaf(qv[c], meanr[d], dotr);
    }
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) gsum += __shfl_xor_sync(0xffffffffu, gsum, offset);
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1)
      peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, offset));
    dot3 = warp_sum(dot3);
    dotr = warp_sum(dotr);
    int exponent = 0;
    frexpf(peak, &exponent);
    const int shift = peak > 0.f ? min(15 - exponent, kMaxQueryShift) : 0;
    const float mul = ldexpf(1.f, shift);
    if ((lane & 7) == 0) s_qsum[i][lane >> 3] = gsum;
    if (lane == 0) {
      s_qmean3[i] = dot3;
      s_qmeanr[i] = kResidual ? dotr : 0.f;
      s_qmul[i] = mul;
      s_qinv[i] = ldexpf(1.f, -shift);
    }
  }
  if (p.splits == 1) new_token_scores<T>(p, row, 0, p.q, s_scores);
  __syncthreads();

  // A fragments of the query-key product: k index 2 * gt + e of step (g, j)
  // is channel 32 g + 8 gt + j + 4 e of query gu. Components FP16 cannot hold
  // exactly go to the second part (row gu + 8), scaled by a further 2^32.
  QueryRegs q;
  {
    const float mul = s_qmul[gu];
    const __half zero = __float2half_rn(0.f);
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const int d0 = 32 * (k >> 2) + 8 * gt + (k & 3);
      const float x0 = s_query[gu][d0] * mul, x1 = s_query[gu][d0 + 4] * mul;
      const __half h0 = __float2half_rn(x0), h1 = __float2half_rn(x1);
      const bool e0 = __half2float(h0) == x0, e1 = __half2float(h1) == x1;
      q.a[k][0] = pack_halves(e0 ? h0 : zero, e1 ? h1 : zero);
      q.a[k][1] = pack_halves(e0 ? zero : __float2half_rn(x0 * 0x1p32f),
                              e1 ? zero : __float2half_rn(x1 * 0x1p32f));
    }
  }
  q.inv = s_qinv[gu];
  q.mean3 = s_qmean3[gu];
  q.meanr = s_qmeanr[gu];
#pragma unroll
  for (int g = 0; g < kGroups; ++g) q.sum[g] = s_qsum[gu][g];
  __syncthreads();  // the query rows share storage with the tiles

  run_partition<T, kResidual>(p, tiles, s_scores, q, row, part, begin, end);
}

template <typename T, bool kResidual>
void launch_typed(const Params& p, cudaStream_t stream) {
  attention_kernel<T, kResidual><<<dim3(p.splits, p.rows), kThreads, 0, stream>>>(p);
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

}  // namespace fast
}  // namespace rkv
