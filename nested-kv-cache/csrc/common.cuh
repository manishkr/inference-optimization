#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

namespace fast {

constexpr int D = 128;
constexpr int BLOCK = 256;
constexpr int WINDOW = 64;
constexpr float EPS = 1e-12f;
constexpr float EPS_SQ = 1e-24f;

__device__ __forceinline__ float fast_inv_sqrt(float sq, float thresh_sq) {
  float x = fmaxf(sq, thresh_sq);
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

struct __align__(16) float8 {
  float4 lo;
  float4 hi;
};

struct Float8Row16 {
  float4 lo[16];
  float4 hi[16];
};

__device__ __forceinline__ void store_f8(Float8Row16& row, int sub_lane, float8 v) {
  row.lo[sub_lane] = v.lo;
  row.hi[sub_lane] = v.hi;
}

__device__ __forceinline__ float8 load_f8(const Float8Row16& row, int sub_lane) {
  float8 v;
  v.lo = row.lo[sub_lane];
  v.hi = row.hi[sub_lane];
  return v;
}

__device__ __forceinline__ uint4 ldg_stream_u4(const uint4* ptr) {
  uint4 res;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
      : "l"(ptr));
  return res;
}

__device__ __forceinline__ float8 load_f8_keep(const Float8Row16& row, int sub_lane) {
  float8 v;
  const float4* lo_ptr = &row.lo[sub_lane];
  const float4* hi_ptr = &row.hi[sub_lane];
  asm("ld.global.nc.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(v.lo.x), "=f"(v.lo.y), "=f"(v.lo.z), "=f"(v.lo.w)
      : "l"(lo_ptr));
  asm("ld.global.nc.L1::evict_last.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(v.hi.x), "=f"(v.hi.y), "=f"(v.hi.z), "=f"(v.hi.w)
      : "l"(hi_ptr));
  return v;
}

__device__ __forceinline__ float4 load_f4_lane(const Float8Row16& row, int lane) {
  return reinterpret_cast<const float4*>(&row)[lane];
}

__device__ __forceinline__ void store_f4_lane(Float8Row16& row, int lane, float4 v) {
  reinterpret_cast<float4*>(&row)[lane] = v;
}

__device__ __forceinline__ float4 add_float4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 scale_float4(float4 v, float s) {
  return make_float4(v.x * s, v.y * s, v.z * s, v.w * s);
}

__device__ __forceinline__ float dot_float4(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ __forceinline__ float warp_allreduce_sum(float val) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, mask);
  }
  return val;
}

__device__ __forceinline__ float8 zero_float8() {
  float8 res;
  res.lo = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  res.hi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  return res;
}

__device__ __forceinline__ float8 scale_float8(float8 v, float s) {
  float8 res;
  res.lo = make_float4(v.lo.x * s, v.lo.y * s, v.lo.z * s, v.lo.w * s);
  res.hi = make_float4(v.hi.x * s, v.hi.y * s, v.hi.z * s, v.hi.w * s);
  return res;
}

__device__ __forceinline__ float8 add_float8(float8 a, float8 b) {
  float8 res;
  res.lo = make_float4(a.lo.x + b.lo.x, a.lo.y + b.lo.y, a.lo.z + b.lo.z, a.lo.w + b.lo.w);
  res.hi = make_float4(a.hi.x + b.hi.x, a.hi.y + b.hi.y, a.hi.z + b.hi.z, a.hi.w + b.hi.w);
  return res;
}

__device__ __forceinline__ float8 fma_float8(float8 v, float s, float8 acc) {
  float8 res;
  res.lo = make_float4(
      fmaf(v.lo.x, s, acc.lo.x), fmaf(v.lo.y, s, acc.lo.y),
      fmaf(v.lo.z, s, acc.lo.z), fmaf(v.lo.w, s, acc.lo.w));
  res.hi = make_float4(
      fmaf(v.hi.x, s, acc.hi.x), fmaf(v.hi.y, s, acc.hi.y),
      fmaf(v.hi.z, s, acc.hi.z), fmaf(v.hi.w, s, acc.hi.w));
  return res;
}

__device__ __forceinline__ float dot_float8(float8 a, float8 b) {
  float s0 = a.lo.x * b.lo.x + a.lo.y * b.lo.y + a.lo.z * b.lo.z + a.lo.w * b.lo.w;
  float s1 = a.hi.x * b.hi.x + a.hi.y * b.hi.y + a.hi.z * b.hi.z + a.hi.w * b.hi.w;
  return s0 + s1;
}

template <bool IS_BF16>
__device__ __forceinline__ float8 raw_to_float8(uint4 raw) {
  float8 res;
  if constexpr (IS_BF16) {
    asm("{\n\tshl.b32 %0, %2, 16;\n\tand.b32 %1, %2, 0xffff0000;\n\t}"
        : "=f"(res.lo.x), "=f"(res.lo.y) : "r"(raw.x));
    asm("{\n\tshl.b32 %0, %2, 16;\n\tand.b32 %1, %2, 0xffff0000;\n\t}"
        : "=f"(res.lo.z), "=f"(res.lo.w) : "r"(raw.y));
    asm("{\n\tshl.b32 %0, %2, 16;\n\tand.b32 %1, %2, 0xffff0000;\n\t}"
        : "=f"(res.hi.x), "=f"(res.hi.y) : "r"(raw.z));
    asm("{\n\tshl.b32 %0, %2, 16;\n\tand.b32 %1, %2, 0xffff0000;\n\t}"
        : "=f"(res.hi.z), "=f"(res.hi.w) : "r"(raw.w));
  } else {
    half2 h0 = *reinterpret_cast<const half2*>(&raw.x);
    half2 h1 = *reinterpret_cast<const half2*>(&raw.y);
    half2 h2 = *reinterpret_cast<const half2*>(&raw.z);
    half2 h3 = *reinterpret_cast<const half2*>(&raw.w);
    float2 f0 = __half22float2(h0);
    float2 f1 = __half22float2(h1);
    float2 f2 = __half22float2(h2);
    float2 f3 = __half22float2(h3);
    res.lo = make_float4(f0.x, f0.y, f1.x, f1.y);
    res.hi = make_float4(f2.x, f2.y, f3.x, f3.y);
  }
  return res;
}

template <bool IS_BF16>
__device__ __forceinline__ void dot_raw_u4(uint4 raw, const float8& g, float& sq, float& dvg) {
  if constexpr (IS_BF16) {
    float2 f0, f1, f2, f3;
    asm("{\n\tshl.b32 %0, %2, 16;\n\tand.b32 %1, %2, 0xffff0000;\n\t}"
        : "=f"(f0.x), "=f"(f0.y) : "r"(raw.x));
    asm("{\n\tshl.b32 %0, %2, 16;\n\tand.b32 %1, %2, 0xffff0000;\n\t}"
        : "=f"(f1.x), "=f"(f1.y) : "r"(raw.y));
    sq = f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y;
    dvg = f0.x * g.lo.x + f0.y * g.lo.y + f1.x * g.lo.z + f1.y * g.lo.w;
    asm("{\n\tshl.b32 %0, %2, 16;\n\tand.b32 %1, %2, 0xffff0000;\n\t}"
        : "=f"(f2.x), "=f"(f2.y) : "r"(raw.z));
    asm("{\n\tshl.b32 %0, %2, 16;\n\tand.b32 %1, %2, 0xffff0000;\n\t}"
        : "=f"(f3.x), "=f"(f3.y) : "r"(raw.w));
    sq += f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
    dvg += f2.x * g.hi.x + f2.y * g.hi.y + f3.x * g.hi.z + f3.y * g.hi.w;
  } else {
    float2 f0 = __half22float2(*reinterpret_cast<const half2*>(&raw.x));
    float2 f1 = __half22float2(*reinterpret_cast<const half2*>(&raw.y));
    sq = f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y;
    dvg = f0.x * g.lo.x + f0.y * g.lo.y + f1.x * g.lo.z + f1.y * g.lo.w;
    float2 f2 = __half22float2(*reinterpret_cast<const half2*>(&raw.z));
    float2 f3 = __half22float2(*reinterpret_cast<const half2*>(&raw.w));
    sq += f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
    dvg += f2.x * g.hi.x + f2.y * g.hi.y + f3.x * g.hi.z + f3.y * g.hi.w;
  }
}

__device__ __forceinline__ float halfwarp_allreduce_sum(float val, uint32_t hw_mask) {
  #pragma unroll
  for (int mask = 8; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(hw_mask, val, mask);
  }
  return val;
}

}  // namespace fast
