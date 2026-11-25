#pragma once

#include <cstdint>

#include <vector_types.h>

namespace rkv {

constexpr int kDim = 128;
constexpr int kGroups = 4;
constexpr int kGroupSize = 32;
constexpr int kHeads = 16;
constexpr int kResidualLoops = 3;
constexpr int kAnchorLoop = 3;
constexpr int kMaxQueries = 8;
constexpr int kMaxLength = 32768;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
static_assert(kThreads == kDim, "the block merge maps one thread to one channel");

struct Cache {
  const uint8_t* anchor_codes;     // [rows, N, 64], two INT4 codes per byte
  const uint8_t* anchor_scale;     // [rows, N, 4], E4M3
  const uint8_t* anchor_offset;    // [rows, N, 4], E4M3
  const uint8_t* residual_codes;   // [rows, 3, N, 32], four INT2 codes per byte
  const uint8_t* residual_scale;   // [rows, 3, N, 4], E4M3
  const uint8_t* residual_offset;  // [rows, 3, N, 4], E4M3
  const uint16_t* alpha;           // [rows, 3, N], bfloat16 bits
};

struct Params {
  const void* query;      // [rows, Q, 128]
  const void* new_key;    // [rows, Q, 128], keys of the Q uncached tokens
  const void* new_value;  // [rows, Q, 128]
  Cache key;
  Cache value;
  const float* key_mean;    // [16, 4, 128]
  float* out;               // [rows, Q, 128]
  float* lse;               // [rows, Q]
  float* partial_o;         // [rows, splits, Q, 128]
  float2* partial_ml;       // [rows, splits, Q]: (max, sum) in the base-2 domain
  int rows;
  int n;
  int q;
  int loop;
  int splits;
  int part_len;
};

}  // namespace rkv
