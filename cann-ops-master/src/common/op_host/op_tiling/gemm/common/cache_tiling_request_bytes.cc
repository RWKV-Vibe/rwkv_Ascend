/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cache_tiling_request_bytes.cpp\
 * \brief function of cache tiling request align bytes count calculator
 */
#include "cache_tiling_request_bytes.h"

using namespace gemm_cache_tiling;

namespace {
const set<int64_t> kOptChannels{32, 64, 128, 160, 192, 224, 256}; // static support dma
using COUNT_FUNC = std::function<AlignCount(int64_t d_size)>;
using ADDRESS_FUNC = std::function<int32_t(int64_t d_size)>;

inline AlignCount Align_512_ND2NZ_Func(int64_t d_size) {
  if (d_size <= 0) {
    return {0, 0, 0};
  }
  int64_t count_512 = d_size / BYTE_512;
  int32_t res = d_size % BYTE_512;
  int64_t count_256 = res / BYTE_256;
  res = res % BYTE_256;
  int64_t count_128 = res / BYTE_128;
  res = res % BYTE_128;
  count_128 += res > 0 ? 1 : 0;
  return {count_512, count_256, count_128};
};

inline AlignCount Align_256_ND2NZ_Func(int64_t d_size) {
  if (d_size <= 0) {
    return {0, 0, 0};
  }
  if (d_size - BYTE_256 < 0) {
    int64_t count_256 = d_size / BYTE_256;
    int32_t res = d_size % BYTE_256;
    int64_t count_128 = res / BYTE_128;
    res = res % BYTE_128;
    count_128 += res > 0 ? 1 : 0;
    return {0, count_256, count_128};
  }
  d_size = d_size - BYTE_256;
  AlignCount align = Align_512_ND2NZ_Func(d_size);
  align.count_256 += 1;
  return align;
}

inline AlignCount Align_384_ND2NZ_Func(int64_t d_size) {
  if (d_size <= 0) {
    return {0, 0, 0};
  }
  d_size = d_size - BYTE_128;
  AlignCount align = Align_512_ND2NZ_Func(d_size);
  align.count_128 += 1;
  return align;
}

inline AlignCount Align_128_ND2NZ_Func(int64_t d_size) {
  if (d_size <= 0) {
    return {0, 0, 0};
  }
  d_size = d_size - BYTE_128;
  AlignCount align = Align_256_ND2NZ_Func(d_size);
  align.count_128 += 1;
  return align;
}

vector<COUNT_FUNC> COUNT_FUNC_ND2NZ_VEC {Align_512_ND2NZ_Func, Align_128_ND2NZ_Func,
                                   Align_256_ND2NZ_Func, Align_384_ND2NZ_Func};

inline AlignCount Align_256_NZ2ND_Func(int64_t d_size) {
  if (d_size <= 0) {
    return {0, 0, 0};
  }
  int64_t count_256 = d_size / BYTE_256;
  int32_t res = d_size % BYTE_256;
  int64_t count_128 = res / BYTE_128;
  res = res % BYTE_128;
  count_128 += res > 0 ? 1 : 0;
  return {0, count_256, count_128};
}

inline AlignCount Align_128_NZ2ND_Func(int64_t d_size) {
  if (d_size <= 0) {
    return {0, 0, 0};
  }
  int64_t count_128 = d_size / BYTE_128;
  int32_t res = d_size % BYTE_128;
  count_128 += res > 0 ? 1 : 0;
  return {0, 0, count_128};
}

vector<COUNT_FUNC> COUNT_FUNC_NZ2ND_VEC {Align_256_NZ2ND_Func, Align_128_NZ2ND_Func};

/**
 * align ori: 1024; d: 1000; addr:512|256|128|104; same addr: 3(256|128|104)
*/
inline int32_t Address_512_ND2NZ_Func(int64_t d_size) {
  if (d_size <= BYTE_128 || d_size == BYTE_256 || d_size >= BYTE_512) {
    return 1; // align ori: 1024; d: 100; addr:100; same addr: 1(100)
  }
  if (d_size < BYTE_256 || d_size <= BYTE_384) {
    return 2; // align ori: 1024; d: 200; addr:128|72; same addr: 2(128|72)
  }
  return 3; // align ori: 1024; d: 1000; addr:512|256|128|104; same addr: 3(256|128|104)
}

inline int32_t Address_384_ND2NZ_Func(int64_t d_size) {
  if (d_size <= BYTE_128) {
    return 1;
  }
  return Address_512_ND2NZ_Func(d_size - BYTE_128);
}

inline int32_t Address_256_ND2NZ_Func(int64_t d_size) {
  if (d_size <= BYTE_128) {
    return 1;
  }
  if (d_size < BYTE_256) {
    return 2; // align ori: 1024+256; d: 200; addr:128|72; same addr: 2(128|72)
  }
  return Address_512_ND2NZ_Func(d_size - BYTE_256); // align ori: 1024+256; d: 400; addr:256|144; same addr: 2(256|144)
}

inline int32_t Address_128_ND2NZ_Func(int64_t d_size) {
  if (d_size <= BYTE_128) {
    return 1;
  }
  if (d_size <= BYTE_256) {
    return 2; // align ori: 1024+128; d: 200; addr:128|72; same addr: 2(128|72)
  }
  if (d_size < BYTE_384) {
    return 3; // align ori: 1024+128; d: 300; addr:128|128|44; same addr: 3(128|128|44)
  }
  // align ori: 512+128; d: 520; addr:128|256|(136); same addr: max(128|256, (136))
  return max(2, Address_512_ND2NZ_Func(d_size - BYTE_384));
}

vector<ADDRESS_FUNC> ADDRESS_FUNC_ND2NZ_VEC {Address_512_ND2NZ_Func, Address_128_ND2NZ_Func,
                                             Address_256_ND2NZ_Func, Address_384_ND2NZ_Func};

inline int32_t Address_NZ2ND_Func(int64_t d_size) {
  if (d_size <= BYTE_128 || d_size == BYTE_256) {
    return 1;
  }

  if (d_size < BYTE_256 || d_size <= BYTE_384 || d_size == BYTE_512) {
    return 2;  // align ori: 1024; d: 200; addr:128|72; same addr: 2(128|72) || 512->256,256
  }
  if (d_size < BYTE_512) {
    return 3; // align ori: 1024; d: 511; addr:256|128|127; same addr: 3(256|128|127);unalign: 127|128|128|127 4
  }
  return 2;  // align ori: 1024; d: 513; addr:256|256|...; same addr: 2(256|256|...)
}

inline int32_t Address_U_NZ2ND_Func(int64_t d_size) {
  if (d_size <= BYTE_128) {
    return 2; // unalign ori: d: 100; addr:30|70; same addr: 2
  }
  if (d_size <= BYTE_256) {
    return 3; // unalign ori: d: 250; addr:30|128|92; same addr: 3
  }
  if (d_size <= BYTE_384) {
    return 5; // unalign ori: d: 300; addr:30|128|98|30|14; same addr: 5
  }
  return 6; // unalign ori: d: 520; addr:30|128|98|30|128|98|-8; same addr: 6
}
}

namespace gemm_cache_tiling {
AlignCount GetRequestND2NZ(GEMM_CUBE_SIZE cube_size) {
  int64_t n = cube_size.n;
  int64_t d = cube_size.d;
  int64_t ori = cube_size.srcD;
  int32_t dtype_size = cube_size.dtype_size;
  int64_t ori_size = ori * dtype_size;
  int64_t ori_d_size = d * dtype_size;
  // burst merge
  if (d == ori && kOptChannels.find(ori_d_size) != kOptChannels.end()) {
    d = n * d;
    ori = n * ori;
    n = 1;
    ori_d_size = d * dtype_size;
    ori_size = ori * dtype_size;
  }
  int64_t ori_n = n - 1;
  int length = COUNT_FUNC_ND2NZ_VEC.size();
  AlignCount first = Align_512_ND2NZ_Func(ori_d_size);
  if (ori_size % BYTE_128 == 0) {
    int32_t align_idx = ori_size % BYTE_512 / BYTE_128;
    if (align_idx == 0) {
      return COUNT_FUNC_ND2NZ_VEC[align_idx](ori_d_size) * ori_n + first;
    }
    if (align_idx == 3 || align_idx == 1) {  // align 384 256 128 512 384 256 128 512
      AlignCount align;
      for (auto iterFunc : COUNT_FUNC_ND2NZ_VEC) {
        align += iterFunc(ori_d_size);
      }
      align *= (ori_n / 4L);  // 128，256, 384, 512
      int32_t end = ori_n % length; // end=3, start=1; end=2, start=2; end=1, start=3
      int32_t start = (ori_size % BYTE_384 == 0) ? (length - end) : 1;
      for (int32_t idx = 0; idx < end; idx++) {
        align += COUNT_FUNC_ND2NZ_VEC[idx + start](ori_d_size);
      }
      return align + first;
    }
    if (align_idx == 2) {  // align 256 512 256 512; 2:align to 256，512
      return COUNT_FUNC_ND2NZ_VEC[0](ori_d_size) * (ori_n / 2) +
             COUNT_FUNC_ND2NZ_VEC[align_idx](ori_d_size) * (ori_n / 2 + ori_n % 2) + first;
    }
  }

  AlignCount align;
  for (int64_t idx = 1; idx < n; idx++) {
    AlignCount alignTmp;
    int64_t addr = idx * ori_size;    // example: k*512+127
    int32_t res = BYTE_128 - addr % BYTE_128;   // example: res = 1
    if (res < BYTE_128) {
      alignTmp.count_128 += 1;
    } else {
      res = 0;
    }
    int64_t last = ori_d_size - res;
    if (last > 0) {
      int64_t pre_addr = addr + res;
      int32_t align_idx = pre_addr % BYTE_512 / BYTE_128;
      alignTmp += COUNT_FUNC_ND2NZ_VEC[align_idx](last);
    }
    align += alignTmp;
  }
  return align + first;
}

AlignCount GetRequestNZ2ND(GEMM_CUBE_SIZE cube_size) {
  int64_t n = cube_size.n;
  int64_t d = cube_size.d;
  int64_t ori = cube_size.srcD;
  int32_t dtype_size = cube_size.dtype_size;
  int64_t ori_size = ori * dtype_size;
  int64_t ori_d_size = d * dtype_size;
  if (ori_d_size <= BYTE_256 && d == ori) {
    d = n * d;
    ori = n * ori;
    n = 1;
    ori_d_size = d * dtype_size;
    ori_size = ori * dtype_size;
  }
  int64_t ori_n = n - 1;
  AlignCount first = Align_256_NZ2ND_Func(ori_d_size);
  if (ori_size % BYTE_128 == 0) {
    int32_t align_idx = ori_size % BYTE_256 /BYTE_128;
    if (align_idx == 0) {
      return COUNT_FUNC_NZ2ND_VEC[align_idx](ori_d_size) * ori_n + first;
    }
    if (align_idx == 1) {  // align 256 512 256 512; 2:align to 256，512
      return COUNT_FUNC_NZ2ND_VEC[0](ori_d_size) * (ori_n / 2) +
             COUNT_FUNC_NZ2ND_VEC[1](ori_d_size) * (ori_n / 2 + ori_n % 2) + first;
    }
  }

  AlignCount alignTmp;
  int64_t n_256_size = ori_d_size / BYTE_256;
  int64_t end_res = ori_d_size % BYTE_256;
  int64_t n_256_size_T = n_256_size * 3; // align 10 118 256 10
  AlignCount align_256 = Align_256_NZ2ND_Func(ori_d_size);
  AlignCount align_128 = Align_128_NZ2ND_Func(ori_d_size);
  for (int idx = 1; idx < n; idx++) {
    int64_t addr = idx * ori_size;    // k*512+127
    if (addr % BYTE_256 == 0) {
      alignTmp += align_256;
    } else if (addr % BYTE_128 == 0) {
      alignTmp += align_128;
    } else {
      if (end_res == 0) {
        alignTmp.count_128 += n_256_size_T; // res 512, add 10, data 118 128 256 10
        continue;
      }
      alignTmp.count_128 += n_256_size_T;
      int64_t cur_end_res = end_res - (BYTE_128 - addr % BYTE_128);
      alignTmp.count_128 += (cur_end_res + BYTE_128 - 1) / BYTE_128 + 1;
    }
  }
  return alignTmp + first;
}

int32_t GetSameAddressND2NZ(GEMM_CUBE_SIZE cube_size) {
  int64_t n = cube_size.n;
  int64_t d = cube_size.d;
  int64_t ori = cube_size.srcD;
  int32_t dtype_size = cube_size.dtype_size;
  int64_t ori_size = ori * dtype_size;
  int64_t ori_d_size = d * dtype_size;
  if (d == ori && kOptChannels.find(ori_d_size) != kOptChannels.end()) {
    d = n * d;
    ori = n * ori;
    n = 1;
    ori_d_size = d * dtype_size;
    ori_size = ori * dtype_size;
  }
  int32_t max_address_count = Address_512_ND2NZ_Func(ori_d_size);
  if (n == 1) {
    return max_address_count;
  }
  int64_t length = ADDRESS_FUNC_ND2NZ_VEC.size();
  int64_t repeat = min(length, n);
  int32_t align_idx = ori_size % BYTE_512 / BYTE_128;
  int64_t start = 1;
  int64_t end = length;
  int64_t shift = 1;
  if (ori_size % BYTE_128 == 0) {
    shift = 0;
    switch (align_idx) {
       case 0: // align 512
        start = 1;
        end = start;
        break;
      case 1: // align 128
        start = 1;
        end = repeat;
        break;
      case 2: // align 256
        start = align_idx;
        end = align_idx + 1;
        break;
      case 3: // align 384
        start = length - repeat;
        end = length;
        break;
      default:
        break;
    }
  }
  for (int64_t i = start; i < end; i++) {
    max_address_count = max(max_address_count, ADDRESS_FUNC_ND2NZ_VEC[i](ori_d_size));
  }
  max_address_count += shift;
  return max_address_count;
}

int32_t GetSameAddressNZ2ND(GEMM_CUBE_SIZE cube_size) {
  int64_t n = cube_size.n;
  int64_t d = cube_size.d;
  int64_t ori = cube_size.srcD;
  int32_t dtype_size = cube_size.dtype_size;
  int64_t ori_d_size = d * dtype_size;
  int64_t ori_size = ori * dtype_size;
  if (ori_d_size <= BYTE_256 && d == ori) {
    d = n * d;
    ori = n * ori;
    n = 1;
    ori_d_size = d * dtype_size;
    ori_size = ori * dtype_size;
  }
  int32_t max_address_count = 0;
  if (ori_size % BYTE_128 == 0) {
    max_address_count = max(Address_NZ2ND_Func(ori_d_size), Address_NZ2ND_Func(ori_d_size - BYTE_512));
  } else {
    max_address_count = Address_U_NZ2ND_Func(ori_d_size);
  }
  return max_address_count;
}
}

