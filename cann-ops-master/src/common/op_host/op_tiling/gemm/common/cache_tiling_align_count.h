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
 * \file cache_tiling_align_count.h\
 * \brief function of cache tiling align count
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_ALIGN_COUNT_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_ALIGN_COUNT_H

#include<iostream>
#include "cache_tiling.h"

namespace gemm_cache_tiling {
using namespace std;

constexpr int32_t BYTE_512 = 512;
constexpr int32_t BYTE_384 = 384;
constexpr int32_t BYTE_256 = 256;
constexpr int32_t BYTE_128 = 128;

class AlignCount {
public:
  int64_t count_512;
  int64_t count_256;
  int64_t count_128;
  int64_t max_same_address_count = 0;

public:
  AlignCount() : count_512(0), count_256(0), count_128(0) {};

  AlignCount(int64_t count_a, int64_t count_b, int64_t count_c) :
    count_512(count_a),
    count_256(count_b),
    count_128(count_c) {};

  AlignCount(const AlignCount& a) :
    count_512(a.count_512),
    count_256(a.count_256),
    count_128(a.count_128) {};

  AlignCount operator+(const AlignCount& a) const;
  AlignCount& operator=(const AlignCount& a);
  AlignCount& operator+=(const AlignCount& a);
  AlignCount operator*(int64_t n) const;
  AlignCount& operator*=(int64_t n);
  bool operator!=(const AlignCount& a) const;
  int64_t AlignSum() const;
  int64_t Size() const;

  friend ostream& operator<<(ostream& out, const AlignCount &a);
};
}
#endif
