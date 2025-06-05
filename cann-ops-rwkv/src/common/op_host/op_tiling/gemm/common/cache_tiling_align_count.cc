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
 * \file cache_tiling_align_count.cpp\
 * \brief function of cache tiling align count
 */
#include "cache_tiling_align_count.h"

namespace gemm_cache_tiling {
AlignCount AlignCount::operator+(const AlignCount& a) const {
  return {this->count_512 + a.count_512, this->count_256 + a.count_256, this->count_128 + a.count_128};
};

AlignCount& AlignCount::operator=(const AlignCount& a) {
  this->count_512 = a.count_512;
  this->count_256 = a.count_256;
  this->count_128 = a.count_128;
  return *this;
}

AlignCount& AlignCount::operator+=(const AlignCount& a) {
  this->count_512 += a.count_512;
  this->count_256 += a.count_256;
  this->count_128 += a.count_128;
  return *this;
};

AlignCount AlignCount::operator*(int64_t n) const {
  return {this->count_512 * n, this->count_256 * n, this->count_128 * n};
};

AlignCount& AlignCount::operator*=(int64_t n) {
  this->count_512 *= n;
  this->count_256 *= n;
  this->count_128 *= n;
  return *this;
};

bool AlignCount::operator!=(const AlignCount& a) const {
  return this->count_512 != a.count_512 || this->count_256 != a.count_256 || this->count_128 != a.count_128;
};

int64_t AlignCount::AlignSum() const {
  return this->count_512 * BYTE_512 + this->count_256 * BYTE_256 + this->count_128 * BYTE_128;
}

int64_t AlignCount::Size() const {
  return this->count_512 + this->count_256 + this->count_128;
}

ostream& operator<<(ostream& out, const AlignCount& a) {
  out<<a.count_512 << " " << a.count_256 << " "<<a.count_128;
  return out;
}
}
