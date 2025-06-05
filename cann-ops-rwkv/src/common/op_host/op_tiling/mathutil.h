/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATHUTIL_H
#define MATHUTIL_H

#include <algorithm>
#include <cmath>
#include <numeric>

#include "op_util.h"

class MathUtil {
 public:
  template <typename T>
  static T CeilDivision(T num1, T num2) {
    return ops::CeilDiv(num1, num2);
  }

  static int64_t CeilDivision(int64_t num1, int32_t num2) {
    return ops::CeilDiv(num1, static_cast<int64_t>(num2));
  }
  static int64_t Align(int32_t num1, int32_t num2) { return static_cast<int64_t>(CeilDivision(num1, num2)) * num2; }

  static uint64_t Align(uint32_t num1, uint32_t num2) {
    return static_cast<uint64_t>(CeilDivision(num1, num2)) * num2;
  }

  static int64_t Align(int64_t num1, int32_t num2) { return CeilDivision(num1, static_cast<int64_t>(num2)) * num2; }

  static int64_t Align(int64_t num1, int64_t num2) { return CeilDivision(num1, num2) * num2; }

  static uint64_t Align(uint64_t num1, uint64_t num2) { return CeilDivision(num1, num2) * num2; }

  static int64_t Align(int64_t num1, uint32_t num2) { return CeilDivision(num1, static_cast<int64_t>(num2)) * num2; }

  static int64_t Align(int32_t num1, int64_t num2) { return CeilDivision(static_cast<int64_t>(num1), num2) * num2; }

  static bool IsEqual(float l_value, float r_value) {
    return std::fabs(l_value - r_value) <= std::numeric_limits<float>::epsilon();
  }
  static int32_t AlignDown(int32_t num1, int32_t num2) {
    if (num2 == 0) {
      return 0;
    }
    return (num1 / num2) * num2;
  }
  static int32_t Min(int64_t num1, int32_t num2) {
    return static_cast<int32_t>(std::min(num1, static_cast<int64_t>(num2)));
  }

  static int32_t Min(int32_t num1, int64_t num2) {
    return static_cast<int32_t>(std::min(num2, static_cast<int64_t>(num1)));
  }

  static int32_t Min(int64_t num1, int64_t num2) {
    return static_cast<int32_t>(std::min(num2, num1));
  }
};

#endif // MATHUTIL_H
