/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include <utility>
#include <type_traits>
#include "register/op_def_registry.h"

#ifndef OP_UTIL_H_
#define OP_UTIL_H_

namespace ops {
/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if <std::is_signed<T>::value, T>::type CeilDiv(T x, T y) {
  if (y != 0 && x != 0) {
    const T quotient = x / y;
    return (x % y != 0 && ((x ^ y) >= 0)) ? (quotient + 1) : quotient;
  }

  return x;
}

/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if <std::is_unsigned<T>::value, T>::type CeilDiv(T x, T y) {
  if (y != 0 && x != 0) {
    const T quotient = x / y;
    return (x % y != 0) ? (quotient + 1) : quotient;
  }

  return x;
}


/**
 * if y is 0, return x
 */
template <typename T>
typename std::enable_if <std::is_integral<T>::value, T>::type FloorDiv(T x, T y) {
  return y == 0 ? x : x / y;
}

/**
 * if align is 0, return 0
 */
template <typename T>
typename std::enable_if <std::is_integral<T>::value, T>::type CeilAlign(T x, T align) {
  return CeilDiv(x, align) * align;
}

/**
 * if align is 0, return 0
 */
template <typename T>
typename std::enable_if <std::is_integral<T>::value, T>::type FloorAlign(T x, T align) {
  return align == 0 ? 0 : x / align * align;
}
}

#endif // OP_UTIL_H_
