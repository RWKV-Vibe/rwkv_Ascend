/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_COMMON_UTILS_KB_STATUS_H_
#define RUNTIME_KB_COMMON_UTILS_KB_STATUS_H_
namespace RuntimeKb {
using Status = uint32_t;

#define KB_DEF_ERRORNO(sysid, modid, name, value, desc)                                                           \
  constexpr RuntimeKb::Status name = ((((static_cast<uint32_t>(0xFFU & (static_cast<uint8_t>(sysid)))) << 24U) |  \
                                       ((static_cast<uint32_t>(0xFFU & (static_cast<uint8_t>(modid)))) << 16U)) | \
                                      (0xFFFFU & (static_cast<uint16_t>(value))));

KB_DEF_ERRORNO(0U, 0U, SUCCESS, 0U, "Success");
KB_DEF_ERRORNO(1U, 1U, KEY_NOT_EXIST, 1U, "Key not exists");
KB_DEF_ERRORNO(0xFFU, 0xFFU, FAILED, 0xFFFFU, "Failed");
}  // namespace RuntimeKb
#endif

