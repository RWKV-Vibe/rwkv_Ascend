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
 * \file cache_tiling_request_bytes.h\
 * \brief function of cache tiling request align bytes count calculator
 */

#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_REQUEST_BYTES_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_REQUEST_BYTES_H

#include "cache_tiling_common.h"
#include "cache_tiling_align_count.h"

namespace gemm_cache_tiling {

AlignCount GetRequestND2NZ(GEMM_CUBE_SIZE cube_size);

AlignCount GetRequestNZ2ND(GEMM_CUBE_SIZE cube_size);

int32_t GetSameAddressND2NZ(GEMM_CUBE_SIZE cube_size);

int32_t GetSameAddressNZ2ND(GEMM_CUBE_SIZE cube_size);
}

#endif
