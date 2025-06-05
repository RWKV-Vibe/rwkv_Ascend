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
 * \file cache_tiling_basic_block.h\
 * \brief function of cache tiling basic block method
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_BLOCK_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_BLOCK_H
#include <list>
#include "cache_tiling.h"

namespace optiling {
enum GenTilingStatus {
  GEN_TILING_EAGAIN, // jump to next calculator
  GEN_TILING_EOF, // basic block calculator eof, jump back
};

optiling::GenTilingStatus GenTilingFromBasicBlock(const string &op_type, optiling::BatchmatmulParas &params,
    optiling::CoreStatus &coreStatus, optiling::SingleCoreStatus &singleCoreStatus);

void GenTuningFromBasicBlock(const string &op_type, optiling::BatchmatmulParas &params,
    std::list<optiling::CoreStatus> &coreStatusList, std::list<optiling::SingleCoreStatus> &singleCoreStatusList);
}

#endif
