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
 * \file cache_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CACHE_TILING_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CACHE_TILING_H_

#include "cube/include/cube_tiling.h"
#include "cube/include/cube_tiling_param.h"
#include "cube/include/cube_run_info.h"

namespace optiling {
namespace cachetiling {
bool GenTiling(const CubeTilingParam &params, CubeTiling &tiling);
void DestoryTilingFactory();
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CACHE_TILING_H_
