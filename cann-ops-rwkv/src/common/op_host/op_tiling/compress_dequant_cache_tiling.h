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
 * \file compress_dequant_cache_tiling.h
 * \brief function of compress_dequant_cache_tiling
 */

#ifndef OPS_BUILT_IN_OP_TILING_COMPRESS_DEQUANT_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_COMPRESS_DEQUANT_CACHE_TILING_H

#include "cache_tiling.h"

namespace compress_dequant_cache_tiling {

void TilingProcess(const string& op_type, optiling::BatchmatmulParas& params, optiling::CoreStatus& coreStatus,
                   optiling::SingleCoreStatus& singleCoreStatus);
}

#endif
