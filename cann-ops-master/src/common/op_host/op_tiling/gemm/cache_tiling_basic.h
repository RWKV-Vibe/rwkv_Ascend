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
 * \file cache_tiling_basic.h
 * \brief function of cache tiling basic
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_H

namespace gemm_cache_tiling {
using namespace std;
constexpr int32_t TILE_M_IDX = 0; // BasicBlock_TILE M index
constexpr int32_t TILE_N_IDX = 1; // BasicBlock_TILE N index
constexpr int32_t TILE_K_IDX = 2; // BasicBlock_TILE K index
const vector<int32_t> BB_IDX = {2, 3, 4, 7, 8, 11, 14, 15, 18, 19, 22, 25,
                              28, 31, 34, 37, 40, 41, 42, 43, 44};
const vector<int32_t> BB_IDX_1982 = {2, 3, 4, 7, 8, 11, 14, 15, 18, 19, 22, 25,
                              28, 31, 34, 37, 40, 41, 42, 43, 44, 50, 51};
const vector<int32_t> BB_IDX_QUANT_BMM_V3 = {45, 46, 47, 48, 49};

const array<array<int64_t, 3>, 52> BASIC_BLOCK_TILE {
  64,   64,   32,
  64,   64,   64,
  64,   64,   256,
  64,   64,   512,
  64,   64,   1024,
  64,   128,  64,
  64,   128,  128,
  64,   128,  256,
  64,   128,  512,
  64,   256,  64,
  64,   256,  128,
  64,   256,  256,
  128,  64,   64,
  128,  64,   128,
  128,  64,   256,
  128,  64,   512,
  128,  128,  64,
  128,  128,  128,
  128,  128,  256,
  128,  128,  512,
  256,  64,   64,
  256,  64,   128,
  256,  64,   256,
  96,   320,  64,
  96,   320,  128,
  96,   320,  256,
  128,  192,  64,
  128,  192,  128,
  128,  192,  256,
  128,  256,  64,
  128,  256,  128,
  128,  256,  256,
  192,  128,  64,
  192,  128,  128,
  192,  128,  256,
  256,  128,  64,
  256,  128,  128,
  256,  128,  256,
  320,  96,   64,
  320,  96,   128,
  320,  96,   256,
  320,  64,   256,
  64,   320,  256,
  512,  64,   128,
  64,   512,  128,
  16,   256,  64,
  16,   256,  128,
  16,   256,  256,
  16,   256,  512,
  16,   64,   32,
  256,  256,  256,
  128,  512,  128,
};
}
#endif

