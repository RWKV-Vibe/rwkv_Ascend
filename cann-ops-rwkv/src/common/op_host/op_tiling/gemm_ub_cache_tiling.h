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

#ifndef OPS_BUILT_IN_OP_TILING_GEMM_UB_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_GEMM_UB_CACHE_TILING_H

#include <unistd.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <ratio>
#include <vector>

#include "cube_tiling_runtime.h"
using json = nlohmann::json;

namespace optiling {
struct Ub2UbBatchmatmulCompileParas {
  int64_t block_m0 = 1;
  int64_t block_n0 = 16;
  int64_t block_a_k0 = 16;
  int64_t block_b_k0 = 16;
  bool is_batch_matmul = false;
  bool bm_fusion_flag = false;
  int64_t fifo_fusion_flag = 0;

  std::string pre_conv = "None";
  std::string pre_activation = "None";
  std::string post_anti_quant = "None";
  std::string post_eltwise = "None";
  std::string post_activation = "None";
  std::string post_quant = "None";
  std::string post_transform = "None";
};

struct Ub2UbBatchmatmulCalPbUbSize {
  int64_t a_ub_size = 0;
  int64_t b_ub_size = 0;
  int64_t c_ub_size = 0;
  int64_t bias_ub_size = 0;
  int64_t available_ub_size = 0;
};

struct Ub2UbBatchmatmulRunParasTemp {
    int64_t m1_ub = 1;
    int64_t n1_ub = 1;
    int64_t k1_aub = 1;
    int64_t k1_bub = 1;
    int64_t a_pb = 1;
    int64_t b_pb = 1;
    int64_t c_pb = 1;
    bool reorder_mn_flag = false;
};

struct Ub2UbBatchmatmulRunParas {
  int64_t core_num = 1;
  int64_t ub_size = 32 * 1024;

  int64_t batch = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t min_k_align = 1;
  int64_t m_align = 1;
  int64_t k_align = 1;
  int64_t n_align = 1;
  int64_t m1 = 1;
  int64_t ka1 = 1;
  int64_t kb1 = 1;
  int64_t n1 = 1;
  int64_t m0 = 1;
  int64_t ka0 = 16;
  int64_t kb0 = 16;
  int64_t n0 = 16;
  int64_t a_dtype_bytes = 1;
  int64_t b_dtype_bytes = 1;
  int64_t c_dtype_bytes = 4;
  int64_t bias_dtype_bytes = 4;
  int64_t a_pb = 1;
  int64_t b_pb = 1;
  int64_t c_pb = 1;

  bool bias_flag = false;
  bool trans_a_flag = false;
  bool trans_b_flag = false;
  bool channel_merge_split = false;
  bool bm_fusion_flag = false;
  int64_t fifo_fusion_flag = 0;
};

struct Ub2UbBatchmatmulParas {
  const Ub2UbBatchmatmulCompileParas *compile_params = nullptr;
  Ub2UbBatchmatmulRunParas *run_params = nullptr;
};

struct Ub2UbCoreStatus {
  int64_t batch_dim = 1;
  int64_t m_dim = 1;
  int64_t n_dim = 1;
  int64_t k_dim = 1;
  int64_t core_num_use = 1;

  int64_t m_single_core_main = 1;
  int64_t n_single_core_main = 1;
  int64_t k_single_core_main = 1;
  int64_t batch_single_core_main = 1;
};

struct Ub2UbSingleCoreStatus {
  int64_t m_single_core = 1;
  int64_t n_single_core = 1;
  int64_t k_single_core = 1;
  int64_t batch_single_core = 1;

  int64_t a_size = 0;
  int64_t b_size = 0;
  int64_t a_ub_size = 0;
  int64_t b_ub_size = 0;
  int64_t bias_ub_size = 0;
  int64_t available_ub_size = 0;

  int64_t m1_ub = 1;
  int64_t n1_ub = 1;
  int64_t k1_aub = 1;
  int64_t k1_bub = 1;
  int64_t m1_mmad = 1;
  int64_t n1_mmad = 1;
  int64_t k1_ammad = 1;
  int64_t k1_bmmad = 1;

  bool reorder_mn_flag = false;
};

class Ub2UbTiling {
public:
  uint64_t tiling_id;
  int64_t block_m = 1;
  int64_t block_n = 1;
  int64_t m_ub = 1;
  int64_t n_ub = 1;
  int64_t k_aub = 1;
  int64_t k_bub = 1;
  int64_t k_mmad = 1;
  int64_t m_mmad = 1;
  int64_t n_mmad = 1;
  Ub2UbTiling() = default;
  void SetParams(const std::string &op_type, const Ub2UbCoreStatus &coreStatus,
                 const Ub2UbSingleCoreStatus &singleCoreStatus, const Ub2UbBatchmatmulParas &params);
  void GetTilingId(const Ub2UbBatchmatmulParas &params, const Ub2UbSingleCoreStatus &singleCoreStatus);
  ~Ub2UbTiling() = default;
};

void Ub2UbGenTiling(const std::string &op_type, const Ub2UbBatchmatmulCompileParas &compile_params,
                    Ub2UbBatchmatmulRunParas &run_params, Ub2UbTiling &tiling, uint64_t &tilingId);
}; // namespace optiling

#endif
