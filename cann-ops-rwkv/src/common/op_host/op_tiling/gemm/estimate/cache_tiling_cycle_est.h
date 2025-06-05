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
 * \file cache_tiling_cycle_est.cc\
 * \brief function of gemm cache cycle est
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_BLOCK_EST_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_BLOCK_EST_H

#include <tuple>

#include "cache_tiling.h"
#include "cache_tiling_est.h"

namespace gemm_cache_tiling {

class GemmCycleUsed {
public:
  int64_t cycle = INT64_MAX;
  int64_t mad_cycle = INT64_MAX;
  int64_t load_size = INT64_MAX;
  int64_t l0c_used = INT64_MAX;
  int64_t load_2d_times = INT64_MAX;
  int64_t repeat_load_size = INT64_MAX;
  int64_t k_l0 = 1;
public:
  void Init() {
    cycle = INT64_MAX;
    mad_cycle = INT64_MAX;
    load_size = INT64_MAX;
    l0c_used = INT64_MAX;
    load_2d_times = INT64_MAX;
    repeat_load_size = INT64_MAX;
    k_l0 = 1;
  }
};

class GemmCycleEstimate final : public GemmEstimate {
public:
  explicit GemmCycleEstimate(const string& op_type, const optiling::BatchmatmulParas *paras);

  ~GemmCycleEstimate() override = default;

public:
  friend class GemmCycleModel;

private:
  void SetBufferParams() override;
  void SetBufferParams(const optiling::CoreStatus &core_status,
      const optiling::SingleCoreStatus &singlecore_status) override;
  void Estimate(int32_t cur_idx) override;
  void GetCycle(GemmCycleUsed &cur_cycle);
  int64_t GetLoadSize();

  int64_t GetLoad2dARepeat();
  int64_t GetLoad2dBRepeat();
  void GetFullLoad2dRepeat(int64_t &load_2d_times, int64_t load_l0_repeat);
  void GetKFullLoad2dRepeat(int64_t &load_2d_times, int64_t load_l0_repeat);
  int64_t GetLoad2dRepeat();
  void SetBestCycle(const GemmCycleUsed &cycle, int32_t cur_idx);
  int64_t GetLoad2dTimes(int64_t load_l0_repeat);

private:
  int64_t al1_full_load_size_ = 0;
  int64_t bl1_full_load_size_ = 0;

  int64_t m_al1_ = 1;
  int64_t n_bl1_ = 1;
  int64_t mad_expansion_rate_ = 1;

  int64_t use_out_cycle_ = -1; // > 0, use cycle from out, such as ub
  GemmCycleUsed best_cycle;
};
}
#endif
