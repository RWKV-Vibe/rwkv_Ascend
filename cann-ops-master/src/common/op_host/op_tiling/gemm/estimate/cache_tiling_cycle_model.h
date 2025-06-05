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
 * \file cache_tiling_cycle_model.h\
 * \brief function of gemm cache cycle model
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_CYCLE_MODEL_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_CYCLE_MODEL_H

#include "cache_tiling.h"
#include "cache_tiling_cycle_est.h"
#include "gemm/common/cache_tiling_common.h"

namespace gemm_cache_tiling {

class GemmCycleModel {
public:
  explicit GemmCycleModel(const optiling::BatchmatmulRunParas &run_params, const optiling::CoreStatus &coreStatus,
      const optiling::SingleCoreStatus &singleCoreStatus);

  explicit GemmCycleModel(const GemmCycleEstimate* const est);

  virtual ~GemmCycleModel() = default;

public:
  virtual int64_t GetCycleModel() = 0;

protected:
  int64_t GetMadCycle();
  int64_t GetMte1Cycle(int64_t n_burst, int64_t burst_length, int64_t bandwidth);
  void GetCycleFinalCycle();

protected:
  const optiling::BatchmatmulRunParas &run_params_;
  const GemmEstCoreStatus core_status_;
  const GemmResultInfo result_;
  int64_t m_al1_ = 1;
  int64_t n_bl1_ = 1;
  int64_t mad_expansion_rate_;
  int64_t cube_k_ = 1;
  int32_t dtype_size_ = 0;
  int32_t out_dtype_size_ = 0;

  int64_t final_cycle_ = 0;
  int64_t mad_cycle_ = 0;
  int64_t mte1_al0_cycle_ = 0;
  int64_t mte1_bl0_cycle_ = 0;
  int64_t mte2_a_cycle_ = 0;
  int64_t mte2_b_cycle_ = 0;

private:
  void BothFullLoadMode();
  void AL1FullLoadMode();
  void BL1FullLoadMode();
  void BothKFullLoadMode();
  void SingleKFullLoadMode();
  void NotFullLoadMode();
};

class GemmCycleModelL0c2out final : public GemmCycleModel {
public:
  explicit GemmCycleModelL0c2out(const optiling::BatchmatmulRunParas &run_params,
      const optiling::CoreStatus &coreStatus,
      const optiling::SingleCoreStatus &singleCoreStatus) :
    GemmCycleModel(run_params, coreStatus, singleCoreStatus) {};

  explicit GemmCycleModelL0c2out(const GemmCycleEstimate* const est) :
    GemmCycleModel(est) {};

  ~GemmCycleModelL0c2out() override = default;

public:
  int64_t GetCycleModel() override;

private:
  int64_t GetMte2Al1Cycle();
  int64_t GetMte2Bl1Cycle();
  int64_t GetFixedPipeCycle();
};

class GemmCycleModelUB final : public GemmCycleModel {
public:
  explicit GemmCycleModelUB(const optiling::BatchmatmulRunParas &run_params, const optiling::CoreStatus &coreStatus,
      const optiling::SingleCoreStatus &singleCoreStatus) :
    GemmCycleModel(run_params, coreStatus, singleCoreStatus),
    ubStatus(singleCoreStatus.ubStatus) {};

  explicit GemmCycleModelUB(const GemmCycleEstimate* const est) = delete;
  ~GemmCycleModelUB() override = default;

  int64_t GetCycleModel() override;

private:
  int64_t GetMte2AubCycle();
  int64_t GetMte2BubCycle();
  int64_t GetMte2Al1Cycle();
  int64_t GetMte2Bl1Cycle();
  int64_t GetMte3CubCycle();
  int64_t GetLoad2dARepeat();
  int64_t GetLoad2dBRepeat();
  int64_t GetBandwidthUsage(int64_t burst_length);
private:
  const optiling::UbStatus& ubStatus;
  int64_t mte3_cycle = 0;
};

int64_t GetCycleByModel(const optiling::BatchmatmulRunParas &run_params, const optiling::CoreStatus &coreStatus,
    const optiling::SingleCoreStatus &singleCoreStatus);
}

#endif

