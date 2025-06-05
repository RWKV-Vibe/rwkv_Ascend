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
 * \file compress_dequant_cache_tiling.cc
 * \brief function of compress_dequant_cache_tiling
 */

#define ASSERT_TRUE(cond, expr_exception_handling) \
  do {                                             \
    if (!(cond)) {                                 \
      expr_exception_handling;                     \
    }                                              \
  } while (0)

#include "compress_dequant_cache_tiling.h"

#include <chrono>
#include <mutex>
#include <thread>

#include "aoe/op_tuning_tiling/gemm_tuning_tiling.h"
#include "aoe/runtime_kb/runtime_bank_manager.h"
#include "cube/algorithm/hash/hash.h"
#include "cube/algorithm/hash/tiling_cache.h"
#include "gemm/estimate/cache_tiling_cycle_est.h"
#include "gemm/estimate/cache_tiling_cycle_model.h"
#include "gemm/estimate/cache_tiling_est.h"
#include "gemm/estimate/cache_tiling_est_mgr.h"
#include "../../mathutil.h"

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

using namespace std;
// using optiling::cachetiling::MathUtil;
using namespace gemm_cache_tiling;
using namespace optiling;

namespace {
using Estimate_Ptr = const std::unique_ptr<GemmEstimate>;
}

namespace compress_dequant_cache_tiling {

static const int64_t kBlockSize = 16;
static const int64_t kint8ReduceBlockSize = 32;
static const int32_t kInt8Bytes = 1;
static const int32_t kFp16Bytes = 2;
static const int32_t kFp32Bytes = 4;
static const int32_t kInt32Bytes = 4;
static const int32_t kInt64Bytes = 4;
static const int32_t kL0FactorLimit = 64;
static const int32_t kL0cFactorLimit = 256;

static bool CheckL1Size(int64_t amat, int64_t bmat, int64_t curExtraL1Size = 0) {
  int64_t load_size_bytes = ((amat + bmat) * kBlockSize * kint8ReduceBlockSize * kInt8Bytes + curExtraL1Size);
  return load_size_bytes <= PlatformInfo::GetInstance().l1_size;
}

static bool GetUbFactors(const string& opType, const BatchmatmulParas& params, CoreStatus& coreStatus, 
                  SingleCoreStatus& singleCoreStatus) {
  (void)opType;
  (void)coreStatus;
  const BatchmatmulRunParas& run_params = *(params.run_params);
  L0Status& l0Status = singleCoreStatus.l0Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  int32_t size_div_n_cub = l0Status.m_l0 * kBlockSize * kBlockSize * ubStatus.db_cub * kFp16Bytes;
  if (run_params.bias_flag) {
    size_div_n_cub += kBlockSize * ubStatus.db_cub * kInt32Bytes;
  }
  // dequant scale need ub
  size_div_n_cub += kBlockSize * ubStatus.db_cub * kInt64Bytes;
  int32_t n_cub_max = min(PlatformInfo::GetInstance().ub_size / size_div_n_cub, l0Status.n_l0);
  if (n_cub_max <= 0) {
    return false;
  }
  for (int32_t n_cub = n_cub_max; n_cub >= 1; n_cub--) {
    if (l0Status.n_l0 % n_cub == 0) {
      ubStatus.n_cub = n_cub;
    }
  }
  return true;
}

static void AddCondition(const string& opType, BatchmatmulParas& params, CoreStatus& coreStatus,
                  SingleCoreStatus& singleCoreStatus, Estimate_Ptr& est) {
  coreStatus.cycle = INT64_MAX;
  BatchmatmulRunParas& run_params = *(params.run_params);
  UpdateL1LoadFlag(coreStatus, singleCoreStatus);
  SetDoubleBuffer(run_params, singleCoreStatus);
  ASSERT_TRUE(GetUbFactors(opType, params, coreStatus, singleCoreStatus), return);
  coreStatus.cycle = GetCycleByModel(run_params, coreStatus, singleCoreStatus);
  est->AddEstimateTask(coreStatus, singleCoreStatus);
}
static void FastFindAl1FullLoad(const string& opType, BatchmatmulParas& params, Estimate_Ptr& est, CoreStatus& coreStatus,
                         SingleCoreStatus& singleCoreStatus) {
  BatchmatmulRunParas& run_params = *(params.run_params);
  L1Status& l1Status = singleCoreStatus.l1Status;
  L0Status& l0Status = singleCoreStatus.l0Status;
  int64_t m_dim = coreStatus.m_dim;
  l1Status.m_l1 = MathUtil::CeilDivision(run_params.m, m_dim);
  l1Status.kal1_16 = run_params.k;
  int64_t amat = l1Status.m_l1 * l1Status.kal1_16;
  ASSERT_TRUE(ExpandShape(run_params, l1Status, m_dim, true), return);
  ASSERT_TRUE(CheckL1Size(amat, 1), return);
  if (l1Status.kal1_16 % l0Status.k_l0 != 0) {
    return;
  }

  l1Status.n_l1 = l0Status.n_l0;
  l1Status.kbl1_16 = l0Status.k_l0;
  int64_t bmat = l1Status.n_l1 * l1Status.kbl1_16;
  ASSERT_TRUE(CheckL1Size(amat, bmat), return);
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return);
  OP_LOGD(opType.c_str(),
          "[L0Status](m0:%ld, k0:%ld, n0:%ld, db_l0c%ld, batch_l0:%ld), "
          "[L1Status](m_l1:%ld, k_al1:%ld, k_bl1:%ld, n_l1:%ld, db_al1:%ld, db_bl1:%ld), "
          "[BlockDim](batch:%ld, m:%ld, k:%ld, n:%ld), ",
          l0Status.m_l0, l0Status.k_l0, l0Status.n_l0, l0Status.db_l0c, l0Status.batch_l0, l1Status.m_l1,
          l1Status.kal1_16, l1Status.kbl1_16, l1Status.n_l1, l1Status.db_al1, l1Status.db_bl1, coreStatus.batch_dim,
          coreStatus.m_dim, coreStatus.k_dim, coreStatus.n_dim);
  AddCondition(opType, params, coreStatus, singleCoreStatus, est);
}
static void FastFindNotFullLoad(const string& opType, BatchmatmulParas& params, Estimate_Ptr& est, CoreStatus& coreStatus,
                         SingleCoreStatus& singleCoreStatus) {
  BatchmatmulRunParas& run_params = *(params.run_params);
  L1Status& l1Status = singleCoreStatus.l1Status;
  L0Status& l0Status = singleCoreStatus.l0Status;
  l1Status.n_l1 = l0Status.n_l0;
  l1Status.m_l1 = l0Status.m_l0;
  l1Status.kal1_16 = min(run_params.k, l0Status.k_l0);
  l1Status.kbl1_16 = min(run_params.k, l0Status.k_l0);

  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  OP_LOGD(opType.c_str(),
          "[L0Status](m0:%ld, k0:%ld, n0:%ld, db_l0c:%ld, batch_l0:%ld), "
          "[L1Status](m_l1:%ld, k_al1:%ld, k_bl1:%ld, n_l1:%ld, db_al1:%ld, db_bl1:%ld), "
          "[BlockDim](batch:%ld, m:%ld, k:%ld, n:%ld), ",
          l0Status.m_l0, l0Status.k_l0, l0Status.n_l0, l0Status.db_l0c, l0Status.batch_l0, l1Status.m_l1,
          l1Status.kal1_16, l1Status.kbl1_16, l1Status.n_l1, l1Status.db_al1, l1Status.db_bl1, coreStatus.batch_dim,
          coreStatus.m_dim, coreStatus.k_dim, coreStatus.n_dim);
  AddCondition(opType, params, coreStatus, singleCoreStatus, est);
}

void TilingProcess(const string& opType, BatchmatmulParas& params, CoreStatus& coreStatus,
                   SingleCoreStatus& singleCoreStatus) {
  BatchmatmulRunParas& run_params = *(params.run_params);
  auto est = GemmEstimateFactory::GetEstimate(CYCLE_ESTIMATE_TYPE, opType, &params);
  coreStatus.cycle = INT64_MAX;
  CoreStatus tmpCoreStatus;
  SingleCoreStatus tmpSingleCoreStatus;
  tmpSingleCoreStatus.l0Status.SetInitLoadStatus();
  tmpSingleCoreStatus.l0Status.n_l0 = run_params.nl0;
  tmpSingleCoreStatus.l0Status.k_l0 = run_params.kl0;
  for (int64_t db_l0c = kDbOff; db_l0c <= kDbOn; db_l0c++) {
    tmpSingleCoreStatus.l0Status.db_l0c = db_l0c;
    tmpSingleCoreStatus.l0Status.m_l0 =
        min(kL0cFactorLimit / (db_l0c * run_params.nl0), kL0FactorLimit / (kDbOn * run_params.kl0));
    tmpSingleCoreStatus.l0Status.m_l0 = min(run_params.m, tmpSingleCoreStatus.l0Status.m_l0);
    while (run_params.m % tmpSingleCoreStatus.l0Status.m_l0 != 0) {
      tmpSingleCoreStatus.l0Status.m_l0--;
    }
    if (tmpSingleCoreStatus.l0Status.m_l0 < kL0cFactorLimit / (kDbOn * run_params.nl0)) {
      tmpSingleCoreStatus.l0Status.db_l0c = kDbOn;
    }
    int64_t m_times = run_params.m / tmpSingleCoreStatus.l0Status.m_l0;
    int64_t n_times = run_params.n / tmpSingleCoreStatus.l0Status.n_l0;
    int64_t m_dim_max = min(PlatformInfo::GetInstance().core_num, m_times);
    for (int64_t m_dim = 1; m_dim <= m_dim_max; m_dim++) {
      int64_t n_dim_max = min(PlatformInfo::GetInstance().core_num / m_dim, n_times);
      for (int64_t n_dim = 1; n_dim <= n_dim_max; n_dim++) {
        tmpCoreStatus.n_dim = n_dim;
        tmpCoreStatus.m_dim = m_dim;
        int64_t al1_min_full_load =
            run_params.k * MathUtil::CeilDivision(run_params.m, m_dim);
        if (CheckL1Size(al1_min_full_load, 1, static_cast<int64_t>(run_params.bias_flag))) {
          FastFindAl1FullLoad(opType, params, est, tmpCoreStatus, tmpSingleCoreStatus);
        }
        FastFindNotFullLoad(opType, params, est, tmpCoreStatus, tmpSingleCoreStatus);
      }
    }
  }
  est->GetEstimateResult(coreStatus, singleCoreStatus);
  return;
}

}  // namespace compress_dequant_cache_tiling
