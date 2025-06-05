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
 * \file cache_tiling_est.h\
 * \brief function of gemm cache tiling est
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_EST_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_EST_H

#include <tuple>
#include <memory>

#include "cache_tiling.h"
#include "gemm/common/cache_tiling_common.h"

namespace gemm_cache_tiling {

using LOAD_DATA = tuple<int32_t, int32_t>;

class  GemmEstimate {
public:
  explicit GemmEstimate(const string& op_type, const optiling::BatchmatmulParas *paras);
  virtual ~GemmEstimate() = default;

public:
  bool AddEstimateTask(const GemmResultInfo &result);
  bool AddEstimateTask(const optiling::CoreStatus &coreStatus, const optiling::SingleCoreStatus &singleCoreStatus);
  bool GetEstimateResult(GemmResultInfo &result);
  bool GetEstimateResult(optiling::CoreStatus &coreStatus, optiling::SingleCoreStatus &singleCoreStatus);
  int32_t GetBestIdx() const { return best_idx_; }
  void Clear();

protected:
  virtual void SetBufferParams();
  virtual void SetBufferParams(const optiling::CoreStatus &out_core_status,
      const optiling::SingleCoreStatus &out_single_core_status);
  virtual void Estimate(int32_t cur_idx) = 0;

  void UpdateLoadFlag();
  LOAD_DATA GetKFullLoadSize();
  LOAD_DATA GetFullLoadSize();
  void EstimateLoadRepeat();

protected:
  const string &op_type_;
  const optiling::BatchmatmulCompileParas &compile_params_;
  const optiling::BatchmatmulRunParas &run_params_;
  vector<GemmResultInfo> result_vec_;
  vector<optiling::CoreStatus> core_status_vec_;
  vector<optiling::SingleCoreStatus> single_core_status_vec_;
  int32_t best_idx_ = 0;

  GemmResultInfo result_;
  GemmEstCoreStatus core_status_;

  std::tuple<int32_t, int32_t> l1_load_repeat_{1, 1};
  std::tuple<int32_t, int32_t> l0_load_repeat_{1, 1};

  int32_t dtype_in_;
  int32_t dtype_size_;
  int32_t out_dtype_size_;
  int64_t cube_k_;
};

using GemmEstimatePtr = std::unique_ptr<GemmEstimate>;
}
#endif
