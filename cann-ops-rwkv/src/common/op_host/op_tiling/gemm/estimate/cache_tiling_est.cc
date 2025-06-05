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
 * \file cache_tiling_est.cc\
 * \brief function of gemm cache tiling est
 */
#include "cache_tiling_est.h"
#include "mathutil.h"

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

static constexpr int32_t kEstimateCapacity = 50;

namespace gemm_cache_tiling {
using namespace std;
using namespace optiling;
// using optiling::cachetiling::MathUtil;

GemmEstimate::GemmEstimate(const string& op_type, const optiling::BatchmatmulParas *paras) :
  op_type_(op_type),
  compile_params_(*(paras->compile_params)),
  run_params_(*(paras->run_params)) {
    result_vec_.reserve(kEstimateCapacity);
    core_status_vec_.reserve(kEstimateCapacity);
    single_core_status_vec_.reserve(kEstimateCapacity);
    if (run_params_.dtype_a == ge::DataType::DT_FLOAT) {
      dtype_size_ = kFp32Bytes;
      cube_k_ = reducekBlockSize;
    } else {
      dtype_size_ = kFp16Bytes;
      cube_k_ = kBlockSize;
    }
    dtype_in_ = (run_params_.dtype_a == static_cast<int32_t>(ge::DT_BF16)) ?
        static_cast<int32_t>(ge::DT_FLOAT16) : run_params_.dtype_a;
    out_dtype_size_ = run_params_.dtype_out == ge::DataType::DT_FLOAT  ? kFp32Bytes
                      : (run_params_.dtype_out == ge::DataType::DT_INT8 ? kInt8Bytes
                                                                       : kFp16Bytes);
};

bool GemmEstimate::AddEstimateTask(const GemmResultInfo &result) {
  if (!result.IsAvaliable()) {
    OP_LOGW(op_type_.c_str(), "Result tiling invalid, m_l0[%ld], k_l0[%ld], n_l0[%ld].",
            result.m_l0, result.k_l0, result.n_l0);
    return false;
  }
  result_vec_.push_back(result);
  return true;
}


bool GemmEstimate::AddEstimateTask(const CoreStatus &coreStatus, const SingleCoreStatus &singleCoreStatus) {
  L0Status l0_status = singleCoreStatus.l0Status;
  if (l0_status.m_l0 <= 0 || l0_status.k_l0 <= 0 || l0_status.n_l0 <= 0 ||
      coreStatus.batch <= 0 || coreStatus.m <= 0 || coreStatus.k <= 0 || coreStatus.n <= 0 ||
      coreStatus.batch_dim <= 0 || coreStatus.m_dim <= 0 || coreStatus.k_dim <= 0 || coreStatus.n_dim <= 0) {
    OP_LOGW(op_type_.c_str(),
            "l0_status tiling invalid, m_l0[%ld], k_l0[%ld], n_l0[%ld], batch[%ld], m[%ld], k[%ld], n[%ld].",
            l0_status.m_l0, l0_status.k_l0, l0_status.n_l0, coreStatus.batch, coreStatus.m, coreStatus.k, coreStatus.n);
    return false;
  }

  core_status_vec_.push_back(coreStatus);
  single_core_status_vec_.push_back(singleCoreStatus);
  return true;
}

bool GemmEstimate::GetEstimateResult(GemmResultInfo &out_result) {
  int32_t size = result_vec_.size();
  if (size == 0) {
    return false;
  }
  for (int32_t cur_idx = 0; cur_idx < size; cur_idx++) {
    this->result_ = result_vec_.at(cur_idx);
    SetBufferParams();
    Estimate(cur_idx);
  }
  out_result = result_vec_.at(best_idx_);
  OP_LOGD(op_type_.c_str(), "end GemmEstimate best_idx[%d]", best_idx_);
  return true;
}

bool GemmEstimate::GetEstimateResult(CoreStatus &out_core_status, SingleCoreStatus &out_single_core_status) {
  int32_t size = core_status_vec_.size();
  if (size == 0) {
    return false;
  }
  for (int32_t cur_idx = 0; cur_idx < size; cur_idx++) {
    SetBufferParams(core_status_vec_.at(cur_idx), single_core_status_vec_.at(cur_idx));
    Estimate(cur_idx);
  }
  out_core_status = core_status_vec_.at(best_idx_);
  out_single_core_status = single_core_status_vec_.at(best_idx_);
  OP_LOGD(op_type_.c_str(), "end GemmEstimate best_idx[%d]", best_idx_);
  return true;
}

void GemmEstimate::Clear() {
  best_idx_ = 0;
  result_vec_.clear();
  core_status_vec_.clear();
  single_core_status_vec_.clear();
}

void GemmEstimate::UpdateLoadFlag() {
  // initialize the full load flag
  core_status_.both_full_load = false;
  core_status_.al1_full_load = false;
  core_status_.bl1_full_load = false;
  core_status_.al1_k_full_load = false;
  core_status_.bl1_k_full_load = false;
  if (core_status_.m_single_core == 1 && core_status_.kal1_factor == 1) {
    core_status_.al1_full_load = true;
  } else if (core_status_.kal1_factor == 1) {
    core_status_.al1_k_full_load = true;
  }
  if (core_status_.n_single_core == 1 && core_status_.kbl1_factor == 1) {
    core_status_.bl1_full_load = true;
  } else if (core_status_.kbl1_factor == 1) {
    core_status_.bl1_k_full_load = true;
  }
  // Update the full_load flag in l1Status to ensure they are correct.
  if (core_status_.al1_full_load && core_status_.bl1_full_load) {
    core_status_.both_full_load = true;
  }
}

void GemmEstimate::SetBufferParams() {
  core_status_.Clear();
  core_status_.m_single_core = MathUtil::CeilDivision(MathUtil::CeilDivision(run_params_.m, result_.m_dim),
                                                      result_.m_l1);
  core_status_.n_single_core = MathUtil::CeilDivision(MathUtil::CeilDivision(run_params_.n, result_.n_dim),
                                                      result_.n_l1);
  core_status_.kal1_factor = MathUtil::CeilDivision(run_params_.k, result_.kal1_16);
  core_status_.kbl1_factor = MathUtil::CeilDivision(run_params_.k, result_.kbl1_16);
  core_status_.m = min(core_status_.m_single_core * result_.m_l1, run_params_.m);
  core_status_.k = run_params_.k;
  core_status_.n = min(core_status_.n_single_core * result_.n_l1, run_params_.n);
  core_status_.batch = MathUtil::CeilDivision(run_params_.batch, result_.batch_dim); // This is batch_single_core
  if (PlatformInfo::GetInstance().support_l0c2out() && result_.k_dim != 1) {
    int64_t k_64 = MathUtil::CeilDivision(run_params_.k, result_.k_dim);
    int64_t k_l1 = max(result_.kal1_16, result_.kbl1_16);
    int64_t k_single_core = MathUtil::CeilDivision(k_64, k_l1);
    core_status_.kal1_factor = k_single_core * (k_l1 / result_.kal1_16);
    core_status_.kbl1_factor = k_single_core * (k_l1 / result_.kbl1_16);
    core_status_.k = min(k_single_core * k_l1, run_params_.k);
  }

  result_.batch_dim = MathUtil::CeilDivision(run_params_.batch, core_status_.batch);
  result_.m_dim = MathUtil::CeilDivision(run_params_.m, core_status_.m);
  result_.n_dim = MathUtil::CeilDivision(run_params_.n, core_status_.n);
  result_.k_dim = MathUtil::CeilDivision(run_params_.k, core_status_.k);
  UpdateLoadFlag();
  l1_load_repeat_ = {1, 1};
  l0_load_repeat_ = {1, 1};
  return;
}

void GemmEstimate::SetBufferParams(const CoreStatus &core_status, const SingleCoreStatus &singlecore_status) {
  result_ = GemmResultInfo{core_status, singlecore_status};
  core_status_ = GemmEstCoreStatus{core_status, singlecore_status};
  l1_load_repeat_ = {1, 1};
  l0_load_repeat_ = {1, 1};
  return;
}

LOAD_DATA GemmEstimate::GetKFullLoadSize() {
  if (core_status_.al1_k_full_load && core_status_.bl1_k_full_load) {
    return {1, core_status_.m / result_.m_l0};
  }
  if (core_status_.al1_k_full_load) {
    return {1, core_status_.m / result_.m_l0};
  }
  return {core_status_.n / result_.n_l0, 1};
}

LOAD_DATA GemmEstimate::GetFullLoadSize() {
  if (core_status_.both_full_load) {
    return {1, 1};
  }
  if (core_status_.bl1_full_load) {
    return {core_status_.n / result_.n_l0, 1};
  }
  if (core_status_.bl1_k_full_load) {
    return {1, 1};
  }
  return {1, core_status_.m / result_.m_l0};
}

void GemmEstimate::EstimateLoadRepeat() {
  l0_load_repeat_ = {core_status_.n / result_.n_l0, core_status_.m / result_.m_l0};
  if (core_status_.al1_full_load || core_status_.bl1_full_load) {
    l1_load_repeat_ = GetFullLoadSize();
    return;
  }
  if (core_status_.al1_k_full_load || core_status_.bl1_k_full_load) {
    l1_load_repeat_ = GetKFullLoadSize();
    return;
  }
  l1_load_repeat_ =  {core_status_.n / result_.n_l0, core_status_.m / result_.m_l0};
  return;
}
}
