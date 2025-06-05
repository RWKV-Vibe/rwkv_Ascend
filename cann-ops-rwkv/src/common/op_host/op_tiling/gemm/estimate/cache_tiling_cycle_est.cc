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
 * \file cache_tiling_cycle_est.h\
 * \brief function of gemm cache cycle est
 */
#include <algorithm>

#include "cache_tiling_cycle_est.h"
#include "cache_tiling_cycle_model.h"
#include "cache_tiling_est_mgr.h"
#include "mathutil.h"

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

namespace {

static const int32_t kHbmBandwidth = 64;
static int64_t band_width[2][257]; // init 1~256 burstlen's real burstlen

int64_t GetBandwidthUsageNd2Nz(int64_t burst_length, int32_t dtype_size_bw) {
  // 512 is base pkg size means 512B
  const int32_t base_pkg_size = 512;
  uint64_t burstLengthByte = static_cast<uint64_t>(burst_length * dtype_size_bw * gemm_cache_tiling::kBlockSize);
  int64_t cache_line_pkg_num = 0;
  // support_cache_line_size's value is the n from "pow(2, n)", example 9 means 512B. we can do fast div by ">> n"
  uint32_t const support_cache_line_size[] = {9, 8, 7};
  for (const uint32_t &cache_line : support_cache_line_size) {
    int64_t curCacheLinePkgNum = (burstLengthByte >> cache_line);
    // support_cache_line_size's value is the n from "pow(2, n)"
    burstLengthByte = burstLengthByte & static_cast<uint64_t>((1L << cache_line) - 1);
    cache_line_pkg_num += curCacheLinePkgNum;
  }
  if (burstLengthByte != 0) {
    cache_line_pkg_num += 1;
  }

  return cache_line_pkg_num * base_pkg_size / kHbmBandwidth;
}

void InitBandWidth() {
  // init 1~256 burstlen's real burstlen
  for (int burst_length = 0; burst_length <= 256; ++burst_length) {
    // Dtype: 0: Float; 1: Float16
    band_width[0][burst_length] = GetBandwidthUsageNd2Nz(burst_length, gemm_cache_tiling::kFp32Bytes);
    band_width[1][burst_length] = GetBandwidthUsageNd2Nz(burst_length, gemm_cache_tiling::kFp16Bytes);
  }
}

__attribute__((constructor)) void InitThread() {
  InitBandWidth();
}
}

namespace gemm_cache_tiling {
using namespace std;
using namespace optiling;
// using optiling::cachetiling::MathUtil;

REG_GEMM_ESTIMATE_FUNC(CYCLE_ESTIMATE_TYPE, GemmCycleEstimate);

GemmCycleEstimate::GemmCycleEstimate(const string& op_type, const optiling::BatchmatmulParas *paras) :
  GemmEstimate(op_type, paras) {
    if (run_params_.dtype_a == ge::DataType::DT_FLOAT) {
      mad_expansion_rate_ = run_params_.hf32_flag ? 1 : 2; // FP32 is 2 times FP16 cube busy cycle
    }
  };

void GemmCycleEstimate::SetBufferParams() {
  GemmEstimate::SetBufferParams();
  m_al1_ = MathUtil::CeilDivision(result_.m_l1, result_.m_l0);
  n_bl1_ = MathUtil::CeilDivision(result_.n_l1, result_.n_l0);
  al1_full_load_size_ = core_status_.batch * core_status_.k * core_status_.m;
  bl1_full_load_size_ = (run_params_.b_have_batch ? core_status_.batch : 1) * core_status_.k * core_status_.n;
  return;
}

void GemmCycleEstimate::SetBufferParams(const CoreStatus &core_status, const SingleCoreStatus &singlecore_status) {
  GemmEstimate::SetBufferParams(core_status, singlecore_status);
  m_al1_ = singlecore_status.l1Status.m_al1;
  n_bl1_ = singlecore_status.l1Status.n_bl1;
  al1_full_load_size_ = core_status.batch * core_status.k * core_status.m;
  bl1_full_load_size_ = (run_params_.b_have_batch ? core_status.batch : 1) * core_status.k * core_status.n;
  use_out_cycle_ = core_status.cycle; // > 0, use cycle from out, such as ub
  return;
}

void GemmCycleEstimate::SetBestCycle(const GemmCycleUsed &cycle, int32_t cur_idx) {
    OP_LOGD(op_type_.c_str(), "update success %d", cur_idx);
    best_cycle = cycle;
    best_idx_ = cur_idx;
}

void GemmCycleEstimate::Estimate(int32_t cur_idx) {
  GemmCycleUsed cur_cycle;
  GetCycle(cur_cycle);
  OP_LOGD(op_type_.c_str(),
          "Estimate cur_idx:%d, "
          "[L0Status](m0:%ld, k0:%ld, n0:%ld, db_l0c:%ld, batch_l0:%ld), "
          "[L1Status](m_l1:%ld, k_al1:%ld, k_bl1:%ld, n_l1:%ld, db_al1:%ld, db_bl1:%ld), "
          "[CoreStatus](batch:%ld, m:%ld,k:%ld, n:%ld), "
          "[BlockDim](batch:%ld, m:%ld, k:%ld, n:%ld), "
          "[Cycle](cur:%ld, min:%ld), [LoadSize](cur:%ld, min:%ld), "
          "[mad_cycle](cur:%ld, min:%ld), [l0c_used](cur:%ld, min:%ld), "
          "[load_2d_times](cur:%ld, min:%ld), [repeat_load_size](cur:%ld, min:%ld), "
          "[k_l0](cur:%ld, min:%ld)",
          cur_idx,
          result_.m_l0, result_.k_l0, result_.n_l0, result_.db_l0c, result_.batch_l0,
          m_al1_ * result_.m_l0, result_.kal1_16, result_.kbl1_16, n_bl1_ * result_.n_l0, result_.db_al1, result_.db_bl1,
          core_status_.batch, core_status_.m, core_status_.k, core_status_.n,
          result_.batch_dim, result_.m_dim, result_.k_dim, result_.n_dim,
          cur_cycle.cycle, best_cycle.cycle, cur_cycle.load_size, best_cycle.load_size,
          cur_cycle.mad_cycle, best_cycle.mad_cycle, cur_cycle.l0c_used, best_cycle.l0c_used,
          cur_cycle.load_2d_times, best_cycle.load_2d_times, cur_cycle.repeat_load_size, best_cycle.repeat_load_size,
          cur_cycle.k_l0, best_cycle.k_l0);
  if (cur_idx == 0) {
    SetBestCycle(cur_cycle, cur_idx);
    return;
  }
  bool update_cycle = cur_cycle.cycle < best_cycle.cycle;
  bool cycle_equal = best_cycle.cycle == cur_cycle.cycle;
  if (PlatformInfo::GetInstance().support_l0c2out() && run_params_.unaligned_flag && cycle_equal) {
    bool m_inner_axis_not_align = run_params_.trans_a_flag && run_params_.ori_shape_m % kBlockSize;
    bool m_dim_priority = result_.m_dim > result_.n_dim && result_.n_dim > 1;
    bool n_inner_axis_not_align = !run_params_.trans_b_flag && run_params_.ori_shape_n % kBlockSize;
    bool n_dim_priority = result_.n_dim > result_.m_dim && result_.m_dim > 1;
    if ((m_inner_axis_not_align && m_dim_priority) || (n_inner_axis_not_align && n_dim_priority)) {
      // if inner axis is m or n, except m or n to bind more core for avoid repeated loading.
      update_cycle = true;
    }
  }
  bool update_mad_cycle = cycle_equal && cur_cycle.mad_cycle < best_cycle.mad_cycle;
  bool mad_cycle_equal = cur_cycle.mad_cycle == best_cycle.mad_cycle;
  if (!PlatformInfo::GetInstance().support_l0c2out()) {
    update_mad_cycle = false;
    mad_cycle_equal = true;
  }
  bool update_load_size = cycle_equal && mad_cycle_equal && cur_cycle.load_size < best_cycle.load_size;
  bool update_l0c_use = cycle_equal && mad_cycle_equal && cur_cycle.load_size == best_cycle.load_size &&
      cur_cycle.l0c_used > best_cycle.l0c_used;
  bool update_load_2d_times = cycle_equal && mad_cycle_equal && cur_cycle.load_size == best_cycle.load_size &&
      cur_cycle.l0c_used == best_cycle.l0c_used && (cur_cycle.load_2d_times < 32 &&
      (best_cycle.load_2d_times >= 32 || (best_cycle.load_2d_times < 32 && best_cycle.k_l0 > cur_cycle.k_l0)));
  // update_small_repeat_load only enabled in 1971\float32, enabled after test in 1980 and float16
  bool update_small_repeat_load = (cycle_equal && mad_cycle_equal && cur_cycle.load_size == best_cycle.load_size &&
      cur_cycle.l0c_used == best_cycle.l0c_used &&
      (cur_cycle.repeat_load_size < best_cycle.repeat_load_size) &&
      PlatformInfo::GetInstance().support_l0c2out() &&
      (run_params_.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) || run_params_.bias_flag));
  if (update_cycle || update_mad_cycle || update_load_size || update_l0c_use || update_load_2d_times ||
      update_small_repeat_load) {
    SetBestCycle(cur_cycle, cur_idx);
  }
}

void GemmCycleEstimate::GetCycle(GemmCycleUsed &cur_cycle) {
  if (use_out_cycle_ >= 0) {
    cur_cycle.cycle = use_out_cycle_;
  } else {
    GemmCycleModelL0c2out cycle_model{this};
    cur_cycle.cycle = cycle_model.GetCycleModel();
  }
  cur_cycle.mad_cycle = result_.m_l0 * result_.k_l0 * result_.n_l0;
  cur_cycle.load_size = GetLoadSize();
  cur_cycle.l0c_used = result_.m_l0 * result_.n_l0 * result_.db_l0c;
  cur_cycle.load_2d_times = GetLoad2dRepeat();
  cur_cycle.repeat_load_size = result_.n_dim * core_status_.m + result_.m_dim * core_status_.n;
  cur_cycle.k_l0 = result_.k_l0;
}

int64_t GemmCycleEstimate::GetLoadSize() {
  EstimateLoadRepeat();
  return al1_full_load_size_ * get<0>(l1_load_repeat_) + bl1_full_load_size_ * get<1>(l1_load_repeat_);
}

int64_t GemmCycleEstimate::GetLoad2dARepeat() {
  if (result_.kal1_16 == result_.k_l0 || (!run_params_.trans_a_flag && result_.k_l0 == 1)) {
    return 1;
  }
  return result_.m_l0;
}

int64_t GemmCycleEstimate::GetLoad2dBRepeat() {
  if (n_bl1_ == 1 || (run_params_.trans_b_flag && result_.n_l0 == 1)) {
    return 1;
  }
  return result_.k_l0;
}

int64_t GemmCycleEstimate::GetLoad2dTimes(int64_t load_l0_repeat) {
  int64_t result = m_al1_ * n_bl1_;
  result = result * (core_status_.k / result_.k_l0) * load_l0_repeat;
  return result;
}

void GemmCycleEstimate::GetFullLoad2dRepeat(int64_t &load_2d_times, int64_t load_l0_repeat) {
  if (core_status_.both_full_load) {
    load_2d_times = load_l0_repeat;
  } else if (core_status_.IsKAnyFullLoad()) {
    load_2d_times = GetLoad2dTimes(load_l0_repeat);
  } else {
    load_2d_times = result_.GetLoadTimes(load_l0_repeat);
  }
}

void GemmCycleEstimate::GetKFullLoad2dRepeat(int64_t &load_2d_times, int64_t load_l0_repeat) {
  if (core_status_.IsKBothFullLoad()) {
    load_2d_times = GetLoad2dTimes(load_l0_repeat);
  } else {
    load_2d_times = result_.GetLoadTimes(load_l0_repeat);
  }
}

int64_t GemmCycleEstimate::GetLoad2dRepeat() {
  int64_t load_2d_a_repeat = GetLoad2dARepeat();
  int64_t load_2d_b_repeat = GetLoad2dBRepeat();
  int64_t load_l0_repeat = load_2d_a_repeat + load_2d_b_repeat;
  int64_t load_2d_times = 0;
  if (core_status_.IsAnyFullLoad()) {
    GetFullLoad2dRepeat(load_2d_times, load_l0_repeat);
  } else if (core_status_.IsKAnyFullLoad()) {
    GetKFullLoad2dRepeat(load_2d_times, load_l0_repeat);
  } else {
    load_2d_times = result_.GetLoadTimes(load_l0_repeat);
  }
  return load_2d_times;
}
}

