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
 * \file cache_tiling_cycle_model.cc\
 * \brief function of gemm cache cycle model
 */
#include <stdint.h>
#include "cache_tiling_cycle_model.h"
#include "../../mathutil.h"

namespace {
using namespace gemm_cache_tiling;
using namespace optiling;
// using optiling::cachetiling::MathUtil;

static constexpr int64_t kCacheLinePkgs[5] = {8, 4, 3, 2, 1};
static int64_t band_width[2][257]; // init 1~256 burstlen's real burstlen

static const int64_t kCopyLoadSize = 512;
static const int64_t kHeadCostNum = 100;
static const int64_t kNd2NzStartCost = 210;
static const int64_t kMinFractalSize = kBlockSize * kBlockSize;

namespace ub {
    // the value for 1980
static const int64_t kFullCacheLine = 8;
static const int64_t kVectorBandWidth = 512; // 512 Bytes per cycle
static const int64_t kMte1L0aBandWidth = 512;  // 512 Bytes per cycle
static const int64_t kMte1L0bBandWidth = 256;  // 256 Bytes per cycle
}

namespace l0c2out {
static const int64_t kHbmBandwidth = 64;
static const int64_t kMte1L0aBandWidth = 256;  // 256 Bytes per cycle
static const int64_t kMte1L0bBandWidth = 128;  // 128 Bytes per cycle
static const int64_t kMte1L0cBandWidth = 256;  // 256 Bytes per cycle
static const int64_t kMte1FixPipeBandWidth = 128;  // 256 Bytes per cycle
static const int64_t kMte1StartCost = 26;
}

int64_t GetBandwidthUsageNd2Nz(int64_t burst_length, int32_t dtype_bytes);

void InitBandWidth() {
  // init 1~256 burstlen's real burstlen
  for (int burst_length = 0; burst_length <= 256; ++burst_length) {
    // Dtype: 0: Float; 1: Float16
    band_width[0][burst_length] = GetBandwidthUsageNd2Nz(burst_length, kFp32Bytes);
    band_width[1][burst_length] = GetBandwidthUsageNd2Nz(burst_length, kFp16Bytes);
  }
}

__attribute__((constructor)) void InitThread() {
  InitBandWidth();
}

int64_t GetBandwidthUsageNd2Nz(int64_t burst_length, int32_t dtype_bytes) {
  // 512 is base pkg size means 512B
  const int64_t base_pkg_size = 512;
  uint64_t burstLengthByte = static_cast<uint64_t>(burst_length * dtype_bytes * kBlockSize);
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

  return cache_line_pkg_num * base_pkg_size / l0c2out::kHbmBandwidth;
}

int64_t GetBandwidthPreCalcuted(int64_t burst_length, int32_t dtype) {
  // get 1~256 burstlen's real burstlen
  if (burst_length <= 256 && (dtype != static_cast<int32_t>(ge::DT_INT8))) {
    return band_width[dtype][burst_length];
  } else {
    int32_t dtype_bytes = kFp16Bytes;
    if (dtype == static_cast<int32_t>(ge::DT_FLOAT)) {
      dtype_bytes = kFp32Bytes;
    } else if (dtype == static_cast<int32_t>(ge::DT_INT8)) {
      dtype_bytes = 1;
    }
    return GetBandwidthUsageNd2Nz(burst_length, dtype_bytes);
  }
}

}
namespace gemm_cache_tiling {
using namespace optiling;

GemmCycleModel::GemmCycleModel(const optiling::BatchmatmulRunParas &run_params, const optiling::CoreStatus &coreStatus,
    const optiling::SingleCoreStatus &singleCoreStatus) :
  run_params_(run_params),
  core_status_(coreStatus, singleCoreStatus),
  result_(coreStatus, singleCoreStatus),
  m_al1_(singleCoreStatus.l1Status.m_al1),
  n_bl1_(singleCoreStatus.l1Status.n_bl1) {
    mad_expansion_rate_ = 1;
    if (run_params_.dtype_a == ge::DataType::DT_FLOAT) {
      mad_expansion_rate_ = run_params_.hf32_flag ? 1 : 2; // FP32 is 2 times FP16 cube busy cycle
      dtype_size_ = kFp32Bytes;
      cube_k_ = reducekBlockSize;
    } else {
      dtype_size_ = kFp16Bytes;
      cube_k_ = kBlockSize;
    }
    out_dtype_size_ = run_params_.dtype_out == ge::DataType::DT_FLOAT ? kFp32Bytes : kFp16Bytes;
};

GemmCycleModel::GemmCycleModel(const GemmCycleEstimate* const est) :
    run_params_(est->run_params_),
    core_status_(est->core_status_),
    result_(est->result_),
    m_al1_(est->m_al1_),
    n_bl1_(est->n_bl1_),
    mad_expansion_rate_(est->mad_expansion_rate_),
    cube_k_(est->cube_k_),
    dtype_size_ (est->dtype_size_),
    out_dtype_size_(est->out_dtype_size_){};

int64_t GemmCycleModel::GetMadCycle() {
  return result_.m_l0 * result_.k_l0 * result_.n_l0 * mad_expansion_rate_;
}

int64_t GemmCycleModel::GetMte1Cycle(int64_t n_burst, int64_t burst_length, int64_t bandwidth) {
  // bandwidth will not be Zero
  return n_burst * burst_length * kBlockSize * cube_k_ * dtype_size_ / bandwidth;
}

void GemmCycleModel::GetCycleFinalCycle() {
  int64_t mte1_cycle = result_.batch_l0 * max(mte1_al0_cycle_, mte1_bl0_cycle_);
  mad_cycle_ = max(GetMadCycle(), mte1_cycle);
  final_cycle_ = 0;
  // al1 full load && bl1 full load
  // Consider multi batch in BatchMatMul
  BothFullLoadMode();
  // al1 full load && (bl1 k full load || bl1 not full load)
  AL1FullLoadMode();
  // bl1 full load && (al1 k full load || al1 not full load)
  BL1FullLoadMode();
  // al1 k full load && bl1 k full load
  BothKFullLoadMode();
  // al1 k full load && bl1 not full load || al1 not full load && bl1 k full load
  SingleKFullLoadMode();
  // al1 not full load && bl1 not full load
  NotFullLoadMode();
}

void GemmCycleModel::BothFullLoadMode() {
  if (core_status_.both_full_load) {
    // total_cycle = MTE2_AUB + MTE2_BUB + m_outer * n_outer * k_outer * MAD
    int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0);
    int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0);
    int64_t k_outer = MathUtil::CeilDivision(core_status_.k, result_.k_l0);
    final_cycle_ =
        mte2_a_cycle_ + mte2_b_cycle_ + m_outer * n_outer * k_outer * mad_cycle_;
  }
}

void GemmCycleModel::AL1FullLoadMode() {
  if (final_cycle_ != 0 || (!core_status_.al1_full_load)) {
    return;
  }
  if (core_status_.bl1_k_full_load) {
    // total_cycle = MTE2_AUB + n_outer * (MTE2_BUB + m_outer * n_bl1 * k_outer * MAD)
    int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0);
    int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0 * n_bl1_);
    int64_t k_outer = MathUtil::CeilDivision(core_status_.k, result_.k_l0);
    final_cycle_ = mte2_a_cycle_ +
                  n_outer * (mte2_b_cycle_ + m_outer * n_bl1_ * k_outer * mad_cycle_);
  } else {
    // total_cycle = MTE2_AUB + m_outer * n_outer * kl1_times * max(MTE2_BUB, k_outer * MAD)
    int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0);
    int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0);
    int64_t kl1_times = MathUtil::CeilDivision(core_status_.k, result_.kbl1_16);
    int64_t k_outer = MathUtil::CeilDivision(result_.kbl1_16, result_.k_l0);
    int64_t inner_cycle = 0;
    if (result_.db_bl1 != kDbOn) {
      inner_cycle = mte2_b_cycle_ + k_outer * mad_cycle_;
    } else {
      inner_cycle = max(mte2_b_cycle_, k_outer * mad_cycle_);
    }
    final_cycle_ = mte2_a_cycle_ + m_outer * n_outer * kl1_times * inner_cycle;
  }
}

void GemmCycleModel::BL1FullLoadMode() {
  if (final_cycle_ != 0 || (!core_status_.bl1_full_load)) {
    return;
  }
  if (core_status_.al1_k_full_load) {
    // total_cycle = MTE2_BUB + m_outer * (MTE2_AUB * n_outer + n_outer * m_al1 * k_outer * MAD)
    int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0 * m_al1_);
    int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0);
    int64_t k_outer = MathUtil::CeilDivision(core_status_.k, result_.k_l0);
    final_cycle_ = mte2_b_cycle_ + m_outer * (mte2_a_cycle_ * n_outer +
                                                        n_outer * m_al1_ * k_outer * mad_cycle_);
  } else {
    // total_cycle = MTE2_BUB + m_outer * n_outer * (kl1_times * max(MTE2_AUB, k_outer * MAD))
    int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0);
    int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0);
    int64_t kl1_times = MathUtil::CeilDivision(core_status_.k, result_.kal1_16);
    int64_t k_outer = MathUtil::CeilDivision(result_.kal1_16, result_.k_l0);
    int64_t inner_cycle = 0;
    if (result_.db_al1 != kDbOn) {
      inner_cycle = mte2_a_cycle_ + k_outer * mad_cycle_;
    } else {
      inner_cycle = max(mte2_a_cycle_, k_outer * mad_cycle_);
    }
    final_cycle_ = mte2_b_cycle_ + m_outer * n_outer * kl1_times * inner_cycle;
  }
}

void GemmCycleModel::BothKFullLoadMode() {
  if (final_cycle_ != 0) {
    return;
  }
  if (core_status_.al1_k_full_load && core_status_.bl1_k_full_load) {
    // total_cycle = m_outer * (MTE2_AUB + n_outer * m_al1 * (MTE2_BUB + n_bl1 * k_outer * MAD))
    int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0 * m_al1_);
    int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0 * n_bl1_);
    int64_t k_outer = MathUtil::CeilDivision(core_status_.k, result_.k_l0);
    final_cycle_ = m_outer * (mte2_a_cycle_ +
                             n_outer * m_al1_ *
                                 (mte2_b_cycle_ + n_bl1_ * k_outer * mad_cycle_));
  }
}

void GemmCycleModel::SingleKFullLoadMode() {
  if (final_cycle_ != 0) {
    return;
  }
  if (core_status_.al1_k_full_load) {
    // total_cycle = m_outer * (MTE2_AUB + n_outer * m_al1 * kl1_times * max(MTE2_BUB, k_outer * MAD))
    int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0 * m_al1_);
    int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0);
    int64_t kl1_times = MathUtil::CeilDivision(core_status_.k, result_.kbl1_16);
    int64_t k_outer = MathUtil::CeilDivision(result_.kbl1_16, result_.k_l0);
    final_cycle_ =
        m_outer * (mte2_a_cycle_ + n_outer * m_al1_ * kl1_times *
                                                  max(mte2_b_cycle_, k_outer * mad_cycle_));
  }
  if (core_status_.bl1_k_full_load) {
    // total_cycle = n_outer * (MTE2_BUB + m_outer * n_bl1 * (kl1_times * max(MTE2_AUB, k_outer * MAD)))
    int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0);
    int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0 * n_bl1_);
    int64_t kl1_times = MathUtil::CeilDivision(core_status_.k, result_.kal1_16);
    int64_t k_outer = MathUtil::CeilDivision(result_.kal1_16, result_.k_l0);
    final_cycle_ =
        n_outer * (mte2_b_cycle_ + m_outer * n_bl1_ * kl1_times *
                                                  max(mte2_a_cycle_, k_outer * mad_cycle_));
  }
}

void GemmCycleModel::NotFullLoadMode() {
  if (final_cycle_ != 0) {
    return;
  }
  // total_cycle = m_outer * n_outer * MAX_FUNC,
  // if k_al1 == k_bl1: MAX_FUNC = k_outer_outer * max(MTE2_AUB + MTE2_BUB, k_outer * MAD)
  // if k_al1 > k_bl1: MAX_FUNC = k_outer_outer * (MTE2_AUB + kl1_times * max(MTE2_BUB, k_outer * MAD))
  // if k_al1 < k_bl1: MAX_FUNC = k_outer_outer * (MTE2_BUB + kl1_times * max(MTE2_AUB, k_outer * MAD))
  int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0);
  int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0);
  int64_t inner_cycle = 0;
  if (result_.kal1_16 == result_.kbl1_16) {
    int64_t k_outer_outer = MathUtil::CeilDivision(core_status_.k, result_.kal1_16);
    int64_t k_outer = MathUtil::CeilDivision(result_.kal1_16, result_.k_l0);
    inner_cycle =
        k_outer_outer * max(mte2_a_cycle_ + mte2_b_cycle_, k_outer * mad_cycle_);
  } else if (result_.kal1_16 > result_.kbl1_16) {
    int64_t k_outer_outer = MathUtil::CeilDivision(core_status_.k, result_.kal1_16);
    int64_t kl1_times = MathUtil::CeilDivision(result_.kal1_16, result_.kbl1_16);
    int64_t k_outer = MathUtil::CeilDivision(result_.kbl1_16, result_.k_l0);
    inner_cycle = k_outer_outer * (mte2_a_cycle_ +
                                   kl1_times * max(mte2_b_cycle_, k_outer * mad_cycle_));
  } else {
    int64_t k_outer_outer = MathUtil::CeilDivision(core_status_.k, result_.kbl1_16);
    int64_t kl1_times = MathUtil::CeilDivision(result_.kbl1_16, result_.kal1_16);
    int64_t k_outer = MathUtil::CeilDivision(result_.kal1_16, result_.k_l0);
    inner_cycle = k_outer_outer * (mte2_b_cycle_ +
                                   kl1_times * max(mte2_a_cycle_, k_outer * mad_cycle_));
  }
  final_cycle_ = m_outer * n_outer * inner_cycle;
}

int64_t GemmCycleModelL0c2out::GetCycleModel() {
  mte2_a_cycle_ = GetMte2Al1Cycle();
  mte2_b_cycle_ = GetMte2Bl1Cycle();
  // Fixpipe cycle is calculated below
  mte1_al0_cycle_ = GetMte1Cycle(result_.m_l0, result_.k_l0, l0c2out::kMte1L0aBandWidth) + l0c2out::kMte1StartCost;
  mte1_bl0_cycle_ = GetMte1Cycle(result_.k_l0, result_.n_l0, l0c2out::kMte1L0bBandWidth) + l0c2out::kMte1StartCost;
  GetCycleFinalCycle();
  final_cycle_ += GetFixedPipeCycle();
  return (run_params_.batch / result_.batch_dim / result_.batch_l0) * final_cycle_;
}

int64_t GemmCycleModelL0c2out::GetMte2Al1Cycle() {
  int64_t n_burst = (run_params_.trans_a_flag ? result_.kal1_16 : m_al1_ * result_.m_l0) * kBlockSize;
  int64_t burst_length = run_params_.trans_a_flag ? m_al1_ * result_.m_l0 : result_.kal1_16;
  int64_t ori_inner_axis = !run_params_.trans_a_flag ? run_params_.k : run_params_.m;
  if (!run_params_.format_a_nd) {
    n_burst = run_params_.trans_a_flag ? m_al1_ * result_.m_l0 : result_.kal1_16;
    burst_length = run_params_.trans_a_flag ? result_.kal1_16 * kBlockSize : m_al1_ * result_.m_l0 * kBlockSize;
  }
  // the 64B~256B's burst_length and not multi core, burst_length can merge with n_burst
  // the number 8 is from 256B / 32B
  int64_t can_merge_max_burst_length = 8;
  // the number 2 is from 64B / 32B
  int64_t can_merge_min_burst_length = 2;
  if (run_params_.dtype_a == static_cast<int32_t>(ge::DT_FLOAT)) {
    can_merge_max_burst_length = can_merge_max_burst_length >> 1;
    can_merge_min_burst_length = can_merge_min_burst_length >> 1;
  }
  if ((burst_length >= can_merge_min_burst_length) && (burst_length <= can_merge_max_burst_length) &&
      (ori_inner_axis == burst_length)) {
    burst_length *= n_burst;
    n_burst = 1;
  }
  auto dtype_a = run_params_.dtype_a;
  dtype_a = (dtype_a == static_cast<int32_t>(ge::DT_BF16)) ? static_cast<int32_t>(ge::DT_FLOAT16) : dtype_a;
  int64_t new_burst_length = GetBandwidthPreCalcuted(burst_length, dtype_a);

  return result_.batch_l0 * n_burst * new_burst_length + kNd2NzStartCost;
}

int64_t GemmCycleModelL0c2out::GetMte2Bl1Cycle() {
  int64_t n_burst = (run_params_.trans_b_flag ? n_bl1_ * result_.n_l0 : result_.kbl1_16) * kBlockSize;
  int64_t burst_length = run_params_.trans_b_flag ? result_.kbl1_16 : n_bl1_ * result_.n_l0;
  int64_t ori_inner_axis = run_params_.trans_b_flag ? run_params_.k : run_params_.n;
  if (!run_params_.format_b_nd) {
    n_burst = run_params_.trans_b_flag ? result_.kbl1_16 : n_bl1_ * result_.n_l0;
    burst_length = run_params_.trans_b_flag ? n_bl1_ * result_.n_l0 * kBlockSize : result_.kbl1_16 * kBlockSize;
  }
  // the 64B~256B's burst_length and not multi core, burst_length can merge with n_burst
  // the number 8 is from 256B / 32B
  int64_t can_merge_max_burst_length = 8;
  // the number 2 is from 64B / 32B
  int64_t can_merge_min_burst_length = 2;
  if (run_params_.dtype_b == static_cast<int32_t>(ge::DT_FLOAT)) {
    can_merge_max_burst_length = can_merge_max_burst_length >> 1;
    can_merge_min_burst_length = can_merge_min_burst_length >> 1;
  }
  if ((burst_length >= can_merge_min_burst_length) && (burst_length <= can_merge_max_burst_length) &&
      (ori_inner_axis == burst_length)) {
    burst_length *= n_burst;
    n_burst = 1;
  }
  auto dtype_a = run_params_.dtype_a;
  dtype_a = (dtype_a == static_cast<int32_t>(ge::DT_BF16)) ? static_cast<int32_t>(ge::DT_FLOAT16) : dtype_a;
  int64_t new_burst_length = GetBandwidthPreCalcuted(burst_length, dtype_a);

  return result_.batch_l0 * n_burst * new_burst_length + kNd2NzStartCost;
}

int64_t GemmCycleModelL0c2out::GetFixedPipeCycle() {
  // FixPipe not consider MTE2 FixePipe overlapping
  int64_t cycle_l0c_2_fixpipe = result_.m_l0 * result_.n_l0 * kMinFractalSize * kFp32Bytes / l0c2out::kMte1L0cBandWidth;
  int64_t cycle_fixpipe_2_out = (
    result_.m_l0 * result_.n_l0 * kMinFractalSize * out_dtype_size_ / l0c2out::kMte1FixPipeBandWidth);
  int64_t m_outer = MathUtil::CeilDivision(core_status_.m, result_.m_l0);
  int64_t n_outer = MathUtil::CeilDivision(core_status_.n, result_.n_l0);
  int64_t fixpipe_cycle = ((run_params_.is_batch_matmul_op) ? result_.batch_l0 * m_outer * n_outer *
                           (cycle_l0c_2_fixpipe + cycle_fixpipe_2_out) : 0);
  return fixpipe_cycle;
}

int64_t GemmCycleModelUB::GetCycleModel() {
  int64_t n_ub_l0_time = MathUtil::CeilDivision(result_.n_l0, ubStatus.n_cub);
  if (run_params_.is_compress_quant) {
    mte2_a_cycle_ = GetMte2Al1Cycle();
    mte2_b_cycle_ = GetMte2Bl1Cycle();
  } else {
    mte2_a_cycle_ = GetMte2AubCycle();
    mte2_b_cycle_ = GetMte2BubCycle();
  }
  mte3_cycle = GetMte3CubCycle() * n_ub_l0_time;
  // When k_l0 = 1 and much smaller than k_l1, vector bank conflict is easy to happen.
  // Need to change the vector bandwidth to 80 bytes/cycle when bank conflict occur.
  int64_t vector_bandwidth = (result_.k_l0 == 1 && max(result_.kal1_16, result_.kbl1_16) >= ub::kFullCacheLine)
                                ? 80
                                : ub::kVectorBandWidth;
  // output vector process contains copy_cc_to_ubuf and vmuls, so need to multiply 2.
  int64_t output_vector_cycle =
      ((result_.m_l0 * GetBandwidthUsage(result_.n_l0) * kMinFractalSize * kFp16Bytes) >> 1) / vector_bandwidth;
  mte1_al0_cycle_ = GetMte1Cycle(result_.m_l0, result_.k_l0, ub::kMte1L0aBandWidth);
  mte1_bl0_cycle_ = GetMte1Cycle(result_.k_l0, result_.n_l0, ub::kMte1L0bBandWidth);
  GetCycleFinalCycle();
  int64_t l0c_to_ub_cycle = 0;
  bool is_l0c_preload = result_.db_l0c == kDbOn && ((core_status_.al1_full_load && !core_status_.bl1_full_load) ||
                                                     (!core_status_.al1_full_load && core_status_.bl1_full_load));
  if (is_l0c_preload) {
    // When l0c preload, mte3 can be coverd by mad.
    int64_t load_2d_a_repeat = GetLoad2dARepeat();
    int64_t load_2d_b_repeat = GetLoad2dBRepeat();
    int64_t mte3_cover_cycle =
        min((32 / (load_2d_a_repeat + load_2d_b_repeat + 1)), MathUtil::CeilDivision(core_status_.k, result_.k_l0)) *
        mad_cycle_;
    l0c_to_ub_cycle = max(output_vector_cycle + mte3_cycle - mte3_cover_cycle, 0L);
  } else {
    l0c_to_ub_cycle = output_vector_cycle + mte3_cycle;
  }
  final_cycle_ += MathUtil::CeilDivision(core_status_.m, result_.m_l0) *
                MathUtil::CeilDivision(core_status_.n, result_.n_l0) * l0c_to_ub_cycle;
  return (run_params_.batch / result_.batch_dim / result_.batch_l0) * final_cycle_;
}

int64_t GemmCycleModelUB::GetMte2Al1Cycle() {
  int64_t n_burst = run_params_.trans_a_flag ? m_al1_ * result_.m_l0 : result_.kal1_16;
  int64_t burst_length =
      run_params_.trans_a_flag ? result_.kal1_16 * kBlockSize : m_al1_ * result_.m_l0 * kBlockSize;
  int64_t new_burst_length = GetBandwidthUsage(burst_length);
  int64_t repeat_load_cycle = n_burst * burst_length < kCopyLoadSize ? kHeadCostNum : 0;
  return n_burst * new_burst_length + repeat_load_cycle;
}

int64_t GemmCycleModelUB::GetMte2Bl1Cycle() {
  int64_t n_burst = MathUtil::CeilDivision(result_.kbl1_16, result_.k_l0) *
                    MathUtil::CeilDivision(n_bl1_, result_.n_l0) * result_.k_l0;
  int64_t burst_length = result_.n_l0 * kBlockSize * kBlockSize;
  int64_t new_burst_length = GetBandwidthUsage(burst_length);
  int64_t repeat_load_cycle = n_burst * burst_length < kCopyLoadSize ? kHeadCostNum : 0;
  return n_burst * new_burst_length + repeat_load_cycle;
}

int64_t GemmCycleModelUB::GetMte2AubCycle() {
  // Calculate MTE2 cost for AUB
  int64_t multi_k_aub_l1 = MathUtil::CeilDivision(result_.kal1_16, ubStatus.k_aub);
  int64_t multi_m_ub_l1 = MathUtil::CeilDivision(m_al1_ * result_.m_l0, ubStatus.m_aub);
  int64_t n_burst = (run_params_.trans_a_flag ? ubStatus.k_aub : ubStatus.m_aub) * kBlockSize;
  int64_t burst_length = run_params_.trans_a_flag ? ubStatus.m_aub : ubStatus.k_aub;
  // Update burst_len when the addresses are consecutive
  if ((!run_params_.trans_a_flag && ubStatus.k_aub == run_params_.k) ||
      (run_params_.trans_a_flag && ubStatus.m_aub == run_params_.m)) {
    burst_length *= n_burst;
    n_burst = 1;
  }
  int64_t new_burst_length = GetBandwidthUsage(burst_length);
  // if copy_load_size < 16kB, has head_cost 100 cycle for each repeat
  int64_t repeat_load_cycle = n_burst * burst_length < kCopyLoadSize ? kHeadCostNum : 0;
  return multi_k_aub_l1 * multi_m_ub_l1 * (n_burst * new_burst_length + repeat_load_cycle);
}

int64_t GemmCycleModelUB::GetMte2BubCycle() {
  // Calculate MTE2 cost for BUB
  int64_t multi_k_bub_l1 = MathUtil::CeilDivision(result_.kbl1_16, ubStatus.k_bub);
  int64_t multi_n_ub_l1 = MathUtil::CeilDivision(n_bl1_ * result_.n_l0, ubStatus.n_bub);
  int64_t n_burst = (run_params_.trans_b_flag ? ubStatus.n_bub : ubStatus.k_bub) * kBlockSize;
  int64_t burst_length = run_params_.trans_b_flag ? ubStatus.k_bub : ubStatus.n_bub;
  // Update burst_len when the addresses are consecutive
  if ((!run_params_.trans_b_flag && ubStatus.n_bub == run_params_.n) ||
      (run_params_.trans_b_flag && ubStatus.k_bub == run_params_.k)) {
    burst_length *= n_burst;
    n_burst = 1;
  }
  int64_t new_burst_length = GetBandwidthUsage(burst_length);
  // if copy_load_size < 16kB, has head_cost 100 cycle for each repeat
  int64_t repeat_load_cycle = n_burst * burst_length < kCopyLoadSize ? kHeadCostNum : 0;
  return multi_k_bub_l1 * multi_n_ub_l1 * (n_burst * new_burst_length + repeat_load_cycle);
}

int64_t GemmCycleModelUB::GetMte3CubCycle() {
  // Calculate MTE3 cost for CUB
  int64_t n_burst = 0;
  int64_t burst_length = 0;
  int64_t new_burst_length = 0;
  n_burst = 1;
  burst_length = result_.m_l0 * ubStatus.n_cub * kBlockSize;
  new_burst_length = GetBandwidthUsage(burst_length);
  // if copy_load_size < 16kB, has head_cost 100 cycle for each repeat
  int64_t repeat_load_cycle = n_burst * burst_length < kCopyLoadSize ? kHeadCostNum : 0;
  return n_burst * new_burst_length + repeat_load_cycle;
}

int64_t GemmCycleModelUB::GetLoad2dARepeat() {
  if (result_.kal1_16 == result_.k_l0 || (!run_params_.trans_a_flag && result_.k_l0 == 1)) {
    return 1;
  }
  return result_.m_l0;
}

int64_t GemmCycleModelUB::GetLoad2dBRepeat() {
  if (n_bl1_ == 1 || (run_params_.trans_b_flag && result_.n_l0 == 1)) {
    return 1;
  }
  return result_.k_l0;
}

int64_t GemmCycleModelUB::GetBandwidthUsage(int64_t burst_length) {
  int64_t res = 0;
  for (int64_t cache_line : kCacheLinePkgs) {
    res += burst_length / cache_line;
    burst_length = burst_length % cache_line;
  }
  return res * ub::kFullCacheLine;
}

int64_t GetCycleByModel(const optiling::BatchmatmulRunParas &run_params, const optiling::CoreStatus &coreStatus,
    const optiling::SingleCoreStatus &singleCoreStatus) {
  if (optiling::PlatformInfo::GetInstance().support_l0c2out()) {
    GemmCycleModelL0c2out cycle_model{run_params, coreStatus, singleCoreStatus};
    return cycle_model.GetCycleModel();
  }
  GemmCycleModelUB cycle_model{run_params, coreStatus, singleCoreStatus};
  return cycle_model.GetCycleModel();
};
}

