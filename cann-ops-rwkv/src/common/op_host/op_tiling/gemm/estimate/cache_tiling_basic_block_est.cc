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
 * \file cache_tiling_basic_block_est.cc
 * \brief function of cache tiling basic block estmate
 */
#include <float.h>
#include "gemm/estimate/cache_tiling_basic_block_est.h"
#include "gemm/estimate/cache_tiling_est_mgr.h"
#include "cube/algorithm/hash/tiling_cache.h"
#include "gemm/common/cache_tiling_align_count.h"
#include "gemm/common/cache_tiling_request_bytes.h"
#include "../../mathutil.h"

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

using namespace std;
using namespace optiling;
// using optiling::cachetiling::MathUtil;
using namespace gemm_cache_tiling;

namespace {
constexpr int32_t L2_SIZE_2 = 192 * 1024 * 1024;
constexpr int32_t BIG_BLOCK = 512;
constexpr int32_t D_LINE = 256;
constexpr int32_t MATA_NUM = 8;
constexpr double AIC_FREQ_2 = 1.8;
constexpr double AIC_FREQ_4 = 1.5;
constexpr int32_t MATA_FREQ = 2; // mata

inline int64_t FindLargestFactor(int64_t x, int64_t y, int64_t default_value = 1) {
  if (y == 0) {
    return default_value;
  }
  for (int64_t tmp = y; tmp > 1; tmp--) {
    if (x % tmp == 0) {
      return tmp;
    }
  }
  return default_value;
}

inline double GetAicFreq() {
  if (PlatformInfo::GetInstance().l2_size == L2_SIZE_2) {
    return AIC_FREQ_2;
  }
  return AIC_FREQ_4;
}

auto MIN_FUNC = [](initializer_list<double> il) {
  return *(set<double>(il).begin());
};

auto MAX_FUNC = [](initializer_list<double> il) {
  return *(set<double>(il).rbegin());
};
}

namespace gemm_cache_tiling {

REG_GEMM_ESTIMATE_FUNC(BASIC_BLOCK_ESTIMATE_TYPE, GemmBBEstimate);

GemmBBEstimate::GemmBBEstimate(const string& op_type, const optiling::BatchmatmulParas *paras) :
  GemmEstimate(op_type, paras) {
    best_perf_ = DBL_MAX;
};

void GemmBBEstimate::SetBufferParams() {
  GemmEstimate::SetBufferParams();
  ResetParams();
  return;
}

void GemmBBEstimate::SetBufferParams(const CoreStatus &core_status, const SingleCoreStatus &singlecore_status) {
  GemmEstimate::SetBufferParams(core_status, singlecore_status);
  ResetParams();
  return;
}

void GemmBBEstimate::ResetParams() {
  memset_s(&cycle_info_, sizeof(cycle_info_), 0, sizeof(cycle_info_));

  m_patch_ = MathUtil::CeilDivision(run_params_.ori_shape_m, result_.m_dim);
  n_patch_ = MathUtil::CeilDivision(run_params_.ori_shape_n, result_.n_dim);
  k_patch_ = MathUtil::CeilDivision(run_params_.ori_shape_k, result_.k_dim);
  m_ = MathUtil::Align(m_patch_, kBlockSize);
  k_ = MathUtil::Align(k_patch_, cube_k_);
  n_ = MathUtil::Align(n_patch_, kBlockSize);

  batch_patch_ = MathUtil::CeilDivision(run_params_.batch, result_.batch_dim);
  t_l1_M_ = result_.m_l1 * kBlockSize;
  t_l1_AK_ = result_.kal1_16 * cube_k_;
  t_l1_BK_ = result_.kbl1_16 * cube_k_;
  t_l1_N_ = result_.n_l1 * kBlockSize;

  t_l0_M_ = result_.m_l0 * kBlockSize;
  t_l0_K_ = result_.k_l0 * cube_k_;
  t_l0_N_ = result_.n_l0 * kBlockSize;
  active_core_num_ = min(result_.n_dim * result_.m_dim * result_.k_dim, PlatformInfo::GetInstance().core_num);
  m_run_ = min(result_.m_dim, active_core_num_);
  n_run_ = min(result_.n_dim, active_core_num_ / m_run_);
  k_run_ = min(result_.k_dim, active_core_num_ / (m_run_ * n_run_));
  if (run_params_.is_batch_matmul_op) {
    batch_run_ = min(result_.batch_dim, active_core_num_ / (m_run_ * n_run_ * k_run_));
    int64_t batch_pc_t = MathUtil::CeilDivision(run_params_.batch, result_.batch_dim);
    int64_t max_batch = 1;
    if (t_l0_M_ >= m_patch_ && t_l0_K_ >= k_patch_ && t_l0_N_ >= n_patch_) {
      max_batch = min(PlatformInfo::GetInstance().l0a_size / (DB_SIZE * t_l0_M_ * t_l0_K_ * dtype_size_),
                      min(PlatformInfo::GetInstance().l0b_size / (DB_SIZE * t_l0_N_ * t_l0_K_ * dtype_size_),
                          PlatformInfo::GetInstance().l0c_size / (DB_SIZE * t_l0_N_ * t_l0_M_ * 4))); // 4:l0c data size
    }
    max_batch = FindLargestFactor(batch_pc_t, max_batch);
    batch_times_ = MathUtil::CeilDivision(batch_pc_t, max_batch);
    batch_patch_ = max_batch;
  }
  k_shift_ = EnableKShuffle();
  // shift_inwards_ need to align 32B
  shift_inwards_ = m_patch_ * dtype_size_ % 32 == 0 && k_patch_ * dtype_size_ % 32 == 0
      && n_patch_ * dtype_size_ % 32 == 0;
}

void GemmBBEstimate::Estimate(int32_t cur_idx) {
  EstimateLoadRepeat();
  EstimateBandwidth();
  EstimateCycle();
  double perf = batch_times_ * EstimatePipe();
  int64_t kernel_start = MathUtil::CeilDivision(result_.n_dim * result_.m_dim * result_.k_dim,
                                                PlatformInfo::GetInstance().core_num);
  double cur_perf = perf * kernel_start;
  // 1800: 1.8 fps(cycle->ns) * 1000(ns->us)
  OP_LOGD(op_type_.c_str(), "gemm gen tiling bb estimate: cur_idx:%d, perf:%lf", cur_idx, perf/1800);
  if (cur_perf < best_perf_) {
    best_perf_ = cur_perf;
    best_idx_ = cur_idx;
  }
}

template<GEMM_CUBE_TAG T>
void GemmBBEstimate::GetDMABiuBw(double &biu_bw, GEMM_CUBE_SIZE cube_size, int32_t latency) {
  int64_t dma_size = 1;
  switch (T) {
    case GEMM_CUBE_TAG::A:
      dma_size = n_run_ * cube_size.n * cube_size.d * cube_size.dtype_size;
      break;
    case GEMM_CUBE_TAG::B:
      dma_size = m_run_ * cube_size.n * cube_size.d * cube_size.dtype_size;
      break;
    case GEMM_CUBE_TAG::C:
      dma_size = min(result_.k_dim, active_core_num_) * cube_size.n * cube_size.d * cube_size.dtype_size;
      break;
    default:
      break;
  }
  double dma_effi = (dma_size / biu_bw) / (dma_size / biu_bw + latency / 2.0);
  if (cube_size.d != cube_size.srcD || cube_size.d * cube_size.dtype_size != 32) { // 32B: no dma
    dma_effi *= 0.75; // ND bawdwidth ratio 1.2T / NZ bandwidtch 1.6T = 0.75
  }
  biu_bw *= dma_effi;
}

bool GemmBBEstimate::EnableKShuffle() {
  bool k_shift_front = (m_patch_ != 1 || n_patch_ != 1) && result_.k_dim == 1;
  int64_t thread = result_.m_dim * result_.n_dim;
  int64_t k_ext = MathUtil::CeilDivision(run_params_.ori_shape_k, t_l1_AK_);
  int64_t block_ext = thread > 24 ? 8 : max(m_run_, n_run_);
  bool k_shift_back = block_ext > 2 && k_ext > block_ext / 2;
  return k_shift_front && k_shift_back;
}

template<GEMM_CUBE_TAG T>
double GemmBBEstimate::GetSameAddressPenalty(GEMM_CUBE_SIZE cube_size, int32_t same_count, int64_t req_size) {
  int64_t core_num = 1;
  switch (T) {
    case GEMM_CUBE_TAG::A:
      core_num = min(active_core_num_, n_run_);
      if (k_shift_) {
        core_num = n_run_ > m_run_ ? 1 : n_run_;
      }
      break;
    case GEMM_CUBE_TAG::B:
      core_num = min(active_core_num_, m_run_);
      if (k_shift_) {
        core_num = m_run_ >= n_run_ ? 1 : m_run_;
      }
      break;
    case GEMM_CUBE_TAG::C:
      break;
    default:
      break;
  }
  double tile_size = 1.0 * core_num * cube_size.n * max(cube_size.d * dtype_size_, 128L) / 1000000;
  double mata_effi = 1 - 0.91 * exp(-0.053 * tile_size); // 0.91 0.053 same address effi
  double same_address_cycle = core_num * (same_count - 1) * req_size / mata_effi * 0.03; // 0.91 0.053 same address coef
  return same_address_cycle;
}

template<GEMM_CUBE_TAG T>
double GemmBBEstimate::MataBandwidth(int64_t sum, int64_t req_size, int32_t same_count, GEMM_CUBE_SIZE cube_size) {
  double cycle = 1.0 * req_size * active_core_num_ / MATA_NUM * MATA_FREQ / GetAicFreq() +
     GetSameAddressPenalty<T>(cube_size, same_count, req_size);
  return 1.0 * sum / cycle;
}

double GemmBBEstimate::MataDDRBandwidth(int64_t req_sum, int64_t align_sum) {
  constexpr int32_t ailin = 128;
  double mata_effi = 1.0 * req_sum / align_sum;
  double mata_phys = 1.0 * ailin * MATA_NUM * MATA_FREQ / active_core_num_ / GetAicFreq();
  return mata_phys * mata_effi;
}


void GemmBBEstimate::EstimateBandwidthC() {
  GEMM_CUBE_SIZE cube_size {
      min(m_patch_, t_l1_M_), min(n_patch_, t_l1_N_), run_params_.ori_shape_n, out_dtype_size_};
  if (shift_inwards_) {
    cube_size = {t_l1_M_, t_l1_N_, run_params_.ori_shape_n, out_dtype_size_};
  }
  AlignCount request_C = GetRequestNZ2ND(cube_size);
  int32_t same_addr_C = GetSameAddressNZ2ND(cube_size);
  int64_t request_C_sum =  cube_size.n * cube_size.d * out_dtype_size_;
  double biu_bw_C = 1.0 * request_C_sum * soc_bw_.WRITE_OTSd / (soc_bw_.L2_LATENCY_WD * request_C.Size());
  GetDMABiuBw<GEMM_CUBE_TAG::C>(biu_bw_C, cube_size, soc_bw_.L2_LATENCY_WD);
  double mata_bw_C = MataBandwidth<GEMM_CUBE_TAG::C>(request_C_sum, request_C.Size(), same_addr_C, cube_size);
  double l0C_l2 = MIN_FUNC({soc_bw_.L0C_RO, soc_bw_.L2_WO / active_core_num_, biu_bw_C, soc_bw_.L2_PHY_RO, mata_bw_C});
  bw_info_.l0C_l2 = (result_.k_dim > 1) ? l0C_l2 * 0.4 : l0C_l2; // 0.4 Simulated loss
}

void GemmBBEstimate::EstimateBandwidthB() {
  int64_t shift_inwards_K = shift_inwards_ ? t_l1_BK_ : min(k_patch_, t_l1_BK_);
  int64_t shift_inwards_N = shift_inwards_ ? t_l1_N_ : min(n_patch_, t_l1_N_);
  GEMM_CUBE_SIZE cube_size {shift_inwards_K, shift_inwards_N, run_params_.ori_shape_n, dtype_size_};
  if (run_params_.trans_b_flag) {
    cube_size = {shift_inwards_N, shift_inwards_K, run_params_.ori_shape_k, dtype_size_};
  }
  AlignCount request_B = GetRequestND2NZ(cube_size);
  int32_t same_addr_B = GetSameAddressND2NZ(cube_size);
  int64_t request_B_sum = cube_size.n * cube_size.d * dtype_size_;

  double l2_biu_bw_B = 1.0 * request_B_sum * soc_bw_.READ_OTSd / (soc_bw_.L2_LATENCY_RD * request_B.Size());
  GetDMABiuBw<GEMM_CUBE_TAG::B>(l2_biu_bw_B, cube_size, soc_bw_.L2_LATENCY_RD);
  double l2_mata_bw_B = MataBandwidth<GEMM_CUBE_TAG::B>(request_B_sum, request_B.Size(), same_addr_B, cube_size);

  double ddr_biu_bw_B = 1.0 * request_B_sum * soc_bw_.READ_OTSd / (soc_bw_.DDR_LATENCY_RD * request_B.Size());
  GetDMABiuBw<GEMM_CUBE_TAG::B>(ddr_biu_bw_B, cube_size, soc_bw_.DDR_LATENCY_RD);

  double ddr_mata_bw_B = min(MataDDRBandwidth(request_B_sum, request_B.AlignSum()), l2_mata_bw_B);

  bw_info_.l2_l1B = MIN_FUNC({soc_bw_.L2_RO / active_core_num_, soc_bw_.L1_WO, l2_biu_bw_B, soc_bw_.L2_PHY_RO, l2_mata_bw_B});
  bw_info_.ddr_l1B = MIN_FUNC({soc_bw_.DDR_RO / active_core_num_, soc_bw_.L1_WO, ddr_biu_bw_B, soc_bw_.DDR_PHY, ddr_mata_bw_B});
  bw_info_.l1_l0B = min(soc_bw_.L1_RO, soc_bw_.L0B_WO);
}

void GemmBBEstimate::EstimateBandwidthA() {
  int64_t shift_inwards_M = shift_inwards_ ? t_l1_M_ : min(m_patch_, t_l1_M_);
  int64_t shift_inwards_K = shift_inwards_ ? t_l1_AK_ : min(k_patch_, t_l1_AK_);
  GEMM_CUBE_SIZE cube_size {shift_inwards_M, shift_inwards_K, run_params_.ori_shape_k, dtype_size_};
  if (run_params_.trans_a_flag) {
    cube_size = {shift_inwards_K, shift_inwards_M, run_params_.ori_shape_m, dtype_size_};
  }
  AlignCount request_A = GetRequestND2NZ(cube_size);
  int32_t same_addr_A = GetSameAddressND2NZ(cube_size);
  int64_t request_A_sum = cube_size.n * cube_size.d * dtype_size_;

  double l2_biu_bw_A = 1.0 * request_A_sum * soc_bw_.READ_OTSd / (soc_bw_.L2_LATENCY_RD * request_A.Size());
  GetDMABiuBw<GEMM_CUBE_TAG::A>(l2_biu_bw_A, cube_size, soc_bw_.L2_LATENCY_RD);
  double l2_mata_bw_A = MataBandwidth<GEMM_CUBE_TAG::A>(request_A_sum, request_A.Size(), same_addr_A, cube_size);

  double ddr_biu_bw_A = 1.0 * request_A_sum * soc_bw_.READ_OTSd / (soc_bw_.DDR_LATENCY_RD * request_A.Size());
  GetDMABiuBw<GEMM_CUBE_TAG::A>(ddr_biu_bw_A, cube_size, soc_bw_.DDR_LATENCY_RD);
  double ddr_mata_bw_A = min(MataDDRBandwidth(request_A_sum, request_A.AlignSum()), l2_mata_bw_A);

  bw_info_.l2_l1A = MIN_FUNC({soc_bw_.L2_RO / active_core_num_, soc_bw_.L1_WO, l2_biu_bw_A, soc_bw_.L2_PHY_RO, l2_mata_bw_A});
  bw_info_.ddr_l1A = MIN_FUNC({soc_bw_.DDR_RO / active_core_num_, soc_bw_.L1_WO, ddr_biu_bw_A, soc_bw_.DDR_PHY, ddr_mata_bw_A});
  bw_info_.l1_l0A = min(soc_bw_.L1_RO, soc_bw_.L0A_WO);
}

void GemmBBEstimate::EstimateBandwidth() {
  EstimateBandwidthA();
  EstimateBandwidthB();
  EstimateBandwidthC();
}

void GemmBBEstimate::EstimateCubeCycle() {
  double l0a_bw = min(soc_bw_.L0A_RO, soc_bw_.L0A_WO);
  double l0b_bw = min(soc_bw_.L0B_RO, soc_bw_.L0B_WO);

  int64_t m_pad = MathUtil::Align(m_, t_l1_M_);
  int64_t k_pad = MathUtil::Align(k_, t_l1_BK_);
  int64_t n_pad = MathUtil::Align(n_, t_l1_N_);
  int64_t compute_k = cube_k_;
  if (run_params_.dtype_a == ge::DataType::DT_FLOAT) {
    compute_k = run_params_.hf32_flag ? 8 : 4; // k_ size a cycle by compute,hf32_flag:8, float32:4
  }
  cycle_info_.cube_tile_cycle = 1.0 * result_.m_l0 * (t_l0_K_ / compute_k) * result_.n_l0;

  double gamma_A = (1.0 * t_l0_M_ * t_l0_K_ * dtype_size_ / l0a_bw) /
          cycle_info_.cube_tile_cycle;
  double gamma_B = (1.0 * t_l0_N_ * t_l0_K_ * dtype_size_ / l0b_bw) /
          cycle_info_.cube_tile_cycle;
  double gamma = MAX_FUNC({gamma_A, gamma_B, 1.0});
  cycle_info_.cube_cycle = 1.0 * batch_patch_ * m_pad * k_pad *n_pad / (kBlockSize * compute_k * kBlockSize) * gamma;
}

void GemmBBEstimate::GetL2HitRate(double &A_hit_rate, double &B_hit_rate) { // match schedule A first
  int64_t tiling_l2_M = m_run_ * t_l1_M_;
  int64_t tiling_l2_Ak = k_run_ * t_l1_AK_;
  int64_t tiling_l2_Bk = k_run_ * t_l1_BK_;
  int64_t tiling_l2_N = n_run_ * t_l1_N_;
  int64_t l1_AK = MathUtil::CeilDivision(run_params_.ori_shape_k, t_l1_AK_);
  int64_t l1_BK = MathUtil::CeilDivision(run_params_.ori_shape_k, t_l1_BK_);
  int64_t A_full_size = batch_run_ * run_params_.ori_shape_m * run_params_.ori_shape_k * dtype_size_ * l1_AK;
  int64_t B_full_size = batch_run_ * run_params_.ori_shape_k * run_params_.ori_shape_n * dtype_size_ * l1_BK;
  int64_t A_k_size = batch_run_ * tiling_l2_M * tiling_l2_Ak * dtype_size_ * l1_AK;
  int64_t B_k_size = batch_run_ * tiling_l2_N * tiling_l2_Bk * dtype_size_ * l1_BK;
  double l2_miss_rate_A = 1.0 / n_run_;
  double l2_miss_rate_B = 1.0 / m_run_;
  int64_t l2_size = PlatformInfo::GetInstance().l2_size;
  if (((A_full_size + B_full_size) <= l2_size) ||
      (A_full_size + B_k_size <= l2_size)) {
    l2_miss_rate_A = 1.0 / (get<0>(l1_load_repeat_) * n_run_);
    l2_miss_rate_B = 1.0 / (get<1>(l1_load_repeat_) * m_run_);
  } else if ((B_full_size + A_k_size <= l2_size) ||
      (A_k_size + B_k_size <= l2_size)) {
    l2_miss_rate_B = 1.0 / (get<1>(l1_load_repeat_) * m_run_);
  }
  A_hit_rate = 1 - l2_miss_rate_A;
  B_hit_rate = 1 - l2_miss_rate_B;
}

void GemmBBEstimate::EstimateMte2Cycle() {
  // This model is a fitting model:4.0 / 3 , 7.0 / 3, 2.0 / 3 statistical value
  auto calculate_mte2cycle = [&](double ddr_l1A, double ddR_l1B, double mte2_A_cycle, double mte2_B_cycle) {
    double mte2_A_tile_cycle =
          1.0 * t_l1_M_ * t_l1_AK_ * dtype_size_ / ddr_l1A;
    double mte2_B_tile_cycle =
            1.0 * t_l1_N_ * t_l1_BK_ * dtype_size_ / ddR_l1B;
    cycle_info_.mte2_cycle = mte2_A_cycle + mte2_B_cycle;
    cycle_info_.mte2_tile_cycle = mte2_A_tile_cycle + mte2_B_tile_cycle;
  };

  double mte2_A_data_size = 1.0 * batch_patch_ * m_patch_ * k_patch_ * dtype_size_;
  double mte2_B_data_size = 1.0 * batch_patch_ * n_patch_ * k_patch_ * dtype_size_;
  if (shift_inwards_) {
    mte2_A_data_size = 1.0 * batch_patch_ * MathUtil::Align(m_patch_, t_l1_M_) *
        MathUtil::Align(k_patch_, t_l1_AK_) * dtype_size_;
    mte2_B_data_size = 1.0 * batch_patch_ * MathUtil::Align(n_patch_, t_l1_N_) *
        MathUtil::Align(k_patch_, t_l1_BK_) * dtype_size_;
  }

  bool is_fit_l2 = k_shift_ ? false : (run_params_.ori_shape_m * run_params_.ori_shape_k * dtype_size_ +
                  run_params_.ori_shape_k * run_params_.ori_shape_n * dtype_size_ +
                  run_params_.ori_shape_m * run_params_.ori_shape_n * out_dtype_size_) <=
                  PlatformInfo::GetInstance().l2_size;
  if (is_fit_l2) {
    double mte2_A_bw = bw_info_.l2_l1A;
    double mte2_B_bw = bw_info_.l2_l1B;
    double mte2_A_cycle =
        mte2_A_data_size / bw_info_.ddr_l1A +  mte2_A_data_size * (get<0>(l1_load_repeat_) - 1) / mte2_A_bw;
    double mte2_B_cycle =
        mte2_B_data_size / bw_info_.ddr_l1B +  mte2_B_data_size * (get<1>(l1_load_repeat_) - 1) / mte2_B_bw;
    calculate_mte2cycle(bw_info_.ddr_l1A, bw_info_.ddr_l1B, mte2_A_cycle, mte2_B_cycle);
  } else {
    double A_hit_rate = 0;
    double B_hit_rate = 0;
    GetL2HitRate(A_hit_rate, B_hit_rate);
    double mte2_A_bw = 1.0 / (A_hit_rate / bw_info_.l2_l1A + (1 - A_hit_rate) / bw_info_.ddr_l1A);
    double mte2_B_bw = 1.0 / (B_hit_rate / bw_info_.l2_l1B + (1 - B_hit_rate) / bw_info_.ddr_l1B);
    double mte2_A_cycle = mte2_A_data_size * (get<0>(l1_load_repeat_)) / mte2_A_bw;
    double mte2_B_cycle = mte2_B_data_size * (get<1>(l1_load_repeat_)) / mte2_B_bw;
    calculate_mte2cycle(bw_info_.ddr_l1A, bw_info_.ddr_l1B, mte2_A_cycle, mte2_B_cycle);
  }
}

void GemmBBEstimate::EstimateMte1Cycle() {
  double mte1_A_data_size = 1.0 * batch_patch_ * m_ * k_ * dtype_size_ * get<0>(l0_load_repeat_);
  double mte1_B_data_size = 1.0 * batch_patch_ * n_ * k_ * dtype_size_ * get<1>(l0_load_repeat_);
  double mte1_A_cycle = mte1_A_data_size / bw_info_.l1_l0A;
  double mte1_A_tile_cycle = 1.0 * t_l0_M_ * t_l0_K_ * dtype_size_ / bw_info_.l1_l0A;
  double mte1_B_cycle = mte1_B_data_size / bw_info_.l1_l0B;
  double mte1_B_tile_cycle = 1.0 * t_l0_N_ * t_l0_K_ * dtype_size_ / bw_info_.l1_l0B;
  cycle_info_.mte1_cycle = mte1_A_cycle + mte1_B_cycle;
  cycle_info_.mte1_tile_cycle = mte1_A_tile_cycle + mte1_B_tile_cycle;
}

void GemmBBEstimate::EstimateFixpCycle() {
  double fixp_data_size = 1.0 * batch_patch_ * m_patch_ * n_patch_ * out_dtype_size_;
  cycle_info_.fix_cycle = fixp_data_size / bw_info_.l0C_l2;
  cycle_info_.fix_tile_cycle = 1.0 * t_l0_M_ * t_l0_N_ * out_dtype_size_ / bw_info_.l0C_l2;
}

void GemmBBEstimate::EstimateCycle() {
  EstimateCubeCycle();
  EstimateMte2Cycle();
  EstimateMte1Cycle();
  EstimateFixpCycle();
}

double GemmBBEstimate::EstimatePipe() {
  double bound_cycle = MAX_FUNC({cycle_info_.cube_cycle, cycle_info_.mte1_cycle,
                                cycle_info_.mte2_cycle + cycle_info_.fix_cycle});
  bool unit_flag = result_.db_l0c == optiling::kDbOff && result_.k_dim == 1 && !k_shift_;
  if (fabs(cycle_info_.cube_cycle - bound_cycle) <= numeric_limits<float>::epsilon()) {
    if (result_.db_l0c == optiling::kDbOn) {
      return bound_cycle + cycle_info_.mte2_tile_cycle + cycle_info_.mte1_tile_cycle + cycle_info_.fix_tile_cycle;
    }
    if (unit_flag) {
      // L0C using unit flag
      return bound_cycle + cycle_info_.mte2_tile_cycle + cycle_info_.mte1_tile_cycle +
          max(cycle_info_.fix_cycle - cycle_info_.cube_tile_cycle,  cycle_info_.fix_tile_cycle);
    }
    return bound_cycle + cycle_info_.mte2_tile_cycle + cycle_info_.mte1_tile_cycle + cycle_info_.fix_cycle;
  }
  if (fabs(cycle_info_.mte2_cycle + cycle_info_.fix_cycle - bound_cycle) <= numeric_limits<float>::epsilon()) {
    if (result_.db_l0c == optiling::kDbOn) {
      return bound_cycle + cycle_info_.cube_tile_cycle + cycle_info_.mte1_tile_cycle;
    }
    if (unit_flag) {
    // L0C using unit flag
      return bound_cycle + cycle_info_.mte1_tile_cycle +
          max(cycle_info_.cube_tile_cycle, cycle_info_.cube_cycle - cycle_info_.cube_tile_cycle - cycle_info_.mte2_cycle);
    }
    return bound_cycle + cycle_info_.mte1_tile_cycle +
          max(cycle_info_.cube_tile_cycle, cycle_info_.cube_cycle - cycle_info_.mte2_cycle);
  }
  return bound_cycle + cycle_info_.cube_tile_cycle + cycle_info_.mte2_tile_cycle + cycle_info_.fix_cycle;
}
}

