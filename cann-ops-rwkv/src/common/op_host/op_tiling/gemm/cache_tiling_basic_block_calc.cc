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
 * \file cache_tiling_basic_block_calc.cc
 * \brief function of cache tiling basic block calculator
 */
#include "cache_tiling_basic_block_calc.h"
#include "cache_tiling_basic.h"
#include "cache_tiling.h"
#include "cube/algorithm/hash/tiling_cache.h"
#include "mathutil.h"
using namespace std;
using namespace optiling;
// using optiling::cachetiling::MathUtil;
using namespace gemm_cache_tiling;

namespace {
static constexpr int64_t MIN_FULL_BW_SIZE = 512;  // 512 is base pkg size means 512B
static constexpr int64_t MIN_FULL_BW_SIZE_1982 = 128;
static constexpr int64_t ALIGN_SIZE = 256;
static constexpr int64_t ALIGN_SIZE_1982 = 128;
static constexpr int64_t CORE_NUM_2 = 24;
static constexpr int64_t L2_SIZE_2 = 192 * 1024 * 1024;
static constexpr double SWEET_UP_THRESHOLD_2 = 140;
static constexpr double SWEET_LOW_THRESHOLD_2 = 73;
static constexpr double SWEET_UP_THRESHOLD_3 = 117;
static constexpr double SWEET_LOW_THRESHOLD_3 = 61;
static constexpr double SWEET_UP_THRESHOLD_4 = 195;
static constexpr double SWEET_LOW_THRESHOLD_4 = 101;
static constexpr int64_t TRANS_SIZE_THRESHOLD = 64 * 1024;
static constexpr double WAVE_WASTE_THRESHOLD = 0.15;
static constexpr int64_t WAST_THRESHOLD = 2;
static constexpr double L2_RWO_EFFI = 0.4;
static constexpr double SAME_ADDR_PENALTY = 1.2;
static constexpr double ACTIVE_UP_THRESHOLD_2 = 0.7;
static constexpr double ACTIVE_LOW_THRESHOLD_2 = 0.8;
static constexpr double ACTIVE_UP_THRESHOLD_3 = 0.85;
static constexpr double ACTIVE_LOW_THRESHOLD_3 = 1;
static constexpr double ACTIVE_UP_THRESHOLD_4 = 0.5;
static constexpr double ACTIVE_LOW_THRESHOLD_4 = 0.75;
static constexpr double L2_HBM_RATION = 2.9;

using K_CUT_VECTOR = vector<array<int64_t, 2>>; // 2: K_CUT_RATIO SIZE{out_ratio, K}
static K_CUT_VECTOR K_CUT_RATIO_2{ {576, 24}, {144, 12}, {64, 8}, {16, 4}, {9, 3}, {4, 2} };
static K_CUT_VECTOR K_CUT_RATIO_3{ {400, 20}, {100, 10}, {25, 5}, {16, 4}, {4, 2} };

using UPLOW_TUPLE = tuple<int64_t, int64_t, int64_t, int64_t>;

template<class T>
inline int32_t GemmCmp (T a, T b) {
  if (a > b) {
    return 1;
  }
  if (a < b) {
      return -1;
  }
  return 0;
};

inline double GetSweetUpThreshold() {
  if (PlatformInfo::GetInstance().core_num == CORE_NUM_2) {
    return SWEET_UP_THRESHOLD_2;
  }
  if (PlatformInfo::GetInstance().l2_size == L2_SIZE_2) {
    return SWEET_UP_THRESHOLD_3;
  }
  return SWEET_UP_THRESHOLD_4;
}

inline double GetSweetLowThreshold() {
  if (PlatformInfo::GetInstance().core_num == CORE_NUM_2) {
    return SWEET_LOW_THRESHOLD_2;
  }
  if (PlatformInfo::GetInstance().l2_size == L2_SIZE_2) {
    return SWEET_LOW_THRESHOLD_3;
  }
  return SWEET_LOW_THRESHOLD_4;
}

inline K_CUT_VECTOR GetKCutRatio() {
  if (PlatformInfo::GetInstance().core_num == CORE_NUM_2) {
    return K_CUT_RATIO_2;
  }
  return K_CUT_RATIO_3;
}

inline double GetActiveUpThreshold() {
  if (PlatformInfo::GetInstance().core_num == CORE_NUM_2) {
    return ACTIVE_UP_THRESHOLD_2;
  }
  if (PlatformInfo::GetInstance().l2_size == L2_SIZE_2) {
    return ACTIVE_UP_THRESHOLD_3;
  }
  return ACTIVE_UP_THRESHOLD_4;
}

inline double GetActiveLowThreshold() {
  if (PlatformInfo::GetInstance().core_num == CORE_NUM_2) {
    return ACTIVE_LOW_THRESHOLD_2;
  }
  if (PlatformInfo::GetInstance().l2_size == L2_SIZE_2) {
    return ACTIVE_LOW_THRESHOLD_3;
  }
  return ACTIVE_LOW_THRESHOLD_4;
}

inline double GetL2RWOEffi(int64_t out_size, ge::DataType dtype_in) {
  double alpha = L2_RWO_EFFI;
  if (dtype_in == ge::DataType::DT_FLOAT) {
    return alpha;
  }
  int64_t l2_size = PlatformInfo::GetInstance().l2_size;
  if (out_size >= l2_size) {
    alpha *= 0.6;   // 0.6 L2 RW bandwidth loss ratio
  } else if (out_size >= l2_size / 2) { // half of L2_RWO_SIZE
    alpha *= 0.8;   // 0.8 L2 RW bandwidth loss ratio
  }
  return alpha;
}

inline set<int64_t> GetFactors(int64_t ori) {
  set<int64_t> factors;
  int64_t base = static_cast<int64_t>(sqrt(ori));
  for (; base > 0; base--) {
    if (ori % base == 0) {
      factors.insert(base);
      factors.insert(ori / base);
    }
  }
  return factors;
}

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

inline int64_t FindSmallestFactor(int64_t in, int64_t low, int64_t up, int64_t default_value = 1, int64_t step = 1) {
  for (int64_t i = low; i <= up; i += step) {
    if (in % i == 0) {
      return i;
    }
  }
  return default_value;
}

inline int64_t ReFindMinRedundant(int64_t in, int64_t low, int64_t up, int64_t step) {
  int64_t min_redundant = in;
  int64_t mark = low;
  for (int64_t tmp = low; tmp <= up; tmp += step) {
    int64_t tmp_redundant = in - (tmp % in);
    if (tmp_redundant % in == 0 || tmp_redundant <= (in * WAVE_WASTE_THRESHOLD)) {
      return tmp;
    }
    if (tmp_redundant < min_redundant) {
      min_redundant = tmp_redundant;
      mark = tmp;
    }
  }
  return mark;
}

UPLOW_TUPLE AppendUpLowCandidates(double cut, int64_t core_num, set<int64_t> &factors) {
  if (cut < 1) {
    return {1, 1, core_num, core_num};
  }
  if (cut >= core_num) {
    return {core_num, core_num, 1, 1};
  }

  int64_t lastData = -1;
  for (auto iter : factors) {
    if (iter >= cut) {
      if (lastData == -1) {
        return {1, 1, core_num, core_num};
      }
      return {iter, lastData, core_num / iter, core_num / lastData};
    }
    lastData = iter;
  }
  return {core_num, core_num, 1, 1};
}

inline int64_t CalculatorKl0(int64_t k_l1, int64_t k_l0_max, int64_t cube_k) {
  int64_t k_l0 = min(k_l0_max, k_l1);
  if (k_l0 != k_l1) {
    int64_t tmp = MathUtil::CeilDivision(k_l1, cube_k);
    for (int64_t i = k_l0 / cube_k; i > 0 ; i--) {
      if (tmp % i == 0) {
        return i;
      }
    }
  }
  return MathUtil::CeilDivision(k_l0, cube_k);
}

}

namespace gemm_cache_tiling {
inline double GemmBBCalculator::CalcEff(int64_t inner_size) const {
  if (inner_size % min_full_dw_size == 0) {
    return 1.0;
  }
  if (inner_size > min_full_dw_size) {
    return 2.0; // 2.0:redundancy bandwidth size factor
  }
    return 2.5; // 2.5:redundancy bandwidth size factor
}

inline int64_t GemmBBCalculator::GetAvaliableL1Size(int64_t t_m_tile, int64_t t_n_tile) const {
  return reserve_l1 ?
    (PlatformInfo::GetInstance().l1_size - DB_SIZE * kBlockSize * (t_m_tile + t_n_tile) * dtype_size) :
    PlatformInfo::GetInstance().l1_size;
}

template<typename T>
bool GemmBBCalculator::NeedUpadateCandidatesLoadInfo(const T& last_info, const T& new_info,
                                                     double cur_AI, double in_AI) {
  constexpr int64_t move_step = 10; // 10：move data to compare res
  int64_t res = 0;
  if (cur_AI >= sweet_up) {
    res = GemmCmp(last_info.bb_counts, new_info.bb_counts);
    res = res * move_step + GemmCmp(new_info.active_ratio, last_info.active_ratio);
    res = res * move_step + GemmCmp(last_info.core_diff, new_info.core_diff);
  } else {
    double active_ratio = min(new_info.active_ratio, last_info.active_ratio);
    if (active_ratio >= active_bound && (in_AI < sweet_point_bound * active_ratio * active_ratio)) {
      res = GemmCmp(new_info.full_score, last_info.full_score);
      res = res * move_step + GemmCmp(last_info.active_ratio, new_info.active_ratio);
    } else {
      res = GemmCmp(new_info.active_ratio, last_info.active_ratio);
      res = res * move_step + GemmCmp(new_info.full_score, last_info.full_score);
    }
    res = res * move_step + GemmCmp(last_info.core_diff, new_info.core_diff);
    res = res * move_step + GemmCmp(last_info.bb_counts, new_info.bb_counts);
  }
  return res > 0;
}

void GemmBBCalculator::UpdateLoadInfo(const GemmCandidatesInfo &info,
                                      GemmCandidatesLoadInfo &loadInfo,
                                      int64_t core_num) {
  int64_t m_core_dim = MathUtil::CeilDivision(info.m_dim, info.m_cut);
  int64_t n_core_dim = MathUtil::CeilDivision(info.n_dim, info.n_cut);
  double bb_counts = m_core_dim * n_core_dim * max(1.0, 1.0 * info.n_cut * info.m_cut / core_num);
  int64_t m_active_cores = MathUtil::CeilDivision(info.m_dim, m_core_dim);
  int64_t n_active_cores = MathUtil::CeilDivision(info.n_dim, n_core_dim);
  int64_t active_cores = m_active_cores * n_active_cores;
  double active_ratio = 1.0 * active_cores / (MathUtil::CeilDivision(active_cores, core_num) * core_num);
  int64_t core_diff = abs(m_active_cores - n_active_cores);

  int64_t bias_status = run_params->bias_flag ? 1 : 0;
  int64_t tile_l1_size = (info.m_tile * k_tile + info.n_tile * (quant_status + bias_status + k_tile)) *
                         dtype_size * DB_SIZE;
  int64_t available_l1_size = GetAvaliableL1Size(info.m_tile, info.n_tile);
  bool m_full = (info.m_dim <= info.m_cut);
  bool n_full = (info.n_dim <= info.n_cut);
  double local_a_eff = info.n_cut == 1 ? a_eff * SAME_ADDR_PENALTY : a_eff;
  double local_b_eff = info.m_cut == 1 ? b_eff * SAME_ADDR_PENALTY : b_eff;

  int64_t m_miss = is_l2_fit ? 1 : MathUtil::CeilDivision(info.n_dim, info.n_cut);
  int64_t n_miss = is_l2_fit ? 1 : MathUtil::CeilDivision(info.m_dim, info.m_cut);
  auto CalcScore = [=](int64_t base, int64_t m_repeats, int64_t n_repeats) {
    int64_t m_hit = m_repeats - m_miss;
    int64_t n_hit = n_repeats - n_miss;
    return base + 1.0 / ((local_a_eff * m_hit + local_b_eff * n_hit) +
        L2_HBM_RATION * (run_params->ori_shape_m * m_miss + run_params->ori_shape_n * n_miss));
  };

  if (k_per_core_tmp <= k_tile && tile_l1_size <= available_l1_size) {
    int64_t n_repeats = (m_full || n_full) ? info.m_cut : info.m_dim;
    double score = CalcScore(3, info.n_cut, n_repeats); // 3: estimate score
    loadInfo = {bb_counts, active_ratio, core_diff, score, k_tile, FULL_LOAD_BOTH};
    return;
  }

  auto min_k_func = [this](int64_t i) {
    int64_t min_k = this->k_per_core_tmp / this->cube_k;
    int64_t sqrtn = static_cast<int64_t>(sqrt(min_k));
    if (i < sqrtn) {
      for (; i < sqrtn; i++) {
        if (min_k % i == 0) {
          min_k = i;
          break;
        }
      }
    } else {
        for (int64_t j = min_k / i; j > 0; j--) {
          if (min_k % j == 0) {
            min_k = min_k / j;
            break;
          }
        }
    }
    return min_k * cube_k;
  };

  if (info.m_dim <= info.m_cut) {
    int64_t i = MathUtil::CeilDivision(TRANS_SIZE_THRESHOLD, info.n_tile * dtype_size * cube_k);
    int64_t min_k = min_k_func(i);
    tile_l1_size = (k_per_core_tmp * info.m_tile + (min_k + bias_status + quant_status) *
                            info.n_tile * DB_SIZE) * dtype_size;
    if (tile_l1_size <= available_l1_size) {
      double score = CalcScore(1, info.n_cut, info.m_cut);
      loadInfo = {bb_counts, active_ratio, core_diff, score, min_k, FULL_LOAD_STATE::FULL_LOAD_A};
      return;
    }
  }

  if (info.n_dim <= info.n_cut) {
    int64_t i = MathUtil::CeilDivision(TRANS_SIZE_THRESHOLD, info.m_tile * dtype_size * cube_k);
    int64_t min_k = min_k_func(i);
    tile_l1_size = ((k_per_core_tmp + bias_status + quant_status) * info.n_tile +
                            min_k * info.m_tile * DB_SIZE) * dtype_size;
    if (tile_l1_size <= available_l1_size) {
      double score = CalcScore(1, info.n_cut, info.m_cut);
      loadInfo = {bb_counts, active_ratio, core_diff, score, min_k, FULL_LOAD_STATE::FULL_LOAD_B};
      return;
    }
  }
  int64_t n_repeats = MathUtil::Align(info.m_dim, info.m_cut);
  int64_t m_repeats = MathUtil::Align(info.n_dim, info.n_cut);
  double score = CalcScore(0, m_repeats, n_repeats);
  loadInfo = {bb_counts, active_ratio, core_diff, score, k_tile, FULL_LOAD_NONE};
  return;
}

CANDIDATES GemmBBCalculator::GenCandidates(int64_t core_num) {
  int64_t m_gcd = __gcd(result_info.m_dim, core_num);
  int64_t n_gcd = __gcd(result_info.n_dim, core_num);
  CANDIDATES datas {{m_gcd, min(result_info.n_dim, core_num / m_gcd)},
      {min(result_info.m_dim, core_num / n_gcd), n_gcd}};
  if (m_gcd == 1) {
    int64_t tmp = max(1L, min(result_info.m_dim, core_num));
    datas.insert({ tmp, core_num / tmp });
  }
  if (n_gcd == 1) {
    int64_t tmp = max(1L, min(result_info.n_dim, core_num));
    datas.insert({ core_num / tmp, tmp });
  }
  set<int64_t> factors = GetFactors(core_num);
  auto data_func = [&](double cut) {
    UPLOW_TUPLE tuple = AppendUpLowCandidates(cut, core_num, factors);
    datas.insert({ min(get<0>(tuple), result_info.m_dim), min(get<2>(tuple), result_info.n_dim) });
    datas.insert({ min(get<1>(tuple), result_info.m_dim), min(get<3>(tuple), result_info.n_dim) });
  };

  double cut = sqrt(1.0 * eff_ratio * core_num * n_tile / m_tile); // L2 hit rate and bandwidth efficiency
  data_func(cut);

  cut = sqrt(1.0 * core_num * result_info.m_dim * m_tile / (result_info.n_dim * n_tile) / eff_ratio); // redundant transfer
  data_func(cut);
  if (m_gcd == core_num || n_gcd == core_num) {
    cut = sqrt(1.0 * core_num);
    data_func(cut);
  }

  if (!divided) {
    return datas;
  }
  {
    CANDIDATES divided_datas;
    for (auto iter : datas) {
      divided_datas.insert({FindSmallestFactor(result_info.m_dim, iter[0], result_info.m_dim, result_info.m_dim),
          FindSmallestFactor(result_info.n_dim, iter[1], result_info.n_dim, result_info.n_dim)});
    }
    return divided_datas;
  }
}

void GemmBBCalculator::AdjustTile(int64_t &dim, int64_t &tile, int64_t ori, int64_t cut, bool be_even) {
  int64_t cut_active = MathUtil::CeilDivision(dim, MathUtil::CeilDivision(dim, cut));
  int64_t core_dim = MathUtil::CeilDivision(dim, cut);
  int64_t ori_map = MathUtil::CeilDivision(ori, kBlockSize) * kBlockSize;
  int64_t tile_adjusted = MathUtil::CeilDivision(ori_map, (core_dim * cut * kBlockSize)) * kBlockSize;
  if (!be_even || tile_adjusted % (2 * kBlockSize) == 0) // 2: align to even
  {
    int64_t dim_adjust = MathUtil::CeilDivision(ori, tile_adjusted);
    int64_t cut_adjust = MathUtil::CeilDivision(dim_adjust, MathUtil::CeilDivision(dim_adjust, cut));
    if (divided && (ori_map % (dim_adjust * tile_adjusted) != 0)) {
      return;
    }
    if (cut_adjust >= cut_active) {
      dim = dim_adjust;
      tile = tile_adjusted;
    }
  }
}

void GemmBBCalculator::AdjustDimTile(int64_t core_num, double cur_AI, double in_AI) {
  k_per_core_tmp = MathUtil::CeilDivision(run_params->k, k_cut) * cube_k;
  GemmCandidatesInfo last_can_info {1, 1, result_info.m_dim, this->m_tile, result_info.n_dim, this->n_tile};
  GemmCandidatesLoadInfo last_can_load_info {INT_MAX, 0, core_num, 0, k_tile, FULL_LOAD_NONE};
  CANDIDATES datas = GenCandidates(core_num);
  for (auto iter : datas) {
    GemmCandidatesInfo can_info {iter[0], iter[1], result_info.m_dim, m_tile, result_info.n_dim, n_tile};
    GemmCandidatesLoadInfo can_load_info;
    if (!is_adjust && in_AI >= sweet_low) {
      if (!run_params->trans_a_flag) {
        AdjustTile(can_info.m_dim, can_info.m_tile, run_params->ori_shape_m, can_info.m_cut, m_be_even);
      }
      if (run_params->trans_b_flag) {
        AdjustTile(can_info.n_dim, can_info.n_tile, run_params->ori_shape_n, can_info.n_cut, n_be_even);
      }
    }
    UpdateLoadInfo(can_info, can_load_info, core_num);
    if (NeedUpadateCandidatesLoadInfo<GemmCandidatesLoadInfo>(last_can_load_info, can_load_info, cur_AI, in_AI)) {
      last_can_info = can_info;
      last_can_load_info = can_load_info;
    }
  }
  result_info.m_dim = min(last_can_info.m_cut, last_can_info.m_dim);
  this->m_tile = last_can_info.m_tile;
  result_info.n_dim = min(last_can_info.n_cut, last_can_info.n_dim);
  this->n_tile = last_can_info.n_tile;
  this->load_info = last_can_load_info;
  m_divided = (run_params->m * kBlockSize) % (result_info.m_dim * m_tile) == 0;
  n_divided = (run_params->n * kBlockSize) % (result_info.n_dim * n_tile) == 0;
  return;
}

void GemmBBCalculator::AdjustDimByDivided(bool &batchDivided) {
  if (!batchDivided) {
    batch_cut = FindLargestFactor(run_params->batch, max(min_batch_cut, max_batch_cut));
  }
  int64_t cores = max(1L, PlatformInfo::GetInstance().core_num / batch_cut / k_cut);
  if (!m_divided) {
    int64_t tmp = FindLargestFactor(run_params->m, m_tile / kBlockSize);
    m_tile = tmp * kBlockSize;
    result_info.m_dim = FindSmallestFactor(run_params->m / tmp, result_info.m_dim, cores);
  }
  if (!n_divided) {
    int64_t tmp = FindLargestFactor(run_params->n, n_tile / kBlockSize);
    n_tile = tmp * kBlockSize;
    result_info.n_dim = FindSmallestFactor(run_params->n / tmp, result_info.n_dim,
        (WAST_THRESHOLD * cores / result_info.m_dim), run_params->n / tmp);
  }

  m_divided = true;
  n_divided = true;
  batchDivided = true;
}

void GemmBBCalculator::AdjustKCut() {
  double in_AI = 2.0 * run_params->ori_shape_m * run_params->ori_shape_n /
      (a_eff * result_info.n_dim + b_eff * result_info.m_dim) / dtype_size;
  double cur_AI = min(2.0 * run_params->ori_shape_k / out_dtype_size, in_AI);
  if (!is_adjust || k_cut > 1) {
    int64_t cores = max(1L, available_core_num / k_cut);
    divided = (k_cut > 1 || must_divided);
    AdjustDimTile(cores, cur_AI, in_AI);
  }
  bool batchDivided = !run_params->is_batch_matmul_op || run_params->batch % batch_cut == 0;
  if ((must_divided || k_cut > 1) && (!m_divided || !n_divided || !batchDivided)) {
    AdjustDimByDivided(batchDivided);
  }
  int64_t core_wst_factor = PlatformInfo::GetInstance().core_num /
      (result_info.m_dim * result_info.n_dim * k_cut * batch_cut);
  batch_cut = (core_wst_factor >= WAST_THRESHOLD && run_params->is_batch_matmul_op)
                  ? FindLargestFactor(run_params->batch, batch_cut * core_wst_factor, batch_cut)
                  : batch_cut;
}

void GemmBBCalculator::CalculateKTileFullNone(int64_t &a_k_tile, int64_t &b_k_tile) {
  int64_t available_l1_size = GetAvaliableL1Size(m_tile, n_tile);
  int64_t bias_status = run_params->bias_flag ? 1 : 0;
  available_l1_size -= DB_SIZE * (bias_status + quant_status) * n_tile;
  int64_t min_k1 = MathUtil::Align(
      MathUtil::CeilDivision(TRANS_SIZE_THRESHOLD, max(m_tile, n_tile) * dtype_size), cube_k);
  int64_t k_shrunk = k_tile;
  int64_t k_align_shrunk = k_tile;
  for (int64_t i = min_k1; i < k_tile + 1; i += cube_k) {
    if (k_tile % i == 0) {
      k_shrunk = i;
      break;
    }
  }
  for (int64_t i = k_shrunk; i < k_tile + 1; i += cube_k) {
    if (i % (min_full_dw_size / dtype_size) == 0) {
      k_align_shrunk = i;
      break;
    }
  }

  a_k_tile = run_params->trans_a_flag ? k_shrunk : k_align_shrunk;
  b_k_tile = run_params->trans_b_flag ? k_align_shrunk : k_shrunk;
  bool expand_ak1 = run_params->trans_a_flag ? false : !k_aligned;
  bool expand_bk1 = run_params->trans_b_flag ? !k_aligned : false;
  if (!run_params->trans_a_flag && run_params->trans_b_flag) {
    expand_ak1 = true;
    expand_bk1 = true;
  }

  if (expand_ak1 && expand_bk1) {
    int64_t max_k1 = min(run_params->k * cube_k, available_l1_size / ((m_tile + n_tile) * DB_SIZE * dtype_size));
    int64_t max_k1_align = (max_k1 / k_align_shrunk) * k_align_shrunk;
    a_k_tile = max_k1_align;
    b_k_tile = a_k_tile;
  }
  if (expand_ak1) {
    int64_t max_k1 = min(run_params->k * cube_k,
        (available_l1_size / dtype_size - b_k_tile * n_tile * DB_SIZE) / (m_tile * DB_SIZE));
    if (max_k1 >= a_k_tile) {
      a_k_tile = (max_k1 / a_k_tile) * a_k_tile;
      int64_t max_k1_fac = min(run_params->k,
          (available_l1_size / dtype_size - a_k_tile * m_tile * DB_SIZE) / n_tile / DB_SIZE / cube_k);
      b_k_tile = cube_k * FindLargestFactor(a_k_tile / cube_k, max_k1_fac);
    }
  } else if (expand_bk1) {
    int64_t max_k1 = min(run_params->k * cube_k,
        (available_l1_size / dtype_size - a_k_tile * m_tile * DB_SIZE) / (n_tile * DB_SIZE));
    if (max_k1 >= b_k_tile) {
      b_k_tile = (max_k1 / b_k_tile) * b_k_tile;
      int64_t max_k1_fac = min(run_params->k,
          (available_l1_size / dtype_size - b_k_tile * n_tile * DB_SIZE) / m_tile / DB_SIZE / cube_k);
      a_k_tile = cube_k * FindLargestFactor(b_k_tile / cube_k, max_k1_fac);
    }
  } else if (available_l1_size != PlatformInfo::GetInstance().l1_size &&
      (m_tile * a_k_tile + n_tile * b_k_tile) * dtype_size * DB_SIZE > available_l1_size) {
    k_tile = max((available_l1_size / dtype_size / DB_SIZE / (m_tile + n_tile) / cube_k), 1L);
    a_k_tile = k_tile;
    b_k_tile = k_tile;
  }
  return;
}

int64_t GemmBBCalculator::CalculatorKl0ByDivided(int64_t &a_k_tile, int64_t &b_k_tile, int64_t k_l0_max) {
  if (a_k_tile > b_k_tile) {
    int64_t tmp_ak = FindLargestFactor(run_params->k, a_k_tile / cube_k);
    a_k_tile = tmp_ak * cube_k;
    int64_t tmp_bk = FindLargestFactor(tmp_ak, b_k_tile / cube_k);
    b_k_tile = tmp_bk * cube_k;
    return FindLargestFactor(tmp_bk, k_l0_max / cube_k);
  }
  if (b_k_tile > a_k_tile) {
    int64_t tmp_bk = FindLargestFactor(run_params->k, b_k_tile / cube_k);
    b_k_tile = tmp_bk * cube_k;
    int64_t tmp_ak = FindLargestFactor(tmp_bk, a_k_tile / cube_k);
    a_k_tile = tmp_ak * cube_k;
    return FindLargestFactor(tmp_ak, k_l0_max / cube_k);
  }
  int64_t tmp_k = FindLargestFactor(run_params->k, a_k_tile / cube_k);
  a_k_tile = tmp_k * cube_k;
  b_k_tile = a_k_tile;
  return FindLargestFactor(tmp_k, k_l0_max / cube_k);
}

void GemmBBCalculator::CalculateAFullKTile(int64_t &a_k_tile, int64_t &b_k_tile,
                                           int64_t bias_status, int64_t bw_type_size) {
  int64_t available_l1_size =  reserve_l1 ?
    (PlatformInfo::GetInstance().l1_size - kBlockSize * (m_tile + DB_SIZE * n_tile) * dtype_size) :
    PlatformInfo::GetInstance().l1_size;
  available_l1_size -= DB_SIZE * bias_status * n_tile;
  a_k_tile = k_per_core_tmp;
  b_k_tile = load_info.min_k;
  if(k_per_core < k_per_core_tmp) {
    int64_t tmp_k = MathUtil::CeilDivision(TRANS_SIZE_THRESHOLD, n_tile * dtype_size * cube_k) * cube_k;
    int64_t min_bk1 = FindSmallestFactor(k_per_core, tmp_k, k_per_core, k_per_core, cube_k);
    if((k_per_core * m_tile + (min_bk1 + bias_status) * DB_SIZE * n_tile) * dtype_size <= available_l1_size) {
      a_k_tile = k_per_core;
      b_k_tile = min_bk1;
    }
  }
  if (run_params->trans_b_flag && (a_k_tile % bw_type_size == 0)) {
    int64_t max_bk1 = min(run_params->k * cube_k,
        (available_l1_size / dtype_size - a_k_tile * m_tile) / n_tile / DB_SIZE);
    for (int64_t tmp_k = load_info.min_k; tmp_k < max_bk1 + 1; tmp_k += cube_k) {
      if (a_k_tile % tmp_k == 0 && tmp_k % bw_type_size == 0) {
        b_k_tile = tmp_k;
        break;
      }
    }
  }
}

void GemmBBCalculator::CalculateBFullKTile(int64_t &a_k_tile, int64_t &b_k_tile,
                                           int64_t bias_status, int64_t bw_type_size) {
  int64_t available_l1_size =  reserve_l1 ?
    (PlatformInfo::GetInstance().l1_size - kBlockSize * (m_tile + DB_SIZE * n_tile) * dtype_size) :
    PlatformInfo::GetInstance().l1_size;
  available_l1_size -= bias_status * n_tile;
  b_k_tile = k_per_core_tmp;
  a_k_tile = load_info.min_k;
  if(k_per_core < k_per_core_tmp) {
    int64_t tmp_k = MathUtil::CeilDivision(TRANS_SIZE_THRESHOLD, n_tile * dtype_size * cube_k) * cube_k;
    int64_t min_ak1 = FindSmallestFactor(k_per_core, tmp_k, k_per_core, k_per_core, cube_k);
    if((min_ak1 * m_tile * DB_SIZE + (k_per_core + bias_status) * n_tile) * dtype_size <= available_l1_size) {
      b_k_tile = k_per_core;
      a_k_tile = min_ak1;
    }
  }
  if (!run_params->trans_a_flag && (b_k_tile % bw_type_size == 0)) {
    int64_t max_ak1 = min(run_params->k * cube_k,
        (available_l1_size / dtype_size - b_k_tile * n_tile) / m_tile / DB_SIZE);
    for (int64_t tmp_k = load_info.min_k; tmp_k < max_ak1 + 1; tmp_k += cube_k) {
      if (b_k_tile % tmp_k == 0 && tmp_k % bw_type_size == 0) {
        a_k_tile = tmp_k;
        break;
      }
    }
  }
}

int64_t GemmBBCalculator::CalculateKTile(int64_t &a_k_tile, int64_t &b_k_tile) {
  int64_t bw_type_size = min_full_dw_size / dtype_size;
  int64_t k_cut_ori = MathUtil::CeilDivision(run_params->ori_shape_k, k_tile);
  int64_t bias_status = run_params->bias_flag ? 1 : 0;
  k_per_core = MathUtil::CeilDivision(run_params->k, k_cut) * cube_k;
  if (k_cut_ori > k_cut) {
    switch (this->load_info.load_state) {
      case FULL_LOAD_NONE: {
        CalculateKTileFullNone(a_k_tile, b_k_tile);
        break;
      }
      case FULL_LOAD_A: {
        CalculateAFullKTile(a_k_tile, b_k_tile, bias_status, bw_type_size);
        break;
      }
      case FULL_LOAD_B: {
        CalculateBFullKTile(a_k_tile, b_k_tile, bias_status, bw_type_size);
        break;
      }
      default:
        break;
    }
  }
  else if(k_cut > 1) {
    if (run_params->trans_a_flag && !run_params->trans_b_flag) {
      a_k_tile = k_per_core;
      b_k_tile = k_per_core;
    } else {
      int64_t min_k = MathUtil::CeilDivision(k_per_core, cube_k) * cube_k;
      int64_t k_align_shrunk = k_tile;
      for(int64_t i = min_k; i <= k_tile; i += cube_k) {
        if( i % bw_type_size == 0) {
          k_align_shrunk = i;
          break;
        }
      }
      a_k_tile = k_align_shrunk;
      b_k_tile = k_align_shrunk;
    }
  }
  int64_t k_l0_max = min(PlatformInfo::GetInstance().l0a_size / dtype_size / DB_SIZE / m_tile,
                         PlatformInfo::GetInstance().l0b_size / dtype_size / DB_SIZE / n_tile);
  if (align_cube_k) {
    k_l0_max = k_l0_max / kBlockSize * kBlockSize;
  }
  int64_t k_l0 = CalculatorKl0(min(a_k_tile, b_k_tile), k_l0_max, cube_k);
  // FP16->FP32 ml1 nl1 nl1 must divide exactly
  if (must_divided && k_cut == 1) {
    k_l0 = CalculatorKl0ByDivided(a_k_tile, b_k_tile, k_l0_max);
  }
  return k_l0;
}

void GemmBBCalculator::UpdateBMMResult() {
  if (run_params->is_batch_matmul_op && !run_params->do_not_multi_batch) {
    result_info.db_l0c = kDbOff;
    if ((n_tile * m_tile * 4 * DB_SIZE <= PlatformInfo::GetInstance().l0c_size) && // 4: l0Cout size
        (n_tile * 4 * DB_SIZE <= PlatformInfo::GetInstance().bt_size)) { // 4: l0Cout size
      result_info.db_l0c = kDbOn;
    }
    int64_t k_l0 = result_info.k_l0 * cube_k;
    if (align_cube_k) {
      k_l0 = MathUtil::Align(result_info.k_l0 * cube_k, kBlockSize);
    }
    bool AB_full_load = false;
    if (result_info.m_dim * m_tile >= run_params->ori_shape_m && result_info.n_dim * n_tile >= run_params->ori_shape_n &&
        k_l0 * k_cut >= run_params->ori_shape_k) {
      AB_full_load = true;
    }
    int64_t batch_per_core = MathUtil::CeilDivision(run_params->batch, batch_cut);
    if (batch_per_core > 1 && AB_full_load) {
      result_info.db_al1 = kDbOff;
      result_info.db_bl1 = kDbOff;
      result_info.db_l0c = kDbOff;
      int64_t max_batch_temp = min(PlatformInfo::GetInstance().l0a_size / dtype_size / DB_SIZE / m_tile / k_l0,
                              PlatformInfo::GetInstance().l0b_size / dtype_size / DB_SIZE / n_tile / k_l0);
      int64_t max_batch = min(max_batch_temp, PlatformInfo::GetInstance().l0c_size / DB_SIZE / m_tile / n_tile / 4);
      result_info.batch_l0 = FindLargestFactor(batch_per_core, max_batch);
    }
  }
}

bool GemmBBCalculator::BasicBlockFit() {
  is_adjust = false;
  k_cut = 1;
  this->load_info = {INT_MAX, 0, available_core_num, 0, k_tile, FULL_LOAD_NONE};
  if (!m_divided || !n_divided) {
    double in_AI = 2.0 * run_params->ori_shape_m * run_params->ori_shape_n /
      (a_eff * result_info.n_dim + b_eff * result_info.m_dim) / dtype_size;
    double cut_AI = min(in_AI,  2 * run_params->ori_shape_k * alpha / out_dtype_size);
    divided = false;
    AdjustDimTile(available_core_num, cut_AI, in_AI);
    is_adjust = true;
  }
  AdjustKCut();

  int64_t a_k_tile = k_tile;
  int64_t b_k_tile = k_tile;
  result_info.k_l0 = CalculateKTile(a_k_tile, b_k_tile);
  result_info.k_dim = k_cut;
  result_info.batch_dim = batch_cut;
  result_info.kal1_16 = MathUtil::CeilDivision(a_k_tile, cube_k);
  result_info.kbl1_16 = MathUtil::CeilDivision(b_k_tile, cube_k);
  result_info.m_l0 = MathUtil::CeilDivision(m_tile, kBlockSize);
  result_info.n_l0 = MathUtil::CeilDivision(n_tile, kBlockSize);
  result_info.m_l1 = result_info.m_l0;
  result_info.n_l1 = result_info.n_l0;

  if (result_info.m_dim * m_tile >= run_params->ori_shape_m && a_k_tile * k_cut >= run_params->ori_shape_k) {
    result_info.db_al1 = kDbOff;
  }
  if (result_info.n_dim * n_tile >= run_params->ori_shape_n && b_k_tile * k_cut >= run_params->ori_shape_k) {
    result_info.db_bl1 = kDbOff;
  }
  UpdateBMMResult();
  return true;
}

void GemmBBCalculator::InitBMMCondition(double tmp_threshold) {
  if (run_params->is_batch_matmul_op && run_params->batch > 1) {
    int64_t core_num = PlatformInfo::GetInstance().core_num;
    int64_t l1_size = PlatformInfo::GetInstance().l1_size;
    double opt_ai = 2.0 * run_params->ori_shape_m * run_params->ori_shape_n / (a_eff + b_eff) / dtype_size;
    int64_t tmp_ratio = static_cast<int64_t>(opt_ai / sweet_point_bound);
    min_batch_cut = MathUtil::CeilDivision(core_num, max(1L, tmp_ratio));
    int64_t min_reduandant = run_params->batch;
    bool l1_fit = (2 * run_params->ori_shape_k * dtype_size *
        sqrt(core_num * run_params->ori_shape_m * run_params->ori_shape_n)) <= core_num * l1_size;
    max_batch_cut = l1_fit ? run_params->batch :
        static_cast<int64_t>(max(1.0, tmp_threshold / (run_params->ori_shape_m + run_params->ori_shape_n)));
    if(max_batch_cut < min_batch_cut) {
      max_batch_cut = run_params->batch;
      min_batch_cut = 1;
    }

    set<int64_t> core_fac_arr = GetFactors(core_num);
    for (auto itr = core_fac_arr.rbegin(); itr != core_fac_arr.rend(); itr++) {
      int64_t tmp = *itr;
      if (tmp > max_batch_cut) {
        continue;
      }
      if (batch_cut != 1 && tmp < min_batch_cut) {
        break;
      }
      int64_t batch_dim_adjusted = MathUtil::CeilDivision(run_params->batch,
          MathUtil::CeilDivision(run_params->batch, tmp));
      if (batch_dim_adjusted != tmp) {
        continue;
      }
      int64_t tmp_redundant = tmp - (run_params->batch % tmp);
      if (tmp_redundant == tmp || tmp_redundant <= run_params->batch * WAVE_WASTE_THRESHOLD) {
        batch_cut = tmp;
        break;
      } else if (tmp_redundant < min_reduandant) {
        min_reduandant = tmp_redundant;
        batch_cut = tmp;
      }
    }
  }
  batch_cut = min(batch_cut, run_params->batch);
  return;
}

bool GemmBBCalculator::Init(int32_t bb_idx) {
  int64_t n_limited_by_bias = PlatformInfo::GetInstance().bt_size / 4; // 4: partial sum size
  if (run_params->bias_flag && run_params->ori_shape_n > n_limited_by_bias
      && BASIC_BLOCK_TILE[bb_idx][TILE_N_IDX] > n_limited_by_bias) {
      return false;
  }
  memset_s(&load_info, sizeof(load_info), 0, sizeof(load_info));
  memset_s(&result_info, sizeof(result_info), 0, sizeof(result_info));
  k_cut = 1;
  batch_cut = 1;
  k_per_core_tmp = 0;
  min_full_dw_size = PlatformInfo::GetInstance().support_l12bt_bf16() ? MIN_FULL_BW_SIZE_1982 : MIN_FULL_BW_SIZE;
  align_size = PlatformInfo::GetInstance().support_l12bt_bf16() ? ALIGN_SIZE_1982 : ALIGN_SIZE;
  result_info.db_l0c = kDbOff;
  if (bb_idx == 4 || bb_idx == 19) { // 4,19: basic block idx need to open db buffer
    result_info.db_l0c = kDbOn;
  }
  result_info.db_al1 = kDbOn;
  result_info.db_bl1 = kDbOn;
  result_info.batch_l0 = 1;
  dtype_in = static_cast<ge::DataType>(run_params->dtype_a);
  dtype_out = static_cast<ge::DataType>(run_params->dtype_out);

  double sweet_up_threshold = GetSweetUpThreshold();
  double sweet_low_threshold = GetSweetLowThreshold();

  if (dtype_in == ge::DataType::DT_FLOAT) { // float32
    dtype_size = kFp32Bytes;
    cube_k = reducekBlockSize;
    sweet_up = run_params->hf32_flag ? sweet_up_threshold / 2 : sweet_up_threshold / 4; // 2,4 sweet_threshold
    sweet_low = run_params->hf32_flag ? sweet_low_threshold / 2 : sweet_low_threshold / 4; // 2,4 sweet_threshold
  } else if (dtype_in == ge::DataType::DT_INT8) { // int8
    dtype_size = kInt8Bytes;
    cube_k = increasekBlockSize;
    sweet_up = sweet_up_threshold * 2; // 2: Twice as much arithmetic of fp32
    sweet_low = sweet_low_threshold * 2; // 2: Twice as much arithmetic of fp32
  } else { // float16 and bf16
    dtype_size = kFp16Bytes;
    cube_k = kBlockSize;
    sweet_up = sweet_up_threshold;
    sweet_low = sweet_low_threshold;
  }
  alpha = GetL2RWOEffi(run_params->ori_shape_m * run_params->ori_shape_n * dtype_size, dtype_in);
  out_dtype_size = dtype_out == ge::DataType::DT_FLOAT  ? kFp32Bytes
                   :(dtype_out == ge::DataType::DT_INT8 ? kInt8Bytes
                                                        : kFp16Bytes);

  // 在量化场景pertensor情况，需要在L1上占有一块和l0C上n方向相同大小的uint64数据类型的空间存放数据
  quant_status = run_params->vector_pre_conv_mode ? 4 : 0; // uint64为8字节 / fp16为2字节 = 4

  m_be_even = run_params->trans_a_flag && (dtype_in == ge::DataType::DT_INT8);
  n_be_even = !run_params->trans_b_flag && (dtype_in == ge::DataType::DT_INT8);
  k_tile = MathUtil::Align(min(BASIC_BLOCK_TILE[bb_idx][TILE_K_IDX] * DB_SIZE / dtype_size, run_params->ori_shape_k),
                           cube_k);
  m_tile = MathUtil::Align(min(BASIC_BLOCK_TILE[bb_idx][TILE_M_IDX], run_params->ori_shape_m),
                           !m_be_even ? kBlockSize : 2 * kBlockSize); // 2: align to even
  n_tile = MathUtil::Align(min(BASIC_BLOCK_TILE[bb_idx][TILE_N_IDX], run_params->ori_shape_n),
                           !n_be_even ? kBlockSize : 2 * kBlockSize); // 2: align to even

  int64_t a_inner_size = (run_params->trans_a_flag ? run_params->ori_shape_m : run_params->ori_shape_k) * dtype_size;
  int64_t b_inner_size = (run_params->trans_b_flag ? run_params->ori_shape_k : run_params->ori_shape_n) * dtype_size;
  a_eff = CalcEff(a_inner_size);
  b_eff= CalcEff(b_inner_size);
  eff_ratio = a_eff / b_eff;
  a_eff *= run_params->ori_shape_m;
  b_eff *= run_params->ori_shape_n;

  result_info.m_dim = MathUtil::CeilDivision(run_params->ori_shape_m, m_tile);
  result_info.n_dim = MathUtil::CeilDivision(run_params->ori_shape_n, n_tile);

  m_divided = (run_params->m * kBlockSize) % result_info.m_dim == 0;
  n_divided = (run_params->n * kBlockSize) % result_info.n_dim == 0;
  k_aligned = (run_params->k * cube_k * dtype_size) % align_size == 0;
  align_cube_k = (dtype_in == ge::DataType::DT_FLOAT && (run_params->trans_a_flag || !run_params->trans_b_flag));
  reserve_l1 = (run_params->ori_shape_k % kBlockSize != 0 && align_cube_k);
  // FP16in FP32out need to k align to compare template
  must_divided = (dtype_in == ge::DataType::DT_FLOAT16 && dtype_out == ge::DataType::DT_FLOAT &&
      (run_params->trans_a_flag || !run_params->trans_b_flag)) ;

  double tmp_threshold = 1.0 * PlatformInfo::GetInstance().l2_size / (run_params->ori_shape_k * dtype_size);
  sweet_point_bound = sweet_low;
  active_bound = GetActiveLowThreshold();
  if ((a_inner_size % min_full_dw_size != 0 || run_params->ori_shape_m > tmp_threshold) &&
      (b_inner_size % min_full_dw_size != 0 || run_params->ori_shape_n > tmp_threshold)) {
      sweet_point_bound = sweet_up;
      active_bound = GetActiveUpThreshold();
  }
  InitBMMCondition(tmp_threshold);
  available_core_num = max(1L, PlatformInfo::GetInstance().core_num / batch_cut);
  is_l2_fit = tmp_threshold >= batch_cut * min(run_params->ori_shape_m, run_params->ori_shape_n);
  return true;
}
}

