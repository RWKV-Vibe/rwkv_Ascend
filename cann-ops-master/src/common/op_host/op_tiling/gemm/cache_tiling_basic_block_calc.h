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
 * \file cache_tiling_basic_block_calc.h
 * \brief function of cache tiling basic block calculator
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_BLOCK_CALC_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_BLOCK_CALC_H

#include "cache_tiling.h"
#include "gemm/common/cache_tiling_common.h"

namespace gemm_cache_tiling {

using CANDIDATES = set<array<int64_t, 2>>; // 2: m_cut, n_cut

enum FULL_LOAD_STATE {
  FULL_LOAD_NONE,
  FULL_LOAD_A,
  FULL_LOAD_B,
  FULL_LOAD_BOTH
};

struct GemmCandidatesLoadInfo {
  double bb_counts;
  double active_ratio;
  int64_t core_diff;
  double full_score;
  int64_t min_k;
  FULL_LOAD_STATE load_state;
};

struct GemmCandidatesInfo {
  int64_t m_cut;
  int64_t n_cut;
  int64_t m_dim;
  int64_t m_tile;
  int64_t n_dim;
  int64_t n_tile;
};

class GemmBBCalculator {
public:
  explicit GemmBBCalculator(const optiling::BatchmatmulParas *paras) :
    compile_params(paras->compile_params),
    run_params(paras->run_params) {};

  ~GemmBBCalculator() = default;

public:
  bool Init(int32_t bb_idx);
  bool BasicBlockFit();
  GemmResultInfo GetResultInfo() { return this->result_info; };

private:
  void UpdateLoadInfo(const GemmCandidatesInfo &info, GemmCandidatesLoadInfo &loadInfo, int64_t core_num);
  void UpdateCandidatesdInfo(GemmCandidatesInfo &info, GemmCandidatesLoadInfo &loadInfo);
  void AdjustDimTile(int64_t core_num, double cur_AI, double in_AI);
  void AdjustTile(int64_t &dim, int64_t &tile, int64_t ori, int64_t cut, bool be_even);
  void AdjustDimByDivided(bool &batchDivided);
  void AdjustKCut();
  int64_t CalculatorKl0ByDivided(int64_t &a_k_tile, int64_t &b_k_tile, int64_t k_l0_max);
  void CalculateKTileFullNone(int64_t &a_k_tile, int64_t &b_k_tile);
  void CalculateAFullKTile(int64_t &a_k_tile, int64_t &b_k_tile, int64_t bias_status, int64_t bw_type_size);
  void CalculateBFullKTile(int64_t &a_k_tile, int64_t &b_k_tile, int64_t bias_status, int64_t bw_type_size);
  int64_t GetAvaliableL1Size(int64_t t_m_tile, int64_t t_n_tile) const;
  int64_t CalculateKTile(int64_t &a_k_tile, int64_t &b_k_tile);
  double CalcEff(int64_t inner_size) const;
  CANDIDATES GenCandidates(int64_t core_num);
  template<typename T>
  bool NeedUpadateCandidatesLoadInfo(const T& last_info, const T& new_info, double cur_AI, double in_AI);
  void InitBMMCondition(double tmp_threshold);
  void UpdateBMMResult();

private:
  const optiling::BatchmatmulCompileParas * const compile_params;
  const optiling::BatchmatmulRunParas * const run_params;
  GemmResultInfo result_info;
  GemmCandidatesLoadInfo load_info;

  double a_eff{0};
  double b_eff{0};
  double eff_ratio{0};

  int64_t k_tile{1};
  int64_t m_tile{1};
  int64_t n_tile{1};
  int64_t k_cut{1};
  int64_t batch_cut{1};
  int64_t available_core_num{1};

  ge::DataType dtype_in;
  ge::DataType dtype_out;
  int64_t dtype_size{1};
  int64_t out_dtype_size{1};
  int64_t cube_k{16};
  int64_t k_per_core_tmp{1};
  int64_t k_per_core{1};
  int64_t min_batch_cut{1};
  int64_t max_batch_cut{1};
  int64_t quant_status{0};
  int64_t min_full_dw_size{512}; // 512 默认block size
  int64_t align_size{256}; // 256 默认对齐Byte
  double alpha{0};
  double sweet_up{1.0};
  double sweet_low{1.0};
  double sweet_point_bound{1.0};
  double active_bound{0};

  bool align_cube_k{false};
  bool must_divided{false};
  bool m_divided{false};
  bool n_divided{false};
  bool k_aligned{false};
  bool reserve_l1{false};
  bool is_adjust{false};
  bool is_l2_fit{false};
  bool m_be_even{false};
  bool n_be_even{false};
  bool divided{false};
};
}
#endif

