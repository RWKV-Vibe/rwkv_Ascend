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
 * \file cache_tiling_basic_block_est.h
 * \brief function of cache tiling basic block estmate
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_BLOCK_EST_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_BASIC_BLOCK_EST_H

#include "cache_tiling.h"
#include "cache_tiling_est.h"
#include "gemm/common/cache_tiling_common.h"
#include "gemm/common/cache_tiling_soc_spec.h"
#include <tuple>


namespace gemm_cache_tiling {
enum GEMM_CUBE_TAG {
  A,
  B,
  C
}GEMM_CUBE_TAG_;

enum ND_NZ_TYPE {
  ND_2_NZ,
  NZ_2_ND
};

struct GemmBBBandWidthInfo {
  double l2_l1A{0};
  double l2_l1B{0};
  double l2_l0A{0};
  double l2_l0B{0};
  double l1_l0A{0};
  double l1_l0B{0};
  double l0C_l2{0};
  double ddr_l1A{0};
  double ddr_l1B{0};
  double ddr_l0A{0};
  double ddr_l0B{0};
};

struct GemmBBCycleInfo {
  double cube_tile_cycle{0};
  double cube_cycle{0};
  double mte2_cycle{0};
  double mte2_tile_cycle{0};
  double mte1_cycle{0};
  double mte1_tile_cycle{0};
  double fix_cycle{0};
  double fix_tile_cycle{0};
};

class GemmBBEstimate final : public GemmEstimate {
public:
  explicit GemmBBEstimate(const string& op_typ, const optiling::BatchmatmulParas *paras);

  ~GemmBBEstimate() override = default;

private:
  void SetBufferParams() override;
  void SetBufferParams(const optiling::CoreStatus &core_status,
      const optiling::SingleCoreStatus &singlecore_status) override;
  void Estimate(int32_t cur_idx) override;
  void ResetParams();
  template<GEMM_CUBE_TAG T>
  void GetDMABiuBw(double &biu_bw, GEMM_CUBE_SIZE cube_size, int32_t latency);
  template<GEMM_CUBE_TAG T>
  double MataBandwidth(int64_t sum, int64_t req_size, int32_t same_count, GEMM_CUBE_SIZE cube_size);
  double MataDDRBandwidth(int64_t req_sum, int64_t align_sum);
  bool EnableKShuffle();
  template<GEMM_CUBE_TAG T>
  double GetSameAddressPenalty(GEMM_CUBE_SIZE cube_size, int32_t same_count, int64_t req_size);
  void EstimateBandwidth();
  void EstimateBandwidthA();
  void EstimateBandwidthB();
  void EstimateBandwidthC();
  void EstimateCycle();
  void EstimateCubeCycle();
  void GetL2HitRate(double &A_hit_rate, double &B_hit_rate);
  void EstimateMte2Cycle();
  void EstimateMte1Cycle();
  void EstimateFixpCycle();
  double EstimatePipe();

private:
  GemmBBBandWidthInfo bw_info_;
  GemmBBCycleInfo cycle_info_;
  GemmSocSpecBandwidth soc_bw_;
  int64_t batch_times_{1};  // single core total batch times
  int64_t batch_patch_{1}; // single core 1 time batch
  int64_t m_patch_{1};
  int64_t k_patch_{1};
  int64_t n_patch_{1};

  int64_t m_{1};
  int64_t k_{1};
  int64_t n_{1};

  int64_t t_l1_M_{1};
  int64_t t_l1_AK_{1};
  int64_t t_l1_BK_{1};
  int64_t t_l1_N_{1};

  int64_t t_l0_M_{1};
  int64_t t_l0_K_{1};
  int64_t t_l0_N_{1};
  int64_t active_core_num_{1};
  int64_t m_run_{1}; // A repeate load cores
  int64_t n_run_{1}; // B repeate load cores
  int64_t k_run_{1};
  int64_t batch_run_{1};

  bool k_shift_{false};
  bool shift_inwards_{false};
  double best_perf_{0};
};
}
#endif
