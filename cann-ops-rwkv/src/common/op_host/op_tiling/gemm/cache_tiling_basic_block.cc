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
 * \file cache_tiling_basic_block.cc\
 * \brief function of cache tiling basic block method
 */

#include "cache_tiling_basic_block.h"
#include "cache_tiling_basic_block_calc.h"
#include "cache_tiling_basic.h"
#include "gemm/estimate/cache_tiling_est.h"
#include "gemm/estimate/cache_tiling_est_mgr.h"

#include "cache_tiling.h"
#include "cube/algorithm/hash/tiling_cache.h"

using namespace std;
using namespace optiling;
// using optiling::cachetiling::MathUtil;
using namespace gemm_cache_tiling;

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

namespace {
inline bool CheckDtype(ge::DataType dtype_A, ge::DataType dtype_B, ge::DataType dtype_C) {
  if (dtype_A != dtype_B) {
    return false;
  }
  // support fp16->fp16  fp32->fp32
  if (dtype_A == ge::DataType::DT_FLOAT16 && dtype_C == ge::DataType::DT_FLOAT16) {
    return true;
  }
  // support batchmatmulfixpipe A16W16 -> int8
  if (dtype_A == ge::DataType::DT_FLOAT16 && dtype_B == ge::DataType::DT_FLOAT16 && dtype_C == ge::DataType::DT_INT8) {
    return true;
  }
  if (dtype_A == ge::DataType::DT_BF16 && dtype_C == ge::DataType::DT_BF16) {
    return true;
  }
  if (dtype_A == ge::DataType::DT_FLOAT && dtype_C == ge::DataType::DT_FLOAT) {
    return true;
  }
  if (dtype_A == ge::DataType::DT_INT8 && dtype_C == ge::DataType::DT_FLOAT16) {
    return true;
  }
  return false;
}

bool CheckBasicParam(const string &op_type, const BatchmatmulParas &params) {
  ge::DataType dtype_A = static_cast<ge::DataType>(params.run_params->dtype_a);
  ge::DataType dtype_B = static_cast<ge::DataType>(params.run_params->dtype_b);
  ge::DataType dtype_C = static_cast<ge::DataType>(params.run_params->dtype_out);
  if (!CheckDtype(dtype_A, dtype_B, dtype_C)) {
    OP_LOGD(op_type.c_str(), "gemm cache tiling basic_block calculator unsupport dtype %d, %d, %d",
        dtype_A, dtype_B, dtype_C);
    return false;
  }
  if (!PlatformInfo::GetInstance().support_l0c2out() || PlatformInfo::GetInstance().support_fix_pipe_l0c2ub()) {
    OP_LOGD(op_type.c_str(), "gemm cache tiling basic_block calculator unsupport hardware");
    return false;
  }
  if (!(params.run_params->nd_flag || params.run_params->weight_nz_flag) || !params.run_params->format_out_nd) {
    OP_LOGD(op_type.c_str(), "gemm cache tiling basic_block calculator only support nd %d, %d, %d, %ld",
        params.run_params->format_a_nd, params.run_params->format_b_nd, params.run_params->format_out_nd,
        params.run_params->ori_shape_m);
    return false;
  }
  return true;
}

bool CheckParam(const string &op_type, const BatchmatmulParas &params) {
  if (!CheckBasicParam(op_type, params)) {
    return false;
  }
  ge::DataType dtype_A = static_cast<ge::DataType>(params.run_params->dtype_a);
  ge::DataType dtype_C = static_cast<ge::DataType>(params.run_params->dtype_out);
  if (dtype_A == ge::DataType::DT_INT8 && dtype_C == ge::DataType::DT_FLOAT16) {
    return true;
  }
  const int32_t fp16_dsize = 2;
  const int32_t fp32_dsize = 4;
  const int32_t min_support_shape = 256;
  const int32_t mode_effi = 32;
  int32_t dsize = dtype_A == ge::DataType::DT_FLOAT ? fp32_dsize : fp16_dsize;
  if ((params.run_params->ori_shape_m > min_support_shape) &&
      (params.run_params->ori_shape_k > min_support_shape) &&
      (params.run_params->ori_shape_n > min_support_shape) &&
      (params.run_params->ori_shape_k * (params.run_params->ori_shape_m +  params.run_params->ori_shape_n) *
        DB_SIZE * dsize / PlatformInfo::GetInstance().core_num > PlatformInfo::GetInstance().l1_size)) {
        bool mode_m = params.run_params->ori_shape_m * dsize % mode_effi == 0;
        bool mode_k = params.run_params->ori_shape_k * dsize % mode_effi == 0;
        bool mode_n = params.run_params->ori_shape_n * dsize % mode_effi == 0;
        bool mode_A = params.run_params->trans_a_flag ? mode_m : mode_k;
        bool mode_B = params.run_params->trans_b_flag ? mode_k : mode_n;
        if (mode_A && mode_B) {
          return true;
        }
  }
  OP_LOGD(op_type.c_str(), "gemm cache tiling basic_block calculator unsupport size %ld, %ld, %ld",
      params.run_params->ori_shape_m, params.run_params->ori_shape_k, params.run_params->ori_shape_n);
  return false;
}

void UpdateCoreStatus(const GemmResultInfo &result, optiling::CoreStatus &coreStatus,
                      optiling::SingleCoreStatus &singleCoreStatus) {
  coreStatus.batch_dim = result.batch_dim;
  coreStatus.m_dim = result.m_dim;
  coreStatus.n_dim = result.n_dim;
  coreStatus.k_dim = result.k_dim;
  singleCoreStatus.l1Status.db_al1 = result.db_al1;
  singleCoreStatus.l1Status.db_bl1 = result.db_bl1;
  singleCoreStatus.l1Status.m_l1 = result.m_l1;
  singleCoreStatus.l1Status.kal1_16 = result.kal1_16;
  singleCoreStatus.l1Status.kbl1_16 = result.kbl1_16;
  singleCoreStatus.l1Status.n_l1 = result.n_l1;
  singleCoreStatus.l0Status.m_l0 = result.m_l0;
  singleCoreStatus.l0Status.k_l0 = result.k_l0;
  singleCoreStatus.l0Status.n_l0 = result.n_l0;
  singleCoreStatus.l0Status.batch_l0 = result.batch_l0;
  singleCoreStatus.l0Status.db_l0c = result.db_l0c;
  singleCoreStatus.l0Status.db_l0a = kDbOn;
  singleCoreStatus.l0Status.db_l0b = kDbOn;
}
}

namespace optiling {
GenTilingStatus GenTilingFromBasicBlock(const string &op_type, BatchmatmulParas &params,
                                        CoreStatus &coreStatus,  SingleCoreStatus &singleCoreStatus)
{
  if (!CheckParam(op_type, params)) {
    return GEN_TILING_EAGAIN;
  }
  OP_LOGD(op_type.c_str(), "start gemm cache tiling basic_block calculator");

  GemmBBCalculator bb_calc(&params);
  auto est = GemmEstimateFactory::GetEstimate(BASIC_BLOCK_ESTIMATE_TYPE, op_type, &params);
  int32_t cur_idx = 0;
  auto bb_idx_tmp = PlatformInfo::GetInstance().support_l12bt_bf16() ? BB_IDX_1982 : BB_IDX;
  // m小于16的增量图场景下，使用增量图基本块
  if (op_type == "QuantBatchMatmulV3" and params.run_params->ori_shape_m <= 16) {
    bb_idx_tmp = BB_IDX_QUANT_BMM_V3;
  }
  for (auto i : bb_idx_tmp) {
    if (!bb_calc.Init(i)) {
      OP_LOGD(op_type.c_str(), "gemm cache tiling basic_block init, unsupport idx:%d", i);
      continue;
    }
    if (!bb_calc.BasicBlockFit()) {
      OP_LOGD(op_type.c_str(), "gemm cache tiling basic_block fit, unsupport idx:%d", i);
      continue;
    }
    const GemmResultInfo& result = bb_calc.GetResultInfo();
    OP_LOGD(op_type.c_str(), "gemm gen tiling basic_block: cur_idx %d, idx:%d, "
        "batch_dim:%ld, m_dim:%ld, k_dim:%ld, n_dim:%ld, batch_l0:%ld, m_l0:%ld, k_l0:%ld, n_l0:%ld, "
        "kal1_16:%ld, kbl1_16:%ld, m_l1:%ld, n_l1:%ld",
        cur_idx, i, result.batch_dim, result.m_dim, result.k_dim, result.n_dim, result.batch_l0, result.m_l0,
        result.k_l0, result.n_l0, result.kal1_16, result.kbl1_16, result.m_l1, result.n_l1);
    bool is_super_dim = result.batch_dim * result.m_dim * result.k_dim * result.n_dim >
                        PlatformInfo::GetInstance().core_num;
    if ((params.run_params->pad_flag != 0 || params.run_params->nz_fusion_flag != 0) && is_super_dim) {
      continue;
    }
    cur_idx++;
    est->AddEstimateTask(result);
  }
  GemmResultInfo best_result;
  if (est->GetEstimateResult(best_result) == false) {
    return GEN_TILING_EAGAIN;
  }
  int32_t best_idx_ = est->GetBestIdx();
  OP_LOGD(op_type.c_str(), "end gemm cache tiling basic_block best cur_idx:%d", best_idx_);
  UpdateCoreStatus(best_result, coreStatus, singleCoreStatus);
  params.run_params->pattern_flag = true;

  return GEN_TILING_EOF;
}

void GenTuningFromBasicBlock(const string &op_type, BatchmatmulParas &params,
      list<CoreStatus> &coreStatusList, list<SingleCoreStatus> &singleCoreStatusList)
{
  CoreStatus coreStatus = coreStatusList.front();
  SingleCoreStatus singleCoreStatus = singleCoreStatusList.front();
  coreStatusList.pop_front();
  singleCoreStatusList.pop_front();
  if (!CheckBasicParam(op_type, params)) {
    return;
  }
  OP_LOGD(op_type.c_str(), "start gemm cache tiling basic_block calculator");

  GemmBBCalculator bb_calc(&params);
  for (auto i : BB_IDX) {
    CoreStatus tmpCoreStatus = coreStatus;
    SingleCoreStatus tmpSingleCoreStatus = singleCoreStatus;
    if (!bb_calc.Init(i)) {
      OP_LOGD(op_type.c_str(), "gemm cache tiling basic_block init, unsupport idx:%d", i);
      continue;
    }
    if (!bb_calc.BasicBlockFit()) {
      OP_LOGD(op_type.c_str(), "gemm cache tiling basic_block fit, unsupport idx:%d", i);
      continue;
    }
    const GemmResultInfo& result = bb_calc.GetResultInfo();
    UpdateCoreStatus(result, tmpCoreStatus, tmpSingleCoreStatus);
    coreStatusList.emplace_back(tmpCoreStatus);
    singleCoreStatusList.emplace_back(tmpSingleCoreStatus);
  }
  return;
}
}
