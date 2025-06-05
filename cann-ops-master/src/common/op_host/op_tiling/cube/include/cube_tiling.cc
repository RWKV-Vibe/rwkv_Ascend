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
 * \file cube_tiling.cc
 * \brief
 */

#include "cube_tiling.h"
#include <sstream>

#define unlikely(x) __builtin_expect((x), 0)
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__)
#define OP_LOGE_IF(condition, return_value, op_name, fmt, ...)                                                 \
  static_assert(std::is_same<bool, std::decay<decltype(condition)>::type>::value, "condition should be bool"); \
  do {                                                                                                         \
    if (unlikely(condition)) {                                                                                 \
      OP_LOGE(op_name, fmt, ##__VA_ARGS__);                                                                    \
      return return_value;                                                                                     \
    }                                                                                                          \
  } while (0)

namespace optiling {
namespace cachetiling {
constexpr char kOpName[] = "Cube";

std::string CubeTiling::ToString() const {
  std::stringstream ss;
  ss << "tiling_id: " << tiling_id
     << " batch_dim: " << batch_dim
     << " group_dim: " << group_dim
     << " m_dim: " << m_dim
     << " k_dim: " << k_dim
     << " n_dim: " << n_dim
     << " m_l0: " << m_l0
     << " k_l0: " << k_l0
     << " n_l0: " << n_l0
     << " db_l0c: " << db_l0c
     << " m_al1: " << m_al1
     << " k_al1: " << k_al1
     << " k_bl1: " << k_bl1
     << " n_bl1: " << n_bl1
     << " db_al1: " << db_al1
     << " db_bl1: " << db_bl1
     << " ho_bl1: " << ho_bl1
     << " al1_bound: " << al1_bound
     << " bl1_bound: " << bl1_bound
     << " m_aub: " << m_aub
     << " k_aub: " << k_aub
     << " k_bub: " << k_bub
     << " n_bub: " << n_bub
     << " n_cub: " << n_cub
     << " db_aub: " << db_aub
     << " db_bub: " << db_bub
     << " db_cub: " << db_cub;
  return ss.str();
}

bool CubeTiling::IsValid() const {
  return true;
}

void CubeTiling::UpdateRunInfoCube(gert::TilingContext *context) const {
  int32_t block_dim = batch_dim * n_dim * k_dim * m_dim * group_dim;
  context->SetBlockDim(static_cast<uint32_t>(block_dim));
  context->SetTilingKey(static_cast<uint64_t>(tiling_id));
}

std::string Conv2DBpFilterTiling::ToString() const {
  std::string base = CubeTiling::ToString();
  std::stringstream ss;
  ss << base << " wi_bub: " << wi_bub;
  return ss.str();
}

bool Conv2DBpFilterTiling::IsValid() const {
  OP_LOGE_IF(k_dim == 0, false, kOpName, "Invalid param, k_dim is 0.");
  OP_LOGE_IF(batch_dim == 0, false, kOpName, "Invalid param, batch_dim is 0.");
  OP_LOGE_IF(n_dim * n_bl1 * n_l0 == 0, false, kOpName, "Invalid param, n_dim * n_bl1 * n_l0 is 0.");
  OP_LOGE_IF(n_cub == 0, false, kOpName, "Invalid param, n_cub is 0.");
  OP_LOGE_IF(m_dim * m_al1 * m_l0 == 0, false, kOpName, "Invalid param, m_dim * m_al1 * m_l0 is 0.");
  OP_LOGE_IF(k_al1 == 0, false, kOpName, "Invalid param, k_al1 is 0");
  OP_LOGE_IF(k_bl1 == 0, false, kOpName, "Invalid param, k_bl1 is 0");
  OP_LOGE_IF(k_l0 == 0, false, kOpName, "Invalid param, k_l0 is 0.");
  OP_LOGE_IF(n_bub == 0, false, kOpName, "Invalid param, n_bub is 0.");
  OP_LOGE_IF(m_aub == 0, false, kOpName, "Invalid param, m_aub is 0.");
  OP_LOGE_IF(k_aub == 0, false, kOpName, "Invalid param, k_aub is 0.");
  OP_LOGE_IF(k_bub == 0, false, kOpName, "Invalid param, k_bub is 0.");
  return true;
}

std::string Conv3DBpInputTiling::ToString() const {
  std::stringstream ss;
  ss << CubeTiling::ToString()
     << " wo_aub: " << wo_aub
     << " aub_bound: " << aub_bound
     << " n_l0_div_ub: " << n_l0_div_ub
     << " d_dim: " << d_dim
     << " d_al1: " << d_al1
     << " d_bl1: " << d_bl1
     << " d_dim: " << d_dim
     << " d_al0: " << d_al1
     << " d_bl0: " << d_dim
     << " d_cl0: " << d_al1;
  return ss.str();
}

std::string Conv2DBpInputTiling::ToString() const {
  std::stringstream ss;
  ss << CubeTiling::ToString()
     << " wo_aub: " << wo_aub
     << " aub_bound: " << aub_bound
     << " bias_table_bound: " << bias_table_bound
     << " simply_loop_mn_from_sc_to_l0_is_1: " << simply_loop_mn_from_sc_to_l0_is_1
     << " co1g_ci1g_is_1: " << co1g_ci1g_is_1
     << " min_kl1_div_kl0_is_1: " << min_kl1_div_kl0_is_1;
  return ss.str();
}

std::string GemmTiling::ToString() const {
  std::string base = CubeTiling::ToString();
  std::stringstream ss;
  ss << base
     << " kal1_16: " << kal1_16
     << " kbl1_16: " << kbl1_16
     << " kal1_factor: " << kal1_factor
     << " kbl1_factor: " << kbl1_factor
     << " k_org_dim: " << k_org_dim
     << " batch_l0: " << batch_l0
     << " batch_aub: " << batch_aub
     << " batch_bub: " << batch_bub
     << " batch_cub: " << batch_cub
     << " out_branch_flag: " << out_branch_flag
     << " bias_flag: " << bias_flag
     << " aub_multi_flag: " << aub_multi_flag
     << " bub_multi_flag: " << bub_multi_flag
     << " a_align_value: " << a_align_value
     << " b_align_value: " << b_align_value
     << " aub_align_bound: " << aub_align_bound
     << " bub_align_bound: " << bub_align_bound
     << " min_kl1_cmp_kl0: " << min_kl1_cmp_kl0
     << " al1_attach_flag: " << al1_attach_flag
     << " bl1_attach_flag: " << bl1_attach_flag
     << " abkl1_attach_flag: " << abkl1_attach_flag
     << " l0c_multi_batch: " << l0c_multi_batch
     << " m_single_core: " << m_single_core
     << " n_single_core: " << n_single_core
     << " flag_cub_solving_bank_conflict: " << flag_cub_solving_bank_conflict
     << " al1_full_load: " << al1_full_load
     << " bl1_full_load: " << bl1_full_load
     << " hf32_flag: " << hf32_flag;
  return ss.str();
}

std::string Conv3DTiling::ToString() const {
  std::string base = CubeTiling::ToString();
  std::stringstream ss;
  ss << base << " d_dim: " << d_dim;
  return ss.str();
}

void Conv3DTiling::UpdateRunInfoCube(gert::TilingContext *context) const {
  int32_t block_dim = batch_dim * n_dim * k_dim * m_dim * group_dim * d_dim;
  context->SetBlockDim(static_cast<uint32_t>(block_dim));
  context->SetTilingKey(static_cast<uint64_t>(tiling_id));
}
}  // namespace cachetiling
}  // namespace optiling
