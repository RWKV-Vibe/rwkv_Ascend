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
 * \file cube_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_TILING_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_TILING_H_

#include <cstdint>
#include <string>

#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
namespace cachetiling {
class CubeTiling {
 public:
  virtual ~CubeTiling() = default;
  virtual std::string ToString() const;
  virtual bool IsValid() const;
  virtual void UpdateRunInfoCube(gert::TilingContext *context) const;

  int32_t tiling_id = 0;
  int32_t batch_dim = 1;
  int32_t group_dim = 1;
  int32_t m_dim = 1;
  int32_t k_dim = 1;
  int32_t n_dim = 1;

  int32_t m_l0 = 1;
  int32_t k_l0 = 1;
  int32_t n_l0 = 1;
  int32_t db_l0c = 1;
  int32_t m_al1 = 1;
  int32_t k_al1 = 1;
  int32_t n_al1 = 1;
  int32_t db_al1 = 1;
  int32_t al1_bound = 1;
  int32_t db_bl1 = 1;
  int32_t ho_bl1 = 1;
  int32_t n_bl1 = 1;
  int32_t k_bl1 = 1;
  int32_t bl1_bound = 1;
  int32_t m_aub = 1;
  int32_t k_aub = 1;
  int32_t k_bub = 1;
  int32_t n_bub = 1;
  int32_t n_cub = 1;
  int32_t db_aub = 1;
  int32_t db_bub = 1;
  int32_t db_cub = 1;
};

class Conv2DBpFilterTiling : public CubeTiling {
 public:
  virtual ~Conv2DBpFilterTiling() = default;
  std::string ToString() const override;
  bool IsValid() const override;

  int32_t wi_bub = 1;
};

class Conv3DTiling : public CubeTiling {
 public:
  virtual ~Conv3DTiling() = default;
  std::string ToString() const override;
  void UpdateRunInfoCube(gert::TilingContext *context) const override;

  int32_t d_dim = 1;
};

using Conv3DBpFilterTiling = Conv3DTiling;

class Conv2DBpInputTiling : public CubeTiling {
 public:
  ~Conv2DBpInputTiling() override = default;
  std::string ToString() const override;
  int32_t wo_aub = 1;
  int32_t aub_bound = 0;
  int32_t bias_table_bound = 0;
  bool simply_loop_mn_from_sc_to_l0_is_1 = false;
  bool co1g_ci1g_is_1 = false;
  bool min_kl1_div_kl0_is_1 = false;
};

class Conv3DBpInputTiling : public CubeTiling {
 public:
  ~Conv3DBpInputTiling() override = default;
  std::string ToString() const override;

  int32_t wo_aub = 1;
  int32_t aub_bound = 1;
  int32_t n_l0_div_ub = 1;
  int32_t d_dim = 1;
  int32_t d_al1 = 1;
  int32_t d_bl1 = 1;
  int32_t d_al0 = 1;
  int32_t d_bl0 = 1;
  int32_t d_cl0 = 1;
};

class GemmTiling : public CubeTiling {
 public:
  virtual ~GemmTiling() = default;
  std::string ToString() const override;
  int32_t kal1_16 = 1;
  int32_t kbl1_16 = 1;
  int32_t kal1_factor = 1;
  int32_t kbl1_factor = 1;
  int32_t k_org_dim = 1;
  int32_t batch_l0 = 1;
  int32_t batch_aub = 1;
  int32_t batch_bub = 1;
  int32_t batch_cub = 1;
  int32_t out_branch_flag = 1;
  int32_t bias_flag = 0;
  int32_t aub_multi_flag = 0;
  int32_t bub_multi_flag = 0;
  int32_t a_align_value = 1;
  int32_t b_align_value = 1;
  int32_t aub_align_bound = 0;
  int32_t bub_align_bound = 0;
  int32_t min_kl1_cmp_kl0 = 0;
  int32_t al1_attach_flag = 0;
  int32_t bl1_attach_flag = 0;
  int32_t abkl1_attach_flag = 0;
  int32_t l0c_multi_batch = 0;
  int32_t m_single_core = 1;
  int32_t n_single_core = 1;
  bool flag_cub_solving_bank_conflict = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  int8_t hf32_flag = 1;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_TILING_H_
