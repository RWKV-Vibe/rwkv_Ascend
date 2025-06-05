/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_DX_TUNING_TILING_H_
#define CONV2D_DX_TUNING_TILING_H_

#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"
namespace tuningtiling {
#pragma pack(push, 1)
struct Conv2DDxInputArgs {
  ge::DataType a_dtype;
  ge::DataType b_dtype;
  ge::DataType c_dtype;
  ge::DataType bias_dtype;
  int64_t a_shape_n;
  int64_t a_shape_h;
  int64_t a_shape_w;
  int64_t b_shape_n;
  int64_t b_shape_c;
  int64_t b_shape_h;
  int64_t b_shape_w;
  int64_t c_shape_h;
  int64_t c_shape_w;
  ge::Format a_format;
  ge::Format b_format;
  ge::Format c_format;
  int64_t groups;
  int64_t stride_expand_h;
  int64_t stride_expand_w;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t pad_u;
  int64_t pad_d;
  int64_t pad_l;
  int64_t pad_r;
  bool bias_flag;
  bool hf32_flag;
  float cub_double_num;
  float fused_double_operand_num;
  int64_t reserved_params1;
  int64_t reserved_params2;
  int64_t reserved_params3;
  int64_t reserved_params4;
  int64_t reserved_params5;
};
#pragma pack(pop)

BEGIN_TUNING_TILING_DEF(Conv2DDxTunnerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, group_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, batch_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_al1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_bl1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_aub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_aub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, wo_aub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_l0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_l0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_cub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_l0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_al1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_bl1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, al1_bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, bl1_bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, aub_bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, bias_table_bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, tiling_id);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Conv2DDxTunnerTiling,
  FIELD(Conv2DDxTunnerTiling, group_dim),
  FIELD(Conv2DDxTunnerTiling, batch_dim),
  FIELD(Conv2DDxTunnerTiling, n_dim),
  FIELD(Conv2DDxTunnerTiling, m_dim),
  FIELD(Conv2DDxTunnerTiling, m_al1),
  FIELD(Conv2DDxTunnerTiling, n_bl1),
  FIELD(Conv2DDxTunnerTiling, k_aub),
  FIELD(Conv2DDxTunnerTiling, m_aub),
  FIELD(Conv2DDxTunnerTiling, wo_aub),
  FIELD(Conv2DDxTunnerTiling, m_l0),
  FIELD(Conv2DDxTunnerTiling, n_l0),
  FIELD(Conv2DDxTunnerTiling, n_cub),
  FIELD(Conv2DDxTunnerTiling, k_l0),
  FIELD(Conv2DDxTunnerTiling, k_al1),
  FIELD(Conv2DDxTunnerTiling, k_bl1),
  FIELD(Conv2DDxTunnerTiling, al1_bound),
  FIELD(Conv2DDxTunnerTiling, bl1_bound),
  FIELD(Conv2DDxTunnerTiling, aub_bound),
  FIELD(Conv2DDxTunnerTiling, bias_table_bound),
  FIELD(Conv2DDxTunnerTiling, tiling_id));
}  // namespace tuningtiling

#endif
