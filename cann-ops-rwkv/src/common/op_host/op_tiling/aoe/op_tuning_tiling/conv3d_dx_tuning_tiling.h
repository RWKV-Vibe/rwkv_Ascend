/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3D_DX_TUNING_TILING_H_
#define CONV3D_DX_TUNING_TILING_H_

#include "register/tuning_bank_key_registry.h"
#include "register/tuning_tiling_registry.h"
namespace tuningtiling {
#pragma pack(push, 1)
struct Conv3DDxInputArgs {
  ge::DataType a_dtype;
  ge::DataType b_dtype;
  ge::DataType c_dtype;
  int64_t a_shape_n;
  int64_t a_shape_d;
  int64_t a_shape_h;
  int64_t a_shape_w;
  int64_t b_shape_n;
  int64_t b_shape_c;
  int64_t b_shape_d;
  int64_t b_shape_h;
  int64_t b_shape_w;
  int64_t c_shape_d;
  int64_t c_shape_h;
  int64_t c_shape_w;
  ge::Format a_format;
  ge::Format b_format;
  ge::Format c_format;
  int64_t groups;
  int64_t stride_expand_d;
  int64_t stride_expand_h;
  int64_t stride_expand_w;
  int64_t dilation_d;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t pad_h;
  int64_t pad_t;
  int64_t pad_u;
  int64_t pad_d;
  int64_t pad_l;
  int64_t pad_r;
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

BEGIN_TUNING_TILING_DEF(Conv3DDxTunnerTiling)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, group_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, batch_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, d_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_dim);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_al1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_bl1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, d_al1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, d_bl1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_aub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_aub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, wo_aub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, m_l0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_l0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, d_al0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, d_bl0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, d_cl0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, n_cub);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_l0);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_al1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, k_bl1);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, al1_bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, bl1_bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, aub_bound);
TUNING_TILING_DATA_FIELD_DEF(uint32_t, tiling_id);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(Conv3DDxTunnerTiling,
  FIELD(Conv3DDxTunnerTiling, group_dim),
  FIELD(Conv3DDxTunnerTiling, batch_dim),
  FIELD(Conv3DDxTunnerTiling, d_dim),
  FIELD(Conv3DDxTunnerTiling, n_dim),
  FIELD(Conv3DDxTunnerTiling, m_dim),
  FIELD(Conv3DDxTunnerTiling, m_al1),
  FIELD(Conv3DDxTunnerTiling, n_bl1),
  FIELD(Conv3DDxTunnerTiling, d_al1),
  FIELD(Conv3DDxTunnerTiling, d_bl1),
  FIELD(Conv3DDxTunnerTiling, k_aub),
  FIELD(Conv3DDxTunnerTiling, m_aub),
  FIELD(Conv3DDxTunnerTiling, wo_aub),
  FIELD(Conv3DDxTunnerTiling, m_l0),
  FIELD(Conv3DDxTunnerTiling, n_l0),
  FIELD(Conv3DDxTunnerTiling, d_al0),
  FIELD(Conv3DDxTunnerTiling, d_bl0),
  FIELD(Conv3DDxTunnerTiling, d_cl0),
  FIELD(Conv3DDxTunnerTiling, n_cub),
  FIELD(Conv3DDxTunnerTiling, k_l0),
  FIELD(Conv3DDxTunnerTiling, k_al1),
  FIELD(Conv3DDxTunnerTiling, k_bl1),
  FIELD(Conv3DDxTunnerTiling, al1_bound),
  FIELD(Conv3DDxTunnerTiling, bl1_bound),
  FIELD(Conv3DDxTunnerTiling, aub_bound),
  FIELD(Conv3DDxTunnerTiling, tiling_id));
}  // namespace tuningtiling

#endif
