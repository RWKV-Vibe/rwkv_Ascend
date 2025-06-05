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
 * \file cube_run_info.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_RUN_INFO_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_RUN_INFO_H_

#include <cstdint>

#include "cube/include/cube_tiling.h"
#include "cube/include/cube_tiling_param.h"

namespace optiling {
struct Conv2dBpFilterRunInfo {
  void Update(const cachetiling::Conv2DBpFilterTilingParam &params, const cachetiling::Conv2DBpFilterTiling &tiling);

private:
  void SetShape(const cachetiling::Conv2DBpFilterTilingParam &params);
  void SetTiling(const cachetiling::Conv2DBpFilterTilingParam &params, const cachetiling::Conv2DBpFilterTiling &tiling);
  void SetSpecialScene(const cachetiling::Conv2DBpFilterTilingParam &params);

public:
  // transdata vars
  int32_t batch;
  int32_t ci;
  int32_t hi;
  int32_t wi;
  int32_t co;
  int32_t ho;
  int32_t wo;
  int32_t kh;
  int32_t kw;
  // dw vars
  int32_t ci1;
  int32_t co1;
  int32_t stride_h;
  int32_t stride_w;
  int32_t pad_u;
  int32_t pad_d;
  int32_t pad_l;
  int32_t pad_r;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t groups;
  int32_t cin1_g;
  int32_t cout1_g;
  int32_t real_g;
  int32_t mag_factor;
  int32_t hf32_flag;
  // tiling vars
  int32_t group_dim;
  int32_t batch_dim;
  int32_t h_dim;
  int32_t batch_single_core;
  int32_t n_single_core;
  int32_t n_dim;
  int32_t n_bl1;
  int32_t n_ub_l0_time;
  int32_t cub_n1;
  int32_t m_dim;
  int32_t m_single_core;
  int32_t m_al1;
  int32_t m_l0;
  int32_t k_l0;
  int32_t k_al1_factor;
  int32_t k_bl1_factor;
  int32_t k_al0_factor;
  int32_t k_bl0_factor;
  int32_t k_al1_16;
  int32_t k_bl1_16;
  int32_t kl1_times;
  int32_t bl1_bound;
  int32_t m_aub;
  int32_t n_bub;
  int32_t k_aub;
  int32_t k_bub;
  int32_t wi_bub;
  int32_t ho_bl1;
  int32_t multi_n_ub_l1;
  int32_t multi_m_ub_l1;
  int32_t multi_k_aub_l1;
  int32_t multi_k_bub_l1;
  int32_t db_al1;
  int32_t db_bl1;
  int32_t db_l0c;
  // special scene
  int32_t load3d_special;
  int32_t is_bf16;
};

struct Conv3DRunInfo {
  void Update(const cachetiling::Conv3DTilingParam &params, const cachetiling::Conv3DTiling &tiling);

private:
  void SetShape(const cachetiling::Conv3DTilingParam &params);
  void SetTiling(const cachetiling::Conv3DTilingParam &params, const cachetiling::Conv3DTiling &tiling);
  void SetSpecialScene(const cachetiling::Conv3DTilingParam &params);

public:
  // shape vars
  int32_t batch_n;
  int32_t fmap_c;
  int32_t fmap_d;
  int32_t fmap_h;
  int32_t fmap_w;
  int32_t c_out;
  int32_t d_out;
  int32_t h_out;
  int32_t w_out;
  int32_t kernel_d;
  int32_t kernel_h;
  int32_t kernel_w;
  int32_t fmap_c1;
  int32_t c1_out;

  // attr vars
  int32_t stride_d;
  int32_t stride_h;
  int32_t stride_w;
  int32_t pad_f;
  int32_t pad_b;
  int32_t pad_u;
  int32_t pad_d;
  int32_t pad_l;
  int32_t pad_r;
  int32_t dilation_d;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t cin1_g;
  int32_t cout1_g;
  int32_t real_g;
  int32_t mag_factor;
  int32_t hf32_flag;

  // tiling vars
  int32_t group_dim;
  int32_t batch_dim;
  int32_t d_dim;
  int32_t batch_dout_single_core;
  int32_t n_single_core;
  int32_t n_dim;
  int32_t n_bl1;
  int32_t n_ub_l0_time;
  int32_t cub_n1;
  int32_t m_dim;
  int32_t m_single_core;
  int32_t m_al1;
  int32_t m_l0;
  int32_t k_l0;
  int32_t k_al1_factor;
  int32_t k_bl1_factor;
  int32_t k_al0_factor;
  int32_t k_bl0_factor;
  int32_t kl1_times;
  int32_t al1_bound;
  int32_t bt_bound;
  // special scene
  int32_t load3d_special;
};

struct Conv3dBpFilterRunInfo {
  void Update(const cachetiling::Conv3DBpFilterTilingParam &params, const cachetiling::Conv3DBpFilterTiling &tiling);

private:
  void SetShape(const cachetiling::Conv3DBpFilterTilingParam &params);
  void SetTiling(const cachetiling::Conv3DBpFilterTilingParam &params, const cachetiling::Conv3DBpFilterTiling &tiling);
  void SetSpecialScene(const cachetiling::Conv3DBpFilterTilingParam &params);

public:
  int32_t batch;
  int32_t ci;
  int32_t di;
  int32_t hi;
  int32_t wi;
  int32_t co;
  int32_t dout;
  int32_t ho;
  int32_t wo;
  int32_t kd;
  int32_t kh;
  int32_t kw;
  int32_t ci1;
  int32_t co1;
  int32_t stride_d;
  int32_t stride_h;
  int32_t stride_w;
  int32_t pad_f;
  int32_t pad_b;
  int32_t pad_u;
  int32_t pad_d;
  int32_t pad_l;
  int32_t pad_r;
  int32_t dilation_d;
  int32_t dilation_h;
  int32_t dilation_w;
  int32_t cin1_g;
  int32_t cout1_g;
  int32_t real_g;
  int32_t mag_factor;

  // tiling vars
  int32_t group_dim;
  int32_t batch_dim;
  int32_t h_dim;
  int32_t batch_dout_single_core;
  int32_t n_single_core;
  int32_t n_dim;
  int32_t n_bl1;
  int32_t n_ub_l0_time;
  int32_t cub_n1;
  int32_t m_dim;
  int32_t m_single_core;
  int32_t m_al1;
  int32_t m_l0;
  int32_t k_l0;
  int32_t k_al1_factor;
  int32_t k_bl1_factor;
  int32_t k_al0_factor;
  int32_t k_bl0_factor;
  int32_t kl1_times;
  int32_t bl1_bound;
  int32_t ho_bl1;
  // special scene
  int32_t load3d_special;
  int32_t is_bf16;
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_RUN_INFO_H_
