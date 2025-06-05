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
 * \file tiling_cache.cc
 * \brief function of tiling_cache
 */
#include "tiling_cache.h"

namespace optiling {
namespace cachetiling {
MatmulHashInput::MatmulHashInput(const BatchmatmulCompileParas &compile_params, const BatchmatmulRunParas &run_params) {
  bit_field_.binary_mode_flag = compile_params.binary_mode_flag;
  bit_field_.bias_flag = compile_params.bias_flag;
  bit_field_.at_l1_flag = compile_params.at_l1_flag;
  bit_field_.split_k_flag = compile_params.split_k_flag;
  bit_field_.nd_flag = run_params.nd_flag;
  bit_field_.trans_a_flag = run_params.trans_a_flag;
  bit_field_.trans_b_flag = run_params.trans_b_flag;
  bit_field_.format_a_nd = run_params.format_a_nd;
  bit_field_.format_b_nd = run_params.format_b_nd;
  bit_field_.b_have_batch = run_params.b_have_batch;
  bit_field_.is_batch_matmul_mode = run_params.is_batch_matmul_mode;
  bit_field_.is_batch_matmul_op = run_params.is_batch_matmul_op;
  bit_field_.used_aligned_pattern = run_params.used_aligned_pattern;
  bit_field_.non_factor_k = run_params.non_factor_k;
  bit_field_.non_factor_bmn = run_params.non_factor_bmn;
  bit_field_.performance_flag = run_params.performance_flag;
  bit_field_.pad_flag = run_params.pad_flag;
  bit_field_.nz_fusion_flag = run_params.nz_fusion_flag;
  bit_field_.vector_pre_conv_mode = run_params.vector_pre_conv_mode;
  bit_field_.reserved = 0;
  bit_field_.hf32_flag = run_params.hf32_flag;
  fused_double_operand_num_ = compile_params.fused_double_operand_num;
  aub_double_num_ = compile_params.aub_double_num;
  bub_double_num_ = compile_params.bub_double_num;
  m_ori_ = run_params.ori_shape_m;
  n_ori_ = run_params.ori_shape_n;
  k_ori_ = run_params.ori_shape_k;
  m_pad_ = run_params.m_pad;
  k_pad_ = run_params.k_pad;
  n_pad_ = run_params.n_pad;
  batch_a1_ = run_params.batch_a1;
  batch_a2_ = run_params.batch_a2;
  batch_a3_ = run_params.batch_a3;
  batch_a4_ = run_params.batch_a4;
  batch_b1_ = run_params.batch_b1;
  batch_b2_ = run_params.batch_b2;
  batch_b3_ = run_params.batch_b3;
  batch_b4_ = run_params.batch_b4;
  a_dtype_ = run_params.dtype_a;
  b_dtype_ = run_params.dtype_b;
  out_dtype_ = run_params.dtype_out;
  eltwise_src_ = compile_params.eltwise_src;
}

std::string MatmulHashInput::ToString() const {
  std::stringstream ss;
  ss << "fused_double_operand_num_: " << fused_double_operand_num_
    << " aub_double_num_: " <<  aub_double_num_
    << " bub_double_num_: " << bub_double_num_
    << " m_ori_: " << m_ori_
    << " n_ori_: " <<  n_ori_
    << " k_ori_: " << k_ori_
    << " m_pad_: " << m_pad_
    << " k_pad_: " << k_pad_
    << " n_pad_: " << n_pad_
    << " batch_a1_: " << batch_a1_
    << " batch_a2_: " << batch_a2_
    << " batch_a3_: " << batch_a3_
    << " batch_a4_: " << batch_a4_
    << " batch_b1_: " <<  batch_b1_
    << " batch_b2_: " << batch_b2_
    << " batch_b3_: " << batch_b3_
    << " batch_b4_: " << batch_b4_
    << " a_dtype_: " << a_dtype_
    << " b_dtype_: " << b_dtype_
    << " out_dtype_: " << out_dtype_
    << " eltwise_src_: " << eltwise_src_
    << " binary_mode_flag: " << bit_field_.binary_mode_flag
    << " bias_flag: " << bit_field_.bias_flag
    << " at_l1_flag: " << bit_field_.at_l1_flag
    << " split_k_flag: " << bit_field_.split_k_flag
    << " nd_flag: " << bit_field_.nd_flag
    << " trans_a_flag: " << bit_field_.trans_a_flag
    << " trans_b_flag: " << bit_field_.trans_b_flag
    << " format_a_nd: " << bit_field_.format_a_nd
    << " format_b_nd: " << bit_field_.format_b_nd
    << " b_have_batch: " << bit_field_.b_have_batch
    << " is_batch_matmul_mode: " << bit_field_.is_batch_matmul_mode
    << " is_batch_matmul_op: " << bit_field_.is_batch_matmul_op
    << " used_aligned_pattern: " << bit_field_.used_aligned_pattern
    << " non_factor_k: " << bit_field_.non_factor_k
    << " non_factor_bmn: " << bit_field_.non_factor_bmn
    << " performance_flag: " << bit_field_.performance_flag
    << " pad_flag: " << bit_field_.pad_flag
    << " nz_fusion_flag: " << bit_field_.nz_fusion_flag
    << " vector_pre_conv_mode: " << bit_field_.vector_pre_conv_mode
    << " reserved: " << bit_field_.reserved;
  return ss.str();
};

Conv2DBpTilingHashParam::Conv2DBpTilingHashParam(const CubeTilingParam &param)
    : a_shape_(param.a_shape),
      b_shape_(param.b_shape),
      c_shape_(param.c_shape),
      pad_u_(static_cast<uint16_t>(param.pad_u)),
      pad_d_(static_cast<uint16_t>(param.pad_d)),
      pad_l_(static_cast<uint16_t>(param.pad_l)),
      pad_r_(static_cast<uint16_t>(param.pad_r)),
      groups_(static_cast<uint16_t>(param.groups)),
      stride_h_(static_cast<uint8_t>(param.stride_h)),
      stride_w_(static_cast<uint8_t>(param.stride_w)),
      kernel_h_(static_cast<uint8_t>(param.kernel_h)),
      kernel_w_(static_cast<uint8_t>(param.kernel_w)),
      aub_fused_num_(static_cast<uint8_t>(param.aub_fused_num)),
      bub_fused_num_(static_cast<uint8_t>(param.bub_fused_num)),
      cub_fused_num_(static_cast<uint8_t>(param.cub_fused_num)),
      a_dtype_(static_cast<uint8_t>(param.a_dtype)),
      b_dtype_(static_cast<uint8_t>(param.b_dtype)),
      c_dtype_(static_cast<uint8_t>(param.c_dtype)),
      binary_mode_(static_cast<uint8_t>(param.binary_mode)),
      load3d_special_(static_cast<uint8_t>(param.load3d_special)) {}

HashShape::HashShape(const Shape &shape) {
  batch = static_cast<int32_t>(shape.batch);
  c = static_cast<int32_t>(shape.c);
  h = static_cast<int32_t>(shape.h);
  w = static_cast<int32_t>(shape.w);
};

bool HashShape::operator==(const HashShape &param) const {
  if (&param == this) {
    return true;
  }

  return memcmp(this, &param, sizeof(param)) == 0;
}

bool Conv2DBpTilingHashParam::operator==(const Conv2DBpTilingHashParam &param) const {
  if (&param == this) {
    return true;
  }

  return memcmp(this, &param, sizeof(param)) == 0;
}

Conv2DBpInputHashParam::Conv2DBpInputHashParam(const CubeTilingParam &param)
    : Conv2DBpTilingHashParam(param), bias_flag_(static_cast<uint8_t>(param.bias_flag)) {}

bool Conv2DBpInputHashParam::operator==(const Conv2DBpInputHashParam &param) const {
  if (&param == this) {
    return true;
  }

  return memcmp(this, &param, sizeof(param)) == 0;
}

Conv3DTilingHashParam::Conv3DTilingHashParam(const Conv3DTilingParam &param)
    : a_shape_(param.a_shape),
      cout_(param.c_shape.c),
      kernel_d_(param.kernel_d),
      kernel_h_(static_cast<uint16_t>(param.kernel_h)),
      kernel_w_(static_cast<uint16_t>(param.kernel_w)),
      pad_f_(static_cast<uint16_t>(param.pad_f)),
      pad_b_(static_cast<uint16_t>(param.pad_b)),
      pad_u_(static_cast<uint16_t>(param.pad_u)),
      pad_d_(static_cast<uint16_t>(param.pad_d)),
      pad_l_(static_cast<uint16_t>(param.pad_l)),
      pad_r_(static_cast<uint16_t>(param.pad_r)),
      groups_(static_cast<uint16_t>(param.groups)),
      bias_flag_(static_cast<uint16_t>(param.bias_flag)),
      stride_d_(static_cast<uint8_t>(param.stride_d)),
      stride_h_(static_cast<uint8_t>(param.stride_h)),
      stride_w_(static_cast<uint8_t>(param.stride_w)),
      dilation_d_(static_cast<uint8_t>(param.dilation_d)),
      dilation_h_(static_cast<uint8_t>(param.dilation_h)),
      dilation_w_(static_cast<uint8_t>(param.dilation_w)),
      a_dtype_(static_cast<uint8_t>(param.a_dtype)),
      b_dtype_(static_cast<uint8_t>(param.b_dtype)),
      c_dtype_(static_cast<uint8_t>(param.c_dtype)),
      bias_dtype_(static_cast<uint8_t>(param.bias_dtype)),
      load3d_special_(static_cast<uint8_t>(param.load3d_special)) {}

bool Conv3DTilingHashParam::operator==(const Conv3DTilingHashParam &param) const {
  if (&param == this) {
    return true;
  }

  return memcmp(this, &param, sizeof(param)) == 0;
}

Conv3DBpFilterHashParam::Conv3DBpFilterHashParam(const Conv3DBpFilterTilingParam &param)
    : Conv2DBpTilingHashParam(param) {
  fmap_d_ = static_cast<int32_t>(param.b_shape.d);
  dedy_d_ = static_cast<int32_t>(param.a_shape.d);

  kernel_d_ = static_cast<int32_t>(param.kernel_d);

  pad_f_ = static_cast<uint16_t>(param.pad_f);
  pad_b_ = static_cast<uint16_t>(param.pad_b);
  stride_d_ = static_cast<uint8_t>(param.stride_d);
  dilation_d_ = static_cast<uint8_t>(param.dilation_d);
}

bool Conv3DBpFilterHashParam::operator==(const Conv3DBpFilterHashParam &param) const {
  if (&param == this) {
    return true;
  }

  return memcmp(this, &param, sizeof(param)) == 0;
}

Conv3DBpInputHashParam::Conv3DBpInputHashParam(const Conv3DBpInputTilingParam &param) {
  a_shape_ = param.a_shape;
  b_shape_ = param.b_shape;
  c_shape_ = param.c_shape;
  pad_h_ = static_cast<uint16_t>(param.pad_h);
  pad_t_ = static_cast<uint16_t>(param.pad_t);
  pad_u_ = static_cast<uint16_t>(param.pad_u);
  pad_d_ = static_cast<uint16_t>(param.pad_d);
  pad_l_ = static_cast<uint16_t>(param.pad_l);
  pad_r_ = static_cast<uint16_t>(param.pad_r);
  groups_ = static_cast<uint16_t>(param.groups);
  stride_d_ = static_cast<uint8_t>(param.stride_d);
  stride_h_ = static_cast<uint8_t>(param.stride_h);
  stride_w_ = static_cast<uint8_t>(param.stride_w);
  dilation_d_ = static_cast<uint8_t>(param.dilation_d);
  dilation_h_ = static_cast<uint8_t>(param.dilation_h);
  dilation_w_ = static_cast<uint8_t>(param.dilation_w);
  aub_fused_num_ = static_cast<uint8_t>(param.aub_fused_num);
  bub_fused_num_ = static_cast<uint8_t>(param.bub_fused_num);
  cub_fused_num_ = static_cast<uint8_t>(param.cub_fused_num);
  a_dtype_ = static_cast<uint8_t>(param.a_dtype);
  b_dtype_ = static_cast<uint8_t>(param.b_dtype);
  c_dtype_ = static_cast<uint8_t>(param.c_dtype);
  load3d_special_ = static_cast<uint8_t>(param.load3d_special);
}

bool Conv3DBpInputHashParam::operator==(const Conv3DBpInputHashParam &param) const {
  if (&param == this) {
    return true;
  }

  return memcmp(this, &param, sizeof(param)) == 0;
}

template class TilingCache<Conv2DBpInputHashParam, Conv2DBpInputHashItem>;
template class TilingCache<Conv2DBpTilingHashParam, Conv2DBpFilterHashItem>;
template class TilingCache<MatmulHashInput, MatmulHashItem>;
template class TilingCache<Conv3DTilingHashParam, Conv3DHashItem>;
template class TilingCache<Conv3DBpFilterHashParam, Conv3DBpFilterHashItem>;
template class TilingCache<Conv3DBpInputHashParam, Conv3DBpInputHashItem>;
}  // namespace cachetiling
}  // namespace optiling

