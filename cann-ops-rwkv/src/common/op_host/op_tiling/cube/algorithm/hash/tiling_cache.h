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
 * \file tiling_cache.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_TILING_CACHE_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_TILING_CACHE_H_

#include <map>

#include "aoe/op_tuning_tiling/conv2d_dx_tuning_tiling.h"
#include "aoe/op_tuning_tiling/conv3d_dx_tuning_tiling.h"
#include "aoe/runtime_kb/runtime_bank_manager.h"
#include "cache_tiling.h"
#include "cube/algorithm/hash/hash.h"
#include "cube/include/cube_cache_tiling.h"
#include "cube/constants/constants_define.h"
#include "lock.h"
#include "../../../mathutil.h"
namespace optiling {
namespace cachetiling {
constexpr uint32_t kMaxTilingCacheEntryNum = 500;

struct MatmulBitField {
  // 4 Bytes aligned
  uint16_t binary_mode_flag : 1;
  uint16_t bias_flag : 1;
  uint16_t at_l1_flag : 1;
  uint16_t split_k_flag : 1;
  uint16_t nd_flag : 1;
  uint16_t trans_a_flag : 1;
  uint16_t trans_b_flag : 1;
  uint16_t format_a_nd : 1;
  uint16_t format_b_nd : 1;
  uint16_t b_have_batch : 1;
  uint16_t is_batch_matmul_mode : 1;
  uint16_t is_batch_matmul_op : 1;
  uint16_t used_aligned_pattern : 1;
  uint16_t non_factor_k : 1;
  uint16_t non_factor_bmn : 1;
  uint16_t performance_flag : 1;
  uint16_t pad_flag : 2;
  uint16_t nz_fusion_flag : 2;
  uint16_t vector_pre_conv_mode : 1;
  uint16_t hf32_flag : 1; 
  uint16_t reserved : 10;
  bool operator==(const MatmulBitField &bit_field) const {
    return (binary_mode_flag == bit_field.binary_mode_flag && bias_flag == bit_field.bias_flag &&
            at_l1_flag == bit_field.at_l1_flag && split_k_flag == bit_field.split_k_flag &&
            nd_flag == bit_field.nd_flag && trans_a_flag == bit_field.trans_a_flag &&
            trans_b_flag == bit_field.trans_b_flag && format_a_nd == bit_field.format_a_nd &&
            format_b_nd == bit_field.format_b_nd && b_have_batch == bit_field.b_have_batch &&
            is_batch_matmul_mode == bit_field.is_batch_matmul_mode && non_factor_k == bit_field.non_factor_k &&
            is_batch_matmul_op == bit_field.is_batch_matmul_op && non_factor_bmn == bit_field.non_factor_bmn &&
            performance_flag == bit_field.performance_flag && used_aligned_pattern == bit_field.used_aligned_pattern &&
            pad_flag == bit_field.pad_flag && nz_fusion_flag == bit_field.nz_fusion_flag &&
            vector_pre_conv_mode == bit_field.vector_pre_conv_mode && hf32_flag == bit_field.hf32_flag);
  }
};

class MatmulHashInput {
 public:
  MatmulHashInput(const BatchmatmulCompileParas &compile_params, const BatchmatmulRunParas &run_params);
  bool operator==(const MatmulHashInput &input) const {
    return (MathUtil::IsEqual(fused_double_operand_num_, input.fused_double_operand_num_) &&
            MathUtil::IsEqual(aub_double_num_, input.aub_double_num_) &&
            MathUtil::IsEqual(bub_double_num_, input.bub_double_num_) &&
            (!bit_field_.nd_flag || (m_ori_ == input.m_ori_ && k_ori_ == input.k_ori_ && n_ori_ == input.n_ori_)) &&
            m_pad_ == input.m_pad_ && k_pad_ == input.k_pad_ && n_pad_ == input.n_pad_ &&
            batch_a1_ == input.batch_a1_ && batch_a2_ == input.batch_a2_ && batch_a3_ == input.batch_a3_ &&
            batch_a4_ == input.batch_a4_ && batch_b1_ == input.batch_b1_ && batch_b2_ == input.batch_b2_ &&
            batch_b3_ == input.batch_b3_ && batch_b4_ == input.batch_b4_ && bit_field_ == input.bit_field_ &&
            a_dtype_ == input.a_dtype_ && b_dtype_ == input.b_dtype_ && out_dtype_ == input.out_dtype_ &&
            eltwise_src_ == input.eltwise_src_);
  }
  std::string ToString() const;

 private:
  float fused_double_operand_num_ = 0;
  float aub_double_num_ = 0;
  float bub_double_num_ = 0;
  int32_t a_dtype_ = 0;
  int32_t b_dtype_ = 0;
  int32_t out_dtype_ = 0;
  int64_t m_ori_ = 1;
  int64_t n_ori_ = 1;
  int64_t k_ori_ = 1;
  int64_t m_pad_ = 0;
  int64_t k_pad_ = 0;
  int64_t n_pad_ = 0;
  int64_t batch_a1_ = 1;
  int64_t batch_a2_ = 1;
  int64_t batch_a3_ = 1;
  int64_t batch_a4_ = 1;
  int64_t batch_b1_ = 1;
  int64_t batch_b2_ = 1;
  int64_t batch_b3_ = 1;
  int64_t batch_b4_ = 1;
  int64_t eltwise_src_ = 0;
  MatmulBitField bit_field_;
  int32_t reserved_ = 0;
};

class MatmulHashItem {
 public:
  MatmulHashItem(const Tiling &tiling, const BatchmatmulRunParas &run_params, const MatmulHashInput &hash_input)
      : input_(hash_input), tiling_(tiling), run_params_(run_params) {}
  const MatmulHashInput &input() const { return input_; }
  const Tiling &tiling() const { return tiling_; }
  const BatchmatmulRunParas &run_params() const { return run_params_; }
  void set_tiling(const Tiling &tiling) { tiling_ = tiling; }
  void set_run_param(const BatchmatmulRunParas &run_param) { run_params_ = run_param; }

 private:
  MatmulHashInput input_;
  Tiling tiling_;
  BatchmatmulRunParas run_params_;
};

class HashShape {
 public:
  explicit HashShape(const Shape &shape);
  bool operator==(const HashShape &param) const;

 private:
  int32_t batch = 1;
  int32_t c = 16;
  int32_t h = 1;
  int32_t w = 1;
};

class Conv2DBpTilingHashParam {
 public:
  explicit Conv2DBpTilingHashParam(const CubeTilingParam &param);
  bool operator==(const Conv2DBpTilingHashParam &param) const;

 private:
  HashShape a_shape_;
  HashShape b_shape_;
  HashShape c_shape_;
  uint16_t pad_u_ = 0;
  uint16_t pad_d_ = 0;
  uint16_t pad_l_ = 0;
  uint16_t pad_r_ = 0;
  uint16_t groups_ = 1;
  uint8_t stride_h_ = 1;
  uint8_t stride_w_ = 1;
  uint8_t kernel_h_ = 1;
  uint8_t kernel_w_ = 1;
  uint8_t aub_fused_num_ = 0;
  uint8_t bub_fused_num_ = 0;
  uint8_t cub_fused_num_ = 0;
  uint8_t a_dtype_;
  uint8_t b_dtype_;
  uint8_t c_dtype_;
  uint8_t binary_mode_ = kBinaryModeNC1HWC0;
  uint8_t load3d_special_ = 1;
};

class Conv2DBpFilterHashItem {
 public:
  Conv2DBpFilterHashItem(const Conv2DBpTilingHashParam &input, const Conv2DBpFilterTiling &tiling,
                         const Conv2dBpFilterRunInfo &run_info)
      : input_(input), tiling_(tiling), run_info_(run_info) {}
  const Conv2DBpTilingHashParam &input() const { return input_; }
  const Conv2DBpFilterTiling &tiling() const { return tiling_; }
  const Conv2dBpFilterRunInfo &run_info() const { return run_info_; }
  void set_input(const Conv2DBpTilingHashParam &input) { input_ = input; }
  void set_tiling(const Conv2DBpFilterTiling &tiling) { tiling_ = tiling; }
  void set_run_info(const Conv2dBpFilterRunInfo &run_info) { run_info_ = run_info; }

 private:
  Conv2DBpTilingHashParam input_;
  Conv2DBpFilterTiling tiling_;
  Conv2dBpFilterRunInfo run_info_;
};

class Conv2DBpInputHashParam : public Conv2DBpTilingHashParam {
 public:
  explicit Conv2DBpInputHashParam(const CubeTilingParam &param);
  bool operator==(const Conv2DBpInputHashParam &param) const;

 private:
  uint8_t bias_flag_;
};

class Conv2DBpInputHashItem {
 public:
  Conv2DBpInputHashItem(const Conv2DBpInputHashParam &input, const Conv2DBpInputTiling &tiling)
      : input_(input), tiling_(tiling) {}
  const Conv2DBpInputHashParam &input() const { return input_; }
  const Conv2DBpInputTiling &tiling() const { return tiling_; }
  void set_input(const Conv2DBpInputHashParam &input) { input_ = input; }
  void set_tiling(const Conv2DBpInputTiling &tiling) { tiling_ = tiling; }

 private:
  Conv2DBpInputHashParam input_;
  Conv2DBpInputTiling tiling_;
};

class Conv3DTilingHashParam {
 public:
  explicit Conv3DTilingHashParam(const Conv3DTilingParam &param);
  bool operator==(const Conv3DTilingHashParam &param) const;

 private:
  Shape a_shape_;
  int32_t cout_;
  int32_t kernel_d_ = 1;
  uint16_t kernel_h_ = 1;
  uint16_t kernel_w_ = 1;
  uint16_t pad_f_ = 0;
  uint16_t pad_b_ = 0;
  uint16_t pad_u_ = 0;
  uint16_t pad_d_ = 0;
  uint16_t pad_l_ = 0;
  uint16_t pad_r_ = 0;
  uint16_t groups_ = 1;
  uint8_t bias_flag_ = 0;
  uint8_t stride_d_ = 1;
  uint8_t stride_h_ = 1;
  uint8_t stride_w_ = 1;
  uint8_t dilation_d_ = 1;
  uint8_t dilation_h_ = 1;
  uint8_t dilation_w_ = 1;
  uint8_t a_dtype_ = 2;
  uint8_t b_dtype_ = 2;
  uint8_t c_dtype_ = 2;
  uint8_t bias_dtype_ = 2;
  uint8_t load3d_special_ = 1;
};

class Conv3DHashItem {
 public:
  Conv3DHashItem(const Conv3DTilingHashParam &input, const Conv3DTiling &tiling,
                         const Conv3DRunInfo &run_info)
      : input_(input), tiling_(tiling), run_info_(run_info) {}
  const Conv3DTilingHashParam &input() const { return input_; }
  const Conv3DTiling &tiling() const { return tiling_; }
  const Conv3DRunInfo &run_info() const { return run_info_; }
  void set_input(const Conv3DTilingHashParam &input) { input_ = input; }
  void set_tiling(const Conv3DTiling &tiling) { tiling_ = tiling; }
  void set_run_info(const Conv3DRunInfo &run_info) { run_info_ = run_info; }

 private:
  Conv3DTilingHashParam input_;
  Conv3DTiling tiling_;
  Conv3DRunInfo run_info_;
};

class Conv3DBpFilterHashParam : public Conv2DBpTilingHashParam {
 public:
  explicit Conv3DBpFilterHashParam(const Conv3DBpFilterTilingParam &param);
  bool operator==(const Conv3DBpFilterHashParam &param) const;

 private:
  uint8_t stride_d_ = 1;
  uint8_t dilation_d_ = 1;
  int32_t pad_f_ = 0;
  int32_t pad_b_ = 0;
  int32_t fmap_d_ = 1;
  int32_t dedy_d_ = 1;
  int32_t kernel_d_ = 1;
};

class Conv3DBpFilterHashItem {
 public:
  Conv3DBpFilterHashItem(const Conv3DBpFilterHashParam &input, const Conv3DBpFilterTiling &tiling,
                         const Conv3dBpFilterRunInfo &run_info)
      : input_(input), tiling_(tiling), run_info_(run_info) {}
  const Conv3DBpFilterHashParam &input() const { return input_; }
  const Conv3DBpFilterTiling &tiling() const { return tiling_; }
  const Conv3dBpFilterRunInfo &run_info() const { return run_info_; }
  void set_input(const Conv3DBpFilterHashParam &input) { input_ = input; }
  void set_tiling(const Conv3DBpFilterTiling &tiling) { tiling_ = tiling; }
  void set_run_info(const Conv3dBpFilterRunInfo &run_info) { run_info_ = run_info; }

 private:
  Conv3DBpFilterHashParam input_;
  Conv3DBpFilterTiling tiling_;
  Conv3dBpFilterRunInfo run_info_;
};

class Conv3DBpInputHashParam {
 public:
  explicit Conv3DBpInputHashParam(const Conv3DBpInputTilingParam &param);
  bool operator==(const Conv3DBpInputHashParam &param) const;

 private:
  Shape a_shape_;
  Shape b_shape_;
  Shape c_shape_;
  uint16_t pad_h_ = 0;
  uint16_t pad_t_ = 0;
  uint16_t pad_u_ = 0;
  uint16_t pad_d_ = 0;
  uint16_t pad_l_ = 0;
  uint16_t pad_r_ = 0;
  uint16_t groups_ = 1;
  uint8_t stride_d_ = 1;
  uint8_t stride_h_ = 1;
  uint8_t stride_w_ = 1;
  uint8_t dilation_d_ = 1;
  uint8_t dilation_h_ = 1;
  uint8_t dilation_w_ = 1;
  uint8_t aub_fused_num_ = 0;
  uint8_t bub_fused_num_ = 0;
  uint8_t cub_fused_num_ = 0;
  uint8_t a_dtype_ = 2;
  uint8_t b_dtype_ = 2;
  uint8_t c_dtype_ = 2;
  uint8_t load3d_special_ = 0;
};

class Conv3DBpInputHashItem {
 public:
  Conv3DBpInputHashItem(const Conv3DBpInputHashParam &input, const Conv3DBpInputTiling &tiling)
      : input_(input), tiling_(tiling) {}
  const Conv3DBpInputHashParam &input() const { return input_; }
  const Conv3DBpInputTiling &tiling() const { return tiling_; }
  void set_input(const Conv3DBpInputHashParam &input) { input_ = input; }
  void set_tiling(const Conv3DBpInputTiling &tiling) { tiling_ = tiling; }

 private:
  Conv3DBpInputHashParam input_;
  Conv3DBpInputTiling tiling_;
};

template <typename HashInput, typename HashItem>
class TilingCache {
 public:
  void Add(uint32_t key, const HashInput &hash_input, const HashItem &value) {
    rwlock_.wrlock();
    if (size_ >= kMaxTilingCacheEntryNum) {
      rwlock_.unlock();
      return;
    }

    if (map_.find(key) != map_.end()) {
      rwlock_.unlock();
      return;
    }

    map_.emplace(key, value);
    size_++;
    rwlock_.unlock();
    return;
  }

  void Replace(uint32_t key, const HashInput &hash_input, const HashItem &value) {
    rwlock_.wrlock();
    if (size_ >= kMaxTilingCacheEntryNum) {
      rwlock_.unlock();
      return;
    }

    if (map_.find(key) == map_.end()) {
      size_++;
    }
    map_.erase(key);
    map_.emplace(key, value);
    rwlock_.unlock();
    return;
  }

  bool Get(uint32_t key, const HashInput &hash_input, HashItem &value) {
    rwlock_.rdlock();
    auto iter = map_.find(key);
    if (iter == map_.end()) {
      rwlock_.unlock();
      return false;
    }
    if (!(hash_input == iter->second.input())) {
      rwlock_.unlock();
      return false;
    }

    value = iter->second;
    rwlock_.unlock();
    return true;
  }

 private:
  std::map<uint32_t, HashItem> map_;
  uint32_t size_ = 0;
  RWLock rwlock_;
};

template <typename Param, typename Tiling, typename RunInfo, typename HashParam, typename HashItem>
bool GetTiling(const Param &params, Tiling &tiling, RunInfo &run_info) {
  static TilingCache<HashParam, HashItem> tiling_cache;
  HashParam hash_param(params);
  uint32_t hash_key = MurmurHash(&hash_param, sizeof(hash_param));
  HashItem hash_value(hash_param, tiling, run_info);
  if (!tiling_cache.Get(hash_key, hash_param, hash_value)) {
    if (!GenTiling(params, tiling)) {
      return false;
    };
    hash_value.set_input(hash_param);
    hash_value.set_tiling(tiling);
    run_info.Update(params, tiling);
    hash_value.set_run_info(run_info);
    tiling_cache.Add(hash_key, hash_param, hash_value);
    OP_LOGI(params.op_type, "Get tiling by calculator. Shape[Backprop[%s], Fmap[%s], Output[%s]], Tiling[%s]",
            params.a_shape.ToString().c_str(), params.b_shape.ToString().c_str(), params.c_shape.ToString().c_str(),
            tiling.ToString().c_str());
  } else {
    tiling = hash_value.tiling();
    run_info = hash_value.run_info();
    OP_LOGI(params.op_type, "Get tiling from hash. Shape[Backprop[%s], Fmap[%s], Output[%s]], Tiling[%s]",
            params.a_shape.ToString().c_str(), params.b_shape.ToString().c_str(), params.c_shape.ToString().c_str(),
            tiling.ToString().c_str());
  }
  return true;
}

bool TransRepoTiling(tuningtiling::TuningTilingDefPtr &repo_tiling, cachetiling::Conv2DBpInputTiling &tiling,
                     gert::TilingContext *context);
bool GetTilingFromRepo(const cachetiling::CubeTilingParam &params, cachetiling::Conv2DBpInputTiling &tiling,
                       gert::TilingContext *context, cachetiling::OpType op_type);

bool TransRepoTiling(tuningtiling::TuningTilingDefPtr &repo_tiling, cachetiling::Conv3DBpInputTiling &tiling,
                     gert::TilingContext *context);
bool GetTilingFromRepo(const cachetiling::CubeTilingParam &params, cachetiling::Conv3DBpInputTiling &tiling,
                       gert::TilingContext *context, cachetiling::OpType op_type);

template <typename Param, typename Tiling, typename HashParam, typename HashItem>
bool GetTiling(const Param &params, Tiling &tiling, gert::TilingContext *context, cachetiling::OpType op_type) {
  static TilingCache<HashParam, HashItem> tiling_cache;
  HashParam hash_param(params);
  uint32_t hash_key = MurmurHash(&hash_param, sizeof(hash_param));
  HashItem hash_value(hash_param, tiling);
  if (!tiling_cache.Get(hash_key, hash_param, hash_value)) {
    if (GetTilingFromRepo(params, tiling, context, op_type)) {
      OP_LOGI(params.op_type,
              "Get tiling by repo. Shape[Backprop[%s], Filter[%s], Output[%s]], Tiling[%s]",
              params.a_shape.ToString().c_str(), params.b_shape.ToString().c_str(), params.c_shape.ToString().c_str(),
              tiling.ToString().c_str());
    } else if (GenTiling(params, tiling)) {
      OP_LOGI(params.op_type,
              "Get tiling by calculator. Shape[Backprop[%s], Filter[%s], Output[%s]], Tiling[%s]",
              params.a_shape.ToString().c_str(), params.b_shape.ToString().c_str(), params.c_shape.ToString().c_str(),
              tiling.ToString().c_str());
    } else {
      // can't get tiling from cache, repo and calculator
      return false;
    }
    hash_value.set_input(hash_param);
    hash_value.set_tiling(tiling);
    tiling_cache.Add(hash_key, hash_param, hash_value);
  } else {
    tiling = hash_value.tiling();
    OP_LOGI(params.op_type, "Get tiling from hash. Shape[Backprop[%s], Filter[%s], Output[%s]], Tiling[%s]",
            params.a_shape.ToString().c_str(), params.b_shape.ToString().c_str(), params.c_shape.ToString().c_str(),
            tiling.ToString().c_str());
  }
  return true;
}

using Conv2dBpTilingHash = TilingCache<Conv2DBpTilingHashParam, Conv2DBpFilterHashItem>;
using Conv2dBpInputTilingHash = TilingCache<Conv2DBpInputHashParam, Conv2DBpInputHashItem>;
using MMTilingHash = TilingCache<MatmulHashInput, MatmulHashItem>;
using Conv3dBpInputTilingHash = TilingCache<Conv3DBpInputHashParam, Conv3DBpInputHashItem>;
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_ALGORITHM_HASH_TILING_CACHE_H_

