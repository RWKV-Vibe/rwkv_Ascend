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
 * \file cache_tiling_param.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_TILING_PARAM_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_TILING_PARAM_H_

#include <cstdint>
#include <string>

#include "cube/platform/platform_info.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
namespace cachetiling {
using gert::TilingContext;
using gert::CompileTimeTensorDesc;
using gert::StorageShape;

const size_t kConv2DPadsDim = 4;
const size_t kConv3DPadsDim = 6;
const size_t kConv2DOriShapeDim = 4;
const size_t kConv3DOriShapeDim = 5;

// 新增算子：1. 需添加在kOpTypeNum之前，否则会出现地址越界问题 2. 在kOpType2Str中同步添加对应的opName
enum OpType : size_t {
  kConv2DBackpropFilter,
  kConv2DBackpropInput,
  kConv2DTranspose,
  kDepthwiseConv2DBackpropInput,
  kGemm,
  kConv3D,
  kConv3DBackpropFilter,
  kConv3DBackpropInput,
  kConv3DTranspose,
  kConv2DBackpropFilterV2,
  kConv3DBackpropFilterV2,
  kOpTypeNum,
};

struct FactorConfig
{
  int32_t max;
  int32_t min;
  int32_t step;
};


class Shape {
 public:
  std::string ToString() const;
  bool operator==(const Shape &param) const;

  int64_t batch = 1;
  int64_t c = 16;
  int64_t d = 1;
  int64_t h = 1;
  int64_t w = 1;
  int64_t c1 = 1;
  int64_t c0 = 16;
};

class Conv3DCommonTilingParam;
class CubeTilingParam {
 public:
  friend class Conv3DCommonTilingParam;
  explicit CubeTilingParam(OpType cube_type);
  virtual ~CubeTilingParam() {}
  virtual std::string ToString() const;
  virtual bool IsValid() const;
  virtual bool Fp32Input() const { return false; }
  bool ParseOpInfo(const TilingContext *context, const CubeCompileInfo &compile_info);

 protected:
  virtual bool SetShape(const TilingContext *context);
  virtual bool SetAttrs(const TilingContext *context);
  virtual void SetSpecialSceneParams() {}
  virtual bool CheckGEMMLimit() const { return true; }
  virtual bool IsExtraInfoValid() const { return true; }

  virtual bool CheckFormat(const TilingContext * /* context */) const { return true; }
  virtual bool SetStrides(const TilingContext *context, const int64_t *strides, size_t dim_num);
  virtual bool SetPads(const TilingContext *context, const int64_t *pads, size_t dim_num);
  virtual bool SetDilations(const int64_t *dilations, size_t dim_num);
  virtual void SetDtype(ge::DataType /* fmap_dtype */, ge::DataType /* filter_dtype */, ge::DataType /* y_dtype */) {}
  virtual void SetDtypeWithBias(ge::DataType /* fmap_dtype */, ge::DataType /* filter_dtype */, ge::DataType /* y_dtype */,
                                ge::DataType /* in_bias_dtype */) {}
  virtual void SetC0(const TilingContext * /* context */) {}
  virtual bool CalcGroupsParams(const char *op_name);

  virtual const CompileTimeTensorDesc *GetFmapTensorDesc(const TilingContext * /* context */) const { return nullptr; }
  virtual const CompileTimeTensorDesc *GetFilterTensorDesc(const TilingContext * /* context */) const { return nullptr; }
  virtual const CompileTimeTensorDesc *GetYTensorDesc(const TilingContext * /* context */) const { return nullptr; }
  virtual const CompileTimeTensorDesc *GetBiasTensorDesc(const TilingContext * /* context */) const { return nullptr; }

  virtual const StorageShape *GetFmapTensorShape(const TilingContext * /* context */) const { return nullptr; }
  virtual const StorageShape *GetFilterTensorShape(const TilingContext * /* context */) const { return nullptr; }
  virtual const StorageShape *GetYTensorShape(const TilingContext * /* context */) const { return nullptr; }

  virtual Shape &fmap() { return a_shape; }
  virtual Shape &filter() { return b_shape; }
  virtual Shape &y() { return c_shape; }

  virtual size_t GetStridesIdx() const { return 0; }
  virtual size_t GetPadsIdx() const { return 1; }
  virtual size_t GetDialtionsIdx() const {
    return 2; // 2: dilation idx
  }
  virtual size_t GetGroupsIdx() const {
    return 3; // 3: groups idx
  }
  virtual size_t GetPaddingIdx() const { return std::numeric_limits<uint64_t>::max(); }
  virtual size_t GetPrecisionModeIdx() const { return std::numeric_limits<uint64_t>::max(); }

  virtual size_t GetValidOriShapeDimNum() const { return kConv2DOriShapeDim; }
  virtual size_t GetValidStridesDimNum() const { return kConv2DOriShapeDim; }
  virtual size_t GetValidPadsDimNum() const { return kConv2DPadsDim; }
  virtual size_t GetValidDilationsDimNum() const { return kConv2DOriShapeDim; }

 private:
  bool SetOneShapeTensor(const int64_t *normalized_shape, size_t dim_num, Shape &shape);
  void UpdateInstrictionParam();

 public:
  OpType type;
  const char *op_type = nullptr;
  PlatformInfo platform_info;
  // ori shape of a/b/c
  Shape a_shape;
  Shape b_shape;
  Shape c_shape;
  int32_t bias_flag = 0;
  int32_t groups = 1;
  int32_t real_g = 1;
  int32_t mag_factor = 1;
  int32_t k0 = 16;
  int32_t stride_h = 1;
  int32_t stride_w = 1;
  int32_t kernel_h = 1;
  int32_t kernel_w = 1;
  int32_t dilation_h = 1;
  int32_t dilation_w = 1;
  int32_t kernel_h_dilation = 1;
  int32_t kernel_w_dilation = 1;
  int32_t pad_u = 0;
  int32_t pad_d = 0;
  int32_t pad_l = 0;
  int32_t pad_r = 0;
  int32_t aub_fused_num = 0;
  int32_t bub_fused_num = 0;
  int32_t cub_fused_num = 0;
  ge::DataType a_dtype = ge::DT_FLOAT16;
  ge::DataType b_dtype = ge::DT_FLOAT16;
  ge::DataType c_dtype = ge::DT_FLOAT16;
  int32_t a_dtype_bytes = 2;
  int32_t b_dtype_bytes = 2;
  int32_t c_dtype_bytes = 2;
  int32_t binary_mode = 1;
  int32_t load3d_special = 1;
  bool conv1d_flag = false;
  bool split_w_flag = false;
  bool load2d_flag = false;
  uint32_t strideh_read_flag = 0;
  int32_t hf32_flag = 0;
  int32_t zero_flag = 0;
  bool dma_flag = false;
  bool load3d_flag = false;
  bool linear_embedding_opti_flag = false;
};

class Conv2DBpFilterTilingParam : public CubeTilingParam {
 public:
  explicit Conv2DBpFilterTilingParam(OpType cube_type) : CubeTilingParam(cube_type) {}
  ~Conv2DBpFilterTilingParam() override {}

  bool Fp32Input() const override { return b_dtype == ge::DT_FLOAT; }
  bool IsLoad3dWSplitMode();
  bool IsL0bDmaCopyMode();

  // in strideh read scene, fmaph and strideh will be recalculated,
  // but the original values will still be saved for later use
  int32_t sr_fmap_h = 1;
  int32_t sr_stride_h = 1;
  bool c04_flag = false;
  bool static_flag = false;
  bool depthwise = false;

 protected:
  virtual bool IsStrideHRead() const;
  void SetSpecialSceneParams() override;
  bool CheckGEMMLimit() const override;
  bool IsExtraInfoValid() const override;

  bool CheckFormat(const TilingContext *context) const override;
  void SetDtype(ge::DataType fmap_dtype, ge::DataType filter_dtype, ge::DataType y_dtype) override;
  void SetC0(const TilingContext *context) override;

  const CompileTimeTensorDesc *GetFmapTensorDesc(const TilingContext *context) const override {
    return context->GetInputDesc(0);
  }
  const CompileTimeTensorDesc *GetFilterTensorDesc(const TilingContext *context) const override {
    return context->GetOutputDesc(0);
  }
  const CompileTimeTensorDesc *GetYTensorDesc(const TilingContext *context) const override {
    return context->GetInputDesc(2); // 2: out_backprop idx
  }

  const StorageShape *GetFmapTensorShape(const TilingContext *context) const override {
    return context->GetInputShape(0);
  }
  const StorageShape *GetFilterTensorShape(const TilingContext *context) const override {
    return context->GetOutputShape(0);
  }
  const StorageShape *GetYTensorShape(const TilingContext *context) const override {
    return context->GetInputShape(2); // 2: out_backprop idx
  }

  Shape &fmap() override { return b_shape; }
  Shape &filter() override { return c_shape; }
  Shape &y() override { return a_shape; }

  size_t GetPadsIdx() const override {
    return depthwise ? 2 : 1; // depthwise pads idx is 2
  }
  size_t GetDialtionsIdx() const override {
    return depthwise ? 1: 2; // depthwise dw dilation idx is 1, dw is 2
  }
  size_t GetGroupsIdx() const override {
    // depthwise dw has no groups, dw groups idx is 3
    return depthwise ? std::numeric_limits<uint64_t>::max(): 3;
  }
  size_t GetPaddingIdx() const override {
    return depthwise ? 4: 5; // depthwise dw padding idx is 4, dw is 5
  }
  size_t GetPrecisionModeIdx() const override {
    return depthwise ? 5: 7; // depthwise dw hf32 idx is 5, dw is 7
  }

 private:
  bool IsLoad3dValid() const;
  bool CheckMinL1Size() const;
  bool CanEnableSplitW() const;
  bool IsLoad2dMode() const;
  int32_t IsLoad3dSpecial() const;
  bool IsConv1d();
  bool IsUseLoad3dFlag() const;
  void SetStrideHReadFlag();
  bool IsLinearEmbeddingOptiMode();
};

class Conv2DBpInputTilingParam : public CubeTilingParam {
 public:
  explicit Conv2DBpInputTilingParam(OpType cube_type) : CubeTilingParam(cube_type) {}
  ~Conv2DBpInputTilingParam() override {}
  std::string ToString() const override;
  void StrideOptimize();

 public:
  int32_t filter_h_dilation = 0;
  int32_t filter_w_dilation = 0;
  int32_t stride_expand_flag = 0;
  bool dx_no_overlap_condition_1 = false;
  bool dx_no_overlap_condition_2 = false;
  int32_t g_extend = 0;
  int32_t co1g = 0;
  int32_t ci1g = 0;
  int32_t filter_co0 = 16;  // co0 in fractal_z
  int32_t filter_ci0 = 16;  // cin0 in fractal_z
  int32_t co1g_reduce = 0;  // co1g calculated by block_reduce depend on dtype
  // 0 表示默认情况, hw 轴合并, 作为 m 轴处理; 1 表示切 w 场景, h、w 轴分开描述, w 轴作为 m 轴并且可切
  int32_t split_axis_mode = 0;
};

class GemmTilingParam : public CubeTilingParam {
 public:
  explicit GemmTilingParam(OpType cube_type) :
      CubeTilingParam(cube_type) {}
  ~GemmTilingParam() override {}
  bool is_batch_matmul_op = false;
  bool bias_flag = false;
  bool unaligned_flag = false;
  bool trans_a_flag;
  bool trans_b_flag;
  bool nd_flag;
  bool b_have_batch;
  bool performance_flag;
  bool split_k_flag;
  bool use_pre_ub;
  bool at_l1_flag;
  bool pattern_flag;
  float aub_double_num;
  float bub_double_num;
  bool format_out_nd;
  bool do_not_multi_batch;
  bool m_quant_check;
  bool n_quant_check;
  bool binary_mode_flag;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t batch = 1;
  int32_t m_mapped = 1;
  int32_t k_mapped = 1;
  int32_t n_mapped = 1;
  int64_t ori_shape_m = 1;
  int64_t ori_shape_k = 1;
  int64_t ori_shape_n = 1;
  int32_t batch_mapped = 1;
  int32_t block_size = 16;
  int32_t reduce_block_size = 32;
  ge::DataType bias_dtype = ge::DT_FLOAT16;
  int32_t bias_dtype_bytes = 2;
  int32_t fused_double_operand_num = 0;
};

class Conv3DCommonTilingParam {
 public:
  Conv3DCommonTilingParam() = default;
  virtual ~Conv3DCommonTilingParam() = default;
  virtual std::string ToString() const;

  bool CheckConv3DFormat(const TilingContext *context, const CubeTilingParam *params) const;
  bool SetConv3DPads(const TilingContext *context, const int64_t *pads, size_t dim_num, CubeTilingParam *params);

  int32_t stride_d = 1;
  int32_t dilation_d = 1;
  int32_t kernel_d_dilation = 1;
  int32_t pad_f = 0;
  int32_t pad_b = 0;
  int64_t kernel_d = 1;
  bool pad_greater_than_filter = false;
};

class Conv3DTilingParam : public CubeTilingParam, public Conv3DCommonTilingParam {
 public:
  explicit Conv3DTilingParam(OpType cube_type) : CubeTilingParam(cube_type) {}
  ~Conv3DTilingParam() override = default;
  std::string ToString() const override;
  bool CheckCanLoadBL1() const;
  int64_t CalcMinAL1Size() const;
  bool Fp32Input() const override { return a_dtype == ge::DT_FLOAT; }

  ge::DataType bias_dtype = ge::DT_FLOAT16;
  int32_t bias_dtype_bytes = 2;

 protected:
  bool CheckGEMMLimit() const override;
  bool IsExtraInfoValid() const override;

  bool CheckFormat(const TilingContext *context) const override { return CheckConv3DFormat(context, this); }
  bool SetStrides(const TilingContext *context, const int64_t *strides, size_t dim_num) override;
  bool SetPads(const TilingContext *context, const int64_t *pads, size_t dim_num) override {
    return SetConv3DPads(context, pads, dim_num, this);
  }
  bool SetDilations(const int64_t *dilations, size_t dim_num) override;
  void SetDtype(ge::DataType fmap_dtype, ge::DataType filter_dtype, ge::DataType y_dtype) override;
  void SetDtypeWithBias(ge::DataType fmap_dtype, ge::DataType filter_dtype, ge::DataType y_dtype,
                ge::DataType in_bias_dtype) override;
  bool CalcGroupsParams(const char *op_name) override;

  const CompileTimeTensorDesc *GetFmapTensorDesc(const TilingContext *context) const override {
    return context->GetInputDesc(0);
  }
  const CompileTimeTensorDesc *GetFilterTensorDesc(const TilingContext *context) const override {
    return context->GetInputDesc(1);
  }
  const CompileTimeTensorDesc *GetYTensorDesc(const TilingContext *context) const override {
    return context->GetOutputDesc(0);
  }
  const CompileTimeTensorDesc *GetBiasTensorDesc(const TilingContext *context) const override {
    return context->GetInputDesc(2); // 2: bias idx
  }

  const StorageShape *GetFmapTensorShape(const TilingContext *context) const override {
    return context->GetInputShape(0);
  }
  const StorageShape *GetFilterTensorShape(const TilingContext *context) const override {
    return context->GetInputShape(1);
  }
  const StorageShape *GetYTensorShape(const TilingContext *context) const override {
    return context->GetOutputShape(0);
  }

  Shape &fmap() override { return a_shape; }
  Shape &filter() override { return b_shape; }
  Shape &y() override { return c_shape; }

  size_t GetPaddingIdx() const override {
    return 6; // 6: padding idx
  }

  size_t GetPrecisionModeIdx() const override {
    return 7; // 7: _op_impl_mode_enum idx, for hf32
  }

  size_t GetValidOriShapeDimNum() const override { return kConv3DOriShapeDim; }
  size_t GetValidStridesDimNum() const override { return kConv3DOriShapeDim; }
  size_t GetValidPadsDimNum() const override { return kConv3DPadsDim; }
  size_t GetValidDilationsDimNum() const override { return kConv3DOriShapeDim; }

 private:
  bool IsLoad3dValid() const;
  int64_t CalcMinBL1Size() const;
  int64_t CalcMinBiasL1Size() const;
};

class Conv3DBpFilterTilingParam : public Conv2DBpFilterTilingParam, public Conv3DCommonTilingParam {
 public:
  explicit Conv3DBpFilterTilingParam(OpType cube_type) : Conv2DBpFilterTilingParam(cube_type) {}
  ~Conv3DBpFilterTilingParam() override {}
  std::string ToString() const override;

 protected:
  bool IsStrideHRead() const override { return false; }
  void SetSpecialSceneParams() override;
  bool CheckGEMMLimit() const override;
  bool IsExtraInfoValid() const override;
  bool CheckFormat(const TilingContext *context) const override { return CheckConv3DFormat(context, this); }
  bool SetStrides(const TilingContext *context, const int64_t *strides, size_t dim_num) override;
  bool SetPads(const TilingContext *context, const int64_t *pads, size_t dim_num) override {
    return SetConv3DPads(context, pads, dim_num, this);
  }

  bool SetDilations(const int64_t *dilations, size_t dim_num) override;

  size_t GetPaddingIdx() const override {
    return 5; // 5: padding idx
  }

  size_t GetPrecisionModeIdx() const override {
    return 6; // 6: _op_impl_mode_enum idx, for conv3ddw hf32
  }

  size_t GetValidOriShapeDimNum() const override { return kConv3DOriShapeDim; }
  size_t GetValidStridesDimNum() const override { return kConv3DOriShapeDim; }
  size_t GetValidPadsDimNum() const override { return kConv3DPadsDim; }
  size_t GetValidDilationsDimNum() const override { return kConv3DOriShapeDim; }
};


class Conv3DBpInputTilingParam : public CubeTilingParam {
 public:
  explicit Conv3DBpInputTilingParam(OpType cube_type) : CubeTilingParam(cube_type) {}
  ~Conv3DBpInputTilingParam() override {}
  std::string ToString() const override;

  int32_t split_axis_mode = 0;
  int32_t stride_expand_flag = 0;
  int32_t dilation_d_gt_one_flag = 0;
  int32_t stride_d = 1;
  int32_t dilation_d = 1;
  int64_t filter_d_dilation = 1;
  int64_t filter_h_dilation = 1;
  int64_t filter_w_dilation = 1;
  int32_t pad_h = 0;
  int32_t pad_t = 0;
  int32_t backprop_pad_h = 0;
  int32_t backprop_pad_t = 0;
  int32_t backprop_pad_u = 0;
  int32_t backprop_pad_d = 0;
  int32_t backprop_pad_l = 0;
  int32_t backprop_pad_r = 0;
  int32_t co1g = 0;
  int32_t ci1g = 0;
  int32_t filter_co0 = 16;  // co0 in fractal_z
  int32_t filter_ci0 = 16;  // cin0 in fractal_z
  int32_t co1g_reduce = 0;  // co1g calculated by block_reduce depend on dtype
  int32_t load3d_special = 0;
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_INCLUDE_CUBE_TILING_PARAM_H_
