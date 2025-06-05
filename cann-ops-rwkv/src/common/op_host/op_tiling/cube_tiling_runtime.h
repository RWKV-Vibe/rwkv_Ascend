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
 * \file cube_tiling_runtime.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_

#include <map>
#include <nlohmann/json.hpp>
#include <vector>

#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"

namespace optiling {
#define CUBE_INNER_ERR_REPORT(op_name, err_msg, ...) std::printf(err_msg, ##__VA_ARGS__)
#define OP_TILING_CHECK(cond, log_func, expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      expr;                                   \
    }                                         \
  } while (0)
}  // namespace optiling

namespace optiling {
constexpr uint64_t kInvalidTilingId = std::numeric_limits<uint64_t>::max();

constexpr uint32_t kVarBatchN = 0x0001;  // conv2d dx, conv3d, conv3d dx
constexpr uint32_t kVarDxH = 0x0002;     // conv2d dx
constexpr uint32_t kVarDxW = 0x0004;     // conv2d dx

constexpr uint32_t kVarBatch = 0x0008;  // conv2d dw
constexpr uint32_t kVarFmapH = 0x0010;  // conv2d dw, conv3d
constexpr uint32_t kVarFmapW = 0x0020;  // conv2d dw, conv3d
constexpr uint32_t kVarFmapD = 0x0040;  // conv2d dw, conv3d
constexpr uint32_t kVarDedyD = 0x0080;  // conv3d dx
constexpr uint32_t kVarDedyH = 0x0100;  // conv3d dx
constexpr uint32_t kVarDedyW = 0x0200;  // conv3d dx

const std::map<std::string, uint32_t> kVar2Flag = {{"batch_n", kVarBatchN}, {"dx_h", kVarDxH},
                                                   {"dx_w", kVarDxW},       {"batch", kVarBatch},
                                                   {"fmap_h", kVarFmapH},   {"fmap_w", kVarFmapW},
                                                   {"fmap_d", kVarFmapD},   {"dedy_d", kVarDedyD},
                                                   {"dedy_h", kVarDedyH},   {"dedy_w", kVarDedyW}};

enum CubeTilingType {
  CUBE_DYNAMIC_SHAPE_TILING,
  CUBE_DEFAULT_TILING,
  CUBE_BINARY_TILING,
};

class CubeCompileInfo {
 public:
  CubeCompileInfo() = default;
  virtual ~CubeCompileInfo() = default;

  bool AnalyzeCompileInfo(const char *op_name, const char *compile_info_str);
  bool CheckRangeSize(size_t shape_dim_num) const;
  static void GetChipFeature(fe::PlatFormInfos &platform_info, const string &lable, const string &platform_info_key,
                             const string &true_value, bool &value);
  static void GetLocalMemSize(fe::PlatFormInfos &platform_info, const string &lable, const string &mem_type,
                              uint64_t &size);
  static void GetAICoreIntrinsicDtype(fe::PlatFormInfos &platform_info, const string &intrinsic_name, bool &value);
  void ParseRuntimePlatformInfo(const char *op_name, fe::PlatFormInfos &platform_info);
  uint64_t CheckTilingInRepo(const char *op_name, const int64_t *shape, size_t dim_num, bool conv = false) const;
  uint64_t CheckTilingInCostModel(const char *op_name, const int64_t *shape, size_t dim_num) const;
  uint64_t CheckDefaultTiling(const char *op_name, const int64_t *shape, size_t dim_num) const;
  uint64_t CubeTilingBatch(const char *op_name, const int64_t *shape) const;

  // virtual bool AnalyzeExtendInfo(const nlohmann::json &compile_info) = 0;
  bool AnalyzeCommonCompileInfo(const nlohmann::json &compile_info);

  bool correct_range_flag = false;
  CubeTilingType tiling_type = CUBE_DYNAMIC_SHAPE_TILING;
  uint64_t default_tiling_id = kInvalidTilingId;
  std::vector<int64_t> default_range;
  std::vector<std::vector<int64_t>> repo_seeds;
  std::vector<std::vector<int64_t>> repo_range;
  std::vector<std::vector<int64_t>> cost_range;
  std::vector<std::vector<int64_t>> batch_range;  // for dynamic batch
  std::vector<uint64_t> repo_tiling_ids;
  std::vector<uint64_t> cost_tiling_ids;
  std::vector<uint64_t> batch_tiling_ids;  // for dynamic batch
  std::map<uint64_t, uint32_t> block_dim;
  std::string soc_version = "";

  uint32_t core_num = 0;
  uint64_t ub_size = 0;
  uint64_t l1_size = 0;
  uint64_t l2_size = 0;
  uint64_t l0a_size = 0;
  uint64_t l0b_size = 0;
  uint64_t l0c_size = 0;
  uint64_t bt_size = 0;
  int32_t cube_freq = 0;
  bool load3d_constraints = true;
  bool intrinsic_data_move_l12ub = true;
  bool intrinsic_matmul_ub_to_ub = false;
  bool intrinsic_conv_ub_to_ub = false;
  bool intrinsic_data_move_l0c2ub = true;
  bool intrinsic_fix_pipe_l0c2out = false;
  bool intrinsic_fix_pipe_l0c2ub = false;
  bool intrinsic_data_move_out2l1_nd2nz = false;
  bool intrinsic_data_move_l12bt_bf16 = false;
};

struct ConvTCompileParas {
  int32_t fm_c0 = 16;
  int32_t w_k0 = 16;
  int32_t w_n0 = 16;
  int32_t res_c0 = 16;
  int32_t forward_dilation_w = 0;
  int32_t forward_dilation_h = 0;
  int32_t batch_ub = 0;
  int32_t c_cub = 0;
  int32_t h_cub = 0;
  int32_t w_cub = 0;
  bool need_prepad = false;
  // only support NHWC
  std::string ori_format = "NHWC";

  std::string pre_conv = "None";
  std::string pre_activation = "None";
  std::string post_anti_quant = "None";
  std::string post_eltwise = "None";
  std::string post_activation = "None";
  std::string post_quant = "None";
  std::string post_transform = "None";
};

class Conv2DBackPropCompileInfo : public CubeCompileInfo {
 public:
  Conv2DBackPropCompileInfo() = default;
  ~Conv2DBackPropCompileInfo() override = default;

  // bool AnalyzeExtendInfo(const nlohmann::json &compile_info) override;
  void AnalyzeExtendInfoUb(const nlohmann::json &compile_info);

  bool repo_binary_flag = false;
  bool is_static_type = false;
  bool fifo_fusion_flag = false;
  uint32_t var_bit_flags = 0;
  uint32_t lib_api_workspace_size = 0;
  int32_t aub_num = 0;
  int32_t bub_num = 0;
  int32_t cub_num = 1;
  int32_t binary_mode = 1;
  std::string soc_version = "";
  ConvTCompileParas convt_params;
};

class Conv3DCompileInfo : public CubeCompileInfo {
 public:
  Conv3DCompileInfo() = default;
  ~Conv3DCompileInfo() override = default;

  // bool AnalyzeExtendInfo(const nlohmann::json &compile_info) override;
  bool repo_binary_flag = false;
  int32_t binary_mode = 1;
  uint32_t var_bit_flags = 0;
  int64_t fmap_c1 = -1;
  bool is_ascend_c = false;
};

class Conv3DBackPropInputCompileInfo : public CubeCompileInfo {
 public:
  Conv3DBackPropInputCompileInfo() = default;
  ~Conv3DBackPropInputCompileInfo() override = default;

  // bool AnalyzeExtendInfo(const nlohmann::json &compile_info) override;
  bool repo_binary_flag = false;
  uint32_t var_bit_flags = 0;
  int64_t dedy_c1 = -1;
  int32_t binary_mode = 1;
  int32_t fusion_mode = 0;
  std::string soc_version = "";
  bool is_ascend_c = false;
};

class DilationCompileInfo : public CubeCompileInfo {
 public:
  DilationCompileInfo() = default;
  ~DilationCompileInfo() override = default;

  // bool AnalyzeExtendInfo(const nlohmann::json &compile_info) override;
};

template <class CompileInfo, size_t range_size>
ge::graphStatus ParseCubeCompileInfo(gert::TilingParseContext *context) {
  OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("nil", "context is null"), return ge::GRAPH_FAILED);
  auto op_name = context->GetNodeName();
  OP_TILING_CHECK(op_name == nullptr, CUBE_INNER_ERR_REPORT("nil", "compute_node_info is null"),
                  return ge::GRAPH_FAILED);
  auto compile_info = context->GetCompiledInfo<CompileInfo>();
  auto json_str = context->GetCompiledJson();
  fe::PlatFormInfos *platform_info = context->GetPlatformInfo();
  OP_TILING_CHECK(compile_info == nullptr || json_str == nullptr || platform_info == nullptr,
                  CUBE_INNER_ERR_REPORT(op_name, "compile_info/json/PlatFormInfos is null"), return ge::GRAPH_FAILED);
  compile_info->ParseRuntimePlatformInfo(op_name, *platform_info);
  OP_TILING_CHECK(!compile_info->AnalyzeCompileInfo(op_name, json_str),
                  CUBE_INNER_ERR_REPORT(op_name, "failed to analyze compile info"), return ge::GRAPH_FAILED);
  OP_TILING_CHECK(!compile_info->CheckRangeSize(range_size),
                  CUBE_INNER_ERR_REPORT(op_name, "repo_range/repo_seeds/cost_range invalid"), return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CubeTiling(const int64_t *input_shape, size_t input_shape_dim_num, const gert::Shape &var_value,
                           const CubeCompileInfo &compile_info, gert::TilingContext *context);

std::string TensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor);
std::string DebugTilingContext(gert::TilingContext *context);
std::string DebugTilingData(gert::TilingContext *context);

ge::graphStatus TilingForConv3D(gert::TilingContext *context, const gert::StorageShape *fmap_shape,
                                const gert::StorageShape *out_shape, bool check_c1);
ge::graphStatus TilingForConv3DBpInput(gert::TilingContext *context, size_t dedy_idx, bool check_c1);

ge::graphStatus TilingForConv2DBp(gert::TilingContext *context, size_t dedy_idx, size_t inputs_size_limit);

size_t InitVarsValuesForConv2DBp(uint32_t var_bit_flags, const gert::Shape &in_shape, const gert::Shape &out_shape,
                                 gert::Shape &var_value, int64_t *shape_for_range_match);

int64_t Lcm(int64_t param1, int64_t param2);
int64_t Lcm(int32_t param1, int32_t param2);
int64_t Lcm(int64_t param1, int32_t param2);
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_CUBE_TILING_RUNTIME_H_
