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
 * \file gemm_estimate_create_mgr.h\
 * \brief function of gemm cache tiling estimate create manager
 */
#ifndef     OPS_BUILT_IN_OP_TILING_GEMM_ESTIMATE_CREATE_FUNC_MGR_H
#define     OPS_BUILT_IN_OP_TILING_GEMM_ESTIMATE_CREATE_FUNC_MGR_H

#include <map>
#include <functional>
#include "cache_tiling_est.h"

namespace gemm_cache_tiling {

enum GemmEstimateType {
  BASIC_BLOCK_ESTIMATE_TYPE,
  CYCLE_ESTIMATE_TYPE,
};

using GEMM_ESTIMATE_CREATE_FUNC = std::function<GemmEstimatePtr(const std::string& op_type,
    const optiling::BatchmatmulParas *paras)>;

class GemmEstimateCreateMgr {
public:
  static GemmEstimateCreateMgr& Instance() {
    static GemmEstimateCreateMgr inst;
    return inst;
  };

  void RegFunc(GemmEstimateType type, const GEMM_ESTIMATE_CREATE_FUNC& func) {
    container[type] = func;
  }

  GEMM_ESTIMATE_CREATE_FUNC GetFunc(GemmEstimateType type) {
    auto it = container.find(type);
    if (it == container.end()) {
      return nullptr;
    }
    return it->second;
  }

private:
  GemmEstimateCreateMgr() = default;
  ~GemmEstimateCreateMgr() = default;

private:
  std::map<GemmEstimateType, GEMM_ESTIMATE_CREATE_FUNC> container{};
};

class GemmEstimateCreateFuncRegister {
public:
  GemmEstimateCreateFuncRegister(GemmEstimateType type, const GEMM_ESTIMATE_CREATE_FUNC& func) {
    GemmEstimateCreateMgr::Instance().RegFunc(type, func);
  }
  ~GemmEstimateCreateFuncRegister() = default;
};

class GemmEstimateFactory {
public:
  static GemmEstimatePtr GetEstimate(GemmEstimateType type, const std::string& op_type,
    const optiling::BatchmatmulParas *paras) {
    auto createFunc = GemmEstimateCreateMgr::Instance().GetFunc(type);
    if (createFunc == nullptr) {
      return nullptr;
    }
    return createFunc(op_type, paras);
  }
};

#define REG_GEMM_ESTIMATE_FUNC(type, class_name) \
  static GemmEstimateCreateFuncRegister g_reg##class_name##type(type,           \
      [](const std::string& op_type, const optiling::BatchmatmulParas *paras) { \
          return std::unique_ptr<class_name>(new class_name(op_type, paras));   \
      }                                                                         \
  )
}
#endif
