/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_RUNTIME_BANK_MANAGER_H_
#define RUNTIME_KB_RUNTIME_BANK_MANAGER_H_
#include <unordered_map>
#include <string>
#include <mutex>
#include "exe_graph/runtime/tiling_context.h"
#include "register/tuning_tiling_registry.h"
#include "aoe/runtime_kb/op_runtime_bank.h"
#include "aoe/runtime_kb/common/kb_status.h"

namespace RuntimeKb {
struct PlatformInfo {
  PlatformInfo() = default;
  PlatformInfo(uint32_t in_core_num, const std::string& in_soc_version)
      : core_num(in_core_num), soc_version(in_soc_version) {
  }
  uint32_t core_num = 1U;
  std::string soc_version;
};
class RuntimeBankManager {
 public:
  __attribute__((visibility("default"))) static RuntimeBankManager& Instance();
  __attribute__((visibility("default"))) Status InitAoeOpBank(const PlatformInfo& plat,
    const std::set<std::string>& opLists);
  __attribute__((visibility("default"))) Status Query(const void* src, size_t src_len, const std::string& optype,
    const PlatformInfo& plat, tuningtiling::TuningTilingDefPtr& tiling);
  __attribute__((visibility("default"))) Status Query(const gert::TilingContext* op, const std::string& optype,
    const PlatformInfo& plat, tuningtiling::TuningTilingDefPtr& tiling);
  __attribute__((visibility("default"))) Status Update(const gert::TilingContext* op, const std::string& optype,
    const tuningtiling::TuningTilingDefPtr& tiling);
  __attribute__((visibility("default"))) Status Save();
  __attribute__((visibility("default"))) Status SetTuningTiling(const gert::TilingContext* op,
    const std::string& optype, const std::string& tiling);
  Status SetTuningTiling(const uint32_t pid, const std::string& optype, const std::string& tiling);

 private:
  Status InitAoeOpBank(const std::string& op);
  void InitAtcOpBank(const PlatformInfo& plat, const std::string& op);
  Status InitOpBank(const std::string& op, const std::string& custom_path, const std::string& builtin_path,
                    RuntimeBankType type);
  std::vector<std::string> GetAllOpBankId(const std::string& optype) const;
  std::string GetAoeOpBankId(const std::string& optype) const;
  RuntimeBankManager() = default;
  ~RuntimeBankManager() = default;
  RuntimeBankManager(const RuntimeBankManager&) = delete;
  RuntimeBankManager& operator=(const RuntimeBankManager&) = delete;
  static uint32_t core_num_;
  static std::string soc_version_;
  std::string bank_prefix_;
  // optiling::RWLock rwlock_;
  std::mutex mtx_;
  std::unordered_map<std::string, OpRuntimeBankPtr> bank_cache_;
  std::unordered_map<std::string, std::vector<std::string>> bank_id_;  // ky: optype, value: all bank index
  std::unordered_map<std::string, std::string> aoe_bank_id_;           // ky: optype, value aoe bank index
};
extern "C" __attribute__((visibility ("default"))) Status SetTuningTiling(const uint32_t pid, const char* optype,
  const char* tiling);
extern "C" __attribute__((visibility ("default"))) Status RuntimeKBSetTuningTiling(const gert::TilingContext *op,
  const std::string &op_type, const std::string &tiling);
extern "C" __attribute__((visibility ("default"))) Status RuntimeKBQuery(const gert::TilingContext *op,
  const std::string &op_type, const PlatformInfo &plat, tuningtiling::TuningTilingDefPtr &tiling);
extern "C" __attribute__((visibility ("default"))) Status RuntimeKBUpdate(const gert::TilingContext *op,
  const std::string &op_type, const tuningtiling::TuningTilingDefPtr &tiling);
extern "C" __attribute__((visibility ("default"))) Status RuntimeKBSave();
extern "C" __attribute__((visibility ("default"))) Status RuntimeKBInitAoeOpBank(const PlatformInfo &plat,
  const std::set<std::string> &op_list);
}  // namespace RuntimeKb
#endif

