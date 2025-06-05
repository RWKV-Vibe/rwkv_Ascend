/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RUNTIME_KB_OP_RUNTIME_BANK_H_
#define RUNTIME_KB_OP_RUNTIME_BANK_H_

#include <unordered_map>
#include <string>

#include "register/tuning_tiling_registry.h"
#include "register/tuning_bank_key_registry.h"
#include "aoe/runtime_kb/common/kb_common.h"
#include "aoe/runtime_kb/common/kb_status.h"

namespace RuntimeKb {
enum class RuntimeBankType {
  ATC = 0,
  AOE,
};
struct Record {
  std::string Serialize();
  bool Deserialize(const std::string& record);
  uint32_t ky = 0U;
  uint32_t version = 0U;  // reverse for compatable
  std::string optype;
  std::shared_ptr<void> ori_ky = nullptr;
  size_t bank_key_size = 0U;
  tuningtiling::TuningTilingDefPtr tiling = nullptr;
};

class CustomRepoGuard;
class OpRuntimeBank {
 public:
  OpRuntimeBank(const std::string& custom_path, const std::string& builtin_path, const RuntimeBankType type);
  OpRuntimeBank(const std::string& custom_path, const std::string& builtin_path, const RuntimeBankType type,
    const std::vector<std::string>& kb_str);
  ~OpRuntimeBank() = default;
  Status Initialize(bool check_custom_path);
  Status Query(uint32_t k, uint32_t pid, const void* src, size_t src_size, tuningtiling::TuningTilingDefPtr& tiling);
  Status Update(uint32_t k, const std::shared_ptr<void>& ori_ky, size_t ori_ky_size, const std::string& optype,
                const tuningtiling::TuningTilingDefPtr& tiling);
  Status Save();
  RuntimeBankType GetType();
  void SetTuningTiling(uint32_t ky, const tuningtiling::TuningTilingDefPtr& tiling);

 private:
  Status UnpackageStaticLib();
  Status CheckObj();
  Status LoadFromFile(const std::string& file, bool builtin_repo = false);
  // optiling::RWLock rwlock_;
  std::string custom_path_;
  std::string builtin_path_;
  RuntimeBankType type_;
  std::vector<std::string> kb_str_;
  std::mutex mtx_;
  std::shared_ptr<std::mutex> async_mtx_ = MakeShared<std::mutex>();
  std::shared_ptr<bool> async_ready_ = MakeShared<bool>(false);
  std::shared_ptr<bool> async_ready_custom_ = MakeShared<bool>(false);
  std::shared_ptr<int8_t> async_actives_ = MakeShared<int8_t>(0);
  std::shared_ptr<int8_t> async_actives_custom_ = MakeShared<int8_t>(0);
  friend class CustomRepoGuard;
  std::shared_ptr<std::unordered_map<uint32_t, Record>> builtin_ = MakeShared<std::unordered_map<uint32_t, Record>>();
  std::shared_ptr<std::unordered_map<uint32_t, Record>> custom_ = MakeShared<std::unordered_map<uint32_t, Record>>();
  std::unordered_map<uint32_t, tuningtiling::TuningTilingDefPtr> tuning_cache_;
};

class CustomRepoGuard {
 public:
  explicit CustomRepoGuard(OpRuntimeBank& op_bank, bool auto_create = true);
  CustomRepoGuard(const CustomRepoGuard&) = delete;
  CustomRepoGuard& operator=(const CustomRepoGuard&) = delete;
  CustomRepoGuard(const CustomRepoGuard&&) = delete;
  CustomRepoGuard& operator=(const CustomRepoGuard&&) = delete;
  ~CustomRepoGuard();

 protected:
  void Lock(const std::string& custom_path, bool auto_create);
  void UnLock() const;

  int32_t handle_lock_ = 0;
};
using OpRuntimeBankPtr = std::shared_ptr<OpRuntimeBank>;
}  // namespace RuntimeKb
#endif

