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
 * \file platform_info.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_PLATFORM_PLATFORM_INFO_H_
#define OPS_BUILT_IN_OP_TILING_CUBE_PLATFORM_PLATFORM_INFO_H_

#include <cstdint>
#include <sstream>
#include <string>

#include "cube_tiling_runtime.h"

namespace optiling {
namespace cachetiling {
constexpr int32_t kKiloByte = 1024;

class PlatformInfo {
 public:
  PlatformInfo();
  ~PlatformInfo() = default;

  void SetRuntimePlatformInfo(const CubeCompileInfo &compile_info);
  bool SetRuntimePlatformInfo(const nlohmann::json &compile_info);
  void set_core_num(uint32_t core_num) { core_num_ = core_num; }
  void set_l1_size(uint64_t l1_size) { l1_size_ = l1_size; }
  void set_l2_size(uint64_t l2_size) { l2_size_ = l2_size; }
  void set_l0a_size(uint64_t l0a_size) { l0a_size_ = l0a_size; }
  void set_l0b_size(uint64_t l0b_size) { l0b_size_ = l0b_size; }
  void set_l0c_size(uint64_t l0c_size) { l0c_size_ = l0c_size; }
  void set_ub_size(uint64_t ub_size) { ub_size_ = ub_size; }
  void set_bt_size(uint64_t bt_size) { bt_size_ = bt_size; }
  void set_load3d_constraints(bool flag) { load3d_constraints_ = flag; }
  void set_data_move_l12ub(bool flag) { intrinsic_data_move_l12ub_ = flag; }
  void set_data_move_l0c2ub(bool flag) { intrinsic_data_move_l0c2ub_ = flag; }
  void set_data_move_out2l1_nd2nz(bool flag) { intrinsic_data_move_out2l1_nd2nz_ = flag; }

  uint32_t core_num() const { return core_num_; }
  uint64_t l2_size() const { return l2_size_; }
  uint64_t l1_size() const { return l1_size_; }
  uint64_t l0a_size() const { return l0a_size_; }
  uint64_t l0b_size() const { return l0b_size_; }
  uint64_t l0c_size() const { return l0c_size_; }
  uint64_t ub_size() const { return ub_size_; }
  uint64_t bt_size() const { return bt_size_; }
  const std::string& soc_version() const { return soc_version_; }
  bool support_data_move_l12ub() const { return intrinsic_data_move_l12ub_; }
  bool support_data_move_l0c2ub() const { return intrinsic_data_move_l0c2ub_; }
  bool support_ub() const { return support_data_move_l12ub() || support_data_move_l0c2ub(); }
  bool support_l0c2out() const { return intrinsic_fix_pipe_l0c2out_; }
  bool support_data_move_out2l1_nd2nz() const { return intrinsic_data_move_out2l1_nd2nz_; }
  bool load3d_constraints() const { return load3d_constraints_; }

  bool IsValid() const;
  bool IsValidL1Size(int32_t l1_size) const;
  bool IsValidL1Size(int64_t l1_size) const;
  bool IsValidL0ASize(int32_t l0a_size) const;
  bool IsValidL0BSize(int32_t l0b_size) const;
  bool IsValidL0CSize(int32_t l0c_size) const;
  std::string ToString() const;

 private:
  uint32_t core_num_ = 0;
  uint64_t l2_size_ = 0;
  uint64_t l1_size_ = 0;
  uint64_t l0a_size_ = 0;
  uint64_t l0b_size_ = 0;
  uint64_t l0c_size_ = 0;
  uint64_t ub_size_ = 0;
  uint64_t bt_size_ = 0;
  bool load3d_constraints_ = true;
  bool intrinsic_data_move_l12ub_ = true;
  bool intrinsic_data_move_l0c2ub_ = true;
  bool intrinsic_fix_pipe_l0c2out_ = false;
  bool intrinsic_fix_pipe_l0c2ub_ = false;
  bool intrinsic_data_move_out2l1_nd2nz_ = false;
  std::string soc_version_ = "";
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_PLATFORM_PLATFORM_INFO_H_

