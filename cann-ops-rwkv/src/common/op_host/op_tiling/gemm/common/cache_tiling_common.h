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
 * \file cache_tiling_common.h\
 * \brief function of cache tiling common info
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_COMMON_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_COMMON_H
#include "cache_tiling.h"

namespace gemm_cache_tiling {
using namespace std;

constexpr int32_t increasekBlockSize = 32;
constexpr int32_t kBlockSize = 16;
constexpr int32_t reducekBlockSize = 8;
constexpr int32_t kInt8Bytes = 1;
constexpr int32_t kFp16Bytes = 2;
constexpr int32_t kFp32Bytes = 4;
constexpr int32_t DB_SIZE = 2;

struct GEMM_CUBE_SIZE{
  GEMM_CUBE_SIZE(int64_t N, int64_t B, int64_t src_d, int32_t dsize) {
    n = N;
    d = B;
    srcD = src_d;
    dtype_size = dsize;
  }
  int64_t n{1};
  int64_t d{1};
  int64_t srcD{1};
  int32_t dtype_size{1};
};

struct GemmResultInfo {
public:
  int64_t batch_dim = 1;
  int64_t m_dim = 1;
  int64_t n_dim = 1;
  int64_t k_dim = 1;

  int64_t m_l0 = 1;
  int64_t n_l0 = 1;
  int64_t k_l0 = 1;
  int64_t batch_l0 = 1;
  int64_t db_l0c = optiling::kDbOff;

  int64_t kal1_16 = 1;
  int64_t kbl1_16 = 1;
  int64_t m_l1 = 1;
  int64_t n_l1 = 1;
  int64_t db_al1 = optiling::kDbOff;
  int64_t db_bl1 = optiling::kDbOff;
public:
  GemmResultInfo() :
    batch_dim(1),
    m_dim(1),
    n_dim(1),
    k_dim(1),
    m_l0(1),
    n_l0(1),
    k_l0(1),
    batch_l0(1),
    db_l0c(optiling::kDbOff),
    kal1_16(1),
    kbl1_16(1),
    m_l1(1),
    n_l1(1),
    db_al1(optiling::kDbOff),
    db_bl1(optiling::kDbOff) {}

  GemmResultInfo(const optiling::CoreStatus &core_status, const optiling::SingleCoreStatus &single_core_status) :
    batch_dim(core_status.batch_dim),
    m_dim(core_status.m_dim),
    n_dim(core_status.n_dim),
    k_dim(core_status.k_dim),
    m_l0(single_core_status.l0Status.m_l0),
    n_l0(single_core_status.l0Status.n_l0),
    k_l0(single_core_status.l0Status.k_l0),
    batch_l0(single_core_status.l0Status.batch_l0),
    db_l0c(single_core_status.l0Status.db_l0c),
    kal1_16(single_core_status.l1Status.kal1_16),
    kbl1_16(single_core_status.l1Status.kbl1_16),
    m_l1(single_core_status.l1Status.m_l1),
    n_l1(single_core_status.l1Status.n_l1),
    db_al1(single_core_status.l1Status.db_al1),
    db_bl1(single_core_status.l1Status.db_bl1) {}

  bool IsAvaliable() const {
    return !(batch_dim <= 0 || m_dim <= 0 || n_dim <= 0 || k_dim <= 0 ||
             m_l0 <= 0 || n_l0 <= 0 || k_l0 <= 0 || batch_l0 <= 0 ||
             kal1_16 <= 0 || kbl1_16 <= 0 || m_l1 <= 0 || n_l1 <= 0);
  }
  int64_t GetLoadTimes(int64_t load_l0_repeat) const {
    return (min(kal1_16, kbl1_16) / k_l0) * load_l0_repeat;
  }
  friend ostream& operator<<(ostream& out, const GemmResultInfo &a);
};


struct GemmEstCoreStatus {
public:
  int64_t batch = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t kal1_factor = 1;
  int64_t kbl1_factor = 1;
  int64_t m_single_core = 1;
  int64_t n_single_core = 1;
  bool both_full_load = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  bool al1_k_full_load = false;
  bool bl1_k_full_load = false;
public:
  GemmEstCoreStatus() = default;
  GemmEstCoreStatus(const optiling::CoreStatus &core_status, const optiling::SingleCoreStatus &single_core_status) :
    batch(core_status.batch),
    m(core_status.m),
    k(core_status.k),
    n(core_status.n),
    kal1_factor(core_status.kal1_factor),
    kbl1_factor(core_status.kbl1_factor),
    m_single_core(core_status.m_single_core),
    n_single_core(core_status.m_single_core),
    both_full_load(single_core_status.l1Status.both_full_load),
    al1_full_load(single_core_status.l1Status.al1_full_load),
    bl1_full_load(single_core_status.l1Status.bl1_full_load),
    al1_k_full_load(single_core_status.l1Status.al1_k_full_load),
    bl1_k_full_load(single_core_status.l1Status.bl1_k_full_load) {};
  void Clear() {
    batch = 1;
    m = 1;
    k = 1;
    n = 1;
    kal1_factor = 1;
    kbl1_factor = 1;
    m_single_core = 1;
    n_single_core = 1;
    both_full_load = false;
    al1_full_load = false;
    bl1_full_load = false;
    al1_k_full_load = false;
    bl1_k_full_load = false;
  };
  bool IsKBothFullLoad() const {
    return al1_k_full_load && bl1_k_full_load;
  }
  bool IsKAnyFullLoad() const {
    return al1_k_full_load || bl1_k_full_load;
  }
  bool IsAnyFullLoad() const {
    return al1_full_load || bl1_full_load;
  }
  friend ostream& operator<<(ostream& out, const GemmEstCoreStatus &a);
};
}
#endif

