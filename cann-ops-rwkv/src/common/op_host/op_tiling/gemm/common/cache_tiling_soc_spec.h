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
 * \file cache_tiling_soc_spec.h\
 * \brief function of cache tiling soc spec bandwidth info
 */
#ifndef     OPS_BUILT_IN_OP_TILING_CACHE_TILING_SOC_SPEC_H
#define     OPS_BUILT_IN_OP_TILING_CACHE_TILING_SOC_SPEC_H
#include "cache_tiling.h"

namespace gemm_cache_tiling {
using namespace std;
static constexpr int32_t L2_SIZE_2 = 192 * 1024 * 1024;
enum DEVICE_ID {
  DDR,
  L2,
  L1,
  L0A,
  L0B,
  L0C,
};

struct GemmSocSpecBandwidth {
  const double L0A_RO = 256;
  const double L0A_WO = 256;
  const double L0A_RW = 512;
  const double L0A_LATENCY_RD = 0;
  const double L0A_LATENCY_WR = 0;

  const double L0B_RO = 128;
  const double L0B_WO = 128;
  const double L0B_RW = 512;
  const double L0B_LATENCY_RD = 0;
  const double L0B_LATENCY_WR = 0;

  const double L0C_RO = 128;
  const double L0C_WO = 128;
  const double L0C_RW = 128;
  const double L0C_LATENCY_RD = 0;
  const double L0C_LATENCY_WR = 0;

  const double L1_RO = 256;
  const double L1_WO = 256;
  const double L1_RW = 256;
  const double L2_RO = 2550;
  const double L2_WO = 2550;
  const int32_t L2_LATENCY_RD = 236;
  const int32_t L2_LATENCY_WD = 96;

  double DDR_RO = 850;      // ddr read bw
  double DDR_WO = 850;      // ddr write bw
  double DDR_RW = 850;      // ddr read write bw
  const int32_t DDR_LATENCY_RD = 347;
  const int32_t DDR_LATENCY_WD = 203;

  const double L2_PHY_RO = 128;
  const double L2_PHY_WO = 128;
  double DDR_PHY = 37;     // ddr phy bw

  const int32_t READ_OTSd = 64;
  const int32_t WRITE_OTSd = 48;

public:
  GemmSocSpecBandwidth() {
    if (optiling::PlatformInfo::GetInstance().l2_size != L2_SIZE_2) {
      DDR_RO = 530;  // ddr read bw
      DDR_WO = 530;  // ddr write bw
      DDR_RW = 530;  // ddr read write bw
      DDR_PHY = 27;  // ddr phy bw
    }
  };
};
}
#endif
