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
 * \file constants_define.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_CUBE_CONSTANTS_CONSTANTS_DEFINE_H
#define OPS_BUILT_IN_OP_TILING_CUBE_CONSTANTS_CONSTANTS_DEFINE_H

#include <cstdint>
#include <array>
#include <map>

namespace optiling {
namespace cachetiling {
constexpr int32_t kL0cNzSize = 128;
constexpr int32_t kL0aNzSize = 64;
constexpr int32_t kL0bZnSize = 64;
constexpr int32_t kL0aSize = (64 * 1024);
constexpr int32_t kL0bSize = (64 * 1024);
constexpr int32_t kL0cSize = (256 * 1024);

constexpr int32_t kBlockSize = 16;
constexpr int32_t kFP16BlockReduce = 16;
constexpr int32_t kFP32BlockReduce = 8;
constexpr int32_t kFp16Bytes = 2;
constexpr int32_t kFp32Bytes = 4;
constexpr int32_t kSmallChannelSize = 4;

constexpr int32_t kFractalSize = kBlockSize * kBlockSize;
constexpr int32_t k1971HbmBandwidth = 96;
constexpr int32_t k1971L1ReadBandwidth = 256;
constexpr int32_t k1971Mte1L0ABandWidth = 256;  // 256 Bytes per cycle
constexpr int32_t k1971Mte1L0BBandWidth = 128;  // 128 Bytes per cycle
constexpr int32_t k1971L0CBandWidth = 256;
constexpr int32_t k1971FixpBandWidth = 128;
constexpr int32_t k1971BiasTableBandWidth = 32;
constexpr int32_t k1971Fp32FixpBandWidth = 21;
constexpr int32_t k1971MadCost = 20;
constexpr int32_t k1971Mte2Latency = 300;
constexpr int32_t kIssueQueue = 32;
constexpr int32_t k1971Mte1Load2dCost = 20;
constexpr int32_t k1971Mte1Load3dCost = 42;
constexpr std::array<int32_t, 6> kCachelinePkgSize = {16, 8, 4, 3, 2, 1};  // 1971 cacheline

constexpr int32_t kMinSet2dCost = 15;
constexpr int32_t k1980Mte1L0aBandWidth = 512;  // 512 Bytes per cycle
constexpr int32_t k1980Mte1L0bBandWidth = 256;  // 256 Bytes per cycle
constexpr int32_t k1980Mte3BandWidth = 64; // 64 Bytes per cycle
constexpr int32_t k1980VectorBandWidth = 512; // 512 Bytes per cycle
constexpr int32_t k1980MinCoreNum = 30;
constexpr int32_t k1980L0cFactorLimit = 256;

// kDtypeCompensateFactor作为数据类型相关的补偿因子，比如dx算子fp32类型输入的Co1g是以16为对齐单位计算的
// 实际使用时，需要使用补偿因子计算出真实的Co1g(以8为对齐单位计算)
constexpr int32_t kDtypeCompensateFactor = 2;
constexpr int32_t kCubeTileNumSize = kBlockSize * kBlockSize;
constexpr int32_t kFp32CubeTileNumSize = kBlockSize * kFP32BlockReduce;
constexpr int32_t kDbOff = 1;
constexpr int32_t kDbOn = 2;
// The maximum fraction calculated at one time(with db on) in l0c is
// kL0cSize / kFp32Bytes / kDbOn / kCubeTileNumSize = 128
// The maximum fraction calculated at one time(with db on) in l0a/l0b is
// fp16:kL0aSize / kFp16Bytes / kDbOn / kCubeTileNumSize = 64
// fp32:kL0aSize / kFp32Bytes / kDbOn / kFp32CubeTileNumSize = 64
// Combined l0a, l0b, l0c utilization, m/n=16/8, k=4 utilization is the highest
constexpr int32_t kMinSingleCoreK = 4;
constexpr int32_t kMaxLoad3dV2Kstart = 65535;
constexpr int32_t kSplitWAxisMode = 1;

// attach flag
constexpr int32_t kNone = INT32_MIN;
constexpr int32_t kAttachFullLoad = 0;
constexpr int32_t kAttachEqual = 1;
constexpr int32_t kAttachLess = 2;
constexpr int32_t kAttachNotLoad = 3;

static const int32_t kCandidateLen = 2;
static const int32_t kL1FactorLimit = 128;
static const int32_t kL1FactorsLen = 6;
static const int32_t kL0FactorNumLimit = 2;
static const int32_t kL1FactorNumLimit = 4;
static const int32_t kSeedMapMin = 16;
static const int32_t kSeedMapMax = 1024;
static const int32_t kMNPntMax = 16;
static const int32_t kMNPntMaxWithBt = 8;
static const int32_t kMaxFactor = 128;
static const int32_t kMaxFactorWithBt = 16;
static const int32_t kMinMte1Load = 32;
static const int32_t kMinFractalSize = 256;
static const int32_t kMinFactorLimit = 32;
static const bool kL0DbFlag = false;
static const int32_t kL0ParasComboLen = kL0DbFlag ? 8 : 2;
static const int32_t kKbytes = 1024;
static const int32_t kIdxZero = 0;
static const int32_t kIdxOne = 1;
static const int32_t kIdxThree = 3;
static const int32_t kIdxFour = 4;
static const int32_t kIdxFive = 5;
static const int32_t kIdxSix = 6;
static const int32_t kIdxSeven = 7;
static const int32_t kIdxEight = 8;

enum BinaryMode : uint32_t {
    kBinaryModeNC1HWC0 = 1,
    kBinaryModeNCHW = 2,
    kBinaryModeNHWC = 3
};

const std::map<int32_t, int32_t> kDtypeBlockReduceMap = {
    {ge::DT_FLOAT16, kBlockSize}, {ge::DT_FLOAT, kFP32BlockReduce}
};

const std::map<int32_t, int32_t> kDtypeCubeTileNumSizeMap = {
    {ge::DT_FLOAT16, kCubeTileNumSize}, {ge::DT_FLOAT, kFp32CubeTileNumSize}
};
}  // namespace cachetiling
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_CUBE_CONSTANTS_CONSTANTS_DEFINE_H

