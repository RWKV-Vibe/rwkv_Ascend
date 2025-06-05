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
 * \file cache_tiling.cc
 * \brief function of cacheTiling
 */
#define ASSERT_TRUE(cond, expr_exception_handling) \
  do {                                             \
    if (!(cond)) {                                 \
      expr_exception_handling;                     \
    }                                              \
  } while (0)

#include "cache_tiling.h"

#include <chrono>
#include <mutex>
#include <thread>
#include <sys/prctl.h>

#include "aoe/op_tuning_tiling/gemm_tuning_tiling.h"
#include "aoe/runtime_kb/runtime_bank_manager.h"
#include "cube/algorithm/hash/hash.h"
#include "cube/algorithm/hash/tiling_cache.h"
#include "gemm/cache_tiling_basic_block.h"
#include "gemm/estimate/cache_tiling_est.h"
#include "gemm/estimate/cache_tiling_est_mgr.h"
#include "gemm/estimate/cache_tiling_cycle_est.h"
#include "gemm/estimate/cache_tiling_cycle_model.h"
#include "compress_dequant_cache_tiling.h"
#include "mathutil.h"

#define OP_LOGD(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGI(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGW(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)
#define OP_LOGE_WITHOUT_REPORT(nodeName, fmt, ...) do {std::printf(fmt, ##__VA_ARGS__); std::printf("\n"); } while(0)

using namespace std;
// using optiling::cachetiling::MathUtil;
using namespace gemm_cache_tiling;

namespace {
using namespace optiling;

static const int64_t kMinFractalSize = kBlockSize * kBlockSize;
static const int32_t kAttachFlagZero = 0;
static const int32_t kAttachFlagOne = 1;
static const int32_t kAttachFlagTwo = 2;
static const int64_t kDiagnoalMatrixSize = 1024;
static const int32_t kIdxZero = 0;
static const int32_t kIdxOne = 1;
static const int32_t kIdxTwo = 2;
static const int32_t kIdxThree = 3;
static const int32_t kIdxFour = 4;
static const int32_t kIdxFive = 5;
static const int32_t kIdxSix = 6;
static const int32_t kIdxSeven = 7;
static const int32_t kIdxEight = 8;
static const int32_t kNumZero = 0;
static const int32_t kNumOne = 1;
static const int32_t kNumTwo = 2;
static const int32_t kNumThree = 3;
static const int32_t kNumFour = 4;
static const int64_t kKbytes = 1024;
static const int64_t kMaxFactor = 128;
static const int64_t kMaxFactorWithBt = 16;
static const int64_t kMinMte1Load = 32;
static const bool kL0DbFlag = false;
static const int32_t kL0ParasComboLen = kL0DbFlag ? 8 : 2;
static const int64_t kBankConflictFactor = 4;
static const int32_t kL1FactorsLen = 6;
static const int32_t kCandidateLen = 2;
static const int64_t kMinSplitKCoreNum = 8;
static const int64_t K_WORST_BANDWIDTH_UTIL_MULTI = 8;
static const int64_t kHbmBandwidth8Core = 250;
static const int64_t kHbmBandwidth32Core = 1100;
static const int64_t kL2Bandwidth8Core = 1300;
static const int64_t kL2Bandwidth32Core = 3300;
static const int64_t kMNPntMax = 16;
static const int64_t kMNPntMaxWithBt = 8;
static const int64_t kSeedMapMin = 16;
static const int64_t FractalSize = 16 * 16 * 4;
static const int64_t PingPong = 2;

// minimum factor number requirement for the data amount in single-core
static const int64_t kL0FactorNumLimit = 2;
static const int64_t kL1FactorNumLimit = 4;
// the lower bound of the factor number check
static const int64_t kMinFactorLimit = 32;
static const int64_t kL0FactorLimit = 64;
static const int64_t kL1FactorLimit = 128;
static const uint32_t kNd2NzOnTheFly = 1;
static const uint32_t kNd2NzMixL2 = 2;
static const uint32_t kWeightNz = 3;
// Pattern mode support non factor ub, so need more factors.
static const int32_t kUbFactorNums = 500;
static const int64_t kMNL0PreferSize = 16;
// the number 8 is a good split from experience for outer axis
static const int64_t kL0PreferSizeOuterAxis = 8;

thread_local static int64_t kUbFp16Size = PlatformInfo::GetInstance().ub_size / kFp16Bytes;
thread_local static int64_t kUbFp32Size = PlatformInfo::GetInstance().ub_size / kFp32Bytes;
// 160M
static const int64_t kRightInputSizeThreshold = 160 * 1024 *1024;
static const int64_t kLeftInputSizeThreshold = 100 * 1024 *1024;
// 1G
static const int64_t kOutputSizeThreshold1 = 1024 * 1024 *1024;
static const int64_t kOutputSizeThreshold2 = 400 * 1024 *1024;
static const int64_t kFractalSize = 512;
// the value for 1980
static const int64_t k1980FullCacheLine = 8;
static const int64_t k1980Mte1L0aBandWidth = 512;  // 512 Bytes per cycle
static const int64_t k1980Mte1L0bBandWidth = 256;  // 256 Bytes per cycle
static const int64_t k1980Mte3BandWidth = 64; // 64 Bytes per cycle
static const int64_t k1980MinCoreNum = 30;
static const int64_t k1980L0cFactorLimit = 256;

// the value for 1971
static const int64_t k1971FullCacheLine = 16;
static const int64_t k1971FullCacheLineUnalign = 32;
static const int64_t k1971FullCacheLineBytes = 512;
static const int64_t k1971Mte2BaseBandWidth = 64;
static const int64_t k1971HbmBandwidth = 64;
static const int64_t k1971L0cFactorMax = 128;

// the value for 1982
static const int64_t k1982FullCacheLine = 4;
static const int64_t k1982FullCacheLineUnalign = 8;
static const int64_t k1982FullCacheLineBytes = 128;
static const int64_t k1982Mte2BaseBandWidth = 24;
static const int64_t k1982HbmBandwidth = 24;
static const int64_t k1982L0cFactorMax = 256;

static const int64_t kML0PreferSize = 8;
static const int64_t kNL0PreferSize = 16;
static const int64_t kML0PreferSizeOuterAxis = 8;
static const int64_t kKL0PreferSize = 4;
static const int64_t kKBL1PreferSize = 4;
static const int64_t kML1MiniSize = 8;
static const int64_t kNd2NzLimitNum = 65536;
// kl0 * kl0 /2 *16 *32 < l0a_size and kl0 must be odd
static const int64_t kWeightQuantBmmKL0Max = 4;
static const int64_t kWeightQuantBmmNL0Max = 8;
static const int64_t kMadInt32kThreshold = 4;
static const int64_t kKl0Align = 0b11111100;
static const int64_t kPadFusionSize = 256;
static const int64_t kPadFusionSizeFp32 = 128;
static const int64_t kSuperCoreLimit = 512;

static std::mutex str_soc_mutex;

// use pattern thread to calculate pattern mode tiling
static std::mutex *pattern_lock = nullptr;
static std::thread *pattern_thread = nullptr;
static std::list<PatternParams> *pattern_list = nullptr;
static cachetiling::MMTilingHash *tiling_hash_cache = new cachetiling::MMTilingHash;
static bool thread_exit_flag = false;
static bool thread_init_succ = false;
static bool pattern_cache_enable = true;
static const size_t kMMThreadItemNum = 10000;
static std::once_flag pattern_once_flag;
// Thread local variables
thread_local static int64_t l0cFactorLimit1971;
thread_local static int64_t inputDtypeBytes;
thread_local static int64_t outputDtypeBytes;
thread_local static int64_t reducedBlockSize;
thread_local static int64_t l1AvailableSize;
thread_local static int64_t madExpansionRate = kNumOne;
thread_local static int64_t fullCacheLine = k1971FullCacheLine;
thread_local static int64_t fullCacheLineUnalign = k1971FullCacheLineUnalign;
thread_local static int64_t l0FactorLimit = kL0FactorLimit;
thread_local static int64_t nL0PreferSize = kNL0PreferSize;
thread_local static int64_t kL0PreferSize = kKL0PreferSize;
thread_local static int64_t mL0PreferSizeOuterAxis = kML0PreferSizeOuterAxis;
thread_local static int64_t kBL1PreferSize = kKBL1PreferSize;
thread_local static int64_t kAlignValue = 1;
thread_local static int64_t nAlignValue = 1;
thread_local static int64_t mAlignValue = 1;

static const int8_t kTransdataB = 1;
static const int8_t kTransdataA = 2;
static const int8_t kTransdataAB = 3;

static const int32_t kAllL2EnableBit = 0;
static const int32_t kAL2DisableBit = 1;
static const int32_t kBL2DisableBit = 2;
static const int32_t kBiasL2DisableBit = 3;
static const int32_t kCL2DisableBit = 4;

const std::map<ge::DataType, int32_t> kDataSizeMap = {
  {ge::DT_INT8, kInt8Bytes},
  {ge::DT_FLOAT16, kFp16Bytes},
  {ge::DT_BF16, kFp16Bytes},
  {ge::DT_FLOAT, kFp32Bytes},
  {ge::DT_INT32, kFp32Bytes}
};

int32_t GetDataSize(const ge::DataType &input_dtype) {
  int32_t data_size = kFp16Bytes;
  auto iter = kDataSizeMap.find(input_dtype);
  if (iter != kDataSizeMap.end()) {
    data_size = iter->second;
  }
  return data_size;
}

void CalcTilingPatternMode();

void MMInitThread() {
  pattern_lock = new std::mutex;
  pattern_list = new std::list<PatternParams>;
  pattern_thread = new std::thread(CalcTilingPatternMode);
  if (pattern_lock && pattern_list && tiling_hash_cache && pattern_thread) {
    thread_init_succ = true;
  }
}

__attribute__((destructor)) void MMDestroyThread() {
  thread_exit_flag = true;
  if (pattern_thread && pattern_thread->joinable()) {
    pattern_thread->join();
  }
  delete pattern_thread;
  delete tiling_hash_cache;
  delete pattern_list;
  delete pattern_lock;
}

void MMDoOnce() {
  std::call_once(pattern_once_flag, MMInitThread);
}

int64_t MapShape(int64_t shape, bool round_up_flag = true) {
  // map numbers between 32 to number of power of 2.
  int64_t seed = kSeedMapMin;
  if (shape < seed) {
    return shape;
  }
  while ((seed << 1) < shape) {
    seed = seed << 1;
  }
  if (round_up_flag) {
    return seed << 1;
  }
  return seed;
}

int64_t GetBiasBtSize(int64_t tiling_nl0) {
   return tiling_nl0 * kBlockSize * kFp32Bytes;
}

int64_t GetBiasL1Size(const BatchmatmulRunParas &run_params, int64_t tiling_nl1) {
  return tiling_nl1 * kBlockSize * run_params.dtype_bias;
}

int64_t GetChannelWiseL1Size(const L1Status &l1Status, int64_t tiling_nl1) {
  return l1Status.channel_wise_times * tiling_nl1 * kBlockSize * kFp16Bytes;
}

bool CheckL1Size(int64_t amat, int64_t bmat, int64_t cur_extra_l1_size = 0) {
  int64_t load_size_bytes = ((amat + bmat) * kBlockSize * reducedBlockSize * inputDtypeBytes + cur_extra_l1_size);
  return load_size_bytes <= l1AvailableSize;
}

bool CheckMulOverflow(int64_t a, int64_t b, int64_t &c) {
  if (a > 0 && b > 0) {
    if (a > (INT64_MAX / b)) {
      return false;
    }
  } else {
    return false;
  }
  c = a * b;
  return true;
}

bool CheckAddOverflow(int64_t a, int64_t b, int64_t &c) {
  if (((b > 0) && (a > (INT64_MAX - b))) || ((b < 0) && (a < (INT64_MIN - b)))) {
    return false;
  }
  c = a + b;
  return true;
}

void GetShapeFactors(int64_t &cnt, int64_t *factorList, int64_t ori_shape, int64_t shape, int64_t max_factor)
{
  // get all factors of ori_shape and shape which smaller or equal to maxNum
  for (int64_t i = 1; i <= max_factor; ++i) {
    if (ori_shape % i == 0 || shape % i == 0) {
      factorList[(cnt)++] = i;
    }
  }
}

void GetTwoFactors(int64_t *res, int64_t base, int64_t dim, int64_t maxNum = 32, int64_t cnt = 0)
{
  // for up bigger or equal to base + 1, find the smallest num which is a factor of dim
  // form down smaller or equal to base, find the biggest num which is a factor of dim
  int64_t up = base + 1;
  int64_t maxCnt = 2;
  while (up <= dim) {
    if (up > maxNum) {
      break;
    }
    if (dim % up == 0) {
      res[cnt++] = up;
      break;
    }
    up++;
  }
  int64_t down = base;
  while (down >= 1) {
    if (dim % down == 0) {
      res[cnt++] = down;
      if (cnt == maxCnt) {
        break;
      }
    }
    down--;
  }
}

void GetAllFactors(int64_t *res, int64_t base, int64_t dim)
{
  // form down smaller or equal to base, find all nums which is a factor of dim
  int64_t down = base;
  size_t cnt = 0;
  while (down >= 1) {
    if (dim % down == 0) {
      res[cnt++] = down;
    }
    down--;
  }
}

void GetNearestFactor(int64_t base, int64_t &factor, int64_t cap_value = INT64_MAX)
{
  if (cap_value == INT64_MAX) {
    cap_value = base;
  }
  while ((factor > cap_value) || (factor > 0 && base % factor != 0)) {
    factor--;
  }
}

void BL1FullLoadBlock(const CoreStatus &coreStatus, BlockDimCalculator &blockDimCalculator,
                      int64_t &n0, bool b_have_batch)
{
  if (n0 >= 1) {
    while (coreStatus.n % n0 != 0) {
      n0--;
    }
    blockDimCalculator.amat_size = blockDimCalculator.ori_amat_size * MathUtil::CeilDivision(coreStatus.n , n0);
    blockDimCalculator.bmat_size = b_have_batch ? coreStatus.batch * coreStatus.n : coreStatus.n;
    blockDimCalculator.total_load_size = blockDimCalculator.amat_size + blockDimCalculator.bmat_size;
    blockDimCalculator.tmp_value = n0;
  }
}

void UpdateBlockDimCalculator(BlockDimCalculator &blockDimCalculator)
{
  if (blockDimCalculator.total_load_size > blockDimCalculator.tmp_load_size) {
      blockDimCalculator.bmat_size = blockDimCalculator.tmp_bmat_size;
      blockDimCalculator.amat_size = blockDimCalculator.tmp_amat_size;
      blockDimCalculator.total_load_size = blockDimCalculator.tmp_load_size;
      blockDimCalculator.tmp_value = 0;
  }
}

void AL1FullLoadBlock(const CoreStatus &coreStatus, BlockDimCalculator &blockDimCalculator, int64_t &m0)
{
  if (m0 >= 1) {
    while (coreStatus.m % m0 != 0) {
      m0--;
    }
    blockDimCalculator.tmp_amat_size = blockDimCalculator.ori_amat_size;
    blockDimCalculator.tmp_bmat_size = coreStatus.n * MathUtil::CeilDivision(blockDimCalculator.ori_amat_size,  m0);
    blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_amat_size + blockDimCalculator.tmp_bmat_size;
    UpdateBlockDimCalculator(blockDimCalculator);
  }
}

void NeitherFullLoadBlock(const CoreStatus &coreStatus, BlockDimCalculator &blockDimCalculator,
                          const vector<int64_t> &nFactorTwoCandidates,
                          const vector<int64_t> &mFactorTwoCandidates,
                          const BatchmatmulRunParas &run_params)
{
  for (auto const &n0: nFactorTwoCandidates) {
    if (n0 <= 0) {
      continue;
    }
    int64_t max_m0 = PlatformInfo::GetInstance().l0c_size / (kKbytes * n0);
    int64_t m0_arr[kCandidateLen] = {0};
    GetTwoFactors(m0_arr, max_m0, coreStatus.m, max_m0);
    for (auto const &m0: m0_arr) {
      if (m0 <= 0) {
        continue;
      }
      blockDimCalculator.tmp_amat_size = blockDimCalculator.ori_amat_size * MathUtil::CeilDivision(coreStatus.n, n0);
      blockDimCalculator.tmp_bmat_size = coreStatus.n * MathUtil::CeilDivision(blockDimCalculator.ori_amat_size, m0);
      blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_amat_size + blockDimCalculator.tmp_bmat_size;
      UpdateBlockDimCalculator(blockDimCalculator);
    }
  }
  for (auto const &m0: mFactorTwoCandidates) {
    if (m0 <= 0) {
      continue;
    }
    int64_t max_n0 = PlatformInfo::GetInstance().l0c_size / (kKbytes * m0);
    if (PlatformInfo::GetInstance().support_l0c2out() && run_params.bias_flag) {
      max_n0 = PlatformInfo::GetInstance().bt_size / kBlockSize / kFp32Bytes;
    }
    int64_t n0_arr[kCandidateLen] = {0};
    GetTwoFactors(n0_arr, max_n0, coreStatus.n, max_n0);
    for (auto const &n0: n0_arr) {
      if (n0 <= 0) {
        continue;
      }
      blockDimCalculator.tmp_amat_size = blockDimCalculator.ori_amat_size * MathUtil::CeilDivision(coreStatus.n, n0);
      blockDimCalculator.tmp_bmat_size = coreStatus.n * MathUtil::CeilDivision(blockDimCalculator.ori_amat_size, m0);
      blockDimCalculator.tmp_load_size = blockDimCalculator.tmp_amat_size + blockDimCalculator.tmp_bmat_size;
      UpdateBlockDimCalculator(blockDimCalculator);
    }
  }
}

void GetBlockDimHelper(CoreStatus &coreStatus, BlockDimCalculator &blockDimCalculator,
                       const std::vector<vector<int64_t>> &m0s, const std::vector<vector<int64_t>> &n0s,
                       const BatchmatmulParas &params)
{
  const BatchmatmulRunParas &run_params = *(params.run_params);
  int64_t bFactor = blockDimCalculator.batch_dim_array[blockDimCalculator.batch_idx];
  int64_t nFactor = blockDimCalculator.n_dim_array[blockDimCalculator.n_idx];
  blockDimCalculator.tmp_core_use = bFactor * nFactor;
  bool need_cal_load_size =
      blockDimCalculator.tmp_core_use > PlatformInfo::GetInstance().core_num || blockDimCalculator.tmp_core_use == 0;
  if (need_cal_load_size) {
    return;
  }
  for (int64_t mIdx = 0; mIdx < blockDimCalculator.m_dim_cnt; mIdx++) {
    int64_t mFactor = blockDimCalculator.m_dim_array[mIdx];
    blockDimCalculator.tmp_core_use = bFactor * nFactor * mFactor;
    need_cal_load_size = mFactor == 0 || blockDimCalculator.tmp_core_use > PlatformInfo::GetInstance().core_num;
    if (need_cal_load_size) {
      continue;
    }
    for (int64_t kIdx = 0; kIdx < blockDimCalculator.k_dim_cnt; kIdx++) {
      int64_t kFactor = blockDimCalculator.k_dim_array[kIdx];
      blockDimCalculator.tmp_core_use = bFactor * nFactor * mFactor * kFactor;
      need_cal_load_size = kFactor == 0 || blockDimCalculator.tmp_core_use > PlatformInfo::GetInstance().core_num;
      if (need_cal_load_size) {
        continue;
      }
      blockDimCalculator.k_num = run_params.k / kFactor * kBlockSize * reducedBlockSize;
      blockDimCalculator.k_bytes = blockDimCalculator.k_num * inputDtypeBytes;
      coreStatus.batch = MathUtil::CeilDivision(run_params.batch, bFactor);
      coreStatus.m = MathUtil::CeilDivision(run_params.m, mFactor);
      coreStatus.n = MathUtil::CeilDivision(run_params.n, nFactor);
      if ((run_params.m_quant_check && coreStatus.m % kNumTwo != 0) ||
        (run_params.n_quant_check && coreStatus.n % kNumTwo != 0)) {
        return;
      }
      coreStatus.k = run_params.k / kFactor;
      if (run_params.k_mapped != run_params.k && kIdx < kIdxTwo) {
        blockDimCalculator.k_num = run_params.k_mapped / kFactor * kNumTwo * kBlockSize * reducedBlockSize;
        coreStatus.k = run_params.k_mapped / kFactor * kNumTwo;
      }
      // load size of A matrix is batch * m
      // load size of B matrix is n
      blockDimCalculator.ori_amat_size = coreStatus.batch * coreStatus.m;
      blockDimCalculator.ori_bmat_size = run_params.b_have_batch ? coreStatus.batch * coreStatus.n : coreStatus.n;
      blockDimCalculator.amat_size = blockDimCalculator.ori_amat_size;
      blockDimCalculator.bmat_size = blockDimCalculator.ori_bmat_size;
      blockDimCalculator.total_load_size = blockDimCalculator.amat_size + blockDimCalculator.bmat_size;
      blockDimCalculator.tmp_value = 0;
      if (blockDimCalculator.total_load_size * blockDimCalculator.k_bytes > PlatformInfo::GetInstance().l1_size) {
        blockDimCalculator.total_load_size = INT64_MAX;
        // BL1 full load
        int64_t n0 = min(min((PlatformInfo::GetInstance().l1_size / inputDtypeBytes - kBlockSize * reducedBlockSize) /
                              blockDimCalculator.k_num, coreStatus.n), kMaxFactor);
        BL1FullLoadBlock(coreStatus, blockDimCalculator, n0, run_params.b_have_batch);
        // AL1 full load
        int64_t m0 = min(min((PlatformInfo::GetInstance().l1_size / inputDtypeBytes - kBlockSize * reducedBlockSize) /
                              blockDimCalculator.k_num / blockDimCalculator.ori_amat_size,
                              blockDimCalculator.ori_amat_size), kMaxFactor);
        AL1FullLoadBlock(coreStatus, blockDimCalculator, m0);
        // neither full load max_m max_n
        // closest m and n
        NeitherFullLoadBlock(coreStatus, blockDimCalculator, n0s[nFactor], m0s[mFactor], run_params);
      }
      if (run_params.k_mapped != run_params.k) {
        blockDimCalculator.total_load_size *= coreStatus.k;
      }
      // updateSolution: bool whether update to a new block factor solution
      // has smaller LoadSize or the same LoadSize but batch
      bool update_condition_loadsize = blockDimCalculator.total_load_size < blockDimCalculator.min_load_size;
      bool update_condition_batch_n_dim = (blockDimCalculator.total_load_size == blockDimCalculator.min_load_size) &&
        ((blockDimCalculator.n_dim_factor * blockDimCalculator.batch_dim_factor < bFactor * nFactor) ||
        (blockDimCalculator.n_dim_factor * blockDimCalculator.batch_dim_factor == bFactor * nFactor &&
        blockDimCalculator.batch_dim_factor < bFactor));

      if (update_condition_loadsize || update_condition_batch_n_dim) {
        blockDimCalculator.min_load_size = blockDimCalculator.total_load_size;
        blockDimCalculator.n_dim_factor = nFactor;
        blockDimCalculator.batch_dim_factor = bFactor;
        blockDimCalculator.m_dim_factor = mFactor;
        blockDimCalculator.k_dim_factor = kFactor;
        blockDimCalculator.core_use = blockDimCalculator.tmp_core_use;
        blockDimCalculator.final_value = blockDimCalculator.tmp_value;
      }
    }
  }
}

void GetBandwidth(const int64_t &use_out_buffer_size, int64_t &hbm_bandwidth, int64_t &l2_bandwidth,
                  int64_t &cur_bandwidth) {
  int64_t abs_core_num_8 = abs(PlatformInfo::GetInstance().core_num - kMinSplitKCoreNum);
  int64_t abs_core_num_32 = abs(PlatformInfo::GetInstance().core_num - 32);  // 32 is Ascend910A's core num
  if (abs_core_num_8 < abs_core_num_32) {
    hbm_bandwidth = kHbmBandwidth8Core;
    l2_bandwidth = kL2Bandwidth8Core;
  } else {
    hbm_bandwidth = kHbmBandwidth32Core;
    l2_bandwidth = kL2Bandwidth32Core;
  }
  cur_bandwidth = use_out_buffer_size < PlatformInfo::GetInstance().l2_size ? l2_bandwidth : hbm_bandwidth;
}

void ComputePerfSplitK(const int64_t block_dims[], const int64_t single_core_shape[], int64_t &min_cost,
                       const BatchmatmulRunParas &run_params, CoreStatus &coreStatus)
{
  int64_t m_dim = block_dims[0];
  int64_t k_dim = block_dims[1];
  int64_t n_dim = block_dims[kIdxTwo];
  int64_t batch_dim_max = block_dims[kIdxThree];
  if (k_dim * n_dim * m_dim > PlatformInfo::GetInstance().core_num) {
    return;
  }
  for (int64_t batch_dim = 1; batch_dim <= batch_dim_max; batch_dim++) {
    if (k_dim * n_dim * m_dim * batch_dim > PlatformInfo::GetInstance().core_num) {
      return;
    }
    int64_t single_core_m = single_core_shape[0];
    int64_t single_core_k = single_core_shape[1];
    int64_t single_core_n = single_core_shape[kIdxTwo];
    int64_t single_core_batch = run_params.batch / batch_dim;
    int64_t atomic_add_bw_lose = k_dim == 1 ? 1 : kNumTwo;
    int64_t mte3_cost = k_dim * (single_core_batch * single_core_m * single_core_n * kFp32Bytes) * atomic_add_bw_lose;
    int64_t base_load_cost =
      single_core_batch * (single_core_m * single_core_k + single_core_k * single_core_n) * inputDtypeBytes;
    int64_t b_repeat_load_cost = (batch_dim * m_dim - 1) * single_core_k * single_core_n * inputDtypeBytes;
    int64_t a_repeat_load_cost = (batch_dim * n_dim - 1) * single_core_k * single_core_m * inputDtypeBytes;
    int64_t cur_cost = base_load_cost + mte3_cost + a_repeat_load_cost + b_repeat_load_cost;
    if (cur_cost < min_cost) {
      min_cost = cur_cost;
      coreStatus.k_dim = k_dim;
    }
  }
}

void GetSplitKdim(const string &op_type, const BatchmatmulRunParas &run_params, CoreStatus &coreStatus) {
  // support multi cores slicing along k dim
  // get batch_dim, m_dim, n_dim and k_dim
  // batch_dim, m_dim, n_dim, k_dim is a factor of input batch, m, n, k
  OP_LOGD(op_type.c_str(), "GetSplitKdim input shape batch:%ld, m:%ld, k:%ld, n:%ld",
          run_params.batch, run_params.m, run_params.k, run_params.n);
  if (PlatformInfo::GetInstance().core_num < kMinSplitKCoreNum) {
    coreStatus.k_dim = 1;
    OP_LOGD(op_type.c_str(), "CORENUM < 8 so multi-core slicing factor k_dim:%ld", coreStatus.k_dim);
    return;
  }
  int64_t use_out_buffer_size = inputDtypeBytes * run_params.batch *
      (run_params.m * run_params.k + run_params.k * run_params.n + run_params.m * run_params.n);
  int64_t cur_bandwidth = 0;
  int64_t hbm_bandwidth = 0;
  int64_t l2_bandwidth = 0;
  GetBandwidth(use_out_buffer_size, hbm_bandwidth, l2_bandwidth, cur_bandwidth);
  int64_t min_cost = PlatformInfo::GetInstance().core_num * use_out_buffer_size / hbm_bandwidth * cur_bandwidth;
  int64_t batch_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.batch);
  int64_t m_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.m);
  int64_t k_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.k);
  int64_t n_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.n);
  int64_t block_dims[kNumFour] = {1, 1, 1, 1};
  int64_t single_core_shape[kNumFour] = {run_params.m, run_params.k, run_params.n, run_params.batch};
  block_dims[kIdxThree] = batch_dim_max;
  for (int64_t k = 1; k <= k_dim_max; k++) {
    for (int64_t n = 1; n <= n_dim_max; n++) {
      if (k * n > PlatformInfo::GetInstance().core_num) {
        break;
      }
      for (int64_t m = 1; m <= m_dim_max; m++) {
        block_dims[0] = m;
        block_dims[1] = k;
        block_dims[kIdxTwo] = n;
        single_core_shape[0] = run_params.m / m;
        single_core_shape[1] = run_params.k / k;
        single_core_shape[kIdxTwo] = run_params.n / n;
        ComputePerfSplitK(block_dims, single_core_shape, min_cost, run_params, coreStatus);
      }
    }
  }
  OP_LOGD(op_type.c_str(), "multi-core slicing factor k_dim:%ld", coreStatus.k_dim);
}

void UpdateMultiCore(const string &op_type, const BatchmatmulRunParas &run_params, CoreStatus &coreStatus,
                     const BlockDimCalculator &blockDimCalculator) {
  // Due to the modification of data amount in single-core, the number of multi-core needs to be updated.
  coreStatus.batch_dim = MathUtil::CeilDivision(run_params.batch, coreStatus.batch);
  coreStatus.n_dim = MathUtil::CeilDivision(run_params.n, coreStatus.n);
  coreStatus.m_dim = MathUtil::CeilDivision(run_params.m, coreStatus.m);
  coreStatus.k_dim = blockDimCalculator.k_dim_factor;
  OP_LOGD(op_type.c_str(),
          "Get final multi-core strategy: batch_dim:%ld, n_dim:%ld, m_dim:%ld, k_dim:%ld",
          coreStatus.batch_dim, coreStatus.n_dim, coreStatus.m_dim, coreStatus.k_dim);
}

int64_t GetFactorCnt(int64_t number, int64_t range_start, int64_t range_end) {
  int64_t real_end = std::min(number, range_end);
  int64_t cnt = 0;
  for (int64_t i = range_start; i <= real_end; i++) {
    if ((i != 0) && (number % i == 0)) {
      ++cnt;
    }
  }
  return cnt;
}

bool CheckFactorNumSatisfy(const int64_t number) {
  // Check whether numbers greater than 32 have more than 2 factors,
  // and numbers greater than 1024 have more than 4 factors
  if (number <= kMinFactorLimit) {
    return true;
  }
  int64_t factor_l0_cnt = GetFactorCnt(number, 1L, l0FactorLimit);
  bool satisfied = (factor_l0_cnt > kL0FactorNumLimit);

  if (number > kL1FactorLimit) {
    int64_t factor_l1_cnt = GetFactorCnt(number, l0FactorLimit + 1, kL1FactorLimit);
    satisfied = satisfied && (factor_l0_cnt + factor_l1_cnt > kL1FactorNumLimit);
  }
  return satisfied;
}

int64_t FindBestSingleCore(int64_t oriShape, int64_t &mappedShape, int64_t coreNum, bool isKDim)
{
  OP_TILING_CHECK(coreNum == 0, OP_LOGW("MatMul", "coreNum is zero, return oriShape"), return oriShape);
  // starting from the lower bound, find the value that satisfies the requiement of the number of factors as
  // the optimal solution
  // because non-factor strategies of k and m/n are different, no need to updata k_single_core here
  if (isKDim) {
    mappedShape = oriShape % coreNum == 0 ? oriShape : mappedShape;
    return mappedShape / coreNum;
  }
  // when multi-core is not opened, the original shape is used as the single-core data
  if (coreNum == 1) {
    return oriShape;
  }

  // in order to maximize the use of multi-core, take the maximum single-core that can fullfill all core as
  // the lower bound, and the current single-core data as the upper bound, find an optimal single-core data
  int64_t realSingleCore = MathUtil::CeilDivision(oriShape, coreNum);
  int64_t mappedSingleCore = MathUtil::CeilDivision(mappedShape, coreNum);
  int64_t best_single_core = realSingleCore;
  while (best_single_core != mappedSingleCore) {
    if (CheckFactorNumSatisfy(best_single_core)) {
      return best_single_core;
    }
    if (best_single_core < mappedSingleCore) {
      ++best_single_core;
    } else {
      --best_single_core;
    }
  }
  return best_single_core;
}

bool PreProcessMiniShape(const string &op_type, CoreStatus &coreStatus, BatchmatmulRunParas &run_params,
                         int64_t core_num, bool split_k_flag)
{
  // experience value for mini shape
  int64_t mini_l0c_threshold = PlatformInfo::GetInstance().l0c_size / kMinFractalSize / kFp16Bytes;
  int64_t mini_l0ab_threshold =
      PlatformInfo::GetInstance().l0a_size / (kBlockSize * reducedBlockSize) / inputDtypeBytes;
  // tend to use less cores for shapes with batch less than core_num and m/k/n can full load in aicore buffers
  // split_k is conflict with m/n shift_inwards
  bool special_scenario = false;
  if (run_params.n > kMinMte1Load) {
    special_scenario |= split_k_flag && ((run_params.n_mapped & (kMinMte1Load - 1)) != 0);
  }
  if (run_params.m > kMinMte1Load) {
    special_scenario |= split_k_flag && ((run_params.m_mapped & (kMinMte1Load - 1)) != 0);
  }
  if (run_params.batch <= core_num &&
      run_params.m * run_params.k <= mini_l0ab_threshold &&
      run_params.n * run_params.k <= mini_l0ab_threshold &&
      run_params.m * run_params.n * run_params.k <= mini_l0c_threshold && !special_scenario) {
    coreStatus.batch_dim = run_params.batch;
    coreStatus.n_dim = run_params.n <= kMinMte1Load ? 1 : run_params.n_mapped / kMinMte1Load;
    coreStatus.m_dim = run_params.m <= kMinMte1Load ? 1 : run_params.m_mapped / kMinMte1Load;
    int64_t k_dim_candidate[2] = {0}; // storage 2 factors of k around k_dim
    GetTwoFactors(k_dim_candidate, coreStatus.k_dim, run_params.k, core_num);
    coreStatus.k_dim = (run_params.k <= kMinMte1Load || !split_k_flag) ? 1 :
      (k_dim_candidate[1] > 1 ? k_dim_candidate[1] : k_dim_candidate[0]);
    coreStatus.batch = 1;
    // In unalign scene, single_core data must > 16 to avoid overwriting between different core
    // data of last core must > 16 because reg_buf borrow data forward
    if (run_params.format_out_nd && !PlatformInfo::GetInstance().support_l0c2out() &&
        run_params.ori_shape_m * run_params.ori_shape_n < kBlockSize) {
      coreStatus.batch_dim = 1;
      coreStatus.batch = run_params.batch;
    }
    OP_LOGD(op_type.c_str(),
        "Get final multi-core strategy: batch_dim:%ld, n_dim:%ld, m_dim:%ld, k_dim:%ld",
        coreStatus.batch_dim, coreStatus.n_dim, coreStatus.m_dim, coreStatus.k_dim);
    coreStatus.n = coreStatus.n_dim == 1 ? run_params.n :
      MathUtil::CeilDivision(run_params.n_mapped, coreStatus.n_dim);
    coreStatus.m = coreStatus.m_dim == 1 ? run_params.m :
      MathUtil::CeilDivision(run_params.m_mapped, coreStatus.m_dim);
    coreStatus.k = coreStatus.k_dim == 1 ? run_params.k :
      MathUtil::CeilDivision(run_params.k_mapped, coreStatus.k_dim);
    run_params.non_factor_k = run_params.k % coreStatus.k_dim == 0 ? false : true;
    if (run_params.m_quant_check && coreStatus.m % kNumTwo != 0) {
      coreStatus.m++;
    }
    if (run_params.n_quant_check && coreStatus.n % kNumTwo != 0) {
      coreStatus.n++;
    }
    OP_LOGD(op_type.c_str(), "get data in single core batch: %ld, n: %ld, m: %ld, k: %ld",
            coreStatus.batch, coreStatus.n, coreStatus.m, coreStatus.k);
    return true;
  }
  return false;
}

bool CheckTailBlock(int64_t n_dim, const BatchmatmulRunParas &run_params) {
  if (PlatformInfo::GetInstance().support_l0c2out() || !(run_params.format_out_nd &&
      run_params.ori_shape_n % kBlockSize != 0)) {
    return true;
  }
  if (run_params.ori_shape_n % (MathUtil::CeilDivision(run_params.n, n_dim) * kBlockSize) < kBlockSize) {
    return false;
  }
  return true;
}

void CoreStatusAdjust(const UbStatus &ubStatus, const BatchmatmulRunParas &run_params, CoreStatus &coreStatus) {
  if (run_params.m_quant_check && coreStatus.m % kNumTwo != 0) {
    coreStatus.m++;
    coreStatus.m_dim = MathUtil::CeilDivision(run_params.m, coreStatus.m);
  }
  if (run_params.n_quant_check && coreStatus.n % kNumTwo != 0) {
    coreStatus.n++;
    coreStatus.n_dim = MathUtil::CeilDivision(run_params.n, coreStatus.n);
  }
  if (!ubStatus.n_cub_tail_block_limit) {
    return;
  }
  // n_cub_tail_block_limit scene, exp: n_32 is 7, coreStatus.n = 2 is illegal, after n++, n is 3, n is still illegal
  while (coreStatus.n < run_params.n) {
    if (run_params.n % coreStatus.n != 1) {
      break;
    }
    coreStatus.n++;
  }
  coreStatus.n_dim = MathUtil::CeilDivision(run_params.n, coreStatus.n);
}

int64_t GetBlockDim(const string &op_type, const UbStatus &ubStatus, BatchmatmulParas &params, CoreStatus &coreStatus)
{
  // get batch_dim, k_dim, m_dim and n_dim for single core
  // support multi cores slicing along k_dim
  // single core batch_dim, m_dim, n_dim, k_dim is a factor of input batch, m, n, k
  BlockDimCalculator blockDimCalculator;
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  BatchmatmulRunParas &run_params = *(params.run_params);
  OP_LOGD(op_type.c_str(), "GetBlockDim input batch:%ld, m:%ld, k:%ld, n:%ld, k_dim:%ld", run_params.batch_mapped,
          run_params.m_mapped, run_params.k_mapped, run_params.n_mapped, coreStatus.k_dim);
  // multi-core strategy for mini shape's is different from other situations and requires preprocess
  if (PreProcessMiniShape(op_type, coreStatus, run_params, PlatformInfo::GetInstance().core_num,
                          compile_params.split_k_flag)) {
    // Due to the modification of data amount in single-core, the number of multi-core needs to be updated.
    CoreStatusAdjust(ubStatus, run_params, coreStatus);
    coreStatus.batch_dim = MathUtil::CeilDivision(run_params.batch, coreStatus.batch);
    coreStatus.n_dim = MathUtil::CeilDivision(run_params.n, coreStatus.n);
    coreStatus.m_dim = MathUtil::CeilDivision(run_params.m, coreStatus.m);
    coreStatus.k_dim = MathUtil::CeilDivision(run_params.k, coreStatus.k);
    OP_LOGD(op_type.c_str(), "Get final multi-core strategy: batch_dim:%ld, n_dim:%ld, m_dim:%ld, k_dim:%ld",
            coreStatus.batch_dim, coreStatus.n_dim, coreStatus.m_dim, coreStatus.k_dim);
    return 0;
  }
  // first get k_dim candidate
  std::vector<int64_t> kDimArray(PlatformInfo::GetInstance().core_num, 0);
  if (coreStatus.k_dim == 1) {
    kDimArray[0] = 1;
    blockDimCalculator.k_dim_cnt = 1;
    run_params.k_mapped = run_params.k;
  } else {
    if (run_params.k_mapped != run_params.k) {
      GetTwoFactors(kDimArray.data(), coreStatus.k_dim, run_params.k_mapped, PlatformInfo::GetInstance().core_num);
      blockDimCalculator.k_dim_cnt += kCandidateLen;
    }
    GetTwoFactors(kDimArray.data(), coreStatus.k_dim, run_params.k, PlatformInfo::GetInstance().core_num,
                  blockDimCalculator.k_dim_cnt);
    blockDimCalculator.k_dim_cnt += kCandidateLen;
  }
  std::vector<int64_t> batchDimArray(PlatformInfo::GetInstance().core_num, 0);
  std::vector<int64_t> nDimArray(PlatformInfo::GetInstance().core_num, 0);
  std::vector<int64_t> mDimArray(PlatformInfo::GetInstance().core_num, 0);
  GetShapeFactors(blockDimCalculator.batch_dim_cnt, batchDimArray.data(),
                  run_params.batch, run_params.batch_mapped, PlatformInfo::GetInstance().core_num);
  GetShapeFactors(blockDimCalculator.n_dim_cnt, nDimArray.data(),
                  run_params.n, run_params.n_mapped, PlatformInfo::GetInstance().core_num);
  GetShapeFactors(blockDimCalculator.m_dim_cnt, mDimArray.data(),
                  run_params.m, run_params.m_mapped, PlatformInfo::GetInstance().core_num);
  std::vector<vector<int64_t>> m0s(PlatformInfo::GetInstance().core_num + 1, vector<int64_t>(kCandidateLen, 0));
  std::vector<vector<int64_t>> n0s(PlatformInfo::GetInstance().core_num + 1, vector<int64_t>(kCandidateLen, 0));
  for (int64_t idx = 0; idx < blockDimCalculator.n_dim_cnt; idx++) {
    int64_t tmpNDim = nDimArray[idx];
    int64_t tmpNSingleCore = run_params.n_mapped / tmpNDim;
    if (run_params.n % tmpNDim == 0) {
      tmpNSingleCore = run_params.n / tmpNDim;
    }
    if (PlatformInfo::GetInstance().support_l0c2out() && run_params.bias_flag) {
      GetTwoFactors(n0s[tmpNDim].data(), kMNPntMaxWithBt, tmpNSingleCore, kMaxFactorWithBt);
    } else {
      GetTwoFactors(n0s[tmpNDim].data(), kMNPntMax, tmpNSingleCore, kMaxFactor);
    }
  }
  for (int64_t idx = 0; idx < blockDimCalculator.m_dim_cnt; idx++) {
    int64_t tmpMDim = mDimArray[idx];
    int64_t tmpMSingleCore = run_params.m_mapped / tmpMDim;
    if (run_params.m % tmpMDim == 0) {
      tmpMSingleCore = run_params.m / tmpMDim;
    }
    GetTwoFactors(m0s[tmpMDim].data(), kMNPntMax, tmpMSingleCore, kMaxFactor);
  }
  blockDimCalculator.n_dim_factor = 1;
  blockDimCalculator.batch_dim_factor = 1;
  blockDimCalculator.m_dim_factor = 1;
  blockDimCalculator.k_dim_factor = 1;
  blockDimCalculator.min_load_size = INT64_MAX;
  blockDimCalculator.batch_dim_array = batchDimArray.data();
  blockDimCalculator.m_dim_array = mDimArray.data();
  blockDimCalculator.n_dim_array = nDimArray.data();
  blockDimCalculator.k_dim_array = kDimArray.data();
  for (int64_t batch_idx = 0; batch_idx < blockDimCalculator.batch_dim_cnt; batch_idx++) {
    for (int64_t n_idx = 0; n_idx < blockDimCalculator.n_dim_cnt; n_idx++) {
      blockDimCalculator.batch_idx = batch_idx;
      blockDimCalculator.n_idx = n_idx;
      if (!CheckTailBlock(blockDimCalculator.n_dim_array[n_idx], run_params)) {
        continue;
      }
      GetBlockDimHelper(coreStatus, blockDimCalculator, m0s, n0s, params);
    }
  }
  OP_LOGD(op_type.c_str(), "Get multi-core strategy by minimize load_size: "
          "batch_dim:%ld, n_dim:%ld, m_dim:%ld, k_dim:%ld, final_value: %ld",
          blockDimCalculator.batch_dim_factor, blockDimCalculator.n_dim_factor,
          blockDimCalculator.m_dim_factor, blockDimCalculator.k_dim_factor,
          blockDimCalculator.final_value);
  coreStatus.batch = MathUtil::CeilDivision(run_params.batch_mapped, blockDimCalculator.batch_dim_factor);
  coreStatus.n = FindBestSingleCore(run_params.n, run_params.n_mapped, blockDimCalculator.n_dim_factor, false);
  coreStatus.m = FindBestSingleCore(run_params.m, run_params.m_mapped, blockDimCalculator.m_dim_factor, false);
  coreStatus.k = FindBestSingleCore(run_params.k, run_params.k_mapped, blockDimCalculator.k_dim_factor, true);
  CoreStatusAdjust(ubStatus, run_params, coreStatus);
  OP_LOGD(op_type.c_str(), "The amount of data in a single core is batch: %ld, n: %ld, m: %ld, k: %ld",
          coreStatus.batch, coreStatus.n, coreStatus.m, coreStatus.k);
  run_params.non_factor_k = run_params.k == run_params.k_mapped ? false : true;
  UpdateMultiCore(op_type, run_params, coreStatus, blockDimCalculator);
  return blockDimCalculator.final_value;
}

int64_t GetLoadSize(const CoreStatus &coreStatus, const L0Status &l0Status) {
  bool al1FullLoad = ((coreStatus.m * coreStatus.k + l0Status.n_l0 * l0Status.k_l0) *
                        kBlockSize * reducedBlockSize * inputDtypeBytes <= PlatformInfo::GetInstance().l1_size);
  bool bl1FullLoad = ((l0Status.m_l0 * l0Status.k_l0 + l0Status.n_l0 * coreStatus.k) * kBlockSize * reducedBlockSize *
                        inputDtypeBytes <= PlatformInfo::GetInstance().l1_size);
  bool bothFullLoad = ((coreStatus.m * coreStatus.k + l0Status.n_l0 * coreStatus.k) *
                        kBlockSize * reducedBlockSize * inputDtypeBytes <= PlatformInfo::GetInstance().l1_size);
  int64_t num0a = bl1FullLoad ? coreStatus.n : MathUtil::CeilDivision(coreStatus.m, l0Status.m_l0) * coreStatus.n;
  int64_t num0b = al1FullLoad ? coreStatus.m : MathUtil::CeilDivision(coreStatus.n, l0Status.n_l0) * coreStatus.m;
  if ((al1FullLoad && bl1FullLoad) && !bothFullLoad) {
    // 其中一个FULL load但是又不能全部FULL_LOAD的情况
    return min(coreStatus.n + MathUtil::CeilDivision(coreStatus.n, l0Status.n_l0) * coreStatus.m,
               coreStatus.m + MathUtil::CeilDivision(coreStatus.m, l0Status.m_l0) * coreStatus.n);
  }
  return num0a + num0b;
}

bool GetFinalMkn(SingleCoreStatus &singleCoreStatus, const CoreStatus &coreStatus, int64_t k0,
                 int64_t majorDimFactor, int64_t minorDimFactor)
{
  if (k0 == 0) {
    return false;
  }
  L0Status &l0Status = singleCoreStatus.l0Status;
  UbStatus ubStatus = singleCoreStatus.ubStatus;
  l0Status.m_l0 = (l0Status.max_axis_idx == 0) ? majorDimFactor : minorDimFactor;
  l0Status.n_l0 = (l0Status.max_axis_idx == 0) ? minorDimFactor : majorDimFactor;
  l0Status.k_l0 = k0;
  float tmpL0cUse = l0Status.m_l0 * l0Status.n_l0 * l0Status.db_l0c * kMinFractalSize * kFp32Bytes * 1.0f /
                    PlatformInfo::GetInstance().l0c_size;
  // kNumTwo means L0A and L0B double buffer is default-on.
  int64_t tmp_mte1_cycle =
      max(static_cast<int64_t>(kNumTwo * kNumThree),
          l0Status.m_l0 * l0Status.k_l0 * inputDtypeBytes * kBlockSize * reducedBlockSize / k1980Mte1L0aBandWidth) +
      max(static_cast<int64_t>(kNumTwo * kNumThree),
          l0Status.k_l0 * l0Status.n_l0 * inputDtypeBytes * kBlockSize * reducedBlockSize / k1980Mte1L0bBandWidth);
  int64_t tmp_mad_cycle = l0Status.m_l0 * l0Status.k_l0 * l0Status.n_l0 * madExpansionRate;
  int64_t tmpLoadSize = GetLoadSize(coreStatus, l0Status);
  int64_t tmpMte1Loop = ((l0Status.n_l0 != 1) ? l0Status.k_l0 : 1) + ((l0Status.k_l0 != 1) ? l0Status.m_l0 : 1);
  auto tmpMinCubSize = l0Status.m_l0 * ubStatus.db_cub * kMinFractalSize * kFp16Bytes *
                       ubStatus.cub_dtype_multi * (1 + ubStatus.fused_double_operand_num);

  auto condition1 = l0Status.final_ml0 == 0;
  auto condition2 =
      (tmpLoadSize < l0Status.final_load_size) || (tmp_mte1_cycle < tmp_mad_cycle && !l0Status.update_using_mte1);
  auto condition3 = (tmpLoadSize == l0Status.final_load_size && tmp_mad_cycle > l0Status.final_mul &&
                     tmp_mad_cycle * tmpL0cUse >= l0Status.final_mul * l0Status.final_l0c_use);
  auto condition4 = tmp_mad_cycle == l0Status.final_mul && tmpLoadSize == l0Status.final_load_size &&
                    tmpMte1Loop < l0Status.final_mte1Loop;
  // Considering pipeline parallelism between MTE1 and MAD
  auto conditionUb = tmpMinCubSize < PlatformInfo::GetInstance().ub_size;
  // Milan not consider ub
  auto conditionEnableUb = PlatformInfo::GetInstance().support_l0c2out() ? 1 : conditionUb;
  auto condition5 = conditionEnableUb &&
                    ((tmp_mte1_cycle < tmp_mad_cycle && l0Status.update_using_mte1) || !l0Status.update_using_mte1);
  bool validL0 = (condition1 || condition2 || condition3 || condition4) && condition5;
  if (validL0) {
    l0Status.final_ml0 = l0Status.m_l0;
    l0Status.final_kl0 = l0Status.k_l0;
    l0Status.final_nl0 = l0Status.n_l0;
    l0Status.final_load_size = tmpLoadSize;
    l0Status.final_l0c_use = tmpL0cUse;
    l0Status.final_mul = tmp_mad_cycle;
    l0Status.final_mte1_cycles = tmp_mte1_cycle;
    l0Status.final_mte1Loop = tmpMte1Loop;
    l0Status.update_using_mte1 = l0Status.update_using_mte1 || (tmp_mte1_cycle < tmp_mad_cycle);
  }
  return validL0;
}

MKNParasCombo GetParasCombo(int32_t index, int64_t blockValue, const BatchmatmulRunParas &params)
{
  map<int32_t, MKNParasCombo> parasComboMap;
  int64_t mn_max = PlatformInfo::GetInstance().l0c_size / kMinFractalSize / kFp32Bytes;
  int64_t max_n = PlatformInfo::GetInstance().bt_size / kBlockSize / kFp32Bytes;
  bool bias_bt = PlatformInfo::GetInstance().support_l0c2out() && params.bias_flag;
  if (blockValue == 0) {
    // db_l0a, db_l0b, db_l0c, max_mk, max_nk, max_mn, max_axis_idx, max_axis_num, max_axis_pnt, max_n
    MKNParasCombo comboZero = {2, 2, 2, 64, 64, mn_max / kDbOn, 0, 64, 11, bias_bt ? max_n / kDbOn : 64}; // L0C DB on
    MKNParasCombo comboOne = {2, 2, 1, 64, 64, mn_max, 0, 64, 16, bias_bt ? max_n : 64}; // L0C DB off
    parasComboMap = {{0, comboZero}, {1, comboOne}};
  } else { // BL1 Full Load Case
    MKNParasCombo comboZero = {2, 2, 2, 64, 64, mn_max / kDbOn, 1, 64, blockValue, bias_bt ? max_n / kDbOn : 64};
    MKNParasCombo comboOne = {2, 2, 1, 64, 64, mn_max, 1, 64, blockValue, bias_bt ? max_n : 64};
    parasComboMap = {{0, comboZero}, {1, comboOne}};
  }
  return parasComboMap[index];
}

void GetL0StatusFromParasCombo(L0Status &l0Status, int64_t *parasCombo)
{
  l0Status.SetInitLoadStatus();
  size_t kIdx = 0;
  l0Status.db_l0a = parasCombo[kIdx++];
  l0Status.db_l0b = parasCombo[kIdx++];
  l0Status.db_l0c = parasCombo[kIdx++];
  l0Status.max_mk = parasCombo[kIdx++];
  l0Status.max_nk = parasCombo[kIdx++];
  l0Status.max_mn = parasCombo[kIdx++];
  l0Status.max_axis_idx = parasCombo[kIdx++];
  l0Status.max_axis_num = parasCombo[kIdx++];
  l0Status.max_axis_pnt = parasCombo[kIdx++];
  l0Status.max_n = parasCombo[kIdx++];
  l0Status.max_axis_pnt = min(l0Status.max_axis_pnt, l0Status.max_axis_num);
}

void SetResFactors(const BatchmatmulRunParas &params, L0Factors &resFactors, const L0Status &l0Status)
{
  resFactors.final_ml0 = l0Status.final_ml0;
  resFactors.final_kl0 = l0Status.final_kl0;
  resFactors.final_nl0 = l0Status.final_nl0;
  resFactors.final_load_size = l0Status.final_load_size;
  resFactors.final_l0c_use = l0Status.final_l0c_use;
  resFactors.final_mte1Loop = l0Status.final_mte1Loop;
  resFactors.final_mul = l0Status.final_mul;
  resFactors.final_mte1_cycles = l0Status.final_mte1_cycles;
  // quant batch matmul v2 mc2 secne use fixed base MNK
  if (params.is_quant_batch_matmul_v3 && params.use_pre_ub) {
    // mc2 case: K is 5504 or 2048, N is 4096
    bool white_list_case1 = params.ori_shape_n == 4096 && (params.ori_shape_k == 5504 || params.ori_shape_k == 2048);
    // mc2 case: K is 4384, N is 3760
    bool white_list_case2 = params.ori_shape_n == 3760 && params.ori_shape_k == 4384;
    if (white_list_case1 || white_list_case2) {
      resFactors.final_ml0 = 16;  // 16: baseM 256
      resFactors.final_kl0 = 4;   // 4:  baseK 128
      resFactors.final_nl0 = 16;  // 16: baseN 256
    }
  }
}

void UpdateFinalMkn(const bool &updateL0Status, SingleCoreStatus &singleCoreStatus, const BatchmatmulRunParas &params) {
  if ((!params.n_quant_check && !params.m_quant_check) || !updateL0Status) {
    return;
  }
  L0Status &l0Status = singleCoreStatus.l0Status;
  int64_t m = params.m;
  int64_t n = params.n;
  if (PlatformInfo::GetInstance().support_l0c2out()) {
    m = params.m_quant_check ? MathUtil::Align(params.m, kNumTwo) : params.m;
    n = params.n_quant_check ? MathUtil::Align(params.n, kNumTwo) : params.n;
  }
  l0Status.n_l0 = min(l0Status.n_l0, n);
  l0Status.m_l0 = min(l0Status.m_l0, m);
  l0Status.final_ml0 = l0Status.m_l0;
  l0Status.final_nl0 = l0Status.n_l0;
}

bool CheckM0N0Combin(const BatchmatmulRunParas &params, const SingleCoreStatus &singleCoreStatus,
                     int64_t &majorDimFactor, int64_t &minorDimFactor) {
  int64_t n0 = majorDimFactor;
  int64_t m0 = minorDimFactor;
  const L0Status &l0Status = singleCoreStatus.l0Status;
  const UbStatus &ubStatus = singleCoreStatus.ubStatus;
  if (l0Status.max_axis_idx == 0) {
    m0 = majorDimFactor;
    n0 = minorDimFactor;
  }
  // consider bias table buffer
  int64_t max_n0 = PlatformInfo::GetInstance().bt_size / kBlockSize / kFp32Bytes / l0Status.db_l0c;
  if ((n0 > max_n0) && PlatformInfo::GetInstance().support_l0c2out() && params.bias_flag) {
    return false;
  }

  if (!ubStatus.n_cub_tail_block_limit) {
    return true;
  }
  // in nd_out and unalign, shape_n1 > 1 scene:
  // reg_buf is used for tail block to solve data overwriting between different block, cub_n1 must >= 2
  int64_t min_cub = 2;
  if (n0 < min_cub || params.n % n0 == 1) {
    return false;
  }
  int64_t n0_factor = n0;
  while (n0_factor > 1) {
    auto tmpCubSize = m0 * (n0_factor + 1) * ubStatus.db_cub * kMinFractalSize *
                      (1 + ubStatus.fused_double_operand_num);
    // make sure any factor of n0 stasify n_cub limit in unalign scene
    if (n0 % n0_factor == 0 && params.n % n0_factor != 1 && tmpCubSize < kUbFp16Size) {
      return true;
    }
    n0_factor--;
  }
  return false;
}

bool CheckL0Size(const BatchmatmulCompileParas& compile_params, const BatchmatmulRunParas& run_params,
                 int64_t m_l0, int64_t k0, int64_t n_l0) {
  int64_t sparse_l0b_multi_size = compile_params.sparse_4to2_flag ? 2 : 1;
  int64_t k0_a_size = ((run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) && run_params.trans_a_flag) ||
                       compile_params.sparse_4to2_flag) ? MathUtil::Align(k0, 2) : k0;
  int64_t l0a_buffer_size = k0_a_size * reducedBlockSize * m_l0 * kBlockSize * inputDtypeBytes * kDbOn;
  int64_t k0_b_size = (run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) &&
                       (run_params.trans_a_flag || !run_params.trans_b_flag)) ? MathUtil::Align(k0, 2) : k0;
  int64_t l0b_buffer_size = MathUtil::CeilDivision((k0_b_size * reducedBlockSize * n_l0 * kBlockSize *
                                                    inputDtypeBytes * kDbOn), sparse_l0b_multi_size);
  return (l0a_buffer_size <= PlatformInfo::GetInstance().l0a_size &&
          l0b_buffer_size <= PlatformInfo::GetInstance().l0b_size);
}

void GetAllFactorsCand(const CoreStatus &coreStatus, SingleCoreStatus &singleCoreStatus, L0Status &l0Status,
                       const BatchmatmulParas &params) {
  const BatchmatmulRunParas& run_params = *(params.run_params);
  const BatchmatmulCompileParas& compile_params = *(params.compile_params);
  int64_t majorDim = coreStatus.m;
  int64_t minorDim = coreStatus.n;
  int64_t majorDimK = l0Status.max_mk;
  int64_t minorDimK = l0Status.max_nk;
  int64_t max_n = l0Status.max_n;
  if (l0Status.max_axis_idx != 0) {
    majorDim = coreStatus.n;
    minorDim = coreStatus.m;
    majorDimK = l0Status.max_nk;
    minorDimK = l0Status.max_mk;
  }
  std::vector<int64_t> majorDimFactors(majorDim, 0);
  if (l0Status.max_axis_idx != 0 && PlatformInfo::GetInstance().support_l0c2out() && run_params.bias_flag) {
    l0Status.max_axis_pnt = min(l0Status.max_axis_pnt, max_n);
    l0Status.max_axis_num = min(l0Status.max_axis_num, max_n);
  }
  GetAllFactors(majorDimFactors.data(), l0Status.max_axis_pnt, majorDim);
  if (l0Status.max_axis_idx != 0 && singleCoreStatus.ubStatus.n_cub_tail_block_limit) {
    // n_cub need > 1, majorDimFactors[0] == 1 means has no legal factor, majorDim maybe a big prime number
    while (majorDimFactors[0] == 1) {
      GetAllFactors(majorDimFactors.data(), l0Status.max_axis_pnt, ++majorDim);
    }
  }
  bool major_quant_check = l0Status.max_axis_idx != 0 ? run_params.n_quant_check : run_params.m_quant_check;
  bool minor_quant_check = l0Status.max_axis_idx != 0 ? run_params.m_quant_check : run_params.n_quant_check;
  for (auto &majorDimFactor: majorDimFactors) {
    if (majorDimFactor == 0 || (major_quant_check && majorDimFactor % kNumTwo != 0)) {
      continue;
    }
    int64_t minorFactorMax = min(l0Status.max_mn / majorDimFactor, minorDimK);
    std::vector<int64_t> minorDimFactors(minorDim, 0);
    if (l0Status.max_axis_idx == 0 && PlatformInfo::GetInstance().support_l0c2out() && run_params.bias_flag) {
      minorFactorMax = min(minorFactorMax, max_n);
    }
    GetAllFactors(minorDimFactors.data(), minorFactorMax, minorDim);
    if (l0Status.max_axis_idx == 0 && singleCoreStatus.ubStatus.n_cub_tail_block_limit) {
      while (minorDimFactors[0] == 1) {
        GetAllFactors(minorDimFactors.data(), minorFactorMax, ++minorDim);
      }
    }
    for (auto &minorDimFactor: minorDimFactors) {
      if (minorDimFactor == 0 || (minor_quant_check && minorDimFactor % kNumTwo != 0)) {
        continue;
      }
      if (!CheckM0N0Combin(run_params, singleCoreStatus, majorDimFactor, minorDimFactor)) {
        continue;
      }
      int64_t k0Max = min(majorDimK / majorDimFactor, minorDimK / minorDimFactor);
      std::vector<int64_t> k0Factors(coreStatus.k, 0);
      GetAllFactors(k0Factors.data(), k0Max, coreStatus.k);
      for (auto &k0 : k0Factors) {
        // Check if the buffer size allocated exceed the hardware buffer size in Float Mode
        if (run_params.dtype_a != static_cast<int32_t>(ge::DT_FLOAT16) || compile_params.sparse_4to2_flag) {
          int64_t m_l0 = minorDimFactor;
          int64_t n_l0 = majorDimFactor;
          if (l0Status.max_axis_idx == 0) {
            n_l0 = minorDimFactor;
            m_l0 = majorDimFactor;
          }
          if (!CheckL0Size(compile_params, run_params, m_l0, k0, n_l0) ||
              (compile_params.sparse_4to2_flag && (k0 % 2 != 0 && k0 != 1))) { // 2 kl0 must even multiple
            continue;
          }
        }
        bool updateL0Status = GetFinalMkn(singleCoreStatus, coreStatus, k0, majorDimFactor, minorDimFactor);
        UpdateFinalMkn(updateL0Status, singleCoreStatus, run_params);
      }
    }
  }
}

void ResetL0StatusParams(L0Status &l0Status, const BatchmatmulRunParas &run_params) {
  l0Status.final_ml0 = run_params.is_quant_batch_matmul_v3 ? 1 : l0Status.final_ml0;
  l0Status.final_kl0 = run_params.is_quant_batch_matmul_v3 ? 1 : l0Status.final_kl0;
  l0Status.final_nl0 = run_params.is_quant_batch_matmul_v3 ? 1 : l0Status.final_nl0;
}

void GetL0FactorsCand(L0Factors &resFactors, const CoreStatus &coreStatus,
                      SingleCoreStatus &singleCoreStatus, int64_t *parasCombo, const BatchmatmulParas &params) {
  const BatchmatmulRunParas& run_params = *(params.run_params);
  const BatchmatmulCompileParas& compile_params = *(params.compile_params);
  L0Status &l0Status = singleCoreStatus.l0Status;
  GetL0StatusFromParasCombo(l0Status, parasCombo);
  int64_t majorDim = coreStatus.m;
  int64_t minorDim = coreStatus.n;
  int64_t majorDimK = l0Status.max_mk;
  int64_t minorDimK = l0Status.max_nk;
  int64_t max_n = l0Status.max_n;
  if (l0Status.max_axis_idx != 0) {
    majorDim = coreStatus.n;
    minorDim = coreStatus.m;
    majorDimK = l0Status.max_nk;
    minorDimK = l0Status.max_mk;
  }
  int64_t majorDimFactors[kCandidateLen] = {0};
  // n dim condition
  if (l0Status.max_axis_idx != 0 && PlatformInfo::GetInstance().support_l0c2out() && run_params.bias_flag) {
    GetTwoFactors(majorDimFactors, min(l0Status.max_axis_pnt, max_n), majorDim, min(l0Status.max_axis_num, max_n));
  } else {
    GetTwoFactors(majorDimFactors, l0Status.max_axis_pnt, majorDim, l0Status.max_axis_num);
  }
  bool major_quant_check = l0Status.max_axis_idx != 0 ? run_params.n_quant_check : run_params.m_quant_check;
  bool minor_quant_check = l0Status.max_axis_idx != 0 ? run_params.m_quant_check : run_params.n_quant_check;
  for (auto &majorDimFactor: majorDimFactors) {
    if (majorDimFactor == 0 || (major_quant_check && majorDimFactor % kNumTwo != 0)) {
      continue;
    }
    int64_t minorFactorMax = min(l0Status.max_mn / majorDimFactor, minorDimK);
    int64_t minorDimFactors[kCandidateLen] = {0};
    if (l0Status.max_axis_idx == 0 && PlatformInfo::GetInstance().support_l0c2out() && run_params.bias_flag) {
      GetTwoFactors(minorDimFactors, min(minorFactorMax, max_n), minorDim, min(minorFactorMax, max_n));
    } else {
      GetTwoFactors(minorDimFactors, minorFactorMax, minorDim, minorFactorMax);
    }
    for (auto &minorDimFactor: minorDimFactors) {
      if (minorDimFactor == 0 || (minor_quant_check && minorDimFactor % kNumTwo != 0)) {
        continue;
      }
      if (!CheckM0N0Combin(run_params, singleCoreStatus, majorDimFactor, minorDimFactor)) {
        continue;
      }
      int64_t k0Max = min(majorDimK / majorDimFactor, minorDimK / minorDimFactor);
      int64_t k0Factors[kCandidateLen] = {0};
      GetTwoFactors(k0Factors, k0Max, coreStatus.k, k0Max);
      for (auto &k0 : k0Factors) {
        // Check if the buffer size allocated exceed the hardware buffer size in Float Mode
        if (run_params.dtype_a != static_cast<int32_t>(ge::DT_FLOAT16) || compile_params.sparse_4to2_flag) {
          int64_t m_l0 = minorDimFactor;
          int64_t n_l0 = majorDimFactor;
          if (l0Status.max_axis_idx == 0) {
            n_l0 = minorDimFactor;
            m_l0 = majorDimFactor;
          }
          if (!CheckL0Size(compile_params, run_params, m_l0, k0, n_l0) ||
              (compile_params.sparse_4to2_flag && (k0 % 2 != 0 && k0 != 1))) {
            continue;
          }
        }
        bool updateL0Status = GetFinalMkn(singleCoreStatus, coreStatus, k0, majorDimFactor, minorDimFactor);
        UpdateFinalMkn(updateL0Status, singleCoreStatus, run_params);
      }
    }
  }
  // there are no correct factors with two loops, so find all factors which are smaller than base
  if (l0Status.final_ml0 == 0 || l0Status.final_kl0 == 0 || l0Status.final_nl0 == 0) {
    ResetL0StatusParams(l0Status, run_params);
    GetAllFactorsCand(coreStatus, singleCoreStatus, l0Status, params);
  }
  SetResFactors(run_params, resFactors, l0Status);
}

void GetL0BatchFactor(const CoreStatus &coreStatus, L0Status &l0Status, const BatchmatmulRunParas &params,
                      bool non_factor_split = false) {
  // when L0A L0B L0C full load, L0 support multi batch
  bool l0_not_full_load_flag = (non_factor_split || l0Status.m_l0 < coreStatus.m ||
                                l0Status.n_l0 < coreStatus.n || l0Status.k_l0 < coreStatus.k);
  if (coreStatus.batch == 1 || l0_not_full_load_flag || params.do_not_multi_batch) {
    OP_LOGD("Multi Batch Fail", "coreStatus.m = %ld, and coreStatus.n = %ld and coreStatus.k = %ld"
            "l0Status.m_l0 = %ld, l0Status.n_l0 = %ld , l0Status.k_l0 = %ld, "
            "batch = %ld and l0_not_full_load_flag = %d, and do_not_multi_batch = %d",
            coreStatus.m, coreStatus.n, coreStatus.k, l0Status.m_l0, l0Status.n_l0, l0Status.k_l0,
            coreStatus.batch, l0_not_full_load_flag, params.do_not_multi_batch);
    return;
  }
  int64_t k_l0a = l0Status.k_l0;
  int64_t k_l0b = l0Status.k_l0;
  if (params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT)) {
    // Need to Align Kl0 to a even number while using Load3D
    k_l0a = params.trans_a_flag ? MathUtil::Align(l0Status.k_l0, kNumTwo) : k_l0a;
    k_l0b = (params.trans_a_flag || !params.trans_b_flag) ? MathUtil::Align(l0Status.k_l0, kNumTwo) : k_l0b;
  }
  int64_t l0a_use_size = l0Status.m_l0 * k_l0a * kBlockSize * reducedBlockSize * l0Status.db_l0a * inputDtypeBytes;
  int64_t l0b_use_size = l0Status.n_l0 * k_l0b * kBlockSize * reducedBlockSize * l0Status.db_l0b * inputDtypeBytes;
  int64_t l0c_use_size = l0Status.m_l0 * l0Status.n_l0 * kMinFractalSize * l0Status.db_l0c * kFp32Bytes;
  int64_t max_l0c_batch = PlatformInfo::GetInstance().l0c_size / l0c_use_size;
  int64_t max_l0a_batch = PlatformInfo::GetInstance().l0a_size / l0a_use_size;
  int64_t max_l0b_batch = PlatformInfo::GetInstance().l0b_size / l0b_use_size;
  l0Status.batch_l0 = min(min(min(max_l0a_batch, max_l0b_batch), max_l0c_batch), coreStatus.batch);
  l0Status.batch_l0 = max(l0Status.batch_l0, 1L);
  GetNearestFactor(coreStatus.batch, l0Status.batch_l0);
  if (l0Status.batch_l0 > 1) {
    l0Status.l0c_multi_batch = 1;
  }
}

void GetL0Factors(const string &op_type, const CoreStatus &coreStatus, int64_t blockValue,
                  SingleCoreStatus &singleCoreStatus, const BatchmatmulParas &params)
{
  // get m_l0, n_l0, k_l0 factor when singlecore m, n, k is know
  // m_l0, n_l0, k_l0 is a factor of single core m, n, k
  L0Status &l0Status = singleCoreStatus.l0Status;
  int64_t dbAOnBOnCOnIdx = 0;
  int64_t dbAOnBOnCOffIdx = 1;
  L0Factors resFactors[kL0ParasComboLen];
  const BatchmatmulRunParas& run_params = *(params.run_params);
  for (int32_t i = 0; i < kL0ParasComboLen; ++i) { // This forloop must be 2
    MKNParasCombo mknParasCombo = GetParasCombo(i, blockValue, run_params);
    GetL0FactorsCand(resFactors[i], coreStatus, singleCoreStatus, mknParasCombo.parasCombo, params);
  }
  // check both L0C utilization and loadsize to control LOC LOA LOB DB
  int64_t m0L0cDbOn = resFactors[dbAOnBOnCOnIdx].final_ml0;
  int64_t k0L0cDbOn = resFactors[dbAOnBOnCOnIdx].final_kl0;
  int64_t n0L0cDbOn = resFactors[dbAOnBOnCOnIdx].final_nl0;
  int64_t loadSizeL0cDbOn = resFactors[dbAOnBOnCOnIdx].final_load_size;
  int64_t mte1CyclesL0cDbOn = resFactors[dbAOnBOnCOnIdx].final_mte1_cycles;

  int64_t m0L0cDbOff = resFactors[dbAOnBOnCOffIdx].final_ml0;
  int64_t k0L0cDbOff = resFactors[dbAOnBOnCOffIdx].final_kl0;
  int64_t n0L0cDbOff = resFactors[dbAOnBOnCOffIdx].final_nl0;
  int64_t loadSizeL0cDbOff = resFactors[dbAOnBOnCOffIdx].final_load_size;
  int64_t mte1CyclesL0cDbOff = resFactors[dbAOnBOnCOffIdx].final_mte1_cycles;

  int64_t mte3_cost_db_on = m0L0cDbOn * n0L0cDbOn * kMinFractalSize * kFp16Bytes *
                            singleCoreStatus.ubStatus.cub_dtype_multi / k1980Mte3BandWidth;
  int64_t mte3_cost_db_off = m0L0cDbOff * n0L0cDbOff * kMinFractalSize * kFp16Bytes *
                             singleCoreStatus.ubStatus.cub_dtype_multi / k1980Mte3BandWidth;
  int64_t mad_cyles_db_on = max(m0L0cDbOn * k0L0cDbOn * n0L0cDbOn, static_cast<int64_t>(mte1CyclesL0cDbOn * 0.7));
  int64_t mad_cyles_db_off = max(m0L0cDbOff * k0L0cDbOff * n0L0cDbOff, static_cast<int64_t>(mte1CyclesL0cDbOff * 0.7));
  int64_t db_on_pipe_time = MathUtil::CeilDivision(coreStatus.m, m0L0cDbOn) *
                                MathUtil::CeilDivision(coreStatus.n, n0L0cDbOn) *
                                ((MathUtil::CeilDivision(coreStatus.k, k0L0cDbOn) - 1) * mad_cyles_db_on +
                            max(mad_cyles_db_on, mte3_cost_db_on));
  int64_t db_off_pipe_time = MathUtil::CeilDivision(coreStatus.m, m0L0cDbOff) *
                             MathUtil::CeilDivision(coreStatus.n, n0L0cDbOff) *
                             (MathUtil::CeilDivision(coreStatus.k, k0L0cDbOff) * mad_cyles_db_off + mte3_cost_db_off);
  db_on_pipe_time = db_on_pipe_time == 0 ? INT64_MAX : db_on_pipe_time;
  db_off_pipe_time = db_off_pipe_time == 0 ? INT64_MAX : db_off_pipe_time;

  if ((db_off_pipe_time < db_on_pipe_time) || (loadSizeL0cDbOff < loadSizeL0cDbOn)) {
    l0Status.db_l0c = kDbOff;
    l0Status.db_l0a = kDbOn;
    l0Status.db_l0b = kDbOn;
    l0Status.m_l0 = m0L0cDbOff;
    l0Status.k_l0 = k0L0cDbOff;
    l0Status.n_l0 = n0L0cDbOff;
  } else {
    l0Status.db_l0c = kDbOn;
    l0Status.db_l0a = kDbOn;
    l0Status.db_l0b = kDbOn;
    l0Status.m_l0 = m0L0cDbOn;
    l0Status.k_l0 = k0L0cDbOn;
    l0Status.n_l0 = n0L0cDbOn;
  }
  l0Status.db_cub = kDbOn;
  // Traditional tiling algorithm does not support non_factor_split
  bool non_factor_split = (run_params.batch_mapped != run_params.batch || run_params.m_mapped != run_params.m ||
                           run_params.n_mapped != run_params.n || run_params.k_mapped != run_params.k);
  GetL0BatchFactor(coreStatus, l0Status, run_params, non_factor_split);
  OP_LOGD(op_type.c_str(), "tiling m_l0:%ld, n_l0:%ld, k_l0:%ld, batch_l0:%ld",
          l0Status.m_l0, l0Status.n_l0, l0Status.k_l0, l0Status.batch_l0);
  OP_LOGD(op_type.c_str(), "tiling db_l0a:%ld, db_l0b:%ld, db_l0c:%ld",
          l0Status.db_l0a, l0Status.db_l0b, l0Status.db_l0c);
  OP_LOGD(op_type.c_str(), "tiling db_cub:%ld", l0Status.db_cub);
}

void GetABL1KAlignValue(const BatchmatmulRunParas &params, int64_t &ka_align_value, int64_t &kb_align_value) {
  if (params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT)) {
    // when in FP32 mode, k_a must be an even number if k-alignment is needed. So make ka_align_value as 2.
    ka_align_value = params.trans_a_flag ? 2 : 1;
    // Same as previous one, make kb_align_value as 2 when k-alignment is needed
    kb_align_value = (params.trans_a_flag || !params.trans_b_flag) ? 2 : 1;
  }
}

void GetABKL1Bound(const BatchmatmulRunParas &run_params, const L1Status &l1Status,
                   int64_t &kal1_bound, int64_t &kbl1_bound) {
  int64_t ka_align_value = 1;
  int64_t kb_align_value = 1;
  GetABL1KAlignValue(run_params, ka_align_value, kb_align_value);
  kal1_bound = MathUtil::Align(l1Status.kal1_16, ka_align_value);
  kbl1_bound = MathUtil::Align(l1Status.kbl1_16, kb_align_value);
  if (run_params.dtype_a != static_cast<int32_t>(ge::DT_FLOAT)) {
    return;
  }
  int64_t offset_kal1 = run_params.ori_shape_k - l1Status.kal1_16 * reducedBlockSize;
  // in shift_inward and fp32 scene
  // if offset_kal1 is not 16_align for the last k, the offset address will move left by 8
  kal1_bound = (offset_kal1 % (ka_align_value * reducedBlockSize) == 0 || (run_params.k % kb_align_value == 0))
               ? kal1_bound : MathUtil::Align(l1Status.kal1_16 + 1, ka_align_value);
  int64_t offset_kbl1 = run_params.ori_shape_k - l1Status.kbl1_16 * reducedBlockSize;
  kbl1_bound = (offset_kbl1 % (kb_align_value * reducedBlockSize) == 0  || (run_params.k % kb_align_value == 0))
               ? kbl1_bound : MathUtil::Align(l1Status.kbl1_16 + 1, kb_align_value);
}

int64_t GetL1Size(const BatchmatmulRunParas &params, const L1Status &l1Status, const L0Status &l0Status) {
  int64_t cur_al1_size = 0;
  int64_t cur_bl1_size = 0;
  int64_t channel_wise_l1_size = 0;
  int64_t al1_const = kBlockSize * reducedBlockSize * l1Status.db_al1 * inputDtypeBytes;
  int64_t bl1_const = kBlockSize * reducedBlockSize * l1Status.db_bl1 * inputDtypeBytes;
  int64_t channel_wise_l1_const = l1Status.channel_wise_times * kBlockSize * l1Status.db_bl1 * l0Status.dtype_bias;
  int64_t ka_align_value = 1;
  int64_t kb_align_value = 1;
  GetABL1KAlignValue(params, ka_align_value, kb_align_value);
  if (!CheckMulOverflow(l1Status.m_al1, l0Status.m_l0, cur_al1_size) ||
      !CheckMulOverflow(cur_al1_size, al1_const, cur_al1_size) ||
      !CheckMulOverflow(cur_al1_size,  MathUtil::Align(l1Status.kal1_16, ka_align_value), cur_al1_size)) {
    return 0;
  }
  if (!CheckMulOverflow(l1Status.n_bl1, l0Status.n_l0, cur_bl1_size) ||
      !CheckMulOverflow(cur_bl1_size, bl1_const, cur_bl1_size) ||
      !CheckMulOverflow(cur_bl1_size, MathUtil::Align(l1Status.kbl1_16, kb_align_value), cur_bl1_size)) {
    return 0;
  }
  if (l1Status.channel_wise_times > 0) {
    if (!CheckMulOverflow(l1Status.n_bl1, l0Status.n_l0, channel_wise_l1_size) ||
      !CheckMulOverflow(channel_wise_l1_size, channel_wise_l1_const, channel_wise_l1_size)) {
      return 0;
    }
  }
  return cur_al1_size + cur_bl1_size + channel_wise_l1_size + l1Status.element_wise_size;
}

int64_t CalL1MaxLen(int64_t res_l1_size, L1Status &l1_status, const L0Status &l0_status, int64_t align_value,
                    const L1TilingType axis_name) {
  int64_t axis_max_len = 1;
  res_l1_size -= l1_status.element_wise_size;
  if (axis_name == L1TilingType::KAL1_16) {
    axis_max_len = res_l1_size / (l1_status.m_al1 * l0_status.m_l0 * l1_status.db_al1 * kBlockSize * reducedBlockSize *
                                  inputDtypeBytes);
  }
  if (axis_name == L1TilingType::KBL1_16) {
    axis_max_len = res_l1_size / (l1_status.n_bl1 * l0_status.n_l0 * l1_status.db_bl1 * kBlockSize * reducedBlockSize *
                                  inputDtypeBytes);
  }
  axis_max_len = MathUtil::AlignDown(axis_max_len, align_value);
  if (axis_name == L1TilingType::M_AL1) {
    axis_max_len = res_l1_size / (MathUtil::Align(l1_status.kal1_16, align_value) * l0_status.m_l0 * l1_status.db_al1 *
                                  kBlockSize * reducedBlockSize * inputDtypeBytes);
  }
  if (axis_name == L1TilingType::N_BL1) {
    axis_max_len =
        res_l1_size / (MathUtil::Align(l1_status.kbl1_16, align_value) * l0_status.n_l0 * l1_status.db_bl1 *
                           kBlockSize * reducedBlockSize * inputDtypeBytes +
                       l1_status.channel_wise_times * l0_status.n_l0 * kBlockSize * reducedBlockSize * inputDtypeBytes);
  }
  return axis_max_len;
}

int64_t CalcQuantBmmTransLength(const L0Status &l0Status, L1Status &l1Status) {
  int64_t transLength = l0Status.m_l0 * kBlockSize * l1Status.kal1_16 * reducedBlockSize * inputDtypeBytes;
  if (l1Status.kal1_16 * reducedBlockSize % kFractalSize == 0 && l1Status.kal1_16 < reducedBlockSize) {
    transLength += l0Status.m_l0 * kBlockSize * l1Status.m_al1 * inputDtypeBytes;
  }
  return transLength;
}

void CalcQuantBmmKaL1(const L0Status &l0Status, L1Status &l1Status) {
  std::string op_type = "QuantBatchMatmul";
  int64_t ubSize = PlatformInfo::GetInstance().ub_size;
  if (ubSize <= 0) {
    OP_LOGE_WITHOUT_REPORT(op_type.c_str(), "EXCEPTION: the ub_size is not greater than 0");
    return;
  }
  int64_t halfUbSize = (uint64_t)ubSize >> 1;
  while (CalcQuantBmmTransLength(l0Status, l1Status) > halfUbSize) {
    l1Status.kal1_16 -= l0Status.k_l0;
  }
}

void L1StatusBothFullLoad(const BatchmatmulRunParas &params, const CoreStatus &coreStatus, const L0Status &l0Status,
                          L1Status &l1Status, int64_t res[][7]) {
  int64_t curL1Size = GetL1Size(params, l1Status, l0Status);
  if (curL1Size > 0 && curL1Size <= PlatformInfo::GetInstance().l1_size) {
    if (params.is_quant_batch_matmul_v3 && params.use_pre_ub &&
        CalcQuantBmmTransLength(l0Status, l1Status)  > (PlatformInfo::GetInstance().ub_size / kNumTwo)) {
      // ND2NZ need half of UB at most to store ND, the other half to store NZ
      return;
    }

    l1Status.both_full_load = true;
    l1Status.load_size = coreStatus.m + coreStatus.n;
    res[kIdxZero][kIdxZero] = l1Status.kal1_16;
    res[kIdxZero][kIdxOne] = l1Status.m_al1;
    res[kIdxZero][kIdxTwo] = l1Status.db_al1;
    res[kIdxZero][kIdxThree] = l1Status.kbl1_16;
    res[kIdxZero][kIdxFour] = l1Status.n_bl1;
    res[kIdxZero][kIdxFive] = l1Status.db_bl1;
    res[kIdxZero][kIdxSix] = l1Status.load_size;
  }
}

void L1StatusAl1FullLoad(const BatchmatmulRunParas &params, const CoreStatus &coreStatus, const L0Status &l0Status,
                         L1Status &l1Status, int64_t res[][7]) {
  int64_t curL1Size = 0;
  int64_t mRepeat = ops::CeilDiv(coreStatus.m, l0Status.m_l0);
  int64_t nRepeat = ops::CeilDiv(coreStatus.n, l0Status.n_l0);
  // Align value is used in FP32 in FP32 out data flow mode
  int64_t ka_align_value = 1;
  int64_t kb_align_value = 1;
  GetABL1KAlignValue(params, ka_align_value, kb_align_value);
  curL1Size = GetL1Size(params, l1Status, l0Status);
  if (curL1Size > 0 && curL1Size <= PlatformInfo::GetInstance().l1_size) {
    // ND2NZ need half of UB at most to store ND, the other half to store NZ
    if (params.is_quant_batch_matmul_v3 && params.use_pre_ub &&
        CalcQuantBmmTransLength(l0Status, l1Status) > (PlatformInfo::GetInstance().ub_size / kNumTwo)) {
      return;
    }

    l1Status.al1_full_load = true;
    l1Status.al1_size = MathUtil::Align(coreStatus.k, ka_align_value) * coreStatus.m *
                        kBlockSize * reducedBlockSize * inputDtypeBytes;
    l1Status.bl1_size = PlatformInfo::GetInstance().l1_size - l1Status.al1_size;
    l1Status.db_bl1 = kDbOn;
    if (GetL1Size(params, l1Status, l0Status) > PlatformInfo::GetInstance().l1_size) {
      l1Status.db_bl1 = kDbOff;
    }
    int64_t bias_size = l1Status.channel_wise_times * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize *
                        kFp16Bytes * l1Status.db_bl1;
    bias_size = PlatformInfo::GetInstance().support_l0c2out() ? bias_size : 0;
    l1Status.kbl1_16 =
        min(CalL1MaxLen((l1Status.bl1_size - bias_size), l1Status, l0Status, kb_align_value, L1TilingType::KBL1_16),
            coreStatus.k);
    l1Status.bl1_times = min(l1Status.kbl1_16 / l0Status.k_l0, l1Status.max_k_bl1);
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    if (l1Status.kbl1_16 == coreStatus.k) {
      l1Status.n_bl1 = min(CalL1MaxLen(l1Status.bl1_size, l1Status, l0Status, kb_align_value, L1TilingType::N_BL1),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    }
    bool invalid_l1_status = (l1Status.n_bl1 == 0 || l1Status.kbl1_16 == 0) ? true : false;
    int64_t possible_m_repeat = (l1Status.kbl1_16 == coreStatus.k) ? 1 : mRepeat;
    l1Status.load_size = invalid_l1_status ? INT64_MAX : (coreStatus.m + possible_m_repeat * coreStatus.n);
    res[kIdxOne][kIdxZero] = l1Status.kal1_16;
    res[kIdxOne][kIdxOne] = l1Status.m_al1;
    res[kIdxOne][kIdxTwo] = l1Status.db_al1;
    res[kIdxOne][kIdxThree] = l1Status.kbl1_16;
    res[kIdxOne][kIdxFour] = l1Status.n_bl1;
    res[kIdxOne][kIdxFive] = l1Status.db_bl1;
    res[kIdxOne][kIdxSix] = l1Status.load_size;
  }
}

void L1StatusBl1FullLoad(const BatchmatmulRunParas &params, const CoreStatus &coreStatus, const L0Status &l0Status,
                         L1Status &l1Status, int64_t res[][7]) {
  int64_t curL1Size = 0;
  int64_t mRepeat = ops::CeilDiv(coreStatus.m, l0Status.m_l0);
  int64_t nRepeat = ops::CeilDiv(coreStatus.n, l0Status.n_l0);
  curL1Size = GetL1Size(params, l1Status, l0Status);
  // Align value is used in FP32 in FP32 out data flow mode
  int64_t ka_align_value = 1;
  int64_t kb_align_value = 1;
  GetABL1KAlignValue(params, ka_align_value, kb_align_value);
  if (curL1Size > 0 && curL1Size <= PlatformInfo::GetInstance().l1_size) {
    l1Status.bl1_full_load = true;
    l1Status.bl1_size =
        MathUtil::Align(coreStatus.k, kb_align_value) * coreStatus.n * kBlockSize * reducedBlockSize * inputDtypeBytes;
    l1Status.al1_size = PlatformInfo::GetInstance().l1_size - l1Status.bl1_size;
    l1Status.db_al1 = kDbOn;
    if (GetL1Size(params, l1Status, l0Status) > PlatformInfo::GetInstance().l1_size) {
      l1Status.db_al1 = kDbOff;
    }
    int64_t bias_size = l1Status.channel_wise_times * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize *
                        l0Status.dtype_bias * l1Status.db_bl1;
    bias_size = PlatformInfo::GetInstance().support_l0c2out() ? bias_size : 0;
    l1Status.kal1_16 =
        min(CalL1MaxLen((l1Status.al1_size - bias_size), l1Status, l0Status, ka_align_value, L1TilingType::KAL1_16),
            coreStatus.k);
    if (params.is_quant_batch_matmul_v3 && params.use_pre_ub) {
      // ND2NZ need half of UB at most to store ND, the other half to store NZ
      l1Status.kal1_16 = min(l1Status.kal1_16, (PlatformInfo::GetInstance().ub_size >> 1) /
                                                   (l0Status.m_l0 * kBlockSize * reducedBlockSize * inputDtypeBytes));
      CalcQuantBmmKaL1(l0Status, l1Status);
    }

    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
    if (l1Status.kal1_16 == coreStatus.k) {
      l1Status.m_al1 =
          min(CalL1MaxLen(l1Status.al1_size - bias_size, l1Status, l0Status, ka_align_value, L1TilingType::M_AL1),
              l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
    bool invalid_l1_status = (l1Status.m_al1 == 0 || l1Status.kal1_16 == 0) ? true : false;
    int64_t possible_n_repeat =
        (coreStatus.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == coreStatus.k) ? 1 : nRepeat;
    l1Status.load_size = invalid_l1_status ? INT64_MAX : (coreStatus.n + possible_n_repeat * coreStatus.m);
    res[kIdxTwo][kIdxZero] = l1Status.kal1_16;
    res[kIdxTwo][kIdxOne] = l1Status.m_al1;
    res[kIdxTwo][kIdxTwo] = l1Status.db_al1;
    res[kIdxTwo][kIdxThree] = l1Status.kbl1_16;
    res[kIdxTwo][kIdxFour] = l1Status.n_bl1;
    res[kIdxTwo][kIdxFive] = l1Status.db_bl1;
    res[kIdxTwo][kIdxSix] = l1Status.load_size;
  }
}

void NeitherFullLoadDb(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                       const BatchmatmulRunParas &params, const int64_t &kbl1Db) {
  int64_t tmpKbl116 = l1Status.kbl1_16;
  l1Status.kbl1_16 = kbl1Db;
  if (GetL1Size(params, l1Status, l0Status) > PlatformInfo::GetInstance().l1_size) {
    l1Status.db_bl1 = kDbOff;
    if (GetL1Size(params, l1Status, l0Status) > PlatformInfo::GetInstance().l1_size) {
      l1Status.db_al1 = kDbOff;
    }
  }
  l1Status.kbl1_16 = coreStatus.k;
  bool bothDoubleBuffer = coreStatus.m != l0Status.m_l0 && coreStatus.k > l0Status.k_l0 &&
    GetL1Size(params, l1Status, l0Status) > PlatformInfo::GetInstance().l1_size;
  l1Status.kbl1_16 = tmpKbl116;
  if (bothDoubleBuffer) {
    l1Status.db_al1 = kDbOn;
    l1Status.db_bl1 = kDbOn;
    if (GetL1Size(params, l1Status, l0Status) > PlatformInfo::GetInstance().l1_size) {
      l1Status.db_bl1 = kDbOff;
      if (GetL1Size(params, l1Status, l0Status) > PlatformInfo::GetInstance().l1_size) {
        l1Status.db_al1 = kDbOff;
      }
    }
  }
}

void NeitherFullLoadMN(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                       const BatchmatmulRunParas &params)
{
  int64_t mRepeat = ops::CeilDiv(coreStatus.m, l0Status.m_l0);
  int64_t nRepeat = ops::CeilDiv(coreStatus.n, l0Status.n_l0);
  if (l0Status.dtype_bias == kFp32Bytes && l1Status.channel_wise_times > 0) {
    l1Status.channel_wise_times++;
  }
  int64_t bias_size =
      l1Status.channel_wise_times * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * kFp16Bytes * l1Status.db_bl1;
  bias_size = PlatformInfo::GetInstance().support_l0c2out() ? bias_size : 0;
  // Align value is used in FP32 in FP32 out data flow mode
  int64_t ka_align_value = 1;
  int64_t kb_align_value = 1;
  GetABL1KAlignValue(params, ka_align_value, kb_align_value);
  if (l0Status.k_l0 == coreStatus.k) {
    if (params.m_mapped > params.n_mapped) {
      l1Status.bl1_size = MathUtil::Align(coreStatus.k, ka_align_value) * l0Status.n_l0 * kBlockSize *
                          reducedBlockSize * l1Status.db_bl1 * inputDtypeBytes;
      l1Status.al1_size = PlatformInfo::GetInstance().l1_size - l1Status.bl1_size;
      l1Status.m_al1 =
          min(CalL1MaxLen(l1Status.al1_size - bias_size, l1Status, l0Status, ka_align_value, L1TilingType::M_AL1),
              l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
      l1Status.al1_size = MathUtil::Align(l1Status.kal1_16, ka_align_value) * l1Status.m_al1 * l0Status.m_l0 *
                          kBlockSize * reducedBlockSize * l1Status.db_al1 * inputDtypeBytes;
      l1Status.bl1_size = PlatformInfo::GetInstance().l1_size - l1Status.al1_size;
      l1Status.n_bl1 = min(CalL1MaxLen(l1Status.bl1_size, l1Status, l0Status, kb_align_value, L1TilingType::N_BL1),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
    } else {
      l1Status.al1_size = MathUtil::Align(coreStatus.k, ka_align_value) * l0Status.m_l0 * kBlockSize *
                          reducedBlockSize * l1Status.db_al1 * inputDtypeBytes;
      l1Status.bl1_size = PlatformInfo::GetInstance().l1_size - l1Status.al1_size;
      l1Status.n_bl1 = min(CalL1MaxLen(l1Status.bl1_size, l1Status, l0Status, kb_align_value, L1TilingType::N_BL1),
                           l1Status.max_n_bl1);
      GetNearestFactor(nRepeat, l1Status.n_bl1);
      l1Status.bl1_size = MathUtil::Align(coreStatus.k, kb_align_value) * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize *
                          reducedBlockSize * l1Status.db_bl1 * inputDtypeBytes;
      l1Status.al1_size = PlatformInfo::GetInstance().l1_size - l1Status.bl1_size;
      bias_size = bias_size * l1Status.n_bl1;
      l1Status.m_al1 =
          min(CalL1MaxLen(l1Status.al1_size - bias_size, l1Status, l0Status, ka_align_value, L1TilingType::M_AL1),
              l1Status.max_m_al1);
      GetNearestFactor(mRepeat, l1Status.m_al1);
    }
  }
}

void NeitherFullLoadKforNZ(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                           const BatchmatmulRunParas &params) {
  l1Status.kbl1_16 = coreStatus.k;
  int64_t bias_size = l1Status.channel_wise_times * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize *
                      l0Status.dtype_bias * l1Status.db_bl1;
  bias_size = PlatformInfo::GetInstance().support_l0c2out() ? bias_size : 0;
  // Align value is used in FP32 in FP32 out data flow mode
  int64_t ka_align_value = 1;
  int64_t kb_align_value = 1;
  GetABL1KAlignValue(params, ka_align_value, kb_align_value);
  if (GetL1Size(params, l1Status, l0Status) > 0 &&
      GetL1Size(params, l1Status, l0Status) <= PlatformInfo::GetInstance().l1_size) {
    l1Status.bl1_size = MathUtil::Align(coreStatus.k, kb_align_value) * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize *
                        reducedBlockSize * l1Status.db_bl1 * inputDtypeBytes;
    l1Status.al1_size = PlatformInfo::GetInstance().l1_size - l1Status.bl1_size;
    l1Status.kal1_16 =
        min(CalL1MaxLen(l1Status.al1_size - bias_size, l1Status, l0Status, ka_align_value, L1TilingType::KAL1_16),
            coreStatus.k);
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
  } else {
    // when NeitherFullLoadMN change the n_bl1 and m_al1
    int64_t perK = min((PlatformInfo::GetInstance().l1_size - bias_size - l1Status.element_wise_size) /
                         (l0Status.m_l0 * kBlockSize * reducedBlockSize * l1Status.db_al1 * inputDtypeBytes +
                           kBlockSize * l0Status.n_l0 * reducedBlockSize * l1Status.db_bl1 * inputDtypeBytes) /
                         l0Status.k_l0 * l0Status.k_l0,
                       coreStatus.k);
    int64_t bias_factor = params.bias_flag ? l1Status.n_bl1 * l0Status.n_l0 : 0;
    int64_t a_aligned_perK = MathUtil::Align(perK, ka_align_value);
    int64_t b_aligned_perK = MathUtil::Align(perK, kb_align_value);
    if (params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) &&
        !CheckL1Size(l1Status.m_al1 * l0Status.m_l0 * a_aligned_perK * l1Status.db_al1,
                     l1Status.n_bl1 * l0Status.n_l0 * b_aligned_perK * l1Status.db_bl1, bias_factor)) {
      perK -= 1;
    }
    int64_t perTimes = min(perK / l0Status.k_l0, max(l1Status.max_k_al1, l1Status.max_k_bl1));
    GetNearestFactor(l1Status.all_times, perTimes);
    perK = perTimes * l0Status.k_l0;
    l1Status.kal1_16 = perK;
    l1Status.kbl1_16 = perK;
  }
}

void NeitherFullLoadKforND(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                           const int &kmax_axis, const BatchmatmulRunParas &params)
{
  int64_t bias_size = l1Status.channel_wise_times * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize *
                      l0Status.dtype_bias * l1Status.db_bl1;
  bias_size = PlatformInfo::GetInstance().support_l0c2out() ? bias_size : 0;
  if (kmax_axis == kNumOne) {
    // first get k_al1, second get k_bl1
    l1Status.kbl1_16 = l0Status.k_l0;
    l1Status.bl1_size = l1Status.kbl1_16 * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * reducedBlockSize *
                        l1Status.db_bl1 * inputDtypeBytes;
    l1Status.al1_size = PlatformInfo::GetInstance().l1_size - l1Status.bl1_size;
    l1Status.kal1_16 =
        min((l1Status.al1_size - bias_size - l1Status.element_wise_size) /
                (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * inputDtypeBytes * reducedBlockSize),
            coreStatus.k);

    if (params.is_quant_batch_matmul_v3) {
      // ND2NZ need half of UB at most to store ND, the other half to store NZ
      l1Status.kal1_16 = min(l1Status.kal1_16, (PlatformInfo::GetInstance().ub_size / kNumTwo) /
                                                   (l0Status.m_l0 * kBlockSize * reducedBlockSize * inputDtypeBytes));
      CalcQuantBmmKaL1(l0Status, l1Status);
    }

    l1Status.al1_times = l1Status.kal1_16 / l0Status.k_l0;
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
    l1Status.al1_size = l1Status.kal1_16 * l1Status.m_al1 * l0Status.m_l0 * kBlockSize * reducedBlockSize *
                        l1Status.db_al1 * inputDtypeBytes;
    l1Status.bl1_size = PlatformInfo::GetInstance().l1_size - l1Status.al1_size;
    l1Status.kbl1_16 =
        min((l1Status.bl1_size - bias_size - l1Status.element_wise_size) /
                (l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * inputDtypeBytes * reducedBlockSize),
            coreStatus.k);
    l1Status.bl1_times = min(l1Status.kbl1_16 / l0Status.k_l0, l1Status.max_k_bl1);
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
  } else if (kmax_axis == kNumTwo) {
    // first get k_bl1, second get k_al1
    l1Status.kal1_16 = l0Status.k_l0;
    l1Status.al1_size = l1Status.kal1_16 * l1Status.m_al1 * l0Status.m_l0 * kBlockSize * reducedBlockSize *
                        l1Status.db_al1 * inputDtypeBytes;
    l1Status.bl1_size = PlatformInfo::GetInstance().l1_size - l1Status.al1_size;
    l1Status.kbl1_16 =
        min((l1Status.bl1_size - bias_size - l1Status.element_wise_size) /
                (l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * l1Status.db_bl1 * inputDtypeBytes * reducedBlockSize),
            coreStatus.k);
    l1Status.bl1_times = l1Status.kbl1_16 / l0Status.k_l0;
    GetNearestFactor(l1Status.all_times, l1Status.bl1_times);
    l1Status.kbl1_16 = l1Status.bl1_times * l0Status.k_l0;
    l1Status.bl1_size = l1Status.kbl1_16 * l1Status.n_bl1 * l0Status.n_l0 * kBlockSize * reducedBlockSize *
                        l1Status.db_bl1 * inputDtypeBytes;
    l1Status.al1_size = PlatformInfo::GetInstance().l1_size - l1Status.bl1_size;
    l1Status.kal1_16 =
        min((l1Status.al1_size - bias_size - l1Status.element_wise_size) /
                (l1Status.m_al1 * l0Status.m_l0 * kBlockSize * l1Status.db_al1 * inputDtypeBytes * reducedBlockSize),
            coreStatus.k);
    if (params.is_quant_batch_matmul_v3) {
      CalcQuantBmmKaL1(l0Status, l1Status);
    }
    l1Status.al1_times = min(l1Status.kal1_16 / l0Status.k_l0, l1Status.max_k_al1);
    GetNearestFactor(l1Status.all_times, l1Status.al1_times);
    l1Status.kal1_16 = l1Status.al1_times * l0Status.k_l0;
  }
}

void NeitherFullLoadK(const CoreStatus &coreStatus, const L0Status &l0Status, L1Status &l1Status,
                      const BatchmatmulRunParas &params)
{
  // 1 -> let k_al1 bigger, 2 -> let k_bl1 bigger, 0 -> no matter
  int kmax_axis = kNumZero;
  if (!params.trans_a_flag && !params.trans_b_flag) {
    kmax_axis = kNumOne;
  } else if (params.trans_a_flag && params.trans_b_flag) {
    kmax_axis = kNumTwo;
  } else if (!params.trans_a_flag && params.trans_b_flag) {
    kmax_axis = l0Status.m_l0 > l0Status.n_l0 ? kNumOne : kNumTwo;
  }
  // Not Support FP32 mode for NZ format and hardware with pre_ub
  if (params.use_pre_ub && kmax_axis != kNumZero) {
    NeitherFullLoadKforND(coreStatus, l0Status, l1Status, kmax_axis, params);
  } else {
    NeitherFullLoadKforNZ(coreStatus, l0Status, l1Status, params);
  }
}

void L1StatusNeitherFullLoad(const CoreStatus &coreStatus, const BatchmatmulParas &params, const L0Status &l0Status,
                             L1Status &l1Status, int64_t res[][7]) {
  int64_t mRepeat = ops::CeilDiv(coreStatus.m, l0Status.m_l0);
  int64_t nRepeat = ops::CeilDiv(coreStatus.n, l0Status.n_l0);
  int64_t kBl1Db = (coreStatus.m == l0Status.m_l0) ? l0Status.k_l0 : coreStatus.k;
  NeitherFullLoadDb(coreStatus, l0Status, l1Status, *(params.run_params), kBl1Db);
  NeitherFullLoadMN(coreStatus, l0Status, l1Status, *(params.run_params));
  NeitherFullLoadK(coreStatus, l0Status, l1Status, *(params.run_params));
  // k_al1 and k_bl1 must be a factor of each other
  if (l1Status.kal1_16 > l1Status.kbl1_16 && l1Status.kal1_16 % l1Status.kbl1_16 != 0) {
    while (l1Status.kal1_16 % l1Status.kbl1_16 != 0 || coreStatus.k % l1Status.kal1_16 != 0) {
      l1Status.kal1_16 -= 1;
    }
  } else if (l1Status.kal1_16 < l1Status.kbl1_16 && l1Status.kbl1_16 % l1Status.kal1_16 != 0) {
    while (l1Status.kbl1_16 % l1Status.kal1_16 != 0 || coreStatus.k % l1Status.kbl1_16 != 0) {
      l1Status.kbl1_16 -= 1;
    }
  }
  l1Status.load_size =
    ((coreStatus.m == l1Status.m_al1 * l0Status.m_l0 && l1Status.kal1_16 == coreStatus.k) ? 1 : nRepeat) *
      coreStatus.m + (l1Status.kbl1_16 == coreStatus.k ? 1 : mRepeat) * coreStatus.n;
  res[kIdxThree][kIdxZero] = l1Status.kal1_16;
  res[kIdxThree][kIdxOne] = l1Status.m_al1;
  res[kIdxThree][kIdxTwo] = l1Status.db_al1;
  res[kIdxThree][kIdxThree] = l1Status.kbl1_16;
  res[kIdxThree][kIdxFour] = l1Status.n_bl1;
  res[kIdxThree][kIdxFive] = l1Status.db_bl1;
  res[kIdxThree][kIdxSix] = l1Status.load_size;
}

void GetL1Factors(const string &op_type, const BatchmatmulParas &params, const CoreStatus &coreStatus,
                  const L0Status &l0Status, L1Status &l1Status)
{
  // get m_al1, n_bl1, kal1_16, kbl1_16 factors when L0, singlecore factor is know
  // get al1, bl1 double buffer factors

  int64_t mte1Loop = 50 / ((l0Status.n_l0 == 1 ? 1 : l0Status.k_l0) + (l0Status.k_l0 == 1 ? 1 : l0Status.m_l0));
  int64_t res[4][7] = {0};
  l1Status.all_times = coreStatus.k / l0Status.k_l0;
  l1Status.max_m_al1 = (coreStatus.m + l0Status.m_l0 - 1) / l0Status.m_l0;
  l1Status.max_n_bl1 = (coreStatus.n + l0Status.n_l0 - 1) / l0Status.n_l0;
  l1Status.max_k_al1 =
    max(mte1Loop, ((kMinMte1Load + l0Status.m_l0 - 1) / l0Status.m_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  l1Status.max_k_bl1 =
    max(mte1Loop, ((kMinMte1Load + l0Status.n_l0 - 1) / l0Status.n_l0 + l0Status.k_l0 - 1) / l0Status.k_l0);
  if (PlatformInfo::GetInstance().support_l0c2out() && params.run_params->bias_flag) {
    l1Status.channel_wise_times++;
  }
  // 2 means times of int64 quant scale to int32 bias
  l1Status.channel_wise_times += (params.compile_params->quant_scale * 2);
  // add/sub l1-fixpipe fusion
  l1Status.element_wise_size = params.compile_params->eltwise_src * l0Status.m_l0 * l0Status.n_l0 * kBlockSize *
                               kBlockSize * l0Status.db_l0c * outputDtypeBytes;
  // both AL1 and Bl1 full load
  int64_t both_full_load_factors[kL1FactorsLen] =
      {coreStatus.k, coreStatus.k, l1Status.max_m_al1, l1Status.max_n_bl1, kDbOff, kDbOff};
  // Need to consider L1 extension in FP32 Mode
  l1Status.SetStatus(both_full_load_factors);
  L1StatusBothFullLoad(*params.run_params, coreStatus, l0Status, l1Status, res);
  // only AL1 full load
  int64_t al1_full_load_factors[kL1FactorsLen] = {coreStatus.k, l0Status.k_l0, l1Status.max_m_al1, 1, kDbOff, kDbOff};
  l1Status.SetStatus(al1_full_load_factors);
  L1StatusAl1FullLoad(*params.run_params, coreStatus, l0Status, l1Status, res);
  // only BL1 full load
  int64_t bl1_full_load_factors[kL1FactorsLen] = {l0Status.k_l0, coreStatus.k, 1, l1Status.max_n_bl1, kDbOff, kDbOff};
  l1Status.SetStatus(bl1_full_load_factors);
  L1StatusBl1FullLoad(*params.run_params, coreStatus, l0Status, l1Status, res);
  // neither AL1 nor Bl1 full load
  int64_t neither_full_load_factors[kL1FactorsLen] = {l0Status.k_l0, l0Status.k_l0, 1, 1, kDbOn, kDbOn};
  l1Status.SetStatus(neither_full_load_factors);
  L1StatusNeitherFullLoad(coreStatus, params, l0Status, l1Status, res);
  // choose the final factors
  int64_t *tmpFactors = res[kIdxThree];
  int64_t tmpLoadSize = tmpFactors[kIdxSix];
  int64_t kAl1FactorOne =
      res[kIdxOne][kIdxZero] > 0
          ? MathUtil::CeilDivision(params.run_params->k, coreStatus.k_dim * res[kIdxOne][kIdxZero])
          : 1;
  int64_t kBl1FactorTwo =
      res[kIdxTwo][kIdxThree] > 0
          ? MathUtil::CeilDivision(params.run_params->k, coreStatus.k_dim * res[kIdxTwo][kIdxThree])
          : 1;
  int64_t kAl1FactorZero =
      res[kIdxZero][kIdxZero] > 0
          ? MathUtil::CeilDivision(params.run_params->k, coreStatus.k_dim * res[kIdxZero][kIdxZero])
          : 1;
  int64_t kBl1FactorZero =
      res[kIdxZero][kIdxThree] > 0
          ? MathUtil::CeilDivision(params.run_params->k, coreStatus.k_dim * res[kIdxZero][kIdxThree])
          : 1;
  bool al1FullLoad =
      params.run_params->nd_flag ? (l1Status.al1_full_load && kAl1FactorOne == 1) : l1Status.al1_full_load;
  bool bl1FullLoad =
      params.run_params->nd_flag ? (l1Status.bl1_full_load && kBl1FactorTwo == 1) : l1Status.bl1_full_load;
  bool bothFullLoad = params.run_params->nd_flag
                          ? (l1Status.both_full_load && kAl1FactorZero == 1 && kBl1FactorZero == 1)
                          : l1Status.both_full_load;
  if (al1FullLoad && (res[kIdxOne][kIdxSix] < tmpLoadSize ||
                      (res[kIdxOne][kIdxSix] == tmpLoadSize &&
                       res[kIdxOne][kIdxOne] + res[kIdxOne][kIdxFour] > tmpFactors[kIdxOne] + tmpFactors[kIdxFour]))) {
    tmpFactors = res[kIdxOne];
    tmpLoadSize = tmpFactors[kIdxSix];
  }
  if (bl1FullLoad && (res[kIdxTwo][kIdxSix] < tmpLoadSize ||
                      (res[kIdxTwo][kIdxSix] == tmpLoadSize &&
                       res[kIdxTwo][kIdxOne] + res[kIdxTwo][kIdxFour] > tmpFactors[kIdxOne] + tmpFactors[kIdxFour]))) {
    tmpFactors = res[kIdxTwo];
    tmpLoadSize = tmpFactors[kIdxSix];
  }
  if (bothFullLoad && (res[kIdxZero][kIdxSix] < tmpLoadSize ||
                       (res[kIdxZero][kIdxSix] == tmpLoadSize && res[kIdxZero][kIdxOne] + res[kIdxZero][kIdxFour] >
                                                                     tmpFactors[kIdxOne] + tmpFactors[kIdxFour]))) {
    tmpFactors = res[kIdxZero];
  }
  int64_t res_l1_factors[kL1FactorsLen] = {tmpFactors[kIdxZero], tmpFactors[kIdxThree], tmpFactors[kIdxOne],
                                           tmpFactors[kIdxFour], tmpFactors[kIdxTwo], tmpFactors[kIdxFive]};
  l1Status.SetStatus(res_l1_factors);
  OP_LOGD(op_type.c_str(), "tiling kal1_16:%ld, kbl1_16:%ld", l1Status.kal1_16, l1Status.kbl1_16);
  OP_LOGD(op_type.c_str(), "tiling m_al1:%ld, n_bl1:%ld", l1Status.m_al1, l1Status.n_bl1);
  OP_LOGD(op_type.c_str(), "tiling db_al1:%ld, db_bl1:%ld", l1Status.db_al1, l1Status.db_bl1);
}

void GetUbReusedFlag(const BatchmatmulRunParas& run_params, const L1Status& l1Status, const L0Status& l0Status,
                     UbStatus& ubStatus) {
  // Initialization
  ubStatus.cub_reuse_aub_flag = l1Status.al1_full_load;
  ubStatus.cub_reuse_bub_flag = l1Status.bl1_full_load;
  ubStatus.aub_multi_flag = kNumZero;
  ubStatus.bub_multi_flag = kNumZero;
  // Get AUB Full Load Flag
  if (l1Status.kal1_16 == ubStatus.k_aub && l1Status.m_al1 * l0Status.m_l0 == ubStatus.m_aub) {
    ubStatus.aub_multi_flag = kAttachFlagOne;
  }
  // Get BUB Full Load Flag
  if (l1Status.kbl1_16 == ubStatus.k_bub && l1Status.n_bl1 * l0Status.n_l0 == ubStatus.n_bub) {
    ubStatus.bub_multi_flag = kAttachFlagOne;
  }
  bool aub_full_load = ubStatus.aub_multi_flag == kAttachFlagOne;
  // remove invalid reused scenario(preload is effected)
  if (l1Status.al1_full_load && aub_full_load) {
    ubStatus.cub_reuse_aub_flag = false;
  }
  bool bub_full_load = ubStatus.bub_multi_flag == kAttachFlagOne;
  // remove invalid reused scenario(preload is effected)
  if (l1Status.bl1_full_load && bub_full_load) {
    ubStatus.cub_reuse_bub_flag = false;
  }
  // Calculate reuse condition for aub/bub
  bool flag_aub_pb_fail = l1Status.al1_full_load && aub_full_load;
  bool flag_bub_pb_fail = l1Status.bl1_full_load && bub_full_load;
  ubStatus.flag_pre_ub_not_reused = (flag_aub_pb_fail != flag_bub_pb_fail) ||
                                    (flag_aub_pb_fail && flag_bub_pb_fail && run_params.is_batch_matmul_op) ||
                                    (!l1Status.al1_full_load && !l1Status.bl1_full_load);
  if (run_params.is_batch_matmul_op && !ubStatus.flag_pre_ub_not_reused) {
    ubStatus.cub_reuse_aub_flag = false;
    ubStatus.cub_reuse_bub_flag = false;
  }
  ubStatus.flag_pre_ub_not_reused = ubStatus.flag_pre_ub_not_reused ||
                                    (ubStatus.cub_reuse_aub_flag && !ubStatus.cub_reuse_bub_flag) ||
                                    (!ubStatus.cub_reuse_aub_flag && ubStatus.cub_reuse_bub_flag);
}

void UpdateUbReuseFlagAndRestSize(const BatchmatmulRunParas& run_params, const L1Status& l1Status,
                                  const L0Status& l0Status, UbStatus& ubStatus) {
  // Calculate reuse condition for aub/bub
  GetUbReusedFlag(run_params, l1Status, l0Status, ubStatus);
  // Update UB rest Size ---> available space for AUb and BUb
  if (!ubStatus.cub_reuse_aub_flag && !ubStatus.cub_reuse_bub_flag) {
    ubStatus.ub_rest_size = kUbFp16Size - ubStatus.min_dma_size;
  } else {
    ubStatus.ub_rest_size = kUbFp16Size;
  }
  if (run_params.bias_flag) {
    ubStatus.ub_rest_size -= l0Status.n_l0 * kBlockSize * ubStatus.db_cub * ubStatus.cub_dtype_multi;
  }
}

bool CheckABUbSize(const PreUbTiling& pre_ub_tiling, const BatchmatmulParas& params,
                   SingleCoreStatus& singleCoreStatus, const bool unsafe_ubStatus = false) {
  const BatchmatmulRunParas& run_params = *(params.run_params);
  const BatchmatmulCompileParas& compile_params = *(params.compile_params);
  const int64_t& k_aub = pre_ub_tiling.k_aub;
  const int64_t& m_aub = pre_ub_tiling.m_aub;
  const int64_t& k_bub = pre_ub_tiling.k_bub;
  const int64_t& n_bub = pre_ub_tiling.n_bub;
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  UbStatus tmp_ubStatus = ubStatus;
  tmp_ubStatus.aub_size = 0;
  tmp_ubStatus.bub_size = 0;
  int64_t aubConst = static_cast<int64_t>(kBlockSize * reducedBlockSize * ubStatus.db_aub * (1 + compile_params.aub_double_num));
  int64_t bubConst = static_cast<int64_t>(kBlockSize * reducedBlockSize * ubStatus.db_bub * (1 + compile_params.bub_double_num));
  if (!CheckMulOverflow(k_aub, m_aub, tmp_ubStatus.aub_size) ||
      !CheckMulOverflow(tmp_ubStatus.aub_size, aubConst, tmp_ubStatus.aub_size)) {
    return false;
  }
  if (!CheckMulOverflow(k_bub, n_bub, tmp_ubStatus.bub_size) ||
      !CheckMulOverflow(tmp_ubStatus.bub_size, bubConst, tmp_ubStatus.bub_size)) {
    return false;
  }
  tmp_ubStatus.k_aub = k_aub;
  tmp_ubStatus.m_aub = m_aub;
  tmp_ubStatus.k_bub = k_bub;
  tmp_ubStatus.n_bub = n_bub;
  // This Param is RuntimeParams
  UpdateUbReuseFlagAndRestSize(run_params, l1Status, l0Status, tmp_ubStatus);
  int64_t occupied_ub_size = 0;
  if (!unsafe_ubStatus && !compile_params.split_k_flag) {
    if (tmp_ubStatus.cub_reuse_aub_flag && !tmp_ubStatus.cub_reuse_bub_flag) {
      tmp_ubStatus.aub_size = max(tmp_ubStatus.aub_size, ubStatus.min_dma_size);
    } else if (!tmp_ubStatus.cub_reuse_aub_flag && tmp_ubStatus.cub_reuse_bub_flag) {
      tmp_ubStatus.bub_size = max(tmp_ubStatus.bub_size, ubStatus.min_dma_size);
    } else if (tmp_ubStatus.cub_reuse_aub_flag && tmp_ubStatus.cub_reuse_bub_flag) {
      // AUB BUB and CUB used the same space
      tmp_ubStatus.aub_size = max(tmp_ubStatus.aub_size, max(tmp_ubStatus.bub_size, ubStatus.min_dma_size));
    }
    occupied_ub_size = max(tmp_ubStatus.aub_size, tmp_ubStatus.bub_size);
    if (tmp_ubStatus.flag_pre_ub_not_reused &&
        !CheckAddOverflow(tmp_ubStatus.aub_size, tmp_ubStatus.bub_size, occupied_ub_size)) {
      return false;
    }
    return occupied_ub_size <= tmp_ubStatus.ub_rest_size;
  }
  occupied_ub_size = max(tmp_ubStatus.aub_size, tmp_ubStatus.bub_size);
  if (tmp_ubStatus.flag_pre_ub_not_reused &&
      !CheckAddOverflow(tmp_ubStatus.aub_size, tmp_ubStatus.bub_size, occupied_ub_size)) {
    return false;
  }
  // Dont know if reused can be enabled so do not reused.
  return occupied_ub_size <= tmp_ubStatus.safe_ub_rest_size;
}

void GetMaxUbSizeWhenPartialFullLoad(const BatchmatmulParas& params, SingleCoreStatus& singleCoreStatus,
                                     const int max_num) {
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  // max_num: 0->k_aub, 1->k_bub, 2->m_aub, 3->n_bub
  ubStatus.aub_size = static_cast<int64_t>(ubStatus.k_aub * kBlockSize * ubStatus.m_aub * reducedBlockSize * ubStatus.db_aub *
                      (1 + compile_params.aub_double_num));
  ubStatus.bub_size = static_cast<int64_t>(ubStatus.k_bub * kBlockSize * ubStatus.n_bub * reducedBlockSize * ubStatus.db_bub *
                      (1 + compile_params.bub_double_num));
  bool is_aub = (max_num == kNumZero || max_num == kNumTwo) ? true : false;
  int64_t db_value = is_aub ? ubStatus.db_aub : ubStatus.db_bub;
  int64_t doubleNum = is_aub ? static_cast<int64_t>(compile_params.aub_double_num) : static_cast<int64_t>(compile_params.bub_double_num);
  int64_t ub_occupied_size = is_aub ? ubStatus.bub_size : ubStatus.aub_size;
  int64_t pending_cap = 0;
  int64_t calculated_var = 0;
  int64_t* pending_var = nullptr;
  if (max_num == kNumZero) {
    pending_cap = l1Status.kal1_16;
    pending_var = &ubStatus.k_aub;
    calculated_var = ubStatus.m_aub;
  } else if (max_num == kNumOne) {
    pending_cap = l1Status.kbl1_16;
    pending_var = &ubStatus.k_bub;
    calculated_var = ubStatus.n_bub;
  } else if (max_num == kNumTwo) {
    pending_cap = l1Status.m_al1 * l0Status.m_l0;
    pending_var = &ubStatus.m_aub;
    calculated_var = ubStatus.k_aub;
  } else {
    // Max Number equals three
    pending_cap = l1Status.n_bl1 * l0Status.n_l0;
    pending_var = &ubStatus.n_bub;
    calculated_var = ubStatus.k_bub;
  }
  *pending_var =
      min(ubStatus.safe_ub_rest_size / (kBlockSize * calculated_var * reducedBlockSize * db_value * (1 + doubleNum)),
          pending_cap);
  GetUbReusedFlag(run_params, l1Status, l0Status, ubStatus);
  *pending_var = ubStatus.flag_pre_ub_not_reused
                     ? (ubStatus.safe_ub_rest_size - ub_occupied_size) /
                           (kBlockSize * calculated_var * reducedBlockSize * db_value * (1 + doubleNum))
                     : *pending_var;
  GetNearestFactor(pending_cap, *pending_var);
}

bool FilterBadFactor(int64_t cur_outer_axis, int64_t cur_inner_axis, int64_t max_outer_axis, int64_t max_inner_axis,
                     bool trans) {
  int64_t cur_burst_len = trans ? cur_outer_axis : cur_inner_axis;
  int64_t max_burst_len = trans ? max_outer_axis : max_inner_axis;
  // allow ub full load
  if (cur_outer_axis == max_outer_axis || cur_inner_axis == max_inner_axis) {
    return false;
  }
  // filter case which cann't use full bandwidth
  if (max_burst_len >= k1980FullCacheLine && cur_burst_len % k1980FullCacheLine) {
    return true;
  }
  return false;
}

void UpdateAUbCandidateStatusPhase1(const CoreStatus &coreStatus, const BatchmatmulParas &params,
                                    const AUbStatusCondition &ub_condition, SingleCoreStatus &singleCoreStatus,
                                    int64_t (*aub_results)[2]) {
  // Get All AUB Candidate Tiling result.
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  int64_t al1_m = l1Status.m_al1 * l0Status.m_l0;

  if (ub_condition.condition_m2_k2) {
    aub_results[ubStatus.aub_cnt][0] = coreStatus.k;
    aub_results[ubStatus.aub_cnt][1] = coreStatus.m;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml1_kl1) {
    aub_results[ubStatus.aub_cnt][0] = l1Status.kal1_16;
    aub_results[ubStatus.aub_cnt][1] = al1_m;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml1_kl0) {
    aub_results[ubStatus.aub_cnt][0] = l0Status.k_l0;
    aub_results[ubStatus.aub_cnt][1] = al1_m;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml1_k0) {
    ubStatus.k_aub = 1;
    ubStatus.m_aub = al1_m;
    GetMaxUbSizeWhenPartialFullLoad(params, singleCoreStatus, kNumZero);
    aub_results[ubStatus.aub_cnt][0] = ubStatus.k_aub;
    aub_results[ubStatus.aub_cnt][1] = ubStatus.m_aub;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml0_kl1) {
    aub_results[ubStatus.aub_cnt][0] = l1Status.kal1_16;
    aub_results[ubStatus.aub_cnt][1] = l0Status.m_l0;
    ubStatus.aub_cnt += 1;
  }
}

void UpdateAUbCandidateStatusPhase2(const BatchmatmulParas &params, const AUbStatusCondition &ub_condition,
                                    SingleCoreStatus &singleCoreStatus, int64_t (*aub_results)[2]) {
  // Get All AUB Candidate Tiling result.
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  if (ub_condition.condition_m0_kl1) {
    ubStatus.k_aub = l1Status.kal1_16;
    ubStatus.m_aub = 1;
    GetMaxUbSizeWhenPartialFullLoad(params, singleCoreStatus, kNumTwo);
    aub_results[ubStatus.aub_cnt][0] = ubStatus.k_aub;
    aub_results[ubStatus.aub_cnt][1] = ubStatus.m_aub;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml0_kl0) {
    aub_results[ubStatus.aub_cnt][0] = l0Status.k_l0;
    aub_results[ubStatus.aub_cnt][1] = l0Status.m_l0;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_ml0_k0) {
    aub_results[ubStatus.aub_cnt][0] = 1;
    aub_results[ubStatus.aub_cnt][1] = l0Status.m_l0;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_m0_kl0) {
    aub_results[ubStatus.aub_cnt][0] = l0Status.k_l0;
    aub_results[ubStatus.aub_cnt][1] = 1;
    ubStatus.aub_cnt += 1;
  }
  if (ub_condition.condition_m0_k0) {
    aub_results[ubStatus.aub_cnt][0] = 1;
    aub_results[ubStatus.aub_cnt][1] = 1;
    ubStatus.aub_cnt += 1;
  }
}

void GetPatternAUbFactors(const BatchmatmulParas &params, SingleCoreStatus &singleCoreStatus,
                          PreUbTiling &pre_ub_tiling, int64_t (*aub_results)[2]) {
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  for (int64_t tmp_aub_m = 1; tmp_aub_m <= l1Status.m_l1; tmp_aub_m++) {
    for (int64_t tmp_aub_k = 1; tmp_aub_k <= l1Status.kal1_16; tmp_aub_k++) {
      pre_ub_tiling.update_tiling(tmp_aub_k, tmp_aub_m, ubStatus.k_bub, ubStatus.n_bub);
      if (FilterBadFactor(tmp_aub_m, tmp_aub_k, l1Status.m_l1, l1Status.kal1_16, run_params.trans_a_flag) ||
          !CheckABUbSize(pre_ub_tiling, params, singleCoreStatus)) {
        continue;
      }
      aub_results[ubStatus.aub_cnt][0] = tmp_aub_k;
      aub_results[ubStatus.aub_cnt][1] = tmp_aub_m;
      ubStatus.aub_cnt += 1;
      if (ubStatus.aub_cnt >= kUbFactorNums) {
        return;
      }
    }
  }
}

void GetAUbFactors(const CoreStatus &coreStatus, const BatchmatmulParas &params, SingleCoreStatus &singleCoreStatus,
                   int64_t (*aub_results)[2]) {
  // Get All AUB Candidate Tiling result.
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  AUbStatusCondition ub_condition;
  PreUbTiling pre_ub_tiling;
  ubStatus.aub_cnt = 0;

  if (run_params.pattern_flag) {
    GetPatternAUbFactors(params, singleCoreStatus, pre_ub_tiling, aub_results);
    return;
  }

  int64_t al1_m = l1Status.m_al1 * l0Status.m_l0;
  pre_ub_tiling.update_tiling(coreStatus.k, coreStatus.m, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_m2_k2 =
      (!compile_params.at_l1_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(l1Status.kal1_16, al1_m, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml1_kl1 = CheckABUbSize(pre_ub_tiling, params, singleCoreStatus);
  pre_ub_tiling.update_tiling(l0Status.k_l0, al1_m, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml1_kl0 =
      (run_params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(1, al1_m, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml1_k0 =
      (run_params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus, true));
  pre_ub_tiling.update_tiling(l1Status.kal1_16, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml0_kl1 =
      (!run_params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(l1Status.kal1_16, 1, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_m0_kl1 =
      (!run_params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus, true));
  pre_ub_tiling.update_tiling(l0Status.k_l0, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml0_kl0 = CheckABUbSize(pre_ub_tiling, params, singleCoreStatus);
  pre_ub_tiling.update_tiling(1, l0Status.m_l0, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_ml0_k0 =
      (run_params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(l0Status.k_l0, 1, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_m0_kl0 =
      (!run_params.trans_a_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(1, 1, ubStatus.k_bub, ubStatus.n_bub);
  ub_condition.condition_m0_k0 = CheckABUbSize(pre_ub_tiling, params, singleCoreStatus);

  UpdateAUbCandidateStatusPhase1(coreStatus, params, ub_condition, singleCoreStatus, aub_results);
  UpdateAUbCandidateStatusPhase2(params, ub_condition, singleCoreStatus, aub_results);
}

void UpdateBUbCandidateStatusPhase1(const CoreStatus &coreStatus, const BatchmatmulParas &params,
                                    const BUbStatusCondition &ub_condition, SingleCoreStatus &singleCoreStatus,
                                    int64_t (*bub_results)[2]) {
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  int64_t bl1_n = l1Status.n_bl1 * l0Status.n_l0;
  if (ub_condition.condition_k2_n2) {
    bub_results[ubStatus.bub_cnt][0] = coreStatus.k;
    bub_results[ubStatus.bub_cnt][1] = coreStatus.n;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_kl1_nl1 && ubStatus.bub_cnt < kNumTwo) {
    bub_results[ubStatus.bub_cnt][0] = l1Status.kbl1_16;
    bub_results[ubStatus.bub_cnt][1] = bl1_n;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_kl0_nl1 && ubStatus.bub_cnt < kNumTwo) {
    bub_results[ubStatus.bub_cnt][0] = l0Status.k_l0;
    bub_results[ubStatus.bub_cnt][1] = bl1_n;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_k0_nl1 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.k_bub = 1;
    ubStatus.n_bub = bl1_n;
    GetMaxUbSizeWhenPartialFullLoad(params, singleCoreStatus, kNumOne);
    if (ubStatus.k_bub != 0) {
      bub_results[ubStatus.bub_cnt][0] = ubStatus.k_bub;
      bub_results[ubStatus.bub_cnt][1] = ubStatus.n_bub;
      ubStatus.bub_cnt += 1;
    }
  }
  if (ub_condition.condition_kl1_nl0 && ubStatus.bub_cnt < kNumTwo) {
    bub_results[ubStatus.bub_cnt][0] = l1Status.kbl1_16;
    bub_results[ubStatus.bub_cnt][1] = l0Status.n_l0;
    ubStatus.bub_cnt += 1;
  }
}

void UpdateBUbCandidateStatusPhase2(const BatchmatmulParas &params, const BUbStatusCondition &ub_condition,
                                    SingleCoreStatus &singleCoreStatus, int64_t (*bub_results)[2]) {
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  if (ub_condition.condition_kl1_n0 && ubStatus.bub_cnt < kNumTwo) {
    ubStatus.k_bub = l1Status.kbl1_16;
    ubStatus.n_bub = 1;
    GetMaxUbSizeWhenPartialFullLoad(params, singleCoreStatus, kNumThree);
    if (ubStatus.n_bub != 0) {
      bub_results[ubStatus.bub_cnt][0] = ubStatus.k_bub;
      bub_results[ubStatus.bub_cnt][1] = ubStatus.n_bub;
      ubStatus.bub_cnt += 1;
    }
  }
  if (ub_condition.condition_kl0_nl0 && ubStatus.bub_cnt < kNumTwo) {
    bub_results[ubStatus.bub_cnt][0] = l0Status.k_l0;
    bub_results[ubStatus.bub_cnt][1] = l0Status.n_l0;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_k0_nl0 && ubStatus.bub_cnt < kNumTwo) {
    bub_results[ubStatus.bub_cnt][0] = 1;
    bub_results[ubStatus.bub_cnt][1] = l0Status.n_l0;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_kl0_n0 && ubStatus.bub_cnt < kNumTwo) {
    bub_results[ubStatus.bub_cnt][0] = l0Status.k_l0;
    bub_results[ubStatus.bub_cnt][1] = 1;
    ubStatus.bub_cnt += 1;
  }
  if (ub_condition.condition_k0_n0 && ubStatus.bub_cnt < kNumTwo) {
    bub_results[ubStatus.bub_cnt][0] = 1;
    bub_results[ubStatus.bub_cnt][1] = 1;
    ubStatus.bub_cnt += 1;
  }
}

void GetPatternBUbFactors(const BatchmatmulParas &params, SingleCoreStatus &singleCoreStatus,
                          PreUbTiling &pre_ub_tiling, int64_t (*bub_results)[2]) {
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const L1Status &l1Status = singleCoreStatus.l1Status;
  UbStatus &ubStatus = singleCoreStatus.ubStatus;
  for (int64_t tmp_bub_n = 1; tmp_bub_n <= l1Status.n_l1; tmp_bub_n++) {
    for (int64_t tmp_bub_k = 1; tmp_bub_k <= l1Status.kbl1_16; tmp_bub_k++) {
      pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, tmp_bub_k, tmp_bub_n);
      if (FilterBadFactor(tmp_bub_k, tmp_bub_n, l1Status.kbl1_16, l1Status.n_bl1, run_params.trans_b_flag) ||
          !CheckABUbSize(pre_ub_tiling, params, singleCoreStatus)) {
        continue;
      }
      bub_results[ubStatus.bub_cnt][0] = tmp_bub_k;
      bub_results[ubStatus.bub_cnt][1] = tmp_bub_n;
      ubStatus.bub_cnt += 1;
      if (ubStatus.bub_cnt >= kUbFactorNums) {
        return;
      }
    }
  }
}

void GetBUbFactors(const CoreStatus &coreStatus, const BatchmatmulParas &params, SingleCoreStatus &singleCoreStatus,
                   int64_t (*bub_results)[2]) {
  // Initialize the candidate array. Data will be overwritten so we can keep it dirty
  // Get All AUB Candidate Tiling result.
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  BUbStatusCondition ub_condition;
  PreUbTiling pre_ub_tiling;
  ubStatus.bub_cnt = 0;

  if (run_params.pattern_flag) {
    GetPatternBUbFactors(params, singleCoreStatus, pre_ub_tiling, bub_results);
    return;
  }

  int64_t bl1_n = l1Status.n_bl1 * l0Status.n_l0;
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, coreStatus.k, coreStatus.n);
  ub_condition.condition_k2_n2 =
      (!compile_params.at_l1_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, bl1_n);
  ub_condition.condition_kl1_nl1 = CheckABUbSize(pre_ub_tiling, params, singleCoreStatus);
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, bl1_n);
  ub_condition.condition_kl0_nl1 =
      (!run_params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, 1, bl1_n);
  ub_condition.condition_k0_nl1 =
      (!run_params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus, true));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, l0Status.n_l0);
  ub_condition.condition_kl1_nl0 =
      (run_params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l1Status.kbl1_16, 1);
  ub_condition.condition_kl1_n0 =
      (run_params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus, true));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, l0Status.n_l0);
  ub_condition.condition_kl0_nl0 = CheckABUbSize(pre_ub_tiling, params, singleCoreStatus);
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, 1, l0Status.n_l0);
  ub_condition.condition_k0_nl0 =
      (!run_params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, l0Status.k_l0, 1);
  ub_condition.condition_kl0_n0 =
      (run_params.trans_b_flag && CheckABUbSize(pre_ub_tiling, params, singleCoreStatus));
  pre_ub_tiling.update_tiling(ubStatus.k_aub, ubStatus.m_aub, 1, 1);
  ub_condition.condition_k0_n0 = CheckABUbSize(pre_ub_tiling, params, singleCoreStatus);

  UpdateBUbCandidateStatusPhase1(coreStatus, params, ub_condition, singleCoreStatus, bub_results);
  UpdateBUbCandidateStatusPhase2(params, ub_condition, singleCoreStatus, bub_results);
}

void GetABUbSize(const BatchmatmulCompileParas &params, const BatchmatmulRunParas &run_params, UbStatus &ubStatus) {
  int64_t k_aub = ubStatus.k_aub;
  int64_t m_aub = ubStatus.m_aub;
  int64_t k_bub = ubStatus.k_bub;
  int64_t n_bub = ubStatus.n_bub;
  // Initialized split tensor
  ubStatus.aub_single_tensor_size = k_aub * reducedBlockSize * m_aub * kBlockSize * ubStatus.db_aub;
  ubStatus.bub_single_tensor_size = k_bub * reducedBlockSize * n_bub * kBlockSize * ubStatus.db_bub;
  ubStatus.aub_aligned_tensor_size = ubStatus.aub_single_tensor_size;
  ubStatus.bub_aligned_tensor_size = ubStatus.bub_single_tensor_size;

  ubStatus.aub_bank_size =
      static_cast<int64_t>(k_aub * reducedBlockSize * m_aub * kBlockSize * ubStatus.db_aub * (1 + params.aub_double_num));
  ubStatus.bub_bank_size =
      static_cast<int64_t>(k_bub * reducedBlockSize * n_bub * kBlockSize * ubStatus.db_bub * (1 + params.bub_double_num));
  ubStatus.aub_size = ubStatus.aub_bank_size;
  ubStatus.bub_size = ubStatus.bub_bank_size;
  ubStatus.a_align_value = 1;
  ubStatus.b_align_value = 1;
  ubStatus.a_bank_conflict = false;
  ubStatus.b_bank_conflict = false;
  if (run_params.trans_a_flag && m_aub % kBankConflictFactor == 0) {
    ubStatus.a_bank_conflict = true;
    ubStatus.a_align_value = (m_aub + 1) * kBlockSize;
    ubStatus.aub_bank_size += k_aub * reducedBlockSize * kBlockSize * ubStatus.db_aub;
    ubStatus.aub_align_bound += k_aub * reducedBlockSize * kBlockSize;
  } else if (!run_params.trans_a_flag && k_aub % kBankConflictFactor == 0) {
    ubStatus.a_bank_conflict = true;
    ubStatus.a_align_value = (k_aub + 1) * kBlockSize;
    ubStatus.aub_bank_size += reducedBlockSize * m_aub * kBlockSize * ubStatus.db_aub;
    ubStatus.aub_align_bound += reducedBlockSize * m_aub * kBlockSize;
  }
  if (run_params.trans_b_flag && k_bub % kBankConflictFactor == 0) {
    ubStatus.b_bank_conflict = true;
    ubStatus.b_align_value = (k_bub + 1) * kBlockSize;
    ubStatus.bub_bank_size += reducedBlockSize * n_bub * kBlockSize * ubStatus.db_bub;
    ubStatus.bub_align_bound += reducedBlockSize * n_bub * kBlockSize;
  } else if (!run_params.trans_b_flag && n_bub % kBankConflictFactor == 0) {
    ubStatus.b_bank_conflict = true;
    ubStatus.b_align_value = (n_bub + 1) * kBlockSize;
    ubStatus.bub_bank_size += k_bub * reducedBlockSize * kBlockSize * ubStatus.db_bub;
    ubStatus.bub_align_bound += k_bub * reducedBlockSize * kBlockSize;
  }
}

void GetABUbStorageSize(const BatchmatmulCompileParas& params, const UbStatus& ubStatus, int64_t storage_array[2][2]) {
  /*
  There are two ways of ub tensors reusing:
  1. Not split K scene: Aub/Bub_aligned(tensor1:gm-->ub) reused by Cub_nz_to_nd; Aub_nd_to_nz/Bub_nd_to_nz
  reused by Cub (tensor1:l0c-->ub)
  2. Split K scene: Aub/Bub_aligned(tensor1:gm-->ub) reused by Cub(tensor1:l0c-->ub); Aub_nd_to_nz/Bub_nd_to_nz(tensor2)
  occupies an independent ub space. In this scene, the ub storage size of ub_nd_to_nz is considered as
  ub_cannot_reused_size.
  */

  // After Reused, the storage size is the actual tensor storage size used.
  // storage_array[0] contains aub_storage_size and aub_bank_storage_size;
  // storage_array[1] contains bub_storage_size and bub_bank_storage_size;
  int64_t aub_reused_tensors_size = params.split_k_flag ? ubStatus.aub_single_tensor_size : ubStatus.aub_size;
  int64_t aub_cannot_reused_size = params.split_k_flag ? ubStatus.aub_single_tensor_size : 0;
  storage_array[0][0] = ubStatus.cub_reuse_aub_flag
                            ? max(aub_reused_tensors_size, ubStatus.min_dma_size) + aub_cannot_reused_size
                            : ubStatus.aub_size;

  int64_t bub_reused_tensors_size = params.split_k_flag ? ubStatus.bub_single_tensor_size : ubStatus.bub_size;
  int64_t bub_cannot_reused_size = params.split_k_flag ? ubStatus.bub_single_tensor_size : 0;
  storage_array[1][0] = ubStatus.cub_reuse_bub_flag
                            ? max(bub_reused_tensors_size, ubStatus.min_dma_size) + bub_cannot_reused_size
                            : ubStatus.bub_size;

  int64_t aub_extend_tensor_size = ubStatus.aub_bank_size - ubStatus.aub_single_tensor_size;
  int64_t bub_extend_tensor_size = ubStatus.bub_bank_size - ubStatus.bub_single_tensor_size;
  if (params.split_k_flag) {
    storage_array[0][1] = ubStatus.cub_reuse_aub_flag ? max(aub_extend_tensor_size, ubStatus.min_dma_size) +
                                                            ubStatus.aub_single_tensor_size
                                                      : ubStatus.aub_bank_size;
    storage_array[1][1] = ubStatus.cub_reuse_bub_flag ? max(bub_extend_tensor_size, ubStatus.min_dma_size) +
                                                            ubStatus.bub_single_tensor_size
                                                      : ubStatus.bub_bank_size;
  } else {
    storage_array[0][1] = ubStatus.cub_reuse_aub_flag
                              ? (max(aub_extend_tensor_size, ubStatus.min_dma_size / kNumTwo) +
                                    max(ubStatus.aub_single_tensor_size, ubStatus.min_dma_size / kNumTwo))
                              : ubStatus.aub_bank_size;
    storage_array[1][1] = ubStatus.cub_reuse_bub_flag
                              ? (max(bub_extend_tensor_size, ubStatus.min_dma_size / kNumTwo) +
                                    max(ubStatus.bub_single_tensor_size, ubStatus.min_dma_size / kNumTwo))
                              : ubStatus.bub_bank_size;
  }
}

int GetAllUbReusedAlignMode(const UbStatus& ubStatus, const int& aub_bank_storage_size,
                            const int& bub_bank_storage_size, const BatchmatmulCompileParas& params) {
  int align_mode = kNumZero;
  int64_t max_bank_storage_size = 0;
  int64_t aub_extend_tensor_size = ubStatus.aub_bank_size - ubStatus.aub_single_tensor_size;
  int64_t bub_extend_tensor_size = ubStatus.bub_bank_size - ubStatus.bub_single_tensor_size;
  if (params.split_k_flag) {
    max_bank_storage_size =
        max(max(aub_extend_tensor_size, ubStatus.min_dma_size), bub_extend_tensor_size);
    // When K_dim is split, there is no nd_to_nz tensor.
    max_bank_storage_size = max_bank_storage_size + ubStatus.aub_single_tensor_size + ubStatus.bub_single_tensor_size;
  } else {
    max_bank_storage_size =
        max(max(aub_extend_tensor_size, ubStatus.min_dma_size / kNumTwo), bub_extend_tensor_size);
    max_bank_storage_size =
        max_bank_storage_size +
        max(max(ubStatus.aub_single_tensor_size, ubStatus.min_dma_size / kNumTwo), ubStatus.bub_single_tensor_size);
  }

  if (max_bank_storage_size <= ubStatus.ub_rest_size) {
    align_mode = kNumThree;
  } else if (ubStatus.a_align_value != 1 && ubStatus.b_align_value != 1) {
    if (ubStatus.aub_bank_size > ubStatus.bub_bank_size) {
      // Aub_bank_storage_size > Bub_bank_storage_size and AUB_bank_storage_size exceeds UB buffer size
      align_mode = (bub_bank_storage_size <= ubStatus.ub_rest_size) ? kNumTwo : kNumZero;
    } else {
      // Bub_bank_storage_size > Aub_bank_storage_size and Bub_bank_storage_size exceeds UB buffer size
      align_mode = (aub_bank_storage_size <= ubStatus.ub_rest_size) ? kNumOne : kNumZero;
    }
  }
  return align_mode;
}

int GetAlignMode(const BatchmatmulCompileParas& params, const UbStatus& ubStatus) {
  int align_mode = kNumZero;
  // storage_array[0] contains aub_storage_size and aub_bank_storage_size;
  // storage_array[1] contains bub_storage_size and bub_bank_storage_size;
  int64_t storage_array[2][2] = {0};
  GetABUbStorageSize(params, ubStatus, storage_array);
  // Parse the storage info from storage array.
  int64_t aub_storage_size = storage_array[0][0];
  int64_t bub_storage_size = storage_array[1][0];
  int64_t aub_bank_storage_size = storage_array[0][1];
  int64_t bub_bank_storage_size = storage_array[1][1];
  int64_t aub_aligned_tensor_size = aub_bank_storage_size - aub_storage_size + ubStatus.aub_single_tensor_size;
  int64_t bub_aligned_tensor_size = bub_bank_storage_size - bub_storage_size + ubStatus.bub_single_tensor_size;
  // Process storage_size when all ub tensors are reused together.
  bool a_b_c_ub_reused_together = ubStatus.cub_reuse_aub_flag && ubStatus.cub_reuse_bub_flag;
  // There‘s 4 align_mode here, 0->neither align, 1->AUB align and BUB not, 2->AUB not and BUB align, 3-> all aligned
  if (a_b_c_ub_reused_together) {
    align_mode = GetAllUbReusedAlignMode(ubStatus, aub_bank_storage_size, bub_bank_storage_size, params);
  } else {
    int64_t full_occupied_size = ubStatus.flag_pre_ub_not_reused
                                     ? aub_bank_storage_size + bub_bank_storage_size
                                     : max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size) +
                                           max(aub_aligned_tensor_size, bub_aligned_tensor_size);
    if (full_occupied_size <= ubStatus.ub_rest_size) {
      align_mode = kNumThree;
    } else if (ubStatus.a_align_value != 1 && ubStatus.b_align_value != 1) {
      int64_t only_aub_bank_size = ubStatus.flag_pre_ub_not_reused
                                       ? aub_bank_storage_size + bub_storage_size
                                       : max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size) +
                                             max(aub_aligned_tensor_size, ubStatus.bub_single_tensor_size);
      int64_t only_bub_bank_size = ubStatus.flag_pre_ub_not_reused
                                       ? bub_bank_storage_size + aub_storage_size
                                       : max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size) +
                                             max(ubStatus.aub_single_tensor_size, bub_aligned_tensor_size);
      if (ubStatus.aub_bank_size > ubStatus.bub_bank_size) {
        align_mode = (only_aub_bank_size <= ubStatus.ub_rest_size)
                         ? kNumOne
                         : ((only_bub_bank_size <= ubStatus.ub_rest_size) ? kNumTwo : kNumZero);
      } else {
        align_mode = (only_bub_bank_size <= ubStatus.ub_rest_size)
                         ? kNumTwo
                         : ((only_aub_bank_size <= ubStatus.ub_rest_size) ? kNumOne : kNumZero);
      }
    }
  }
  return align_mode;
}

void CheckBankConflict(const BatchmatmulCompileParas& params,
                       const BatchmatmulRunParas& run_params, UbStatus& ubStatus) {
  GetABUbSize(params, run_params, ubStatus);
  // There‘s 4 align_mode here, 0->neither align, 1->AUB align and BUB not, 2->AUB not and BUB align, 3-> all aligned
  int align_mode = GetAlignMode(params, ubStatus);
  if (align_mode == kNumZero) {
    ubStatus.a_align_value = 1;
    ubStatus.b_align_value = 1;
    ubStatus.aub_align_bound = ubStatus.k_aub * reducedBlockSize * ubStatus.m_aub * kBlockSize;
    ubStatus.bub_align_bound = ubStatus.k_bub * reducedBlockSize * ubStatus.n_bub * kBlockSize;
  } else if (align_mode == kNumOne) {
    ubStatus.a_bank_conflict = false;
    ubStatus.b_align_value = 1;
    ubStatus.aub_aligned_tensor_size = ubStatus.aub_bank_size - ubStatus.aub_single_tensor_size;
    ubStatus.bub_align_bound = ubStatus.k_bub * reducedBlockSize * ubStatus.n_bub * kBlockSize;
    ubStatus.aub_size = ubStatus.aub_bank_size;
  } else if (align_mode == kNumTwo) {
    ubStatus.b_bank_conflict = false;
    ubStatus.a_align_value = 1;
    ubStatus.bub_aligned_tensor_size = ubStatus.bub_bank_size - ubStatus.bub_single_tensor_size;
    ubStatus.aub_align_bound = ubStatus.k_aub * reducedBlockSize * ubStatus.m_aub * kBlockSize;
    ubStatus.bub_size = ubStatus.bub_bank_size;
  } else {
    // align_mode is 3-> all aligned.
    ubStatus.aub_size = ubStatus.aub_bank_size;
    ubStatus.bub_size = ubStatus.bub_bank_size;
    ubStatus.aub_aligned_tensor_size = ubStatus.aub_bank_size - ubStatus.aub_single_tensor_size;
    ubStatus.bub_aligned_tensor_size = ubStatus.bub_bank_size - ubStatus.bub_single_tensor_size;
    ubStatus.a_bank_conflict = false;
    ubStatus.b_bank_conflict = false;
  }
}

int64_t GetAvgUbBatch(UbStatus &ubStatus, bool aub_full_load_flag, bool bub_full_load_flag, bool cub_full_load_flag) {
  int64_t batch_ub_use_size = 0;
  int64_t batch_ub_rest_size = kUbFp16Size;
  if (aub_full_load_flag) {
    batch_ub_use_size += ubStatus.aub_size;
  } else {
    batch_ub_rest_size -= ubStatus.aub_size;
  }
  if (bub_full_load_flag) {
    batch_ub_use_size += ubStatus.bub_size;
  } else {
    batch_ub_rest_size -= ubStatus.bub_size;
  }
  if (cub_full_load_flag) {
    batch_ub_use_size += ubStatus.cub_size;
  } else {
    batch_ub_rest_size -= ubStatus.cub_size;
  }
  return batch_ub_rest_size / batch_ub_use_size;
}

void GetUbBatchHelper(L0Status &l0Status, UbStatus &ubStatus, const int64_t (&batch_aub_arr)[kCandidateLen],
                      const int64_t (&batch_bub_arr)[kCandidateLen], const int64_t (&batch_cub_arr)[kCandidateLen]) {
  // choose the best ub batch tiling from candidates
  int64_t min_outer_loop = l0Status.batch_l0 * l0Status.batch_l0 * l0Status.batch_l0;
  for (auto const &batch_aub: batch_aub_arr) {
    for (auto const &batch_bub: batch_bub_arr) {
      for (auto const &batch_cub: batch_cub_arr) {
        bool condition_buffer_size =
          (batch_aub * ubStatus.aub_size + batch_bub * ubStatus.bub_size + batch_cub * ubStatus.cub_size) <=
          (kUbFp16Size);
        int64_t tmp_outer_loop =
          (l0Status.batch_l0 / batch_aub) * (l0Status.batch_l0 / batch_bub) * (l0Status.batch_l0 / batch_cub);
        bool condition_outer_loop = tmp_outer_loop < min_outer_loop;
        if (condition_buffer_size && condition_outer_loop) {
          ubStatus.batch_aub = batch_aub;
          ubStatus.batch_bub = batch_bub;
          ubStatus.batch_cub = batch_cub;
          min_outer_loop = tmp_outer_loop;
        }
      }
    }
  }
  // update aub_size, bub_size, aub_align_bound, bub_align_bound
  ubStatus.aub_size *= ubStatus.batch_aub;
  ubStatus.aub_align_bound *= ubStatus.batch_aub;
  ubStatus.bub_size *= ubStatus.batch_bub;
  ubStatus.bub_align_bound *= ubStatus.batch_bub;
  // update l0c_multi_batch
  if (ubStatus.batch_aub < l0Status.batch_l0) {
    l0Status.l0c_multi_batch += 4;  // batch_aub_axis_flag * 4
  }
  if (ubStatus.batch_bub < l0Status.batch_l0) {
    l0Status.l0c_multi_batch += 2;  // batch_bub_axis_flag * 2
  }
  if (ubStatus.batch_cub < l0Status.batch_l0) {
    l0Status.l0c_multi_batch += 1;  // batch_cub_axis_flag * 1
  }
}

void GetUbBatchFactor(const BatchmatmulParas &params, const CoreStatus &coreStatus,
                      L0Status &l0Status, UbStatus &ubStatus)
{
  const BatchmatmulRunParas &run_params = *(params.run_params);
  if (coreStatus.batch == 1 || l0Status.l0c_multi_batch == 0 || run_params.do_not_multi_batch) {
    return;
  }
  // when AUB full load, AUB support multi batch
  bool aub_full_load_flag = (run_params.use_pre_ub && ubStatus.m_aub == coreStatus.m &&
                             ubStatus.k_aub == coreStatus.k);
  // when BUB full load, BUB support multi batch
  bool bub_full_load_flag = (run_params.use_pre_ub && run_params.b_have_batch &&
                             ubStatus.n_bub == coreStatus.n && ubStatus.k_bub == coreStatus.k);
  // when CUB full load, CUB support multi batch
  bool cub_full_load_flag = (ubStatus.n_cub == coreStatus.n && l0Status.m_l0 == coreStatus.m);
  if (!aub_full_load_flag && !bub_full_load_flag && !cub_full_load_flag) {
    return;
  }
  // 1. calculate max batch_aub, batch_bub, batch_cub
  int64_t ub_fp16_size = kUbFp16Size;
  int64_t max_batch_aub = aub_full_load_flag ?
    min((ub_fp16_size - ubStatus.bub_size - ubStatus.cub_size) / ubStatus.aub_size, l0Status.batch_l0) : 1;
  int64_t max_batch_bub = bub_full_load_flag ?
    min((ub_fp16_size - ubStatus.aub_size - ubStatus.cub_size) / ubStatus.bub_size, l0Status.batch_l0) : 1;
  int64_t max_batch_cub = cub_full_load_flag ?
    min((ub_fp16_size - ubStatus.aub_size - ubStatus.bub_size) / ubStatus.cub_size, l0Status.batch_l0) : 1;
  // 2. calculate average batch
  int64_t batch_ub_avg = GetAvgUbBatch(ubStatus, aub_full_load_flag, bub_full_load_flag, cub_full_load_flag);
  // 3. calculate candidates of batch_aub, batch_bub, batch_cub
  int64_t batch_aub_arr[kCandidateLen] = {1, 1};
  int64_t batch_bub_arr[kCandidateLen] = {1, 1};
  int64_t batch_cub_arr[kCandidateLen] = {1, 1};
  if (aub_full_load_flag) {
    GetTwoFactors(batch_aub_arr, batch_ub_avg, l0Status.batch_l0, max_batch_aub);
  }
  if (bub_full_load_flag) {
    GetTwoFactors(batch_bub_arr, batch_ub_avg, l0Status.batch_l0, max_batch_bub);
  }
  if (cub_full_load_flag) {
    GetTwoFactors(batch_cub_arr, batch_ub_avg, l0Status.batch_l0, max_batch_cub);
  }
  // 4. choose the best candidate
  GetUbBatchHelper(l0Status, ubStatus, batch_aub_arr, batch_bub_arr, batch_cub_arr);
}

bool SolveBankConflictInConditionNcub(bool &flag_cub_bank_conflict, const L0Status& l0Status,
                                      const BatchmatmulParas& params, UbStatus& ubStatus, int64_t factor_cap) {
  const BatchmatmulRunParas& run_params = *(params.run_params);
  const BatchmatmulCompileParas& compile_params = *(params.compile_params);
  bool flag_bank_conflict_solved = false;
  int64_t extra_size = l0Status.m_l0 * kMinFractalSize * ubStatus.db_cub;
  bool need_nz_to_nd = run_params.format_out_nd && !compile_params.split_k_flag;
  // Determine whether it is needed to solve bank conflict in condition_cub_n
  if (ubStatus.n_cub % (kBlockSize / kNumTwo) == 0 && need_nz_to_nd) {
    flag_cub_bank_conflict = true;
    int64_t nCubTmp = static_cast<int64_t>((ubStatus.max_dma_size - extra_size) / (l0Status.m_l0 * kMinFractalSize *
                                                                (1 + compile_params.fused_double_operand_num) *
                                                                ubStatus.db_cub * ubStatus.cub_dtype_multi));
    GetNearestFactor(l0Status.n_l0, nCubTmp, factor_cap);
    bool not_solve_bank_flag = (ubStatus.cub_reuse_aub_flag && ubStatus.cub_reuse_bub_flag &&
                                l0Status.db_l0c == kDbOn && nCubTmp != ubStatus.n_cub);
    if (nCubTmp >= 1 && !not_solve_bank_flag) {
      flag_bank_conflict_solved = true;
      ubStatus.n_cub = nCubTmp;
    } else {
      flag_bank_conflict_solved = false;
    }
  }
  return flag_bank_conflict_solved;
}

void GetReusedBufferSizeForCub(const BatchmatmulParas &params, const UbStatus &ubStatus, int64_t reused_tensor_array[2])
{
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  int64_t cub_reused_size = 0;
  int64_t cub_nz_to_nd_reused_size = 0;
  // In perticular circumstances, cub and pre_ub have cross reuse.
  bool cross_reuse_flag = !(compile_params.split_k_flag || run_params.b_have_batch);
  if (ubStatus.cub_reuse_aub_flag && !ubStatus.cub_reuse_bub_flag) {
    cub_reused_size = cross_reuse_flag ? ubStatus.aub_single_tensor_size : ubStatus.aub_aligned_tensor_size;
    cub_nz_to_nd_reused_size = cross_reuse_flag ? ubStatus.aub_aligned_tensor_size : ubStatus.aub_single_tensor_size;
  } else if (!ubStatus.cub_reuse_aub_flag && ubStatus.cub_reuse_bub_flag) {
    cub_reused_size = cross_reuse_flag ? ubStatus.bub_single_tensor_size : ubStatus.bub_aligned_tensor_size;
    cub_nz_to_nd_reused_size = cross_reuse_flag ? ubStatus.bub_aligned_tensor_size : ubStatus.bub_single_tensor_size;
  } else if (ubStatus.cub_reuse_aub_flag && ubStatus.cub_reuse_bub_flag) {
    cub_reused_size = cross_reuse_flag ? max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size)
                                        : max(ubStatus.aub_aligned_tensor_size, ubStatus.bub_aligned_tensor_size);
    cub_nz_to_nd_reused_size = cross_reuse_flag
                                    ? max(ubStatus.aub_aligned_tensor_size, ubStatus.bub_aligned_tensor_size)
                                    : max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size);
  }
  cub_nz_to_nd_reused_size = compile_params.split_k_flag ? 0 : cub_nz_to_nd_reused_size;
  reused_tensor_array[0] = cub_reused_size;
  reused_tensor_array[1] = cub_nz_to_nd_reused_size;
}

void GetCubNForNoNl0FullLoad(const L0Status &l0Status, const BatchmatmulParas &params, UbStatus &ubStatus,
                             bool &flag_cub_bank_conflict, bool &flag_bank_conflict_solved) {
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  ubStatus.max_dma_size = kUbFp16Size - ubStatus.min_load_size;
  if (run_params.bias_flag) {
    ubStatus.max_dma_size -= l0Status.n_l0 * kBlockSize * ubStatus.db_cub * ubStatus.cub_dtype_multi;
  }
  bool is_invalid = true;
  int64_t factor_cap = INT64_MAX;
  int64_t cub_size = 0;
  int64_t cub_nz_to_nd_size = 0;
  // reused_tensor_array[0] is cub_reused_size and reused_tensor_array[1] is cub_nz_to_nd_reused_size
  int64_t reused_tensor_array[2] = {0};
  GetReusedBufferSizeForCub(params, ubStatus, reused_tensor_array);
  int64_t cub_reused_size = reused_tensor_array[0];
  int64_t cub_nz_to_nd_reused_size = reused_tensor_array[1];
  while (is_invalid && factor_cap > 0) {
    ubStatus.n_cub = static_cast<int64_t>(ubStatus.max_dma_size /
                      (l0Status.m_l0 * kMinFractalSize * (1 + compile_params.fused_double_operand_num) *
                      ubStatus.db_cub * ubStatus.cub_dtype_multi));
    ubStatus.db_cub = ubStatus.n_cub > 0 ? kDbOn : kDbOff;
    GetNearestFactor(l0Status.n_l0, ubStatus.n_cub, factor_cap);
    flag_bank_conflict_solved =
        SolveBankConflictInConditionNcub(flag_cub_bank_conflict, l0Status, params, ubStatus, factor_cap);
    cub_size = ubStatus.n_cub * l0Status.m_l0 * kMinFractalSize * ubStatus.cub_dtype_multi * ubStatus.db_cub;
    cub_nz_to_nd_size =
        flag_bank_conflict_solved ? cub_size + l0Status.m_l0 * kMinFractalSize * ubStatus.db_cub : cub_size;
    cub_nz_to_nd_size = compile_params.split_k_flag ? 0 : cub_nz_to_nd_size;
    is_invalid =
        ubStatus.min_load_size + (max(cub_size, cub_reused_size) + max(cub_nz_to_nd_size, cub_nz_to_nd_reused_size)) >
        kUbFp16Size;
    is_invalid = is_invalid || (ubStatus.n_cub_tail_block_limit && ubStatus.n_cub == 1);
    ubStatus.max_dma_size = static_cast<int64_t>(ubStatus.n_cub * l0Status.m_l0 * kMinFractalSize * ubStatus.cub_dtype_multi *
                                (1 + compile_params.fused_double_operand_num) * ubStatus.db_cub - 1);
    factor_cap = ubStatus.n_cub - 1;
  }
  // In UB Fusion, n_cub could be 0 in case of large fused_double_operand_num
  ubStatus.n_cub = max(ubStatus.n_cub, 1L);
}

void GetCUbFactors(const L0Status& l0Status, const BatchmatmulParas& params, UbStatus& ubStatus) {
  const BatchmatmulRunParas& run_params = *(params.run_params);
  const BatchmatmulCompileParas& compile_params = *(params.compile_params);
  // Initialize n_cub status.
  ubStatus.n_cub = l0Status.n_l0;
  ubStatus.flag_cub_solving_bank_conflict = false;
  bool flag_cub_bank_conflict = false;
  bool flag_bank_conflict_solved = false;
  bool condition_cub_n = ubStatus.max_dma_size + ubStatus.min_load_size > kUbFp16Size;
  // Try to solve cub bank conflict ---> Current algorithm does not take parallelism into consideration.
  bool need_nz_to_nd =
      run_params.format_out_nd && !compile_params.split_k_flag && !PlatformInfo::GetInstance().support_l0c2out();
  int64_t extra_size = 0;
  if (ubStatus.n_cub % (kBlockSize / kNumTwo) == 0 && need_nz_to_nd) {
    flag_cub_bank_conflict = true;
    // Solving CUB Bank Conflict needs to check cub_size.
    extra_size = l0Status.m_l0 * kBlockSize * kBlockSize * ubStatus.db_cub;
    condition_cub_n = ubStatus.max_dma_size + extra_size + ubStatus.min_load_size > kUbFp16Size;
  }
  if (condition_cub_n && (!PlatformInfo::GetInstance().support_l0c2out() || std::fabs(compile_params.fused_double_operand_num - 0.0f) > std::numeric_limits<float>::epsilon())) {
    GetCubNForNoNl0FullLoad(l0Status, params, ubStatus, flag_cub_bank_conflict, flag_bank_conflict_solved);
  } else {
    // Having enough space even if extra space is needed to solve bank conflict.
    flag_bank_conflict_solved = true;
  }
  ubStatus.cub_size = static_cast<int64_t>(ubStatus.n_cub * l0Status.m_l0 * kBlockSize * kBlockSize *
                      (1 + compile_params.fused_double_operand_num) * ubStatus.db_cub * ubStatus.cub_dtype_multi);
  if (flag_cub_bank_conflict && flag_bank_conflict_solved) {
    ubStatus.flag_cub_solving_bank_conflict = true;
    ubStatus.cub_size = static_cast<int64_t>(ubStatus.n_cub * l0Status.m_l0 * kBlockSize * kBlockSize *
                            (1 + compile_params.fused_double_operand_num) * ubStatus.db_cub * ubStatus.cub_dtype_multi +
                        extra_size);
  }
  if (run_params.bias_flag) {
    ubStatus.cub_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub * ubStatus.cub_dtype_multi;
  }
}

void CalculateLoadSizeForSingleReusedCondition(const BatchmatmulCompileParas& params, UbStatus& ubStatus,
                                               bool is_aub) {
  int64_t reused_tensor_size_in_split_k = 0;
  int64_t cannot_occupied_size = is_aub ? ubStatus.bub_size : ubStatus.aub_size;
  int64_t single_tensor_size = is_aub ? ubStatus.aub_single_tensor_size : ubStatus.bub_single_tensor_size;
  int64_t aligned_tensor_size = is_aub ? ubStatus.aub_aligned_tensor_size : ubStatus.bub_aligned_tensor_size;

  ubStatus.min_load_size = ubStatus.flag_pre_ub_not_reused
                               ? cannot_occupied_size + static_cast<int64_t>(params.split_k_flag) * single_tensor_size
                               : static_cast<int64_t>(params.split_k_flag) *
                                     max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size);
  reused_tensor_size_in_split_k = ubStatus.flag_pre_ub_not_reused
                                      ? aligned_tensor_size
                                      : max(ubStatus.aub_aligned_tensor_size, ubStatus.bub_aligned_tensor_size);
  ubStatus.max_dma_size =
      params.split_k_flag ? max(ubStatus.max_dma_size, reused_tensor_size_in_split_k)
                          : max(static_cast<int64_t>(ubStatus.max_dma_size / (1 + params.fused_double_operand_num)),
                                single_tensor_size) +
                                max(static_cast<int64_t>(ubStatus.max_dma_size / (1 + params.fused_double_operand_num)),
                                    aligned_tensor_size);
}

void UpdateUbLoadSize(const BatchmatmulCompileParas& params, UbStatus& ubStatus) {
  if (!ubStatus.cub_reuse_aub_flag && !ubStatus.cub_reuse_bub_flag) {
    ubStatus.min_load_size = ubStatus.flag_pre_ub_not_reused
                                 ? ubStatus.aub_size + ubStatus.bub_size
                                 : max(ubStatus.aub_aligned_tensor_size, ubStatus.bub_aligned_tensor_size) +
                                       max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size);
  } else if (ubStatus.cub_reuse_aub_flag && !ubStatus.cub_reuse_bub_flag) {
    CalculateLoadSizeForSingleReusedCondition(params, ubStatus, true);
  } else if (!ubStatus.cub_reuse_aub_flag && ubStatus.cub_reuse_bub_flag) {
    CalculateLoadSizeForSingleReusedCondition(params, ubStatus, false);
  } else {
    int64_t reused_single_tensor_size = max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size);
    int64_t reused_aligned_tensor_size = max(ubStatus.aub_aligned_tensor_size, ubStatus.bub_aligned_tensor_size);
    ubStatus.min_load_size = static_cast<int64_t>(params.split_k_flag) *
                             max(ubStatus.aub_single_tensor_size, ubStatus.bub_single_tensor_size);
    ubStatus.max_dma_size =
        params.split_k_flag
            ? max(ubStatus.max_dma_size, reused_aligned_tensor_size)
            : max(static_cast<int64_t>(ubStatus.max_dma_size / (1 + params.fused_double_operand_num)),
                  reused_single_tensor_size) +
                  max(static_cast<int64_t>(ubStatus.max_dma_size / (1 + params.fused_double_operand_num)),
                      reused_aligned_tensor_size);
  }
}

void UpdateUbStatus(const UbStatus &src_ub, UbStatus &dst_ub) {
  // Update dst UbStatus from source ubStatus
  dst_ub.k_aub = src_ub.k_aub;
  dst_ub.m_aub = src_ub.m_aub;
  dst_ub.k_bub = src_ub.k_bub;
  dst_ub.n_bub = src_ub.n_bub;
  dst_ub.aub_multi_flag = src_ub.aub_multi_flag;
  dst_ub.bub_multi_flag = src_ub.bub_multi_flag;
  dst_ub.n_cub = src_ub.n_cub;

  dst_ub.flag_cub_solving_bank_conflict = src_ub.flag_cub_solving_bank_conflict;
  dst_ub.a_align_value = src_ub.a_align_value;
  dst_ub.b_align_value = src_ub.b_align_value;
  dst_ub.aub_align_bound = src_ub.aub_align_bound;
  dst_ub.bub_align_bound = src_ub.bub_align_bound;

  dst_ub.aub_size = src_ub.aub_size;
  dst_ub.bub_size = src_ub.bub_size;
  dst_ub.cub_size = src_ub.cub_size;
}

int64_t GetUbInstructionCost(const BatchmatmulRunParas& run_params, const CoreStatus& coreStatus,
                             const SingleCoreStatus& singleCoreStatus) {
  // Get The Cost of UB MTE2 process which is constructed as copy_gm_to_ub and nd2nz
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  const UbStatus& ubStatus = singleCoreStatus.ubStatus;
  // Calculate MTE2 cost for AUB
  int64_t multi_k_aub_l1 = MathUtil::CeilDivision(l1Status.kal1_16, ubStatus.k_aub);
  int64_t multi_m_ub_l1 = MathUtil::CeilDivision(l1Status.m_al1 * l0Status.m_l0, ubStatus.m_aub);
  // Do compensation if Pre Ub still have bank conflict.
  multi_m_ub_l1 = ubStatus.a_bank_conflict ? multi_m_ub_l1 * kNumFour : multi_m_ub_l1;
  double aub_brand_width_utilization = run_params.trans_a_flag
                                           ? max((K_WORST_BANDWIDTH_UTIL_MULTI / static_cast<double>(ubStatus.m_aub)), 1.0)
                                           : max((K_WORST_BANDWIDTH_UTIL_MULTI / static_cast<double>(ubStatus.k_aub)), 1.0);
  int64_t aub_mte2_cost = static_cast<int64_t>(multi_k_aub_l1 * multi_m_ub_l1 * aub_brand_width_utilization);
  // Calculate MTE2 cost for BUB
  int64_t multi_k_bub_l1 = MathUtil::CeilDivision(l1Status.kbl1_16, ubStatus.k_bub);
  int64_t multi_n_ub_l1 = MathUtil::CeilDivision(l1Status.n_bl1 * l0Status.n_l0, ubStatus.n_bub);
  // Do compensation if Pre Ub still have bank conflict.
  multi_n_ub_l1 = ubStatus.b_bank_conflict ? multi_n_ub_l1 * kNumFour : multi_n_ub_l1;
  double bub_brand_width_utilization = run_params.trans_b_flag
                                           ? max((K_WORST_BANDWIDTH_UTIL_MULTI / static_cast<double>(ubStatus.k_bub)), 1.0)
                                           : max((K_WORST_BANDWIDTH_UTIL_MULTI / static_cast<double>(ubStatus.n_bub)), 1.0);
  int64_t bub_mte2_cost = static_cast<int64_t>(multi_k_bub_l1 * multi_n_ub_l1 * bub_brand_width_utilization);
  // Calculate MTE3 cost which represents the proccess of UB-->GM
  int64_t multi_n_ub_l0 = l0Status.n_l0 / ubStatus.n_cub;
  return aub_mte2_cost * coreStatus.kal1_factor + bub_mte2_cost * coreStatus.kbl1_factor + multi_n_ub_l0;
}

bool FilterSmallUbRes(const UbStatus &ubStatus, const L1Status &l1Status,
                      const BatchmatmulCompileParas &compile_params) {
  // Allow aub and bub both full load
  if (ubStatus.m_aub == l1Status.m_l1 && ubStatus.k_aub == l1Status.kal1_16 && ubStatus.k_bub == l1Status.kbl1_16 &&
      ubStatus.n_bub == l1Status.n_l1) {
    return false;
  }
  int64_t aubSizeBytes = static_cast<int64_t>(ubStatus.m_aub * ubStatus.k_aub * kBlockSize * reducedBlockSize * ubStatus.db_aub *
                           (1 + compile_params.aub_double_num) * inputDtypeBytes);
  int64_t bubSizeBytes = static_cast<int64_t>(ubStatus.k_bub * ubStatus.n_bub * kBlockSize * reducedBlockSize * ubStatus.db_bub *
                           (1 + compile_params.bub_double_num) * inputDtypeBytes);
  // Filter to solutions with a UB utilization of less than 25%
  if (aubSizeBytes + bubSizeBytes < (PlatformInfo::GetInstance().ub_size >> 2)) {
    return true;
  }
  return false;
}

void GetUbFactorsInND(const string& op_type, CoreStatus& coreStatus, const BatchmatmulParas& params,
                      SingleCoreStatus& singleCoreStatus) {
  const L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  // multi_m_ub_l1 * multi_k_aub_l1 * kal1_factor + multi_n_ub_l1 * multi_k_bub_l1 * kbl1_factor + multi_n_ub_l1
  // with bank conflict kNumTwo and worst brand utilization: AUB cycle + BUB cycle + Cub Cycle
  int64_t minUbCost =static_cast<int64_t>(
      l1Status.kal1_16 * l1Status.m_al1 * l0Status.m_l0 * kNumFour * K_WORST_BANDWIDTH_UTIL_MULTI * coreStatus.kal1_factor +
      l1Status.kbl1_16 * l1Status.n_bl1 * l0Status.n_l0 * kNumFour * K_WORST_BANDWIDTH_UTIL_MULTI * coreStatus.kbl1_factor +
      l0Status.n_l0);
  // Two means both aub and bub have bank conflict.
  int8_t min_bank_conflict_level = kNumTwo;
  UbStatus final_ub_status;
  int64_t aub_results[kUbFactorNums][2] = {0};
  int64_t bub_results[kUbFactorNums][2] = {0};
  GetAUbFactors(coreStatus, params, singleCoreStatus, aub_results);
  for (int32_t aub_idx = 0; aub_idx < ubStatus.aub_cnt; aub_idx++) {
    ubStatus.k_aub = aub_results[aub_idx][0];
    ubStatus.m_aub = aub_results[aub_idx][1];
    GetBUbFactors(coreStatus, params, singleCoreStatus, bub_results);
    for (int32_t bub_idx = 0; bub_idx < ubStatus.bub_cnt; bub_idx++) {
      ubStatus.k_bub = bub_results[bub_idx][0];
      ubStatus.n_bub = bub_results[bub_idx][1];
      if (run_params.pattern_flag && FilterSmallUbRes(ubStatus, l1Status, compile_params)) {
        continue;
      }
      // Initialize min/max dma copy size before updating it.
      ubStatus.min_dma_size = static_cast<int64_t>(l0Status.m_l0 * kBlockSize * kBlockSize * (1 + compile_params.fused_double_operand_num) *
                              ubStatus.db_cub * ubStatus.cub_dtype_multi);
      ubStatus.max_dma_size = l0Status.n_l0 * ubStatus.min_dma_size;
      ubStatus.aub_align_bound = ubStatus.k_aub * reducedBlockSize * ubStatus.m_aub * kBlockSize;
      ubStatus.bub_align_bound = ubStatus.k_bub * reducedBlockSize * ubStatus.n_bub * kBlockSize;
      UpdateUbReuseFlagAndRestSize(run_params, l1Status, l0Status, ubStatus);
      CheckBankConflict(compile_params, run_params, ubStatus);
      UpdateUbLoadSize(compile_params, ubStatus);
      if (run_params.bias_flag) {
        ubStatus.min_dma_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub * ubStatus.cub_dtype_multi;
        ubStatus.max_dma_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub * ubStatus.cub_dtype_multi;
      }
      GetCUbFactors(l0Status, params, ubStatus);
      // Calculate CUB cost
      OP_TILING_CHECK(ubStatus.n_cub == 0,
                      OP_LOGW(op_type.c_str(),
                              "The current Tiling Candidate in MatMul/BatchMatMaul optiling exist "
                              "one invalid zero n_cub tiling result."),
                      return);
      int64_t tmp_ub_cost = GetUbInstructionCost(run_params, coreStatus, singleCoreStatus);
      int8_t tmp_bank_conflict_level =
          static_cast<int8_t>(ubStatus.a_bank_conflict) + static_cast<int8_t>(ubStatus.b_bank_conflict);
      int64_t cycle = run_params.pattern_flag ? GetCycleByModel(run_params, coreStatus, singleCoreStatus) : INT64_MAX;
      bool oldCondition = tmp_ub_cost < minUbCost ||
                           (tmp_ub_cost == minUbCost && tmp_bank_conflict_level < min_bank_conflict_level);
      if (cycle < coreStatus.cycle || (cycle == coreStatus.cycle && oldCondition)) {
        coreStatus.cycle = cycle;
        minUbCost = tmp_ub_cost;
        min_bank_conflict_level = tmp_bank_conflict_level;
        UpdateUbStatus(ubStatus, final_ub_status);
      }
    }
  }
  UpdateUbStatus(final_ub_status, ubStatus); // Update final result to ubStatus.
}

void SetUbReuseFlag(const L1Status& l1Status, UbStatus& ubStatus) {
  // Set UB Reused Flag
  if (l1Status.both_full_load) {
    ubStatus.cub_reuse_aub_flag = true;
    ubStatus.cub_reuse_bub_flag = true;
  } else if (l1Status.al1_full_load && !l1Status.bl1_full_load) {
    ubStatus.cub_reuse_aub_flag = true;
  } else if (!l1Status.al1_full_load && l1Status.bl1_full_load) {
    ubStatus.cub_reuse_bub_flag = true;
  }
}

static void UpdateL1FullLoadFlag(const string &op_type, const BatchmatmulRunParas &params, CoreStatus &coreStatus,
                                 SingleCoreStatus &singleCoreStatus) {
  L0Status& l0Status = singleCoreStatus.l0Status;
  L1Status& l1Status = singleCoreStatus.l1Status;
  if (l1Status.kal1_16 >= l1Status.kbl1_16) {
    coreStatus.kal1_factor = MathUtil::CeilDivision(params.k, coreStatus.k_dim * l1Status.kal1_16);
    coreStatus.kbl1_factor = coreStatus.kal1_factor * l1Status.kal1_16 / l1Status.kbl1_16;
  }
  else {
    coreStatus.kbl1_factor = MathUtil::CeilDivision(params.k, coreStatus.k_dim * l1Status.kbl1_16);
    coreStatus.kal1_factor = coreStatus.kbl1_factor * l1Status.kbl1_16 / l1Status.kal1_16;
  }
  // when kal1/kbl1 is not full_loaded, m_al1/n_bl1 should be set as 1
  if (!params.is_weight_quant_bmm) {
    l1Status.n_bl1 = coreStatus.kbl1_factor > 1 ? 1 : l1Status.n_bl1;
    l1Status.m_al1 = coreStatus.kal1_factor > 1 ? 1 : l1Status.m_al1;
  }
  int64_t n_single_core = coreStatus.n / (l1Status.n_bl1 * l0Status.n_l0);
  int64_t m_single_core = coreStatus.m / (l1Status.m_al1 * l0Status.m_l0);
  // initialize the full load flag
  l1Status.both_full_load = false;
  l1Status.al1_full_load = false;
  l1Status.bl1_full_load = false;
  if (m_single_core == 1 && coreStatus.kal1_factor == 1) {
    l1Status.al1_full_load = true;
    OP_LOGD(op_type.c_str(), "check special template, tiling al1 changed to full load");
  }
  if (n_single_core == 1 && coreStatus.kbl1_factor == 1) {
    l1Status.bl1_full_load = true;
    OP_LOGD(op_type.c_str(), "check special template, tiling bl1 changed to full load");
  }
  // disable l0c_preload when m_al1 * n_bl1 less than 3
  if (!PlatformInfo::GetInstance().support_l0c2out() &&
    l1Status.al1_full_load != l1Status.bl1_full_load && l1Status.m_al1 * l1Status.n_bl1 <= kNumTwo) {
    l0Status.db_l0c = kDbOff;
  }
  // Update the full_load flag in l1Status to ensure they are correct.
  if (l1Status.al1_full_load && l1Status.bl1_full_load) {
    l1Status.both_full_load = true;
  }
}

void UpdateSingleCoreStatus(const BatchmatmulParas &params, SingleCoreStatus &singleCoreStatus)
{
  const BatchmatmulCompileParas compile_params = *(params.compile_params);
  BatchmatmulRunParas run_params = *(params.run_params);
  singleCoreStatus.l0Status.SetInitLoadStatus();
  singleCoreStatus.l0Status.dtype_bias = run_params.dtype_bias;
  if (run_params.dtype_out == static_cast<int32_t>(ge::DT_FLOAT)) {
    // data in cub is fp32 so the size used is double
    singleCoreStatus.ubStatus.cub_dtype_multi = kNumTwo;
  }
  singleCoreStatus.ubStatus.fused_double_operand_num = compile_params.fused_double_operand_num;
  if (!PlatformInfo::GetInstance().support_l0c2out() && run_params.format_out_nd &&
      run_params.ori_shape_n % kBlockSize != 0 && run_params.n > 1 && !run_params.is_quant_batch_matmul_v3) {
      singleCoreStatus.ubStatus.n_cub_tail_block_limit = true;
  }
  if (run_params.bias_flag && PlatformInfo::GetInstance().support_l0c2out()) {
    singleCoreStatus.l1Status.channel_wise_times =
        MathUtil::CeilDivision(run_params.dtype_bias, kFp16Bytes);
  }
  if (run_params.vector_pre_conv_mode) {
    // q_bias(int32)，deq_scale(uint64_t) use L1 space.
    singleCoreStatus.l1Status.channel_wise_times =
        MathUtil::CeilDivision(run_params.dtype_bias, kFp16Bytes) +
        MathUtil::CeilDivision(ge::GetSizeByDataType(ge::DT_UINT64), kFp16Bytes);
  }
  if (run_params.is_weight_quant_bmm) {
    // q_bias(int32)，deq_scale(uint64_t), bias(fp16, fp32) use L1 space.
    singleCoreStatus.l1Status.channel_wise_times =
        MathUtil::CeilDivision(kFp32Bytes, kFp16Bytes) +
        MathUtil::CeilDivision(run_params.dtype_bias, kFp16Bytes) +
        MathUtil::CeilDivision(ge::GetSizeByDataType(ge::DT_UINT64), kFp16Bytes);
  }
}

void GetUbFactors(const string &op_type, const BatchmatmulParas &params, CoreStatus &coreStatus,
                  SingleCoreStatus &singleCoreStatus) {
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  L0Status& l0Status = singleCoreStatus.l0Status;
  const L1Status& l1Status = singleCoreStatus.l1Status;
  UbStatus& ubStatus = singleCoreStatus.ubStatus;
  // Set reused condition based on L1 attach situation
  SetUbReuseFlag(l1Status, ubStatus);
  ubStatus.n_cub = l0Status.n_l0;
  ubStatus.min_dma_size = static_cast<int64_t>(l0Status.m_l0 * kBlockSize * kBlockSize * (1 + compile_params.fused_double_operand_num) *
                          ubStatus.db_cub * ubStatus.cub_dtype_multi);
  ubStatus.max_dma_size = l0Status.n_l0 * ubStatus.min_dma_size;

  if (run_params.bias_flag) {
    ubStatus.min_dma_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub * ubStatus.cub_dtype_multi;
    ubStatus.max_dma_size += l0Status.n_l0 * kBlockSize * ubStatus.db_cub * ubStatus.cub_dtype_multi;
  }
  if (run_params.use_pre_ub) {
    ubStatus.safe_ub_rest_size = (kUbFp16Size) - ubStatus.min_dma_size;
    GetUbFactorsInND(op_type, coreStatus, params, singleCoreStatus);
  } else {
    // Get CUB factors for NZ in Mode
    GetCUbFactors(l0Status, params, ubStatus);
  }
  OP_LOGD(op_type.c_str(), "tiling n_cub:%ld, n_bub:%ld, k_bub:%ld, k_aub:%ld, m_aub:%ld",
          ubStatus.n_cub, ubStatus.n_bub, ubStatus.k_bub, ubStatus.k_aub, ubStatus.m_aub);
  GetUbBatchFactor(params, coreStatus, l0Status, ubStatus);
  OP_LOGD(op_type.c_str(), "tiling batch_cub:%ld, batch_aub:%ld, batch_bub:%ld",
          ubStatus.batch_cub, ubStatus.batch_aub, ubStatus.batch_bub);
}

bool CheckFactor(const BatchmatmulRunParas &run_params, const CoreStatus &coreStatus,
                      const SingleCoreStatus &singleCoreStatus) {
  if (coreStatus.n_dim == 1 && (run_params.n != singleCoreStatus.l1Status.n_bl1 * singleCoreStatus.l0Status.n_l0 *
      coreStatus.n_single_core * coreStatus.n_dim)) {
    return false;
  }
  if (coreStatus.m_dim == 1 && (run_params.m != singleCoreStatus.l1Status.m_al1 * singleCoreStatus.l0Status.m_l0 *
      coreStatus.m_single_core * coreStatus.m_dim)) {
    return false;
  }
  return true;
}

void SetDbFullLoad(L1Status &l1Status, int64_t cur_al1_size, int64_t cur_bl1_size,
                   int64_t cur_bias_l1_size, const BatchmatmulRunParas &run_params) {
  if (l1Status.both_full_load) {
      if (run_params.is_weight_quant_bmm && l1Status.al1_full_load &&
          CheckL1Size(cur_al1_size, cur_bl1_size * kDbOn, cur_bias_l1_size * kDbOn)) {
        l1Status.db_bl1 = kDbOn;
      }
    return;
  }
  if (l1Status.al1_full_load && CheckL1Size(cur_al1_size, cur_bl1_size * kDbOn, cur_bias_l1_size * kDbOn)) {
    l1Status.db_bl1 = kDbOn;
  }
  if (l1Status.bl1_full_load && CheckL1Size(cur_al1_size * kDbOn, cur_bl1_size, cur_bias_l1_size)) {
    l1Status.db_al1 = kDbOn;
  }
}

void SetDbNotFullLoad(L1Status &l1Status, int64_t cur_al1_size, int64_t cur_bl1_size,
                      int64_t cur_bias_l1_size) {
  if (CheckL1Size(cur_al1_size * kDbOn, cur_bl1_size * kDbOn, cur_bias_l1_size * kDbOn)) {
    l1Status.db_al1 = kDbOn;
    l1Status.db_bl1 = kDbOn;
  } else if (CheckL1Size(cur_al1_size * kDbOn, cur_bl1_size, cur_bias_l1_size)) {
    l1Status.db_al1 = kDbOn;
  } else if (CheckL1Size(cur_al1_size, cur_bl1_size * kDbOn, cur_bias_l1_size * kDbOn)) {
    l1Status.db_bl1 = kDbOn;
  } else {
    return;
  }
}

void AddCondition(const string &op_type, BatchmatmulParas &params, CoreStatus &coreStatus,
                  SingleCoreStatus &singleCoreStatus, GemmEstimate *est) {
  coreStatus.cycle = -1;
  BatchmatmulRunParas &run_params = *(params.run_params);
  run_params.pattern_flag = true;
  UpdateL1LoadFlag(coreStatus, singleCoreStatus);
  SetDoubleBuffer(run_params, singleCoreStatus);
  if (!PlatformInfo::GetInstance().support_l0c2out() || std::fabs(params.compile_params->fused_double_operand_num - 0.0f) > std::numeric_limits<float>::epsilon()) {
    coreStatus.cycle = INT64_MAX; // cycle ==-1, calculate in model;
    GetUbFactors(op_type, params, coreStatus, singleCoreStatus);
  }
  est->AddEstimateTask(coreStatus, singleCoreStatus);
}

bool CheckTilingBuffer(const BatchmatmulRunParas &run_params, const SingleCoreStatus &singleCoreStatus) {
  const L0Status &l0Status = singleCoreStatus.l0Status;
  const L1Status &l1Status = singleCoreStatus.l1Status;
  int64_t channel_wise_l1_size = GetChannelWiseL1Size(l1Status, l1Status.n_l1);
  if (PlatformInfo::GetInstance().support_l0c2out() && run_params.bias_flag) {
    if (GetBiasBtSize(l0Status.n_l0) > PlatformInfo::GetInstance().bt_size) {
      return false;
    }
  }
  int64_t kal1_bound = 0;
  int64_t kbl1_bound = 0;
  GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
  int64_t bl1_size = kbl1_bound * l1Status.n_l1 +
                     static_cast<int64_t>(run_params.is_weight_quant_bmm) * kbl1_bound * l1Status.n_l1 / kFp16Bytes;
  int64_t l0c_factor_limit = PlatformInfo::GetInstance().support_l12bt_bf16() ? k1982L0cFactorMax : k1971L0cFactorMax;
  if (!CheckL1Size(l1Status.m_l1 * kal1_bound, bl1_size, channel_wise_l1_size) ||
      !(l0Status.batch_l0 * l0Status.k_l0 * l0Status.m_l0 <= l0FactorLimit) ||
      !(l0Status.batch_l0 * l0Status.k_l0 * l0Status.n_l0 <= l0FactorLimit) ||
      !(l0Status.batch_l0 * l0Status.n_l0 * (l0Status.m_l0 +
        static_cast<int64_t>(run_params.is_weight_quant_bmm) * l0Status.k_l0) <= l0c_factor_limit)) {
    return false;
  }
  return true;
}

bool IsFactorBlockDimSplitk(const BatchmatmulParas& params, int64_t batch_dim, int64_t m_dim, int64_t n_dim) {
  // used in fp16 2 fp32
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  if (!compile_params.split_k_flag || !PlatformInfo::GetInstance().support_l0c2out()) {
    return false;
  }
  return (n_dim == 0) || (m_dim == 0) || (run_params.n % n_dim != 0) || (run_params.m % m_dim != 0) ||
         (run_params.batch % batch_dim != 0);
}

void WQuantBmmAlignInit(const BatchmatmulRunParas &run_params, L1Status &l1Status, L0Status &l0Status, int64_t &k_64) {
  if (!run_params.is_weight_quant_bmm) {
    return;
  }
  k_64 = MathUtil::Align(k_64, kAlignValue);
  l1Status.n_l1 = MathUtil::Align(l1Status.n_l1, nAlignValue);
  l1Status.kbl1_16 = MathUtil::Align(l1Status.kbl1_16, kAlignValue);
  l1Status.kal1_16 = MathUtil::Align(l1Status.kal1_16, kAlignValue);
  l0Status.n_l0 = min(l1Status.n_l1, nL0PreferSize); // 支持到16和N1的最小值
  while ((l1Status.n_l1 % l0Status.n_l0 != 0)) {
    l0Status.n_l0 -= nAlignValue; // 基本块退化逻辑
  }
}

bool QuantAlign(const BatchmatmulRunParas &run_params, L1Status &l1Status, L0Status &l0Status) {
  if (run_params.is_weight_quant_bmm || run_params.dtype_a != static_cast<int32_t>(ge::DT_INT8)) {
    return true;
  }
  OP_LOGD("MatMul", "before fix tiling for int8 load2d_transpose, n_l0:%ld, n_l1:%ld, m_l0:%ld, m_l1:%ld.",
          l0Status.n_l0, l1Status.n_l1, l0Status.m_l0, l1Status.m_l1);
  if (run_params.m_quant_check) {
    l1Status.m_l1 = MathUtil::Align(l1Status.m_l1, mAlignValue);
    l0Status.m_l0 = MathUtil::Align(l0Status.m_l0, mAlignValue);
  }
  if (run_params.n_quant_check) {
    l1Status.n_l1 = MathUtil::Align(l1Status.n_l1, nAlignValue);
    l0Status.n_l0 = MathUtil::Align(l0Status.n_l0, nAlignValue);
  }
  int64_t nl0_nparts = l1Status.n_l1 / l0Status.n_l0;
  int64_t ml0_nparts = l1Status.m_l1 / l0Status.m_l0;
  while (l0Status.k_l0 * (l0Status.m_l0 + l0Status.n_l0) > k1971L0cFactorMax) {
    if (l0Status.n_l0 >= l0Status.m_l0) {
      l0Status.n_l0 = l0Status.n_l0 - nAlignValue;
    } else {
      l0Status.m_l0 = l0Status.m_l0 - mAlignValue;
    }
  }
  if (l0Status.n_l0 <= 0 || l0Status.m_l0 <= 0) {
    return false;
  }
  l1Status.n_l1 = l0Status.n_l0 * nl0_nparts;
  l1Status.m_l1 = l0Status.m_l0 * ml0_nparts;
  OP_LOGD("MatMul", "after fix tiling for int8 load2d_transpose, n_l0:%ld, n_l1:%ld, m_l0:%ld, m_l1:%ld.",
          l0Status.n_l0, l1Status.n_l1, l0Status.m_l0, l1Status.m_l1);
  return true;
}

void WeightQuantBmmNl0Fix(const BatchmatmulRunParas &run_params, L1Status &l1Status, L0Status &l0Status) {
  int64_t nl0_nparts = l1Status.n_l1 / l0Status.n_l0;
  int64_t ml0_nparts = l1Status.m_l1 / l0Status.m_l0;
  int64_t nbl1 = 2; // nbl1 is 2 * nbl0 to enable l0c_db
  while (l0Status.n_l0 * (l0Status.m_l0 + l0Status.k_l0) * kDbOn > k1971L0cFactorMax) {
    if (l0Status.n_l0 >= l0Status.m_l0) {
      l0Status.n_l0 = l0Status.n_l0 - nAlignValue;
    } else {
      l0Status.m_l0 = l0Status.m_l0 - 1;
    }
  }
  while (l0Status.n_l0 > run_params.n + 1) {
    l0Status.n_l0 = l0Status.n_l0 - nAlignValue;
  }
  l0Status.n_l0 = max(l0Status.n_l0, 1L);
  l1Status.n_l1 = l0Status.n_l0 * nl0_nparts;
  l1Status.m_l1 = l0Status.m_l0 * ml0_nparts;
  if (l0Status.n_l0 > kWeightQuantBmmNL0Max) {
    l0Status.n_l0 = MathUtil::CeilDivision(l0Status.n_l0, nbl1);
    l0Status.n_l0 = MathUtil::Align(l0Status.n_l0, nAlignValue);
    if (l0Status.n_l0 * (l0Status.m_l0 * nbl1 + l0Status.k_l0) * kDbOn > k1971L0cFactorMax) {
      l0Status.n_l0 = l0Status.n_l0 - nAlignValue;
    }
    l1Status.n_l1 = l0Status.n_l0 * nbl1;
    while (l1Status.n_l1 > run_params.n)
    {
      l0Status.n_l0 = l0Status.n_l0 - nAlignValue;
      l1Status.n_l1 = l0Status.n_l0 * nbl1;
    }
  }
}

void WeightQuantBmmAlign(const BatchmatmulRunParas &run_params, L1Status &l1Status,
                         L0Status &l0Status, int64_t &k_64) {
  if (!run_params.is_weight_quant_bmm) {
    return;
  }
  k_64 = MathUtil::Align(k_64, kAlignValue);
  l1Status.n_l1 = MathUtil::Align(l1Status.n_l1, nAlignValue);
  l1Status.kbl1_16 = MathUtil::Align(l1Status.kbl1_16, kAlignValue);
  l1Status.kal1_16 = MathUtil::Align(l1Status.kal1_16, kAlignValue);
  l0Status.k_l0 = MathUtil::Align(l0Status.k_l0, kAlignValue);
  l0Status.n_l0 = MathUtil::Align(l0Status.n_l0, nAlignValue);
  // kal1 can't be larger than kbl1
  l1Status.kal1_16 = min(l1Status.kal1_16, l1Status.kbl1_16);
  if (l1Status.kbl1_16 % l1Status.kal1_16 != 0) {
    l1Status.kal1_16 -= kAlignValue;
  }
  l0Status.k_l0 = min(l0Status.k_l0, kWeightQuantBmmKL0Max);
  // tiling of diagonal_matrix in l0a (k_l0, kl0//2, 16, 32)
  while (min(l1Status.kbl1_16, l1Status.kal1_16) % l0Status.k_l0 != 0 || k_64 % l0Status.k_l0 != 0) {
    l0Status.k_l0 -= kAlignValue;
  }
  WeightQuantBmmNl0Fix(run_params, l1Status, l0Status);
  if ((l0Status.k_l0 > kMadInt32kThreshold) and (l0Status.k_l0 % kMadInt32kThreshold == 0)) {
    l0Status.k_l0 = kMadInt32kThreshold;
  }
  // 2 means pingpong
  bool l0c_flag = (l0Status.n_l0 * l0Status.k_l0 * FractalSize * PingPong >
                   MathUtil::CeilDivision(PlatformInfo::GetInstance().l0c_size, PingPong));
  if (l0c_flag && (l0Status.k_l0 % PingPong == 0)) {
    l0Status.k_l0 = PingPong;
  }
  if (run_params.k % kAlignValue != 0) {
    l0Status.m_l0 = 1;
  }
  OP_LOGD("MatMul", "before fix tiling for int8 load2d_transpose, n_l0:%ld, n_l1:%ld, m_l0:%ld, m_l1:%ld.",
          l0Status.n_l0, l1Status.n_l1, l0Status.m_l0, l1Status.m_l1);
}

bool SetBothFullLoadParams(const BatchmatmulRunParas &run_params, CoreStatus &coreStatus,
                           SingleCoreStatus &singleCoreStatus) {
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  int64_t m_dim = coreStatus.m_dim;
  int64_t n_dim = coreStatus.n_dim;
  int64_t k_64 = run_params.k;
  l1Status.n_l1 = MathUtil::CeilDivision(run_params.n, n_dim);
  l1Status.m_l1 = MathUtil::CeilDivision(run_params.m, m_dim);
  // Initialization
  l0Status.batch_l0 = 1;
  l0Status.l0c_multi_batch = 0;
  bool b_full_load_invalid = !run_params.trans_b_flag && l1Status.n_l1 % fullCacheLine != 0 && n_dim != 1;
  bool a_full_load_invalid = run_params.trans_a_flag && l1Status.m_l1 % fullCacheLine != 0 && m_dim != 1;
  if (b_full_load_invalid  || a_full_load_invalid) {
    return false;
  }
  l1Status.kal1_16 = run_params.k;
  l1Status.kbl1_16 = run_params.k;

  l0Status.n_l0 = min(l1Status.n_l1, nL0PreferSize); // 支持到16和N1的最小值
  l0Status.m_l0 = min(l1Status.m_l1, kML0PreferSize); // 支持到8和M1的最小值
  while ((l1Status.n_l1 % l0Status.n_l0 != 0)) {
    l0Status.n_l0 -= 1; // 基本块退化逻辑
  }
  WQuantBmmAlignInit(run_params, l1Status, l0Status, k_64);
  l0Status.k_l0 = min(l0FactorLimit / l0Status.n_l0, k_64);

  while ((l1Status.m_l1 % l0Status.m_l0 != 0)) {
    l0Status.m_l0 -= 1; // 基本块退化逻辑
  }
  l0Status.k_l0 = min(l0FactorLimit / l0Status.m_l0, l0Status.k_l0);
  if (run_params.is_weight_quant_bmm) {
    l0Status.k_l0 = min(l0Status.k_l0, kWeightQuantBmmKL0Max);
  }

  while ((l1Status.kal1_16 % l0Status.k_l0 != 0 || l0Status.k_l0 % kAlignValue != 0)) {
    l0Status.k_l0 -= 1; // 基本块退化逻辑
  }
  if (run_params.is_weight_quant_bmm) {
    while (k_64 % l0Status.k_l0 != 0) {
      l0Status.k_l0 -= kAlignValue;
    }
  }
  bool is_fp16_nz = run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT16) && !run_params.format_a_nd &&
                    !run_params.format_b_nd && PlatformInfo::GetInstance().support_l0c2out();
  bool mkn_full_load = l1Status.m_l1 * l1Status.kal1_16 < kL0FactorLimit &&
                       l1Status.m_l1 * l1Status.n_l1 < kL0FactorLimit &&
                       l1Status.kal1_16 * l1Status.n_l1 < kL0FactorLimit && is_fp16_nz;
  bool m_full_load = l1Status.m_l1 * l0Status.k_l0 < kL0FactorLimit &&
                     l1Status.m_l1 * l0Status.n_l0 < kL0FactorLimit && is_fp16_nz;
  bool k_full_load = l0Status.m_l0 * l1Status.kal1_16 < kL0FactorLimit &&
                     l1Status.kal1_16 * l0Status.n_l0 < kL0FactorLimit && is_fp16_nz;
  bool n_full_load = l0Status.m_l0 * l1Status.n_l1 < kL0FactorLimit &&
                     l0Status.k_l0 * l1Status.n_l1 < kL0FactorLimit && is_fp16_nz;
  if (mkn_full_load) {
    l0Status.m_l0 = l1Status.m_l1;
    l0Status.k_l0 = l1Status.kal1_16;
    l0Status.n_l0 = l1Status.n_l1;
  }
  if (m_full_load) {
    l0Status.m_l0 = l1Status.m_l1;
  }
  if (k_full_load) {
    l0Status.k_l0 = l1Status.kal1_16;
  }
  if (n_full_load) {
    l0Status.n_l0 = l1Status.n_l1;
  }
  if (run_params.is_weight_quant_bmm && l1Status.n_l1 > l0Status.n_l0) {
    return false;
  }
  WeightQuantBmmAlign(run_params, l1Status, l0Status, k_64);
  ASSERT_TRUE(QuantAlign(run_params, l1Status, l0Status), return false);
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  GetL0BatchFactor(coreStatus, l0Status, run_params, false);
  singleCoreStatus.ubStatus.batch_cub = l0Status.batch_l0;
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return false);
  ASSERT_TRUE(CheckTilingBuffer(run_params, singleCoreStatus), return false);
  return true;
}

void FastFindBothFullLoad(const string &op_type, BatchmatmulParas &params, GemmEstimate *est) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  CoreStatus tmpCoreStatus;
  SingleCoreStatus tmpSingleCoreStatus;
  UpdateSingleCoreStatus(params, tmpSingleCoreStatus);
  int64_t batch_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.batch);
  for (int64_t batch_dim = 1; batch_dim <= batch_dim_max; batch_dim++) {
    int64_t n_dim_max = min(PlatformInfo::GetInstance().core_num / batch_dim, run_params.n);
    for (int64_t n_dim = 1; n_dim <= n_dim_max; n_dim++) {
      int64_t m_dim_max = min(PlatformInfo::GetInstance().core_num / n_dim / batch_dim, run_params.m);
      for (int64_t m_dim = 1; m_dim <= m_dim_max; m_dim++) {
        tmpCoreStatus.batch_dim = batch_dim;
        tmpCoreStatus.n_dim = n_dim;
        tmpCoreStatus.m_dim = m_dim;
        if (IsFactorBlockDimSplitk(params, batch_dim, m_dim, n_dim)) {
          continue;
        }
        if (!SetBothFullLoadParams(run_params, tmpCoreStatus, tmpSingleCoreStatus)) {
          continue;
        }
        AddCondition(op_type, params, tmpCoreStatus, tmpSingleCoreStatus, est);
      }
    }
  }
}

bool SetAl1FullLoadParams(const BatchmatmulRunParas &run_params, CoreStatus &coreStatus,
                          SingleCoreStatus &singleCoreStatus) {
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  int64_t m_dim = coreStatus.m_dim;
  int64_t n_dim = coreStatus.n_dim;
  l1Status.m_l1 = MathUtil::CeilDivision(run_params.m, m_dim);
  ASSERT_TRUE(ExpandShape(run_params, l1Status, m_dim, true), return false);
  ASSERT_TRUE(CheckL1Size(l1Status.m_l1 * l1Status.kal1_16, 1), return false);
  int64_t amat = l1Status.m_l1 * l1Status.kal1_16;
  if (l1Status.m_l1 <= k1980FullCacheLine) {
    l0Status.m_l0 = l1Status.m_l1;
  } else if (l1Status.m_l1 % k1980FullCacheLine == 0) {
    // al1 full load prefer pattern (m0=16, k0=4, n0=8)
    l0Status.m_l0 = l1Status.m_l1 % kMNL0PreferSize ? k1980FullCacheLine : kMNL0PreferSize;
  } else {
    return false;
  }
  l0Status.k_l0 = min(l0FactorLimit / l0Status.m_l0, k1980FullCacheLine);
  while ((l1Status.kal1_16 % l0Status.k_l0) || (!CheckL1Size(amat, l0Status.k_l0 * k1980FullCacheLine))) {
    l0Status.k_l0 -= 1;
  }

  int64_t core_n = MathUtil::CeilDivision(run_params.n, n_dim);
  l0Status.n_l0 = core_n >= k1980FullCacheLine ? k1980FullCacheLine : core_n;
  // bl1 not full load prefer to be same as bl0
  l1Status.n_l1 = l0Status.n_l0;
  l1Status.kbl1_16 = l0Status.k_l0;
  ASSERT_TRUE(CheckL1Size(amat, l1Status.kbl1_16 * l1Status.n_l1), return false);
  bool is_k_full_bandwidth = run_params.trans_b_flag && run_params.k % k1980FullCacheLine == 0 &&
                             k1980FullCacheLine % l0Status.k_l0 == 0 &&
                             CheckL1Size(amat, k1980FullCacheLine * l1Status.n_l1);
  l1Status.kbl1_16 = is_k_full_bandwidth ? k1980FullCacheLine : l1Status.kbl1_16;
  bool is_nl0_16 = core_n >= kMNL0PreferSize && (l0Status.k_l0 * kMNL0PreferSize <= l0FactorLimit) &&
                   (l0Status.m_l0 * kMNL0PreferSize * kDbOn <= k1980L0cFactorLimit) &&
                   CheckL1Size(amat, l1Status.kbl1_16 * kMNL0PreferSize);
  if (is_nl0_16) {
    l0Status.n_l0 = kMNL0PreferSize;
    l1Status.n_l1 = l0Status.n_l0;
  }
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return false);
  return true;
}

bool FixOuterAxisTiling(const BatchmatmulParas &params, SingleCoreStatus &singleCoreStatus, int64_t m_dim,
                        int64_t n_dim, bool split_k_flag);

// milan and david will use this method
bool SetAl1FullLoadParamsOpti(const BatchmatmulParas &params, CoreStatus &coreStatus,
                              SingleCoreStatus &singleCoreStatus) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  int64_t m_dim = coreStatus.m_dim;
  int64_t n_dim = coreStatus.n_dim;
  int64_t k_64 = run_params.k;
  l1Status.m_l1 = MathUtil::CeilDivision(run_params.m, m_dim);
  // enabled in fp32, enabled in fp16 after test
  if (run_params.trans_a_flag && l1Status.m_l1 % fullCacheLine != 0 && m_dim != 1 &&
      (run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) || run_params.bias_flag)) {
    return false;
  }
  int64_t kal1_bound = 0;
  int64_t kbl1_bound = 0;
  GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
  ASSERT_TRUE(ExpandShape(run_params, l1Status, m_dim, true), return false);
  ASSERT_TRUE(CheckL1Size(l1Status.m_l1 * kal1_bound, 1,
                          static_cast<int64_t>(run_params.bias_flag)), return false);
  if (run_params.trans_a_flag && (l1Status.m_l1 < kML1MiniSize) && (m_dim != 1)) {
    return false;
  }
  int64_t amat = l1Status.m_l1 * kal1_bound;

  if (l1Status.m_l1 <= fullCacheLine) {
    l0Status.m_l0 = l1Status.m_l1;
  } else if (l1Status.m_l1 % kML0PreferSize == 0) {
    l0Status.m_l0 = kML0PreferSize;
  } else {
    return false;
  }
  WQuantBmmAlignInit(run_params, l1Status, l0Status, k_64);
  l0Status.k_l0 = min(l0FactorLimit / l0Status.m_l0, k_64);

  int64_t core_n = MathUtil::CeilDivision(run_params.n, n_dim);
  l0Status.n_l0 = min(core_n, kNL0PreferSize);
  while (((l1Status.kal1_16 % l0Status.k_l0) != 0) || !(l0Status.k_l0 * l0Status.m_l0 <= l0FactorLimit) ||
         !(l0Status.k_l0 * l0Status.n_l0 <= l0FactorLimit)) {
    l0Status.k_l0 -= 1;
    if (l0Status.k_l0 == 0) {
      return false;
    }
  }
  l1Status.n_l1 = l0Status.n_l0;
  l1Status.kbl1_16 = l0Status.k_l0;
  int64_t cur_bias_l1_size = run_params.bias_flag ? GetBiasL1Size(run_params, l1Status.n_l1) : 0;
  bool is_k_full_bandwidth = run_params.trans_b_flag && run_params.k % fullCacheLine == 0 &&
                             fullCacheLine % l0Status.k_l0 == 0 &&
                             CheckL1Size(amat, fullCacheLine * l1Status.n_l1, cur_bias_l1_size);
  l1Status.kbl1_16 = is_k_full_bandwidth ? fullCacheLine : l1Status.kbl1_16;
  WeightQuantBmmAlign(run_params, l1Status, l0Status, k_64);
  ASSERT_TRUE(CheckL1Size(l1Status.m_l1 * kal1_bound, kbl1_bound * l1Status.n_l1 * kDbOn,
                          cur_bias_l1_size * kDbOn), return false);
  ASSERT_TRUE(QuantAlign(run_params, l1Status, l0Status), return false);
  ASSERT_TRUE(FixOuterAxisTiling(params, singleCoreStatus, m_dim, n_dim, compile_params.split_k_flag), return false);
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return false);
  ASSERT_TRUE(CheckTilingBuffer(run_params, singleCoreStatus), return false);
  if (run_params.is_weight_quant_bmm) {
    kal1_bound = 0;
    kbl1_bound = 0;
    GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
    int64_t cur_al1_size = l1Status.m_al1 * l0Status.m_l0 * kal1_bound;
    int64_t cur_bl1_size = l1Status.n_bl1 * l0Status.n_l0 * kbl1_bound * kDbOn;
    cur_bl1_size = cur_bl1_size + cur_bl1_size / kFp16Bytes;
    int64_t channel_wise_l1_size = GetChannelWiseL1Size(l1Status, l1Status.n_bl1 * l0Status.n_l0) * kDbOn;
    ASSERT_TRUE(CheckL1Size(cur_al1_size, cur_bl1_size, channel_wise_l1_size), return false);
  }
  return true;
}

void FastFindAl1FullLoad(const string &op_type, BatchmatmulParas &params, GemmEstimate *est) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  CoreStatus tmpCoreStatus;
  SingleCoreStatus tmpSingleCoreStatus;
  UpdateSingleCoreStatus(params, tmpSingleCoreStatus);
  // AL1 full load的最小化条件就是K维度能全部载入AL1
  tmpSingleCoreStatus.l1Status.kal1_16 = run_params.k;
  int64_t batch_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.batch);
  for (int64_t batch_dim = 1; batch_dim <= batch_dim_max; batch_dim++) {
    int64_t m_dim_max = min(PlatformInfo::GetInstance().core_num / batch_dim, run_params.m);
    for (int64_t m_dim = 1; m_dim <= m_dim_max; m_dim++) {
      int64_t n_dim_max = min(PlatformInfo::GetInstance().core_num / batch_dim / m_dim, run_params.n);
      // filter small case which use core_num smaller than 24 in tuscany, 16 in milan
      int64_t perf_min_core_num = PlatformInfo::GetInstance().support_l0c2out() ? 16 : 24;
      // filter small case which use core_num smaller than 24 in david
      perf_min_core_num = PlatformInfo::GetInstance().support_l12bt_bf16() ? 24 : perf_min_core_num;
      int64_t n_dim_min = MathUtil::CeilDivision(MathUtil::CeilDivision(perf_min_core_num, batch_dim), m_dim);
      if (run_params.m <= k1980FullCacheLine && m_dim > 1) {
        break;
      }
      for (int64_t n_dim = n_dim_min; n_dim <= n_dim_max; n_dim++) {
        tmpCoreStatus.batch_dim = batch_dim;
        tmpCoreStatus.n_dim = n_dim;
        tmpCoreStatus.m_dim = m_dim;
        if (IsFactorBlockDimSplitk(params, batch_dim, m_dim, n_dim)) {
          continue;
        }
        bool is_set_al1 = PlatformInfo::GetInstance().support_l0c2out()
                              ? SetAl1FullLoadParamsOpti(params, tmpCoreStatus, tmpSingleCoreStatus)
                              : SetAl1FullLoadParams(run_params, tmpCoreStatus, tmpSingleCoreStatus);
        if (!is_set_al1) {
          continue;
        }
        AddCondition(op_type, params, tmpCoreStatus, tmpSingleCoreStatus, est);
      }
    }
  }
}

bool SetBl1FullLoadParams(const BatchmatmulRunParas &run_params, CoreStatus &coreStatus,
                          SingleCoreStatus &singleCoreStatus) {
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  int64_t m_dim = coreStatus.m_dim;
  int64_t n_dim = coreStatus.n_dim;
  l1Status.n_l1 = MathUtil::CeilDivision(run_params.n, n_dim);
  ASSERT_TRUE(ExpandShape(run_params, l1Status, n_dim, false), return false);
  ASSERT_TRUE(CheckL1Size(1, l1Status.kbl1_16 * l1Status.n_l1), return false);
  int64_t bmat = l1Status.kbl1_16 * l1Status.n_l1;
  if (l1Status.n_l1 <= k1980FullCacheLine) {
    l0Status.n_l0 = l1Status.n_l1;
  } else if (l1Status.n_l1 % k1980FullCacheLine == 0) {
    // bl1 full load prefer pattern (m0=4, k0=4, n0=16)
    l0Status.n_l0 = l1Status.n_l1 % kMNL0PreferSize ? k1980FullCacheLine : kMNL0PreferSize;
  } else {
    return false;
  }
  l0Status.k_l0 = min(l0FactorLimit / l0Status.n_l0, k1980FullCacheLine);
  while (l1Status.kbl1_16 % l0Status.k_l0) {
    l0Status.k_l0 -= 1;
  }

  int64_t core_m = MathUtil::CeilDivision(run_params.m, m_dim);
  // bl1 full load prefer pattern (m0=4, k0=4, n0=16)
  l0Status.m_l0 = min(core_m, 4L);
  // bl1 not full load prefer to be same as bl0
  l1Status.m_l1 = l0Status.m_l0;
  l1Status.kal1_16 = l0Status.k_l0;
  ASSERT_TRUE(CheckL1Size(l1Status.m_l1 * l1Status.kal1_16, bmat), return false);
  bool is_k_full_bandwidth = !run_params.trans_a_flag && run_params.k % k1980FullCacheLine == 0 &&
                             k1980FullCacheLine % l0Status.k_l0 == 0 &&
                             CheckL1Size(l1Status.m_l1 * k1980FullCacheLine, bmat);
  l1Status.kal1_16 = is_k_full_bandwidth ? k1980FullCacheLine : l1Status.kal1_16;
  bool is_m_full_bandwidth = core_m >= k1980FullCacheLine && CheckL1Size(k1980FullCacheLine * l1Status.kal1_16, bmat);
  if (is_m_full_bandwidth) {
    l1Status.m_l1 = k1980FullCacheLine;
    bool is_m_expand = (l1Status.m_l1 * l0Status.k_l0 <= l0FactorLimit) &&
                       (l1Status.m_l1 * l0Status.n_l0 * kDbOn <= k1980L0cFactorLimit);
    if (is_m_expand) {
      l0Status.m_l0 = k1980FullCacheLine;
    }
  }
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return false);
  return true;
}

// milan and david will use this method
bool SetBl1FullLoadParamsOpti(const BatchmatmulParas &params, CoreStatus &coreStatus,
                              SingleCoreStatus &singleCoreStatus) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  int64_t m_dim = coreStatus.m_dim;
  int64_t n_dim = coreStatus.n_dim;
  int64_t k_64 = run_params.k;
  l1Status.n_l1 = MathUtil::CeilDivision(run_params.n, n_dim);
  // enabled in float32, enabled in float16 after test
  bool bl1_full_load_invalid = !run_params.trans_b_flag && l1Status.n_l1 % fullCacheLine != 0 && n_dim != 1 &&
      (run_params.dtype_b == static_cast<int32_t>(ge::DT_FLOAT) || run_params.bias_flag) &&
      !run_params.is_weight_quant_bmm;
  if (bl1_full_load_invalid) {
    return false;
  }
  ASSERT_TRUE(ExpandShape(run_params, l1Status, n_dim, false), return false);
  int64_t kal1_bound = 0;
  int64_t kbl1_bound = 0;
  GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
  ASSERT_TRUE(CheckL1Size(1, kbl1_bound * l1Status.n_l1,
                          static_cast<int64_t>(run_params.bias_flag)), return false);

  if (l1Status.n_l1 <= fullCacheLine) {
    l0Status.n_l0 = l1Status.n_l1;
  } else if (l1Status.n_l1 % nL0PreferSize == 0 || run_params.is_weight_quant_bmm) {
    l0Status.n_l0 = nL0PreferSize;
  } else {
    return false;
  }
  WQuantBmmAlignInit(run_params, l1Status, l0Status, k_64);
  int64_t bmat = kbl1_bound * l1Status.n_l1;
  // In weight_quant_bmm scene,
  // bl1 needs to occupy two space, one for FP16 data(kbl1_bound * l1Status.n_l1) and one for int8 data
  bmat = bmat + static_cast<int64_t>(run_params.is_weight_quant_bmm) * bmat / kFp16Bytes;

  int64_t core_m = MathUtil::CeilDivision(run_params.m, m_dim);
  l0Status.m_l0 = min(core_m, kML0PreferSize);
  l0Status.k_l0 = min(l0FactorLimit / l0Status.n_l0, k_64);
  while ((l1Status.kbl1_16 % l0Status.k_l0 > 0) || !(l0Status.k_l0 * l0Status.n_l0 <= l0FactorLimit) ||
         !(l0Status.k_l0 * l0Status.m_l0 <= l0FactorLimit) || l0Status.k_l0 % kAlignValue != 0) {
    l0Status.k_l0 -= 1;
    if (l0Status.k_l0 == 0) {
      return false;
    }
  }
  int64_t channel_wise_l1_size = GetChannelWiseL1Size(l1Status, l1Status.n_l1);
  // al1 not full load prefer to be same as al0
  l1Status.m_l1 = l0Status.m_l0;
  l1Status.kal1_16 = l0Status.k_l0;
  bool is_k_full_bandwidth = !run_params.trans_a_flag && k_64 % fullCacheLine == 0 &&
                             fullCacheLine % l0Status.k_l0 == 0 &&
                             CheckL1Size(l1Status.m_l1 * fullCacheLine * kDbOn, bmat, channel_wise_l1_size);
  l1Status.kal1_16 = is_k_full_bandwidth ? fullCacheLine : l1Status.kal1_16;
  GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
  bool is_m_full_bandwidth =
      core_m >= fullCacheLine && CheckL1Size(fullCacheLine * kal1_bound * kDbOn, bmat, channel_wise_l1_size);
  if (is_m_full_bandwidth) {
    bool is_m_expand = (fullCacheLine * l0Status.k_l0 <= l0FactorLimit) &&
                       (fullCacheLine * l0Status.n_l0 <= l0cFactorLimit1971);
    if (l1Status.kal1_16 == k_64) {
      l1Status.m_l1 = fullCacheLine;
    }
    if (is_m_expand) {
      l1Status.m_l1 = fullCacheLine;
      l0Status.m_l0 = fullCacheLine;
    }
  }
  WeightQuantBmmAlign(run_params, l1Status, l0Status, k_64);
  ASSERT_TRUE(QuantAlign(run_params, l1Status, l0Status), return false);
  ASSERT_TRUE(CheckL1Size(l1Status.m_l1 * kal1_bound * kDbOn, bmat, channel_wise_l1_size), return false);
  ASSERT_TRUE(FixOuterAxisTiling(params, singleCoreStatus, m_dim, n_dim, compile_params.split_k_flag), return false);
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return false);
  ASSERT_TRUE(CheckTilingBuffer(run_params, singleCoreStatus), return false);
  return true;
}

void FastFindBl1FullLoad(const string &op_type, BatchmatmulParas &params, GemmEstimate *est) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  CoreStatus tmpCoreStatus;
  SingleCoreStatus tmpSingleCoreStatus;
  UpdateSingleCoreStatus(params, tmpSingleCoreStatus);

  tmpSingleCoreStatus.l1Status.kbl1_16 = run_params.k;
  int64_t full_cache_line = PlatformInfo::GetInstance().support_l0c2out() ? fullCacheLine : k1980FullCacheLine;
  int64_t batch_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.batch);
  for (int64_t batch_dim = 1; batch_dim <= batch_dim_max; batch_dim++) {
    int64_t n_dim_max = min(PlatformInfo::GetInstance().core_num / batch_dim, run_params.n);
    for (int64_t n_dim = 1; n_dim <= n_dim_max; n_dim++) {
      int64_t m_dim_max = min(PlatformInfo::GetInstance().core_num / batch_dim / n_dim, run_params.m);
      // filter small case which use core_num smaller than 24 in tuscany, 16 in milan
      int64_t perf_min_core_num = PlatformInfo::GetInstance().support_l0c2out() ? 16 : 24;
      // filter small case which use core_num smaller than 24 in david
      perf_min_core_num = PlatformInfo::GetInstance().support_l12bt_bf16() ? 24 : perf_min_core_num;
      int64_t m_dim_min = MathUtil::CeilDivision(MathUtil::CeilDivision(perf_min_core_num, batch_dim), n_dim);
      if (run_params.n <= full_cache_line && n_dim > 1) {
        break;
      }
      for (int64_t m_dim = m_dim_min; m_dim <= m_dim_max; m_dim++) {
        tmpCoreStatus.batch_dim = batch_dim;
        tmpCoreStatus.n_dim = n_dim;
        tmpCoreStatus.m_dim = m_dim;
        if (IsFactorBlockDimSplitk(params, batch_dim, m_dim, n_dim)) {
          continue;
        }
        bool is_set_bl1 = PlatformInfo::GetInstance().support_l0c2out()
                              ? SetBl1FullLoadParamsOpti(params, tmpCoreStatus, tmpSingleCoreStatus)
                              : SetBl1FullLoadParams(run_params, tmpCoreStatus, tmpSingleCoreStatus);
        if (!is_set_bl1) {
          continue;
        }
        AddCondition(op_type, params, tmpCoreStatus, tmpSingleCoreStatus, est);
      }
    }
  }
}

bool SetNotFullLoadL1Params(int64_t ori_shape, int64_t dim, int64_t &l1_shape,
                            int64_t prefer_size) {
  int64_t core_shape = MathUtil::CeilDivision(ori_shape, dim);
  int64_t core_shape_max = dim == 1 ? core_shape : (MathUtil::CeilDivision(ori_shape, dim - 1) - 1);
  core_shape_max = max(core_shape, core_shape_max);
  l1_shape = min(core_shape, prefer_size);
  while (dim != 1 && (MathUtil::Align(core_shape, l1_shape) > core_shape_max)) {
    l1_shape -= 1;
  }
  if (PlatformInfo::GetInstance().support_l0c2out()) {
    if ((prefer_size == fullCacheLine) && ((l1_shape & (kML0PreferSize - 1)) != 0) &&
        core_shape >= fullCacheLine) {
      return false;
    }
  } else {
    if (l1_shape < k1980FullCacheLine) {
      return false;
    }
  }
  return true;
}

bool SetNotFullLoadParams(const BatchmatmulRunParas &run_params, CoreStatus &coreStatus,
                          SingleCoreStatus &singleCoreStatus, int64_t m_dim, int64_t n_dim) {
  ASSERT_TRUE((m_dim * n_dim >= k1980MinCoreNum), return false);
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;

  ASSERT_TRUE(SetNotFullLoadL1Params(run_params.n, n_dim, l1Status.n_l1, kMNL0PreferSize), return false);
  ASSERT_TRUE(SetNotFullLoadL1Params(run_params.m, m_dim, l1Status.m_l1, kMNL0PreferSize), return false);

  // not full load prefer k_l1 equal to k_l0 and k_l0 smaller
  // k_l0 prefer 2 when k is outer axis, othen condition prefer 4.
  int64_t k_pref_value = (run_params.trans_a_flag && !run_params.trans_b_flag) ? 2 : 4;

  l1Status.kal1_16 = min(run_params.k, k_pref_value);
  l1Status.kbl1_16 = min(run_params.k, k_pref_value);
  if (run_params.trans_b_flag && run_params.k >= k1980FullCacheLine) {
    l1Status.kbl1_16 = k1980FullCacheLine;
  }
  if (!run_params.trans_a_flag && run_params.k >= k1980FullCacheLine) {
    l1Status.kal1_16 = k1980FullCacheLine;
  }
  l0Status.m_l0 = l1Status.m_l1;
  l0Status.n_l0 = l1Status.n_l1;
  l0Status.k_l0 = min(l1Status.kal1_16, l1Status.kbl1_16) >= k_pref_value ? k_pref_value : run_params.k;
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  ASSERT_TRUE(CheckFactor(run_params, coreStatus, singleCoreStatus), return false);
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return false);
  return true;
}

bool FixOuterAxisTiling(const BatchmatmulParas &params, SingleCoreStatus &singleCoreStatus, int64_t m_dim,
                        int64_t n_dim, bool split_k_flag) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  if (!split_k_flag) {
    return true;
  }
  int64_t m_single_core = run_params.m / m_dim;
  if (m_single_core % l1Status.m_l1 != 0) {
    while (m_single_core % l1Status.m_l1 != 0) {
      l1Status.m_l1--;
    }
    l0Status.m_l0 = l1Status.m_l1;
  }
  int64_t n_single_core = run_params.n / n_dim;
  if (n_single_core % l1Status.n_l1 != 0) {
    while (n_single_core % l1Status.n_l1 != 0) {
      l1Status.n_l1--;
    }
    l0Status.n_l0 = l1Status.n_l1;
  }
  return true;
}

void LargeKTiling(const BatchmatmulParas &params, SingleCoreStatus &singleCoreStatus, bool split_k_flag) {
  const BatchmatmulRunParas &run_params = *(params.run_params);

  if (!split_k_flag || (run_params.k * reducedBlockSize < kNd2NzLimitNum)) {
    return;
  }
  L1Status &l1Status = singleCoreStatus.l1Status;
  int64_t bias_value = run_params.bias_flag ? 1 : 0;
  int64_t k_space = (PlatformInfo::GetInstance().l1_size / kBlockSize / reducedBlockSize / inputDtypeBytes);
  k_space = (k_space / (l1Status.n_l1 + l1Status.m_l1) - bias_value) / kNumTwo;
  int64_t k_factor = k_space;

  while ((((k_factor + k1971FullCacheLine - 1) >> kNumFour) << kNumFour) > k_space) {
    k_factor = k_factor / kNumTwo;
  }
  if (run_params.is_weight_quant_bmm && k_factor % kAlignValue != 0) {
    k_factor -= 1;
  }
  // can try to align to k1971FullCacheLine
  l1Status.kal1_16 = k_factor;
  l1Status.kbl1_16 = k_factor;
  int64_t kal1_bound = 0;
  int64_t kbl1_bound = 0;
  GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
  while (max(kal1_bound, kbl1_bound) > k_space && l1Status.kal1_16 > kNumTwo) {
    l1Status.kal1_16 = l1Status.kal1_16 - kNumTwo;
    l1Status.kbl1_16 = l1Status.kbl1_16 - kNumTwo;
    GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
  }
}

void NotUseKfullLoad(const BatchmatmulRunParas &run_params,
                     SingleCoreStatus &singleCoreStatus, int64_t k_dim, bool split_k_flag) {
  L1Status &l1Status = singleCoreStatus.l1Status;
  if (split_k_flag && (l1Status.kal1_16 == MathUtil::CeilDivision(run_params.k, k_dim))) {
    l1Status.kal1_16 = l1Status.kal1_16 / kNumTwo;
  }
  if (split_k_flag && (l1Status.kbl1_16 == MathUtil::CeilDivision(run_params.k, k_dim))) {
    l1Status.kbl1_16 = l1Status.kbl1_16 / kNumTwo;
  }
  if (l1Status.kbl1_16 > l1Status.kal1_16) {
    l1Status.kbl1_16 = l1Status.kbl1_16 / l1Status.kal1_16 * l1Status.kal1_16;
  }
  if (l1Status.kal1_16 > l1Status.kbl1_16) {
    l1Status.kal1_16 = l1Status.kal1_16 / l1Status.kbl1_16 * l1Status.kbl1_16;
  }
}

bool SetNotFullLoadMaxKl0(const BatchmatmulParas &params, CoreStatus &coreStatus,
                          SingleCoreStatus &singleCoreStatus, bool split_k_flag = false) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  int64_t m_dim = coreStatus.m_dim;
  int64_t n_dim = coreStatus.n_dim;
  int64_t single_core_m1 = MathUtil::CeilDivision(run_params.m, m_dim);
  int64_t single_core_n1 = MathUtil::CeilDivision(run_params.n, n_dim);
  if (single_core_m1 > mL0PreferSizeOuterAxis || single_core_n1 > nL0PreferSize ||
      run_params.is_weight_quant_bmm || run_params.format_a_nd) {
    return false;
  }
  int64_t k_64 = MathUtil::CeilDivision(run_params.k, coreStatus.k_dim);
  l1Status.n_l1 = single_core_n1;
  l1Status.m_l1 = single_core_m1;
  l0Status.m_l0 = single_core_m1;
  l0Status.n_l0 = single_core_n1;
  int64_t max_k_l0 = min(l0FactorLimit / l0Status.m_l0, l0cFactorLimit1971 / l0Status.n_l0);
  // align k_l0 to multiply of 4;
  max_k_l0 = max_k_l0 < kKL0PreferSize ? max_k_l0 : max_k_l0 & kKl0Align;
  l0Status.k_l0 = min(k_64, max_k_l0);
  int64_t channel_wise_l1_size = GetChannelWiseL1Size(l1Status, l1Status.n_l1);
  int64_t max_k_l1 = (l1AvailableSize - channel_wise_l1_size) / (kBlockSize * reducedBlockSize * inputDtypeBytes *
                     (l1Status.m_l1 + l1Status.n_l1) * kNumTwo);
  max_k_l1 = min(max_k_l1, k_64) / l0Status.k_l0 * l0Status.k_l0;
  l1Status.kal1_16 = max_k_l1;
  l1Status.kbl1_16 = max_k_l1;
  ASSERT_TRUE(QuantAlign(run_params, l1Status, l0Status), return false);
  ASSERT_TRUE(FixOuterAxisTiling(params, singleCoreStatus, m_dim, n_dim, split_k_flag), return false);
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return false);
  ASSERT_TRUE(CheckTilingBuffer(run_params, singleCoreStatus), return false);
  return true;
}

// milan and david will use this method
bool SetNotFullLoadParamsOpti(const BatchmatmulParas &params, CoreStatus &coreStatus,
                              SingleCoreStatus &singleCoreStatus, bool split_k_flag = false) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  if (SetNotFullLoadMaxKl0(params, coreStatus, singleCoreStatus, split_k_flag)) {
    return true;
  }
  int64_t m_dim = coreStatus.m_dim;
  int64_t n_dim = coreStatus.n_dim;
  int64_t k_64 = MathUtil::CeilDivision(run_params.k, coreStatus.k_dim);
  // for both k inner
  if (run_params.unaligned_flag && (!run_params.trans_a_flag && run_params.trans_b_flag) &&
      (k_64 > kL0PreferSizeOuterAxis && k_64 < fullCacheLine)) {
    k_64 = fullCacheLine;
  }
  bool can_merge_n = false;
  if (!run_params.trans_b_flag && (n_dim == 1) && (run_params.n != 1 && run_params.n <= kML1MiniSize)) {
    l1Status.n_l1 = run_params.n;
    can_merge_n = true;
  } else {
    int64_t prefer_n_l1 = run_params.trans_b_flag ? kL0PreferSizeOuterAxis : fullCacheLine;
    ASSERT_TRUE(SetNotFullLoadL1Params(run_params.n, n_dim, l1Status.n_l1, prefer_n_l1), return false);
  }
  int64_t prefer_m_l1 = run_params.trans_a_flag ? fullCacheLine : mL0PreferSizeOuterAxis;
  ASSERT_TRUE(SetNotFullLoadL1Params(run_params.m, m_dim, l1Status.m_l1, prefer_m_l1), return false);
  // kal1 prefer use 8

  l1Status.kbl1_16 = k_64;
  l1Status.kal1_16 = k_64;
  int64_t kal1_bound = 0;
  int64_t kbl1_bound = 0;
  GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
  WQuantBmmAlignInit(run_params, l1Status, l0Status, k_64);
  int64_t channel_wise_l1_size = GetChannelWiseL1Size(l1Status, l1Status.n_l1);
  int64_t bmat = kbl1_bound * l1Status.n_l1 +
                 static_cast<int64_t>(run_params.is_weight_quant_bmm) * kbl1_bound * l1Status.n_l1 / kFp16Bytes;
  if ((!can_merge_n || split_k_flag) ||
      !CheckL1Size(l1Status.m_l1 * kal1_bound, bmat, channel_wise_l1_size)) {
    l1Status.kal1_16 = min(k_64, kL0PreferSizeOuterAxis);
    l1Status.kbl1_16 = min(k_64, kBL1PreferSize);

    int64_t prefer_k_l1 = fullCacheLine;
    if (run_params.unaligned_flag && k_64 >= fullCacheLineUnalign) {
      prefer_k_l1 = fullCacheLineUnalign;
    }
    if (can_merge_n) {
      // when merge n, k*n not big than 256
      int64_t tempKbl1 = min(256 / l1Status.n_l1, k_64);
      if (tempKbl1 == k_64 && tempKbl1 % kL0PreferSize == 0) {
        tempKbl1 = tempKbl1 / l1Status.kal1_16 * l1Status.kal1_16;
        l1Status.kbl1_16 = tempKbl1;
      }
    } else if (run_params.trans_b_flag && k_64 >= prefer_k_l1) {
      l1Status.kbl1_16 = prefer_k_l1;
    }
    if (!run_params.trans_a_flag && k_64 >= prefer_k_l1) {
      l1Status.kal1_16 = prefer_k_l1;
    }
    LargeKTiling(params, singleCoreStatus, split_k_flag);
  }
  // k_al1 is inner axis, hope align to 512B, so need mod 16
  if (can_merge_n && l1Status.kbl1_16 > l1Status.kal1_16 &&
      (!run_params.trans_a_flag && (l1Status.kbl1_16 & (reducedBlockSize - 1)) == 0)) {
    l1Status.kal1_16 = l1Status.kbl1_16;
  }
  // Avoiding fp32 precision issues, fix this
  NotUseKfullLoad(run_params, singleCoreStatus, coreStatus.k_dim, split_k_flag);
  l0Status.m_l0 = l1Status.m_l1;
  l0Status.n_l0 = l1Status.n_l1;
  if (l0Status.m_l0 == l0Status.n_l0 && l0Status.m_l0 == kMNL0PreferSize) {
    l0Status.m_l0 = kML0PreferSize;
    if (k_64 != l1Status.kal1_16) {
      l1Status.m_l1 = l0Status.m_l0;
    }
  }
  int64_t min_k_l1 = min(l1Status.kal1_16, l1Status.kbl1_16);
  if (can_merge_n && (l0Status.m_l0 * min_k_l1 <= l0FactorLimit) && (min_k_l1 * l0Status.n_l0 <= l0FactorLimit)) {
    l0Status.k_l0 = min_k_l1;
  } else {
    // k_l0 prefer 4
    l0Status.k_l0 = min(l1Status.kal1_16, l1Status.kbl1_16) >= kL0PreferSize ? kL0PreferSize : k_64;
  }

  // if small k, use bigger ml1
  if (l1Status.kal1_16 == k_64 && !run_params.trans_a_flag) {
    int64_t min_m1 = max(MathUtil::CeilDivision(l0FactorLimit, l1Status.kal1_16), l1Status.m_l1);
    int64_t max_m1 = min(min_m1 * kNumTwo, MathUtil::CeilDivision(run_params.m, m_dim));
    for (int64_t i = min_m1; i <= max_m1; i += l0Status.m_l0) {
      if (run_params.m % i == 0) {
        l1Status.m_l1 = i;
        break;
      }
    }
  }

  // if small m0 and n0, use bigger k0
  int64_t max_k_l0 =
      min(MathUtil::CeilDivision(l0FactorLimit, l0Status.m_l0), MathUtil::CeilDivision(l0FactorLimit, l0Status.n_l0));
  if (l0Status.k_l0 < max_k_l0) {
    for (int64_t i = l0Status.k_l0 + 1; i <= max_k_l0; ++i) {
      if (max_k_l0 % i == 0 && l1Status.kal1_16 % i == 0) {
        l0Status.k_l0 = i;
        l1Status.kbl1_16 = max(l0Status.k_l0, l1Status.kbl1_16);
        break;
      }
    }
  }
  while ((l1Status.kbl1_16 % l0Status.k_l0 != 0)) {
    l0Status.k_l0--;
  }
  WeightQuantBmmAlign(run_params, l1Status, l0Status, k_64);
  ASSERT_TRUE(QuantAlign(run_params, l1Status, l0Status), return false);
  ASSERT_TRUE(FixOuterAxisTiling(params, singleCoreStatus, m_dim, n_dim, split_k_flag), return false);
  SetBufferParams(run_params, coreStatus, singleCoreStatus);
  ASSERT_TRUE(CheckExpandRatio(run_params, coreStatus, singleCoreStatus), return false);
  ASSERT_TRUE(CheckTilingBuffer(run_params, singleCoreStatus), return false);
  if (run_params.is_weight_quant_bmm) {
    GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
    int64_t cur_al1_size = l1Status.m_al1 * l0Status.m_l0 * kal1_bound * kDbOn;
    int64_t cur_bl1_size = l1Status.n_bl1 * l0Status.n_l0 * kbl1_bound * kDbOn;
    cur_bl1_size = cur_bl1_size + cur_bl1_size / kFp16Bytes;
    channel_wise_l1_size = GetChannelWiseL1Size(l1Status, l1Status.n_bl1 * l0Status.n_l0) * kDbOn;
    ASSERT_TRUE(CheckL1Size(cur_al1_size, cur_bl1_size, channel_wise_l1_size), return false);
    if (coreStatus.kal1_factor != 1 && coreStatus.kbl1_factor == 1) {
      return false;
    }
  }
  return true;
}

void DoAlignCoreShape(int64_t shape, int64_t &dim) {
  uint32_t core_shape = static_cast<uint32_t>(MathUtil::CeilDivision(shape, dim));
  // align core_shape to 8, the number 7 is "8 - 1", the number 3 is from "pow(2, 3)"
  uint32_t core_shape_align = ((core_shape + 7) >> 3) << 3;
  // if expand ratio big than 2, bad perf
  if ((core_shape != 0) && (core_shape_align / core_shape >= 2)) {
    // align core_shape to 4, the number 3 is "4 - 1", the number 2 is from "pow(2, 2)"
    core_shape_align = ((core_shape + 3) >> 2) << 2;
  }
  dim = MathUtil::CeilDivision(shape, core_shape_align);
  return;
}

void FixmDimForAlignMcore(int64_t &m_dim, const BatchmatmulRunParas &run_params) {
  if (!run_params.trans_a_flag && m_dim == run_params.m) {
    return;
  }
  DoAlignCoreShape(run_params.m, m_dim);
  return;
}

void FastFindNotFullLoadSplitK(const string &op_type, BatchmatmulParas &params, bool split_k_flag, GemmEstimate *est) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  CoreStatus tmpCoreStatus;
  SingleCoreStatus tmpSingleCoreStatus;
  UpdateSingleCoreStatus(params, tmpSingleCoreStatus);
  int64_t n_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.n);
  for (int64_t n_dim = 1; n_dim <= n_dim_max; n_dim++) {
    int64_t m_dim_max = min(PlatformInfo::GetInstance().core_num / n_dim, run_params.m);
    if (run_params.n % n_dim != 0) {
      continue;
    }
    for (int64_t m_dim = 1; m_dim <= m_dim_max; m_dim++) {
      if (run_params.m % m_dim != 0) {
        continue;
      }
      tmpCoreStatus.k_dim = min(PlatformInfo::GetInstance().core_num / n_dim / m_dim, run_params.k);
      if (!PlatformInfo::GetInstance().support_l0c2out()) {
        return;
      }
      while (MathUtil::CeilDivision(run_params.k, tmpCoreStatus.k_dim) * reducedBlockSize * inputDtypeBytes <
                 k1971FullCacheLineBytes &&
             tmpCoreStatus.k_dim != 0) {
        tmpCoreStatus.k_dim--;
      }
      if (tmpCoreStatus.k_dim <= 1) {
        continue;
      }
      DoAlignCoreShape(run_params.k, tmpCoreStatus.k_dim);
      tmpCoreStatus.m_dim = m_dim;
      tmpCoreStatus.n_dim = n_dim;
      if (IsFactorBlockDimSplitk(params, 1, m_dim, n_dim)) {
        continue;
      }
      if (!SetNotFullLoadParamsOpti(params, tmpCoreStatus, tmpSingleCoreStatus, split_k_flag)) {
        continue;
      }
      AddCondition(op_type, params, tmpCoreStatus, tmpSingleCoreStatus, est);
    }
  }
}

void FastFindNotFullLoad(const string &op_type, BatchmatmulParas &params, GemmEstimate *est) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  CoreStatus tmpCoreStatus;
  SingleCoreStatus tmpSingleCoreStatus;
  UpdateSingleCoreStatus(params, tmpSingleCoreStatus);
  int64_t batch_dim_max = min(PlatformInfo::GetInstance().core_num, run_params.batch);
  for (int64_t batch_dim = 1; batch_dim <= batch_dim_max; batch_dim++) {
    int64_t n_dim_max = min(PlatformInfo::GetInstance().core_num / batch_dim, run_params.n);
    for (int64_t n_dim = 1; n_dim <= n_dim_max; n_dim++) {
      int64_t m_dim = min(PlatformInfo::GetInstance().core_num / batch_dim / n_dim, run_params.m);
      tmpCoreStatus.m_dim = m_dim;
      tmpCoreStatus.n_dim = n_dim;
      tmpCoreStatus.batch_dim = batch_dim;
      if (PlatformInfo::GetInstance().support_l0c2out()) {
        if (run_params.format_a_nd) {
          FixmDimForAlignMcore(m_dim, run_params);
        }
        tmpCoreStatus.m_dim = m_dim;
        if (!SetNotFullLoadParamsOpti(params, tmpCoreStatus, tmpSingleCoreStatus)) {
          continue;
        }
      } else {
        if (!SetNotFullLoadParams(run_params, tmpCoreStatus, tmpSingleCoreStatus, m_dim, n_dim)) {
          continue;
        }
      }
      AddCondition(op_type, params, tmpCoreStatus, tmpSingleCoreStatus, est);
    }
  }
}

void FastFindParams(const string &op_type, BatchmatmulParas &params, CoreStatus &coreStatus,
                    SingleCoreStatus &singleCoreStatus) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  auto est = GemmEstimateFactory::GetEstimate(CYCLE_ESTIMATE_TYPE, op_type, &params);
  auto est_ptr = est.get();
  if (est_ptr == nullptr) {
    run_params.pattern_flag = false;
    OP_LOGD(op_type.c_str(), "FastFindParams new estimate error.");
    return;
  }
  coreStatus.cycle = INT64_MAX;
  run_params.pattern_flag = false;
  int64_t bl1_min_full_load = run_params.k * MathUtil::CeilDivision(run_params.n,
      PlatformInfo::GetInstance().core_num);
  // In weight_quant_bmm scend, bl1 need 1.5 * bl1_size
  bl1_min_full_load = bl1_min_full_load +
      static_cast<int64_t>(run_params.is_weight_quant_bmm) * bl1_min_full_load / kFp16Bytes;
  int64_t al1_min_full_load = run_params.k * MathUtil::CeilDivision(run_params.m,
      PlatformInfo::GetInstance().core_num);
  // be disabled in float16, can be enabled after test
  if (run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) ||
      run_params.bias_flag || run_params.is_batch_matmul_op) {
    FastFindBothFullLoad(op_type, params, est_ptr);
  }
  // In weight_quant_bmm scene, kbL1 >= kaL1
  if (CheckL1Size(al1_min_full_load, 1, static_cast<int64_t>(run_params.bias_flag))) {
    FastFindAl1FullLoad(op_type, params, est_ptr);
  }
  if (CheckL1Size(1, bl1_min_full_load, static_cast<int64_t>(run_params.bias_flag)) &&
      !run_params.is_weight_quant_bmm) {
    FastFindBl1FullLoad(op_type, params, est_ptr);
  }
  if (compile_params.split_k_flag) {
    FastFindNotFullLoadSplitK(op_type, params, compile_params.split_k_flag, est_ptr);
  } else {
    FastFindNotFullLoad(op_type, params, est_ptr);
  }

  if (est_ptr->GetEstimateResult(coreStatus, singleCoreStatus)) {
    run_params.pattern_flag = true;
  }
  return;
}

void InitilizationProcess(const string &opType, const BatchmatmulRunParas &runParams) {
  OP_LOGD(opType.c_str(), "cache tiling input shape batch:%ld, m:%ld, k:%ld, n:%ld",
          runParams.batch, runParams.m, runParams.k, runParams.n);
  if (PlatformInfo::GetInstance().core_num == 0) {
    OP_LOGE_WITHOUT_REPORT(opType.c_str(), "EXCEPTION: core num is 0");
    return;
  }
  OP_LOGD(opType.c_str(), "The Data type of input is: input_a :%d, input_b:%d",
          static_cast<int32_t>(runParams.dtype_a), static_cast<int32_t>(runParams.dtype_b));
  nAlignValue = runParams.n_quant_check ? kNumTwo : 1;
  mAlignValue = runParams.m_quant_check ? kNumTwo : 1;
  OP_LOGD(opType.c_str(), "cache tiling nAlignValue:%ld, mAlignValue:%ld.", nAlignValue, mAlignValue);
  // In weight_quant_bmm input_weight is int8， k need align to 2.
  kAlignValue = runParams.is_weight_quant_bmm ? 2 : 1;
  l1AvailableSize = PlatformInfo::GetInstance().l1_size;
  if (runParams.is_weight_quant_bmm) {
    l1AvailableSize = PlatformInfo::GetInstance().l1_size - kDiagnoalMatrixSize;
  }
  l0cFactorLimit1971 = runParams.is_weight_quant_bmm ? k1971L0cFactorMax >> 1 : k1971L0cFactorMax;
  inputDtypeBytes = GetDataSize(static_cast<ge::DataType>(runParams.dtype_a));
  outputDtypeBytes = GetDataSize(static_cast<ge::DataType>(runParams.dtype_out));
  fullCacheLine = k1971FullCacheLine;
  fullCacheLineUnalign = k1971FullCacheLineUnalign;
  if (PlatformInfo::GetInstance().support_l12bt_bf16()) {
    fullCacheLine = k1982FullCacheLine;
    fullCacheLineUnalign = k1982FullCacheLineUnalign;
  }
  if (runParams.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) &&
      runParams.dtype_b == static_cast<int32_t>(ge::DT_FLOAT)) {
    // When the date type of Input is Float, the block size in reduce dimension will be 8;
    reducedBlockSize = 8;
    madExpansionRate = (runParams.hf32_flag != 0) ? 1 : 2; // FP32 is 2 times FP16 cube busy cycle
    fullCacheLine = (uint64_t)fullCacheLine >> 1;
    fullCacheLineUnalign = (uint64_t)fullCacheLineUnalign >> 1;
    l0FactorLimit = kL0FactorLimit >> 1;
    nL0PreferSize = kNL0PreferSize >> 1;
    kL0PreferSize = kKL0PreferSize >> 1;
    kBL1PreferSize = kKBL1PreferSize << 1;
    mL0PreferSizeOuterAxis = kML0PreferSizeOuterAxis << 1;
    OP_LOGD(opType.c_str(), "Triggle FP32 process");
  } else if (runParams.dtype_a == static_cast<int32_t>(ge::DT_FLOAT16) ||
             runParams.dtype_a == static_cast<int32_t>(ge::DT_BF16)) {
    reducedBlockSize = kBlockSize;
    l0FactorLimit = kL0FactorLimit;
    nL0PreferSize = kNL0PreferSize;
    kL0PreferSize = kKL0PreferSize;
    kBL1PreferSize = kKBL1PreferSize;
    mL0PreferSizeOuterAxis = kML0PreferSizeOuterAxis;
    OP_LOGD(opType.c_str(), "Triggle FP16 process");
  } else {
    // int8
    // When the date type of Input is int8, the block size in reduce dimension will be 32;
    reducedBlockSize = 32;
    fullCacheLine = (uint64_t)fullCacheLine << 1;
    fullCacheLineUnalign = (uint64_t)fullCacheLineUnalign << 1;
    l0FactorLimit = kL0FactorLimit;
    nL0PreferSize = kNL0PreferSize << 1;
    kL0PreferSize = kKL0PreferSize << 1;
    kBL1PreferSize = kKBL1PreferSize >> 1;
    mL0PreferSizeOuterAxis = kML0PreferSizeOuterAxis >> 1;
    if (runParams.dtype_a != static_cast<int32_t>(ge::DT_INT8)) {
      OP_LOGW(opType.c_str(), "Need set init quick pattern's params for this dtype %d", runParams.dtype_a);
    }
    OP_LOGD(opType.c_str(), "Triggle INT8 process");
  }
}

void CalcTilingPatternMode() {
  prctl(PR_SET_NAME, (unsigned long)("mmCacheTiling"));
  do {
    if (thread_exit_flag) {
      return;
    }
    if (!thread_init_succ) {
      return;
    }
    // sleep 1 us to get list
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    pattern_lock->lock();
    if (pattern_list->empty()) {
      pattern_lock->unlock();
      continue;
    }
    PatternParams pattern_params = pattern_list->front();
    pattern_list->pop_front();
    pattern_lock->unlock();

    const BatchmatmulCompileParas compile_params = pattern_params.compile_params;
    BatchmatmulRunParas run_params = pattern_params.run_params;
    BatchmatmulParas params;
    params.compile_params = &compile_params;
    params.run_params = &run_params;
    CoreStatus coreStatus;
    SingleCoreStatus singleCoreStatus;
    Tiling tiling;
    InitilizationProcess("MatMul", run_params);
    cachetiling::MatmulHashInput hash_input(compile_params, run_params);
    cachetiling::MatmulHashItem hash_value(tiling, run_params, hash_input);
    uint32_t tiling_key = cachetiling::MurmurHash(&hash_input, sizeof(hash_input));
    UpdateSingleCoreStatus(params, singleCoreStatus);
    FastFindParams("MatMul", params, coreStatus, singleCoreStatus);
    if (run_params.pattern_flag) {
      tiling.SetParams(coreStatus, singleCoreStatus.l0Status, singleCoreStatus.l1Status, singleCoreStatus.ubStatus,
                       params);
      tiling.SetAttachFlag();
      tiling.SetL2CacheFlag(params);
      if (run_params.zero_flag) {
        tiling.SetZeroFlagTiling(params);
      }
      tiling.GetTilingId(params);
      if (run_params.zero_flag) {
        continue;
      }
      hash_value.set_tiling(tiling);
      hash_value.set_run_param(run_params);
      // replace tiling to cache
      tiling_hash_cache->Replace(tiling_key, hash_input, hash_value);
    }
  } while (1);
}

void NotifyTiling(const PatternParams &pattern_params) {
  MMDoOnce();
  if (!thread_init_succ) {
    return;
  }
  pattern_lock->lock();
  if (pattern_list->size() < kMMThreadItemNum) {
    pattern_list->push_back(pattern_params);
  }
  pattern_lock->unlock();
  return;
}

void TilingPatternProcess(const string &op_type, BatchmatmulParas &params, CoreStatus &coreStatus,
                          SingleCoreStatus &singleCoreStatus) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  // tiling pattern only support following scenarios now:
  // 1. only support Ascend910A and Ascend910B
  // 2. input and output must be fp16 in Ascend910A
  // 3. Ascend910A only support aligned nd shape
  // 4. Do not split k
  // 5. Do not has bias and BatchMatMul in Ascend910A
  // 6. weight_quant_bmm
  // 7. int8/uint8 is not supported
  bool support_fp32 = (run_params.dtype_out != static_cast<int32_t>(ge::DT_FLOAT) ||
                       PlatformInfo::GetInstance().support_l0c2out());
  bool is_supported_version =
      (PlatformInfo::GetInstance().core_num >= k1980MinCoreNum) || PlatformInfo::GetInstance().support_l0c2out();
  bool is_supported_align_mode = !run_params.unaligned_flag || PlatformInfo::GetInstance().support_l0c2out();
  bool is_supported_bias = !run_params.bias_flag || PlatformInfo::GetInstance().support_l0c2out();
  bool is_supported_split_k = (!compile_params.split_k_flag && !PlatformInfo::GetInstance().support_l0c2out()) ||
      (PlatformInfo::GetInstance().support_l0c2out() && !(run_params.nd_flag && !run_params.format_out_nd));
  bool invalid_nz_shape = run_params.m == 1 || run_params.k == 1 || run_params.n == 1;
  invalid_nz_shape = invalid_nz_shape && (run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT16));
  bool is_support_nz = run_params.nd_flag || (PlatformInfo::GetInstance().support_l0c2out() &&
                                              (run_params.format_a_nd || run_params.format_b_nd || !invalid_nz_shape));
  bool is_batch_matmul_supported = ((!PlatformInfo::GetInstance().support_l0c2out() &&
                                     !run_params.is_batch_matmul_op) ||
                                    PlatformInfo::GetInstance().support_l0c2out());
  bool allow_pattern_flag = is_supported_bias && is_supported_split_k &&
      ((is_supported_version && is_support_nz && is_supported_align_mode &&
        support_fp32 && is_batch_matmul_supported) ||
       (run_params.pattern_flag && !PlatformInfo::GetInstance().support_l0c2out()));
  if (allow_pattern_flag || run_params.is_weight_quant_bmm || run_params.is_compress_quant) {
    PatternParams pattern_params;
    pattern_params.compile_params = compile_params;
    pattern_params.run_params = run_params;
    if (pattern_cache_enable && !PlatformInfo::GetInstance().support_l0c2out() && !run_params.is_compress_quant) {
      NotifyTiling(pattern_params);
    }
    // 1971 version can calculate tiling directly
    if (run_params.is_compress_quant) {
      compress_dequant_cache_tiling::TilingProcess(op_type, params, coreStatus, singleCoreStatus);
      run_params.pattern_flag = true;
    }
    else if (run_params.pattern_flag || PlatformInfo::GetInstance().support_l0c2out()) {
      FastFindParams(op_type, params, coreStatus, singleCoreStatus);
    }
  }
}


void GetTransdataUb(BatchmatmulRunParas &run_params, CoreStatus &core_status, SingleCoreStatus &single_core_status) {
  if (run_params.nz_fusion_flag == 0) {
    return;
  }
  L0Status &l0_status = single_core_status.l0Status;
  UbStatus &ub_status = single_core_status.ubStatus;
  int64_t k1_aub = l0_status.k_l0;
  int64_t m1_aub = l0_status.m_l0;
  int64_t k1_bub = l0_status.k_l0;
  int64_t n1_bub = l0_status.n_l0;
  // 多Core的策略还需要重新思考和设计
  int64_t m_aub_dim = core_status.m_dim;
  int64_t n_bub_dim = core_status.n_dim;
  int64_t k_aub_dim = min(core_status.n_dim, MathUtil::CeilDivision(run_params.ori_shape_k, kBlockSize));
  int64_t k_bub_dim = min(core_status.m_dim, MathUtil::CeilDivision(run_params.ori_shape_k, kBlockSize));

  int64_t aub_size = kUbFp16Size + 1;
  int64_t bub_size = kUbFp16Size + 1;
  bool unvalid_tiling = false;
  // Hardware allow subcore doing nothing now. So there is no need to keep subblock nparts reserved.
  bool a_nz_fusion_flag = (run_params.nz_fusion_flag == kTransdataA) || (run_params.nz_fusion_flag == kTransdataAB);
  bool b_nz_fusion_flag = (run_params.nz_fusion_flag == kTransdataB) || (run_params.nz_fusion_flag == kTransdataAB);
  OP_LOGD("MatMul-NzFusion", "---> nz_fusion_flag = %d", run_params.nz_fusion_flag);
  OP_LOGD("MatMul-NzFusion", "---> kUbFp16Size = %ld", kUbFp16Size);
  if (a_nz_fusion_flag) {
    if (run_params.trans_a_flag && (run_params.ori_shape_m / m_aub_dim < m1_aub * kBlockSize)) {
      m1_aub = MathUtil::CeilDivision(run_params.m, m_aub_dim);
      OP_LOGD("MatMul-NzFusion", "---> m1_aub change from %ld to %ld", l0_status.m_l0, m1_aub);
    } else if (run_params.ori_shape_k / k_aub_dim < k1_aub * kBlockSize) {
      k1_aub = MathUtil::CeilDivision(run_params.k, k_aub_dim);
      OP_LOGD("MatMul-NzFusion", "---> k1_aub change from %ld to %ld", l0_status.k_l0, k1_aub);
    }
    // 补充扩大AUB\BUB的逻辑
    int64_t m_single_vec = MathUtil::CeilDivision(run_params.ori_shape_m, m_aub_dim);
    int64_t ka_single_vec = MathUtil::CeilDivision(run_params.ori_shape_k, k_aub_dim);
    aub_size = k1_aub * m1_aub * kMinFractalSize * kNumTwo;
    OP_LOGD("MatMul-NzFusion", "a_nz_fusion. m_single_core = %ld, k_single_core = %ld\n", m_single_vec, ka_single_vec);
  }
  if (b_nz_fusion_flag) {
    if (run_params.trans_b_flag && (run_params.ori_shape_k / k_bub_dim < k1_bub * kBlockSize)) {
      k1_bub = MathUtil::CeilDivision(run_params.k, k_bub_dim);
      OP_LOGD("MatMul-NzFusion", "---> k1_bub change from %ld to %ld", l0_status.k_l0, k1_bub);
    } else if (run_params.ori_shape_n / n_bub_dim < n1_bub * kBlockSize) {
      n1_bub = MathUtil::CeilDivision(run_params.n, n_bub_dim);
      OP_LOGD("MatMul-NzFusion", "---> n1_bub change from %ld to %ld", l0_status.n_l0, n1_bub);
    }
    // 补充扩大AUB\BUB的逻辑
    bub_size = k1_bub * n1_bub * kMinFractalSize * kNumTwo;
    int64_t kb_single_vec = MathUtil::CeilDivision(run_params.ori_shape_k, k_aub_dim);
    int64_t n_single_vec = MathUtil::CeilDivision(run_params.ori_shape_n, n_bub_dim);
    OP_LOGD("MatMul-NzFusion", "b_nz_fusion_flag. n_single_core = %ld, k_single_core = %ld", n_single_vec,
            kb_single_vec);
  }
  unvalid_tiling |= (aub_size > kUbFp16Size && a_nz_fusion_flag) || (bub_size > kUbFp16Size && b_nz_fusion_flag);
  unvalid_tiling |= (k1_aub == 0) || (m1_aub == 0);
  unvalid_tiling |= (k1_bub == 0) || (n1_bub == 0);
  if (unvalid_tiling) {
    OP_LOGD("MatMul-NzFusion", "Transdata ub params, ori_shape_m:%ld, ori_shape_k:%ld, ori_shape_n:%ld, k1_aub:%ld, "
            "m1_aub:%ld, k1_bub:%ld, n1_bub:%ld, m_aub_dim:%ld, n_bub_dim:%ld, k_aub_dim:%ld, k_bub_dim:%ld\n",
            run_params.ori_shape_m, run_params.ori_shape_k, run_params.ori_shape_n,
            k1_aub, m1_aub, k1_bub, n1_bub, m_aub_dim, n_bub_dim, k_aub_dim, k_bub_dim);
    OP_LOGD("MatMul-NzFusion", "aub_size:%ld, bub_size:%ld\n", aub_size, bub_size);
    return;
  }
  ub_status.k1_aub = k1_aub;
  ub_status.m1_aub = m1_aub;
  ub_status.k1_bub = k1_bub;
  ub_status.n1_bub = n1_bub;
  core_status.m_aub_dim = m_aub_dim;
  core_status.n_bub_dim = n_bub_dim;
  core_status.k_aub_dim = k_aub_dim;
  core_status.k_bub_dim = k_bub_dim;
  OP_LOGD("MatMul-NzFusion", "Transdata ub params, ori_shape_m:%ld, ori_shape_k:%ld, ori_shape_n:%ld, k1_aub:%ld, "
          "m1_aub:%ld, k1_bub:%ld, n1_bub:%ld, m_aub_dim:%ld, n_bub_dim:%ld, k_aub_dim:%ld, k_bub_dim:%ld\n",
          run_params.ori_shape_m, run_params.ori_shape_k, run_params.ori_shape_n,
          k1_aub, m1_aub, k1_bub, n1_bub, m_aub_dim, n_bub_dim, k_aub_dim, k_bub_dim);
}

void GetPadUb(const string &op_type, BatchmatmulRunParas &run_params, CoreStatus &coreStatus, UbStatus &ubStatus) {
  if (run_params.pad_flag == 0 || run_params.nz_fusion_flag > 0) {
    return;
  }
  // disable db
  int64_t total_vector_dim = coreStatus.m_dim * coreStatus.n_dim * coreStatus.k_dim;
  int64_t a_shape[kNumTwo] = {1, 1};
  int64_t b_shape[kNumTwo] = {1, 1};
  int64_t a_tiling[kNumTwo] = {1, 1};
  int64_t b_tiling[kNumTwo] = {1, 1};
  // step1: init output shape
  if (run_params.trans_a_flag) {
    a_shape[0] = run_params.ori_shape_k;
    a_shape[1] = run_params.m_pad;
  } else {
    a_shape[0] = run_params.ori_shape_m;
    a_shape[1] = run_params.k_pad;
  }
  if (run_params.trans_b_flag) {
    b_shape[0] = run_params.ori_shape_n;
    b_shape[1] = run_params.k_pad;
  } else {
    b_shape[0] = run_params.ori_shape_k;
    b_shape[1] = run_params.n_pad;
  }

  // step2: get the tiling of inner axis, bigger is better
  int64_t ub_ele_size = run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) ? kUbFp32Size : kUbFp16Size;
  int64_t align_size = run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) ? kPadFusionSizeFp32 : kPadFusionSize;
  int64_t outer_loop = run_params.dtype_a == static_cast<int32_t>(ge::DT_FLOAT) ? kFp32Bytes : kFp16Bytes;
  a_tiling[1] = MathUtil::Min(MathUtil::Align(a_shape[1], align_size), ub_ele_size);
  b_tiling[1] = MathUtil::Min(MathUtil::Align(b_shape[1], align_size), ub_ele_size);

  // step3: bind all core in outer axis. Even if the outer axis is small, bind all core.
  int64_t vector_dims[kNumTwo] = {total_vector_dim, total_vector_dim};

  // step4: get the tiling of outer axis
  int64_t max_tiling_outer = ub_ele_size / a_tiling[1];
  a_tiling[0] = min(MathUtil::CeilDivision(a_shape[0], vector_dims[0] * outer_loop), max_tiling_outer);
  max_tiling_outer = ub_ele_size / b_tiling[1];
  b_tiling[0] = min(MathUtil::CeilDivision(b_shape[0], vector_dims[1] * outer_loop), max_tiling_outer);

  // step5: output result
  ubStatus.k_aub = a_tiling[1];
  ubStatus.m_aub = a_tiling[0];
  ubStatus.k_bub = b_tiling[0];
  ubStatus.n_bub = b_tiling[1];
  if (run_params.trans_a_flag) {
    std::swap(ubStatus.k_aub, ubStatus.m_aub);
  }
  if (run_params.trans_b_flag) {
    std::swap(ubStatus.k_bub, ubStatus.n_bub);
  }
  coreStatus.aub_dim = vector_dims[0];
  coreStatus.bub_dim = vector_dims[1];

  OP_LOGD(op_type.c_str(),
          "Get pad ub params success, k_aub:%ld, m_aub:%ld, k_bub:%ld, n_bub:%ld, aub_dim:%ld, bub_dim:%ld",
          ubStatus.k_aub, ubStatus.m_aub, ubStatus.k_bub, ubStatus.n_bub, coreStatus.aub_dim, coreStatus.bub_dim);
}

bool IsInvalidTilingForPad(tuningtiling::TuningTilingDefPtr &tuning_tiling, const BatchmatmulRunParas &run_params) {
  auto aoe_mm_tiling = std::static_pointer_cast<tuningtiling::GemmTunnerTiling>(tuning_tiling);
  int64_t total_core_num = aoe_mm_tiling->m_dim * aoe_mm_tiling->k_dim * aoe_mm_tiling->n_dim;
  return (run_params.pad_flag || run_params.nz_fusion_flag) && (total_core_num > PlatformInfo::GetInstance().core_num);
}

void SetAoeMultiBatch(L0Status &l0_status, UbStatus &ub_status) {
  l0_status.l0c_multi_batch = l0_status.batch_l0 > 1 ? 1 : 0;
  if (ub_status.batch_aub < l0_status.batch_l0) {
    // 4 means 0b100 of l0c_multi_batch template
    l0_status.l0c_multi_batch += 4;
  }
  if (ub_status.batch_bub < l0_status.batch_l0) {
    // 2 means 0b010
    l0_status.l0c_multi_batch += 2;
  }
  if (ub_status.batch_cub < l0_status.batch_l0) {
    l0_status.l0c_multi_batch += 1;
  }
}

void UpdateUbTilingParam(const BatchmatmulParas &params, const L0Status &l0_status, const L1Status &l1_status,
                         const CoreStatus &core_status,  UbStatus &ub_status) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  if ((!run_params.format_a_nd && !run_params.format_b_nd) ||
      PlatformInfo::GetInstance().support_l0c2out() ||
      run_params.dtype_a != static_cast<int32_t>(ge::DT_FLOAT16) ||
      run_params.dtype_b != static_cast<int32_t>(ge::DT_FLOAT16) ) {
    return;
  }
  // if need solve bank conflict, tiling of inner_axis + 1
  GetABUbSize(compile_params, run_params, ub_status);
  ub_status.aub_align_bound += ub_status.k_aub * kBlockSize * ub_status.m_aub * reducedBlockSize;
  ub_status.bub_align_bound += ub_status.k_bub * kBlockSize * ub_status.n_bub * reducedBlockSize;
  bool need_nz_to_nd = run_params.format_out_nd && !compile_params.split_k_flag;
  if (need_nz_to_nd && ub_status.n_cub % (kBlockSize / kNumTwo) == 0) {
    int64_t aub_size = run_params.format_a_nd ?
                       static_cast<int64_t>(ub_status.aub_align_bound * ub_status.db_aub * (1 + compile_params.aub_double_num)) : 0;
    int64_t bub_size = run_params.format_b_nd ?
                       static_cast<int64_t>(ub_status.bub_align_bound * ub_status.db_bub * (1 + compile_params.bub_double_num)) : 0;
    int64_t cubSize = static_cast<int64_t>(ub_status.n_cub * l0_status.m_l0 * kBlockSize * \
                       kBlockSize * ub_status.db_cub * (1 + compile_params.fused_double_operand_num));
    // cub solve bank_conflict use extra size
    int64_t extra_size = l0_status.m_l0 * kBlockSize * kBlockSize * ub_status.db_cub;
    int64_t ub_used_size = 0;
    if (run_params.bias_flag) {
       ub_used_size = l0_status.n_l0 * kBlockSize * ub_status.db_cub;
    }
    bool al1_full_load = l1_status.kal1_16 >= run_params.k && core_status.m_dim * l1_status.m_l1 >= run_params.m;
    bool bl1_full_load = l1_status.kbl1_16 >= run_params.k && core_status.n_dim * l1_status.n_l1 >= run_params.n;
    bool aub_full_load = ub_status.k1_aub >= run_params.k && ub_status.m_aub * core_status.m_dim >= run_params.m;
    bool bub_full_load = ub_status.k1_bub >= run_params.k && ub_status.n_bub * core_status.n_dim >= run_params.n;
    bool cub_reuse_aub = al1_full_load && !aub_full_load && run_params.format_a_nd;
    bool cub_reuse_bub = bl1_full_load && !bub_full_load && run_params.format_b_nd;
    if (cub_reuse_aub && cub_reuse_bub) {
      ub_used_size += max(max(aub_size, bub_size), cubSize + extra_size);
    } else if (cub_reuse_aub) {
      ub_used_size += max(aub_size, cubSize + extra_size) + bub_size;
    } else if (cub_reuse_bub) {
      ub_used_size += max(bub_size, cubSize + extra_size) + aub_size;
    } else {
      ub_used_size += aub_size + bub_size + cubSize + extra_size;
    }
    // solve cub conflict when ubuf size is enough
    ub_status.flag_cub_solving_bank_conflict = (!run_params.is_batch_matmul_op && (ub_used_size <= kUbFp16Size));
  }
}

void TranslateAoeTiling(tuningtiling::TuningTilingDefPtr &tuning_tiling, BatchmatmulParas &params,
                        CoreStatus &core_status, SingleCoreStatus &single_core_status) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  auto aoe_mm_tiling = std::static_pointer_cast<tuningtiling::GemmTunnerTiling>(tuning_tiling);
  std::string op_name = "MatMulV2";
  L0Status &l0_status = single_core_status.l0Status;
  L1Status &l1_status = single_core_status.l1Status;
  UbStatus &ub_status = single_core_status.ubStatus;
  core_status.batch_dim = aoe_mm_tiling->batch_dim;
  core_status.m_dim = aoe_mm_tiling->m_dim;
  core_status.k_dim = aoe_mm_tiling->k_dim;
  core_status.n_dim = aoe_mm_tiling->n_dim;
  l0_status.batch_l0 = aoe_mm_tiling->batch_l0;
  l0_status.m_l0 = aoe_mm_tiling->m_l0;
  l0_status.k_l0 = aoe_mm_tiling->k_l0;
  l0_status.n_l0 = aoe_mm_tiling->n_l0;
  l1_status.m_l1 = aoe_mm_tiling->m_l0 * aoe_mm_tiling->m_al1;
  l1_status.kal1_16 = aoe_mm_tiling->k_al1;
  l1_status.kbl1_16 = aoe_mm_tiling->k_bl1;
  l1_status.n_l1 = aoe_mm_tiling->n_l0 * aoe_mm_tiling->n_bl1;

  ub_status.n_cub = aoe_mm_tiling->n_cub;
  ub_status.m_aub = aoe_mm_tiling->m_aub;
  ub_status.k_aub = aoe_mm_tiling->k_aub;
  ub_status.k_bub = aoe_mm_tiling->k_bub;
  ub_status.n_bub = aoe_mm_tiling->n_bub;
  ub_status.batch_cub = aoe_mm_tiling->batch_cub;

  l1_status.db_al1 = aoe_mm_tiling->db_al1;
  l1_status.db_bl1 = aoe_mm_tiling->db_bl1;
  l0_status.db_l0a = aoe_mm_tiling->db_l0a;
  l0_status.db_l0b = aoe_mm_tiling->db_l0b;
  l0_status.db_l0c = aoe_mm_tiling->db_l0c;
  ub_status.db_aub = aoe_mm_tiling->db_aub;
  ub_status.db_bub = aoe_mm_tiling->db_bub;
  ub_status.db_cub = aoe_mm_tiling->db_cub;

  SetAoeMultiBatch(l0_status, ub_status);
  SetBufferParams(run_params, core_status, single_core_status);
  UpdateL1FullLoadFlag(op_name, run_params, core_status, single_core_status);
  UpdateUbTilingParam(params, l0_status, l1_status, core_status, ub_status);
  run_params.pattern_flag = true;

  OP_LOGD(op_name.c_str(), "[L0Status](m0:%ld, k0:%ld, n0:%ld, db_l0c:%ld, batch_l0:%ld), "
          "[L1Status](m_l1:%ld, k_al1:%ld, k_bl1:%ld, n_l1:%ld, db_al1:%ld, db_bl1:%ld), "
          "[UbStatus](m_aub:%ld, k_aub:%ld, k_bub:%ld, n_bub:%ld, n_cub:%ld, batch_cub:%ld), "
          "[CoreStatus](batch:%ld, m:%ld,k:%ld, n:%ld), [BlockDim](batch:%ld, m:%ld, k:%ld, n:%ld)",
          l0_status.m_l0, l0_status.k_l0, l0_status.n_l0, l0_status.db_l0c, l0_status.batch_l0,
          l1_status.m_al1 * l0_status.m_l0, l1_status.kal1_16, l1_status.kbl1_16, l1_status.n_bl1 * l0_status.n_l0,
          l1_status.db_al1, l1_status.db_bl1, ub_status.m_aub, ub_status.k_aub, ub_status.k_bub, ub_status.n_bub,
          ub_status.n_cub, ub_status.batch_cub, core_status.batch, core_status.m, core_status.k, core_status.n,
          core_status.batch_dim, core_status.m_dim, core_status.k_dim, core_status.n_dim);
}

const int64_t kBlockReduce = 16;
const int64_t kBlockReduceFp32 = 8;
const int64_t kBlockReduceS8 = 32;
const int64_t kBlockReduceS4 = 64;
const int64_t kBlockIn = 16;
const int64_t kBlockOut = 16;

const std::map<ge::DataType, int64_t> kBlockReduceMap = {
  {ge::DT_INT4, kBlockReduceS4},
  {ge::DT_INT8, kBlockReduceS8},
  {ge::DT_FLOAT16, kBlockReduce},
  {ge::DT_BF16, kBlockReduce},
  {ge::DT_FLOAT, kBlockReduceFp32}
};

int64_t GetBlockReduce(const ge::DataType &input_dtype) {
  int64_t block_size = kBlockReduce;
  auto iter = kBlockReduceMap.find(input_dtype);
  if (iter != kBlockReduceMap.end()) {
    block_size = iter->second;
  }
  return block_size;
}

void SetGemmInputArgsFromBatchmatmulParas(tuningtiling::GemmInputArgs &input_args, const BatchmatmulParas &params,
                                          uint64_t extern_params) {
  // set input_args from params instead of context_
  const BatchmatmulRunParas &run_params = *(params.run_params);
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  int64_t block_reduce = GetBlockReduce(static_cast<ge::DataType>(run_params.dtype_a));
  input_args.m = run_params.m * kBlockIn;
  input_args.k = run_params.k * block_reduce;
  input_args.n = run_params.n * kBlockOut;
  input_args.batch_a1 = run_params.batch_a1;
  input_args.batch_a2 = run_params.batch_a2;
  input_args.batch_a3 = run_params.batch_a3;
  input_args.batch_a4 = run_params.batch_a4;
  input_args.batch_b1 = run_params.batch_b1;
  input_args.batch_b2 = run_params.batch_b2;
  input_args.batch_b3 = run_params.batch_b3;
  input_args.batch_b4 = run_params.batch_b4;
  input_args.l1_fused_num = 0;
  input_args.aub_double_num = compile_params.aub_double_num;
  input_args.bub_double_num = compile_params.bub_double_num;
  input_args.fused_double_operand_num = compile_params.fused_double_operand_num;
  input_args.a_dtype = static_cast<ge::DataType>(run_params.dtype_a);
  input_args.b_dtype = static_cast<ge::DataType>(run_params.dtype_b);
  input_args.out_dtype = static_cast<ge::DataType>(run_params.dtype_out);
  input_args.a_format = run_params.format_a;
  input_args.b_format = run_params.format_b;
  input_args.bias_flag = run_params.bias_flag;
  input_args.out_format = run_params.format_out;
  input_args.trans_a_flag = run_params.trans_a_flag;
  input_args.trans_b_flag = run_params.trans_b_flag;
  input_args.reserved_bool = run_params.reserved_bool;
  input_args.m_align_flag = input_args.m == run_params.ori_shape_m;
  input_args.k_align_flag = input_args.k == run_params.ori_shape_k;
  input_args.n_align_flag = input_args.n == run_params.ori_shape_n;
  input_args.reserved_params1 = extern_params;
  input_args.reserved_params2 = 0;
  input_args.reserved_params3 = 0;
  input_args.reserved_params4 = 0;
  input_args.reserved_params5 = 0;
  input_args.reserved_params6 = 0;
}

void NonFactorMap(const string &op_type, BatchmatmulParas &params) {
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  BatchmatmulRunParas &run_params = *(params.run_params);
  run_params.batch_mapped = run_params.batch;
  run_params.m_mapped = run_params.m;
  run_params.k_mapped = run_params.k;
  run_params.n_mapped = run_params.n;
  // Split k will introduce atomic_add which can't be used with shift_inwards.
  // Thus in split k mode, batch/m/n/ can't use non-factorial segmentation.
  if (compile_params.split_k_flag) {
    // it is only necessary to consider the non-factor splitting of k when split_k_flag is true
    if (!CheckFactorNumSatisfy(run_params.k)) {
      // Non-factors of the k dimension use a down-aligned number of powers of 2
      run_params.k_mapped = MapShape(run_params.k, false);
    }
  } else {
    int64_t batch_factor_cnt = GetFactorCnt(run_params.batch, 1L, PlatformInfo::GetInstance().core_num);
    if (batch_factor_cnt <= kL0FactorNumLimit) {
      run_params.batch_mapped = MapShape(run_params.batch);
    }
    run_params.m_mapped = MapShape(run_params.m);
    run_params.n_mapped = MapShape(run_params.n);
  }
  OP_LOGD(op_type.c_str(),
          "NonFactorMap get mapped shape: batch_mapped: %ld, m_mapped: %ld, k_mapped: %ld, n_mapped: %ld.",
          run_params.batch_mapped, run_params.m_mapped, run_params.k_mapped, run_params.n_mapped);
}

void GetTilingFromDataBase(const string &op_type, BatchmatmulParas &params, CoreStatus &coreStatus,
                           SingleCoreStatus &singleCoreStatus) {
  coreStatus.cycle = INT64_MAX;
  NonFactorMap(op_type, params);
  int64_t blockValue = GetBlockDim(op_type, singleCoreStatus.ubStatus, params, coreStatus);
  GetL0Factors(op_type, coreStatus, blockValue, singleCoreStatus, params);
  GetL1Factors(op_type, params, coreStatus, singleCoreStatus.l0Status, singleCoreStatus.l1Status);
  UpdateL1FullLoadFlag(op_type, *params.run_params, coreStatus, singleCoreStatus);
  GetUbFactors(op_type, params, coreStatus, singleCoreStatus);
}

bool IsTilingValid(const string &op_type, const BatchmatmulRunParas &run_params) {
  OP_TILING_CHECK((run_params.batch > INT_MAX ||
                  run_params.batch <= 0 ||
                  run_params.ori_shape_m > INT_MAX ||
                  run_params.ori_shape_m <= 0 ||
                  run_params.ori_shape_k > INT_MAX ||
                  run_params.ori_shape_k <= 0 ||
                  run_params.ori_shape_n > INT_MAX ||
                  run_params.ori_shape_n <= 0),
                  CUBE_INNER_ERR_REPORT(op_type,
                                      "The batch, m,k,n of input ori_shape has exceeded INT_MAX"),
                  return false);
  return true;
}

tuningtiling::GemmTunnerTiling Trans2Tuning(const string &op_type, BatchmatmulRunParas &run_params,
                                           CoreStatus &coreStatus, SingleCoreStatus &singleCoreStatus){
  GetPadUb(op_type, run_params, coreStatus, singleCoreStatus.ubStatus);
  tuningtiling::GemmTunnerTiling tunner;
  L0Status &l0_status = singleCoreStatus.l0Status;
  L1Status &l1_status = singleCoreStatus.l1Status;
  UbStatus &ub_status = singleCoreStatus.ubStatus;
  tunner.batch_dim = coreStatus.batch_dim;
  tunner.n_dim = coreStatus.n_dim;
  tunner.m_dim = coreStatus.m_dim;
  tunner.k_dim = coreStatus.k_dim;
  tunner.m_al1 = l1_status.m_al1;
  tunner.k_al1 = l1_status.kal1_16;
  tunner.k_bl1 = l1_status.kbl1_16;
  tunner.n_bl1 = l1_status.n_bl1;
  tunner.batch_l0 = l0_status.batch_l0;
  tunner.batch_aub = ub_status.batch_aub;
  tunner.batch_bub = ub_status.batch_bub;
  tunner.batch_cub = ub_status.batch_cub;
  tunner.m_l0 = l0_status.m_l0;
  tunner.k_l0 = l0_status.k_l0;
  tunner.n_l0 = l0_status.n_l0;
  tunner.n_cub = ub_status.n_cub;
  tunner.m_aub = ub_status.m_aub;
  tunner.k_aub = ub_status.k_aub;
  tunner.k_bub = ub_status.k_bub;
  tunner.n_bub = ub_status.n_bub;
  tunner.db_al1 = l1_status.db_al1;
  tunner.db_bl1 = l1_status.db_bl1;
  tunner.db_l0a = l0_status.db_l0a;
  tunner.db_l0b = l0_status.db_l0b;
  tunner.db_l0c = l0_status.db_l0c;
  tunner.db_aub = ub_status.db_aub;
  tunner.db_bub = ub_status.db_bub;
  tunner.db_cub = ub_status.db_cub;
  OP_LOGD("TUNING_MatMulV2",
          "[L0Status](m0:%ld, k0:%ld, n0:%ld, db_l0c:%ld, batch_l0:%ld), "
          "[L1Status](m_l1:%ld, k_al1:%ld, k_bl1:%ld, n_l1:%ld, db_al1:%ld, db_bl1:%ld), "
          "[UbStatus](m_aub:%ld, k_aub:%ld, k_bub:%ld, n_bub:%ld, n_cub:%ld, batch_cub:%ld), "
          "[CoreStatus](batch:%ld, m:%ld,k:%ld, n:%ld), "
          "[BlockDim](batch:%ld, m:%ld, k:%ld, n:%ld)",
          l0_status.m_l0, l0_status.k_l0, l0_status.n_l0, l0_status.db_l0c, l0_status.batch_l0,
          l1_status.m_al1 * l0_status.m_l0, l1_status.kal1_16, l1_status.kbl1_16, l1_status.n_bl1 * l0_status.n_l0,
          l1_status.db_al1, l1_status.db_bl1, ub_status.m_aub, ub_status.k_aub, ub_status.k_bub, ub_status.n_bub,
          ub_status.n_cub, ub_status.batch_cub, coreStatus.batch, coreStatus.m, coreStatus.k, coreStatus.n,
          coreStatus.batch_dim, coreStatus.m_dim, coreStatus.k_dim, coreStatus.n_dim);

  return tunner;
}

inline bool CheckSupperCoreTuning(BatchmatmulRunParas &run_params) {
  if (run_params.dtype_a != static_cast<int32_t>(ge::DT_FLOAT16) ||
      run_params.dtype_out != static_cast<int32_t>(ge::DT_FLOAT16)) {
    return false;
  }
  if (((run_params.ori_shape_m + run_params.ori_shape_n) *
        run_params.ori_shape_k * inputDtypeBytes) <= PlatformInfo::GetInstance().l2_size) {
    return false;
  }
  if (run_params.pad_flag != 0 || run_params.nz_fusion_flag != 0) {
    return false;
  }
  return true;
}

inline bool CheckSupperCoreTuning(const tuningtiling::GemmTunnerTiling &tuning) {
  if (tuning.k_dim > 1) {
    return false;
  }
  if (tuning.batch_dim > 1) {
    return false;
  }
  return true;
}

void GenSupperCoreTuning(BatchmatmulRunParas &run_params, vector<tuningtiling::GemmTunnerTiling> &gemm_tiling_list) {
  if (!CheckSupperCoreTuning(run_params)) {
    return;
  }
  constexpr double core_threshold = 0.8;
  constexpr double single_threshold = 0.6;
  vector<tuningtiling::GemmTunnerTiling> super_tiling_list;
  int64_t cores = PlatformInfo::GetInstance().core_num;
  for (auto tuning : gemm_tiling_list) {
    if(!CheckSupperCoreTuning(tuning)) {
      continue;
    }
    int64_t m_cores = MathUtil::CeilDivision(run_params.m, tuning.m_al1 * tuning.m_l0);
    int64_t n_cores = MathUtil::CeilDivision(run_params.n, tuning.n_bl1 * tuning.n_l0);
    auto add_super_func = [&super_tiling_list, &tuning](int64_t m_dim, int64_t n_dim) {
        tuning.m_dim = m_dim;
        tuning.n_dim = n_dim;
        super_tiling_list.emplace_back(tuning);
    };
    for (int64_t m_dim = tuning.m_dim; m_dim <= m_cores; m_dim++) {
      for (int64_t n_dim = tuning.n_dim; n_dim <= n_cores; n_dim++) {
        if (m_dim * n_dim >= kSuperCoreLimit) { // 限制超核的最大核数
          break;
        }
        if (m_dim * n_dim <= cores || (m_dim == tuning.m_dim && n_dim == tuning.n_dim)) {
          continue;
        }
        int64_t m_single_last = m_cores % m_dim;
        int64_t n_single_last = n_cores % n_dim;
        if (m_single_last == 0 && n_single_last == 0) {
          add_super_func(m_dim, n_dim);
          continue;
        }

        if (((m_dim * n_dim % cores < core_threshold * cores) && (m_dim * n_dim % cores > 0)) ||
            ((m_single_last <= single_threshold * m_dim) && (m_single_last > 0)) ||
            ((n_single_last <= single_threshold * n_dim) && (n_single_last > 0))) {
          continue;
        }
        add_super_func(m_dim, n_dim);
      }
    }
  }
  gemm_tiling_list.insert(gemm_tiling_list.end(), super_tiling_list.begin(), super_tiling_list.end());
}

} // namespace


namespace optiling {
void DisablePatternCache() {
  pattern_cache_enable = false;
}

void EnablePatternCache() {
  pattern_cache_enable = true;
}

void GetTilingFromCache(uint32_t tiling_key, const cachetiling::MatmulHashInput &hash_input,
                        cachetiling::MatmulHashItem &hash_value) {
  tiling_hash_cache->Get(tiling_key, hash_input, hash_value);
}
void Tiling::SetParams(const CoreStatus &coreStatus, const L0Status &l0Status, const L1Status &l1Status,
                       const UbStatus &ubStatus, const BatchmatmulParas &params)
{
  batch_dim = coreStatus.batch_dim;
  n_dim = coreStatus.n_dim;
  m_dim = coreStatus.m_dim;
  k_dim = coreStatus.k_dim;
  m_l0 = l0Status.m_l0;
  k_l0 = l0Status.k_l0;
  n_l0 = l0Status.n_l0;
  kal1_16 = l1Status.kal1_16;
  kbl1_16 = l1Status.kbl1_16;
  // only pattern mode update single_core params which is calculated differently with main process.
  m_single_core = coreStatus.m_single_core;
  n_single_core = coreStatus.n_single_core;
  kal1_factor = coreStatus.kal1_factor;
  kbl1_factor = coreStatus.kbl1_factor;
  m_al1 = l1Status.m_al1;
  n_bl1 = l1Status.n_bl1;
  db_al1 = l1Status.db_al1;
  db_bl1 = l1Status.db_bl1;
  n_cub = ubStatus.n_cub;
  db_cub = ubStatus.db_cub;
  k_org_dim = kal1_factor * kal1_16 * reducedBlockSize;
  db_l0c = l0Status.db_l0c;
  al1_full_load = l1Status.al1_full_load;
  bl1_full_load = l1Status.bl1_full_load;
  batch_l0 = l0Status.batch_l0;
  batch_cub = ubStatus.batch_cub;
  l0c_multi_batch = l0Status.l0c_multi_batch;
  // 2: fp32 branch, 1: fp16 branch
  out_branch_flag = params.run_params->dtype_out == 0 ? 2 : 1;
  bias_flag = params.run_params->bias_flag;
  hf32_flag = params.run_params->hf32_flag;
  if (params.run_params->use_pre_ub) {
    k_aub = ubStatus.k_aub;
    m_aub = ubStatus.m_aub;
    db_aub = ubStatus.db_aub;
    k_bub = ubStatus.k_bub;
    n_bub = ubStatus.n_bub;
    db_bub = ubStatus.db_bub;
    batch_aub = ubStatus.batch_aub;
    batch_bub = ubStatus.batch_bub;
    aub_multi_flag = ubStatus.aub_multi_flag;
    bub_multi_flag = ubStatus.bub_multi_flag;
    a_align_value = ubStatus.a_align_value;
    b_align_value = ubStatus.b_align_value;
    aub_align_bound = ubStatus.aub_align_bound;
    bub_align_bound = ubStatus.bub_align_bound;
    flag_cub_solving_bank_conflict = ubStatus.flag_cub_solving_bank_conflict;
  } else if (params.run_params->pad_flag > 0) {
    k_aub = ubStatus.k_aub;
    m_aub = ubStatus.m_aub;
    k_bub = ubStatus.k_bub;
    n_bub = ubStatus.n_bub;
    aub_dim = coreStatus.aub_dim;
    bub_dim = coreStatus.bub_dim;
  } else if (params.run_params->nz_fusion_flag > 0) {
    k1_aub = ubStatus.k1_aub;
    m1_aub = ubStatus.m1_aub;
    k1_bub = ubStatus.k1_bub;
    n1_bub = ubStatus.n1_bub;
    m_aub_dim = coreStatus.m_aub_dim;
    n_bub_dim = coreStatus.n_bub_dim;
    k_aub_dim = coreStatus.k_aub_dim;
    k_bub_dim = coreStatus.k_bub_dim;
  }

  if (al1_full_load && coreStatus.batch == 1) {
    db_al1 = 1;
  }
  if (bl1_full_load && (!params.run_params->b_have_batch || coreStatus.batch == 1)) {
    db_bl1 = 1;
  }
  min_kl1_cmp_kl0 = (min(kal1_16, kbl1_16) == k_l0) ? 0 : 1;
  datatype_bf16 = params.run_params->dtype_a == static_cast<int32_t>(ge::DT_BF16);
}

void Tiling::SetZeroFlagTiling(BatchmatmulParas &params) {
  BatchmatmulRunParas &run_params = *(params.run_params);
  k_l0 = run_params.k;
  k_dim = 1;
  kal1_16 = k_l0;
  kbl1_16 = k_l0;
  kal1_factor = 1;
  kbl1_factor = 1;
  m_l0 = min(m_l0, kML0PreferSize);
  n_l0 = min(n_l0, kML0PreferSize);
  n_cub = n_l0;
  m_al1 = 1;
  n_bl1 = 1;
  k_aub = k_l0;
  k_bub = k_l0;
  m_aub = 1;
  n_bub = 1;
  batch_l0 = 1;
  batch_aub = 1;
  batch_bub = 1;
  batch_cub = 1;
  al1_attach_flag = kAttachFlagOne;
  bl1_attach_flag = kAttachFlagOne;
  abkl1_attach_flag = kAttachFlagZero;
  min_kl1_cmp_kl0 = 0;
  l0c_multi_batch = 0;
  bub_multi_flag = 0;
  aub_multi_flag = 0;
  al1_full_load = false;
  bl1_full_load = false;
  db_l0c = kDbOn;
  db_bl1 = kDbOn;
  db_al1 = kDbOn;
  m_single_core = 1;
  n_single_core = MathUtil::CeilDivision(MathUtil::CeilDivision(run_params.n, n_dim), n_bl1 * n_l0);
  m_single_core = MathUtil::CeilDivision(MathUtil::CeilDivision(run_params.m, m_dim), m_al1 * m_l0);
  run_params.non_factor_k = false;
}

void Tiling::SetWeightQuantBmmAttachFlag()
{
  bool kAl1FullLoad = kal1_16 * reducedBlockSize == k_org_dim;
  bool kBl1FullLoad = kbl1_16 * reducedBlockSize == k_org_dim;
  al1_attach_flag = kAl1FullLoad ? kAttachFlagOne : kAttachFlagZero;
  bl1_attach_flag = kBl1FullLoad ? kAttachFlagOne : kAttachFlagZero;
  abkl1_attach_flag = kal1_16 > kbl1_16 ? kAttachFlagZero : kAttachFlagOne;
}

void Tiling::SetAttachFlag()
{
  // find kernel ID
  bool kAl1FullLoad = kal1_16 * reducedBlockSize == k_org_dim;
  bool kBl1FullLoad = kbl1_16 * reducedBlockSize == k_org_dim;
  bool template1 = al1_full_load && bl1_full_load;
  bool template2 = al1_full_load && !bl1_full_load && kBl1FullLoad;
  bool template3 = al1_full_load && !bl1_full_load && !kBl1FullLoad;
  bool template4 = !al1_full_load && bl1_full_load && kAl1FullLoad;
  bool template5 = !al1_full_load && bl1_full_load && !kAl1FullLoad;
  bool template6 = !al1_full_load && !bl1_full_load && kAl1FullLoad && kBl1FullLoad;
  bool template7 = !al1_full_load && !bl1_full_load && kAl1FullLoad && !kBl1FullLoad;
  bool template8 = !al1_full_load && !bl1_full_load && !kAl1FullLoad && kBl1FullLoad;
  bool template9 = !al1_full_load && !bl1_full_load && !kAl1FullLoad && !kBl1FullLoad;
  bool condition1 = template1 || template2 || template3;
  bool condition2 = template4 || template6 || template7;
  bool condition3 = template5 || template8 || template9;
  bool condition4 = template1 || template4 || template5;
  bool condition5 = template2 || template6 || template8;
  bool condition6 = template3 || template7 || template9;
  bool condition7 = template1 || template2 || template4 || template6;
  bool condition8 = template3 || template7;
  bool condition9 = template5 || template8;

  if (condition1) {
    al1_attach_flag = kAttachFlagZero;
  }
  if (condition2) {
    al1_attach_flag = kAttachFlagOne;
  }
  if (condition3) {
    al1_attach_flag = kAttachFlagTwo;
  }
  if (condition4) {
    bl1_attach_flag = kAttachFlagZero;
  }
  if (condition5) {
    bl1_attach_flag = kAttachFlagOne;
  }
  if (condition6) {
    bl1_attach_flag = kAttachFlagTwo;
  }
  if (condition7) {
    abkl1_attach_flag = kAttachFlagZero;
  }
  if (condition8) {
    abkl1_attach_flag = kAttachFlagOne;
  }
  if (condition9) {
    abkl1_attach_flag = kAttachFlagTwo;
  }
  if (template9) {
    if (kal1_16 == kbl1_16) {
      abkl1_attach_flag = kAttachFlagZero;
    } else if (kal1_16 > kbl1_16) {
      abkl1_attach_flag = kAttachFlagOne;
    } else if (kal1_16 < kbl1_16) {
      abkl1_attach_flag = kAttachFlagTwo;
    }
  }
}

void Tiling::SetL2CacheFlag(BatchmatmulParas &params)
{
  BatchmatmulRunParas &runParams = *params.run_params;
  bool aL2Enable = false;
  bool bL2Enable = false;
  bool cL2Enable = false;
  bool biasL2Enable = false;
  bool aSingleCoreLoadRepeatly = (al1_attach_flag == kAttachFlagTwo && bl1_attach_flag != kAttachFlagZero);
  bool bSingleCoreLoadRepeatly = (bl1_attach_flag == kAttachFlagTwo && al1_attach_flag != kAttachFlagZero) ||
                                     (bl1_attach_flag == kAttachFlagOne && al1_attach_flag == kAttachFlagOne);
  aSingleCoreLoadRepeatly = (aSingleCoreLoadRepeatly && m_single_core > 1);
  bSingleCoreLoadRepeatly = (bSingleCoreLoadRepeatly && n_single_core > 1);
  if (k_dim > 1) {
    biasL2Enable = true;
    cL2Enable = true;
  } else {
    int64_t batch = runParams.batch;
    int64_t m = runParams.m;
    int64_t n = runParams.n;
    cL2Enable = (batch * m * n * outputDtypeBytes) <= PlatformInfo::GetInstance().l2_size;
  }

  OP_LOGI("matmul",
    "m_dim: %ld n_dim: %ld k_dim: %ld al1_attach_flag: %d bl1_attach_flag: %d",
    m_dim, n_dim, k_dim, al1_attach_flag, bl1_attach_flag);

  // 切K场景,AB只载入一次且载入数据大于x倍的L2,不需要经过L2
  // 只切MN场景,AB只载入一次不需要经过L2
  // ?

  // 判断一根轴还是两根
  bool aCoreLoadRepeatly = (m_dim > 1 && n_dim > 1);
  bool bCoreLoadRepeatly = (m_dim > 1 && n_dim > 1);
  aL2Enable = aSingleCoreLoadRepeatly || aCoreLoadRepeatly;
  bL2Enable = bSingleCoreLoadRepeatly || bCoreLoadRepeatly;

  if (aL2Enable && bL2Enable && cL2Enable && biasL2Enable) {
    l2_cache_flag |= (1 << kAllL2EnableBit);
    OP_LOGI("matmul", "l2_cache_flag: %d", l2_cache_flag);
    return;
  }

  if (!aL2Enable) {
    l2_cache_flag |= (1 << kAL2DisableBit);
  }

  if (!bL2Enable) {
    l2_cache_flag |= (1 << kBL2DisableBit);
  }

  if (!cL2Enable) {
    l2_cache_flag |= (1 << kCL2DisableBit);
  }

  if (!biasL2Enable) {
    l2_cache_flag |= (1 << kBiasL2DisableBit);
  }

  OP_LOGI("matmul", "l2_cache_flag: %d", l2_cache_flag);
}

bool Tiling::GetReorderFlag(const BatchmatmulParas &params) const
{
  const BatchmatmulCompileParas& compile = *params.compile_params;
  const BatchmatmulRunParas& run = *params.run_params;
  bool reorder_flag = false;
  int64_t a_size = run.m * run.k * kFractalSize;
  int64_t b_size = run.n * run.k * kFractalSize;
  int64_t c_size = run.m * run.n * kFractalSize;
  bool not_reorder = (!PlatformInfo::GetInstance().support_l0c2out()) ||
      run.is_batch_matmul_op ||
      (a_size + b_size < PlatformInfo::GetInstance().l2_size) ||
      c_size > kOutputSizeThreshold1 ||
      run.pad_flag > 0 || run.nz_fusion_flag > 0 ||
      run.dtype_a == static_cast<int32_t>(ge::DT_INT8) ||
      run.dtype_b == static_cast<int32_t>(ge::DT_INT8) ||
      run.dtype_out == static_cast<int32_t>(ge::DT_INT8) ||
      (run.performance_flag && run.dtype_a == static_cast<int32_t>(ge::DT_FLOAT)) ||
      compile.split_k_flag;
  OP_LOGD("matmul",
          "pad_flag: %d, nz_flag: %d, performance_flag: %d, split_k_flag: %d, not_reorder: %d.",
          run.pad_flag, run.nz_fusion_flag,
          run.performance_flag, compile.split_k_flag, not_reorder);
  if (not_reorder) {
    return reorder_flag;
  }
  if (a_size > kRightInputSizeThreshold && b_size <= kRightInputSizeThreshold) {
    reorder_flag = run.unaligned_flag;
  }
  if (a_size <= kLeftInputSizeThreshold && b_size > kLeftInputSizeThreshold &&
      c_size < kOutputSizeThreshold2) {
    reorder_flag = run.unaligned_flag ? false : true;
    reorder_flag = reorder_flag && run.format_out_nd;
  }
  reorder_flag = reorder_flag && al1_attach_flag == kAttachFlagTwo && bl1_attach_flag == kAttachFlagTwo;
  OP_LOGD("matmul",
          "a_size: %ld, b_size: %ld, unaligned_flag: %d, reorder_flag: %d.",
          a_size, b_size,
          run.unaligned_flag, reorder_flag);
  return reorder_flag;
}

void Tiling::GetTilingId(const BatchmatmulParas &params)
{
  const BatchmatmulCompileParas &compile_params = *(params.compile_params);
  const BatchmatmulRunParas &run_params = *(params.run_params);
  uint32_t nd2nz_type = 0;
  if (PlatformInfo::GetInstance().support_l0c2out() && run_params.nd_flag) {
    nd2nz_type = run_params.use_pre_ub ? kNd2NzMixL2 : kNd2NzOnTheFly;
  }
  if (run_params.weight_nz_flag) {
    nd2nz_type = kWeightNz;
  }
  bool is_quant = ((run_params.dtype_a == static_cast<int32_t>(ge::DT_INT8)) &&
                   PlatformInfo::GetInstance().support_l0c2out());
  uint32_t non_factor_bmn = static_cast<uint32_t>(!run_params.is_bmm_fixp &&
    ((run_params.unaligned_flag && (run_params.nd_flag || run_params.format_out_nd)) ||
    (!run_params.nd_flag && is_quant) ? 0 : 1));
  if (run_params.is_weight_quant_bmm) {
    uint32_t tilingIDLongLong = static_cast<uint32_t>(run_params.vector_pre_conv_mode);
    tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(bl1_attach_flag);
    tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(al1_attach_flag);
    tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(abkl1_attach_flag);
    tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(db_l0c - 1);
    tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(db_bl1 - 1);
    tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(db_al1 - 1);
    this->tiling_id = static_cast<uint64_t>(tilingIDLongLong);
    return;
  }

  uint32_t tilingIDLongLong = GetReorderFlag(params) ? 1 : 0;
  tilingIDLongLong = (tilingIDLongLong << kNumTwo) + static_cast<uint32_t>(run_params.nz_fusion_flag);
  tilingIDLongLong = (tilingIDLongLong << kNumTwo) + static_cast<uint32_t>(run_params.pad_flag);
  tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(run_params.vector_pre_conv_mode);
  tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(run_params.zero_flag);
  tilingIDLongLong = (tilingIDLongLong << kNumTwo) + nd2nz_type;
  tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(run_params.performance_flag);
  tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(compile_params.split_k_flag);
  tilingIDLongLong = (tilingIDLongLong << kNumFour) + static_cast<uint32_t>(l0c_multi_batch);
  // use non-factor-bmn as default, run_params.non_factor_bmn equal to 1
  tilingIDLongLong = (tilingIDLongLong << kNumOne) + non_factor_bmn;
  tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(run_params.non_factor_k);
  if (run_params.use_pre_ub) {
    tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(bub_multi_flag);
    tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(aub_multi_flag);
  }
  tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(min_kl1_cmp_kl0);
  tilingIDLongLong = (tilingIDLongLong << kNumTwo) + static_cast<uint32_t>(bl1_attach_flag);
  tilingIDLongLong = (tilingIDLongLong << kNumTwo) + static_cast<uint32_t>(al1_attach_flag);
  tilingIDLongLong = (tilingIDLongLong << kNumTwo) + static_cast<uint32_t>(abkl1_attach_flag);
  tilingIDLongLong = (tilingIDLongLong << kNumOne) + static_cast<uint32_t>(db_l0c - 1);
  this->tiling_id = static_cast<uint64_t>(tilingIDLongLong);
}

void SetBufferParams(const BatchmatmulRunParas &run_params, CoreStatus &coreStatus,
                     SingleCoreStatus &singleCoreStatus) {
  L1Status &l1Status = singleCoreStatus.l1Status;
  L0Status &l0Status = singleCoreStatus.l0Status;
  l0Status.db_l0a = kDbOn;
  l0Status.db_l0b = kDbOn;
  l1Status.m_al1 = MathUtil::CeilDivision(l1Status.m_l1, l0Status.m_l0);
  l1Status.n_bl1 = MathUtil::CeilDivision(l1Status.n_l1, l0Status.n_l0);
  coreStatus.m_single_core = MathUtil::CeilDivision(MathUtil::CeilDivision(run_params.m, coreStatus.m_dim),
                                                    l1Status.m_l1);
  coreStatus.n_single_core = MathUtil::CeilDivision(MathUtil::CeilDivision(run_params.n, coreStatus.n_dim),
                                                    l1Status.n_l1);
  coreStatus.kal1_factor = MathUtil::CeilDivision(run_params.k, l1Status.kal1_16);
  coreStatus.kbl1_factor = MathUtil::CeilDivision(run_params.k, l1Status.kbl1_16);
  coreStatus.m = min(coreStatus.m_single_core * l1Status.m_l1, run_params.m);
  coreStatus.k = run_params.k;
  coreStatus.n = min(coreStatus.n_single_core * l1Status.n_l1, run_params.n);
  coreStatus.batch = MathUtil::CeilDivision(run_params.batch, coreStatus.batch_dim); // This is batch_single_core
  if (PlatformInfo::GetInstance().support_l0c2out() && coreStatus.k_dim != 1) {
    int64_t k_64 = MathUtil::CeilDivision(run_params.k, coreStatus.k_dim);
    int64_t k_l1 = max(l1Status.kal1_16, l1Status.kbl1_16);
    int64_t k_single_core = MathUtil::CeilDivision(k_64, k_l1);
    coreStatus.kal1_factor = k_single_core * (k_l1 / l1Status.kal1_16);
    coreStatus.kbl1_factor = k_single_core * (k_l1 / l1Status.kbl1_16);
    coreStatus.k = min(k_single_core * k_l1, run_params.k);
  }
  coreStatus.batch_dim = MathUtil::CeilDivision(run_params.batch, coreStatus.batch);
  coreStatus.n_dim = MathUtil::CeilDivision(run_params.n, coreStatus.n);
  coreStatus.m_dim = MathUtil::CeilDivision(run_params.m, coreStatus.m);
  coreStatus.k_dim = MathUtil::CeilDivision(run_params.k, coreStatus.k);
  return;
}

bool CheckExpandRatio(const BatchmatmulRunParas &run_params, const CoreStatus &coreStatus,
                      const SingleCoreStatus &singleCoreStatus) {
  if (run_params.is_weight_quant_bmm) {
    return true;
  }
  int64_t al1_load_size = singleCoreStatus.l1Status.m_l1 * coreStatus.m_single_core * coreStatus.m_dim;
  int64_t bl1_load_size = singleCoreStatus.l1Status.n_l1 * coreStatus.n_single_core * coreStatus.n_dim;
  double a_expand_ratio = static_cast<double>(al1_load_size) / run_params.m;
  double b_expand_ratio = static_cast<double>(bl1_load_size) / run_params.n;
  // Filter params which expand ratio over 1.35 for align
  double expand_ratio = 1.35;
  if (run_params.unaligned_flag) {
    // 3 for unalign
    expand_ratio = 3;
  }
  if (a_expand_ratio < expand_ratio && b_expand_ratio < expand_ratio) {
    return true;
  }
  return false;
}

void UpdateL1LoadFlag(CoreStatus &coreStatus, SingleCoreStatus &singleCoreStatus) {
  L1Status& l1Status = singleCoreStatus.l1Status;
  // initialize the full load flag
  l1Status.both_full_load = false;
  l1Status.al1_full_load = false;
  l1Status.bl1_full_load = false;
  l1Status.al1_k_full_load = false;
  l1Status.bl1_k_full_load = false;
  if (coreStatus.m_single_core == 1 && coreStatus.kal1_factor == 1) {
    l1Status.al1_full_load = true;
  } else if (coreStatus.kal1_factor == 1) {
    l1Status.al1_k_full_load = true;
  }
  if (coreStatus.n_single_core == 1 && coreStatus.kbl1_factor == 1) {
    l1Status.bl1_full_load = true;
  } else if (coreStatus.kbl1_factor == 1) {
    l1Status.bl1_k_full_load = true;
  }
  // Update the full_load flag in l1Status to ensure they are correct.
  if (l1Status.al1_full_load && l1Status.bl1_full_load) {
    l1Status.both_full_load = true;
  }
}

void SetDoubleBuffer(const BatchmatmulRunParas &run_params, SingleCoreStatus &singleCoreStatus) {
  L0Status &l0Status = singleCoreStatus.l0Status;
  L1Status &l1Status = singleCoreStatus.l1Status;
  int64_t cur_bias_bt_size = 0;
  int64_t channel_wise_l1_size = GetChannelWiseL1Size(l1Status, l1Status.n_bl1 * l0Status.n_l0);
  if (PlatformInfo::GetInstance().support_l0c2out() && run_params.bias_flag) {
    cur_bias_bt_size = GetBiasBtSize(l0Status.n_l0);
  }
  l0Status.db_l0c = kDbOff;
  int64_t l0c_factor_limit = PlatformInfo::GetInstance().support_l0c2out() ? k1971L0cFactorMax : k1980L0cFactorLimit;
  l0c_factor_limit = PlatformInfo::GetInstance().support_l12bt_bf16() ? k1982L0cFactorMax : l0c_factor_limit;
  if (l0Status.batch_l0 * (l0Status.m_l0  + static_cast<int64_t>(run_params.is_weight_quant_bmm) * l0Status.k_l0) *
      l0Status.n_l0 * kDbOn <= l0c_factor_limit && (cur_bias_bt_size * kDbOn <= PlatformInfo::GetInstance().bt_size)) {
    l0Status.db_l0c = kDbOn;
  }
  int64_t kal1_bound = 0;
  int64_t kbl1_bound = 0;
  GetABKL1Bound(run_params, l1Status, kal1_bound, kbl1_bound);
  int64_t cur_al1_size = l1Status.m_al1 * l0Status.m_l0 * kal1_bound;
  int64_t cur_bl1_size = l1Status.n_bl1 * l0Status.n_l0 * kbl1_bound;
  cur_bl1_size = cur_bl1_size + static_cast<int64_t>(run_params.is_weight_quant_bmm) * cur_bl1_size / kFp16Bytes;
  l1Status.db_al1 = kDbOff;
  l1Status.db_bl1 = kDbOff;
  if (l1Status.al1_full_load || l1Status.bl1_full_load) {
    SetDbFullLoad(l1Status, cur_al1_size, cur_bl1_size, channel_wise_l1_size, run_params);
  } else {
    SetDbNotFullLoad(l1Status, cur_al1_size, cur_bl1_size, channel_wise_l1_size);
  }
}

bool ExpandShape(const BatchmatmulRunParas &run_params, L1Status &l1Status, int64_t dim, bool al1_flag) {
  if (dim == 1) {
    return true;
  }
  int64_t ori_shape = al1_flag ? run_params.m : run_params.n;
  int64_t l1_shape = al1_flag ? l1Status.m_l1 : l1Status.n_l1;
  int64_t l1_max_shape = MathUtil::CeilDivision(ori_shape, dim - 1) - 1;
  l1_max_shape = max(l1_max_shape, l1_shape);
  int64_t full_cache_line = PlatformInfo::GetInstance().support_l0c2out() ? fullCacheLine : k1980FullCacheLine;
  int64_t l1_align_shape = MathUtil::Align(l1_shape, full_cache_line);
  if (l1_align_shape <= l1_max_shape) {
    if (al1_flag) {
      l1Status.m_l1 = l1_align_shape;
    } else {
      l1Status.n_l1 = l1_align_shape;
    }
    return true;
  }
  return false;
}

bool GenTiling(const string &op_type, const BatchmatmulCompileParas &compile_params, BatchmatmulRunParas &run_params,
               Tiling &tiling, gert::TilingContext *context) {
  if (!IsTilingValid(op_type, run_params)) {
    return false;
  }
  InitilizationProcess(op_type, run_params);
  // check tiling in cache
  cachetiling::MatmulHashInput hash_input(compile_params, run_params);
  cachetiling::MatmulHashItem hash_value(tiling, run_params, hash_input);
  uint32_t tiling_key = cachetiling::MurmurHash(&hash_input, sizeof(hash_input));
  OP_LOGD(op_type, "hash_key %u, rt_bank %u, zero_flag: %d, hash_input %s", tiling_key, compile_params.enable_rt_bank_cache,
      run_params.zero_flag, hash_input.ToString().c_str());
  bool enable_hash = !compile_params.enable_rt_bank_cache && !run_params.zero_flag;
  if (tiling_hash_cache->Get(tiling_key, hash_input, hash_value) && enable_hash) {
    tiling = hash_value.tiling();
    tiling.datatype_bf16 = run_params.dtype_a == static_cast<int32_t>(ge::DT_BF16);
    run_params = hash_value.run_params(); // if use run_params.bf16, need reupdate
    OP_LOGI(op_type, "the tiling id from cache tiling is: %lu, hash_key %u", tiling.tiling_id, tiling_key);
    return true;
  }
  kUbFp16Size = PlatformInfo::GetInstance().ub_size / kFp16Bytes;
  kUbFp32Size = PlatformInfo::GetInstance().ub_size / kFp32Bytes;
  if (run_params.unaligned_flag && run_params.format_out_nd && (!run_params.pad_flag && !run_params.nz_fusion_flag)) {
    // remove reg_buf used size, reg_buf is 16, if open double buffer need 2 *16
    kUbFp16Size -= 2 * kBlockSize;
  }
  // calculate tiling
  CoreStatus coreStatus;
  SingleCoreStatus singleCoreStatus;
  BatchmatmulParas params;
  params.compile_params = &compile_params;
  params.run_params = &run_params;

  if (compile_params.split_k_flag) {
    GetSplitKdim(op_type, run_params, coreStatus);
  }
  UpdateSingleCoreStatus(params, singleCoreStatus);

  GenTilingStatus status = GenTilingFromBasicBlock(op_type, params, coreStatus, singleCoreStatus);
  if (status == GEN_TILING_EOF) {
    SetBufferParams(run_params, coreStatus, singleCoreStatus);
    UpdateL1LoadFlag(coreStatus, singleCoreStatus);
    SetDoubleBuffer(run_params, singleCoreStatus);
  } else {
    TilingPatternProcess(op_type, params, coreStatus, singleCoreStatus);
    // calculate tiling use load_size
    if (!run_params.pattern_flag) {
      GetTilingFromDataBase(op_type, params, coreStatus, singleCoreStatus);
    }
  }
  GetTransdataUb(run_params, coreStatus, singleCoreStatus);
  GetPadUb(op_type, run_params, coreStatus, singleCoreStatus.ubStatus);
  tiling.SetParams(coreStatus, singleCoreStatus.l0Status, singleCoreStatus.l1Status, singleCoreStatus.ubStatus, params);
  if (run_params.is_weight_quant_bmm) {
    tiling.SetWeightQuantBmmAttachFlag();
  }
  else {
    tiling.SetAttachFlag();
  }
  tiling.SetL2CacheFlag(params);

  if (run_params.zero_flag) {
    tiling.SetZeroFlagTiling(params);
  }
  tiling.GetTilingId(params);
  if (enable_hash) {
    hash_value.set_tiling(tiling);
    hash_value.set_run_param(run_params);
    // add tiling to cache
    tiling_hash_cache->Add(tiling_key, hash_input, hash_value);
  }
  OP_LOGI(op_type.c_str(), "the tiling id from cache tiling is: %lu, hash_key %u", tiling.tiling_id, tiling_key);
  return true;
}

void GenTuning(const string &op_type, BatchmatmulCompileParas &compile_params, BatchmatmulRunParas &run_params,
               vector<tuningtiling::GemmTunnerTiling> &gemm_tiling_list) {
  if (!IsTilingValid(op_type, run_params)) {
    return;
  }
  InitilizationProcess(op_type, run_params);
  kUbFp16Size = PlatformInfo::GetInstance().ub_size / kFp16Bytes;
  kUbFp32Size = PlatformInfo::GetInstance().ub_size / kFp32Bytes;
  if (run_params.unaligned_flag && run_params.format_out_nd && run_params.pad_flag == 0) {
    // remove reg_buf used size, reg_buf is 16, if open double buffer need 2 *16
    kUbFp16Size -= 2 * kBlockSize;
  }
  // calculate tiling
  CoreStatus coreStatus;
  SingleCoreStatus singleCoreStatus;
  BatchmatmulParas params;
  params.compile_params = &compile_params;
  params.run_params = &run_params;
  UpdateSingleCoreStatus(params, singleCoreStatus);
  {
    list<CoreStatus> csList {coreStatus};
    list<SingleCoreStatus> scsList {singleCoreStatus};
    GenTuningFromBasicBlock(op_type, params, csList, scsList);
    auto csIter = csList.begin();
    auto scsIter = scsList.begin();
    for (; csIter != csList.end() && scsIter != scsList.end(); ++csIter, ++scsIter)
    {
      CoreStatus tmpCoreStatus = *csIter;
      SingleCoreStatus tmpSingleCoreStatus = *scsIter;
      SetBufferParams(run_params, tmpCoreStatus, tmpSingleCoreStatus);
      UpdateL1LoadFlag(tmpCoreStatus, tmpSingleCoreStatus);
      SetDoubleBuffer(run_params, tmpSingleCoreStatus);
      tuningtiling::GemmTunnerTiling tuner = Trans2Tuning(op_type, run_params, tmpCoreStatus, tmpSingleCoreStatus);
      gemm_tiling_list.emplace_back(tuner);
    }
  }
  {
    auto quickFunc = [&gemm_tiling_list, op_type, &params, &run_params](
        CoreStatus tmpCoreStatus, SingleCoreStatus tmpSingleCoreStatus) {
      run_params.pattern_flag = false;
      TilingPatternProcess(op_type, params, tmpCoreStatus, tmpSingleCoreStatus);
      if(run_params.pattern_flag) {
        tuningtiling::GemmTunnerTiling tuner = Trans2Tuning(op_type, run_params, tmpCoreStatus, tmpSingleCoreStatus);
        gemm_tiling_list.emplace_back(tuner);
      }
    };
    auto dataBaseFuc = [&gemm_tiling_list, op_type, &params, &run_params](
        CoreStatus tmpCoreStatus, SingleCoreStatus tmpSingleCoreStatus) {
      GetTilingFromDataBase(op_type, params, tmpCoreStatus, tmpSingleCoreStatus);
      tuningtiling::GemmTunnerTiling tuner = Trans2Tuning(op_type, run_params, tmpCoreStatus, tmpSingleCoreStatus);
      gemm_tiling_list.emplace_back(tuner);
    };
    if (run_params.dtype_a != static_cast<int32_t>(ge::DT_FLOAT) &&
        run_params.dtype_out == static_cast<int32_t>(ge::DT_FLOAT) &&
        !run_params.bias_flag) {
      compile_params.split_k_flag = true;
      GetSplitKdim(op_type, run_params, coreStatus);
    }
    quickFunc(coreStatus, singleCoreStatus);
    dataBaseFuc(coreStatus, singleCoreStatus);
  }
  GenSupperCoreTuning(run_params, gemm_tiling_list);
  OP_LOGI(op_type.c_str(), "the cache tiling tuning gemm_tiling_list size %zu", gemm_tiling_list.size());
}

void PlatformInfo::SetInstance(const CubeCompileInfo &compile_info) {
  core_num = compile_info.core_num;
  l1_size = compile_info.l1_size;
  l2_size = compile_info.l2_size;
  l0a_size = compile_info.l0a_size;
  l0b_size = compile_info.l0b_size;
  l0c_size = compile_info.l0c_size;
  ub_size = compile_info.ub_size;
  bt_size = compile_info.bt_size;
  load3d_constraints = compile_info.load3d_constraints;
  intrinsic_data_move_l12ub = compile_info.intrinsic_data_move_l12ub;
  intrinsic_data_move_l0c2ub = compile_info.intrinsic_data_move_l0c2ub;
  intrinsic_fix_pipe_l0c2out = compile_info.intrinsic_fix_pipe_l0c2out;
  intrinsic_fix_pipe_l0c2ub = compile_info.intrinsic_fix_pipe_l0c2ub;
  intrinsic_data_move_out2l1_nd2nz = compile_info.intrinsic_data_move_out2l1_nd2nz;
  intrinsic_matmul_ub_to_ub = compile_info.intrinsic_matmul_ub_to_ub;
  intrinsic_conv_ub_to_ub = compile_info.intrinsic_conv_ub_to_ub;
  intrinsic_data_move_l12bt_bf16 = compile_info.intrinsic_data_move_l12bt_bf16;
  std::lock_guard<std::mutex> lock(str_soc_mutex);
  soc_version = compile_info.soc_version;
  cube_freq = compile_info.cube_freq;
  OP_LOGD("NO_OP_NAME", "PLATFORM INFO in runtime2.0: %s", ToString().c_str());
}

std::string PlatformInfo::GetSocVersion() {
  std::lock_guard<std::mutex> lock(str_soc_mutex);
  return soc_version;
}

std::string PlatformInfo::ToString() const {
  std::stringstream ss;
  ss << " load3d_constraints: " << load3d_constraints
     << " support_l0c2out: " << intrinsic_fix_pipe_l0c2out
     << " support_fix_pipe_l0c2ub: " << intrinsic_fix_pipe_l0c2ub
     << " support_data_move_l12ub: " << intrinsic_data_move_l12ub
     << " support_data_move_l0c2ub: " << intrinsic_data_move_l0c2ub
     << " support_data_move_out2l1_nd2nz: " << intrinsic_data_move_out2l1_nd2nz
     << " support_matmul_ub_to_ub: " << intrinsic_matmul_ub_to_ub
     << " core_num: " << core_num
     << " l2_size: " << l2_size
     << " l1_size: " << l1_size
     << " l0a_size: " << l0a_size
     << " l0b_size: " << l0b_size
     << " l0c_size: " << l0c_size
     << " ub_size: " << ub_size
     << " bt_size: " << bt_size;
  return ss.str();
}
} // namespace optiling
