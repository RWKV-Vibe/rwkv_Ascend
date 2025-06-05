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
 * \file cache_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_CACHE_TILING_H
#define OPS_BUILT_IN_OP_TILING_CACHE_TILING_H

#include <unistd.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <ratio>
#include <vector>

#include "cube_tiling_runtime.h"
#include "gemm_ub_cache_tiling.h"
#include "conv2dtranspose_ub_cache_tiling.h"
#include "aoe/op_tuning_tiling/gemm_tuning_tiling.h"
#include "op_tiling_util.h"
using json = nlohmann::json;

namespace optiling {
const int64_t kDbOn = 2;
const int64_t kDbOff = 1;

enum Optype {
  NULL_TYPE,
  MAT_MUL,
  BATCH_MATMUL,
  TRANSPOSE_BATCH_MATMUL,
  QUANT_BATCH_MATMUL,
  COMPRESS_QUANT_BATCH_MATMUL,
  BATCH_MATMUL_FIXPIPE,
  WEIGHT_QUANT_BATCH_MATMUL
};

struct BatchmatmulCompileParas {
  bool binary_mode_flag = false;
  bool bias_flag = false;
  bool at_l1_flag = true;
  bool split_k_flag = false;
  bool pattern_flag = false;
  bool zero_flag = false;
  bool sparse_4to2_flag = false;
  bool binary_constant_flag = false;
  bool vector_pre_conv_mode = false;
  float fused_double_operand_num = 0;
  float aub_double_num = 0;
  float bub_double_num = 0;
  int64_t quant_scale = 0;
  int64_t eltwise_src = 0;
  int8_t enable_pad = 0;
  bool enable_nz_fusion = false;
  bool enable_rt_bank_cache = false;
};

struct BatchmatmulRunParas {
  bool nd_flag = false;
  bool use_pre_ub = false;
  bool trans_a_flag = false;
  bool trans_b_flag = false;
  bool format_a_nd = false;
  bool format_b_nd = false;
  bool format_out_nd = false;
  Format format_a = ge::FORMAT_ND;
  Format format_b = ge::FORMAT_ND;
  Format format_out = ge::FORMAT_ND;
  bool reserved_bool = false;
  bool b_have_batch = false;  // dim num > 2
  bool is_batch_matmul_mode = false;  // dynamic_mode == "dynamic_mknb"
  bool is_batch_matmul_op = false;  // BatchMatMulV2 or BatchMatMul
  bool used_aligned_pattern = false;
  bool non_factor_k = false;
  bool non_factor_bmn = false;
  bool bias_flag = false;
  bool pattern_flag = false;
  bool do_not_multi_batch = false;
  bool performance_flag = false;
  bool unaligned_flag = false;
  bool zero_flag = false;
  bool is_compress_quant = false;
  bool is_bmm_fixp = false;
  bool enable_nz_fusion = false;
  bool weight_nz_flag = false;
  int8_t enable_pad = 0;
  int8_t hf32_flag = 1;
  int8_t pad_flag = 0;
  int8_t nz_fusion_flag = 0;
  int32_t dtype_a = 0;
  int32_t dtype_b = 0;
  int32_t dtype_out = 0;
  int32_t dtype_bias = 0;
  int64_t m_mapped = 1;
  int64_t k_mapped = 1;
  int64_t n_mapped = 1;
  int64_t batch_mapped = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t batch = 1;
  int64_t ori_shape_m = 1;
  int64_t ori_shape_k = 1;
  int64_t ori_shape_n = 1;
  int64_t m_pad = 0;
  int64_t k_pad = 0;
  int64_t n_pad = 0;
  int64_t nl0 = 1;
  int64_t kl0 = 1;
  int64_t dim0_a = 0;
  int64_t dim1_a = 0;
  int64_t dim2_a = 0;
  int64_t dim0_b = 0;
  int64_t dim1_b = 0;
  int64_t dim2_b = 0;
  int64_t batch_a1 = 1;
  int64_t batch_a2 = 1;
  int64_t batch_a3 = 1;
  int64_t batch_a4 = 1;
  int64_t batch_b1 = 1;
  int64_t batch_b2 = 1;
  int64_t batch_b3 = 1;
  int64_t batch_b4 = 1;
  int64_t batch_c1 = 1;
  int64_t batch_c2 = 1;
  int64_t batch_c3 = 1;
  int64_t batch_c4 = 1;
  int32_t offset_x = 0;
  int32_t index_size = 0;
  bool m_quant_check = false;
  bool n_quant_check = false;
  bool is_weight_quant_bmm = false;
  bool vector_pre_conv_mode = false;
  bool is_quant_batch_matmul_v3 = false;
  bool is_weight_quant_batch_matmul_v2 = false;
  bool is_pertoken = false;
  // 3 is perm_a dim
  std::array<size_t, 3> perm_a = {0, 0, 0};
  // 3 is perm_b dim
  std::array<size_t, 3> perm_b = {0, 0, 0};
  ge::DataType bias_dtype = ge::DT_FLOAT16;
};


struct BatchmatmulParas {
  const BatchmatmulCompileParas *compile_params = nullptr;
  BatchmatmulRunParas *run_params = nullptr;
};

struct PatternParams {
  BatchmatmulCompileParas compile_params;
  BatchmatmulRunParas run_params;
};

struct CoreStatus {
  int64_t batch = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t batch_dim = 1;
  int64_t m_dim = 1;
  int64_t n_dim = 1;
  int64_t k_dim = 1;
  int64_t aub_dim = 1;
  int64_t bub_dim = 1;
  int64_t m_aub_dim = 1;
  int64_t n_bub_dim = 1;
  int64_t k_aub_dim = 1;
  int64_t k_bub_dim = 1;
  int64_t kal1_factor = 1;
  int64_t kbl1_factor = 1;
  int64_t m_single_core = 1;
  int64_t n_single_core = 1;
  int64_t cycle = INT64_MAX;
};

struct BlockDimCalculator {
  int64_t batch = 1;
  int64_t m = 1;
  int64_t k = 1;
  int64_t n = 1;
  int64_t k_num = 1;
  int64_t k_bytes = 1;
  int64_t n_dim_factor = 1;
  int64_t batch_dim_factor = 1;
  int64_t m_dim_factor = 1;
  int64_t k_dim_factor = 1;
  int64_t min_load_size = 1;
  int64_t core_use = 1;
  int64_t tmp_core_use = 1;
  int64_t batch_idx = 0;
  int64_t n_idx = 0;
  int64_t batch_dim_cnt = 0;
  int64_t m_dim_cnt = 0;
  int64_t n_dim_cnt = 0;
  int64_t k_dim_cnt = 0;
  int64_t ori_amat_size = 0;
  int64_t ori_bmat_size = 0;
  int64_t amat_size = 0;
  int64_t bmat_size = 0;
  int64_t tmp_amat_size = 0;
  int64_t tmp_bmat_size = 0;
  int64_t tmp_load_size = 0;
  int64_t total_load_size = 0;
  int64_t* batch_dim_array;
  int64_t* m_dim_array;
  int64_t* n_dim_array;
  int64_t* k_dim_array;
  int64_t tmp_value = 0;
  int64_t final_value = 0;
  bool init_flag = false;
};

struct L0Status {
  int64_t m_l0 = 1;
  int64_t n_l0 = 1;
  int64_t k_l0 = 1;
  int64_t batch_l0 = 1;
  int32_t l0c_multi_batch = 0;
  int64_t db_l0a = 1;
  int64_t db_l0b = 1;
  int64_t db_l0c = 1;
  int64_t db_cub = 1;
  int64_t final_ml0 = 0;
  int64_t final_kl0 = 0;
  int64_t final_nl0 = 0;
  int64_t final_load_size = INT64_MAX;
  float final_l0c_use = 0;
  int64_t final_mul = 0;
  int64_t final_mte1Loop = INT64_MAX;
  int64_t final_mte1_cycles = 0;
  int64_t max_mk = 1;
  int64_t max_nk = 1;
  int64_t max_mn = 1;
  int64_t max_axis_idx = 0;
  int64_t max_axis_num = 0;
  int64_t max_axis_pnt = 1;
  int64_t max_n = 1;
  int32_t dtype_bias = 0;
  bool update_using_mte1 = false;
  void SetInitLoadStatus()
  {
    final_ml0 = 0;
    final_kl0 = 0;
    final_nl0 = 0;
    final_load_size = INT64_MAX;
    final_l0c_use = 0;
    final_mul = 0;
    final_mte1Loop = INT64_MAX;
    final_mte1_cycles = 0;
    update_using_mte1 = false;
  }
};

struct L0Factors {
  int64_t final_ml0 = 0;
  int64_t final_kl0 = 0;
  int64_t final_nl0 = 0;
  int64_t final_load_size = INT64_MAX;
  float final_l0c_use = 0;
  int64_t final_mul = 0;
  int64_t final_mte1Loop = INT64_MAX;
  int64_t final_mte1_cycles = 0;
};

struct MKNParasCombo {
  int64_t parasCombo[10];
};

struct L1Status {
  int64_t kal1_16 = 1;
  int64_t kbl1_16 = 1;
  int64_t m_l1;
  int64_t n_l1;
  int64_t m_al1 = 1;
  int64_t n_bl1 = 1;
  int64_t db_al1 = 1;
  int64_t db_bl1 = 1;
  int64_t al1_size = 0;
  int64_t bl1_size = 0;
  int64_t al1_times = 1;
  int64_t bl1_times = 1;
  int64_t all_times = 1;
  int64_t load_size = 0;
  int64_t max_m_al1 = 1;
  int64_t max_n_bl1 = 1;
  int64_t max_k_al1 = 1;
  int64_t max_k_bl1 = 1;
  bool both_full_load = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  bool al1_k_full_load = false;
  bool bl1_k_full_load = false;
  int64_t channel_wise_times = 0;
  int64_t element_wise_size = 0;
  void SetStatus(const int64_t tmp_l1_factors[6])  // the size of tmp_l1_factors must be 6
  {
    this->kal1_16 = tmp_l1_factors[0];
    this->kbl1_16 = tmp_l1_factors[1];
    this->m_al1 = tmp_l1_factors[2]; // 2 means m_al1 factor index
    this->n_bl1 = tmp_l1_factors[3]; // 3 means n_bl1 factor index
    this->db_al1 = tmp_l1_factors[4]; // 4 means db_al1 factor index
    this->db_bl1 = tmp_l1_factors[5]; // 5 means db_bl1 factor index
  }
};

struct AUbStatusCondition {
  /* This struct is used to storage the tiling condition of AUb.
  condition_m2_k2 : all data in m_ori and k_ori are loaded to Ub
  condition_ml1_kl1 : all data of m_al1 and k_al1 are loaded to Ub
  condition_ml1_kl0: all data of m_al1 and the data of k_l0 are loaded to Ub.
  condition_ml1_k0: all data of m_al1 and partial data of k_l0 data are loaded to Ub.
  condition_ml0_kl1 : all data of k_al1 and the data of ml0 are loaded to Ub.
  condition_m0_kl1 : all data of k_al1 and the partial data of ml0 are loaded to Ub.
  condition_ml0_kl0 : the data of k_l0 and ml0 are loaded to Ub.
  condition_ml0_k0 : the data of ml0 in m dimension and one block of data in k dimension are loaded to Ub.
  condition_m0_kl0 : the data of k_l0 in m dimension and one block of data in m dimension are loaded to Ub.
  condition_m0_k0 : one block of data in m dimension and one block of data in k dimension are loaded to Ub.
  */
  bool condition_m2_k2 = false;
  bool condition_ml1_kl1 = false;
  bool condition_ml1_kl0 = false;
  bool condition_ml1_k0 = false;
  bool condition_ml0_kl1 = false;
  bool condition_m0_kl1 = false;
  bool condition_ml0_kl0 = false;
  bool condition_ml0_k0 = false;
  bool condition_m0_kl0 = false;
  bool condition_m0_k0 = false;
};

struct BUbStatusCondition {
  /* This struct is used to storage the tiling condition of BUb.
  condition_k2_n2 : all data in k_ori and n_ori are loaded to Ub
  condition_kl1_nl1 : all data of k_al1 and n_bl1 are loaded to Ub
  condition_kl0_nl1: all data of n_bl1 and the data of kl0 are loaded to Ub.
  condition_k0_nl1: all data of n_bl1 and partial data of kl0 are loaded to Ub.
  condition_kl1_nl0 : all data of k_bl1 and the data of nl0 are loaded to Ub.
  condition_kl1_n0 : all data of k_bl1 and the partial data of nl0 are loaded to Ub.
  condition_kl0_nl0 : the data of kl0 and nl0 are loaded to Ub.
  condition_k0_nl0 : the data of nl0 and one block of data in k dimension are loaded to Ub.
  condition_kl0_n0 : the data of kl0 in k dimension and one block of data in n dimension are loaded to Ub.
  condition_k0_n0 : one block of data in n dimension and one block of data in k dimension are loaded to Ub.
  */
  bool condition_k2_n2 = false;
  bool condition_kl1_nl1 = false;
  bool condition_kl0_nl1 = false;
  bool condition_k0_nl1 = false;
  bool condition_kl1_nl0 = false;
  bool condition_kl1_n0 = false;
  bool condition_kl0_nl0 = false;
  bool condition_k0_nl0 = false;
  bool condition_kl0_n0 = false;
  bool condition_k0_n0 = false;
};

class PreUbTiling
{
public:
  int64_t k_aub = 1;
  int64_t m_aub = 1;
  int64_t k_bub = 1;
  int64_t n_bub = 1;
  PreUbTiling() = default;
  void update_tiling(int64_t new_k_aub, int64_t new_m_aub, int64_t new_k_bub, int64_t new_n_bub)
  {
    k_aub = new_k_aub;
    m_aub = new_m_aub;
    k_bub = new_k_bub;
    n_bub = new_n_bub;
  }
  ~PreUbTiling() = default;
};

struct UbStatus {
  int64_t k_aub = 1;
  int64_t m_aub = 1;
  // UB double Buffer default On
  int64_t db_aub = kDbOn;
  int64_t batch_aub = 1;
  int64_t k_bub = 1;
  int64_t n_bub = 1;
  int64_t db_bub = kDbOn;
  int64_t batch_bub = 1;
  int64_t n_cub = 1;
  int64_t db_cub = kDbOn;
  int64_t batch_cub = 1;
  int64_t k1_aub = 1;
  int64_t m1_aub = 1;
  int64_t k1_bub = 1;
  int64_t n1_bub = 1;
  int64_t max_dma_size = 0;
  int64_t min_dma_size = 0;
  int64_t min_load_size = 0;
  int64_t aub_size = 0;
  int64_t bub_size = 0;
  int64_t cub_size = 0;
  int64_t aub_multi_flag = 0;
  int64_t bub_multi_flag = 0;
  int64_t a_align_value = 1;
  int64_t b_align_value = 1;
  int64_t aub_bank_size = 0;
  int64_t bub_bank_size = 0;
  int64_t aub_align_bound = 0;
  int64_t bub_align_bound = 0;
  int64_t aub_aligned_tensor_size = 0;
  int64_t bub_aligned_tensor_size = 0;
  int64_t aub_single_tensor_size = 0;
  int64_t bub_single_tensor_size = 0;
  int64_t ub_rest_size = 0;
  int64_t safe_ub_rest_size = 0;
  int32_t aub_cnt = 0;
  int32_t bub_cnt = 0;
  int64_t cub_dtype_multi = 1;
  bool flag_pre_ub_not_reused = false;
  bool cub_reuse_aub_flag = false;
  bool cub_reuse_bub_flag = false;
  bool a_bank_conflict = false;
  bool b_bank_conflict = false;
  bool flag_cub_solving_bank_conflict = false;
  // format_out_nd and n_ori is unalign, tail_block of n_cub must > 16;
  bool n_cub_tail_block_limit = false;
  // cub fusion params
  float fused_double_operand_num = 0.0;
};

struct SingleCoreStatus {
  L0Status l0Status;
  L1Status l1Status;
  UbStatus ubStatus;
};

enum L1TilingType {
  KAL1_16,
  KBL1_16,
  M_AL1,
  N_BL1
};

class PlatformInfo {
 public:
  int64_t core_num = 32;
  int64_t l1_size = (1024 * 1024);
  int64_t l2_size = (32 * 1024 * 1024);
  int64_t l0a_size = (64 * 1024);
  int64_t l0b_size = (64 * 1024);
  int64_t l0c_size = (256 * 1024);
  int64_t ub_size = (256 * 1024);
  int64_t bt_size = 1024;
  bool load3d_constraints = true;
  bool intrinsic_data_move_l12ub = true;
  bool intrinsic_data_move_l0c2ub = true;
  bool intrinsic_fix_pipe_l0c2out = false;
  bool intrinsic_fix_pipe_l0c2ub = false;
  bool intrinsic_data_move_out2l1_nd2nz = false;
  bool intrinsic_matmul_ub_to_ub = false;
  bool intrinsic_data_move_l12bt_bf16 = false;
  bool intrinsic_conv_ub_to_ub = false;
  int64_t cube_freq = 1000; // KHz
  std::string GetSocVersion();
  bool support_l0c2out() { return intrinsic_fix_pipe_l0c2out; }
  bool support_fix_pipe_l0c2ub() { return intrinsic_fix_pipe_l0c2ub; }
  bool support_l12bt_bf16() const { return intrinsic_data_move_l12bt_bf16; }
  static PlatformInfo &GetInstance() {
    static PlatformInfo instance;
    return instance;
  }
  bool SetInstance(const json &compile_info);
  void SetInstance(const CubeCompileInfo &compile_info);

 private:
  // disable new and destory outside
  PlatformInfo() = default;
  ~PlatformInfo() = default;
  // disable copy/move and operator '=' outside
  PlatformInfo(const PlatformInfo &) = delete;
  PlatformInfo(PlatformInfo &&) = delete;
  PlatformInfo &operator=(const PlatformInfo &) = delete;
  PlatformInfo &operator=(PlatformInfo &&) = delete;
  std::string ToString() const;

  std::string soc_version = "";
};

class Tiling {
public:
  uint64_t tiling_id;
  int64_t n_cub = 1;
  int64_t db_cub = 1;
  int64_t m_l0 = 1;
  int64_t k_l0 = 1;
  int64_t n_l0 = 1;
  int64_t batch_dim = 1;
  int64_t n_dim = 1;
  int64_t m_dim = 1;
  int64_t k_dim = 1;
  int64_t kal1_16 = 1;
  int64_t kbl1_16 = 1;
  int64_t kal1_factor = 1;
  int64_t kbl1_factor = 1;
  int64_t m_al1 = 1;
  int64_t n_bl1 = 1;
  int64_t db_al1 = 1;
  int64_t db_bl1 = 1;
  int64_t k_aub = 1;
  int64_t m_aub = 1;
  int64_t db_aub = 1;
  int64_t k_bub = 1;
  int64_t n_bub = 1;
  int64_t db_bub = 1;
  int64_t aub_dim = 1;
  int64_t bub_dim = 1;
  int64_t m1_aub = 1;
  int64_t n1_bub = 1;
  int64_t k1_aub = 1;
  int64_t k1_bub = 1;
  int64_t m_aub_dim = 1;
  int64_t n_bub_dim = 1;
  int64_t k_aub_dim = 1;
  int64_t k_bub_dim = 1;
  int64_t k_org_dim = 1;
  int64_t db_l0c = 1;
  int64_t batch_l0 = 1;
  int64_t batch_aub = 1;
  int64_t batch_bub = 1;
  int64_t batch_cub = 1;
  int32_t out_branch_flag = 1;
  int32_t bias_flag = 0;
  int32_t aub_multi_flag = 0;
  int32_t bub_multi_flag = 0;
  int64_t a_align_value = 1;
  int64_t b_align_value = 1;
  int64_t aub_align_bound = 0;
  int64_t bub_align_bound = 0;
  int64_t min_kl1_cmp_kl0 = 0;
  int32_t al1_attach_flag = 0;
  int32_t bl1_attach_flag = 0;
  int32_t abkl1_attach_flag = 0;
  int32_t l0c_multi_batch = 0;
  int64_t m_single_core = 1;
  int64_t n_single_core = 1;
  bool flag_cub_solving_bank_conflict = false;
  bool al1_full_load = false;
  bool bl1_full_load = false;
  int8_t hf32_flag = 1;
  int32_t zero_flag = 0;
  bool datatype_bf16 = false;
  uint64_t deq_scale_var = 0x3F800000;
  uint32_t l2_cache_flag = 0;
  Tiling() = default;
  void SetParams(const CoreStatus& coreStatus, const L0Status& l0Status, const L1Status& l1Status,
                 const UbStatus& ubStatus, const BatchmatmulParas& params);
  void SetWeightQuantBmmAttachFlag();
  void SetAttachFlag();
  void SetL2CacheFlag(BatchmatmulParas& params);
  void SetZeroFlagTiling(BatchmatmulParas& params);
  void GetTilingId(const BatchmatmulParas& params);
  bool GetReorderFlag(const BatchmatmulParas &params) const;
  ~Tiling() = default;
};

bool GenTiling(const std::string &op_type, const BatchmatmulCompileParas &compile_params,
               BatchmatmulRunParas &run_params, Tiling &tiling, gert::TilingContext *context);

void GenTuning(const std::string &op_type,
                     BatchmatmulCompileParas &compile_params,
                     BatchmatmulRunParas &run_params,
                     vector<tuningtiling::GemmTunnerTiling> &gemm_tiling_list);

void SetBufferParams(const BatchmatmulRunParas &run_params, CoreStatus &coreStatus,
                     SingleCoreStatus &singleCoreStatus);

bool CheckExpandRatio(const BatchmatmulRunParas &run_params, const CoreStatus &coreStatus,
                      const SingleCoreStatus &singleCoreStatus);

bool ExpandShape(const BatchmatmulRunParas &run_params, L1Status &l1Status, int64_t dim, bool al1_flag);

void UpdateL1LoadFlag(CoreStatus &coreStatus, SingleCoreStatus &singleCoreStatus);

void SetDoubleBuffer(const BatchmatmulRunParas &run_params, SingleCoreStatus &singleCoreStatus);
}; // namespace optiling

namespace gert {
enum DynamicMode {
  DYNAMIC_MKN,
  DYNAMIC_MKNB,
  WEIGHT_QUANT_BMM
};

class GemmCompileInfo : public optiling::CubeCompileInfo {
 public:
  GemmCompileInfo() = default;
  ~GemmCompileInfo() override = default;

  // bool AnalyzeExtendInfo(const nlohmann::json &compile_info) override;

  bool trans_a = false;
  bool trans_b = false;
  bool repo_seed_flag = false;
  bool repo_costmodel_flag = false;
  uint32_t workspace_num = 0;
  uint32_t ub_size = 0;
  optiling::BatchmatmulCompileParas params;
  optiling::Ub2UbBatchmatmulCompileParas ub2ub_params;
  DynamicMode dynamic_mode = DYNAMIC_MKN;
};
}

#endif
