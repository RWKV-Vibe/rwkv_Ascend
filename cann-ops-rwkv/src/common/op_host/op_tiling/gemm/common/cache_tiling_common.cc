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
 * \file cache_tiling_common.cpp\
 * \brief function of cache tiling common
 */
#include "cache_tiling_common.h"

namespace gemm_cache_tiling {

ostream& operator<<(ostream& out, const GemmEstCoreStatus& a) {
  out<<a.batch <<" "<< a.m <<" "<<a.k<<" "<< a.n <<" "<<a.kal1_factor<<" "<< a.kbl1_factor <<" "<<a.m_single_core<<
  " "<< a.n_single_core <<" "<<a.both_full_load<<" "<< a.al1_full_load <<" "<<a.bl1_full_load<<" "<< a.al1_k_full_load<<
  " "<<a.bl1_k_full_load;
  return out;
}

ostream& operator<<(ostream& out, const GemmResultInfo& a) {
  out<<a.batch_dim <<" "<< a.m_dim <<" "<<a.n_dim<<" "<< a.k_dim <<" "<<a.m_l0<<" "<< a.n_l0 <<" "<<a.k_l0<<
  " "<< a.batch_l0 <<" "<<a.batch_l0<<" "<< a.db_l0c <<" "<<a.kal1_16<<" "<< a.kbl1_16 <<" "<<a.m_l1<<" "<<a.n_l1<<" "<<
  a.db_al1<<" "<<a.db_bl1;
  return out;
}
}
