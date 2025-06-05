/**
 * @file function.h
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef FUNCTION_H_
#define FUNCTION_H_

#include <ATen/ATen.h>

std::tuple<at::Tensor, at::Tensor> wkv7_npu_forward(const at::Tensor& k, const at::Tensor& v, const at::Tensor& w, const at::Tensor& r, const at::Tensor& a, const at::Tensor& b, const at::Tensor& hi);

#endif //  FUNCTION_H_
