/**
 * @file registration.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <torch/extension.h>
#include "function.h"

// 在myops命名空间里注册add_custom和add_custom_backward两个schema，新增自定义aten ir需要在此注册
// 在 wkv7.cpp 中添加以下代码（或合并到 registration.cpp）
TORCH_LIBRARY(myops, m) {
    // 声明 wkv7 的输入输出类型
    m.def("wkv7(Tensor k, Tensor v, Tensor w, Tensor r, Tensor a, Tensor b, Tensor hi) -> (Tensor, Tensor)");
    // // 声明反向传播的 Schema
    // m.def("wkv7_backward(Tensor grad_out, Tensor grad_ht, Tensor k, Tensor v, Tensor w, Tensor r, Tensor a, Tensor b, Tensor hi) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}


// 通过pybind将c++接口和python接口绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv7", &wkv7_autograd, "");
}
