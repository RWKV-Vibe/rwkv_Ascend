/**
 * @file wkv7.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using variable_list = std::vector<at::Tensor>;

// 为NPU设备注册前向实现
std::tuple<at::Tensor, at::Tensor> wkv7_impl_npu(const at::Tensor& k, const at::Tensor& v, const at::Tensor& w, const at::Tensor& r, const at::Tensor& a, const at::Tensor& b, const at::Tensor& hi) {
    // 创建输出内存
    at::Tensor out = at::empty_like(k);
    at::Tensor ht = at::empty_like(hi);

    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnwkv7, k, v, w, r, a, b, hi, out, ht);
    return std::make_tuple(out, ht);
}

// 为NPU设备注册反向实现
std::tuple<at::Tensor, at::Tensor> wkv7_backward_impl_npu(const at::Tensor& grad) {
    at::Tensor result = grad; // 创建输出内存

    return {result, result};
}

// 为Meta设备注册前向实现
at::Tensor wkv7_impl_meta(const at::Tensor& self, const at::Tensor& other) {
    return at::empty_like(self);
}

// 为Meta设备注册反向实现
std::tuple<at::Tensor, at::Tensor> wkv7_backward_impl_meta(const at::Tensor& self) {
    auto result = at::empty_like(self);
    return std::make_tuple(result, result);
}

// 通过继承torch::autograd::Function类实现前反向绑定
class WKV7Function : public torch::autograd::Function<WKV7Function> {
    public:
        static at::Tensor forward(AutogradContext *ctx, const at::Tensor& k, const at::Tensor& v, const at::Tensor& w, const at::Tensor& r, const at::Tensor& a, const at::Tensor& b, const at::Tensor& hi) {
            at::AutoDispatchBelowADInplaceOrView guard;
            static auto op = torch::Dispatcher::singleton()
                            .findSchemaOrThrow("myops::wkv7", "")
                            .typed<decltype(wkv7_impl_npu)>();

            auto out, ht = op.call(k, v, w, r, a, b, hi); 
            return std::make_tuple(out, ht);
        }

        static variable_list backward(AutogradContext *ctx, variable_list grad_outputs) {
            auto grad_output = grad_outputs[0];

            static auto op = torch::Dispatcher::singleton()
                            .findSchemaOrThrow("myops::wkv7_backward", "")
                            .typed<decltype(wkv7_backward_impl_npu)>();
            auto result = op.call(grad_output);
            return {std::get<0>(result), std::get<1>(result)};
        }
};

// 使用的时候调用apply()方法
std::tuple<at::Tensor, at::Tensor> wkv7_autograd(const at::Tensor& k, const at::Tensor& v, const at::Tensor& w, const at::Tensor& r, const at::Tensor& a, const at::Tensor& b, const at::Tensor& hi) {
     return WKV7Function::apply(k, v, w, r, a, b, hi);  // 解包 apply 结果
}

// 为NPU设备注册前反向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("wkv7", &wkv7_impl_npu);
    // m.impl("wkv7_backward", &wkv7_backward_impl_npu);
}

// 给op绑定NPU的自动求导实现
// 如果是pytorch 2.1以下的版本，AutogradPrivateUse1需要改成AutogradXLA
TORCH_LIBRARY_IMPL(myops, AutogradPrivateUse1, m) {
    m.impl("wkv7", &wkv7_autograd);
}

// 为Meta设备注册前反向实现
TORCH_LIBRARY_IMPL(myops, Meta, m) {
    m.impl("wkv7", &wkv7_impl_meta);
    // m.impl("wkv7_backward", &wkv7_backward_impl_meta);
}
