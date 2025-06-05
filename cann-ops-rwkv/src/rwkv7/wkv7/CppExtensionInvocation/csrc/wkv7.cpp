#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

// 前向NPU实现
std::tuple<at::Tensor, at::Tensor> wkv7_npu_forward(
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& w,
    const at::Tensor& r,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& hi) {
    
    at::Tensor out = at::empty_like(k);
    at::Tensor ht = at::empty_like(hi);
    
    EXEC_NPU_CMD(aclnnwkv7, k, v, w, r, a, b, hi, out, ht);
    return {out, ht};
}

// 注册前向算子（不需要Autograd实现）
TORCH_LIBRARY(myops, m) {
    m.def("wkv7(Tensor k, Tensor v, Tensor w, Tensor r, Tensor a, Tensor b, Tensor hi) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("wkv7", wkv7_npu_forward);
}
