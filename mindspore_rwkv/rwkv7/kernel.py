import numpy as np
import mindspore as ms
from mindspore import Tensor, Parameter, nn
from mindspore.ops import CustomOpBuilder


class WKVKernelCustom(nn.Cell):
    def __init__(self):
        super().__init__()
        # self.p = Parameter(2.0, requires_grad=False)
        self.my_ops = CustomOpBuilder("my_ops", ['./function_wkv7.cpp'], backend="Ascend").load()

    def construct(self, k, v, w, r, a, b, hi):
        o, ho = self.my_ops.wkv7(k, v, w, r, a, b, hi)
        return o, ho
