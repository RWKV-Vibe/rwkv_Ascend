import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops
import copy

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

def naive_recurrent_rwkv7(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    S: torch.Tensor,
):
    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    # S = torch.torch.zeros(B, H, K, V, dtype=torch.float, device=q.device)
    
    for t in range(T):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t]
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                w_t = torch.exp(w[bi, hi, t])

                sa = (a_t[None, :] * S[bi, hi]).sum(dim=1)

                S[bi, hi] = (S[bi, hi] * w_t[None, :] +  # [N,V] * [N,1]
                                 k_t[None, :] * v_t[:, None] +     # [N,1] * [1,V]
                                 sa[:, None] * b_t[None, :])                # [V] * [N,1]

                y = (S[bi, hi] * q_t[None, :]).sum(dim=1)

                o[bi, hi, t] = y
    return o, S

class TestCustomAdd(TestCase):
    def test_add_custom(self):
        r_cpu = torch.rand([1, 16, 64, 64], dtype=torch.float32) * 2 - 1
        w_cpu = torch.rand([1, 16, 64, 64], dtype=torch.float32) * 2 - 1
        k_cpu = torch.rand([1, 16, 64, 64], dtype=torch.float32) * 2 - 1
        v_cpu = torch.rand([1, 16, 64, 64], dtype=torch.float32) * 2 - 1
        a_cpu = torch.rand([1, 16, 64, 64], dtype=torch.float32) * 2 - 1
        b_cpu = torch.rand([1, 16, 64, 64], dtype=torch.float32) * 2 - 1
        
        h_cpu = torch.rand([1, 16, 64, 64], dtype=torch.float32) * 2 - 1
        k_npu, v_npu, w_npu, r_npu, a_npu, b_npu = copy.deepcopy(k_cpu).npu(), copy.deepcopy(v_cpu).npu(), copy.deepcopy(w_cpu).npu(), copy.deepcopy(r_cpu).npu(), copy.deepcopy(a_cpu).npu(), copy.deepcopy(b_cpu).npu()
        h_npu = copy.deepcopy(h_cpu).npu()
        

        # calculate on npu
        output, state = custom_ops.wkv7(k_npu, v_npu, w_npu, r_npu, a_npu, b_npu, h_npu)
        # output.backward(output)
        print("output:", output)
        print("state", state)
        o, h = naive_recurrent_rwkv7(k_npu, v_npu, w_npu, r_npu, a_npu, b_npu, h_npu)
        print("o:", o)
        print("h", h)
        # calculate on cpu
        # cpuout = torch.add(x_cpu, y_cpu)
        # cpuout.backward(cpuout)

        # # compare result
        self.assertRtolEqual(output, o)
        # self.assertRtolEqual(x_npu.grad, x_cpu.grad)
        # self.assertRtolEqual(y_npu.grad, y_cpu.grad)


    # def test_add_custom_meta(self):
    #     input1 = torch.randn([8, 2048], dtype=torch.float32)
    #     input2 = torch.randn([8, 2048], dtype=torch.float32)

    #     x_input1 = input1.to("meta")
    #     y_input1 = input2.to("meta")
    #     x_input1.requires_grad = True
    #     y_input1.requires_grad = True
    #     custom_out = custom_ops.add_custom(x_input1, y_input1)
    #     custom_out.backward(custom_out)

    #     x_input2 = input1.to("meta")
    #     y_input2 = input2.to("meta")
    #     x_input2.requires_grad = True
    #     y_input2.requires_grad = True
    #     cpuout = torch.add(x_input2, y_input2)
    #     cpuout.backward(cpuout)

    #     self.assertTrue(custom_out.is_meta)
    #     self.assertRtolEqual(custom_out.size(), cpuout.size())
    #     self.assertRtolEqual(x_input1.grad.size(), x_input2.grad.size())
    #     self.assertRtolEqual(y_input1.grad.size(), y_input2.grad.size())


if __name__ == "__main__":
    run_tests()

