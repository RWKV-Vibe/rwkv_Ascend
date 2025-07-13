import torch
import torch.nn as nn
import torch.nn.functional as F


class Wkv7(nn.Module):
    def __init__(self, num_heads, head_size, custom_wkv=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.custom_wkv = custom_wkv

        # 移除所有aimet自定义模块实例化，使用PyTorch原生操作替代

        if custom_wkv:
            # 保留自定义wkv实现
            from rwkv_src.wkv_custom import wkv_c_impl_src
            module = torch.utils.cpp_extension.load_inline(
                    name='extension', cpp_sources=[wkv_c_impl_src])
            self.wkv_func = torch.ops.rwkv.wkv7

    def forward(self, seq_length, r, w, k, v, a, b, state2):
        if self.custom_wkv:
            # 使用PyTorch reshape替代aimet Reshape
            r = r.reshape(seq_length, self.num_heads, self.head_size)
            w = w.reshape(seq_length, self.num_heads, self.head_size)
            k = k.reshape(seq_length, self.num_heads, self.head_size)
            v = v.reshape(seq_length, self.num_heads, self.head_size)
            a = a.reshape(seq_length, self.num_heads, self.head_size)
            b = b.reshape(seq_length, self.num_heads, self.head_size)

            x, state2_out = self.wkv_func(r, w, k, v, a, b, state2)
            x = x.view(seq_length, self.num_heads, 1, self.head_size)
        else:
            if seq_length == 1:
                r = r.view(self.num_heads, self.head_size, 1)
                v = v.view(self.num_heads, self.head_size, 1)
                k = k.view(self.num_heads, 1, self.head_size)
                w = w.view(self.num_heads, 1, self.head_size)
                b = b.view(self.num_heads, 1, self.head_size)
                a = a.view(self.num_heads, self.head_size, 1)

                # 用PyTorch matmul替代aimet MatMul
                kv = torch.matmul(v, k)
                # 用运算符替代aimet Add和Multiply
                state2_out = (state2 * w) + (torch.matmul(torch.matmul(state2, a), b)) + kv
                x = torch.matmul(state2_out, r).view(seq_length, self.num_heads, 1, self.head_size)
            else:
                r = r.view(seq_length, self.num_heads, self.head_size, 1)
                v = v.view(seq_length, self.num_heads, self.head_size, 1)
                k = k.view(seq_length, self.num_heads, 1, self.head_size)
                w = w.view(seq_length, self.num_heads, 1, self.head_size)
                b = b.view(seq_length, self.num_heads, 1, self.head_size)
                a = a.view(seq_length, self.num_heads, self.head_size, 1)
                kv = torch.matmul(v, k)
                x = torch.zeros(seq_length, self.num_heads, self.head_size, 1, device=k.device, dtype=kv.dtype)
                for i in range(seq_length):
                    # 替换aimet的Add、Multiply和MatMul
                    state2 = (state2 * w[i, :, :, :]) + (torch.matmul(torch.matmul(state2, a[i, :, :, :]), b[i, :, :, :])) + kv[i, :, :, :]
                    x[i, :, :, :] = torch.matmul(state2, r[i, :, :, :])
                state2_out = state2
                x = x.view(seq_length, self.num_heads, 1, self.head_size)

        return x, state2_out


class Rwkv7SelfAttention(nn.Module):
    def __init__(self, state_dict, hidden_size, head_size, layer_id=0, rescale_layer=0, custom_wkv=False):
        super().__init__()
        prefix = f'blocks.{layer_id}.att.'
        self.layer_id = layer_id
        self.num_heads = hidden_size // head_size
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.custom_wkv = custom_wkv

        self.D_DECAY_LORA = state_dict[prefix + 'w1'].shape[-1]
        self.D_AAA_LORA = state_dict[prefix + 'a1'].shape[-1]
        self.D_GATE_LORA = state_dict[prefix + 'g1'].shape[-1]

        self.x_r = nn.Parameter(state_dict[prefix + 'x_r'])
        self.x_w = nn.Parameter(state_dict[prefix + 'x_w'])
        self.x_k = nn.Parameter(state_dict[prefix + 'x_k'])
        self.x_v = nn.Parameter(state_dict[prefix + 'x_v'])
        self.x_a = nn.Parameter(state_dict[prefix + 'x_a'])
        self.x_g = nn.Parameter(state_dict[prefix + 'x_g'])

        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.receptance.weight = nn.Parameter(state_dict[prefix + 'receptance.weight'])
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'])
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.matmul_time_decay_w1   = nn.Linear(hidden_size, self.D_DECAY_LORA, bias=False)
        self.matmul_time_decay_w1.weight = nn.Parameter(state_dict[prefix + 'w1'].t())
        self.matmul_time_decay_w2   = nn.Linear(self.D_DECAY_LORA, hidden_size)
        self.matmul_time_decay_w2.weight = nn.Parameter(state_dict[prefix + 'w2'].t())
        self.matmul_time_decay_w2.bias = nn.Parameter(state_dict[prefix + 'w0'].view(-1))

        self.matmul_a1 = nn.Linear(hidden_size, self.D_AAA_LORA, bias=False)
        self.matmul_a1.weight = nn.Parameter(state_dict[prefix + 'a1'].t())
        self.matmul_a2 = nn.Linear(self.D_AAA_LORA, hidden_size)
        self.matmul_a2.weight = nn.Parameter(state_dict[prefix + 'a2'].t())
        self.matmul_a2.bias = nn.Parameter(state_dict[prefix + 'a0'].view(-1))

        if layer_id != 0:
            self.D_MV_LORA = state_dict[prefix + 'v1'].shape[-1]
            self.matmul_v1 = nn.Linear(hidden_size, self.D_MV_LORA, bias=False)
            self.matmul_v1.weight = nn.Parameter(state_dict[prefix + 'v1'].t())
            self.matmul_v2 = nn.Linear(self.D_MV_LORA, hidden_size)
            self.matmul_v2.weight = nn.Parameter(state_dict[prefix + 'v2'].t())
            self.matmul_v2.bias = nn.Parameter(state_dict[prefix + 'v0'].view(-1))

        self.matmul_g1 = nn.Linear(hidden_size, self.D_GATE_LORA, bias=False)
        self.matmul_g1.weight = nn.Parameter(state_dict[prefix + 'g1'].t())
        self.matmul_g2 = nn.Linear(self.D_GATE_LORA, hidden_size, bias=False)
        self.matmul_g2.weight = nn.Parameter(state_dict[prefix + 'g2'].t())

        self.k_k = nn.Parameter(state_dict[prefix + 'k_k'].view(self.num_heads, self.head_size))
        self.k_a = nn.Parameter(state_dict[prefix + 'k_a'].view(self.num_heads, self.head_size))
        self.r_k = nn.Parameter(state_dict[prefix + 'r_k'].view(self.num_heads, self.head_size))

        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output.weight = nn.Parameter(state_dict[prefix + 'output.weight'])
        self.ln_x = nn.LayerNorm(self.head_size, eps=64e-5)
        self.ln_x_w = nn.Parameter(state_dict[prefix + 'ln_x.weight'])
        self.ln_x_b = nn.Parameter(state_dict[prefix + 'ln_x.bias'])

        self.ln_1                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_1.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.weight'])
        self.ln_1.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln1.bias'])

        self.tanh_w                 = nn.Tanh()
        self.exp_w                  = torch.exp  # 替换aimet Exponential
        self.sigmoid_a              = nn.Sigmoid()
        self.sigmoid_g              = nn.Sigmoid()
        self.sigmoid_v              = nn.Sigmoid()
        self.sigmoid_w              = nn.Sigmoid()


        self.wkv7 = Wkv7(self.num_heads, self.head_size, custom_wkv=self.custom_wkv)
    
    def forward(self, x, state1, state2, v_first):
        last_x = x
        x = self.ln_1(x)
        batch_size, seq_length, _ = x.size()
        assert batch_size == 1
        
        if seq_length == 1:
            state1_out = x
            # 替换aimet Subtract为减法运算符
            sx = state1 - x
        else:
            # 替换aimet CustomGather为索引操作（取最后一个元素）
            state1_out = x[:, -1:]
            # 取0到seq_length-2的元素
            past = x[:, :-1]
            # 替换aimet Concat和Reshape
            past = torch.cat([state1.reshape(1, 1, -1), past], dim=1)
            sx = past - x

        # 替换aimet的lerp_mul和lerp_add为乘法和加法运算符
        xr = x + (sx * self.x_r)
        xw = x + (sx * self.x_w)
        xk = x + (sx * self.x_k)
        xv = x + (sx * self.x_v)
        xa = x + (sx * self.x_a)
        xg = x + (sx * self.x_g)

        receptance = self.receptance(xr).view(seq_length, self.num_heads, self.head_size)
        key = self.key(xk).view(seq_length, self.num_heads, self.head_size)
        value = self.value(xv).view(seq_length, self.num_heads, self.head_size)
        gate = self.matmul_g2(self.sigmoid_g(self.matmul_g1(xg)))
        a = self.sigmoid_a(self.matmul_a2(self.matmul_a1(xa))).view(seq_length, self.num_heads, self.head_size)
        
        time_decay = self.matmul_time_decay_w2(self.tanh_w(self.matmul_time_decay_w1(xw)))
        # 替换aimet Multiply为乘法运算符
        time_decay = self.exp_w(-0.606531 * self.sigmoid_w(time_decay))

        # 替换aimet Multiply为乘法
        kk = key * self.k_k
        kk = torch.nn.functional.normalize(kk.view(seq_length, self.num_heads, self.head_size), dim=-1, p=2.0).view(-1)
        # 替换aimet的混合操作（乘法和加减法）
        key = key * (1 + ( (a - 1) * self.k_a ))

        if self.layer_id == 0:
            v_first = value
        else:
            # 替换aimet的加减乘操作
            value = value + ( (v_first - value) * self.sigmoid_v(self.matmul_v2(self.matmul_v1(xv)).view(seq_length, self.num_heads, self.head_size)) )

        # 替换aimet Neg（取负）和Multiply（乘法）
        b = kk * a
        a = -kk
        
        x, state2_out = self.wkv7(seq_length, receptance, time_decay, key, value, a, b, state2)

        # 替换aimet的LayerNorm相关操作
        x = self.ln_x(x).view(batch_size, seq_length, self.hidden_size)
        x = x * self.ln_x_w
        x = x + self.ln_x_b

        # 替换aimet Sum和乘法操作
        rkv = (torch.sum( (receptance * key) * self.r_k, dim=-1, keepdim=True ) * value).view(seq_length, self.hidden_size)
        x = x + rkv
        x = x * gate
        x = self.output(x)

        if self.layer_id == 0:
            return (last_x + x), state1_out, state2_out, v_first
        else:
            return (last_x + x), state1_out, state2_out

class Rwkv7FeedForward(nn.Module):
    def __init__(self, state_dict, hidden_size, intermediate_size, layer_id=0, layer_total=0, output_last=False):
        super().__init__()
        prefix = f'blocks.{layer_id}.ffn.'
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.layer_total = layer_total
        self.output_last = output_last
        self.x_k = nn.Parameter(state_dict[prefix + 'x_k'])

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.key.weight = nn.Parameter(state_dict[prefix + 'key.weight'])
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.value.weight = nn.Parameter(state_dict[prefix + 'value.weight'])

        self.ln_2                   = nn.LayerNorm(hidden_size, eps=1e-5)
        self.ln_2.weight            = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.weight'])
        self.ln_2.bias              = nn.Parameter(state_dict[f'blocks.{layer_id}.ln2.bias'])

    def forward(self, x, state):
        batch_size, seq_length, _ = x.size()
        if self.output_last and self.layer_id == self.layer_total - 1:
            if seq_length == 1:
                last_x = x
                x = self.ln_2(x)
                state_out = x
                sx = state - x
            else:
                # 替换aimet CustomGather为索引
                last_x = x[:, -1:]
                x = x[:, -2:]  # 取最后两个元素
                x = self.ln_2(x)
                # 替换aimet Split为torch.split
                past, state_out = torch.split(x, 1, dim=1)
                sx = past - state_out
                x = state_out
        else:
            last_x = x
            x = self.ln_2(x)
            assert batch_size == 1
            if seq_length == 1:
                state_out = x
                sx = state - x
            else:
                state_out = x[:, -1:]
                past = x[:, :-1]
                past = torch.cat([state.reshape(1, 1, -1), past], dim=1)
                sx = past - x

        # 替换aimet的加减乘操作
        xk = x + (sx * self.x_k)

        key = F.relu(self.key(xk)) **2  # 替换aimet Pow为**运算符
        value = self.value(key)

        # 替换aimet Add为加法
        return last_x + value, state_out