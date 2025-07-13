import mindspore
import mindnlp
import mindnlp.core.nn as nn
import mindnlp.core.ops as ops
from typing import Tuple


class RWKV_BLOCK(nn.Module):
    """
    RWKV模型的块结构。

    Args:
        block_w (dict): 权重字典。
        n_embd (int): 嵌入维度。
        n_head (int): 头数。
        state (mindspore.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]。
        v_first: 第一层的值。
        i (int): 时间索引。
    """
    def __init__(self, block_w: dict, n_embd: int, n_head: int, state: mindspore.Tensor, v_first: mindspore.Tensor, i: int):
        super().__init__()
        self.layer_id = i
        self.head_size = 64
        self.n_embd = n_embd
        self.n_head = n_head
        
        # 时间状态索引
        i0 = (2 + self.head_size) * i + 0
        i1 = (2 + self.head_size) * i + 1
        i2 = (2 + self.head_size) * i + 2
        i3 = (2 + self.head_size) * (i + 1)

        # 初始化第一层的值
        self.v_first = v_first

        # 初始化时间状态视图
        self.state_view_channel = state[:, i0]
        self.state_view_time_1 = state[:, i1]
        self.state_view_time_2 = state[:, i2: i3, :]
        
        # 初始化层归一化
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln1.weight = nn.Parameter(block_w['ln1.weight'])
        self.ln1.bias = nn.Parameter(block_w['ln1.bias'])
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln2.weight = nn.Parameter(block_w['ln2.weight'])
        self.ln2.bias = nn.Parameter(block_w['ln2.bias'])

        # 初始化激活函数
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        
        # 初始化注意力参数
        self.x = nn.Parameter(ops.stack([block_w['att.x_r'],
                                        block_w['att.x_w'],
                                        block_w['att.x_k'],
                                        block_w['att.x_v'],
                                        block_w['att.x_a'],
                                        block_w['att.x_g']]))
        self.w0 = nn.Parameter(block_w['att.w0'])
        self.r_k = nn.Parameter(block_w['att.r_k'])
        self.w1 = nn.Parameter(block_w['att.w1'])
        self.w2 = nn.Parameter(block_w['att.w2'])
        self.a1 = nn.Parameter(block_w['att.a1'])
        self.a2 = nn.Parameter(block_w['att.a2'])
        self.a0 = nn.Parameter(block_w['att.a0'])
        self.g1 = nn.Parameter(block_w['att.g1'])
        self.g2 = nn.Parameter(block_w['att.g2'])
        if self.layer_id != 0:
            self.v2 = nn.Parameter(block_w['att.v2'])
            self.v1 = nn.Parameter(block_w['att.v1'])
            self.v0 = nn.Parameter(block_w['att.v0'])
        self.k_k = nn.Parameter(block_w['att.k_k'])
        self.k_a = nn.Parameter(block_w['att.k_a'])
        self.att_receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_receptance.weight = nn.Parameter(block_w['att.receptance.weight'])
        self.att_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_key.weight = nn.Parameter(block_w['att.key.weight'])
        self.att_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_value.weight = nn.Parameter(block_w['att.value.weight'])
        self.att_output = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_output.weight = nn.Parameter(block_w['att.output.weight'])        
        self.att_group_norm = nn.GroupNorm(num_groups=n_head, num_channels=n_embd, eps=64e-5, affine=True)
        self.att_group_norm.weight = nn.Parameter(block_w['att.ln_x.weight'])
        self.att_group_norm.bias = nn.Parameter(block_w['att.ln_x.bias'])
            
        # 初始化前馈参数
        self.x_k = nn.Parameter(block_w['ffn.x_k'])
        self.ffn_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_key.weight = nn.Parameter(block_w['ffn.key.weight'])
        self.ffn_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_value.weight = nn.Parameter(block_w['ffn.value.weight'])

    def channel_mixing(self, x: mindspore.Tensor) -> mindspore.Tensor:
        """
        通道混合函数。

        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, N_embd]。
        Returns:
            mindspore.Tensor: 混合后的张量，形状与输入的x相同。
        """
        sx = self.state_view_channel - x
        self.state_view_channel = x
        
        xk = x + sx * self.x_k
        k = self.relu(self.ffn_key(xk)).pow(2)

        return self.ffn_value(k)

    def time_mixing(self, x: mindspore.Tensor, v_first: mindspore.Tensor) -> mindspore.Tensor:
        """
        时间混合函数。

        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, N_embd]。
        Returns:
            mindspore.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, H, S = x.shape[0], self.n_head, self.head_size

        sx = (self.state_view_time_1 - x)
        self.state_view_time_1 = x

        xr, xw, xk, xv, xa, xg = ops.unbind(x.unsqueeze(1) + sx.unsqueeze(1) * self.x, dim=1)

        # 计算注意力机制的权重    
        w = self.w0 + ops.tanh(xw @ self.w1) @ self.w2
        w = ops.exp(-0.606531 * self.sigmoid(w)).view(batch_size, H, 1, S)

        # 计算注意力机制的组件
        r = self.att_receptance(xr).view(batch_size, H, S, 1)
        k = self.att_key(xk)
        v = self.att_value(xv)
        if self.layer_id == 0:
            v_first = v.copy() # 存储第一层的v
        else:
            v = v + (v_first - v) * ops.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        v = v.view(batch_size, H, S, 1)
        a = self.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = self.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = nn.functional.normalize(kk.view(batch_size, H, S), dim=-1, p=2.0).view(batch_size, -1)
        k = (k * (1 + (a-1) * self.k_a)).view(batch_size, H, 1, S)

        # 使用注意力机制更新状态
        vk = v @ k
        ab = (-kk).view(batch_size, H, S, 1) @ (kk * a).view(batch_size, H, 1, S)
        s = self.state_view_time_2.view(batch_size, H, S, S)
        s = s * w + s @ ab.float() + vk.float()
        self.state_view_time_2 = s.view(batch_size, S, -1)
        x = s @ r

        # 展平x并应用组归一化和门控
        x = self.att_group_norm(x.flatten(start_dim=1))
        rkv = (r.squeeze(-1) * k.squeeze(-2) * self.r_k).sum(dim=-1, keepdim=True) * v.squeeze(-1)
        x = (x + rkv.view(batch_size, H * S)) * g

        # 应用输出层并返回结果
        return self.att_output(x), v_first

    def forward(self, x: mindspore.Tensor, v_first: mindspore.Tensor) -> mindspore.Tensor:
        """
        模型的前向传播。
        Args:
            x (mindspore.Tensor): 输入张量，形状为[Batch, N_embd]。
        Returns:
            mindspore.Tensor: 前向传播结果张量，形状与输入的x相同。
        """
        xx, v_first = self.time_mixing(self.ln1(x), v_first)
        x = x + xx
        x = x + self.channel_mixing(self.ln2(x))
        return x, v_first
        

class RWKV_RNN(nn.Module):
    """
    RWKV模型的RNN结构。

    Args:
        args (dict): 参数字典。
    """
    def __init__(self, args: dict):
        super().__init__()
        self.args = args
        self.set_train(False)

        # 加载权重
        w = mindnlp.core.serialization.load(args['MODEL_NAME'] + '.pth')
        
        # 将所有权重转换为float32
        self.num_layer = 0
        for k in w.keys():
            w[k] = w[k].float()
            if '.x_' in k: w[k] = w[k].squeeze()
            if '.k_' in k: w[k] = w[k].squeeze()
            if 'att.r' in k: w[k] = w[k].squeeze()
            if 'att.w' in k: w[k] = w[k].squeeze()
            if 'att.v0' in k: w[k] = w[k].squeeze()
            if 'att.v1' in k: w[k] = w[k].squeeze()
            if 'att.v2' in k: w[k] = w[k].squeeze()
            if 'att.a' in k: w[k] = w[k].squeeze()
            if 'att.g' in k: w[k] = w[k].squeeze()
            if "blocks" in k: self.num_layer = max(self.num_layer, int(k.split(".")[1]))
        
        self.num_layer += 1

        self.head_size = 64
        self.n_head = w['blocks.0.att.r_k'].shape[0]
        self.n_embd = self.n_head * self.head_size
        self.state_size = [self.num_layer * (2 + self.head_size), self.n_embd]
        self.batch_size = args['batch_size']

        print(f"state_size: {self.state_size}") # 这里打印状态的形状
        
        # 初始化模型参数
        self.emb = nn.Embedding.from_pretrained(w['emb.weight'], freeze=True)
        self.ln0 = nn.LayerNorm(self.n_embd)
        self.ln0.weight = nn.Parameter(w['blocks.0.ln0.weight'])
        self.ln0.bias = nn.Parameter(w['blocks.0.ln0.bias'])
        self.blocks = nn.ModuleList()

        # 初始化参数
        self.state = ops.zeros([self.batch_size, *self.state_size])
        self.v_first = ops.zeros([self.batch_size, self.n_embd])
        
        for i in range(self.num_layer):
            # 提取当前块的权重
            block_w = {k[len(f'blocks.{i}.'):]: v for k, v in w.items() if f'blocks.{i}.' in k}
            self.blocks.append(RWKV_BLOCK(block_w, self.n_embd, self.n_head, self.state, self.v_first, i))
            print(f"Loading blocks...[{i + 1}/{self.num_layer}]", end='\r')
        print()

        self.ln_out = nn.LayerNorm(self.n_embd)
        self.ln_out.weight = nn.Parameter(w['ln_out.weight'])
        self.ln_out.bias = nn.Parameter(w['ln_out.bias'])
        self.head = nn.Linear(self.n_embd, args['vocab_size'], bias=False)
        self.head.weight = nn.Parameter(w['head.weight'])

    def forward(self, token: mindspore.Tensor) -> Tuple[mindspore.Tensor, mindspore.Tensor]:
        """
        模型的前向传播。
        Args:
            token (mindspore.Tensor): 输入的令牌张量。[Batch_size]
        Returns:
            mindspore.Tensor: 模型输出。
        """
        x = self.emb(token)
        x = self.ln0(x)
        for block in self.blocks:
            x, self.v_first = block(x, self.v_first)
        x = self.ln_out(x)
        x = self.head(x)
        return x
