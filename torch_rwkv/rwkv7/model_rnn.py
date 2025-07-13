import math
import torch
import torch_npu
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from typing import Optional


def LoRA(
    x: torch.Tensor,                 # 输入张量 [B, D]
    xa: torch.Tensor,               # 低秩 A 权重 [D, r]
    xb: torch.Tensor,               # 低秩 B 权重 [r, D_out]
    activation: Optional[str] = None, 
    bias: Optional[torch.Tensor] = None  # 可选 bias [D_out] 或 [B, D_out]
) -> torch.Tensor:
    """
    LoRA (Low-Rank Adaptation) 层实现。

    Args:
        x: 输入张量，形状为 [B, D]
        xa: LoRA A 矩阵，形状为 [D, r]
        xb: LoRA B 矩阵，形状为 [r, D_out]
        activation: 可选的激活函数（'relu', 'sigmoid', 'tanh'），默认无
        bias: 可选的 bias，形状应为 [D_out] 或 [B, D_out]

    Returns:
        输出张量，形状为 [B, D_out]
    """

    # 选择激活函数
    if activation == 'sigmoid':
        act_fn = torch.sigmoid
    elif activation == 'tanh':
        act_fn = torch.tanh
    elif activation == 'relu':
        act_fn = torch.relu
    else:
        act_fn = lambda x: x  # 等价于 nn.Identity()

    A = x @ xa                      # [B, r]
    AB = act_fn(A) @ xb            # [B, D_out]

    if bias is not None:
        AB = AB + bias             # bias 会自动广播

    return AB

def TimeDecay(xw: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor):
        """
        构造 RWKV 时间衰减权重模块(token-shift + decay)

        Args:
            input_dim: 输入维度(通常为 n_embd)
            hidden_dim: 中间隐藏维度
            output_dim: 输出维度(通常为 head_size)
        """
       
        """
        计算时间衰减权重 w

        Args:
            xw: 输入 [B, n_embd]
            batch_size, n_head, head_size: 输出维度组织格式

        Returns:
            Tensor: [B, n_head, 1, head_size]
        """
        B, N = xw.shape[0],  w1.shape[1]
        H = int(w1.shape[0] / w1.shape[1])
        
        w = torch.tanh(xw @ w1) @ w2
        w = w0 + w.float()
        w = torch.exp(-0.606531 * w.sigmoid())
        
        return w.view(B, H, 1, N)

def fused_addcmul_rwkv7(hidden_states: torch.Tensor, delta: torch.Tensor, xs: torch.Tensor):
    """
    fused_addcmul_rwkv7: 计算 token-shift 后的 6 个向量(xr, xw, xk, xv, xa, xg)
    
    Args:
        hidden_states: [B, emb]，当前 token 的隐状态
        delta:         [B, emb]，与上一 token 的差值 sx
        xs:            [6, emb],token shift 的乘数系数

    Returns:
        Tuple[xr, xw, xk, xv, xa, xg]: 每个都是 [B, emb]
    """
    # [6, emb] -> [6, 1, emb]
    xs_exp = xs.unsqueeze(1)  # [6, 1, emb]
    # delta: [B, emb] -> [1, B, emb]
    delta_exp = delta.unsqueeze(0)  # [1, B, emb]
    # hidden_states: [B, emb] -> [1, B, emb]
    hidden_exp = hidden_states.unsqueeze(0)  # [1, B, emb]

    # [6, B, emb] = [1, B, emb] + [1, B, emb] * [6, 1, emb]
    out = hidden_exp + delta_exp * xs_exp  # broadcast 相乘

    # 拆成 6 个 [B, emb]
    return torch.unbind(out, dim=0)

class RWKV_BLOCK(nn.Module):
    """
    RWKV模型的块结构。

    Args:
        block_w (dict): 权重字典。
        n_embd (int): 嵌入维度。
        n_head (int): 头数。
        state (torch.Tensor): 隐藏状态张量。[Batch_size, State_size, N_embd]。
        v_first: 第一层的值。
        i (int): 时间索引。
    """
    def __init__(self, block_w: dict, n_embd: int, n_head: int, state: torch.Tensor, v_first: torch.Tensor, i: int):
        super().__init__()
        self.layer_id = i
        self.head_size = 64
        self.n_embd = n_embd
        self.n_head = n_head
        
        # self.a_lora = LoRA(self.n_embd, self.n_embd, low_rank_dim=64, activation=None).to('npu:0')
        # self.g_lora = LoRA(self.n_embd, self.n_embd, low_rank_dim=128, activation='sigmoid', bias=False).to('npu:0')
        
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
        self.x = nn.Parameter(torch.stack([block_w['att.x_r'],
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
        # self.att_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_key_weight = nn.Parameter(block_w['att.key.weight'])
        # self.att_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_value_weight = nn.Parameter(block_w['att.value.weight'])
        self.att_output = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.att_output.weight = nn.Parameter(block_w['att.output.weight'])        
        self.att_group_norm = nn.GroupNorm(num_groups=n_head, num_channels=n_embd, eps=64e-5, affine=True)
        self.att_group_norm.weight = nn.Parameter(block_w['att.ln_x.weight'])
        self.att_group_norm.bias = nn.Parameter(block_w['att.ln_x.bias'])
            
        # 初始化前馈参数
        self.x_k = nn.Parameter(block_w['ffn.x_k'])
        # self.ffn_key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_key_weight = nn.Parameter(block_w['ffn.key.weight'])
        # self.ffn_value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ffn_value_weight = nn.Parameter(block_w['ffn.value.weight'])

    def channel_mixing(self, x: torch.Tensor) -> torch.Tensor:
        """
        通道混合函数。

        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, N_embd]。
        Returns:
            torch.Tensor: 混合后的张量，形状与输入的x相同。
        """
        sx = self.state_view_channel - x
        self.state_view_channel = x
        
        # Token shift
        xk = x + sx * self.x_k
        
        # k = self.relu(self.ffn_key(xk)).pow(2)
        # 等价于 self.ffn_key(xk) = xk @ W_key^T
        k = self.relu(xk @ self.ffn_key_weight.T).pow(2)

        # return self.ffn_value(k)
        # 等价于 self.ffn_value(k) = k @ W_value^T
        return k @ self.ffn_value_weight.T

    def time_mixing(self, x: torch.Tensor, v_first: torch.Tensor) -> torch.Tensor:
        """
        时间混合函数。

        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, N_embd]。
        Returns:
            torch.Tensor: 混合后的时间状态张量，形状与输入的state相同。
        """
        batch_size, H, S = x.shape[0], self.n_head, self.head_size
  
        sx = (self.state_view_time_1 - x)
        self.state_view_time_1 = x

        # xr, xw, xk, xv, xa, xg = torch.unbind(x.unsqueeze(1) + sx.unsqueeze(1) * self.x, dim=1)
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(x, sx, self.x)
        
        # token shift(data-dependent)    
        # w = torch.tanh(xw @ self.w1) @ self.w2
        # w = self.w0 + w.float()  
        # w = torch.exp(-0.606531 * self.sigmoid(w)).view(batch_size, H, 1, S)
        w = TimeDecay(xw, self.w0, self.w1, self.w2)

        
        # 计算注意力机制的组件
        r = self.att_receptance(xr).view(batch_size, H, S, 1)
        # k = self.att_key(xk)
        k = xk @ self.att_key_weight.T
        # v = self.att_value(xv)
        v = xv @ self.att_value_weight.T
        if self.layer_id == 0:
            v_first = v.clone() # 存储第一层的v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        v = v.view(batch_size, H, S, 1)
        
        # a = self.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        a = LoRA(xa, self.a1, self.a2, bias=self.a0).sigmoid()
        
        # g = self.sigmoid(xg @ self.g1) @ self.g2
        g = LoRA(xg, self.g1, self.g2, activation='sigmoid')

        
        kk = k * self.k_k
        kk = F.normalize(kk.view(batch_size, H, S), dim=-1, p=2.0).view(batch_size, -1)
        k = (k * (1 + (a-1) * self.k_a)).view(batch_size, H, 1, S)

        # wkv7
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

    def forward(self, x: torch.Tensor, v_first: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状为[Batch, N_embd]。
        Returns:
            torch.Tensor: 前向传播结果张量，形状与输入的x相同。
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

        # 加载权重
        w = torch.load(args['MODEL_NAME'] + '.pth', map_location=args['device'])
        
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
        self.state = torch.zeros([self.batch_size, *self.state_size], device=args['device'])
        self.v_first = torch.zeros([self.batch_size, self.n_embd], device=args['device'])
        
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

    def forward(self, token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型的前向传播。
        Args:
            token (torch.Tensor): 输入的令牌张量。[Batch_size]
        Returns:
            torch.Tensor: 模型输出。
        """
        x = self.emb(token)
        x = self.ln0(x)
        for block in self.blocks:
            x, self.v_first = block(x, self.v_first)
        x = self.ln_out(x)
        x = self.head(x)
        return x
