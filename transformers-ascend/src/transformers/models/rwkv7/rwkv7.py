from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch_npu
import custom_ops
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
import sys
import os
# sys.path.append(os.environ['WKV_LIB'])
# sys.path.append("/home/ma-user/work/wkv7_Ascend/npu/build")
# import rwkv7_vector

from .lora import LoRA


if TYPE_CHECKING:
    from .utils import Cache

def fused_addcmul_rwkv7(hidden_states, xx, xr, xw, xk, xv, xa, xg=None):
    oxr = torch.addcmul(hidden_states, xx, xr)
    oxw = torch.addcmul(hidden_states, xx, xw)
    oxk = torch.addcmul(hidden_states, xx, xk)
    oxv = torch.addcmul(hidden_states, xx, xv)
    oxa = torch.addcmul(hidden_states, xx, xa)
    if xg is not None:
        oxg = torch.addcmul(hidden_states, xx, xg)
        return oxr, oxw, oxk, oxv, oxa, oxg
    else:
        return oxr, oxw, oxk, oxv, oxa, None

def recurrent_dplr_delta_rule_ref(
    q: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    q, k, v, a, b, w = map(lambda x: x.transpose(1, 2).to(torch.float), (q, k, v, a, b, w))

    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.torch.zeros(B, H, K, V, dtype=torch.float, device=q.device)
    if initial_state is not None:
        S = initial_state
    if scale is None:
        scale = K ** -0.5
    q = q * scale

    for i in range(T):
        _q = q[:, :, i]
        _k = k[:, :, i]
        _v = v[:, :, i].clone()
        a_i = a[:, :, i]
        b_i = b[:, :, i]
        # first matmul then decay in DPLR.
        _v2 = (S.clone() * a_i[..., None]).sum(-2)
        S = S.clone() * w[:, :, i].exp()[..., None]
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2) + b_i.unsqueeze(-1) * _v2.unsqueeze(-2)
        o[:, :, i] = torch.einsum('bhd,bhdm->bhm', _q, S)
    if not output_final_state:
        S = None
    o = o.transpose(1, 2)
    return o, S

def naive_recurrent_rwkv7(
    q: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    q, k, v, a, b, w = map(lambda x: x.transpose(1, 2).to(torch.float), (q, k, v, a, b, w))

    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.torch.zeros(B, H, K, V, dtype=torch.float, device=q.device)
    if initial_state is not None:
        S = initial_state
    if scale is None:
        scale = K ** -0.5
    q = q * scale

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
    if not output_final_state:
        S = None
    o = o.transpose(1, 2)
    return o, S

def fused_recurrent_rwkv7(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
):
    """
    Args:
        r (torch.Tensor):
            r of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        w (torch.Tensor):
            log decay of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            k of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            v of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        a (torch.Tensor):
            a of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        b (torch.Tensor):
            b of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        scale (float):
            scale of the attention.
        initial_state (torch.Tensor):
            initial state of shape `[B, H, K, V]` if cu_seqlens is None else `[N, H, K, V]` where N = len(cu_seqlens) - 1.
        output_final_state (bool):
            whether to output the final state.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (bool):
            whether to use head first. Recommended to be False to avoid extra transposes.
            Default: `False`.
    """
    # please remember here w = -log(w)
    B, T, H, K, V = r.shape[0], r.shape[1], r.shape[2], r.shape[3], v.shape[-1]
    r, k, v, a, b, w = map(lambda x: x.transpose(1, 2).to(torch.float).contiguous(), (r, k, v, a, b, w))
    
    state = torch.zeros(B, H, K, V, dtype=torch.float, device=r.device).contiguous()
    o = torch.zeros_like(v)
    
    if initial_state is not None:
        state = initial_state.to(torch.float).contiguous() 
        
    try:
        torch.npu.synchronize()  # 确保之前操作完成
        o, state = custom_ops.wkv7(k, v, w, r, a, b, state)
        torch.npu.synchronize()  # 捕获算子内部错误
    except RuntimeError as e:
        print(f"Operator failed: {e}")
        raise
    if not output_final_state:
        state = None
    o = o.transpose(1, 2)
    # ht = state if output_final_state else None
    return o, state


class RWKV7Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = None,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        fuse_norm: bool = False,
        value_dim: int = None,
        num_hidden_layers: int = None,
        **kwargs
    ) -> RWKV7Attention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        self.head_v_dim = int(self.value_dim // self.num_heads)

        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers
        self.fuse_norm = fuse_norm

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        self.g_norm = nn.GroupNorm(
                num_groups=self.num_heads,
                num_channels=self.value_dim,
                eps=self.head_dim*norm_eps,
                affine=elementwise_affine
            )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return

        # Initialize only when we're processing the RWKV7Attention module itself
        if isinstance(module, RWKV7Attention) and self.layer_idx is not None:
            ratio_0_to_1 = self.layer_idx / (self.num_hidden_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (self.layer_idx / self.num_hidden_layers)  # 1 to ~0

            # Create position-based initialization tensor
            with torch.no_grad():
                ddd = torch.ones(1, 1, self.hidden_size)
                for i in range(self.hidden_size):
                    ddd[0, 0, i] = i / self.hidden_size

                # Initialize x_* parameters directly
                self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_r.dtype)
                self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_w.dtype)
                self.x_k.data = (1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)).to(self.x_k.dtype)
                self.x_v.data = (1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)).to(self.x_v.dtype)
                self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).to(self.x_a.dtype)
                self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).to(self.x_g.dtype)
                # Set specific bias values for LoRA modules
                # w0 initialization - complex decay speed
                decay_speed = torch.ones(self.hidden_size)
                for n in range(self.hidden_size):
                    decay_speed[n] = -7 + 5 * (n / (self.hidden_size - 1)) ** (
                        0.85 + 1.0 * ratio_0_to_1**0.5
                    )
            # Initialize k_k, k_a, r_k
            nn.init.constant_(self.k_k, 0.85)
            nn.init.constant_(self.k_a, 1.0)
            nn.init.zeros_(self.r_k)

            self.w_lora.set_bias_value(decay_speed + 0.5)

            # v0 initialization - ones (for non-first layers)
            if self.layer_idx != 0:
                self.v_lora._initialize_weights(self.v_lora)
                self.v_lora.set_bias_value(1.0)

            self.r_proj.weight.data.uniform_(-0.5/(self.hidden_size**0.5), 0.5/(self.hidden_size**0.5))
            self.k_proj.weight.data.uniform_(-0.05/(self.hidden_size**0.5), 0.05/(self.hidden_size**0.5))
            self.v_proj.weight.data.uniform_(-0.5/(self.hidden_size**0.5), 0.5/(self.hidden_size**0.5))
            self.o_proj.weight.data.zero_()

        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, seq_len, _ = hidden_states.shape

        if self.training:
            # if training, use chunk mode no matter how short the sequence is
            mode = 'chunk'
        else:
            # launching the triton kernel for just one token will actually be slower
            mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if attention_mask is not None:
            hidden_states = hidden_states.mul(attention_mask[:, -hidden_states.shape[-2]:, None])
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state['conv_state'].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state['conv_state']

        # [batch_size, seq_len, hidden_size]
        delta = shifted - hidden_states

        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(hidden_states, delta, self.x_r, self.x_w,
                                                     self.x_k, self.x_v, self.x_a, self.x_g)

        r = self.r_proj(xr)
        # Using bf16 for LoRA computation is numerically safe here because:
        # 1. After sigmoid activation:
        #    - Max absolute error (vs float32): 0.003
        #    - Mean absolute error: 0.0004
        # 2. Subsequent scaling by -0.6065 will further reduce relative error
        #    (error scales linearly with constant multiplication)
        # 3. Final compounded error remains within acceptable bounds for bf16 precision
        # Empirical observation confirms bf16 introduces no practical degradation
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)

        # Prefer addcmul over expanded form for numerical stability in bf16:
        # 1. Fused Multiply-Add (FMA) in addcmul reduces intermediate rounding:
        #    - Single op vs original 3 ops (mul, sub, mul)
        #    - 1 less intermediate value storage (bf16 write->read overhead)
        # 2. Mathematically equivalent to k*(1 + (a-1)*self.k_a)
        #    but with better precision preservation
        # 3. Particularly crucial for bf16 where intermediate values easily lose precision
        k = k.addcmul(k * (a - 1), self.k_a)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]
        r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, a))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)
        # print("input r shape:", r.shape)
        
        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        # if recurrent_state is not None:
        #     print("input state shape:", recurrent_state.shape)
            
        cu_seqlens = kwargs.get('cu_seqlens', None)
        
        
        # o, recurrent_state = recurrent_dplr_delta_rule_ref(
        #     q=r,
        #     w=w,
        #     k=k,
        #     v=v,
        #     a=-kk,
        #     b=kk * a,
        #     scale=1.,
        #     initial_state=recurrent_state,
        #     output_final_state=use_cache,
        #     cu_seqlens=cu_seqlens,
        # )
        # o, recurrent_state = naive_recurrent_rwkv7(
        #     q=r,
        #     w=w,
        #     k=k,
        #     v=v,
        #     a=-kk,
        #     b=kk * a,
        #     scale=1.,
        #     initial_state=recurrent_state,
        #     output_final_state=use_cache,
        #     cu_seqlens=cu_seqlens,
        # )
        o, recurrent_state = fused_recurrent_rwkv7(
            r=r,
            w=w,
            k=k,
            v=v,
            a=-kk,
            b=kk * a,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        # print("out1 o shape:", o)
        # print("out1 state shape:", recurrent_state)
        # # print("out2 o shape:", o2)
        # # print("out2 state shape:", recurrent_state2)
        # print("out3 o shape:", o3)
        # print("out3 state shape:", recurrent_state3)
        
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=hidden_states[:, -1],
                layer_idx=self.layer_idx,
                offset=r.shape[1]
            )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
        else:
            o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)

        o = o + ((r * k * self.r_k).sum(-1, keepdim=True) * v).view(batch_size, seq_len, -1)
        o = self.o_proj(o * g)

        return o, None, past_key_values, v_first
