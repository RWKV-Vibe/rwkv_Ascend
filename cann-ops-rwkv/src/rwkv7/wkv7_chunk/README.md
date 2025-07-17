# WKV7 Vector

## 贡献说明
| 贡献者    | 贡献方  | 贡献算子                | 贡献时间      | 贡献内容                    |
|--------|------|---------------------|-----------|-------------------------|
| appleinsky | rwkv&昇腾 | wkv7 | 2025/7/13 | 新增wkv7算子 |

## 支持的产品型号

- Atlas 推理系列产品
- Atlas 200I/500 A2 推理产品
## 算子描述
- 功能描述
实现RWKV7的time_mixing单元的 prefill & decode wkv7算子计算逻辑:

### 原型信息
- `B`: 输入的Batch数 (shape: [], type: uint32)
- `T`: 输入的序列长度 (shape: [], type: uint32) 
- `C`: 输入的维度 (shape: [], type: uint32)
- `H`: 输入的block数, H能被C整除, N = C // H (shape: [], type: uint32)
  <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">WKV7</td></tr>
    </tr>
    <tr><td rowspan="8" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">query</td><td align="center">B,T,H,N</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">key</td><td align="center">B,T,H,N</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">value</td><td align="center">B,T,H,N</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">weight</td><td align="center">B,T,H,N</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">a</td><td align="center">B,T,H,N</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">b</td><td align="center">B,T,H,N</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">h0</td><td align="center">B,T,N,N</td><td align="center">float16、float</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="2" align="center">算子输出</td><td align="center">output</td><td align="center">B,T,H,N</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <td align="center">ht</td><td align="center">B,T,N,N</td><td align="center">float16、float</td><td align="center">ND</td></tr>
    </tr>
  </table>

- 实现逻辑
```python
def wkv7(k: torch.Tensor, v: torch.Tensor, w: torch.Tensor, r: torch.Tensor,  a: torch.Tensor,  b: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """
    wkv7 kernel

    Args:
        k, v, w, r, a, b (torch.Tensor): 输入张量，形状为[Batch, Seq_length, head_size, head_dim]。
        h0 (torch.Tensor): 输入张量(state状态)，形状为[Batch, 1, head_dim, head_dim]。
    Returns:
        out (torch.Tensor)：输出张量，形状为[Batch, Seq_length, head_size, head_dim]
        ht (torch.Tensor): 输出张量(state状态),形状为[Batch, 1, head_dim, head_dim]。
    """
    B, T, H, N = k.shape
    vk = value.view(B, T, H, N, 1) @ key.view(B, T, H, 1, N)
    w = torch.exp(w)
    if T == 1: 
        ab = a.view(B, T, H, N, 1) @ b.view(B, T, H, 1, N)
        ht = h0 * w + (h0 @ ab) + vk
        x = ht @ r
    else:
        b = b.view(B, T, H, 1, N)
        a = a.view(B, T, H, N, 1)
        x = torch.zeros(B, T, H, N, 1, device=k.device,dtype=vk.dtype)
        w = w.view(B, T, H, 1, N)
        r = r.view(B, T, H, N, 1)
        for i in range(T):
            ht = h0 * w[:, i, : , :, :] + (h0 @ a[:, i, :, :, :] @ b[:, i, :, :, :]) + vk[:, i, :, :, :]
            x[:, i, :, :, :] = ht @ r[:, i, :, :, :]
        x = x.view(B, T, H, 1, N)

# 调用样例
out, ht = wkv7(k, v, w, r, a, b, h0)
```

接口:

```cpp
extern "C" __global__ __aicore__ void wkv7(GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, 
                                            GM_ADDR a, GM_ADDR b, GM_ADDR h0, GM_ADDR o, GM_ADDR ht, 
                                            GM_ADDR workspace, GM_ADDR tiling)
```
### 参考实现
#### fused
- [bo's cuda](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/cuda)
- [fused triton](https://github.com/RWKV-Vibe/rwkv-fla/blob/stable/fla/ops/rwkv7/fused_recurrent.py)
- [prefill AscendC](https://github.com/RWKV-Vibe/rwkv_Ascend/tree/main/cann-ops-rwkv/src/rwkv7/wkv7)
#### chunk
- [chunk dplr triton](https://github.com/RWKV-Vibe/rwkv-fla/blob/stable/fla/ops/generalized_delta_rule/dplr/chunk.py#L22)
- [chunk dplr cuda](https://github.com/johanwind/wind_rwkv/blob/main/wind_rwkv/rwkv7/chunked_cuda/chunked_cuda.cu)