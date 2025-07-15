# channel_mixing
## 贡献说明
| 贡献者    | 贡献方  | 贡献算子                | 贡献时间      | 贡献内容                    |
|--------|------|---------------------|-----------|-------------------------|
| appleinsky | rwkv&昇腾 | channel_mixing | 2025/7/13 | 新增channel_mixing算子 |

## 支持的产品型号
- Atlas 推理系列产品
- Atlas 200I/500 A2 推理产品
## 算子描述
- 功能描述
实现channel_mixing算子计算逻辑:

### 原型信息
- `B`: 输入的Batch数 (shape: [], type: uint32)
- `T`: 输入的序列长度 (shape: [], type: uint32) 
- `C`: 输入的维度 (shape: [], type: uint32)
- `H`: 输入的block数, H能被C整除, N = C // H (shape: [], type: uint32)
  <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">channel_mixing</td></tr>
    </tr>
    <tr><td rowspan="6" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">x</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">h0</td><td align="center">B,1,C</td><td align="center">float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">xk</td><td align="center">1,1,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">kw</td><td align="center">4C,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">vw</td><td align="center">C,4C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    </tr>
    </tr>
    <tr><td rowspan="2" align="center">算子输出</td><td align="center">output</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <td align="center">ht</td><td align="center">B,1,C</td><td align="center">float16、float</td><td align="center">ND</td></tr>
    </tr>
  </table>

- 实现逻辑
```python
def channel_mixing(x: torch.Tensor, h0: torch.Tensor, x_k: torch.Tensor, ffn_key: torch.Tensor, ffn_value: torch.Tensor) -> torch.Tensor:
    """
    通道混合函数。

    Args:
        x (torch.Tensor): 输入张量，形状为[Batch, Seq_length, N_embd]。
        h0 (torch.Tensor): 输入张量(state状态)，形状为[Batch, 1, N_embd]。
        x_k (torch.Tensor): 输入张量(模型权重），形状为[1, 1, N_embd]。
        ffn_key (torch.Tensor): 输入张量(模型权重），形状为[4*N_embd, N_embd]。
        ffn_value (torch.Tensor): 输入张量(模型权重），形状为[N_embd, 4*N_embd]。
    Returns:
        out (torch.Tensor): 输出张量，形状为[Batch, Seq_length, N_embd]。
        ht (torch.Tensor): 输出张量(state状态),形状为[Batch, 1, N_embd]。
    """
    batch_size, seq_length = x.shape[0], x.shape[1]
    # Token shift
    if seq_length == 1:
        sx = h0 - x
        ht = x
    else:
        h0 = h0.view(batch_size,1,-1)
        h0 = torch.cat([h0, x[:, :-1, :]], dim=1)
        sx = (h0 - x)
        ht = x[:, -1, :]
    xk = x + sx * x_k

    k = torch.relu(xk @ ffn_key.T).pow(2)

    return k @ ffn_value.T, ht

# 调用样例
out, state = channel_mixing(x, state, xk, kw, vw)

```

接口:

```cpp
extern "C" __global__ __aicore__ void channel_mixing(GM_ADDR x, GM_ADDR h0, GM_ADDR xk, GM_ADDR kw, GM_ADDR vw, GM_ADDR workspace, GM_ADDR tiling)
```
