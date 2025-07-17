# token_shift

## 贡献说明
| 贡献者    | 贡献方  | 贡献算子                | 贡献时间      | 贡献内容                    |
|--------|------|---------------------|-----------|-------------------------|
| appleinsky | rwkv&昇腾 | token_shift | 2025/7/13 | 新增token_shift算子 |

## 支持的产品型号
- Atlas 推理系列产品
- Atlas 200I/500 A2 推理产品
## 算子描述
- 功能描述
实现time_mixing单元token_shift的token_shift算子计算逻辑:

### 原型信息
- `B`: 输入的Batch数 (shape: [], type: uint32)
- `T`: 输入的序列长度 (shape: [], type: uint32) 
- `C`: 输入的维度 (shape: [], type: uint32)
- `H`: 输入的block数, H能被C整除, N = C // H (shape: [], type: uint32)
  <table>
    <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">token_shift</td></tr>
    </tr>
    <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td align="center">x</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">rwkvag</td><td align="center">6,1,1,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    <tr><td align="center">h0</td><td align="center">B,1,C</td><td align="center">float16、float</td><td align="center">ND</td></tr>

    </tr>
    </tr>
    <tr><td rowspan="7" align="center">算子输出</td>
    <td align="center">xr</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    </tr>
    <td align="center">xw</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    </tr>
    <td align="center">xk</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    </tr>
    <td align="center">xv</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    </tr>
    <td align="center">xa</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    </tr>
    <td align="center">xg</td><td align="center">B,T,C</td><td align="center">int8、float16、float</td><td align="center">ND</td></tr>
    </tr>
    <td align="center">ht</td><td align="center">B,1,C</td><td align="center">float16、float</td><td align="center">ND</td>
    </tr>
  </table>

- 实现逻辑
```python
def token_shift(x: torch.Tensor, rwkvag: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
    """
    通道混合函数。

    Args:
        x (torch.Tensor): 输入张量，形状为[Batch, Seq_length, N_embd]。
        rwkvag (torch.Tensor): 输入张量(模型权重），形状为[5, 1, 1, N_embd]。
        h0 (torch.Tensor): 输入张量(state状态)，形状为[Batch, 1, N_embd]。
    Returns:
        xr，xw, xk, xv, xa, xg (torch.Tensor)：输出r, w, k, v, a, g混合后的张量，形状均为[Batch, Seq_length, N_embd]
        ht (torch.Tensor): 输出张量(state状态),形状为[Batch, 1, N_embd]。
    """
    batch_size, seq_length = x.shape[0], x.shape[1]
    if seq_length == 1:
        sx = (h0 - x)
        ht = x
    else:
        h0 = h0.view(batch_size,1,-1)
        xx = torch.cat([h0, x[:, :-1, :]], dim=1)
        sx = (xx - x)
        ht = x[:, -1, :]
    xr = x + rwkvag[0] * sx
    xw = x + rwkvag[1] * sx
    xk = x + rwkvag[2] * sx
    xv = x + rwkvag[3] * sx
    xa = x + rwkvag[4] * sx
    xg = x + rwkvag[5] * sx

    return xr，xw, xk, xv, xa, xg, ht

# 调用样例
xr，xw, xk, xv, xa, xg, ht = token_shift(x, rwkvag, h0)
```

接口:

```cpp
extern "C" __global__ __aicore__ void token_shift(GM_ADDR x, GM_ADDR rwkvag, GM_ADDR h0, GM_ADDR xr，GM_ADDR xw, GM_ADDR xk, GM_ADDR xv, GM_ADDR xa, GM_ADDR xg, GM_ADDR ht, GM_ADDR workspace, GM_ADDR tiling)
```