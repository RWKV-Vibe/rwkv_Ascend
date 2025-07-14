## 🎯 项目介绍
cann-ops-rwkv是昇腾与rwkv共建的算子仓库，欢迎rwkv爱好者学习、使用和魔改的RNN attention（rwkv、fla）系列算子代码。

## 🔍 仓库结构
cann-ops仓关键目录如下所示：
```
├── cmake
├── src // 算子源码目录
│ ├── common // 公共目录
│ ├── rwkv4 // rwkv4算子目录
│ │ └── wkv4     // wkv4算子目录
| | └── wkv4grad // wkv4grad算子目录
│ └── CMakeLists.txt
├── rwkv6 // rwkv6算子目录
│ │ └── wkv6     // wkv6算子目录
| | └── wkv6grad // wkv6grad算子目录
| | └── CMakeLists.txt
├── rwkv7 // rwkv7算子目录
│ │ └── wkv7     // wkv7算子目录
| | └── wkv7grad // wkv7grad算子目录
| | └── CMakeLists.txt
├── CMakeLists.txt
├── CMakePresets.json // 配置文件
├── LICENSE
├── README.md
└── build.sh // 算子编译脚本
```
## ⚡️ 支持算子
| 算子名称  | 样例介绍  | 开发语言  |
|---|---|---|
| [wkv4](./src/rwkv4/wkv4)  | Time_mixing中的wkv4前向算子(vector)  |  Ascend C |
| [wkv4grad](./src/rwkv4/wkv4grad)  | Time_mixing中的wkv4grad反向算子(vector)  |  Ascend C |
| [wkv6](./src/rwkv6/wkv6)  | Time_mixing中的wkv6前向算子(vector)  |  Ascend C |
| [wkv7](./src/rwkv7/wkv7)  | Time_mixing中的wkv7前向算子(vector)  |  Ascend C |
| [wkv7grad](./src/rwkv7/wkv7grad)  | Time_mixing中的wkv7grad反向算子(cube & vector)  |  Ascend C |

## 📝 硬件支持说明
- Atlas 训练系列产品
- Atlas 推理系列产品
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 📄 许可证书
[CANN Open Software License Agreement Version 1.0](LICENSE)
