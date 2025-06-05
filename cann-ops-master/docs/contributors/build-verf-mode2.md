# 本地构建和验证模式二

## 概述

本地构建和验证模式二是直接在本仓上进行开发验证，本文主要介绍该模式下如何进行本地构建和验证

## 本仓算子目录结构介绍

本仓对应算子的目录结构如下

```
├── add_custom                // cann-ops库算子目录
│ ├── docs                    // 算子文档目录
│ ├── example                 // 算子调用示例目录
│ │ └── AclNNInvocationNaive  // aclnn接口测试目录
│ ├── framework               // 框架插件目录
│ ├── op_host                 // host目录
│ ├── op_kernel               // kernel目录
│ ├── opp_kernel_aicpu        // aicpu目录
│ ├── tests                   // test目录
│ │ ├── st                    // st目录
│ │ └── ut                    // ut目录
│ ├── README.md               // 算子编译部署调用说明文档
│ └── CMakeLists.txt          // CMakeLists文件
```

注：上述本仓对应算子的目录结构除了算子名add_custom，其他目录和文件名不能修改，否则会导致CI失败

## 本地构建和验证流程

以开发新算子Sqrt为例

### 算子包编译部署（必须）

1. 拷贝一份示例算子src/math/add_custom到src/math下，将add_custom重命名为sqrt，命名风格统一用全小写且下划线分割。
2. 修改src/math/sqrt/op_host目录下的文件名，将add_custom.cpp修改成sqrt.cpp，将add_custom_tiling.h修改成sqrt_tiling.h，并将这两个文件中的算子名`AddCustom`替换成`Sqrt`
3. 修改src/math/sqrt/op_kernel目录下的文件名，将add_custom.cpp修改成sqrt.cpp，并将这个文件中的kernel入口函数名`add_custom`替换成`sqrt`，`add_custom_do`替换成`sqrt_do`
4. 打开src/math/sqrt/CMakeLists文件，将其中的`add_custom.cpp`修改成`sqrt.cpp`
5. 完成算子host侧代码开发
6. 完成算子kernel侧代码开发
7. 参考[算子工程说明文档中的算子包编译部署章节](../../src/math/add_custom/README.md#算子包编译部署)完成算子包编译部署，可在步骤4完成后提前编译一把保证和框架相关的代码修改没问题

### aclnn接口自验证（必须）

**请确保已根据算子包编译部署步骤完成本算子的编译部署动作**

1. 修改src/math/sqrt/examples/AclNNInvocationNaive下的main.cpp、gen_data.py和verify_result.py文件使其适配当前算子
2. 参考[aclnn说明文档中的运行样例算子章节](../../src/math/add_custom/examples/AclNNInvocationNaive/README.md#运行样例算子)完成aclnn接口自验证

### ST自验证（必须）

**请确保已根据算子包编译部署步骤完成本算子的编译部署动作**

1. 生成算子测试用例定义文件
   ```
   cd src/math/sqrt
   msopst create -i op_host/example_custom.cpp -out tests/st/
   ```
2. 将生成的src/math/sqrt/tests/st/Sqrt_case_timestamp.json重命名为Sqrt_case_all_type.json，替换原有的json文件
3. 打开src/math/sqrt/tests/st/Sqrt_case_all_type.json文件，填写要测试的shape，并参考src/math/add_custom/tests/st/AddCustom_case_all_type.json添加一行来配置算子期望数据生成脚本
   ```
   "calc_expect_func_file": "test_sqrt.py:calc_expect_func",
   ```
4. 将src/math/sqrt/tests/st/test_add_custom.py重命名为test_sqrt.py，修改该文件中calc_expect_func函数的接口和逻辑以适配当前算子
5. 根据执行机器的架构修改msopst.ini中的atc_singleop_advance_option和HOST_ARCH
6. 参考[ST说明文档中的执行测试用例章节](../../src/math/add_custom/tests/st/README.md#执行测试用例)完成ST自验证

### 撰写文档（必须）

1. 撰写下列上仓必须的文档

| 路径                                                             | 文档内容                  | 参考模板                                                     |
| ------------------------------------------------------------------ | --------------------------- | -------------------------------------------------------------- |
| src/math/sqrt/docs/AddCustom.md                       | 算子文档                  | src/math/add_custom/docs/AddCustom.md                       |
| src/math/sqrt/README.md                               | 算子编译部署调用说明文档  | src/math/add_custom/README.md                               |
| src/math/sqrt/tests/st/README.md                      | 算子ST测试说明文档        | src/math/add_custom/tests/st/README.md                      |
| src/math/sqrt/examples/AclNNInvocationNaive/README.md | 算子aclnn接口测试说明文档 | src/math/add_custom/examples/AclNNInvocationNaive/README.md |

### 补充算子调用（可选）

**补充的算子调用不会在门禁上看护，请在本地验证通过后合入​**

1. 在src/math/sqrt/examples目录下补充算子调用代码和文档，可能的调用方式包括但不限于如下方式

| 目录                 | 描述                                 |
| ---------------------- | -------------------------------------- |
| AclOfflineModel      | 通过aclopExecuteV2调用的方式调用算子 |
| AclOnlineModel       | 通过aclopCompile调用的方式调用算子   |
| CppExtensions        | 通过Pybind方式调用算子               |
| PytorchInvocation    | 通过pytorch调用的方式调用算子        |
| TensorflowInvocation | 通过tensorflow调用的方式调用算子     |

2. 将补充的算子调用方式更新到src/math/sqrt/README.md文档算子调用章节
