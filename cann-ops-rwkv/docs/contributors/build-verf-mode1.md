# 本地构建和验证模式一

## 概述

本地构建和验证模式一是先在单算子工程开发然后迁移到本仓，本文主要介绍该模式下如何进行本地构建和验证

## 单算子工程和本仓算子目录结构介绍

使用msOpGen工具生成的单算子工程目录结构如下

```
├── add_custom                // 单算子工程根目录
│ ├── cmake 			
│ ├── framework               // 框架插件目录
│ ├── op_host                 // host目录
│ ├── op_kernel               // kernel目录
│ ├── scripts
│ ├── CMakePresets.json
│ └── build.sh
```

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

注：上述本仓对应算子的目录结构除了算子名add\_custom，其他**目录和文件名不能删除或修改**

## 本地构建和验证流程

以开发新算子Sqrt为例
注：如无特别指明以下的操作都是针对本仓的

### 算子包编译部署（必须）

1. 完成单算子工程开发验证和优化。
2. 拷贝一份示例算子src/math/add_custom到src/math下，将add_custom重命名为sqrt，命名风格统一用全小写且下划线分割。
3. 单算子工程目录和本仓算子目录下包含同名的op_host、op_kernel目录，直接用单算子工程这两个目录下的内容替换本仓算子目录下的内容，然后删除op_host和op_kernel目录下的CMakeLists.txt文件。
4. 打开src/math/sqrt/CMakeLists文件，将其中的`add_custom.cpp`修改成`sqrt.cpp`
5. 参考[算子工程说明文档中的算子包编译部署章节](../../src/math/add_custom/README.md#算子包编译部署)完成算子包编译部署

### aclnn接口自验证（必须）

**请确保已根据算子包编译部署步骤完成本算子的编译部署动作**

1. 修改src/math/sqrt/examples/AclNNInvocationNaive下的main.cpp、gen_data.py和verify_result.py文件使其适配当前算子
2. 参考[aclnn说明文档中的运行样例算子章节](../../src/math/add_custom/examples/AclNNInvocationNaive/README.md#运行样例算子)完成aclnn接口自验证

### ST自验证（必须）

**请确保已根据算子包编译部署步骤完成本算子的编译部署动作**

1. 生成算子测试用例定义文件
   ```
   cd src/math/sqrt
   msopst create -i op_host/sqrt.cpp -out tests/st/
   ```
2. 将生成的src/math/sqrt/tests/st/Sqrt_case_timestamp.json重命名为Sqrt_case_all_type.json，替换原有的json文件
3. 打开src/math/sqrt/tests/st/Sqrt_case_all_type.json文件，填写要测试的shape，并参考src/math/sqrt/tests/Sqrt_case_all_type.json添加一行来配置算子期望数据生成脚本
   ```
   "calc_expect_func_file": "test_sqrt.py:calc_expect_func",
   ```
4. 将src/math/sqrt/tests/st/test_add_custom.py重命名为test_sqrt.py，修改该文件中calc_expect_func函数的接口和逻辑以适配当前算子
5. 根据执行机器的架构修改msopst.ini中的atc_singleop_advance_option和HOST_ARCH
6. 参考[ST说明文档中的执行测试用例章节](../../src/math/add_custom/tests/st/README.md#执行测试用例)完成ST自验证

### 撰写文档（必须）

1. 撰写下列上库必须的文档

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
