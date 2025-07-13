## 概述
本样例展示了自定义算子通过torch原生提供的cppextension方式注册eager模式与torch.compile模式的注册样例，eager模式与torch.compile模式的介绍参考：[Link](https://pytorch.org/get-started/pytorch-2.0)。

## 目录结构介绍
```
├── build_and_run.sh                // 自定义算子wheel包编译安装并执行用例的脚本
├── csrc                            // 算子适配层c++代码目录
│   ├── wkv7.cpp                    // 自定义算子正反向适配代码以及绑定
│   ├── function.h                  // 正反向接口头文件
│   ├── pytorch_npu_helper.hpp      // 自定义算子调用和下发框架
│   └── registration.cpp            // 自定义算子aten ir注册文件
├── custom_ops                      // 自定义算子包python侧代码
│   ├── wkv7.py                     // 提供自定义算子python调用接口
│   └── __init__.py                 // python初始化文件
├── setup.py                        // wheel包编译文件
└── test                            // 测试用例目录
    └── test_wkv7_custom.py          // 执行eager模式下算子用例脚本
```

## 样例脚本build_and_run.sh关键步骤解析

  - 编译适配层代码并生成wheel包
    ```bash
    python3 setup.py build bdist_wheel
    ```

  - 安装编译生成的wheel包
    ```bash
    cd ${BASE_DIR}
    pip3 install dist/*.whl
    ```
## 运行样例算子
该样例脚本基于Pytorch2.1、python3.9 运行
### 1.编译算子工程
运行此样例前，请参考[编译算子工程](../README.md#operatorcompile)完成前期准备。


#### 其他样例运行说明
  - 环境安装完成后，样例支持单独执行：eager模式与compile模式的测试用例
    - 执行pytorch eager模式的自定义算子测试文件
      ```bash
      python3 test_wkv7_custom.py
      ```
    <!-- - 执行pytorch torch.compile模式的自定义算子测试文件
      ```bash
      python3 test_add_custom_graph.py
      ``` -->

### 其他说明
    更加详细的Pytorch适配算子开发指导可以参考[LINK](https://gitee.com/ascend/op-plugin/wikis)中的“算子适配开发指南”。

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/07/11 | 初版readme |