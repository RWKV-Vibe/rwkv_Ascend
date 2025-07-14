## 目录结构介绍
``` 
├── AclNNInvocationNaive_float    //通过aclnn调用的方式调用wkv7算子
│   ├── CMakeLists.txt
│   ├── compute.ipynb       // 读取numpy数组进行单步计算调试
│   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   ├── verify_result.ipynb // jupyter读取bin进行真值对比误差分析
│   ├── verify_result.py    // 真值对比文件
│   ├── main.cpp            // wkv7单算子调用应用的入口
│   └── run.sh              // 执行命令脚本
``` 
## 代码实现介绍
完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：
   ```cpp    
   aclnnStatus aclnnwkv7GetWorkspaceSize(const aclTensor *key, const aclTensor *value, const aclTensor *weight, const aclTensor *query, const aclTensor *a, const aclTensor *b, const aclTensor *hi, const aclTensor *out, const aclTensor *ht,  uint64_t *workspaceSize, aclOpExecutor **executor);
   aclnnStatus aclnnwkv7(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream);
   ```
其中aclnnwkv7GetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小之后，按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnwkv7执行计算。具体参考[AscendCL单算子调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)>单算子API执行 章节。

## 运行样例算子
### 1. 编译算子工程
运行此样例前，请参考[编译算子工程](../../README.md#2-编译算子工程)完成前期准备。
### 2. aclnn调用样例运行

  - 进入到样例目录

    ```bash
    cd ${cann-ops-rwkv}\src\rwkv7\wkv7\examples\AclNNInvocationNaive_float
    ```
  
  - 样例执行    

    样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后检验运行结果。具体过程可参见run.sh脚本。

    ```bash
    bash run.sh
    ```
## 更新说明
  | 时间 | 更新事项 |
|----|------|
| 2025/07/14 | 更新readme |
