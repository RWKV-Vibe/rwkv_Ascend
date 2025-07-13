# RWKV ASCEND 

## 简介

基于昇腾推理硬件使用离线推理方式部署RWKV7 2.9B大模型

## 环境要求

### 支持的设备型号

本样例支持如下产品型号：

- Atlas 300I Pro/300V Pro
- 香橙派AIPRO 20T

### 软件依赖

| 软件      | 软件版本         |
| --------- | :--------------- |
| CANN      | 8.2.RC1.alpha001 |
| Pytorch   | 2.1.0            |
| Torch_npu | 7.0.0            |

tips:CANN 8.2.RC1.alpha002版本存在bug无法正常运行,其他版本未进行测试

## 推理测试

### 环境准备

1. CANN安装

   参考[CANN安装手册](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)

   **tips:除了toolkit包还需要安装kernels算子包**

2. Torch_npu安装

   参考[Torch_npu安装手册](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html)

3. 安装ais_bench工具

   参考[ais_bench工具仓库](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_bench)中"工具安装-工具安装方式-下载whl包安装"章节

### 推理运行

- #### 快速启动(约束要求:CANN版本为8.2.RC1.alpha001)

  1. 下载tokenizer文件到model目录下

     ```
     cd model
     wget http://obs.appleinsky.top/rwkv_vocab_v20230424.txt
     ```

  2. 下载om模型到model目录下

     | 模型列表      | 下载地址                                                |
     | ------------- | ------------------------------------------------------- |
     | 310P half模型 | http://obs.appleinsky.top/rwkv7/model/rwkv_half_310P.om |
     | 310P w8a8模型 | http://obs.appleinsky.top/rwkv7/model/rwkv_w8a8_310P.om |
     | 310B half模型 | http://obs.appleinsky.top/rwkv7/model/rwkv_half_310B.om |
     | 310B w8a8模型 | http://obs.appleinsky.top/rwkv7/model/rwkv_w8a8_310B.om |

     下载示例:

     ```
     wget http://obs.appleinsky.top/rwkv7/model/rwkv_half_310P.om
     ```

     

  3. 推理运行

     ```
     cd ..
     python3 infer_om_310P.py model/rwkv_half_310P.om
     ```

     **tips:请按照需要运行的om模型名称修改命令**

     **tips:香橙派运行请将运行脚本修改为infer_om_310B4.py**

  

- #### 正常流程

1. 下载tokenizer文件到model目录下

   ```
   cd model
   wget http://obs.appleinsky.top/rwkv_vocab_v20230424.txt
   ```

2. 下载模型onnx文件到model目录下

   - FP16模型

     ```
     wget http://obs.appleinsky.top/rwkv7.onnx.data
     wget http://obs.appleinsky.top/rwkv7.onnx
     ```

     tips:模型导出流程可参考 [RWKV onnx导出](export/README.md)

   - W8A8模型

     ```
     wget http://obs.appleinsky.top/quant_onnx.zip
     unzip quant_onnx.zip
     ```

     tips:量化流程可参考 [RWKV onnx导出](export/README.md)

3. 模型转换

   - FP16模型

     ```
     atc --model=rwkv7.onnx --soc_version=Ascend310P3 --framework=5 --output rwkv_half_310P 
     ```

   - W8A8模型

     ```
     atc --model=rwkv_deploy_model.onnx --soc_version=Ascend310P3 --framework=5 --output rwkv_w8a8_310P 
     ```

   **tips1:香橙派运行请修改soc_version为Ascend310B4**

   **tips2:如果atc转换内存不足失败请参考 [此链接](https://www.hiascend.com/forum/thread-0239142592318174023-1-1.html)**

4. 推理运行

   ```
   cd ..
   python3 infer_om_310P.py model/rwkv_half_310P.om
   ```

   **tips:请按照需要运行的om模型名称修改命令**

   **tips:香橙派运行请将运行脚本修改为infer_om_310B4.py**

