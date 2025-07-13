# RWKV ONNX EXPORT  

## 简介

基于昇腾推理硬件导出RWKV7 2.9B 模型ONNX文件

## 环境要求

### 支持的设备型号

本样例支持如下产品型号：

- Atlas 300I Pro/300V Pro

### 软件依赖

| 软件      | 软件版本         |
| --------- | :--------------- |
| CANN      | 8.2.RC1.alpha001 |
| Pytorch   | 2.1.0            |
| Torch_npu | 7.0.0            |

## 模型导出

### 环境准备

1. CANN安装

   参考[CANN安装手册](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)

   **tips:除了toolkit包还需要安装kernels算子包**

2. Torch_npu安装

   参考[Torch_npu安装手册](https://www.hiascend.com/document/detail/zh/Pytorch/700/configandinstg/instg/insg_0004.html)

### ONNX导出

1. 下载模型权重文件

   ```
   cd model
   wget http://obs.appleinsky.top/rwkv7-g1-2.9b-20250519-ctx4096.pth
   ```

2. 导出onnx

   ```
   cd ..
   python3 export.py model/rwkv7-g1-2.9b-20250519-ctx4096.pth
   ```

   onnx导出到当前目录的output文件夹下

### ONNX AMCT(W8A8)量化(可选)

1.环境准备

参考[AMCT手册](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/devaids/devtools/amct/atlasamct_16_0011.html)中的"准备环境"章节,安装AMCT(ONNX)相关部分

**tips:不要遗漏"安装后处理"的操作**

2.下载量化配置文件

```
wget http://obs.appleinsky.top/config.cfg
```

3.执行量化脚本

```
bash amct_w8a8.sh
```

量化模型生成在quant目录下,其中的rwkv_deploy_model.onnx是模型转换所需要的模型,rwkv_fake_model.onnx是精度模拟使用的.