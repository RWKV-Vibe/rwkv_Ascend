## transformers_rwkv

### 环境准备
#### python环境安装
```
pip install -e 
pip install wheel einops expecttest
```
#### 自定义算子注册
```
cd .. && cd register_op_rwkv
cd CppExtensionInvocation
python3 setup.py build bdist_wheel
pip install dist/*.whl
```
### 适配rwkv6、rwkv7模型
- rwkv6代码路径：.src\transformers\models\rwkv6 
- rwkv7代码路径：.src\transformers\models\rwkv7 
#### 推理rwkv7
```
cd src
python3 test_v7.py
```
#### 测试prefill
```
cd src
python3 test_prefill.py
```