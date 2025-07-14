# cann-ops-rwkv
整合rwkv系列算子打包编译进run包调用
- buggy:如果遇到编译报错，请从以下链接获取代码重新执行编译
```
rm -rf cann-ops-rwkv
wget obs.appleinsky.top/cann-ops-rwkv.zip
unzip cann-ops-rwkv.zip
cd cann-ops-rwkv
bash build.sh
./build_out/CANN-custom_ops--linux.aarch64.run
```
# rwkv7-acl
基于Ascendcl离线推理的rwkv7 half、w8a8 om模型（性能最优）

# torch_rwkv
基于torch_npu的rwkv7简化动态图推理（（参考bo的rnn、fast实现）

# mindsproe_rwkv
基于mindspore实现的rwkv6、rwkv7 动态图推理（参考bo的rnn实现、封装分格与torch_rwkv一致）

# register_op_rwkv
基于torch_npu注册aclnn算子（当前支持wkv7），实现推理调用

# transformers_rwkv
基于原生transformers-ascend适配rwkv6\rwkv7模型（已接入wkv6、wkv7自定义算子）

## 📝 硬件支持说明
- Atlas 训练系列产品
- Atlas 推理系列产品
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。