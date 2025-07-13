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
# mindsproe_rwkv
基于mindspore实现的rwkv6、rwkv7 动态图推理（参考bo的rnn实现）

# register_op_rwkv
基于torch_npu注册wkv7算子，实现推理调用

# rwkv7-acl
基于acl离线推理的rwkv7 om模型（310P：half、w8a8  310B: half、w8a8）

# torch_rwkv
基于torch_npu的rwkv7简化动态图推理（（参考bo的rnn、fast实现）

# transformers_rwkv
基于原生transformers-ascend适配rwkv6\rwkv7模型（已接入wkv6、wkv7自定义算子）