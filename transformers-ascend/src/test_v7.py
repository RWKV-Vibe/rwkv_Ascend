import sys
import os
import torch
import torch_npu
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 指定 so 路径
# custom_lib_path = "/home/ma-user/work/wkv7_Ascend/npu/build"

# os.environ['WKV_LIB'] = custom_lib_path

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.rwkv7 import RWKV7ForCausalLM, RWKV7Tokenizer

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""

device = 'npu:0'
model = RWKV7ForCausalLM.from_pretrained("/root/rwkv7-2.9B-world", torch_dtype=torch.float32).to(device)
tokenizer = AutoTokenizer.from_pretrained("/root/rwkv7-2.9B-world", trust_remote_code=True)


text = "请介绍北京的旅游景点"
prompt = generate_prompt(text)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])  # 显式设置

print("ready to decode!")
import time
start_time = time.time()
output = model.generate(inputs["input_ids"], max_new_tokens=200, do_sample=True, temperature=1.0, top_p=0.3, top_k=0, )
# output = model.generate(inputs["input_ids"], max_new_tokens=1)
end_time = time.time() - start_time
print("out:", tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
print(f"total time: {end_time} s, speed: {200/end_time} token/s")
# print(f"prefill time: {end_time} s")


