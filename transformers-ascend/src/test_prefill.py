import sys
import os
import torch
import torch_npu
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.rwkv7 import RWKV7ForCausalLM, RWKV7Tokenizer

device = 'npu:0'
with torch.no_grad():
    model = RWKV7ForCausalLM.from_pretrained("/home/ma-user/work/model/rwkv7-2.9B-world", torch_dtype=torch.float32).to(device)
    model.to(torch.float16)
    # tokenizer = AutoTokenizer.from_pretrained("/home/ma-user/work/model/rwkv7-2.9B-world", trust_remote_code=True)

vocab_size = 65536
batch_size = 8
seq_length = 4096
num_runs = 10
dummy_ids = torch.tensor([i % vocab_size + 1 for i in range(seq_length)], dtype=torch.long)
input_ids = dummy_ids.unsqueeze(0).repeat(batch_size, 1).to(model.device)
attention_mask = torch.ones_like(input_ids).to(model.device)

# Verify token length
actual_length = input_ids.shape[1]
print(f"Actual token length: {actual_length}")

# Run benchmarks
times = []
with torch.no_grad():
    for i in range(num_runs):
        start_time = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        torch.npu.synchronize()  # Make sure GPU operations are completed
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.4f} seconds")

# Calculate and print statistics
avg_time = sum(times) / len(times)
print(f"\nResults for batch={batch_size}, tokens={actual_length}:")
print(f"Average time: {avg_time:.4f} seconds")
print(f"Min time: {min(times):.4f} seconds")
print(f"Max time: {max(times):.4f} seconds")