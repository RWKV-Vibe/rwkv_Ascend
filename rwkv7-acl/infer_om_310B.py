########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################
import sys
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import types, torch, copy, time
from typing import List
from ais_bench.infer.interface import InferSession
import argparse
import aclruntime
#torch_npu.npu.set_compile_mode(jit_compile = False)
import torch.nn as nn
from torch.nn import functional as F

########################################################################################################

'''
This will load RWKV-7 "Goose" x070 and inference in RNN-mode (slower than GPT-mode for prefilling)
'''
parser = argparse.ArgumentParser(description='inference model')
parser.add_argument('model', type=str, help='Path to RWKV om file')
parser_args = parser.parse_args()

args = types.SimpleNamespace()

# model download: https://huggingface.co/BlinkDL/rwkv-7-world

args.MODEL_NAME = "./rwkv7-g1-2.9b-20250519-ctx4096"

args.n_layer = 32
args.n_embd = 2560
args.vocab_size = 65536
args.head_size = 64
prompt = "The Eiffel tower is in the city of"
NUM_TRIALS = 5
LENGTH_PER_TRIAL = 10
TEMPERATURE = 1.0
TOP_P = 0.0

# DTYPE = torch.bfloat16
DTYPE = torch.half # better

########################################################################################################

#@MyStatic
def sample_logits(logits, temperature:float=1.0, top_p:float=1.0, top_k:int=0):
    probs = F.softmax(logits.float(), dim=-1)
    sorted_probs, c = torch.sort(probs, descending=True)
    sorted_probs = sorted_probs
    if top_k > 0:
        probs[sorted_ids[top_k:]] = 0

    if top_p < 1:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff_index = torch.searchsorted(cumulative_probs, top_p)
        cutoff = sorted_probs[cutoff_index]
        probs[probs < cutoff] = 0

        if top_p > 0:
            idx = torch.where(probs == cutoff)[0]
            if len(idx) > 0:
                probs[idx] = cutoff + (top_p - torch.sum(probs).item()) / len(idx)
                # assert abs(torch.sum(probs).item() - top_p) < 1e-6
    
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    return torch.multinomial(probs, num_samples=1).item()

########################################################################################################
# RWKV Tokenizer (slow version)
########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()

tokenizer = RWKV_TOKENIZER("model/rwkv_vocab_v20230424.txt")

########################################################################################################

print(f'\nUsing CUDA {str(DTYPE).replace("torch.","")}. Loading {args.MODEL_NAME} ...')
#torch.save(model.state_dict(),"model_half.pth")
session = InferSession(device_id=0, model_path=parser_args.model)
print(f'\nPrefilling prompt (note: using RNN mode to prefill is very inefficient)')

init_state = [None for _ in range(args.n_layer * 3)]
for i in range(args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
    init_state[i*3+0] = aclruntime.Tensor(torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cpu").numpy())
    init_state[i*3+0].to_device(0)
    init_state[i*3+1] = aclruntime.Tensor(torch.zeros((args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.half, requires_grad=False, device="cpu").numpy())
    init_state[i*3+1].to_device(0)
    init_state[i*3+2] = aclruntime.Tensor(torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cpu").numpy())
    init_state[i*3+2].to_device(0)
start = time.time()
count = 0
for token in tokenizer.encode(prompt):
    token = aclruntime.Tensor(np.array(token))
    token.to_device(0)
    input = [token] + init_state
    output = session.infer(input,out_array=False)
    output[0].to_host()
    init_state = output[1:]
    count += 1
output[0].to_host()
t1 = time.time()
np.array(output[0])
init_out = torch.from_numpy(np.array(output[0])).view(-1)
probs = F.softmax(init_out, dim=-1) # compute softmax in float (more accurate)

print(f'\n{prompt}')
_, indices = torch.topk(probs, 10) # print top-10 possibilities
for i in range(len(indices)):
    token_id = indices[i].item()
    token = tokenizer.decode([token_id])
    token_prob = probs[token_id].item()
    print(token, f'[probability {token_prob:.2%}]')

########################################################################################################

for TRIAL in range(NUM_TRIALS):
    print(f'\n\n--[ Trial {TRIAL} ]-----------------', prompt, end="")
    all_tokens = []
    out_last = 0
    out, state = init_out, init_state
    
    min_time = 1e10
    min_time_all = 1e10

    t000 = time.perf_counter()

    for i in range(LENGTH_PER_TRIAL):
        t00 = time.perf_counter()
        token = sample_logits(out, TEMPERATURE, TOP_P)
        all_tokens += [token]
        try:
            tmp = tokenizer.decode(all_tokens[out_last:])
            if '\ufffd' not in tmp: # only print when we have a valid utf-8 string
                print(tmp, end="", flush=True)
                out_last = i + 1
        except:
            pass
        t0 = time.perf_counter()
        token = aclruntime.Tensor(np.array(token))
        token.to_device(0)
        input = [token] + state
        output = session.infer(input,out_array=False)
        output[0].to_host()
        out = torch.from_numpy(np.array(output[0])).view(-1)
        state = output[1:]
        
        t1 = time.perf_counter()
        min_time = min(min_time, t1 - t0)
        min_time_all = min(min_time_all, t1 - t00)
    
    print(f'\n[ {round(1/min_time_all,2)} (real) / {round(1/min_time,2)} (ignore sampling & tokenizer) token/s = {round(time.perf_counter()-t000,3)}s ]', end='')

print('\n')
