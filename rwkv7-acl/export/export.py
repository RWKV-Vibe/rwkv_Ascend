from rwkv_src.rwkv_model import RWKV_RNN
import types
import os
import torch
import torch_npu
import numpy as np
import argparse
import json
import copy
from pathlib import Path
from util.model_utils import get_dummy_input_for_rwkv_causal_llm, get_input_output_names

parser = argparse.ArgumentParser(description='Export onnx model')
parser.add_argument('model', type=Path, help='Path to RWKV pth file')
parser_args = parser.parse_args()

model_args = types.SimpleNamespace()
model_args.USE_CUDA = False
model_args.fp16 = True
model_args.wkv_customop = False
model_args.USE_EMBEDDING = True
model_args.RESCALE_LAYER = 6

model_args.MODEL_NAME = str(parser_args.model).replace('.pth', '')
model = RWKV_RNN(model_args).npu()
print(model)

output_dir = 'output'
output_path = str(output_dir / Path(os.path.basename(model_args.MODEL_NAME).replace('.pth', '') + '.onnx'))
dummy_input = get_dummy_input_for_rwkv_causal_llm(1, 1, model.device, model.args)
os.path.exists(output_dir) or os.makedirs(output_dir)

input_names, output_names = get_input_output_names(model.args)
#torch.onnx.export(model, dummy_input, output_path, input_names=input_names, output_names=output_names, opset_version=14,do_constant_folding=False)
torch.onnx.export(model, dummy_input, output_path, input_names=input_names, output_names=output_names, opset_version=14)

