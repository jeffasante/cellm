import sys
import torch
from safetensors.torch import save_file

inp = "models/hf/gemma-3-4b-it-hqq-int8-int4/pytorch_model-00001-of-00002.bin"
out = "test.safetensors"
obj = torch.load(inp, map_location='cpu', weights_only=False)
print("Keys:", list(obj.keys())[:10])
