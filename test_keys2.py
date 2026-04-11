import sys
import torch

inp = "models/hf/gemma-3-4b-it-hqq-int8-int4/pytorch_model-00001-of-00002.bin"
obj = torch.load(inp, map_location='cpu', weights_only=False)
if hasattr(obj, 'state_dict'):
    obj = obj.state_dict()
elif isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
    obj = obj['state_dict']

for k in list(obj.keys())[:20]:
    v = obj[k]
    if hasattr(v, '__tensor_flatten__'):
        t_names, t_ctx = v.__tensor_flatten__()
        for n in t_names:
            att = getattr(v, n)
            if att is not None:
                print(f"{k}.{n}", getattr(att, 'shape', 'No shape'), getattr(att, 'dtype', 'No dtype'))
            else:
                print(f"{k}.{n}", "None")
