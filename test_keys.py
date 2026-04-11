import sys
import torch

inp = "models/hf/gemma-3-4b-it-hqq-int8-int4/pytorch_model-00001-of-00002.bin"
obj = torch.load(inp, map_location='cpu', weights_only=False)
if hasattr(obj, 'state_dict'):
    obj = obj.state_dict()
elif isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
    obj = obj['state_dict']

for k in list(obj.keys())[:10]:
    v = obj[k]
    print(k, type(v), getattr(v, 'shape', 'No shape'))
    if type(v).__name__ != 'Tensor':
        try:
            print("  Attrs:", dir(v))
            if hasattr(v, '__tensor_flatten__'):
                print("  Flatten:", v.__tensor_flatten__())
        except:
            pass
