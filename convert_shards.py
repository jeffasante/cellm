import sys
import torch
from safetensors.torch import save_file
import glob
import os

inputs = glob.glob('models/hf/gemma-3-4b-it-hqq-int8-int4/*.bin')
for inp in inputs:
    out = inp.replace('.bin', '.safetensors')
    print(f"Converting {inp} to {out}")
    obj = torch.load(inp, map_location='cpu', weights_only=False)
    if hasattr(obj, 'state_dict'):
        obj = obj.state_dict()
    elif isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        obj = obj['state_dict']
    elif isinstance(obj, dict) and 'model_state_dict' in obj and isinstance(obj['model_state_dict'], dict):
        obj = obj['model_state_dict']
    
    clean = {}
    for k, v in obj.items():
        if torch.is_tensor(v):
            t = v
            if hasattr(t, "dequantize"):
                try:
                    t = t.dequantize()
                except Exception:
                    pass
            clean[str(k)] = t.detach().cpu().contiguous()
    
    save_file(clean, out)
    print(len(clean))
